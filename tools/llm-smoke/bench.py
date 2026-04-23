#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Memory + prompt-warming bench companion to eval.py.
#
# Run:
#   source tools/llm-smoke/.venv/bin/activate
#   python tools/llm-smoke/bench.py mlx-community/gemma-3-4b-it-4bit
#   python tools/llm-smoke/bench.py mlx-community/gemma-3-4b-it-4bit --mode warm
#
# Reports:
#   * MLX active / peak / cache buffer sizes after load, after first prefill,
#     after first decode — these are the numbers that matter for on-device
#     jetsam headroom, not process RSS which includes python overhead.
#   * Prompt warming: times `prompt_1` (preamble + user-1 + assistant-1 +
#     user-2) twice — once with a fresh cache (cold), once reusing the cache
#     built during turn 1 (warm). Cold / warm ratio tells us whether the
#     family's cache implementation actually reuses tokens cleanly. Qwen 3.5
#     hybrid-attention tripped on this (mlx-swift-lm#157) and we want to
#     verify Gemma 3 doesn't.

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# A preamble long enough to mimic the app's ~7k token system. Used to put
# real pressure on prefill and KV cache size; without this the warm-cache
# ratio is meaningless.
PREAMBLE = (
    "You are MCPZim, an assistant with access to offline Wikipedia and "
    "OpenStreetMap data via tools. The tools exposed to you are for "
    "nearby-places search (near_places, near_named_place, nearby_stories, "
    "nearby_stories_at_place, what_is_here), routing (route_from_places, "
    "route_status), Wikipedia article retrieval (article_overview, "
    "get_article_section, narrate_article, compare_articles), and their "
    "context-aware variants. When the user asks anything that would be "
    "served by a tool, call it; do NOT decline on the basis of not having "
    "real-time access — the tools provide all real-time data needed.\n\n"
    # Pad to ~4 KB. Real app preamble has extensive tool descriptions +
    # examples + behavioural rules; this stand-in is just token ballast so
    # the prefill cost resembles production.
    + (
        "You have memory of the current conversation, the user's approximate "
        "location (via GPS), and any active driving route. Respect these as "
        "context when reasoning about what tool to pick. Prefer the most "
        "specific tool for the situation: if a place name is mentioned, "
        "reach for the `_named_place` / `_at_place` variants instead of the "
        "user-location-centred ones. If no tool fits, answer from your own "
        "training knowledge, concisely. Never hallucinate a tool that is "
        "not in the provided schema.\n"
    ) * 15
)


def mem_snapshot(tag: str) -> dict:
    d = {
        "tag": tag,
        "mlx_active_mb": mx.get_active_memory() / 1024 / 1024,
        "mlx_peak_mb": mx.get_peak_memory() / 1024 / 1024,
        "mlx_cache_mb": mx.get_cache_memory() / 1024 / 1024,
    }
    if _HAS_PSUTIL:
        d["rss_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
    return d


def print_snapshot(s: dict) -> None:
    rss = f" rss={s['rss_mb']:.0f}MB" if "rss_mb" in s else ""
    print(
        f"  [{s['tag']}] mlx.active={s['mlx_active_mb']:.0f}MB  "
        f"peak={s['mlx_peak_mb']:.0f}MB  "
        f"cache_buf={s['mlx_cache_mb']:.0f}MB{rss}"
    )


def time_generate(model, tokenizer, prompt: str, cache, max_tokens: int = 30):
    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    decoded_tokens = 0
    for resp in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=cache,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        if resp.text:
            decoded_tokens += 1
    t_end = time.perf_counter()
    ft = first_token_at if first_token_at is not None else t_end
    return {
        "prefill_s": ft - t0,
        "decode_s": max(t_end - ft, 1e-6),
        "decode_tokens": decoded_tokens,
    }


def bench(model_id: str, mode: str, max_tokens: int) -> None:
    print(f"== bench ==\nmodel:  {model_id}\nmode:   {mode}\n")
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"loaded in {time.perf_counter() - t0:.2f}s")
    print_snapshot(mem_snapshot("post-load"))

    # --------------------------------------------------- memory pass
    if mode in ("mem", "all"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PREAMBLE},
                {"role": "user", "content": "Tell me about Palo Alto."},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer.encode(prompt)
        print(f"\n--- memory pass (prompt={len(tokens)} tokens) ---")
        mx.reset_peak_memory()
        cache = make_prompt_cache(model)
        # Pre-eval: drive prefill only (no decode) by asking for 1 token.
        r = time_generate(model, tokenizer, prompt, cache, max_tokens=1)
        print_snapshot(mem_snapshot("post-prefill"))
        # Continue decoding 60 tokens to measure cache growth during decode.
        r2 = time_generate(model, tokenizer, "", cache, max_tokens=60)
        print_snapshot(mem_snapshot("post-decode-60"))
        prefill_s = r["prefill_s"]
        print(
            f"\nprefill: {prefill_s*1000:.0f}ms "
            f"({len(tokens)} tok → {len(tokens)/prefill_s:.0f} tok/s)"
        )

    # --------------------------------------------------- warm-cache pass
    if mode in ("warm", "all"):
        print("\n--- warm-cache pass ---")
        # This mirrors what our iOS ChatSession does: prefill turn-1's
        # prompt, measure LCP against turn-2's prompt tokens, then only
        # prefill the delta tokens through the cache. `mlx_lm.stream_generate`
        # by itself doesn't do that — it re-ingests the entire prompt every
        # call. So we compute the LCP manually and pass only the delta to
        # the warm run.
        user1 = "Tell me about Palo Alto."
        assistant1 = (
            '<tool_call>{"name":"article_overview","arguments":{"title":"Palo Alto"}}'
            '</tool_call>'
        )
        user2 = "How about Mountain View instead?"
        p1 = tokenizer.apply_chat_template(
            [{"role": "system", "content": PREAMBLE},
             {"role": "user", "content": user1}],
            tokenize=False, add_generation_prompt=True
        )
        p2 = tokenizer.apply_chat_template(
            [{"role": "system", "content": PREAMBLE},
             {"role": "user", "content": user1},
             {"role": "assistant", "content": assistant1},
             {"role": "user", "content": user2}],
            tokenize=False, add_generation_prompt=True
        )
        t1 = tokenizer.encode(p1)
        t2 = tokenizer.encode(p2)
        lcp = 0
        while lcp < min(len(t1), len(t2)) and t1[lcp] == t2[lcp]:
            lcp += 1
        print(f"turn-1 tokens: {len(t1)}")
        print(f"turn-2 tokens: {len(t2)}  (LCP with turn-1: {lcp} → delta {len(t2) - lcp})")
        bpe_stable = (lcp == len(t1))
        print(f"BPE boundary stable: {bpe_stable}"
              + ("" if bpe_stable else
                 f"  (LOST {len(t1) - lcp} tokens of turn-1's suffix at the boundary)"))

        # (a) cold turn-1 — fresh cache, drive 1 decoded token so the cache
        # contains everything through the turn's assistant-open marker.
        mx.reset_peak_memory()
        warm_cache = make_prompt_cache(model)
        r_cold1 = time_generate(model, tokenizer, p1, warm_cache, max_tokens=1)
        print_snapshot(mem_snapshot("after-turn1-prefill"))

        # (b) WARM turn-2 — the cache already has LCP tokens; we pass only
        # the remaining delta as the prompt. stream_generate extends the
        # cache with these new tokens rather than re-prefilling from zero.
        delta_ids = t2[lcp:]
        # stream_generate wants a str OR a list of token IDs via mx.array.
        # Use the token-id path to avoid re-tokenisation drift.
        delta_prompt = mx.array(delta_ids)
        r_warm2 = time_generate(model, tokenizer, delta_prompt, warm_cache, max_tokens=max_tokens)
        print_snapshot(mem_snapshot("after-turn2-warm"))

        # (c) COLD turn-2 — fresh cache, full prompt.
        mx.reset_peak_memory()
        cold2_cache = make_prompt_cache(model)
        r_cold2 = time_generate(model, tokenizer, p2, cold2_cache, max_tokens=max_tokens)
        print_snapshot(mem_snapshot("after-turn2-cold"))

        print("\n--- warm-cache result ---")
        print(f"turn-1 prefill (cold):  {r_cold1['prefill_s']*1000:>6.0f}ms "
              f"({len(t1)/r_cold1['prefill_s']:.0f} tok/s, {len(t1)} tok)")
        print(f"turn-2 prefill (cold):  {r_cold2['prefill_s']*1000:>6.0f}ms "
              f"({len(t2)/r_cold2['prefill_s']:.0f} tok/s, {len(t2)} tok)")
        print(f"turn-2 prefill (warm):  {r_warm2['prefill_s']*1000:>6.0f}ms "
              f"({len(delta_ids)/r_warm2['prefill_s']:.0f} tok/s, {len(delta_ids)} delta tok)")
        saved = r_cold2["prefill_s"] - r_warm2["prefill_s"]
        ratio = r_warm2["prefill_s"] / r_cold2["prefill_s"] if r_cold2["prefill_s"] else 0
        print(f"saved by warm cache:    {saved*1000:>6.0f}ms  (warm/cold={ratio*100:.0f}%)")
        print(f"decode speed (warm):    {r_warm2['decode_tokens']/r_warm2['decode_s']:.0f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--mode", choices=["mem", "warm", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=60)
    args = ap.parse_args()
    bench(args.model_id, args.mode, args.max_tokens)


if __name__ == "__main__":
    main()
