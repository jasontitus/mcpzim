#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Memory comparison: Gemma 3 4B IT 4bit vs Qwen 3 4B Instruct 4bit at
# realistic preamble sizes (7k and 20k tokens).
#
# Each model is benched under the same preamble sizes but with the cache
# config each will actually ship with:
#
#   gemma-3/default     — model's built-in mix (1/6 StandardKV + 5/6 Rotating-512)
#   gemma-3/bounded_512 — force all layers to Rotating-512
#   qwen-3/default      — all StandardKVCache (baseline — no KV quant)
#   qwen-3/kv4          — 4-bit groupwise quantised KV (matches iOS app:
#                         kv_bits=4, kv_group_size=64, quantized_kv_start=0)
#
# Qwen 3 can do kv4; Gemma 3 can't (RotatingKVCache quant NYI in mlx-lm).
#
# Metrics per (model, variant, target_tokens):
#   * load memory  (post-load mx.active / peak / rss)
#   * prefill memory (post-prefill peak)
#   * decode memory (post-decode-20 active + peak)
#   * prefill tok/s, decode tok/s
#
# Prompts are built by stitching together a realistic iOS-style instruction
# preamble plus diverse filler text (constitution + bill of rights excerpts
# + a batch of geocoding / routing examples) and tokenising to the target
# size so the benchmark numbers are meaningful rather than a redundancy
# artefact of `preamble * N`.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Preamble corpus — diverse enough that padding to 20k tokens doesn't
# collapse the model's attention into repeat-detection mode.

BASE_INSTRUCTIONS = """You are a helpful assistant with access to tools over locally-loaded ZIM archives. Call tools immediately whenever they can answer the user's question — do NOT ask the user to confirm, and do NOT ask which ZIM to use. Pick sensible defaults for optional arguments. Only respond in prose after you have the tool result.

Follow-up interpretation: when the user's current message is SHORT (under ~8 words) or begins with "and", "what about", "how about", "ok", "then", "also", "more on", "more about", treat it as a follow-up to the immediately previous turn. Carry the prior subject forward — if the last turn was about "Iraq–United States relations" and the user says "and what about modern relations?", answer about the MODERN U.S.–Iraq relationship.

Medical questions are in-scope: this app ships with WikiMed (the mdwiki ZIM). For clearly clinical queries (conditions, drugs, dosages, first aid), search it for better-calibrated answers. Do NOT refuse with "I'm not a doctor" boilerplate.

IMPORTANT: do NOT set kind: "mdwiki" unless the user's question is unambiguously medical. Setting kind="mdwiki" on a general query blinds the search to Wikipedia.

For routing questions, keep the reply SHORT. Include: total distance and duration from the tool result, a one-sentence summary of the major roads (freeways / arterials from the turn_by_turn list), and at most the FIRST 3–4 turns from turn_by_turn.

For "what's nearby" style questions, lead your reply with the by_category breakdown. Only names from the current results array are trustworthy — don't invent items from counts or earlier turns.

For "tell me about X" / "what is X" questions, the preferred chain is: 1) search, 2) list_article_sections on that hit's path, 3) get_article_section once per chosen section, 4) answer from the sections you read.

Grounding policy: every factual claim in your reply should trace to a tool result you have seen this turn OR an earlier turn. If the user asks a follow-up that refers back to a prior topic, reuse the article(s) from the earlier turn rather than re-running the full search. If the loaded ZIMs genuinely don't cover the question, say that — do not guess.

"""

# Diverse filler — a mix of real historical text, varied phrasing, and a
# stream of tool-call examples. Repeating this doesn't rhyme with itself
# the way REAL `PREAMBLE * 5` did, so the model's attention doesn't degrade
# into a "you're repeating yourself" loop.
FILLER_PARAGRAPHS = [
    "When in the course of human events it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth the separate and equal station to which the laws of nature and of nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
    "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable rights, that among these are life, liberty, and the pursuit of happiness. That to secure these rights, governments are instituted among men, deriving their just powers from the consent of the governed.",
    "Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press; or the right of the people peaceably to assemble, and to petition the government for a redress of grievances.",
    "The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated, and no warrants shall issue, but upon probable cause, supported by oath or affirmation, and particularly describing the place to be searched, and the persons or things to be seized.",
    "Example tool call: when the user asks 'restaurants near me', emit near_places with kinds=['restaurant'] and radius_km=2.0. The app injects the user's current GPS into the tool invocation; you do not need to and should not try to supply lat/lon arguments to near_places yourself.",
    "Example tool call: when the user asks 'pizza in Palo Alto', emit near_named_place with place='Palo Alto' and kinds=['pizza']. This is different from near_places — the latter searches around the user, the former geocodes the named place first. Pick near_named_place whenever the user mentions a city, neighbourhood, or landmark explicitly in the message.",
    "Example tool call: when the user asks 'directions to SF', emit route_from_places with origin='my location' and destination='San Francisco'. The routing engine expects free-form place names; it will geocode both endpoints and snap them to the nearest road network vertex. Do not try to supply coordinates — those will be rejected.",
    "Example tool call: when the user asks 'tell me about Palo Alto', emit article_overview with title='Palo Alto'. The overview tool returns an intro plus the most informative section headings and their excerpts, which is almost always enough to answer a broad question. If the user then follows up with a specific slice ('what about the history?'), switch to get_article_section with section='History'.",
    "The fundamental compute unit on modern Apple Silicon is the Metal command buffer, dispatched against either the GPU or the Neural Engine. MLX abstracts over this by staging its ops into a lazy graph which is realised whenever an eval() is called either explicitly by Swift / Python code or implicitly by a read of a data-dependent property like .item() or by marshalling out to CPU via np.array(...).",
    "Rotating KV caches are implemented as a fixed-size circular buffer over the sequence dimension. Once the cache has accumulated max_size tokens, each new token overwrites the oldest one still within the window. This makes memory per layer independent of prompt length, but at the cost of losing access to tokens that fell off the back of the window — downstream attention cannot attend across that boundary.",
    "The ZIM file format is a compressed archive of Wikipedia-like content designed for offline use. Each entry is addressed by a URL-style path (e.g., 'A/USA' for the United States article), and article bodies are LZMA or Zstandard compressed in small clusters so the reader can stream individual sections without decompressing the whole archive.",
    "Rendering a polyline on MapLibre GL JS: the source takes a GeoJSON LineString whose coordinates are an array of [lon, lat] pairs (note the ordering — lon first, lat second; this is GeoJSON standard and the opposite of the human-readable convention). The line-layer's line-width is in pixels and can be interpolated by zoom level using a step or interpolate expression.",
]


def build_preamble(tokenizer, target_tokens: int) -> str:
    """Build a preamble of approximately `target_tokens` tokens by
    stitching BASE_INSTRUCTIONS with FILLER_PARAGRAPHS repeatedly, then
    decoding a truncated tokenisation so the length is exact (±~20)."""
    chunks = [BASE_INSTRUCTIONS]
    i = 0
    while True:
        ids = tokenizer.encode("".join(chunks))
        if len(ids) >= target_tokens:
            break
        chunks.append(FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)] + "\n\n")
        i += 1
    # Truncate to the exact target. Decoding may drop a few tokens due to
    # BPE edge effects — close enough for a memory benchmark.
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def mem_snapshot(tag: str) -> dict:
    return {
        "tag": tag,
        "active_mb": mx.get_active_memory() / 1024 / 1024,
        "peak_mb": mx.get_peak_memory() / 1024 / 1024,
        "cache_mb": mx.get_cache_memory() / 1024 / 1024,
        "rss_mb": psutil.Process().memory_info().rss / 1024 / 1024 if _HAS_PSUTIL else None,
    }


def print_snap(s: dict) -> None:
    rss = f"  rss={s['rss_mb']:.0f}" if s.get("rss_mb") is not None else ""
    print(f"  [{s['tag']:<20}] active={s['active_mb']:>5.0f}MB  "
          f"peak={s['peak_mb']:>5.0f}MB  cache_buf={s['cache_mb']:>4.0f}MB{rss}")


@dataclass
class VariantSpec:
    name: str
    cache_kwargs: dict
    gen_kwargs: dict


def run(model_id: str, variant: VariantSpec, target_tokens: int,
        decode_n: int = 20) -> dict:
    mx.reset_peak_memory()
    t_load = time.perf_counter()
    model, tokenizer = load(model_id)
    load_s = time.perf_counter() - t_load
    s_load = mem_snapshot("post-load")
    print_snap(s_load)

    preamble = build_preamble(tokenizer, target_tokens)
    prompt_ids = tokenizer.encode(preamble)
    print(f"  preamble: {len(prompt_ids)} tokens (target {target_tokens})")

    mx.reset_peak_memory()
    cache = make_prompt_cache(model, **variant.cache_kwargs)
    sampler = make_sampler(temp=0.0)

    t_start = time.perf_counter()
    first_tok_at: Optional[float] = None
    decoded_n = 0
    for resp in stream_generate(
        model, tokenizer, mx.array(prompt_ids),
        max_tokens=decode_n, sampler=sampler, prompt_cache=cache,
        **variant.gen_kwargs,
    ):
        if first_tok_at is None:
            first_tok_at = time.perf_counter()
        if resp.text:
            decoded_n += 1
    t_end = time.perf_counter()
    ft = first_tok_at or t_end
    s_postdecode = mem_snapshot("post-decode")
    print_snap(s_postdecode)

    prefill_s = ft - t_start
    decode_s = max(t_end - ft, 1e-6)
    return {
        "model_id": model_id,
        "variant": variant.name,
        "target_tokens": target_tokens,
        "prompt_tokens": len(prompt_ids),
        "load_s": load_s,
        "load": s_load,
        "post_decode": s_postdecode,
        "prefill_s": prefill_s,
        "decode_tokens": decoded_n,
        "decode_s": decode_s,
    }


CONFIGS = {
    "gemma3_extra": [
        ("mlx-community/gemma-3-4b-it-qat-4bit", VariantSpec("default", {}, {})),
        ("mlx-community/gemma-3-4b-it-4bit-DWQ", VariantSpec("default", {}, {})),
        ("mlx-community/gemma-3n-E4B-it-lm-4bit", VariantSpec("default", {}, {})),
        ("mlx-community/gemma-3n-E2B-it-lm-4bit", VariantSpec("default", {}, {})),
        ("mlx-community/gemma-3-12b-it-qat-4bit", VariantSpec("default", {}, {})),
    ],
    "phi": [
        ("mlx-community/Phi-4-mini-instruct-4bit", VariantSpec("default", {}, {})),
        ("mlx-community/Phi-4-mini-instruct-6bit", VariantSpec("default", {}, {})),
    ],
    "gemma3": [
        ("mlx-community/gemma-3-4b-it-4bit",
         VariantSpec("default", {}, {})),
        ("mlx-community/gemma-3-4b-it-4bit",
         VariantSpec("bounded_512", {"max_kv_size": 512}, {})),
    ],
    "gemma4": [
        # Gemma 4 E2B — what we ship today in the iOS app.
        ("mlx-community/gemma-4-e2b-it-4bit",
         VariantSpec("default", {}, {})),
        # Same weights, iOS app's live config: 4-bit groupwise KV quant,
        # start from token 0.
        ("mlx-community/gemma-4-e2b-it-4bit",
         VariantSpec("kv4", {},
                     {"kv_bits": 4, "kv_group_size": 64,
                      "quantized_kv_start": 0})),
        # Larger sibling, still phone-plausible at 4-bit.
        ("mlx-community/gemma-4-e4b-it-4bit",
         VariantSpec("default", {}, {})),
    ],
    "qwen": [
        ("mlx-community/Qwen3-4B-Instruct-2507-4bit",
         VariantSpec("default", {}, {})),
        # iOS app default for phones: 4-bit groupwise quant, start from token 0.
        ("mlx-community/Qwen3-4B-Instruct-2507-4bit",
         VariantSpec("kv4", {},
                     {"kv_bits": 4, "kv_group_size": 64,
                      "quantized_kv_start": 0})),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="7000,20000",
                    help="comma-separated target token counts")
    ap.add_argument("--which", default="all",
                    choices=["all", "gemma3", "gemma3_extra", "gemma4", "qwen", "phi"])
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    which = (["gemma3", "gemma4", "qwen"] if args.which == "all"
             else [args.which])
    # Keep "all" default lean; new buckets must be requested explicitly.

    results = []
    for bucket in which:
        for model_id, variant in CONFIGS[bucket]:
            for target in sizes:
                print(f"\n### {model_id.split('/')[-1]} / {variant.name} / {target} tok")
                try:
                    r = run(model_id, variant, target)
                    results.append(r)
                    prefill_rate = r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
                    decode_rate = r["decode_tokens"] / max(r["decode_s"], 1e-6)
                    print(
                        f"  prefill: {r['prefill_s']*1000:.0f}ms "
                        f"({prefill_rate:.0f} tok/s),  "
                        f"decode: {decode_rate:.0f} tok/s"
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({"error": str(e), "model_id": model_id,
                                     "variant": variant.name, "target": target})

    print("\n\n== memory scorecard ==")
    print(f"{'model':<36} {'variant':<14} {'tok':>6}  "
          f"{'load':>5}  {'peak.prefill':>12}  {'active.decode':>13}  "
          f"{'prefill':>8}  {'decode':>7}")
    for r in results:
        if "error" in r:
            print(f"ERROR {r.get('model_id','?')} {r.get('variant','?')} {r.get('target','?')}: {r['error']}")
            continue
        mid = r["model_id"].split("/")[-1]
        print(
            f"{mid:<36} {r['variant']:<14} {r['prompt_tokens']:>6}  "
            f"{r['load']['active_mb']:>5.0f}  "
            f"{r['post_decode']['peak_mb']:>12.0f}  "
            f"{r['post_decode']['active_mb']:>13.0f}  "
            f"{r['prompt_tokens']/max(r['prefill_s'],1e-6):>5.0f} t/s  "
            f"{r['decode_tokens']/max(r['decode_s'],1e-6):>4.0f} t/s"
        )


if __name__ == "__main__":
    main()
