#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# KV-cache variant bench for Gemma 3 4B (and other candidates).
#
# Measures quality + memory across four cache configurations so we can see
# the accuracy / memory trade-off before committing to any on-device
# optimisation:
#
#   default       — what the model ships with. Gemma 3 already mixes
#                   `StandardKVCache` on its global layers (1/6) with
#                   `RotatingKVCache(maxSize=512)` on sliding-window
#                   layers (5/6). This is the baseline to beat.
#
#   bounded_512   — force ALL layers to RotatingKVCache(maxSize=512).
#                   Hits the global layers too, capping the full cache.
#                   Via `make_prompt_cache(model, max_kv_size=512)`.
#
#   q8_from_64    — start quantising K/V to int8 after the first 64
#                   tokens. `kv_bits=8, kv_group_size=64,
#                   quantized_kv_start=64`. Quantised KV layers cost
#                   roughly 1/2 the memory of bf16 layers.
#
#   q4_from_64    — same as above but int4. ~1/4 memory of bf16 at the
#                   cost of more precision loss. This is the aggressive
#                   option and the one most at risk of degrading
#                   tool-calling accuracy.
#
# Reports per variant:
#   * 9-scenario tool-selection score (from eval.py)
#   * turn-1 prefill tok/s, decode tok/s on the preamble
#   * MLX active + peak memory at end-of-turn-1
#
# Usage:
#   source tools/llm-smoke/.venv/bin/activate
#   python tools/llm-smoke/bench_kv.py mlx-community/gemma-3-4b-it-4bit

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from eval import CASES, TOOLS, SYSTEM, build_prompt, extract_tool_call
from bench import PREAMBLE  # ~4KB; `--long-preamble` multiplies this


# A 7k-ish-token preamble that mirrors the shipped iOS system prompt size.
# Built by concatenating the short PREAMBLE from bench.py with itself so we
# don't need to maintain a second long template here. Used to stress the
# KV cache into the regime where RotatingKVCache(512) actually saves memory
# relative to a full KVCacheSimple.
LONG_PREAMBLE = PREAMBLE * 5


@dataclass
class Variant:
    name: str
    # kwargs forwarded to `make_prompt_cache(model, **)`.
    cache_kwargs: dict
    # kwargs forwarded to `stream_generate(...)`.
    gen_kwargs: dict


VARIANTS = [
    Variant("default",     {},                      {}),
    Variant("bounded_512", {"max_kv_size": 512},    {}),
    Variant("q8_from_64",  {},                      {"kv_bits": 8,
                                                     "kv_group_size": 64,
                                                     "quantized_kv_start": 64}),
    Variant("q4_from_64",  {},                      {"kv_bits": 4,
                                                     "kv_group_size": 64,
                                                     "quantized_kv_start": 64}),
]


def _build_long_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": LONG_PREAMBLE,
         "tools": json.dumps(TOOLS, ensure_ascii=False)},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def run_case(model, tokenizer, case, variant: Variant, max_tokens: int = 200,
             long: bool = False):
    prompt = (_build_long_prompt(tokenizer, case.prompt) if long
              else build_prompt(tokenizer, case.prompt))
    prompt_tokens = tokenizer.encode(prompt)
    mx.reset_peak_memory()
    cache = make_prompt_cache(model, **variant.cache_kwargs)
    sampler = make_sampler(temp=0.0)
    t_start = time.perf_counter()
    first_tok_at = None
    decoded = []
    decoded_n = 0
    for resp in stream_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=cache,
        **variant.gen_kwargs,
    ):
        if first_tok_at is None:
            first_tok_at = time.perf_counter()
        if resp.text:
            decoded.append(resp.text)
            decoded_n += 1
    t_end = time.perf_counter()
    ft = first_tok_at or t_end
    output = "".join(decoded)
    call = extract_tool_call(output)
    name_ok = bool(call and call[0] in case.expected_tools)
    args_ok = False
    if call:
        args_lower = json.dumps(call[1]).lower()
        args_ok = all(s.lower() in args_lower for s in case.expected_args_contain)

    return {
        "name_ok": name_ok,
        "args_ok": args_ok,
        "got": call[0] if call else None,
        "prompt_tokens": len(prompt_tokens),
        "prefill_s": ft - t_start,
        "decode_tokens": decoded_n,
        "decode_s": max(t_end - ft, 1e-6),
        "peak_mb": mx.get_peak_memory() / 1024 / 1024,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--variants", default="all",
                    help="comma-separated variant names, or 'all'")
    ap.add_argument("--long-preamble", action="store_true",
                    help="use a ~7-8k-token preamble (matches the iOS app) "
                         "to put KV caches under real pressure")
    args = ap.parse_args()

    want = set(v.strip() for v in args.variants.split(",")) if args.variants != "all" else None
    run_variants = [v for v in VARIANTS if not want or v.name in want]

    print(f"== kv-bench ==\nmodel: {args.model_id}")
    print(f"variants: {[v.name for v in run_variants]}")

    t0 = time.perf_counter()
    model, tokenizer = load(args.model_id)
    print(f"loaded in {time.perf_counter() - t0:.2f}s")
    print(f"post-load mlx.active={mx.get_active_memory()/1024/1024:.0f}MB\n")

    # Per-variant aggregates
    summary = {}
    for variant in run_variants:
        print(f"\n### variant: {variant.name}")
        print(f"  cache_kwargs={variant.cache_kwargs}  gen_kwargs={variant.gen_kwargs}")
        passes = 0
        name_passes = 0
        prefill_rates = []
        decode_rates = []
        peaks = []
        for case in CASES:
            try:
                r = run_case(model, tokenizer, case, variant, long=args.long_preamble)
            except Exception as e:
                print(f"  [ERR] {case.name}: {e}")
                continue
            ok = r["name_ok"] and r["args_ok"]
            if ok: passes += 1
            if r["name_ok"]: name_passes += 1
            prefill_rates.append(r["prompt_tokens"] / max(r["prefill_s"], 1e-6))
            decode_rates.append(r["decode_tokens"] / max(r["decode_s"], 1e-6))
            peaks.append(r["peak_mb"])
            mark = "✓" if ok else ("∼" if r["name_ok"] else "✗")
            print(f"  {mark} {case.name:<28}  got={str(r['got'])[:24]:<24}  "
                  f"prefill={r['prefill_s']*1000:>5.0f}ms  "
                  f"decode={r['decode_tokens']/max(r['decode_s'],1e-6):>3.0f}t/s  "
                  f"peak={r['peak_mb']:>5.0f}MB")
        n = len(CASES)
        summary[variant.name] = {
            "name_pass": f"{name_passes}/{n}",
            "full_pass": f"{passes}/{n}",
            "mean_prefill": sum(prefill_rates) / max(len(prefill_rates), 1),
            "mean_decode": sum(decode_rates) / max(len(decode_rates), 1),
            "peak_mb_max": max(peaks) if peaks else 0,
            "peak_mb_mean": sum(peaks) / max(len(peaks), 1),
        }

    print("\n== cross-variant summary ==")
    print(f"{'variant':<14} {'name':<6} {'full':<6} {'prefill':<10} {'decode':<8} {'peak_max':<10} {'peak_mean':<10}")
    for name, s in summary.items():
        print(f"{name:<14} {s['name_pass']:<6} {s['full_pass']:<6} "
              f"{s['mean_prefill']:>4.0f} t/s   "
              f"{s['mean_decode']:>3.0f} t/s   "
              f"{s['peak_mb_max']:>5.0f}MB    {s['peak_mb_mean']:>5.0f}MB")


if __name__ == "__main__":
    main()
