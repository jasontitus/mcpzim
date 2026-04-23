#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Memory bench for Gemma 4. The mlx-community Gemma 4 weights are packaged
# as multimodal `Gemma4ForConditionalGeneration` — plain `mlx_lm.load()`
# rejects them with "Received 140 parameters not in model" because its
# text-only Gemma4 model doesn't know about Gemma 4's layer-group shared
# K/V weights. `mlx_vlm.load()` handles them natively.
#
# We ignore images/audio here (text-only preamble + text-only decode) so
# the MLX memory numbers are apples-to-apples with bench_memory.py's
# Gemma 3 / Qwen 3 results.
#
# Usage:
#   source tools/llm-smoke/.venv/bin/activate
#   python tools/llm-smoke/bench_memory_gemma4.py --sizes 7000,20000,40000

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_vlm import load, stream_generate

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from bench_memory import FILLER_PARAGRAPHS, BASE_INSTRUCTIONS


def build_preamble(tokenizer, target_tokens: int) -> str:
    chunks = [BASE_INSTRUCTIONS]
    i = 0
    while True:
        ids = tokenizer.encode("".join(chunks))
        if len(ids) >= target_tokens:
            break
        chunks.append(FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)] + "\n\n")
        i += 1
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def mem_snap(tag: str) -> dict:
    return {
        "tag": tag,
        "active_mb": mx.get_active_memory() / 1024 / 1024,
        "peak_mb": mx.get_peak_memory() / 1024 / 1024,
        "cache_mb": mx.get_cache_memory() / 1024 / 1024,
        "rss_mb": (psutil.Process().memory_info().rss / 1024 / 1024
                   if _HAS_PSUTIL else None),
    }


def _p(s):
    rss = f"  rss={s['rss_mb']:.0f}" if s.get("rss_mb") is not None else ""
    print(f"  [{s['tag']:<20}] active={s['active_mb']:>5.0f}MB  "
          f"peak={s['peak_mb']:>5.0f}MB  cache_buf={s['cache_mb']:>4.0f}MB{rss}")


@dataclass
class Variant:
    name: str
    gen_kwargs: dict


VARIANTS = {
    "default": Variant("default", {}),
    # iOS app's production config: 4-bit groupwise KV quant from the first
    # token. Gemma 4 shipped through `Gemma4SwiftCore` + our Gemma4Text.swift
    # patch uses this when `DeviceProfile.useQuantizedKVCache` is true.
    "kv4": Variant("kv4", {"kv_bits": 4, "kv_group_size": 64,
                            "quantized_kv_start": 0}),
}


def run(model_id: str, variant_name: str, target_tokens: int) -> dict:
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, processor = load(model_id)
    load_s = time.perf_counter() - t0
    s_load = mem_snap("post-load")
    _p(s_load)

    # mlx-vlm's processor exposes the tokenizer indirectly — most processors
    # have a `.tokenizer` attribute; some put the text encoder behind
    # `.processor.tokenizer`. Probe.
    tokenizer = getattr(processor, "tokenizer", None) or processor
    preamble = build_preamble(tokenizer, target_tokens)
    prompt_ids = tokenizer.encode(preamble)
    print(f"  preamble: {len(prompt_ids)} tokens (target {target_tokens})")

    mx.reset_peak_memory()
    variant = VARIANTS[variant_name]

    t_start = time.perf_counter()
    first_tok_at = None
    decoded_n = 0
    # mlx_vlm.stream_generate yields GenerationResult objects with .text.
    for resp in stream_generate(
        model, processor, preamble,
        max_tokens=20, temperature=0.0,
        **variant.gen_kwargs,
    ):
        if first_tok_at is None:
            first_tok_at = time.perf_counter()
        # Different mlx-vlm versions yield different shapes; handle both.
        text = getattr(resp, "text", None) if hasattr(resp, "text") else str(resp)
        if text:
            decoded_n += 1
    t_end = time.perf_counter()
    ft = first_tok_at or t_end
    s_dec = mem_snap("post-decode")
    _p(s_dec)
    prefill_s = ft - t_start
    decode_s = max(t_end - ft, 1e-6)

    return {
        "model_id": model_id,
        "variant": variant_name,
        "target": target_tokens,
        "prompt_tokens": len(prompt_ids),
        "load_s": load_s,
        "load": s_load,
        "post_decode": s_dec,
        "prefill_s": prefill_s,
        "decode_tokens": decoded_n,
        "decode_s": decode_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",
                    default="mlx-community/gemma-4-e2b-it-4bit,mlx-community/gemma-4-e4b-it-4bit")
    ap.add_argument("--variants", default="default,kv4")
    ap.add_argument("--sizes", default="7000,20000,40000")
    args = ap.parse_args()

    model_ids = [m.strip() for m in args.models.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]

    results = []
    for model_id in model_ids:
        for variant_name in variants:
            for target in sizes:
                label = f"{model_id.split('/')[-1]} / {variant_name} / {target} tok"
                print(f"\n### {label}")
                try:
                    r = run(model_id, variant_name, target)
                    results.append(r)
                    p_tps = r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
                    d_tps = r["decode_tokens"] / max(r["decode_s"], 1e-6)
                    print(f"  prefill: {r['prefill_s']*1000:.0f}ms "
                          f"({p_tps:.0f} tok/s),  decode: {d_tps:.0f} tok/s")
                except Exception as e:
                    print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
                    results.append({"error": str(e), "label": label})

    print("\n== memory scorecard (Gemma 4) ==")
    print(f"{'model':<36} {'variant':<10} {'tok':>6}  "
          f"{'load':>5}  {'peak':>5}  {'active.dec':>10}  "
          f"{'prefill':>8}  {'decode':>7}")
    for r in results:
        if "error" in r:
            print(f"ERROR {r.get('label','?')}: {r['error']}")
            continue
        mid = r["model_id"].split("/")[-1]
        p_tps = r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
        d_tps = r["decode_tokens"] / max(r["decode_s"], 1e-6)
        print(
            f"{mid:<36} {r['variant']:<10} {r['prompt_tokens']:>6}  "
            f"{r['load']['active_mb']:>5.0f}  "
            f"{r['post_decode']['peak_mb']:>5.0f}  "
            f"{r['post_decode']['active_mb']:>10.0f}  "
            f"{p_tps:>5.0f} t/s  {d_tps:>4.0f} t/s"
        )


if __name__ == "__main__":
    main()
