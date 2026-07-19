# Bonsai 27B — MLX vs llama.cpp runtime A/B

**Date:** 2026-07-19 · **Machine:** Mac Studio M1 Ultra 128 GB · **Branch:** `bonsai-mlx-compare`

Same weights family (Prism's ternary Bonsai 27B), same ChatML template, same
sampling recipe (temp 0.7 · top-p 0.8 · top-k 20 · presence 1.5), same three
grounded turns against the full `wikipedia_en_all_nopic_2026-06` ZIM. Each
runtime ran in its own process (no shared Metal pool). Reproduce with:

```sh
tools/bonsai-ab/compare.sh \
  --zim ~/Downloads/wikipedia_en_all_nopic_2026-06.zim \
  --gguf ~/Library/Caches/huggingface/hub/models--prism-ml--Ternary-Bonsai-27B-gguf/snapshots/main/Ternary-Bonsai-27B-Q2_0.gguf \
  --turn "Tell me about gravitational waves" \
  --turn "When were they first detected?" \
  --turn "How are they detected?"
```

## Results

| turn | runtime | reused/prompt tok | prefill s | TTFT s | out tok | decode tok/s | total s | footprint MB | stop |
|---|---|---|---|---|---|---|---|---|---|
| Tell me about gravitational waves | llamacpp | 0/699 | 1.829 | 2.551 | 75 | 23.57 | 5.691 | 1162 | eog |
| When were they first detected? | llamacpp | 774/959 | 0.005 | 0.754 | 22 | 23.76 | 1.638 | 1171 | eog |
| How are they detected? | llamacpp | 981/1188 | 0.005 | 0.833 | 26 | 23.96 | 1.877 | 1171 | eog |
| Tell me about gravitational waves | mlx | 0/699 | 4.936 | 5.155 | 84 | 17.47 | 9.976 | 8342 | eos_or_max |
| When were they first detected? | mlx | 0/968 | 3.912 | 4.066 | 33 | 17.53 | 5.960 | 8659 | eos_or_max |
| How are they detected? | mlx | 0/1208 | 4.653 | 4.792 | 43 | 17.09 | 7.319 | 8590 | eos_or_max |

**Medians:** llama.cpp — TTFT 0.83 s, decode 23.8 tok/s · MLX — TTFT 4.79 s,
decode 17.5 tok/s.

## Reading the numbers

1. **Cross-turn KV reuse is the decisive gap.** llama.cpp reuses the
   conversation prefix (774 and 981 tokens reused on the follow-ups →
   0.005 s prefill, sub-second TTFT). MLX shows `reused=0` on every turn:
   Bonsai's hybrid linear-attention architecture trips the
   mlx-swift-lm#157 stale-state guard, so every follow-up re-prefills the
   whole transcript. The gap *grows with context* — at a 4K-token
   discussion the MLX follow-up TTFT extrapolates to ~15 s while
   llama.cpp stays sub-second.
2. **Steady decode: llama.cpp ~36 % faster** (23.8 vs 17.5 tok/s) for the
   ternary quant on M1 Ultra.
3. **Footprint columns are NOT directly comparable on macOS.** llama.cpp
   mmaps the 6.8 GB GGUF (weights live in the page cache, outside
   phys_footprint → 1.2 GB shown); MLX materializes weights in-process
   (8.3–8.7 GB shown, matching Prism's "~8.4 GB at 4K" figure). On iOS,
   jetsam counts both — Prism's phone numbers (5.2 GB GGUF vs 5.9 GB MLX
   for the 1-bit pack) remain the honest device comparison.
4. **Answer quality is equivalent** — both runtimes produced correct,
   grounded answers (Einstein prediction, binary-pulsar indirect
   evidence, 2015 LIGO detection) from the same retrieved passages.

## Verdict

The GGUF/llama.cpp runtime keeps its ship position on every axis that
matters for conversation: 6× faster follow-up TTFT (the walking-companion
metric), ~36 % faster decode, and the safer phone memory profile. The MLX
runtime is competitive for single-shot prompts but pays full re-prefill
every turn until mlx-swift-lm#157 is fixed upstream for hybrid caches.

## What it took / follow-ups

- The registered MLX entry is the **ternary 2-bit** pack (stock MLX affine
  quant). The phone-class **1-bit pack cannot load on stock mlx-swift
  0.31.x** — mlx-c rejects `bits=1` (supported: 2,3,4,5,6,8); the 1-bit
  kernels live in `PrismML-Eng/mlx-swift` branch `prism`. A phone-side MLX
  A/B needs that dependency swap (and would also need the #157 hybrid-cache
  fix to be conversationally competitive).
- Instrumentation shipped on this branch: both providers emit identical
  `perf start/prefill/first token/complete` lines and populate a shared
  `GenerationStats`; ChatSession logs one `[Perf]` row per generation
  (`runtime= model= prompt/reused/prefill/ttft/out/decode/total/footprint/
  stop`) on both the tool-loop and grounded-discuss paths;
  `--probe-discuss --runtime llamacpp|mlx` selects the runtime and
  `tools/bonsai-ab/compare.sh` merges two runs into this table.
