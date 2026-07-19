# Bonsai 27B — quant × runtime comparison (1-bit / ternary, llama.cpp / MLX)

**Date:** 2026-07-19 · **Machine:** Mac Studio M1 Ultra 128 GB · **Branch:** `bonsai-mlx-compare`

Same ChatML template, same sampling recipe (temp 0.7 · top-p 0.8 · top-k 20 ·
presence 1.5), same three grounded turns against the full
`wikipedia_en_all_nopic_2026-06` ZIM. One process per leg. Reproduce:

```sh
tools/bonsai-ab/compare.sh \
  --zim ~/Downloads/wikipedia_en_all_nopic_2026-06.zim \
  --turn "Tell me about gravitational waves" \
  --turn "When were they first detected?" \
  --turn "How are they detected?"
# legs default to q1-gguf,ternary-gguf,ternary-mlx; see --legs
```

## Results (MLX hybrid-cache reuse ENABLED — see below)

| turn | leg | reused/prompt tok | prefill s | TTFT s | out tok | decode tok/s | total s | footprint MB | stop |
|---|---|---|---|---|---|---|---|---|---|
| Tell me about gravitational waves | q1-gguf | 0/699 | 1.731 | 2.418 | 98 | 35.34 | 5.163 | 875 | eog |
| When were they first detected? | q1-gguf | 797/982 | 0.005 | 0.718 | 61 | 33.78 | 2.494 | 879 | eog |
| How are they detected? | q1-gguf | 1043/1250 | 0.006 | 0.795 | 79 | 34.12 | 3.081 | 879 | eog |
| Tell me about gravitational waves | ternary-gguf | 0/699 | 1.828 | 2.550 | 75 | 23.69 | 5.673 | 1162 | eog |
| When were they first detected? | ternary-gguf | 774/959 | 0.006 | 0.753 | 22 | 23.75 | 1.637 | 1169 | eog |
| How are they detected? | ternary-gguf | 981/1188 | 0.005 | 0.832 | 26 | 23.94 | 1.876 | 1169 | eog |
| Tell me about gravitational waves | ternary-mlx | 0/699 | 4.358 | 4.480 | 74 | 16.51 | 8.972 | 8519 | eos_or_max |
| When were they first detected? | ternary-mlx | 773/958 | 0.798 | 0.931 | 106 | 16.94 | 7.197 | 8481 | eos_or_max |
| How are they detected? | ternary-mlx | 1064/1271 | 0.854 | 1.022 | 108 | 16.81 | 7.457 | 8641 | eos_or_max |

**Medians:** q1-gguf — TTFT 0.80 s, decode 34.1 tok/s · ternary-gguf — TTFT
0.83 s, decode 23.8 tok/s · ternary-mlx — TTFT 1.02 s, decode 16.8 tok/s.

## Quant axis: 1-bit vs ternary (same runtime, llama.cpp)

- **1-bit decodes ~42 % faster** (34.1 vs 23.8 tok/s) with identical
  sub-second follow-up TTFT and ~25 % lower process footprint
  (879 vs 1169 MB; weights 3.6 vs 6.8 GB on disk).
- Prefill speed and prefix reuse are equivalent.
- Quality: this 3-turn set produced correct grounded answers from both; the
  16-scenario Mac conversational suite (GRID_RESULTS_BONSAI_* in
  tools/llama-smoke) scored the two at near-parity with one wrong-tool turn
  difference, which deterministic routing already covers. Ternary's case is
  a quality-insurance margin, not a measured win.

## Runtime axis: llama.cpp vs MLX (same ternary weights)

- **First run of this A/B exposed a stale guard, not a runtime property.**
  MLX follow-ups initially re-prefilled everything (`reused=0`, TTFT
  3.9–4.8 s) because Gemma4Provider force-disabled cache reuse for any
  hybrid (MambaCache) model — a workaround for the old mlx-swift-lm#157
  `broadcast_shapes` SIGABRT. The vendored library refresh that came with
  Bonsai fixed resumption for the Qwen35 classes; we verified empirically
  (original crasher Qwen 3.5 4B: 835-token reuse, no abort; Bonsai:
  838/1071-token reuse, answers unchanged) and retired the guard
  (per-family kill-switch kept via `ModelTemplate.hasStaleScratchStateBug`).
- **With reuse enabled, MLX is conversationally competitive:** follow-up
  TTFT 0.93–1.02 s vs 0.75–0.83 s GGUF. llama.cpp keeps a ~42 % decode
  advantage on ternary (23.8 vs 16.8 tok/s) and its first-turn prefill is
  ~2.4× faster (1.8 vs 4.4 s).
- **Footprint columns are NOT comparable on macOS**: llama.cpp mmaps the
  GGUF (weights live in the page cache, outside phys_footprint); MLX
  materializes weights in-process (~8.5 GB shown, matching Prism's ~8.4 GB
  figure). Prism's phone numbers (5.2 GB GGUF vs 5.9 GB MLX, 1-bit)
  remain the honest device comparison.
- Answer quality equivalent across all legs on these turns.

## Verdict

- **Quant:** 1-bit GGUF is the right phone operating point — strictly
  faster and smaller, with quality near-parity per the grid suite.
- **Runtime:** llama.cpp stays the ship choice (decode + prefill +
  device-memory margins), but the gap is now honest rather than
  structural: the MLX cache fix took MLX follow-ups from 4× slower to
  ~1.2× slower. The missing cell — **1-bit MLX** — still needs
  `PrismML-Eng/mlx-swift` branch `prism` (stock mlx-c rejects `bits=1`;
  supported: 2,3,4,5,6,8).

## Instrumentation shipped on this branch

Both providers emit identical `perf start/prefill/first token/complete`
lines and populate a shared `GenerationStats`; ChatSession logs one
`[Perf]` row per generation on both the tool-loop and grounded-discuss
paths; `--probe-discuss --runtime llamacpp|mlx` selects the runtime;
`tools/bonsai-ab/compare.sh --legs q1-gguf,ternary-gguf,ternary-mlx`
merges any set of legs into this table. The Mac model picker carries
"Bonsai 27B Ternary (2-bit · MLX · Mac)" beside its GGUF sibling for
interactive A/Bs.
