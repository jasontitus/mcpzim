# Bonsai 27B — the full quant × runtime matrix (1-bit / ternary × llama.cpp / MLX)

**Date:** 2026-07-19 · **Machine:** Mac Studio M1 Ultra 128 GB · **Branch:** `bonsai-mlx-compare`
**MLX:** PrismML-Eng/mlx-swift fork `prism` (1-bit kernels) · hybrid-cache reuse ENABLED (stale guard retired)

Same ChatML template, same sampling recipe (temp 0.7 · top-p 0.8 · top-k 20 ·
presence 1.5), same three grounded turns against the full
`wikipedia_en_all_nopic_2026-06` ZIM. One process per leg. Reproduce:

```sh
tools/bonsai-ab/compare.sh \
  --zim ~/Downloads/wikipedia_en_all_nopic_2026-06.zim \
  --turn "Tell me about gravitational waves" \
  --turn "When were they first detected?" \
  --turn "How are they detected?"
# legs default to q1-gguf,ternary-gguf,q1-mlx,ternary-mlx
```

| turn | leg | reused/prompt tok | prefill s | TTFT s | out tok | decode tok/s | total s | footprint MB | stop |
|---|---|---|---|---|---|---|---|---|---|
| Tell me about gravitational waves | q1-gguf | 0/699 | 1.739 | 2.430 | 98 | 33.80 | 5.300 | 875 | eog |
| When were they first detected? | q1-gguf | 797/982 | 0.010 | 0.733 | 61 | 31.92 | 2.613 | 889 | eog |
| How are they detected? | q1-gguf | 1043/1250 | 0.008 | 0.800 | 79 | 32.83 | 3.176 | 889 | eog |
| Tell me about gravitational waves | ternary-gguf | 0/699 | 1.843 | 2.567 | 75 | 24.54 | 5.582 | 1164 | eog |
| When were they first detected? | ternary-gguf | 774/959 | 0.007 | 0.755 | 22 | 22.74 | 1.679 | 1176 | eog |
| How are they detected? | ternary-gguf | 981/1188 | 0.005 | 0.845 | 26 | 22.85 | 1.939 | 1176 | eog |
| Tell me about gravitational waves | q1-mlx | 0/699 | 3.384 | 3.666 | 142 | 8.38 | 20.708 | 5287 | eos_or_max |
| When were they first detected? | q1-mlx | 841/1026 | 0.879 | 1.056 | 80 | 9.87 | 9.156 | 5375 | eos_or_max |
| How are they detected? | q1-mlx | 1106/1313 | 0.879 | 1.120 | 120 | 9.37 | 13.934 | 5478 | eos_or_max |
| Tell me about gravitational waves | ternary-mlx | 0/699 | 3.183 | 3.420 | 75 | 10.76 | 10.387 | 8505 | eos_or_max |
| When were they first detected? | ternary-mlx | 774/959 | 0.797 | 0.961 | 32 | 11.06 | 3.866 | 8522 | eos_or_max |
| How are they detected? | ternary-mlx | 991/1198 | 0.877 | 1.088 | 68 | 10.29 | 7.707 | 8669 | eos_or_max |

**q1-gguf** · turns=3 · median TTFT=0.80s · median decode=32.8 tok/s · peak footprint=889 MB

**ternary-gguf** · turns=3 · median TTFT=0.84s · median decode=22.9 tok/s · peak footprint=1176 MB

**q1-mlx** · turns=3 · median TTFT=1.12s · median decode=9.4 tok/s · peak footprint=5478 MB

**ternary-mlx** · turns=3 · median TTFT=1.09s · median decode=10.8 tok/s · peak footprint=8669 MB


## Conclusions

1. **Quant axis (llama.cpp):** 1-bit is the right phone operating point —
   ~43 % faster decode (32.8 vs 22.9 tok/s), same sub-second follow-up
   TTFT, ~25 % lower footprint, half the weights on disk. Quality
   near-parity per the 16-scenario grid suite.
2. **Quant axis (MLX):** decode is kernel-bound, not bandwidth-bound —
   1-bit (9.4 tok/s) is no faster than ternary (10.8) despite half the
   bytes. 1-bit's real MLX win is memory: 5.3–5.5 GB in-process vs
   8.5–8.7 GB ternary (consistent with Prism's 5.9 GB @ 4K phone figure).
3. **Runtime axis:** llama.cpp wins decode decisively on Apple Silicon —
   3.5× on 1-bit (32.8 vs 9.4 tok/s), 2.1× on ternary (22.9 vs 10.8).
   With the hybrid-cache fix, MLX follow-up TTFT is competitive
   (~1.0–1.1 s vs 0.8 s), so the conversational gap is decode speed, not
   latency structure.
4. **Fork cost (flag):** swapping mlx-swift to the Prism fork to enable
   1-bit REGRESSED ternary MLX decode ~35 % vs upstream 0.31.3 on this
   Mac (16.8 → 10.8 tok/s, same leg, same machine conditions — llama
   legs varied <5 % between runs). The fork buys capability (bits=1),
   not speed; its merge base also predates upstream 0.31.6's iOS build
   fix. If MLX ternary ever matters for shipping, pin upstream and skip
   1-bit, or get Prism to re-sync.
5. **Cache reuse works on all four legs** (reused>0 on every follow-up)
   after retiring the stale MambaCache guard — verified against the
   original Qwen 3.5 4B crasher too. Per-family kill-switch remains via
   `ModelTemplate.hasStaleScratchStateBug`.

## Ship position

Unchanged: **Bonsai 27B 1-bit GGUF on llama.cpp** — fastest decode,
sub-second follow-ups, smallest weights, safest phone memory. The MLX
stack is now a fully instrumented, honestly comparable alternative
rather than a guess.

## Instrumentation (permanent, this branch)

Both providers emit identical `perf start/prefill/first token/complete`
lines and populate a shared `GenerationStats`; ChatSession logs one
`[Perf]` row per generation on both the tool-loop and grounded-discuss
paths; `--probe-discuss --runtime llamacpp|mlx` selects the runtime;
`tools/bonsai-ab/compare.sh` merges any leg set into this table. MLX
operating points live only in the harness — the picker entry was retired
with the experiment once llama.cpp won the A/B.

## Dependency state after this work

- **llama.cpp:** PrismML fork xcframework `prism-b9591` — rechecked the
  Prism branch on 2026-08-10. Its eight commits since our pin affect
  CUDA, x86, server behavior, packaging, and DSpark integrated-GPU
  selection; its Metal kernels are unchanged. Stock llama.cpp now supports
  Bonsai Q1 and is worth a controlled A/B once the machine is idle, but the
  existing Mac ternary GGUF uses Prism's fork-only group-128 Q2 format.
- **mlx-swift:** back on upstream (`.upToNextMinor("0.31.3")` /
  kokoro-ios `from: "0.29.1"`). The Prism fork pin used for the q1-mlx
  leg (revision e40e0a57a6f7ad08dc3fd87ad598a7aa6407d230) was retired
  with the experiment; re-pin BOTH Package.swift files by revision to
  reproduce the 1-bit MLX rows.
- **mlx-swift-lm:** vendored, refreshed with Bonsai; its Qwen35 classes
  are what fixed hybrid-cache resumption.
