# Bonsai latency: physical iPhone investigation

Tested September 4–5, 2026 on the connected iPhone 17 Pro Max (A19 Pro,
iOS 26.6.1), with text-only launches. No microphone capture or speech
synthesis was requested. This extends the source audit in
[the September 4 review](BONSAI_LATENCY_REVIEW_2026-09-04.md).
The subsequent [adversarial review](ADVERSARIAL_REVIEW_2026-09-05.md) covers
the conversation changes as well as inference and benchmark failure paths.

The signed development build is installed with the pinned runtime, bounded
host-memory checkpoints, and 128-token prefill chunks for phone Bonsai Q1.
The final suite ran without checkpoint/batch overrides and passed every
output comparison. Normal app launch is restored after testing.

## Reproduction and measurement

The phone uses `Bonsai-27B-Q1_0.gguf` (3,803,452,480 bytes), 16,384 context
tokens, Q4 K/V, all layers on Metal, and flash attention. The controlled
suite uses 1,223–1,276 token synthetic encyclopedia prompts, greedy sampling
with the publisher sampling profile explicitly disabled, and at most 32
output tokens. Reference outputs are compared within each runtime for exact
retry, ordinary append, cancellation followed by retry, and edited answers.
The review added changed-evidence equality, explicit cold-reset assertions,
and a requirement that the interrupted run actually reports cancellation.
These are cache correctness checks, not factual-quality evaluations.

Build and signature verification follow [SIGNED_APP_BUILDS.md](SIGNED_APP_BUILDS.md).
The DEBUG launch hook is `MCPZIM_BENCH_LATENCY=1`. Additional launch variables:

| Variable | Purpose |
| --- | --- |
| `MCPZIM_LLAMA_RECOVERY_CHECKPOINT=0` / `1` | Override checkpoint use; phone Bonsai Q1 defaults on |
| `MCPZIM_LLAMA_RECOVERY_STORAGE=device` | Exercise backend device buffers; otherwise use host memory |
| `MCPZIM_LLAMA_PREFILL_BATCH=128` / `512` | Override prefill chunks; phone Bonsai Q1 defaults to 128, other configurations to 512 |
| `MCPZIM_BENCH_VERIFY_CURVE=1` | Measure target-only verification widths 1, 2, 4, 8 |
| `MCPZIM_BENCH_HOLD_OPEN_SECONDS=300` | Keep the phone available briefly for the next launch; capped at 30 minutes |

Pass a JSON dictionary through `devicectl device process launch
--environment-variables`. Benchmarks keep the idle timer disabled while
running, then restore its prior setting. Normal launches do not run them.
The existing question autorun also cancels its stop timer at the end of each
turn, preventing a delayed timer from accidentally stopping a subsequent turn.

Performance-only extracts are in
[benchmarks/bonsai-latency-2026-09-05](benchmarks/bonsai-latency-2026-09-05).
The extractor is `tools/llama-smoke/summarize-phone-latency.py`. It excludes
questions, replies, coordinates, and unrelated startup logs, and ignores the
previous-session tail copied into an unclean-launch diagnostic.

The prefill timer now synchronizes Metal before ending. Previously several
seconds of queued prefill appeared as time between prefill and first sampling.
TTFT remains comparable across this instrumentation change. Checkpoint timing
also separates queued prefill from the actual snapshot copy.
Reported process footprint is not total model residency or a measurement of
remaining jetsam headroom.

## Baseline and checkpoint

In the real Wikipedia path, Einstein's overview reached its first token in
9.83 s and decoded at 8.44 tokens/s. A school follow-up reused 786 tokens but
still introduced 585 new prompt tokens, producing an 8.52 s TTFT. Successful
reuse alone therefore does not guarantee an immediate follow-up: the amount
of new evidence remains important.

The controlled baseline reproduced the larger avoidable penalty:

| Case | Pinned runtime, checkpoint off: TTFT |
| --- | ---: |
| Cold, 1,223 tokens | 15.713 s |
| Exact retry | 18.118 s |
| Ordinary append, 21 fresh tokens | 0.821 s |
| Retry after cancellation | 21.882 s |
| Edited preceding answer | 22.498 s |

The baseline passed every output check. The phone moved from `fair` to
`serious` thermal state during this repeated-cold workload; these are
individual measurements, not confidence intervals or controlled speedup ratios.

The prototype saves one partial/recurrent sequence state immediately before
the final prompt token. It verifies both the incoming token prefix and the
still-live attention prefix before restoring, then removes the attention tail.
It falls back to a full reset for changed evidence, a missing/mismatched
prefix, a failed restore, or a checkpoint without a fresh token for logits.
Reset, prefix replacement, and model unload invalidate the checkpoint.
The real state is capped at 256 MiB; Bonsai requires 149.6 MiB.

An initial device-buffer snapshot reduced exact retry TTFT to 0.399 s and
preserved the output. Later cold-reference steps failed with `llama_decode
rc=-3`. The same failure reproduced with mainline b10816. Its console reported
Metal command buffers discarded during GPU error/recovery. This does not
establish whether thermal load, GPU scheduling, or device snapshot handling
caused the underlying recovery. A successful retry is insufficient to promote
that variant.

Both follow-up isolation tests passed all four output comparisons:

| Configuration | Retry | Interrupted retry | Edited answer | Result |
| --- | ---: | ---: | ---: | --- |
| Device snapshot, 128-token chunks | 0.216 s | 0.273 s | 1.144 s | Passed |
| Host snapshot, 512-token chunks | 0.210 s | 0.321 s | 1.174 s | Passed |
| Final defaults: host, 128-token chunks | 0.265 s | 0.316 s | 1.207 s | Passed |
| **After adversarial review, same defaults** | **0.206 s** | **0.335 s** | **0.902 s** | **Passed strengthened suite** |

The second run demonstrates that host snapshots can recover correctly even
with the original chunk size. The first demonstrates that smaller submissions
also survived this workload with device snapshots. Neither result alone proves
the exact cause of the earlier GPU recovery. The chosen phone default combines
host storage, which releases immediately on reset, with 128-token chunks.
Other models and Mac defaults are preserved. The final configuration passed
again without environment overrides. Ordinary append TTFT was 0.681 s;
cold TTFT was 13.721 s. All final cases started in `serious` thermal state.

Host snapshot data is 156,894,364 bytes, not a multi-gigabyte whole-model or
disk cache. The measured copy took roughly 0.01 s when reusing the buffer and
0.13 s following a reset; initial allocation is included. GPU state restoration
was roughly 0.1–0.15 s. Queued prefill work is timed separately.

## Current llama.cpp comparison

The shipping Prism pin is `62061f9` / `prism-b9591`; the candidate is upstream
`b10816` / `427291b5b3`. The candidate was built from source with the official
XCFramework script. Only the iOS slice was substituted for the experiment;
the original slice was restored afterward. Embedded framework UUIDs were
checked after signing:

- Prism: `A850FEF6-7894-3F80-AF70-151A866F1565`.
- b10816: `1059558A-733D-30A1-9C8A-1D9AD5665C90`.

The new sampler API needs a vocabulary size and explicit history length;
the `MCPZIM_LLAMA_UPSTREAM` build condition supplies these. Greedy probes
bypass that sampler. No global runtime upgrade is justified by these results:
the mainline checkpoint-off run passed correctness, but measured 20.492 s for
retry and 2.106 s for the ordinary append, with `serious` thermal state.
Thermal conditions differ from the initial baseline, so this is not a clean
regression percentage. It does rule out claiming a demonstrated large upgrade
win from this session. The existing Mac ternary group-128 GGUF also needs the
Prism fork; upstream's group-64 Q2 format is not interchangeable.

Relevant upstream changes actually included in the candidate are
[large-batch quantized-KV dequantization before attention](https://github.com/ggml-org/llama.cpp/pull/27390)
and [per-device flash-attention vector tuning](https://github.com/ggml-org/llama.cpp/pull/26570).
The console confirmed the candidate's new Q4-to-F16 attention path on A19 Pro.
These attention improvements do not remove the one-bit projection or hybrid
cache constraints.

## Speculation: measure verification before adding weights

The b10816 phone probe requests logits for every verification row, restores
the same context before each measurement, excludes a warm-up sweep, and
alternates ascending/descending width order. Medians of three timed sweeps:

| Width | Target verification | Equivalent single steps | Optimistic target-only ratio |
| --- | ---: | ---: | ---: |
| 1 | 0.133 s | 0.133 s | 1.00× |
| 2 | 0.472 s | 0.267 s | 1.77× cost |
| 4 | 0.671 s | 0.533 s | 1.26× cost |
| 8 | 1.363 s | 1.067 s | 1.28× cost |

Those **mainline** target costs cannot support a gain even with perfect
acceptance before adding draft execution and recurrent rollback. The subsequent
**Prism** run materially changed the assessment:

| Width | Pinned target verification | Cost in single-step equivalents |
| --- | ---: | ---: |
| 1 | 0.137 s | 1.00 |
| 2 | 0.214 s | 1.56 |
| 4 | 0.393 s | 2.88 |
| 8 | 0.484 s | 3.54 |

These are synthetic, thermally constrained curves, not complete drafter
benchmarks. The pinned Q1 path has **potential headroom**: at width 8, mean
useful output must exceed 3.54 tokens per round before accounting for any
drafter or rollback cost. The target-only perfect-acceptance ceiling is about
2.26×. That is not an expected speedup. The existing GGUF contains no
MTP/DSpark head tensors, and naive recurrent restore plus accepted-token
replay can consume the margin. A matched drafter, acceptance measurements on
Wikipedia/StreetZIM answers, and efficient recurrent rollback are prerequisites.
No speculative decoder is enabled by this change. These results favor working
from the pinned Q1 path rather than assuming newer general kernels improve it.

## What the MLX work contributes

There has been post-release Bonsai work outside Prism's original fork.
[oMLX's Bonsai dispatch](https://github.com/jundot/omlx/blob/main/omlx/utils/model_loading.py)
adds one-bit construction and inference support. Its
[actual one-bit kernel wrapper](https://github.com/jundot/omlx/blob/main/omlx/patches/bonsai_qmv.py)
still explicitly expands weights to float16 for larger prefill batches.
That is a significant constraint for a phone; a fast one-token kernel is not
enough for conversational TTFT.

[mlx-dspark's Bonsai measurements](https://github.com/ARahim3/mlx-dspark#prismml-bonsai-27b-ternary--1-bit-qwen36-27b)
report one-bit generation support through mlx-vlm 0.6.5, but speculative
decoding at only 0.71–0.77× baseline on an M4 Pro because verification rereads
weights per token. Its newer hybrid checkpoint cache is relevant; the modest
ternary-code gains concern a different, larger weight format. These are
maintainer measurements, not phone results reproduced here.

The strongest challenge transfer candidates are:

| Work | Application to Bonsai |
| --- | --- |
| Qwen fused GDN prework and normalization | Closest architectural fit: Bonsai has 48 recurrent layers and 16 full-attention layers. Benchmark fusion separately from quantized projections. |
| Stable hybrid prompt checkpoints | Directly applicable to turn recovery; prototype above uses llama.cpp's existing sequence-state API. |
| Reuse a weight decode across small verification batches | Necessary to change the phone cost curve, but must support Q1 group-128; Q4/Q8 kernels are not a drop-in. |
| Laguna final-prefill-row and projection fusion | Investigate redundant work and launches; preserve Bonsai's recurrent state updates and exact weight format. |
| Gemma/Laguna expert and cohort batching kernels | Lower relevance to one dense-model conversation on a phone. |

This transfer is more than hypothetical:
[oMLX credits its fused GDN prework to mlx-serve's port of the challenge kernel](https://github.com/jundot/omlx#acknowledgments).
The [Laguna challenge](https://github.com/Layr-Labs/mlxfast-challenge)
targets a much larger NVFP4 model on an M5 Max. The
[Gemma challenge](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine)
includes a cohort workload; leaderboard gains should not be used as expected
single-conversation iPhone gains. The useful next kernel project is bounded
Q1 prefill/small-batch matmul plus GDN fusion, with end-to-end phone TTFT and
correctness as the acceptance criteria.

## Final validation and remaining conversation work

- Signed iPhone build passed the repository signature gate, with the original
  Prism framework UUID verified after restoring the comparison slice.
- 155 targeted Swift tests passed after the adversarial review, including the checkpoint prefix guard and
  conversation routing/suggestion tests. `git diff --check` passed.
- Final default inference suite: exact retry, append, interrupted retry, and
  edited-answer output comparisons all passed; no fatal decode errors. The
  review rerun additionally passed changed evidence and cold resets, with
  cancellation confirmed after eight output pieces. Its records are in
  `benchmarks/bonsai-latency-2026-09-05/adversarial-defaults.json`.
- Real Wikipedia smoke: overview TTFT 8.017 s and school follow-up TTFT
  7.495 s. These individual runs are not a controlled speedup estimate.
- Real StreetZIM smoke: `near_named_place` returned cafes in 0.293 s and the
  map logged 25 rendered pins. No inference was needed for that direct route.
- Reviewed conversation smoke: discovery returned three distinct pages in
  0.364/0.073/0.395 s, using candidate cursors 0/7/13. Selecting “Option 1”
  between pages dispatched the exact Wikipedia archive and did not advance
  the discovery cursor. Nearby-story discovery and its exclusion-aware next
  batch returned in 0.121/0.023 s, respectively.
- Normal launch restored without benchmark flags; the repository health
  watcher confirmed the app remained alive for 45 seconds.

The Wikipedia smoke also reproduced an existing extraction-quality issue:
“Tell me about his parents” selected a sentence about the family business
instead of the parents' names. It occurs in the deterministic extraction path,
not the new checkpoint logic, and remains work for the conversation/evidence
selection track. Cold processing of substantial new evidence also remains
several seconds; the large improvement here concerns reusing existing work.

The later Duet smoke also generated a questionable contrast with harmony,
claiming that duet performers take turns playing solo sections. That turn
used a full 624-token prefill with zero reused tokens, so it did not exercise
checkpoint restoration. The model's factual faithfulness needs separate
evaluation. Topic variety also depends on the archive's main page: this
phone's first two batches mostly followed its featured Bach/music article.

## Source-grounding follow-up — September 5

The later physical-phone source audit confirmed that the loaded Wikipedia
ZIM itself contains the questionable duet “taking turns” sentence. That
example was a source error, not demonstrated model invention. Article
answers now use exact ZIM excerpts with an explicit missing-evidence response;
the parents-name selection was corrected as well. See
[SOURCE_GROUNDING_2026-09-05.md](SOURCE_GROUNDING_2026-09-05.md) and the
[full app review](APP_REVIEW_2026-09-05.md). The earlier raw-inference cache
measurements remain valid for that benchmark; ordinary article answers now
avoid that generation pass entirely.
