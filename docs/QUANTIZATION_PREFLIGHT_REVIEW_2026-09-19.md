# Quantization preflight adversarial review

September 19, 2026. Manual adversarial review of the implementation and plan;
not an independent reviewer or a completed GPU/model experiment.

The fixed teacher and quantization source is original full-precision
Qwen3.8-27B. Success means beating the current Bonsai on held-out Zimfo tasks
preferably at the same or smaller total deployed size, with a modest increase
eligible for consideration when the measured quality gain is substantial. No Bonsai reference activations.

| Finding | Action / status |
| --- | --- |
| Projection tests could be mistaken for full 27B RCO feasibility | Reports explicitly identify synthetic projection scope. Added real candidate-generation, whole-model backward/update and packing/product gates. **Full-model feasibility is still open.** |
| A 1.58-bit 27B artifact already exceeds the roughly 3.8 GB goal before metadata | Added serialized-cost accounting before long optimization; Bonsai is a preferred size anchor, not an absolute cutoff. Predominantly one-bit encoding and head/embedding treatment remain unresolved. **Do not claim target-size feasibility.** |
| Saving GPU candidate differences for backward could erase the memory savings | Custom backward reloads one row tile at a time and contracts its gradient. Host weights remain resident; a bounded disk-backed cache is still needed for full-model work. |
| Zero-probability candidates could incorrectly receive zero derivatives | Added an explicit adversarial test plus dense-autograd comparisons. Passed. |
| BF16 accumulation could erase small allocation gradients | Accumulate tile contractions/input-gradient sums in FP32. CPU BF16 parity checked; CUDA BF16 parity must pass before target memory measurements. |
| A CPU fallback, NaN, empty result or interrupted run could falsely pass | Target requires CUDA/BF16. Fail on nonfinite values. Reports begin running, become failed on exception, and are written atomically. Launcher validates returned CUDA profile, precision checks, shapes and memory fields. Tests cover CPU/incomplete/running reports. |
| Repeated steps might retain live tensors | Target runs three iterations per shape and checks post-release CUDA allocations. PyTorch allocator peaks are measured; they do not include every driver allocation. Initial nvidia-smi is diagnostic, not a complete device-memory peak measurement. **Whole-block retention testing remains open.** |
| SSH success could be confused with successful tests | Launcher requires a validated returned CUDA report in addition to zero remote exit status. |
| VM can continue billing after client crash, preemption or failed SSH | Server-side one-hour DELETE policy, no automatic restart or resource retries, auto-delete boot disk. Client cleanup also runs after errors. No paid resources created during preparation. |
| Cleanup could delete another VM, or miss a disk after preemption deletes the VM | Random run name and ownership label required for VM deletion. Always verify boot disk absence. Refuse ambiguous cleanup and record it as unverified. Tests cover unrelated VM and orphan disk. |
| A delayed create response could leave resource state uncertain | Try ownership-checked cleanup after a failed create request. API-side runtime limit remains the final bound. An unverifiable API response is reported, not hidden as confirmed deletion. |
| Returning an archive could overwrite local files | Read only the expected regular JSON member, reject duplicate/oversized entries, never extract the archive automatically. |
| Upload could accidentally include private app logs or credentials | Explicit package allowlist: synthetic harness, bootstrap and numerical tests. No model, app content, venv, authentication or conversation logs. VM has no service account. |
| Stale/partial prices could understate cost | Use directly fetched whole-VM Spot rate and add disk/IP. Distinguish ~$0.53 one-hour preflight estimate from unmeasured full optimization. Recheck pricing before approved launch. |
| Spot preemption could force repeated spending | No automatic recreation, fallback region or larger-GPU substitution. Missing results fail; another paid attempt requires an explicit decision. |

Validation performed locally: 20 tests passed, one Metal test skipped; CPU tiny
profile completed; CPU BF16 relative gradient errors below 0.3%; bootstrap shell
syntax passed; GCP package preparation performed without execution. GPU tests,
actual 27B blocks, full RCO, packing/runtime integration and target model quality
have not passed because they have not run.

The one-hour L4 preflight is a cheap primitive check. It must not become a
substitute for answering whether the candidate representation has a useful measured size/quality tradeoff and whether real GSQ/RCO steps on the chosen model work. See the
[decision gates](QUANTIZATION_PLAN_2026-09-19.md).

## Input capture and comparison review (September 19, implementation pass)

Manual adversarial review, not an independent agent review:

| Finding | Resolution or remaining gap |
|---|---|
| Prefix-cache suffixes omit necessary instructions/history | Record the complete rendered input and full originating token IDs before suffix selection in both providers. Unicode/follow-up round-trip test passed. |
| Chat catches a recorder error and continues, falsely completing a partial dataset | Sticky recorder failures cause CLI completion to fail. Missing-context and forced disk-write failure tests passed. |
| Rerun mixes records with previous work | Reject existing capture directories; atomic records; unfinished manifest stays running. Test passed. |
| Capture accidentally becomes routine private logging | No environment-based activation; explicit evaluation CLI flag only; default-disabled path tested; capture directory created with private permissions. |
| Bonsai token IDs or quantized activations get treated as Qwen reference data | Recorder is explicitly input-only, provenance pending. Full-precision runner must verify original artifacts and retokenization before reference extraction. Still open. |
| Better aggregate hides dropped cases, failed generations or wrong-source answers | Reporter rejects missing/duplicate/skipped cases, includes failures at zero, and displays category deltas and new critical failures. Tests passed. |
| Slightly larger but much better candidate is discarded | Bonsai is a comparison anchor; report exact size delta without automatic rejection or promotion. Test passed. |
| A different checkpoint is mislabeled as compressed Qwen | Candidate source model/revision must match the original reference. Attested hashes still need verification by real runner. |
| Positive retention ratio is misleading when Qwen has no quality headroom | Ratio is null unless Qwen exceeds Bonsai; retain raw score differences. Test passed. |
| CLI MLX branch labels/settings silently treat Qwen as Bonsai | Existing branch limitation documented; full-precision Qwen adapter is not yet implemented. Do not report baseline quality from this path yet. |

Validation: five Swift recorder tests passed; Python harness suite passed 34 tests
with one Metal test skipped, then the expanded comparison module passed all 15
tests (one additional source-revision rejection case). The headless Mac
`MCPZimEvalCLI` build succeeded. Full weights, reference activations, real quality
comparisons and GPU optimization remain unexecuted.

Real CLI integration also passed: two ad-hoc public-topic development turns
recorded two actual llama.cpp invocations; full UTF-8 hashes verified and the
manifest completed. The deterministic preparation turn emitted no model record,
while the follow-up emitted two, confirming turns cannot be used as invocation
counts. No rubric scores were assigned. The reported macOS footprint is not a
complete model-memory measure because mmap-backed weights are under-counted;
do not use it to claim phone memory feasibility.

## Readiness adversarial follow-up

The old launcher installed dependencies after GPU boot and retained results only
on the disposable VM. It is now disabled before any cloud call, and the guest
script contains no package installation. This closes accidental use of that
path; it does not implement the replacement full-model runner. The new readiness
runbook requires an immutable prebuilt environment, prestaged inputs, separate
durable checkpoint storage, manifest-last commits and a tested restore of actual
optimizer/RNG state. A dependency audit also found the real solver pins differ
from the synthetic probe; those images must not be conflated.

The baseline verifier now writes a fresh incomplete status before validation
and a failed status on errors, so corrupt input cannot leave a stale successful
report. Tests cover corruption, missing shards, wrong revisions, quantized
configuration (including empty config), missing coverage and packed integer
weights. All 18 real original shards passed hash/shape/dtype verification.

## Full-model launch review, later September 19

This supersedes the earlier unexecuted-Mac status above. Full Mac calibration
and paired comparison completed; see `benchmarks/quantization-2026-09-19/`.
The committed GCS input package contains 114 bulk data objects, approximately
73.66 GiB, plus one final manifest. A full 27B RTN compatibility file loads and
prefills in the shipped runtime. That fixture does not establish learned
quantization quality.

Parallel independent reviews of the new CPU/GPU preparation paths found and
resolved these issues:

| Finding | Resolution |
|---|---|
| CPU restorer invoked from wrong container directory | Run `python -m restore_inputs` through the image's explicit package path |
| Metadata bootstrap executed without checking its staged identity | Hash verification before execution |
| Timeout could kill Docker client while leaving container running | Unique container name and bounded stop in `finally`; independent guest and API time limits |
| Frozen baseline integrity assumed read-only files, but mount was writable | Nested read-only prepared-input mount, separate writable output area |
| Exit zero could be mistaken for full-model feasibility | Require original-model identity, largest sequence, both hybrid GSQ block types, full-vocabulary RCO gradients, measured CUDA memory, and matching checkpoint next update |
| Checkpoint report could reference absent remote commit | Independently download commits using their exact generation and verify size/SHA256 |
| Spot loss or VM cleanup could delete prepared inputs/progress | Both disks have auto-delete disabled; cleanup checks ownership and disk IDs and verifies detachment |
| Retrying ambiguous create could rent a duplicate VM | Persist launch attempt before create, refuse blind retries and any second existing quantization VM |

The GPU launcher has 12 local tests covering these boundaries, including failed
reports, stale manifests, wrong ownership, corrupt remote commits and immutable
image requirements. No test result here claims actual GPU memory feasibility.
The reviewed job runner and actual-model smoke must be included in the final
image, pass their integration tests, and complete CPU preparation before launch.

## Live-size transport finding and correction

The first GPU attempt exposed a performance gap not covered by the tiny
transport tests: remote SHA verification used1MiB GCS ranges. GPU billing was
stopped after about12minutes, with disks retained. No complete GSQ/RCO smoke
pass is claimed. GCS network chunking now uses64MiB, independently of the local
1MiB streaming hash buffer. Generation pinning, exact hashes and atomic commits
remain unchanged. Tests assert both upload and read SDK chunk sizes and retain
corruption, conflicting-generation, interrupted-upload and RNG invariants.

The new image passed41 solver plus45 transport/restore tests. The actual
4.634GB remote checkpoint was then read and SHA-verified on CPU in47.427seconds
(93.194MiB/s), before another GPU rental. The CPU benchmark fails readiness
below32MiB/s, preserves the existing filesystem and never runs quantization.
The guest used the original model's PyTorch fallback for linear attention;
optimized FLA/causal-convolution kernels were unavailable. This is a performance
limitation to measure, not evidence that full RCO passed.

## Second GPU attempt and timing/recovery review (September 20, 01:06 UTC)

The real 27B run completed both hybrid GSQ updates and reached the full-model
RCO restored-next-update comparison, which failed (76/996 allocation values,
maximum absolute difference 0.0011873245). The stopped VM has been deleted with
prepared disks retained. This gate remains failed; do not reinterpret finite
forward/backward execution as successful recovery or quality validation.

Independent agent review of the subsequent timing/recovery changes found and
then re-reviewed these fixes:

- Timing file/console errors could mask the original solver failure or prevent
  recording a successfully published checkpoint. Telemetry is now best-effort,
  emits an error diagnostic and preserves the solver/checkpoint state machine.
  Recovery diagnostic writes also preserve the original failure.
- Corpus propagation incorrectly squared total tokens for a quadratic workload
  statistic. It now records the sum of each sequence length squared.
- Publication-only timings omitted serialization, hashing and preparation.
  Enclosing checkpoint/stage wall timers retain those costs. They contain their
  child measurements and must not be summed with them.

Recovery diagnostics verify exact pre-update solver/Adam/scheduler/RNG state
and preserve both replay reports and memory on failure; existing post-update
numerical tolerances are unchanged. Strict deterministic CUDA algorithms,
cuDNN settings and cuBLAS workspace configuration are installed before CUDA
initialization. This may reject unsupported kernels or change performance.
It does not prove the failed run's cause or establish that replay now passes.

Validation: the final local Linux amd64 runtime build passed **48 solver tests
and 45 transport/restore tests**. The independent reviewer also ran 11 focused
tests and found no remaining blocker in the reviewed changes. Build evidence:
`tools/calibration/runs/timing-recovery-image-build-final-20260920.log`.
Local image `zimfo-quantization:timing-recovery-20260920`, manifest-list digest
`sha256:f13a4e99a85f4f650a8e3e130757fb4ddee891eeb8e34e6cea973e99dc56c3cd`.
This image has not been published/cached on the cloud disk or validated on CUDA.

The [timing report](QUANTIZATION_TIMING_2026-09-19.md) records measured values,
coverage gaps and the complete-recipe ETA method. No third GPU was launched.

## Third-attempt launch preparation

The user explicitly approved another one-hour Spot attempt. Runtime
`f13a4e99…` was published and cached on the existing boot disk using CPU-only
preparation v8; its 4.63 GB read/verification benchmark reached 110.65 MiB/s.
No package installation or image pull is required on the GPU.

The launcher now freezes and attaches the shutdown-diagnostic script before VM
creation. It binds to the current run metadata, publishes one bounded compressed
object with generation-zero creation, skips malformed/oversized reports while
retaining useful logs, and has a 25-second overall timeout plus 2-second kill
grace. Independent adversarial review found and verified fixes for malformed
JSON aborting the bundle and missing overall time bound. **21 tests passed.**
Shutdown delivery is best-effort and remains supplementary to committed periodic
checkpoints. No tokens, authentication files or environment dumps are included.

The independent offline timing analyzer has **26 passing tests**. It separates
smoke from production, preserves stage/type/token-bucket coverage, excludes
failed samples and never adds enclosing stage/checkpoint totals to child timers.
Missing work remains unknown. This analyzer does not change the running image.

### External continuation and live progress review (September 20 UTC)

The external controller preserves the running `f13a4e99…` solver image and frozen
selected-stage identity. Independent root verification passed 61 tests across
continuation and timing summaries, including actual tiny-model terminal GSQ
resume with zero new updates followed by learned head and RCO functions. Cloud
transport and container execution remain simulated in that test. The controller
is not yet wired into cloud bootstrap or validated on the full CUDA model;
see [continuation limits](QUANTIZATION_CONTINUATION.md).

Live monitoring exposed timing records obscuring the latest optimizer cursor.
The future-host bootstrap now skips records without a valid integer global step;
21 launcher tests passed and an independent adversarial review found no blocker.
The active VM's frozen scripts were not changed. The bounded 16 KiB log-tail
search can still omit a sufficiently old cursor rather than inventing progress.


The final independent third-attempt evidence review found no blocking overclaim:
87 embedding +174 GSQ durable updates, one excluded uncommitted update, nested
timing accounting, the conditional 229-second block interval extrapolation,
timeout diagnosis and compute-only cost scope all match retained evidence.
Commit generation/hash verification and observed VM stop timestamps are retained
with the run. Bulk checkpoint restore on a new host remains untested.

### Overlapped checkpoint and five-hour continuation review

The external controller now runs the pinned solver and a separate bulk GCS
publisher, keeping local progress distinct from cloud durability. Adversarial
review found and fixed expired-deadline process launches, orphaned containers
after Docker CLI exit, insufficient final manifest binding, overly broad
publisher cleanup identity, and recovery receipts losing diagnostics space to
performance reports. The numerical solver image is unchanged.

The targeted checkpoint/controller/cloud/staging suite passed 198 tests; two
subsequent host regressions cover cleanup identity and receipt priority. The
integration test uses actual tiny-model GSQ/head/RCO optimization and checkpoint
state, then a cold continuation; process and network boundaries are simulated.
Independent CPU staging review passed 11 tests with no blocking finding.
Actual cloud staging, disk capacity and overlapped throughput remain unverified.
See [optimization and ETA limits](QUANTIZATION_CHECKPOINT_OPTIMIZATION_2026-09-20.md).
