# Quantization launch readiness

Updated September 20, 2026, 02:10 UTC (September 19 Pacific).
The [approved experiment plan](QUANTIZATION_PLAN_2026-09-19.md) remains a
size-versus-quality comparison of original Qwen3.8-27B, Bonsai and learned
quantized Qwen candidates.

**Third attempt: actual-model GSQ/RCO and checkpoint-replay validation passed;
87 embedding updates and 174 GSQ updates are durable. The VM is deleted and no
GPU instances remain.** Both prepared disks and GCS commits are retained.

The exact pre-update solver/Adam/scheduler/RNG state checks and unchanged
post-update tolerances all passed. GSQ smoke peak allocated/reserved memory was
63.00/66.25 GiB; RCO was 58.61/59.23 GiB. This establishes the tested longest
4,673-token smoke operations, not completion of all production stages or quality.

VM `zimfo-gpu-6583ec8c4659` used cached runtime `f13a4e99…`. It ran about
47.8 minutes from VM start to stop, approximately $1.39 compute at the observed
Spot quote, excluding storage/preparation/network. The supervisor timed out
while a final GSQ checkpoint was still saving: the 180-second stop reserve and
90-second termination grace did not cover the block-boundary path. The prior
commit safely preserves two completed blocks and the start of block 2; one
observed subsequent update is not counted as durable. This was a supervised
timeout, not another numerical replay failure or an observed OOM.

Host-side continuation and asynchronous checkpoint publication are implemented
and tested locally, preserving the exact solver image and selected-stage
identity. The controller reserves at least 15 minutes for checkpointing; actual
overlapped throughput still needs measurement. CPU-side staging and full-run
disk-headroom admission remain operational gates. A local continuation plan reconciles the real terminal
evidence to the block-2 checkpoint and requires 104.39 GiB free by the current
conservative recovery rule. Available disk headroom has not yet been verified.
No fourth VM or longer paid run is authorized by the consumed third-run approval.
See [continuation limits](QUANTIZATION_CONTINUATION.md).
See the [checkpoint optimization and five-hour plan](QUANTIZATION_CHECKPOINT_OPTIMIZATION_2026-09-20.md)
for current implementation, tests, capacity limits and ETA uncertainty.

The steady observed linear block interval was about 3.8 minutes, dominated by
checkpoint handling. Applying that rate unchanged to 64 blocks gives a **4.1-hour
GSQ-only scenario**, not a complete-job ETA. Production full-attention/head/RCO,
packing and quality validation are still unmeasured. See the
[timing report](QUANTIZATION_TIMING_2026-09-19.md) and
[third-attempt evidence](benchmarks/quantization-2026-09-19/gpu-attempt-3.json).

## Results from the second attempt

Spot VM `zimfo-gpu-1aa7601d8015` used `g4-standard-48`, one 96 GB RTX PRO 6000,
and the immutable runtime `a1de1726…`. Guest execution lasted 1,318 seconds
(about 22 minutes); this is not the complete billable VM lifetime. Initialization
took about 139 seconds and the two GSQ smoke checks about 546 seconds including
loading and checkpoint work. GSQ peak allocation was 62.95 GiB and peak reserved
memory 65.86 GiB. The largest complete invocation was 4,673 tokens.

The RCO resumed-versus-uninterrupted comparison failed for 76 of 996 allocation
values, with maximum absolute difference 0.0011873245 (allowed absolute
1e-6 / relative 1e-5). The old report did not preserve isolated RCO update timings
or its peak memory before failure. Production embedding, all-block GSQ, head
and RCO stages did not begin. A defensible whole-run ETA remains unavailable;
see [measurements and ETA method](QUANTIZATION_TIMING_2026-09-19.md).

Evidence:

- `tools/calibration/runs/cloud-gpu-20260919-v2/{plan,result,terminal-diagnostics,timing-observations}.json`
- `tools/calibration/runs/cloud-gpu-20260919-v2/instances-after-cleanup.json`
- `docs/benchmarks/quantization-2026-09-19/gpu-attempt-2.json`

## Work completed before rental

- Original Qwen/Qwen3.8-27B BF16 revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: all 18 shard hashes verified,
  Mac inference and held-out comparison completed.
- Mac capture: 87 invocations, 104,557 tokens, 497 projection inputs per
  invocation, 173,130 tensors, 21.83 GiB; validation passed. Source-overlap review
  and CUDA/MLX parity remain separate quality checks.
- Private GCS staging completed: 114 data objects, 73.66 GiB, one manifest commit.
  Input commit SHA256 is
  `b15a9f032f9d0cdc4d1dcf5c601148f257724093e44508d7048a6ff6a672586a`,
  generation `1789857421832274`. Archives avoid per-tensor cloud writes.
- Prepared disks contain validated model/calibration inputs and a cached
  Linux amd64 CUDA 13 / PyTorch 2.11 runtime. Startup requires no package install
  or image pull. The tested second-attempt image passed 41 solver and 45
  transport/restore tests; it is **not recovery-validated on CUDA**.
- Packed 27B Q1 compatibility fixture loaded and prefills in the shipped runtime.
  It is an RTN layout fixture, not a learned quality result.

## Changes after the failure

The local harness now records synchronized optimizer timings separately from
model loading, checkpoint publication, full checkpoint cost, restore and cache
propagation. An enclosing stage timer captures unitemized preparation/export
cost. Timings stay local/in the existing solver log; there are no new per-update
GCS requests. Nested totals must not be added to their child measurements.

Recovery diagnostics now verify exact restored solver, optimizer, scheduler and
RNG state before replay, preserve both update reports and memory on failure,
and retain the existing post-update tolerance. Deterministic CUDA execution is
required before device initialization. Nondeterminism is a hypothesis, not a
proven cause or a verified fix. These changes passed 48 solver and 45 transport/restore tests in the local Linux
image, plus independent re-review. They must still pass actual GPU validation;
the new image is published and cached; the third GPU attempt is testing it.

## Paid-run and recovery controls

The approved second attempt had a one-hour absolute VM DELETE deadline, a
45-minute solver deadline, a 55-minute guest shutdown timer, no automatic retry,
and retained disks. Future attempts need a concrete bounded approval. At the
previously verified $1.7429/hour compute rate, one hour is approximately $1.74,
plus storage and network. Recheck the quote for a new rental.

The 100 GiB boot and 256 GiB data Hyperdisk Balanced volumes are retained, at
approximately $0.0724/hour combined (~$1.74/day). Bucket/registry storage is
additional. Preserve these prepared inputs until the next decision.

Worker identity is `zimfo-quant-worker@tiltastech-zimfo.iam.gserviceaccount.com`:
private bucket objectViewer/objectCreator and registry reader, with no keys,
overwrite or delete access. Bucket and registry are in `us-central1`; ingress
is denied. Checkpoints use bulk immutable state objects, generation/hash
verification and commit-last publication. A two-minute completed-step cadence
is not a strict two-minute loss bound when an update or transfer takes longer.

Launcher `tools/calibration/cloud_gpu/launch.py` verifies disk IDs and ownership,
consumes a detached-disk receipt, checks the open NVIDIA kernel module and cached
image, validates returned numerical/recovery reports, and deletes only the
stopped owned VM while keeping disks. The legacy synthetic launcher is disabled.
The existing recovery command restores a verified production **stage**; complete
remaining-stage orchestration is not yet automatic.

## First-attempt transport finding

VM `zimfo-gpu-12634bcb83a0` was stopped after roughly 12 minutes because remote
SHA verification used 1 MiB GCS ranges. The fix uses 64 MiB network ranges while
preserving small local buffers and generation/hash checks. CPU preparation
verified a 4,634,634,240-byte checkpoint in 47.427 seconds (93.194 MiB/s).
The refreshed receipt is `tools/calibration/runs/cloud-cpu-prep-20260919-v7/ready.json`.
Large duplicated auxiliary payloads still contributed substantial publication
and restore time in the second attempt; faster ranges do not eliminate that cost.
