# Native Mac GSQ/RCO handoff

Written September 20, 2026 UTC (September 19 evening Pacific). This is the
handoff for a separate session attempting native Mac optimization tonight.
It does not launch a run. Paths and small manifest hashes below were checked
locally while writing; the full tensor/weight validation reports are retained
evidence from collection, not a new complete rehash in this handoff.

## Start here

The full calibration set and original model are already on this Mac. Do not
recollect conversations, download another baseline, or train on Bonsai.

Repository: `/Users/jasontitus/experiments/mcpzim`.

The objective is original **Qwen/Qwen3.8-27B**, quantized for Zimfo's offline
Wikipedia/StreetZIM questions, tool use and conversational follow-ups. Compare
each candidate against both full-precision Qwen and shipping Bonsai, reporting
quality loss/gain, actual complete size, memory and speed. Bonsai's size is the
preferred anchor; a modest increase may be worthwhile for substantial quality
improvement. No arbitrary new hard size cutoff has been agreed.

The existing production solver is CUDA-specific. Successful Mac inference and
activation capture do **not** mean native GSQ/RCO training is ported or validated.
No native MPS/MLX training entry point was found in the inspected solver.
An implementation from another session must be inspected and qualified before use.

**September 20 update:** a new `df -h .` check reports about **985 GiB free**.
The earlier 28 GiB disk blocker below is historical and no longer applies at this
observation. Still budget checkpoint history and recheck before a long run.
See [answers to the port session's questions](QUANTIZATION_MAC_CAPTURE_ANSWERS.md):
the existing runner generates full teacher/student block caches sequentially;
an all-64-layer recapture is not required, and the original head is already local.

**Earlier observation:** `df -h .` reported only about **28 GiB free** on the
repository's APFS volume when this document was written. Hardware reports
`Mac13,2`, **128 GiB unified memory**. Recheck both before starting. There is
enough existing data to work without copies, but a full checkpointing run has
not been admitted against this remaining disk space. Do not consume the last
free space, silently disable checkpointing, or delete source artifacts to fit.
Choose adequate output storage or arrange space first. No cleanup was performed.

## Data map — exact local paths

All paths in this table are relative to the repository above. These artifacts
are ignored local files, not assets available from a clean git checkout.

| What | Path | Use |
|---|---|---|
| Validated activation/replay package | `tools/calibration/runs/mac-pipeline-v1-20260919/activations/` | Primary calibration source; **87 invocations, 104,557 tokens, max 4,673 tokens**, 497 projection inputs per invocation |
| Package manifest | `tools/calibration/runs/mac-pipeline-v1-20260919/activations/manifest.json` | Sequence order, per-file hashes, model/runtime provenance |
| Original exact app inputs | `tools/calibration/runs/qwen-calibration-v1-20260919/` | `invocation-*.json`: full rendered prompts, token IDs, conversation/turn IDs, sampler/tool/evidence context; the activation package also copies each input |
| Calibration answers | `tools/calibration/runs/qwen-calibration-v1-20260919-answers.json` | Generated trajectories for 25 conversations / 85 turns; observed answers are not automatically reviewed gold labels |
| Calibration suite | `eval/quantization_calibration_v1.json` | Scenario definitions; frozen copy is `activations/calibration-suite.json` |
| Original full-precision weights | `tools/calibration/runs/qwen3.8-27b-original/` | 18 safetensors shards plus config/tokenizer files, **55,586,035,724 bytes** total recorded model artifacts |
| Original weight verification | `docs/benchmarks/quantization-2026-09-19/baseline-weight-validation.json` | Original BF16 validation evidence; per-file hashes also in activation manifest |
| Activation integrity report | `tools/calibration/runs/mac-pipeline-v1-20260919/activation-validation.json` | Accepted 173,130 tensors / **23,436,670,016 tensor bytes = 21.83 GiB** |
| Bulk upload copies | `tools/calibration/runs/gcs-staging-v1-20260919/` | 87 invocation tar files, metadata tar, input manifest, generation-pinned upload receipt; not required to use native local inputs |
| Pinned upstream source | `tools/calibration/cuda_runtime/.context/sources/gsq/` and `.../rco/` | Immutable exported upstream source, with `.../sources/manifest.json`; preserve attribution |
| Held-out protocol and suite | `tools/calibration/runs/comparison-v1-20260919/protocol.json` and `suite.json` | Separate evaluation, not training/calibration input |
| Paired answers and corrected report | `tools/calibration/runs/mac-pipeline-v1-20260919/{reference-answers.json,bonsai-answers.json,comparison-retry.json}` | Reuse for comparison; `comparison.json` is the original failed report |

The name `tools/calibration/runs/prepared-20260919/` is misleading for this task:
it contains an old small launch package, **not** the complete calibration data.
The cloud path `/mnt/zimfo-inputs/prepared/` is likewise not a local Mac directory.

Baseline revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Activation manifest SHA-256:
`b4a3584782917bccec8bfbdba91bbc2249290860431cdae8233f624cfa337ead`.

### Reading an invocation correctly

Follow `manifest.json` → `sequences[*].directory` → `sequence.json`. Its `files`
map associates semantic layer names with hashed safetensors filenames. Do not
infer layer order from filenames or a directory listing. `input.json` in each
invocation directory contains exact `tokenIDs` and `promptSHA256`. See
`solver.run.corpus_inputs` for input integrity checks.

The export contains complete first-block inputs and final normalized hidden
states (`first_block_input` and `final_normalized_hidden` in the file map).
For every projection/call, it contains **32 stratified input rows** and
all-row `sum_x` / `sum_x_squared` statistics with row indices and total row counts.
It does **not** contain every raw projection activation or full Gram/Hessian
matrices. Diagonal moments cannot substitute for `XᵀX`, nonlinear GSQ block
reconstruction, or RCO's full-vocabulary objective.

All sequences were captured with fresh recurrent/attention state, no padding and
implicit positions 0..N-1. Preserve complete tokens and conversation trajectories.
The production recipe uses stable largest-first ordering. Do not truncate long
inputs, silently drop failures, or replace this app corpus with coding prompts.

## GCS copy and existing optimization progress

The committed cloud input package is:

```text
gs://tiltastech-zimfo-quantization-us-central1/inputs/b4a3584782917bccec8bfbdba91bbc2249290860431cdae8233f624cfa337ead/inputs-manifest.json
generation: 1789857421832274
SHA-256: b15a9f032f9d0cdc4d1dcf5c601148f257724093e44508d7048a6ff6a672586a
```

The manifest names every original-weight object and bulk calibration archive,
with exact generation, bytes and hashes. Local upload evidence is
`tools/calibration/runs/gcs-staging-v1-20260919/status.json`. Use existing
`restore_inputs.py` / `gcs_checkpoints.py` integrity semantics if recovery is
needed; do not download a second 79 GB input copy onto this nearly full Mac.

Prior actual CUDA work committed **87 embedding updates and 174 GSQ updates**:
blocks 0 and 1 complete, block 2 at its start, plus block 3 smoke warmstart.
This is not a completed GSQ model or completed production RCO search.

```text
gs://tiltastech-zimfo-quantization-us-central1/runs/zimfo-gpu-6583ec8c4659/gsq/commits/gsq-b002-s00000174-e000-q00000-p00004.json
generation: 1789869833752714
bytes: 3918
SHA-256: 75ae90952b053fa2df0d155256fed4cbe2babc00b87b1c835fbad7ead0c89180
```

Local receipt (metadata, **not** all checkpoint tensors):
`tools/calibration/runs/cloud-gpu-20260920-v3/continuation-plan/04-gsq/latest-checkpoint.json`.
Its file SHA-256 is
`8bebdd955cff647de882cc4ce03ab9f9179570a8de15a56d5ed13c76a7c44d19`.
The source payloads total about **16.9 GB**, including candidates, cache, solver,
optimizer, scheduler, RNG, progress, warmstarts, configuration and packing costs.
The verified expanded copy is on the retained GCP data disk, not established as
present on this Mac. `prepare_gsq24_source.py` can restore and verify that exact
source; check its temporary-space requirements before using it locally.

The old CUDA runtime identity is
`f13a4e99a85f4f650a8e3e130757fb4ddee891eeb8e34e6cea973e99dc56c3cd`.
The newer reviewed L4 diagnostic runtime is
`4b4c1c1750cf8ae8e0e92bfa9a77f60e65e7a05487360846d4824fb91482fb39`.
Two L4 requests failed for GCP capacity; neither ran or advanced progress.

A native Mac backend has a **new runtime/solver identity**. Do not change the old
receipt, pretend a CUDA RNG state is an MLX/MPS RNG state, or bypass normal resume
identity checks. Prefer preserving usable learned candidates with an explicit
conversion/provenance record. If exact optimizer/RNG migration cannot be proven,
label it a warm start or restart the affected stage under a new identity. The
current `gsq_migration.py` proposal helper is specific to the 24 GB CUDA test;
its success conditions are not a native Mac migration certificate.

## Code to port and contracts to retain

Read these before changing implementations:

| Code | Contract |
|---|---|
| `tools/calibration/solver/qwen.py` | Original HF BF16 text model, hybrid block adapter, exact projection mapping |
| `tools/calibration/solver/gsq.py` | GSQ quantizer, block loss, hard-forward propagation, candidate export |
| `tools/calibration/solver/run.py` | Stage order, data/identity binding, embedding/head training, checkpoints and cursor semantics |
| `tools/calibration/solver/rco.py` | Differentiable candidate interpolation, allocation logits, manifold/budget operations |
| `tools/calibration/solver/objective.py` | Chunked full-vocabulary loss and gradients |
| `tools/calibration/solver/candidates.py` | Packed Q1 candidates, streamed linear backward, including unselected-candidate gradients |
| `tools/calibration/solver/gsq_residency.py` | One-block CUDA residency and state release; CPU offload does not create extra physical memory on a unified-memory Mac |
| `tools/calibration/checkpoints.py`, `checkpoint_bridge.py` | Durable manifest-last local saves and optional coalesced bulk cloud publication |
| `tools/calibration/mac_collect.py` | Known working MLX original-weight import and reference-capture conventions |
| `tools/calibration/packing/` | Actual encoding/cost manifests and round-trip tests; required for a deployable result |

Pinned GSQ upstream: `03fc16484c369e3127225615d5e03e8d3a6043e3`.
Pinned RCO upstream: `9a1e09c07d468109cbe60a1b87d5036034a79d10`.

Concrete port gaps identified in source:

- `run.main` requires CUDA BF16; `gsq_residency.execution_device` rejects MPS.
  Changing a device string does not produce a working Mac runner.
- The non-CPU GSQ path imports CUDA-oriented upstream code. Its Gumbel autograd
  uses CUDA RNG APIs; its `idx` tensor/device attribute need explicit handling.
  `CPUOneBit` exists for tiny mathematical tests and is **not** a drop-in
  numerically equivalent BF16 production quantizer.
- Preserve BF16 rounding boundaries, FP32 logits/scale reductions and the
  post-Adam scale grid (`scales.half().bfloat16().float()` in PyTorch).
  Test soft and hard forward, gradients, Adam and export,
  not just a low loss value. Do not quietly switch the baseline to FP32/FP16 or
  use a quantized teacher to make the implementation run.
- RCO constructs on-device float64 counts; boundary training uses per-device
  generators. Audit these and operation/backward support on the exact native
  backend before allocating the full model. Do not silently rely on CPU fallback.
- Existing RNG capture covers Python, NumPy, torch CPU and CUDA, not MLX/MPS.
  Add backend-specific state and cold next-update replay tests.
- Existing timing synchronization and peak-memory accounting are CUDA-specific.
  MLX needs explicit evaluation/synchronization so lazy work is included; MPS
  needs its corresponding synchronization. Record total process/system memory,
  allocator use and swap pressure, not a misleading CUDA-zero statistic.

MLX's original-HF import maps names, discards vision/MTP, transposes convolution
axis 2→1, and adds 1 to selected RMSNorm weights. The collector records these
conventions. Apply each transformation exactly once and reconcile export back
to the packer's original tensor layout. Do not feed sanitized MLX weights into
an HF adapter expecting unsanitized originals.

GSQ must compare each student's block output, with its already-quantized prefix,
against the original teacher block output. Preserve distinct teacher/student
caches. RCO must preserve the full-vocabulary normalization, embedding and head
contributions, exact serialized byte costs, allocation feasibility, and gradients
for candidates not chosen in the forward pass. Sampled vocabulary or local
projection MSE is not an equivalent RCO objective.

More specifically, the objective is `KL(teacher || student)` over the shifted
hidden sequence `hidden[:-1]`, with FP32 logits and global log-sum-exp. Token and
vocabulary chunks partition memory, not the probability normalization. Preserve
the allocation update order (project gradient → Adam → retraction → vector
transport) and keep checkpoint recomputation inside the candidate-interpolation
context. The streamed-linear and head backward paths contain deliberate BF16
rounding; generic autodiff through casts may not reproduce their gradients.
Use fixed-noise dense-oracle comparisons to detect this rather than accepting
merely finite losses.

This implementation's candidate pair is learned group-128 **one-bit Q1 plus
original BF16**, not an already implemented ternary recipe. Preserve code/sign
conventions and actual group/scale packing. Size must include protected tensors,
embeddings, head, metadata and padding. A 4,000,000,000-byte target appears in the
old run configuration; it is one experimental point, not the user's acceptance rule.

## Native environment and checks before an overnight run

The existing `tools/calibration/.venv/bin/python` reports Python 3.12 environment
packages: torch **2.8.0**, MLX **0.32.2**, mlx-lm **0.31.3**, transformers **5.17.0**,
safetensors **0.8.0**, google-cloud-storage **3.4.1**; **accelerate is absent**.
The CUDA runtime uses different pinned versions (torch 2.11.0+cu130,
transformers 5.7.0, accelerate 1.13.0). Do not assume interchangeable behavior or
install the Linux CUDA lockfile into the native Mac environment. Preserve the
working collector environment; use a separate pinned native environment for
port development and record its versions/source hashes.

The current cloud runner expects `inputs/{model,calibration,restore-validation.json}`.
The native local files are in the separate paths listed above. Add an explicit
validated local-input adapter or use a tested adapter supplied by the port.
Do not fabricate a cloud restore receipt, weaken hashes, or copy all data merely
to satisfy that directory layout.

Safe inspection/validation commands from the repository (no model optimization):

```sh
df -h .
sysctl hw.memsize hw.model
shasum -a 256 tools/calibration/runs/mac-pipeline-v1-20260919/activations/manifest.json
shasum -a 256 tools/calibration/runs/gcs-staging-v1-20260919/inputs-manifest.json
# Reads the existing package; allow time for a full tensor integrity pass.
tools/calibration/.venv/bin/python tools/calibration/validate_capture.py \
  tools/calibration/runs/mac-pipeline-v1-20260919/activations
```

There is intentionally no invented `--device mlx` launch command here: the
current CLI does not support it. The receiving session should document the real
native entry point and complete frozen configuration after inspecting the port.

Suggested execution sequence:

1. Inventory any existing port, live Metal processes, memory pressure and output
   disk headroom. Keep model-heavy workloads sequential. Retain existing data.
2. Run small **implementation** parity tests for GSQ, streamed backward, RCO
   budget/full-vocabulary loss, candidate packing and cold checkpoint recovery.
   This is not a request to collect or optimize a preliminary dataset.
3. Exercise actual Qwen linear-attention and full-attention blocks with the
   complete 4,673-token invocation. Verify loss, every required gradient, optimizer
   state and candidate export; record tolerances and actual peak memory. Test a
   complete native RCO forward/backward/update too, not only projection kernels.
4. Start the full existing 87-invocation corpus once backend and resource checks
   pass. Stage order is initialization → embedding → GSQ blocks → head → RCO.
   Reuse prior learned work only through validated, explicitly labeled migration.
5. Save at completed update boundaries with all optimizer/scheduler/RNG/cursor
   state. Keep final commits atomic. Test interruption and a cold restore before
   leaving a long run unattended. Plan local retention: existing immutable stores
   accumulate history and have no concurrent garbage collector. Prefer a bounded
   retained output volume and bulk/coalesced GCS backups if enabled; no thousands
   of small cloud writes. Do not delete objects while a writer can reference them.
6. Measure load, updates by block type/sequence length, cache propagation, local
   serialization, upload and restore separately. Produce a measured GSQ estimate
   first; full-job ETA also needs head and RCO measurements. Old G4 timings and
   14.6-minute Mac activation export do not predict native optimization time.
7. Write a human-readable status with a committed checkpoint and exact restart
   command, even if the whole job does not finish tonight. Packing and held-out
   evaluation follow; completing optimization alone is not a replacement model.

## Evaluation and handoff expectations

Keep the held-out comparison out of training and allocation tuning. Existing
automated app checks passed 51/66 Qwen turns and 48/66 Bonsai turns; among the
paired union of 17 model-involved turns, 13 versus 11. This small suite includes
deterministic routing and different closed-loop trajectories, so it is not a
general quality score. Human critical-answer review remains necessary.

The corrected report is `comparison-retry.json`. The pipeline's historical failed
stage is retained under `status.json` with `completed_with_reporting_recovery`;
the correction handled inactive llama.cpp top-k in greedy mode. Do not rerun or
discard useful outputs solely because the old `comparison.json` failed.

At the end of the native session, report: backend/source/environment identity,
data manifest hash, full corpus coverage, tests/parity limitations, migrated vs
new progress, latest durable checkpoint and restart command, disk/memory usage,
measured timings and ETA assumptions, candidate sizes and held-out results if
available. Preserve originals and unrelated app changes. Do not launch another
paid GPU merely because this native attempt encounters a porting obstacle.

Handoff review: independently checked local source/receipt paths, validation CLI,
dependency pins, numerical contracts and evaluation caveats. This document's
review is not a native-backend training validation.

Further context: [24 GB design/reviews](QUANTIZATION_GSQ24_VALIDATION.md),
[checkpoint design](QUANTIZATION_CHECKPOINTS.md),
[checkpoint optimization](QUANTIZATION_CHECKPOINT_OPTIMIZATION_2026-09-20.md),
[continuation controller](QUANTIZATION_CONTINUATION.md),
[corpus](QUANTIZATION_CORPUS.md), and [overall plan](QUANTIZATION_PLAN_2026-09-19.md).
The overall plan contains historical progress sections; prefer current receipts
and this handoff for what has actually completed.
