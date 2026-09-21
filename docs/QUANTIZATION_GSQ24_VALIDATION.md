# GSQ on a 24 GB GPU

The user approved implementing block-at-a-time GSQ and verifying it on an actual
24 GB instance. This is a bounded validation run, not authorization for an
unbounded production run or proof that RCO fits in 24 GB.

## Implementation and preserved work

`solver/gsq_residency.py` keeps the original BF16 baseline in CPU memory. Only the
active transformer block and its rotary buffers move to CUDA. Training state is
released before changing blocks, including on errors and checkpoint stops.
`gsq_run` uses an explicit execution device; embeddings and inactive weights do
not accidentally follow the first model parameter onto CUDA. Original full-model
execution remains available. Memory mode/device are identity-bound.

The original checkpoint remains unchanged: 87 learned embedding updates, 174 GSQ
updates, completed blocks 0/1, block 2 input cache and starting optimizer state,
and the block 3 warmstart. CPU preparation verifies its exact GCS generation,
manifest, all payload hashes, and original identity. Ordinary resume continues
to reject a changed solver identity. Migration requires a separately bound
provenance record; diagnostic validation does not advance production progress.

## Actual-device test

`python -m solver.gsq24` starts six isolated workers: reference, streamed, and cold
checkpoint replay for linear block 2 and full-attention block 3. The unchanged
BlockTrainer source is pinned to SHA256
`e13654823c070e5e71984b1c0cd80a6060855166bc8aff2a999c549afb990e30`, read from the
previous immutable runtime. Reference workers explicitly place only the active
block on CUDA, independently of the new residency manager. This compares the
same mathematics on the same L4; it does not promise bitwise equivalence between
Blackwell and L4 hardware.

Both blocks use the full longest 4,673-token invocation. Block 2 consumes the
committed teacher/student cache. Block 3 uses a diagnostic propagation through
the committed block 2 state and the preserved block 3 warmstart; it is not the
future fully-trained block 2 production output. Each worker verifies losses,
all gradients, Adam/scheduler/RNG state, propagated teacher/student hidden states,
and hard candidate tensors. A separate cold process restores the saved state
and repeats the next update. Fresh CUDA peak counters cover model loading,
placement, updates, propagation, checkpointing and restore within each worker.

The parent rejects incomplete worker reports, wrong source/runtime/calibration,
truncated sequences, nonfinite results and non-24 GB hardware. Candidate equality
uses canonical tensor contents and metadata, not safetensors header byte order.
All six matching worker reports and the additional production-loop canary below
are required for success.

## Preparation before GPU rental

GCP `g2-standard-32` supplies one L4 with 24 GB VRAM and 128 GiB host RAM. The
smaller 32 GiB host shape cannot hold this full BF16 baseline. `g2-standard-24`
has two GPUs and is not the intended one-GPU test.

G2 does not support the Hyperdisk Balanced disks used for the previous G4 run.
A bounded CPU preparation host therefore builds a separate compatible boot disk
and restores inputs onto a new data disk; the old disks remain untouched. Driver,
Docker/toolkit, immutable runtime, original inputs and source checkpoint are
prepared before renting the GPU. The planned GPU attempt has a one-hour maximum,
provider deletion and guest deadlines, no automatic retries, and retained disks.
Reports are uploaded in bulk, not per tensor or optimizer step.

Published Spot compute for the 128 GiB single-L4 shape is approximately $1/hour;
storage and CPU preparation are additional. Sources:
[GCP GPU shapes/disks](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines),
[Spot pricing](https://cloud.google.com/spot-vms/pricing).

## Local evidence and remaining gates

The final Linux/amd64 build passed 156 solver tests and 45 transport/restore tests.
Adversarial harness coverage includes 77 cases. CPU residency parity and
interrupted/restored training tests passed; source migration helpers have 12 tests.
Reviews found and fixed missing calibration-source binding, weak worker report
admission, and a CPU preparation status mismatch.

Registry-verified runtime:
`us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/runtime@sha256:4b4c1c1750cf8ae8e0e92bfa9a77f60e65e7a05487360846d4824fb91482fb39`.

CPU preparation completed on September 20 UTC: original inputs and the exact
source checkpoint verified, final runtime cached, NVIDIA 580.178.04 prepared for
kernel 6.8.0-1067-gcp, and 161.39 GiB free. The CPU VM was removed and both PD disks
were retained and detached. The frozen single-L4 validation plan is in
`tools/calibration/runs/gsq24-l4-validation-20260920-v1/plan.json`.

Actual L4 execution, measured memory/throughput, and a production migration remain
unverified at this revision. A
successful test will still not establish head/RCO timing, complete-file packing,
phone quality, or a whole-job ETA.

The L4 create requests on September 20 at 05:43 and 05:48 UTC were both rejected
with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` in `us-central1-b`. Neither ran
the harness. These are capacity failures, not evidence that 24 GB is insufficient.
No further create attempts are scheduled. The original and prepared disks remain
available, and the original 174 GSQ updates remain unchanged.

After the first rejection, a read-only
[GCP Spot capacity advisor](https://docs.cloud.google.com/compute/docs/instances/view-vm-availability)
query rated both `us-central1-a` and the original zone at 0.9 obtainability with
600 seconds estimated uptime. The fresh second attempt reused the existing disks
because the alternate zone had no better advice. Actual allocation still failed:
capacity advice is probabilistic and does not reserve a GPU. No snapshots,
alternate-zone disks, or quota changes were made. A future alternate-zone attempt
needs documented snapshot lineage and enough regional SSD quota to preserve the
originals (500 GiB limit, 336 GiB used; duplicating both disks needs 672 GiB total).

Attempt evidence is retained under
`tools/calibration/runs/gsq24-l4-validation-20260920-v1/` and `...-v2/`.

## Adversarial review findings and disposition

- **Incomplete CUDA coverage:** the initial harness tested block mathematics,
  not the production run loop. A seventh, separate worker now runs the actual
  `gsq_run` loop for two updates from the verified block 2 source, using the next
  two complete corpus inputs. It exercises ordinary local checkpoints and verified
  restore, with strict cursor and performance admission. Its new identity binds
  diagnostic-only scope and the source commit: 174→176 here does not advance
  production progress. Full-corpus execution remains untested.
- **Weak independent state mapping:** two traversals of the current model did not
  independently ground positional state mapping. The old `qwen.py` adapter is now
  pinned to `aad58276128fcdd0a66d782c97dcae1748b57a9bbc0caf9738f01187988493b5`,
  alongside PyTorch 2.11.0+cu130, Transformers 5.7.0 and Accelerate 1.13.0. These
  values were read from the previous immutable container.
- **False success from partial reports:** tightened source/runtime/calibration,
  six-worker coverage, GPU memory, gradient/state comparisons, canary cursor,
  candidate fingerprints, and measured update-count validation.
- **Unsafe early scratch removal:** prevalidate both blocks' completed proofs and
  every deletion path before removing reproducible diagnostic fixtures. The
  original read-only source is never removed. Large comparison/checkpoint copies
  are discarded only after all six proofs pass, before the canary. Admission
  requires 110 GiB free, and the canary requires 100 GiB after that cleanup.
- **Misleading performance estimate:** mathematical comparison timings include
  gradient copying and comparison overhead. Actual run-loop timing groups measure
  the production path; two longest inputs still cannot establish a full-run ETA
  or the full-attention production timing distribution.
- **Later-stage handoff:** the generic continuation controller carried GSQ-only
  offload fields into head/RCO, which reject them. The external controller now
  supports `--stop-after-stage gsq`: it stops only after completed GSQ artifacts
  and its final cloud checkpoint are verified. New head/RCO configurations omit
  the GSQ-only settings; the selected GSQ checkpoint identity stays unchanged.
  An explicit later attempt restores terminal GSQ with zero new updates before
  proceeding. Independent review and 63 controller/pipeline/integration tests
  passed, including actual tiny-model recovery and failed-publication cases.
  This does not provision a larger GPU or apply a production identity migration.
- **CPU preparation failure:** the first CPU attempt stopped because its strict
  umask made a public package key unreadable by apt. Explicit permissions were
  added and preparation resumed on the same retained disks. No GPU ran.

The tested hardware requirement is one 24 GB GPU **plus 128 GiB host RAM**.
Ordinary checkpoint identity rejection remains in place; neither this test nor
the migration proposal helper applies a production migration.

Static original-checkpoint header inspection confirms block 2 has the same tensor
names, shapes and dtypes as all 48 linear-attention blocks, and block 3 matches
all 16 full-attention blocks. This supports representative block selection; it
does not replace actual memory or full-corpus timing evidence. See
[shape coverage](benchmarks/quantization-2026-09-19/gsq24-block-shape-coverage.json).
