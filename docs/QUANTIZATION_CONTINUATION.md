# Quantization continuation controller

`tools/calibration/continuation.py` prepares and can explicitly execute continuation on an **already provisioned host**, outside the frozen `solver/` package. It does not provision cloud resources, install dependencies, or alter the pinned runtime.

**Current limits:** the GCP launcher/bootstrap now has a tested continuation path, but actual CPU staging, disk-capacity admission and CUDA continuation through it have not been validated. CPU tests cover real tiny-model solver functions and identity binding, with simulated cloud transport; they do not establish full-model memory, timing, or quality. Independently review a prepared plan before using the execution path on a paid host.

## Prepare without executing

Wait for the retained job to stop. Preserve the disk, root status, stage reports, frozen configurations, and `latest-checkpoint.json` receipts. Then run:

```sh
python tools/calibration/continuation.py \
  --job-dir /retained/job \
  --output /new/continuation-plan \
  --runtime-image REGISTRY/runtime@sha256:EXACT_ORIGINAL_DIGEST \
  --inputs /future-host/prepared \
  --recovery-output /future-host/new-recovery
```

Preparation selects the deepest production stage with a valid retained receipt, instead of trusting potentially stale root status. It binds the receipt's canonical manifest, original frozen configuration, baseline, calibration and exact runtime identity. It preserves completed embedding/head reports against retained hashes. Missing or changed required reports fail preparation.

The new directory contains copied small evidence, `reconciled-status.json`, and `continuation-plan.json`. No tensors are downloaded; remote generations/payloads remain unverified until the original runtime's `prepare-resume` runs. Preparation never executes its suggested commands. Keep this directory unchanged and mount it at its recorded location for the standalone suggested command.

## Explicit execution on a prepared host

After review and authorization, the external controller can run the prepared plan:

```sh
python tools/calibration/continuation.py execute \
  --plan-dir /mnt/zimfo-inputs/continuation-plan \
  --output /mnt/zimfo-inputs/new-continuation-attempt \
  --workspace /mnt/zimfo-inputs \
  --inputs /mnt/zimfo-inputs/prepared \
  --deadline-seconds 2700
```

All paths must lie inside the explicit existing workspace; plans and outputs must be outside the read-only prepared inputs. The exact digest-pinned Docker image must already exist locally. The host needs Docker, NVIDIA runtime support, and its existing restricted GCS identity. This command does not enable APIs, create a VM, or set a VM lifetime. **A separately configured host/VM runtime limit remains necessary**, particularly if Docker cleanup itself fails.

The controller first runs the unchanged image's `solver.job prepare-resume`, verifies its returned receipt, then runs the selected stage with the original frozen identity. Subsequent completed stages hand candidate archives, propagated caches, and learned-boundary reports into GSQ, head and RCO as appropriate. Each attempt uses new output directories and checkpoint namespaces. All phases share one deadline. Execution supports an explicit duration up to 24 hours; the default checkpoint reserve is now 900 seconds, and no new stage starts after that work deadline. The pinned recovery helper remains bounded to 2,700 seconds. Container timeout leaves 120 seconds for owned-container cleanup inside the controller cap. A long optimizer update or cache propagation can still exceed the reserve. The cloud launcher derives its provider, guest and supervisor limits from one absolute deadline, so startup consumes the approved budget instead of resetting it.

The conservative recovery admission requirement is **six times the committed payload bytes plus 10 GiB free**. The plan reports this number, and execution checks available space before starting any container; the pinned recovery helper checks again. This covers overlapping verified downloads, expansion and serialization. It can block continuation even when the checkpoint itself would fit. It is not a complete disk forecast for all subsequent stages, especially the head's larger optimizer state. Preserve earlier evidence; do not delete it automatically to satisfy the check.

## Stops and remaining work

After a controlled stop, `status.json` records the current generation-pinned checkpoint and can be passed through preparation again. If the child fails after publishing, the controller still reconciles its latest valid receipt while retaining the failed outcome. If initial recovery fails before any new publication, the copied source checkpoint remains available for another preparation attempt. A terminal checkpoint can be resumed with zero additional optimizer updates to regenerate its candidate/cache/report outputs and advance to the next phase.

### Explicit GSQ handoff before head and RCO

On a host sized for block-at-a-time GSQ, request a stage boundary explicitly:

```sh
python tools/calibration/continuation.py execute \
  --plan-dir /mnt/zimfo-inputs/continuation-plan \
  --output /mnt/zimfo-inputs/new-gsq-attempt \
  --workspace /mnt/zimfo-inputs \
  --inputs /mnt/zimfo-inputs/prepared \
  --deadline-seconds 2700 \
  --stop-after-stage gsq
```

`--stop-after-stage gsq` prevents this attempt from launching head or RCO. It
returns `status: checkpointed` and `stop_reason: requested_stage_boundary` only
after GSQ reports completion, its final checkpoint is committed, and the
candidate archives and final propagated cache are available. In local-spool mode,
the final local checkpoint must first pass the existing GCS final-drain proof.
`stage_handoff` records the completed stage, report hash, committed snapshot,
candidate/cache locations, and remaining `head` and `rco` stages. A deadline stop
midway through GSQ remains a partial checkpoint; it does not receive a completed
stage handoff.

Retain that attempt's disk and evidence, then run preparation again using its
output directory as `--job-dir`. On a separately selected host with sufficient
memory for the remaining stages, execute the newly prepared plan **without**
`--stop-after-stage`. The controller first terminal-resumes the completed GSQ
checkpoint, potentially performing zero new optimizer updates, to reconstruct
portable candidates, caches and reports. It then starts head and RCO. The exact
pinned runtime and selected GSQ identity remain required; GSQ's frozen memory mode
and execution-device settings are preserved. Only newly constructed head/RCO
configurations omit `gsq_memory_mode` and `gsq_execution_device`, so those stages
can use their own full-model execution path.

The stop policy belongs to one controller attempt and is not inserted into solver
identity. A later attempt must explicitly request the flag again if it should
stop at GSQ. A plan already selecting head or RCO rejects the GSQ stop flag before
launching any container. This handoff does **not** provision a larger GPU, change
an instance, extend a runtime limit, or authorize further paid compute. Those
resource actions remain separate from preparing and executing the reviewed plan.

Completed RCO produces `quantization_stages_completed`, **not** a validated replacement model. Packing/export, runtime loading and held-out quality comparisons remain required. An RCO-only cold resume also needs candidate-inventory reconstruction from its restored artifacts before export. The controller does not select a replacement model, claim a complete quality recipe, or automatically return artifacts to the Mac.

## Local validation

Run the controller tests with the pinned upstream RCO `src` directory on `PYTHONPATH`:

```sh
PYTHONPATH=/path/to/pinned-rco/src:tools/calibration \
  tools/calibration/.venv/bin/python -m pytest \
  tools/calibration/tests/test_continuation.py \
  tools/calibration/tests/test_checkpoint_pipeline.py \
  tools/calibration/tests/test_checkpoint_pipeline_integration.py -q
```

The real CPU contract test uses the actual GSQ, head and RCO functions, actual candidate/cache artifacts, and `bind_identity`. It restores terminal GSQ with zero new updates and passes the real return schemas through the external controller to learned-head and full-vocabulary RCO completion. The spool integration also stops after completed GSQ, prepares the retained handoff, and terminal-resumes GSQ with zero updates before head/RCO. Only cloud I/O/process transport are simulated in that test; local and cloud checkpoint formats, hashes and restore operations are real. Separate tests cover stale root status, missing/changed boundary evidence, failed remote verification, checkpoint publication followed by failure, repeated preparation, disk rejection, timeout cleanup, and SIGTERM handler restoration.

## Cloud integration controls and remaining operational gates

The launcher now binds the controller and prepared-plan hashes,
exact source checkpoint generation/hash, runtime digest, and approved absolute
deadline into launch configuration. Host supervision logs stay outside the new
controller-owned output directory. An ownership ledger records producer and
publisher container names before launch. Ancestor reads are restricted to exact
reviewed source/smoke commits; new writes use the new run namespace. Outcome
validation preserves the distinction between completed stages and a validated
model. One bounded terminal bundle captures the selected cloud receipt,
configuration/provenance and supervisor/source evidence. These paths have local
fault tests; real staging and actual-host validation remain necessary.

A longer run must derive provider termination, guest shutdown, bootstrap timeout
and controller deadlines from the same approved absolute cap. The unchanged
runtime's `prepare-resume` helper can retain its own 2,700-second maximum;
subsequent stage deadlines can use the approved remaining duration because
`deadline_unix` is excluded from solver identity. Checkpoint reserve must reflect
measured publication latency, including a block-boundary save before a stop is
observed. Test directory ownership, ancestry rejection, zero-update completion,
all deadline layers and owned-container cleanup locally before renting.


## Overlapped checkpoint publication

Optional `--checkpoint-mode local-spool` uses the unchanged solver image with
local checkpoint storage on the retained data disk. The checkpoint field is
excluded from numerical identity. A separately supervised, digest-bound
`checkpoint_bridge.py` process uploads the newest complete local snapshot in
bulk every five minutes and drains the latest snapshot when the producer exits.
It never shares the training interpreter's RNG. Local cursors are recorded
separately from generation-pinned cloud receipts; only the latter count as GCS
progress. The original configuration and an audit of the transport-only rewrite
are published with every selected cloud snapshot. Final-drain validation binds
the local manifest digest, not just the snapshot name.

The cloud publisher uses one attempt namespace for all stages, so duplicate
content is not stored again under each stage prefix. Its verified-source cache
also reuses immutable artifacts within a stage. Snapshot selection coalesces
intermediate uploads; object manifests are still committed last, and restore
still verifies exact generations and hashes. It does not make per-tensor cloud
writes or periodically upload tiny progress files.

No concurrent local garbage collection is implemented: the producer can install
objects before exposing the corresponding commit. Local history therefore grows,
regardless of upload coalescing. `--spool-min-free-bytes` and a matching CPU-staged
capacity receipt are mandatory; the original 256 GiB disk is not assumed to fit.
`checkpoint_plan.py` exposes the retention and size assumptions and deliberately
does not certify whole-pipeline capacity from two sample blocks.

Reaching a five-hour target does not discard state. Stop new work before the
hard cap, save locally, drain publication, retain both disks, and prepare another
bounded continuation. If a hard stop interrupts publication, the older cloud
checkpoint remains valid and a newer local checkpoint may still be published
from the retained disk before resuming. This is resumability, not automatic
permission to extend a paid VM beyond its approved cap.
