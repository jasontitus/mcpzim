# Quantization checkpoint and resume contract

Current evidence update (September 20 UTC): the third GPU attempt passed
actual-Qwen full-model RCO GCS checkpoint restoration, exact pre-update state
checks and the unchanged next-update tolerances. Production checkpoints preserve
87 embedding and 174 GSQ updates. This does not yet validate the external
continuation controller on CUDA or an actual Spot-kill/cold-host continuation.
See [third-attempt results](QUANTIZATION_TIMING_2026-09-19.md) and
[continuation limits](QUANTIZATION_CONTINUATION.md). The earlier protocol research
and historical gates below retain their original scope.

Updated 2026-09-19. `tools/calibration/checkpoints.py` supplies the tested local
store, and `tools/calibration/gcs_checkpoints.py` supplies the production GCS
transport. The latter passed a **live bucket roundtrip with real CPU optimizer
resume and injected interruption/conflict checks**. This does not establish full
GSQ/RCO CUDA resume, whole-Qwen feasibility, or actual Spot-preemption recovery.

## Implemented publication protocol

`LocalCheckpointStore.publish(snapshot, identity, payloads)` copies frozen files
with a 1 MiB buffer, hashes them with SHA-256, flushes and fsyncs each file, and
installs each object under its digest with atomic create-if-absent hard links.
Existing objects must have the same bytes. The commit manifest is installed
**after** all payloads have been verified; no partially uploaded snapshot appears
in the committed snapshot list. Interrupted work can leave unreferenced objects.
The previous commit remains available. A retry with the same snapshot and bytes
is idempotent; a conflicting rewrite is rejected.

`restore` checks the expected identity, manifest schema, every payload size and
hash before copying. It checks copies again, then exposes a complete destination
directory with an atomic rename. It never deserializes tensors or executes pickle
objects. The caller chooses an explicit snapshot; lexicographic ordering of names
is not an automatic "latest valid" policy. Missing or corrupt commits fail closed.

Each snapshot has five mandatory payload roles: `solver`, `optimizer`, `scheduler`,
`rng`, and `progress`. Additional named shards may be attached. The store validates
presence and bytes, not solver-specific contents. A future adapter must validate
all state fields below; empty placeholder files must never satisfy a launch gate.

Identity binds:

- Original baseline repository and immutable 40-character revision.
- Exact calibration manifest SHA-256, including token IDs, masks, splits, source
  evidence, and reference-cache manifests as applicable.
- Pinned solver revision and the SHA-256 of effective solver configuration,
  including local patch/source digests, precision, parameter ordering and budgets.
- Candidate database manifest SHA-256, including tensor names, representation,
  shapes, bit costs and artifact hashes. During GSQ candidate creation, use a
  manifest of that job's immutable input dependencies; publish generated candidates
  as outputs. Do not change the identity under a running job or silently replace
  missing candidates with another database.
- Runtime manifest SHA-256: container/environment digest, package versions,
  architecture, GPU topology, deterministic settings, and custom kernel versions.

## Safe state boundary and actual upstream requirements

Take a checkpoint after a complete optimizer update and all associated schedule
and constraint operations, with no in-flight gradient accumulation or live
autograd graph. Synchronize the device before making CPU snapshots. Freeze the
source files while publishing. Avoid duplicating the entire model into a single
in-memory serialization buffer: stage state by block/tensor shard, releasing each
temporary device/host buffer. The store itself is bounded, but it cannot enforce
how much memory a caller uses before supplying files.

The following contract comes from the pinned source, not an assumption that a
generic `model.state_dict()` contains everything.

| State | GSQ candidate optimization | RCO allocation optimization |
|---|---|---|
| Solver | All active quantizer parameters. One-bit GSQ specifically has `sign_logits` and FP32 `scales`; other bit widths have their own parameters. Preserve exact tensor and optimizer-group ordering. | Allocation `interp.alpha`; group ordering, candidate ordering, `actual_bits`, `group_param_fracs`, target budget and model/candidate reconstruction dependencies. |
| Optimizer | Lion `state_dict()`, including moments and parameter-group fields. | Adam `state_dict()`, including step, moments and group fields **after** vector transport. |
| Scheduler | Custom scheduler has no `state_dict()` API. Explicitly serialize `current_step`, `initial_lrs`, `total_steps`, `warmup_steps`, `lr_decay_type`, `min_lr`; preserve group `lr_decay_tag`. | No independent LR scheduler in the inspected search loop. Serialize an explicit `kind: none` schedule descriptor plus total steps, next step, `tau_init`, `tau_min`, and the annealing formula/version. |
| Progress | Active layer/block, candidate setting, epoch, next optimizer update, number of total updates, temperature/scale schedule parameters, completed output manifest, and partial epoch metrics if required for reports. | Next optimizer update and completed history; snapshot after gradient projection, Adam step, retraction, and vector transport. Debug `alpha_snapshot` affects diagnostics and should be retained if reproducing logs. |
| Data traversal | Current epoch batch permutation **and cursor**. RNG alone cannot reconstruct an already drawn permutation. Include masks, padding behavior, microbatch size and quantized-prefix/cache dependencies. | Calibration order/masks and batch configuration. At an update boundary, next-step minibatch selection and Gumbel noise come from restored RNG. Mid-step resume is unsupported. |
| RNG | Python, NumPy, torch CPU and every CUDA device, plus any explicit generator states. GSQ's custom backward replays forward CUDA RNG; finish backward before checkpointing. | Same global/explicit generators: the loop uses CPU `randperm` for minibatches and device `rand_like` for Gumbel sampling. Restore RNG **after** constructors, state loads, and any diagnostic initialization. |

GSQ references: [trainer and custom scheduler](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/trainer.py),
[one-bit quantizer](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/quantization/gumbel_quantizer_1bit.py).
The trainer currently writes final hard quantized outputs after finishing a layer;
those files alone are not an interrupted-training resume checkpoint. Its batch
iterator draws a permutation inside a generator and drops a remainder; the adapter
must make traversal explicit and record the chosen remainder policy.

RCO reference: [allocation search](https://github.com/IST-DASLab/RCO/blob/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/search/quant.py).
The inspected routine constructs Adam inside the search function and starts its
loop from zero. It needs an adapter/refactor to accept restored state and a next
step; storing final `alpha` alone is insufficient. Do not repeat initialization,
sensitivity probes, or retraction after restore unless that is explicitly part of
the next operation.

## Tests and what they establish

Run from the repository root:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python -m pytest tools/calibration/tests/test_checkpoints.py -q
```

Measured locally: **18 passed**. Tests cover interrupted payload upload, failure
before and after manifest commit, retries, immutable conflicts, corrupt/missing
payloads, every run-identity dimension, unsafe names, mutation during copying,
mutation between verification and restore, and changed CUDA RNG device count.
A tiny real CPU Adam/StepLR model checkpoints after three updates, restores into
fresh objects, and reproduces the next random target, parameter update, Adam
moment, and scheduler state exactly. This proves the local protocol and generic
state handling, not numerical equivalence of the full solver on CUDA.

## Manual adversarial review and remaining launch blockers

- **Do the five roles prove completeness?** No. They only prevent omission of a
  category. GSQ/RCO adapters need schema validation and interrupted/resumed versus
  uninterrupted tests using their actual step functions and loss/gradient outputs.
- **Is local disk safe after Spot deletion?** Not by itself. The GCS store below
  has now passed transport and tiny optimizer recovery tests. The actual solver
  must call it at safe boundaries and include all referenced cache/candidate state.
  A full CUDA job killed and resumed from durable state is still a separate gate.
- **What about a kill during writing?** An incomplete checkpoint is ignored;
  resume the prior committed one. Abrupt process/power and actual Spot kill tests
  remain necessary in the target environment. Local tests inject failures around
  the same publication boundaries; they do not emulate storage hardware failures.
- **Can the restore silently use another model or candidate set?** Expected
  identity must be independently constructed from pinned input manifests. Never
  trust the checkpoint to tell the caller which identity to expect.
- **Will it be bitwise identical on another GPU?** Not promised. RNG topology is
  checked, but kernels and floating-point execution can vary. Same-environment
  CUDA equivalence and an explicitly bounded tolerance are separate checks.
- **Can publication race the optimizer?** Frozen source files and a completed
  update boundary are mandatory. Source rehashing catches ordinary file changes,
  not arbitrary concurrent mutation of the underlying model during serialization.
- **Is this an untrusted multi-tenant store?** No. The local directory must be
  trusted; symlink attacks and malicious manifests signed by another party are not
  addressed. SHA-256 checks integrity, not authorship. Restored state should use
  restricted tensor loading (`weights_only=True` where supported).
- **Are uploads free of storage leaks?** No. Crashes may leave temporary files or
  orphan objects. Retention/garbage collection must inspect all committed manifests
  and live publishers before deleting anything; no automatic destructive cleanup
  exists yet. Restore requires one owner for the destination directory.

Before renting, complete solver state adapters, actual step/resume equivalence,
artifact staging and restore disk-space checks. The
GPU smoke must then kill/restart a short actual-model job and compare its next
update, confirming progress survives removal of the compute VM and boot disk.

## GCS transport: implemented and live tested

Install `google-cloud-storage==3.4.1` in the prepared runtime. It was installed in
the repository calibration virtual environment for these tests. Use ADC for the
restricted worker service account; the preparation Mac can opt into its existing
gcloud login with `make_client(project, use_gcloud=True)`. Tokens pass through a
captured pipe into memory, refresh as needed, and are never serialized or printed.

```python
from gcs_checkpoints import GCSCheckpointStore

store = GCSCheckpointStore(
    "tiltastech-zimfo-quantization-us-central1", "runs/your-unique-run",
    project="tiltastech-zimfo", staging_dir="/scratch/checkpoint-staging",
)
receipt = store.publish("step-000100", identity, payload_files)
# Keep this small receipt in the job's durable progress record.
store.restore("step-000100", identity, "/scratch/resumed-state",
              commit_generation=receipt["commit"]["generation"])
```

`payload_files` maps role names to frozen local files. The five mandatory roles
are bundled into **one deterministic tar object**, with one small commit manifest
written last. Extra roles should already be bulk archives, such as one completed
candidate block or one rolling cache checkpoint. Do not upload individual tensors
or logs per step. The solver controls cadence: phase boundaries and approximately
two minutes of useful optimization work, rather than every optimizer step.

Schema 3 records each core archive member's name, size and SHA-256, and the
archive object's name, generation, size and SHA-256. Extra archives get the same
generation-bound object descriptors. Every newly uploaded remote object is read
back in 1 MiB chunks and SHA-256 checked before the commit becomes visible. SDK
uploads use resumable 8 MiB chunks and CRC32C. Create-only generation preconditions
make uncertain-request retries safe; an existing object is accepted only after
verification. Restore pins the commit generation, validates independently supplied
job identity, then pins all payload generations. It stages downloaded bytes and
extracts only the five expected regular members, rejecting links, traversal,
duplicates and hash/size mismatches before publishing the restored directory.

Unchanged caller-owned extra archives are reused without repeat upload or full
rehashing: the store caches a previously SHA-verified generation against local
path/inode/size/mtime/ctime and confirms that generation still exists. This cache
depends on frozen files and a trusted local directory; it does not defend against
a hostile process forging filesystem metadata. A new process re-verifies its
inputs. Core state is repacked, while completed historical blocks stay immutable.

The source of these API semantics is Google's [generation preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions)
and [Python Blob API](https://docs.cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob).
The transport does not create buckets, launch VMs, change IAM, delete objects, or
set lifecycle policies. Retention/garbage collection remains an explicit task;
incomplete publications can leave safe but unreferenced payload objects.

### Reproducibility finding from the live test

The first live optimizer continuation test failed even though bytes restored
correctly: Google client retry jitter consumed Python's process-global RNG.
Synchronous transport operations now preserve that RNG, and a unit test deliberately
consumes random values inside its fake network calls to guard the fix. **Background
checkpoint calls are rejected before touching RNG state.** The solver must pause
all training/data-loader RNG users, finish backward/constraint operations, and
synchronize the device before a main-thread checkpoint call. A main-thread check
cannot detect another independently running training thread; such concurrency is
unsupported. Future asynchronous publication needs isolated explicit generators,
not a global RNG reset. Restore saved solver RNG only after reconstructing objects.

### Recorded evidence and remaining limits

`tests/test_gcs_checkpoints.py`: **30 passed** on the Mac. Coverage includes
generation pinning, interrupted uploads and commit responses, idempotent retries,
conflicts, unsafe tar members, corrupted/missing data, wrong identities/generations,
bounded reads, two-object publication for the five roles, historical archive reuse,
SDK preconditions, RNG isolation and rejection of background writes. A separate
agent's adversarial review found two hardening opportunities, both fixed: enforce
a JSON object for the payload map, and verify the completed tar members before
upload to catch source bytes changing and then reverting during archive creation.

Live proof command:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  tools/calibration/checkpoint_cloud_smoke.py \
  --bucket tiltastech-zimfo-quantization-us-central1 --gcloud \
  --report tools/calibration/runs/gcs-checkpoint-live-20260919.json
```

The successful test took **8.69 seconds** and retained **4 objects / 64,575 bytes**
under `gs://tiltastech-zimfo-quantization-us-central1/tests/checkpoints-32e53aa7a6614157b03a1dc5c886bef2/`.
It passed nine checks: publish/read verification, idempotent repeat, exact next
Adam/RNG/scheduler update after restore, wrong generation, wrong identity,
interrupted publication without a commit, successful retry, conflicting commit
rejection, and preservation of the original committed checkpoint. The report is
local ignored evidence; no model weights or private captures were committed.

This is real durable transport with a tiny CPU training state. It is not yet
evidence that a full GSQ/RCO model resumes identically on CUDA or survives a real
Spot interruption. Peak local scratch must accommodate the core archive plus
extracted state; the solver must size that space before running. SDK authentication
and downloads also need validation under the actual restricted VM service account.

## Production stage orchestration and cold recovery

`python -m solver.job` launches isolated processes for initialization, both
actual-model smoke components, embedding training, all GSQ blocks, head training,
and RCO. Isolation also releases CUDA allocations and prevents the upstream
GSQ/RCO packages from sharing conflicting module names. The full captured corpus
is retained. A smoke-only successful run reports zero
`production_optimizer_updates`; its diagnostic checkpoints are explicitly not
advertised as resumable production stages.

A production checkpoint includes the frozen configuration, packing-cost
manifest, learned boundary reports, and hash-bound GSQ warm starts in addition
to optimizer/scheduler/RNG/progress state and candidate/cache archives. The job
status records the entire immutable commit receipt. `status.json` lives on the
retained data disk; the terminal VM report also embeds it in GCS. On an abrupt
preemption the newest stage `latest-checkpoint.json` on that retained disk can
be newer than the controller's status. Preserve that newer receipt when choosing
which checkpoint to recover. The immutable GCS commit and configuration are the
ultimate state, not the mutable local status file.

To resume a production stage on a clean host, first restore the same pinned
input package and use the **identical production image digest**. Copy the job
status (or terminal VM report containing `solver_status`) to the new host and
run inside that image:

```sh
python -m solver.job prepare-resume \
  --status /mnt/zimfo-inputs/recovery/source-status.json \
  --inputs /mnt/zimfo-inputs/prepared \
  --output /mnt/zimfo-inputs/recovery/new-attempt \
  --runtime-sha256 ORIGINAL_IMAGE_DIGEST_HEX \
  --deadline-seconds 2700 \
  --execute
```

Omit `--execute` to prepare and inspect the exact command in
`recovery-receipt.json`. `--use-gcloud` is available for a local Mac preparation
using its existing login; cloud workers use their attached service account.
The destination must not exist. Recovery validates generation, checksum,
baseline, image, calibration and frozen algorithm identity; remaps auxiliary
file paths; then resumes the interrupted stage in a new checkpoint prefix.
The separate read-source/write-destination prefixes prevent an unchanged
optimizer cursor with relocated configuration from conflicting with the old
immutable commit. Old commits remain usable.

This initial recovery path deliberately verifies all remote payloads once while
preparing and again in the stage CLI. It therefore downloads twice and needs
space for both restored copies. It resumes that stage only; subsequent phases
are launched separately from its completed report. It does not silently restart
an entire recipe or treat a diagnostic smoke checkpoint as learned production
state. Checkpoint publication stays synchronous at an optimizer boundary.

Verification includes a real tiny PyTorch Adam update with scheduler and RNG:
create a production checkpoint, delete the entire original worker directory,
restore onto an empty root, rebind the exact identity, and reproduce the next
update bit-for-bit. The test also restores from the old prefix and republishes
the same cursor in a new prefix without changing the original commit. This is
CPU recovery proof; the separate actual-model CUDA smoke establishes the real
GPU next-update resume check before long optimization.

Cold recovery checks available disk space before downloading: six times the
committed payload bytes plus a 10 GiB reserve must be free, conservatively
covering two restores, temporary bundles, expanded archives and another state
serialization. Existing retained jobs are never deleted to make room. Repeated
recovery can require a larger data disk or explicit operator-managed archival.
