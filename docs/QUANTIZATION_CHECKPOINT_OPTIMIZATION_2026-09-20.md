# Checkpoint optimization and five-hour continuation

The third actual-Qwen run proved the strict RCO recovery checks and retained 87
embedding plus 174 GSQ updates. It also showed the bottleneck: 36.16 seconds of
optimizer work for two complete blocks versus 869.37 seconds in five completed
GSQ checkpoint calls. Those calls include the initial dependency upload and
nested publication costs; the numbers must not be added twice.

The new optional path keeps synchronous complete local checkpoints on retained
storage and publishes them using a separate process. Training can proceed while
GCS uploads and remote verification run. The publisher selects the newest
complete pending snapshot every 300 seconds and performs a final drain. It uses
bulk objects, immutable generations and commit-last manifests. Shared attempt
namespaces avoid storing identical artifacts again under each stage. Original
solver code, runtime image, quantization recipe and numerical identity remain
unchanged; the checkpoint location is an explicitly excluded operational field.
A source-configuration payload and audit bind that transport rewrite.

The controller treats local and GCS cursors separately. It requires a matching
final source-manifest digest before accepting a completed publication. Tests
exercise a real tiny hybrid Qwen model, actual optimizer/RNG state, GSQ, learned
embedding/head, full-vocabulary RCO, local checkpoints, cloud-format restoration
and a second cold continuation. Only process execution and network transport are
substituted in that integration test. The host receipt reader also runs under
Python isolated/no-site mode with no model or cloud SDK imports.

Adversarial review caught two cleanup bugs: a Docker CLI could exit while its
container remained alive, and an expired deadline could still launch processes.
Both are fixed and tested. Abnormal cleanup targets both recorded container
names and retains the ownership ledger if cleanup is uncertain. Another review
strengthened the final local-to-cloud manifest binding.

The targeted checkpoint/controller/cloud/staging regression suite passed 198
tests. Two subsequent host regression tests also cover an unrelated publisher
name and recovery receipts taking priority over bulky performance reports.
Independent staging review passed all 11 staging tests with no blocking finding.
Shell syntax and Python compilation checks passed. These are local validations;
they do not establish production upload throughput or full-run disk capacity.

## Timing target and estimate

Five hours is a completion target and can also be configured as a bounded VM
budget. It is not a measured whole-job ETA. The current baseline gives 229 seconds
per observed steady linear block, or 3.94 hours for the 62 remaining GSQ blocks
before head/RCO/packing/evaluation. The new overlap path has not been timed on
the actual retained cloud disk. It should remove serialized network waiting,
but local copying/hashing, uploader contention, startup, final draining and
head/RCO remain material unknowns.

The offline planner reports the conditional overlap model:

`max(producer time, background upload time) + final drain + restore`

Producer time includes synchronous local checkpoint work. This model requires
real release-rate/throughput assumptions; it is not an achieved speed or a bound.
We have not filled missing measurements with zero or advertised a speculative
3–5 hour completion estimate. Use the five-hour budget as a resumable target;
measure the first actual overlapped blocks before narrowing the prediction.

The controller now reserves at least 15 minutes for checkpointing before its
absolute deadline and 120 seconds for container cleanup. The provider, guest and
host supervisor also derive their limits from the same approved absolute cap.
There is no automatic paid extension. If unfinished, retain disks and verified
GCS commits and continue in another bounded attempt. A partially uploaded latest
snapshot never replaces the previous valid commit.

## Disk capacity and remaining launch gates

The unchanged local store still synchronously copies/hashes payloads and retains
all distinct checkpoint objects. Coalescing cloud uploads does not reclaim local
history. Concurrent garbage collection would race unpublished producer commits,
so it is deliberately absent. The existing 256 GiB disk is not admitted for a full
spooled run. `checkpoint_plan.py` requires explicit bounds and includes historical
state/cache objects, recovery, temporary copies and metadata; head/RCO/packing
need separate space allowances. Adequate capacity or safely quiescent retention
must be established before a GPU launch.

Remaining operational work: stage the reviewed plan/helpers on the retained
disk using a CPU host, verify actual identity/free space, measure local-copy and
publication throughput there, then run the actual CUDA continuation with the
approved cap. No GPU has been launched for these code changes.

`cloud_prep/stage_continuation.py` now builds a deterministic hash-bound bundle,
stages it on the CPU host, and emits a provisional receipt. Operator-side
read-only Compute queries then bind that same CPU instance to the retained disk's
numeric ID. The worker needs no additional IAM privileges. Bundle transfer and
actual CPU staging have not run. Startup rechecks staged hashes and free space.
Minor review hardening remains around concurrent empty-directory replacement
and syncing nested directories during CPU staging; neither bypasses the startup
hash checks.

## Proposed 24 GB GSQ / larger-GPU RCO split

The stage boundary allows this in principle, but the current pinned executable
cannot run GSQ on 24 GB. `solver/run.py` loads the entire original text model on
CUDA before selecting a stage; `solver/qwen.py` installs every BF16 tensor on
that device. The measured GSQ smoke reached 63.00 GiB allocated and 66.25 GiB
reserved. This is an implementation footprint, not proof that GSQ inherently
requires that much memory.

`gsq_run` already trains one block at a time and carries teacher/student hidden
states in disk caches. A smaller-memory implementation can keep inactive weights
in host RAM or stream them from original shards, place only the active block and
its quantizers/optimizer on CUDA, and move to the next block after propagation.
The large embedding and head training stages need separate admission; initially
keep those on the larger GPU alongside RCO. Sufficient host RAM or shard streaming
is still required even with only 24 GB VRAM.

Before claiming a 24 GB fit, test both hybrid block types using the longest actual
calibration sequence, including update, propagation, checkpoint and restore.
Compare losses/gradients/updates to the existing implementation and measure peak
VRAM. Code changes alter solver identity, so importing the current checkpoint
requires an explicit validated migration; do not bypass identity checks. Completed
GSQ candidates, final caches and provenance can be handed through GCS to the
larger-GPU head/RCO stages. Neither this refactor nor its CUDA validation has been
implemented. Compare total runtime and transfer overhead before claiming savings.
