# Measured quantization runtime and estimation plan

## Third attempt: recovered numerical correctness, checkpoint-bound production

The September 20 UTC run passed both actual-Qwen GSQ block types, full-model
full-vocabulary RCO, exact pre-update restored solver/Adam/scheduler/RNG checks,
and the unchanged post-update tolerances. It then completed all 87 embedding
updates and two full GSQ blocks (174 updates). A block-2 start checkpoint is
committed. One subsequent observed update was interrupted while saving and is
not counted as durable. The supervisor reported timeout, not success; the VM
was stopped and deleted with both disks preserved. No GPU remains running.

| Measured operation | Time | Scope |
|---|---:|---|
| Embedding optimizer work | 6.29 s | All 87 actual invocations |
| Embedding stage including load/save | 321.99 s | Includes 295.55 s checkpoint |
| GSQ block 0 optimizer work | 18.20 s | All 87 invocations |
| GSQ block 1 optimizer work | 17.96 s | All 87 invocations |
| GSQ propagation | 7.55 s total | Two complete blocks |
| Completed GSQ checkpoint calls | 869.37 s total | Five calls; includes initial dependency upload |
| GSQ initial checkpoint | 458.78 s | One-time stage dependencies |
| Subsequent GSQ checkpoints | 81.88–124.93 s each | Four measured calls |
| RCO update smoke | 76.08–76.82 s | Three computations of the longest 4,673-token invocation |
| RCO checkpoint restore | 117.37 s | Real generation-pinned GCS restoration |

Checkpoint totals include publication totals; they must not be added together.
The interrupted final save has no completed timer and is absent from the
869.37-second total. The log shows why the prior reserve was insufficient:
a block-boundary checkpoint can run before the next optimizer update observes
the stop deadline, then that update requires another checkpoint. A 90-second
termination grace did not cover this path. The next supervisor must budget from
observed checkpoint costs and avoid treating a timed-out partial save as success.

**Useful partial duration estimate:** the interval between committed block 0 and
block 1 completion was 229 seconds. Transferring that interval unchanged to all
64 blocks gives 4.07 hours (about $7.10 compute at $1.7429/hour) for GSQ alone.
This is a conditional planning scenario, not a bound or a full-run prediction:
only two linear-attention production blocks were measured, whereas 16 of the
64 blocks use full attention; checkpoint contents, later accumulated artifacts,
network throughput and interrupted recovery can change the rate. It excludes
initialization/recovery, embedding, head, RCO, packing and evaluation. Resume
would retain the two completed blocks, rather than retrain them.

The RCO smoke gives an actual 76-second longest-prompt update, but does not
establish the rate across the 87-prompt length distribution or with learned
candidates. Head optimization/checkpointing and final packaging remain untimed.
The offline analyzer therefore correctly leaves complete-job ETA unset. Its
bucket-based projection covers the embedding and linear-attention update work;
checkpoint overhead dominates the observed production wall time.

The VM start/stop timestamps span 2,869.506 seconds (47.83 minutes), approximately
$1.39 compute, excluding CPU preparation, retained disks, GCS/registry and network.
The GPU guest itself reported 2,795.77 seconds; these are different scopes.
GSQ smoke peaked at 63.00 GiB allocated / 66.25 GiB reserved, RCO at
58.61 / 59.23 GiB. The tested operations fit the 96 GB GPU.

Evidence: `docs/benchmarks/quantization-2026-09-19/gpu-attempt-3.json`,
`tools/calibration/runs/cloud-gpu-20260920-v3/{terminal-diagnostics,performance-summary}.json`.
Commit timing is retained in `gsq-commits-observed.txt`; exact remote generations
and hashes are independently checked in `checkpoint-commit-verification.json`.
`observed-final-instance-state.json` preserves the pre-deletion start/stop
timestamps. These files are in the same third-attempt run directory.
Local continuation preparation selected the real block-2 receipt successfully;
remote payload re-verification, host integration and available recovery disk
space remain gates before another paid continuation.

## Prior attempt and estimation methodology

Updated September 20, 2026, after the second bounded GPU attempt.

At the end of the second attempt, a complete ETA was unavailable: actual-model
smoke ran, but recovery equivalence failed before production began. Those
checkpoint-heavy smoke totals could not support a compute-throughput claim.
The third-attempt measurements above supersede that status.

## Measurements retained

| Measurement | Observed value | Scope |
|---|---:|---|
| VM guest execution | 1,318 seconds | Startup through failure; excludes some billable VM startup/shutdown |
| Candidate initialization phase | ~139 seconds | Includes original model load and RTN candidate creation |
| GSQ smoke phase | ~546 seconds | Two updates, model/preparation and durable publication |
| Last GSQ checkpoint publication | 224.93 seconds | Includes bulk uploads and remote verification; excludes torch serialization |
| RCO first checkpoint publication | 285.24 seconds | Includes repeated candidate/warmstart auxiliary files |
| GSQ GPU peak allocated / reserved | 62.95 / 65.86 GiB | Full original model plus largest-sequence block tests |
| Production updates completed | 0 | Stopped at mandatory RCO recovery comparison |

The largest sequence is 4,673 tokens. The entire 87-invocation corpus totals
104,557 tokens: 22.375 times the longest sequence by token count, or 11.296 times
its squared length. Neither ratio alone predicts runtime: fixed per-update
work, hybrid attention, candidate disk reads, optimizer state, full-vocabulary
loss and checkpoint transfer all matter. Multiplying longest-prompt smoke
wall time by 87 would be misleading.

Evidence is in `docs/benchmarks/quantization-2026-09-19/gpu-attempt-2.json` and
`tools/calibration/runs/cloud-gpu-20260919-v2/terminal-diagnostics.json`.
Payload-byte counts include reused files; they are not uploaded-byte counters.

## Instrumentation and estimation requirements

`solver/performance.py` records CUDA-synchronized wall times to the existing
solver log and a bounded, aggregated local `performance-report.json`. It makes
no cloud calls. Exact individual sequence lengths remain in log events;
aggregates distinguish block/type and 512-token length buckets. Corpus cache
propagation records **sum of squared sequence lengths**, not square of their
sum. Failed operations are separate from successful throughput samples.

Measure these separately:

1. One-time loading, initialization and recovery validation.
2. Actual GSQ optimizer work for linear-attention and full-attention blocks,
   plus propagation of teacher/student caches and candidate export.
3. Learned embedding and full-vocabulary head updates.
4. Full-model RCO updates, including candidate streaming and backward pass.
5. Checkpoint serialization/hashing, publication/verification, and restore.
6. Final packing, validation and return transfer (still need end-to-end timing).

`stage_total` includes its child operations, and `checkpoint_total` includes
`checkpoint_publish`. **Do not sum parent and child timings.** Use the enclosing
wall total to identify preparation/export overhead absent from sub-operation
samples. Telemetry I/O failure is reported without replacing the original
solver exception or invalidating an already committed checkpoint; missing
measurements must reduce ETA coverage, never become zero time.

For the current one-epoch recipe there are 5,568 GSQ updates (64 blocks × 87
invocations), 87 embedding, 87 head and 87 RCO updates: **5,829 production
updates**. There are 48 linear-attention and 16 full-attention blocks. This is a
recipe workload, not a promise that one epoch achieves acceptable quality.

Build the ETA from measured sequence-length buckets separately for each stage
and block type. Report measured coverage, observed range and unmeasured costs.
Reconcile predicted work against actual elapsed production time as it accumulates.
Include phase-boundary and periodic checkpoint overhead, packing, and explicit
Spot interruption/recovery scenarios. Do not silently extrapolate a production
rate from only a smoke step or advertise a narrow confidence interval from a
single sample. Additional epochs and allocation budgets change the estimate;
shared GSQ candidates should not be charged repeatedly when reused.

Cost is predicted billable hours times the refreshed whole-VM Spot quote, plus
retained disk/GCS/registry and applicable transfer charges. The earlier quote
was $1.7429/hour for this 96 GB GPU VM; it is not a price for a 3090 or L4.

## Prior recovery blocker and subsequent validation

RCO resumed next-update allocation differed in 76/996 values, maximum absolute
difference 0.0011873245. The prior runtime did not require deterministic CUDA
algorithms. PyTorch documents that RNG seeds alone do not rule out algorithmic
nondeterminism and that strict deterministic mode may reduce throughput:
[PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
That is a plausible contributor, **not a confirmed diagnosis**.

The updated harness verifies exact solver/Adam/scheduler/RNG state before replay,
then compares the next update at the unchanged tolerance. It records both loss
and gradient-norm reports and memory even if equality fails. Strict CUDA policy
must be measured in the next actual run; old throughput cannot automatically
be reused after changing the execution policy. No new paid GPU was started for
these code or test changes.

The third attempt above now validates the strict policy and replay checks on CUDA.
It does not retroactively prove the exact cause of the second attempt’s mismatch.
