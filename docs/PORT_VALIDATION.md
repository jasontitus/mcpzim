# MPS port: validation status

Running log of what has been proven on this machine, what has not, and what was
wrong along the way. Updated as work lands. Written to be read by someone deciding
whether to trust a result.

---

## Proven

### Capture parity — the port reproduces the capture from token IDs alone

`solver/tools/check_capture_parity.py`, invocation-000000 (`cal_coral_reef` turn 1,
468 tokens), on MPS:

| Boundary | Result |
|---|---|
| `first_block_input` (embedding, pre-norm) | **mean 0.00 ULP, max 0.00 ULP** — bit-exact |
| `final_normalized_hidden` (64 blocks + norm) | **mean 0.80 ULP**, max 87.4 ULP (tol 3.0 ULP mean) |

This is the strongest available validation: it exercises weight loading, dtype
handling, embedding, rotary convention, hybrid block dispatch, and all 64 blocks,
against numbers produced by a different implementation (MLX) on a different
machine.

**Why the 0.80 ULP is not a defect.** The capture ran on MLX kernels; this runs on
PyTorch, which falls back to reference implementations for gated-delta and
causal-conv (neither `flash-linear-attention` nor `causal_conv1d` is installed).
Different kernels round differently in bf16. Evidence that this is accumulation
and not a wrong path:

- The error **grows with token position**: mean 0.159 over the first 100 tokens,
  0.401 over the last 100, position correlation 0.235. A wrong path would be flat.
- PyTorch is deterministic here: the same input twice gives `0.000e+00`.
- The magnitude is **under one representable step** — bf16 has 8 mantissa bits, and
  0.80 ULP means the two agree to the last step.

For contrast, my first version of the check applied the final norm twice and got
**tens of ULPs**. That is what a real path error looks like, and it is why the
tolerance is expressed in ULPs rather than absolute units.

### The hybrid block runs on MPS at production width

Measured, not inferred:

| Block type | Params | Forward + backward |
|---|---|---|
| `linear_attention` | 385.9 M | 1229 ms |
| `full_attention` | 374.9 M | 1008 ms |

Use `use_cache=False`; a single-layer linear-attention model with
`use_cache=True` raises a cache-type `ValueError`.

### Measured GSQ cost

Per update (`teacher fwd + student fwd + student bwd`), production width:

| Block type | Per update | One epoch over the 87-invocation corpus |
|---|---|---|
| `linear_attention` | 1758 ms | 2.5 min |
| `full_attention` | 864 ms | 1.3 min |

All 64 blocks (48 linear + 16 full): **2.4 h per epoch**, ~24 h at 10 epochs.
Excludes cache propagation and checkpoint serialization.

### Test suite

`cd tools/calibration && PYTHONPATH=. .venv/bin/python -m pytest solver/tests/ -q`
→ **177 passed**.

Was 173 passing with 4 failures caused by environment, not code: the pinned
upstream sources live at `cuda_runtime/.context/sources/` but ten call sites
hardcoded `/opt/upstream` (the container layout). Fixed with
`solver/upstream_paths.py`, one resolver, so a missing path now names what it
looked for instead of raising `ModuleNotFoundError: No module named 'manifold'`
from inside a stage.

### Gumbel quantizer parity, including on MPS

`solver/tests/test_gumbel_mps.py` (17 tests). With noise pinned so the comparison
measures arithmetic rather than RNG draws:

- CPU vs upstream: forward δ 0.0, grad_logits δ 0.0, grad_scales δ 0.0, hard
  weights equal.
- MPS vs upstream-on-MPS: **0.0 on every quantity** at three temperatures.
- MPS vs upstream-on-CPU: `torch.equal` on forward, both gradients, hard weights.

**The headline finding, and it is not a defect in the port:** `torch.rand(5)` under
`manual_seed(0)` gives `[0.4963, 0.7682, …]` on CPU and `[0.3664, 0.0755, …]` on
MPS. Upstream's Gumbel function draws with `torch.rand_like` on the **device**
stream, so upstream's own MPS forward and CPU forward use *different noise under
one seed*. MPS has no arithmetic problem here — only an RNG one. That is why
noise is drawn on the global CPU stream and stored rather than replayed.

---

## Corrections made during this work

Recorded because each was a wrong claim I made and had to fix.

| Claim | Reality |
|---|---|
| "All 64 layers must be captured." | **Wrong.** `run.py:321-335 cached_inputs` builds teacher/student streams from token IDs; `run.py:411-412` propagates them block by block. The student stream depends on the learned quantized prefix and *could not* be captured in advance. No new capture needed. |
| "2,784 rows is the whole corpus." | **Wrong.** That counts the 32-row diagnostic samples. The solver trains on **104,557 token positions** across 87 invocations. |
| "The 4,096-activation knee applies." | Unverified for this objective — it came from a different pipeline (per-layer `gsq_bits` scalar quantization). Adequacy should be judged on held-out Zimfo behaviour. |
| Parity check: apply `model.model.norm` to the output. | **Wrong.** `Qwen3_5TextModel.forward` already applies `self.norm` (verified by source inspection: TextModel `True`, `Qwen3_5Model`/`ForCausalLM` `False`). Double-applying gave std 1.97 vs the capture's 1.83. |

---

## Not yet proven

- **A real run has not been started.** No GSQ, embedding, head or RCO stage has
  executed against the 52 GiB model end to end.
- **Cache propagation cost** is not measured — 64 extra full-corpus forwards.
- **Checkpoint serialization** cost on an MPS state is not measured; the observed
  cloud payload was 12–17 GB.
- **RCO** loss/backward/update at full scale is untested on MPS. It needs
  `model.gradient_checkpointing_enable()` and a full 64-block forward per
  invocation.
- **Head stage** — full-vocabulary KL over 248,320 rows — untested.
- **Cold checkpoint recovery** on MPS untested.

## Adversarial review findings — all fixed

Three independent reviewers audited the port. Every finding below was verified by
reproduction before being fixed, and the fix re-verified.

### R1 (P0, fatal): every MPS checkpoint was unresumable

`capture_rng_state` attached an `mps` entry whenever MPS memory was live, and
`restore_rng_state` **refused unconditionally** on that entry — so no checkpoint
from an MPS run could ever resume.

The refusal was justified by "MPS replay across a fresh process has not been
validated". **That premise was false.** Measured:

```
capture → draw → restore → draw   reproduces the same six values, bit-for-bit
torch.manual_seed(0) → [0.3664, 0.0755, 0.7784, 0.1186]  (same both times)
```

`torch.mps.get_rng_state`/`set_rng_state` round-trip exactly. `restore_rng_state`
now restores the MPS state, and `tests/test_checkpoints.py` asserts the round-trip
instead of the refusal. The guard against a silent misalignment is real; it was
pointed at the wrong target.

### R2 (P0): `rco.py` mixed a CPU tensor with a device tensor

My own fp64 fix moved `counts` to CPU (MPS has no fp64) and hoisted `self.weights`
back to the device — but left the expression at `rco.py:64` consuming `counts`
un-moved. `RCOTrainer` construction raised
`Expected all tensors to be on the same device, but found at least two devices,
mps:0 and cpu!`, so **the RCO stage could not start on MPS at all**.

Invisible in CI: every `rco_run` test uses a CPU model, where the operands
coincidentally share a device. Fixed by hoisting `counts_dev`; verified by
constructing a real `RCOTrainer` on MPS (17 candidates, both `weights` and `costs`
on `mps:0`).

### R3: the capability gate did not probe MPS BF16

It tested `torch.cuda.is_bf16_supported()` but admitted any machine where
`mps.is_available()` was true, never probing MPS bf16 — contradicting its own
comment that it "remains strict about BF16". Now calls
`device_policy.supports_bf16('mps')`, which probes rather than version-sniffs.

### R4 (high): the architecture guard admitted a non-hybrid model

The check only rejected *unknown* layer types, which an all-`full_attention`
config satisfies trivially. The reviewer loaded a synthetic all-full-attention
Qwen3_5 **successfully** through it — the exact case the guard exists to exclude —
and nothing downstream re-checks, so this guard is the last line of defence for
tensor layout.

Now requires **both** `linear_attention` and `full_attention` present. Re-ran four
adversarial configs (all-full-attention, all-linear-attention, unknown type, wrong
head_dim): all rejected.

### R5 (high, provenance): peak-memory receipt was device-mislabeled

On MPS `memory_stats` returns *current* allocation — MPS exposes no peak counter —
and the receipt wrote it into keys named `cuda_peak_*`, sampled *after* training
state had been released. A reader would treat a non-peak, post-release number as
the run's peak and understate memory.

Split into `peak_memory_stats` (returns `None` where no peak exists — the honest
answer) and separate `memory_current_*` keys, plus `memory_peak_available` and
`memory_peak_absent_reason`.

### R6: `upstream_paths.resolve` did not verify the pinned revision

It checked the directory *shape*, not that `gsq`/`rco` were at the pinned commits.
A stale or wrong `GSQ_UPSTREAM_DIR` would be used silently, and the run identity
would not describe what actually ran. Now verifies by content hash against
`cuda_runtime/.context/sources/manifest.json` — 142 files, all matching.

### R7: checkpoint sync read a non-existent attribute

`getattr(self, 'device', None)` — `Checkpointer` carries no `device`, so the sync
silently no-op'd. Now derived from `module.parameters()`.

### Retracted by the reviewer after measuring

Two apparent defects in `gumbel_mps` were the reviewer's own test artifacts — a
`get_hard_weights` divergence from fp32/bf16 state misalignment, and a dtype
binding issue that only occurs with fp32 weights, which `qwen.py` forbids in
production. Recorded so they are not re-investigated.

## A real stage has now run on MPS

`initialize` — the RTN Q1 candidate database, the first stage in the sequence —
ran against the real 52 GiB model and the real 87-invocation corpus:

```
solver_timing  model_load   29.5 s
solver_timing  stage_total  70.3 s
status: completed    candidate_count: 498
```

498 projections initialized — every eligible linear in the model, including
`lm_head` (248320×5120) and both block types. Output 3.5 GB of candidates plus a
tar, under `runs/native-mps/initialize/`.

Candidate integrity spot-checked: `model.layers.0.mlp.gate_proj` (17408×5120),
`model.layers.3.self_attn.q_proj` (12288×5120), `lm_head` (248320×5120) — all
correct shapes for the real model, each with a content hash.

**Two review fixes are visible in the receipts**, which is the point of having
written them:

```json
"memory_device": "mps",
"memory_peak_available": false,
"memory_peak_absent_reason": "this backend exposes no allocator peak counter
                              (only CUDA does); memory_current_* is a post-stage
                              sample, not a high-water mark"
```

Instead of a `cuda_peak_*` field holding a mislabeled MPS sample, the receipt says
what it actually measured. And:

```json
"backend": "mps",
"deterministic_algorithms": true,
"warn_only": true,
"exact_rng_resume": false,
"caveats": ["Metal/MPS has no cuBLAS/cuDNN workspaces and torch reports many
             operations as nondeterministic, so
             torch.use_deterministic_algorithms(True) is not satisfiable ..."]
```

The policy states its own limits rather than claiming determinism it cannot
deliver.

### Supporting tooling, also new

- `solver/tools/prepare_local_inputs.py` — binds the two local artifact trees
  (`activations/`, `qwen3.8-27b-original/`) into the single-root layout the solver
  expects, with a **derived** receipt: `input_commit_sha256` recomputed from the
  GCS manifest (matches the handoff's `b15a9f03…`), calibration hash from the
  manifest, model coverage from the tree. Hard-links the 52 GiB of weights — no
  data copied — and excludes non-model bookkeeping (`download-status.json`) so the
  solver's coverage check passes. It does **not** fabricate a receipt, which the
  handoff explicitly forbids because that field is inside the frozen identity.
- `solver/tools/make_run_configs.py` — writes per-stage frozen configs against
  local paths. Notably it does **not** pre-bake an `identity` block:
  `bind_identity` *computes* that and raises if a supplied one disagrees. It
  supplies the inputs to that computation instead — `runtime_sha256` is now a
  digest of the ported solver source, not a stale container-image digest, so a
  CUDA-era checkpoint is correctly not resumable under it.

## GSQ trains on MPS — the full smoke stage passes

`solver.run --stage smoke`, on the real 52 GiB model and the real 4,673-token
longest invocation, with the real 498-candidate database:

```json
{
  "status": "completed",
  "smoke": {
    "passed": true,
    "sequence_tokens": 4673,
    "linear_attention_gsq": {"block": 0, "loss": 5.5207e-02, "gradient_norm": 1.4326e-01},
    "full_attention_gsq":   {"block": 3, "loss": 8.1711e-02, "gradient_norm": 6.2721e-01}
  },
  "optimizer_updates": 2
}
```

Both block types produced **finite loss and non-zero gradients**, `optimizer.step()`
ran, and two durable checkpoints were published (`warmstart-block-0.pt`,
`warmstart-block-3.pt`). Timings: block 0 update 10.2 s, block 3 update 6.4 s,
checkpoint publish 24–45 s.

**This is the milestone that matters.** The stack that had to work for it:
model load → dtype → embedding → the hybrid block dispatch → the ported Gumbel
quantizer on MPS → `functional_call` weight substitution → the backward through
the whole chain → optimizer → the scale-grid projection → checkpoint
serialization. Every one of those is now exercised against the real model.

Caveat on the timings: block 3 reports 6.4 s here versus 3.3 s in a failing run
and 3.9 s in the first attempt. The spread is real but small, and the publish cost
(24–45 s) dominates a 2-block smoke. The 1758 ms/update figure measured earlier was
a bare block forward+backward without the trainer's full wrapper.

### A false alarm worth recording

The smoke appeared to fail on block 3 with `FloatingPointError: Actual hybrid GSQ
backward failed` across three attempts. It was **not** a port defect: my harness
reused the local checkpoint store between attempts, and `local` checkpoints are
immutable by design, so the second publish collided
(`Immutable object conflict: smoke_gsq-b000-s00000001-e000-q00001-p00000.json`) —
`publish_ordinal` resets per process while the store path did not.

The collision *masked* the block-3 result rather than causing it, which is why the
reported failure looked like a numerical one. With a fresh store per attempt it
passes. The lesson is for the harness, not the code: **clear
`runs/native-mps/checkpoints` between attempts.** The immutability guard itself is
correct and load-bearing — it is what stops a resume from silently reading a
different payload under a name it has seen before.

Also fixed while chasing this: `smoke.py` still had
`if device.type != 'cuda': raise ValueError('Production smoke requires actual
CUDA')`, the last CUDA gate on the path to real training, plus a `cuda_report`
that would have written CUDA-only fields on MPS. Both are now device-aware, and
the MPS report states `peak_available: false` with a reason rather than a
substituted number.

## A real GSQ run is training on MPS

`solver.run --stage gsq` on the real 52 GiB model, the real 498-candidate
database, and the real 87 invocations. Block 0 (`linear_attention`) updates at
**5.7-9.8 s each depending on sequence length**, loss descending from `5.2e-02`
to `3.7e-02` over the first 21 updates. Checkpoints publish durably; the stage
stops cleanly on deadline and resumes.

### The memory mode that makes it fit

The first attempt died with **no traceback and no crash report** after caching 5
invocations. That is the signature of an external kill, and the evidence
confirmed it: swap was **fully exhausted, 42.1 of 43 GB, with 22.3 GB in the
compressor**. `full` residency holds all 50.1 GiB on the accelerator, and on a
128 GB unified-memory machine that does not leave room for the cache.

The config now uses `block_cpu_offload`, which is what the mode was written for.
Measured, not assumed:

```
mode=block_cpu_offload  execution_device=mps
baseline on CPU: True
MPS before block:  0.00 GiB
MPS inside block:  0.71 GiB      <- one block + rotary
  active block tensors: {'mps'}
  rotary tensors:       {'mps'}
  inactive blocks:      {'cpu'}
  embedding:            {'cpu'}
after block:       0.00 GiB
baseline back on CPU: True
```

Peak accelerator residency is **0.71 GiB instead of 50.1 GiB**, a 70x reduction,
with the residency contract asserted rather than hoped for. Two supporting fixes
went in with it: model load drops from 26.4 s to 2.5 s (weights never cross the
bus), and the block teardown now flushes the MPS allocator cache through
`device_policy.empty_cache` where it previously flushed only CUDA.

Projected cost is 15.7 h for all 64 blocks at 9.6 s/update; the run carries a 20 h
deadline and checkpoints per block, so it stops and resumes rather than dying at
a wall.

## A device-comparison bug that would have hit CUDA too

`GSQResidency` rejected a correctly-resident baseline with *"Baseline residency
does not match requested GSQ memory mode/device"*. The cause:

```python
torch.device('mps') == torch.device('mps', 0)   # False
torch.zeros(1, device=torch.device('mps')).device  # mps:0
```

Configs and execution policy name an accelerator by **type**; tensors report the
**index**. A bare `!=` therefore rejects a baseline sitting exactly where it
should be. CUDA has the identical trap (`cuda` vs `cuda:0`) and only escapes it
because the frozen NVIDIA configs happen to specify indexed devices.

The fix is `device_policy.same_device(a, b)`: type must always agree; when both
sides state an index it must agree too, so `cuda:0` and `cuda:1` stay distinct
while `mps` matches `mps:0`. It is used at all three residency checks and in the
embedding-cache guard in `run.py`, which had the same defect. `objective.py`
compares devices taken from real tensors, which always carry an index, so that
site is left alone.

## Independently re-verified

Run against the current tree, not quoted from an earlier session:

```
$ pytest tests/ -q
345 passed, 3 skipped in 8.02s
```

The 3 skips are honest and specific — they need the external `manifold` RCO
package on `PYTHONPATH`, which this repo does not vendor:
`test_checkpoint_pipeline_integration.py:28` (2 cases) and
`test_continuation.py:397` (1 case). They skip rather than silently pass.

```
$ pytest tests/test_checkpoints.py -k mps_rng
tests/test_checkpoints.py::test_mps_rng_state_round_trips_exactly PASSED
```

The capture-parity claim — the port's strongest correctness evidence — is now a
durable artifact at `runs/native-mps/parity-report.txt`, exit 0, rather than a
number that existed only in prose. It reads:

```
first_block_input (embedding output, pre-norm):
  OK   mean=0.000e+00 max=0.000e+00  0.00 ulp mean, 0.00 ulp max  (tol 0.0 ulp mean)
final_normalized_hidden (all blocks, norm applied inside the model):
  OK   mean=2.617e-01 max=2.869e+01  0.80 ulp mean, 87.43 ulp max  (tol 3.0 ulp mean)
PARITY OK — the port reproduces the capture from token IDs alone.
```

`first_block_input` is **bit-exact**. The final hidden state lands at 0.80 ulp
mean against a 3.0 ulp tolerance. That exercises weight loading, dtype, the
attention implementation, the rotary convention and the hybrid block dispatch,
and it does so against tensors produced by a *different runtime* (the MLX
capture) — which is what makes it evidence rather than a self-consistency check.

## Adversarial review and what it changed

An independent review of the port (read-only, current tree) returned 42 findings
and a verdict of *"nothing found corrupts results silently on the validated
single-device configuration"*. Four were real and are fixed; the rest are
recorded below, including the ones I rejected.

### Fixed: the determinism policy could describe a device the run did not use

`configure(seed)` resolved its backend with an internal
`resolve_device('auto')`, which prefers MPS, while `run.main` resolved the
stage's compute device separately and the BF16 gate probed CUDA whenever CUDA
was visible. On a host with **both** accelerators those three could disagree: the
exported `execution-policy-report.json` would name `cuda` for a run that executed
on MPS, with strict `use_deterministic_algorithms(True)` installed on a backend
the run never touched.

`configure(seed, device=None)` now takes the device, and `run.main` resolves
`compute_device` **once** before configuring the policy and running the gate.
The gate also probes the device the run will actually use rather than whichever
vendor happens to be present. A new `tests/test_execution_policy.py` defends the
contract; against the pre-fix code two of its cases fail (`assert 'mps' == 'cpu'`),
so it is a real regression test rather than a restatement of the code.

### Fixed: `exact_rng_resume` contradicted the code and the tests

The MPS policy hardcoded `exact_rng_resume: False` with a caveat saying the RNG
round-trip *"has NOT been established"*. Both were wrong: the generator is
seeded, captured by `checkpoints._mps_rng_state`, restored by `restore_rng_state`,
and `test_mps_rng_state_round_trips_exactly` passes. The field now reports `True`
and a new `exact_arithmetic_replay: False` carries the genuinely unguaranteed
property, so the report distinguishes *RNG resumes exactly* from *arithmetic may
not replay bit-for-bit* — which is what `warn_only=True` has always meant.

### Fixed: `full` residency could kill the machine instead of refusing

`memory_mode` defaults to `full`, which holds all 50.1 GiB on the accelerator,
and nothing checked that against the device. That is the configuration that
produced the silent external kill documented above. `GSQResidency` now calls
`require_baseline_fits` with the baseline's real byte count, and
`device_policy.total_memory_bytes` supplies the budget.

On MPS that budget is `None`, and the guard declines to decide — consistent with
the port's rule that absence must never be reported as a number. The guard
therefore *helps on CUDA and does not pretend to help on MPS*. Verified at all
four cases: unknown budget allowed, oversized baseline refused, fitting baseline
allowed, offload mode never refused.

### Fixed: the checkpoint store would have exhausted the disk

Not a correctness bug, but it would have ended the run. Measured from the run's
own receipts: **18.34 GB per commit**, and the store reached 43 GB after one
block. Projecting the real commit rate gave **~1.8 TB against 902 GB free**.

The lever was the cadence. `Checkpointer` reads `checkpoint_seconds`; the config
set `checkpoint_interval_seconds`, which nothing reads, so the run silently used
the 120 s default and published far more often than the 15-hour schedule needs.
With `checkpoint_seconds = 1200` a full block publishes **2 commits instead of
~3–6**, and the projection drops to **693 GB against 902 GB free**.

The payload contents were checked rather than assumed. The 8.5 GB of warmstart
archives per commit is *not* duplication: `LocalCheckpointStore.restore` reads
exactly one manifest, and `remap_restored_auxiliary` refuses a resume whose
warmstart is missing or changed, so the anchors must be present in every commit
that a resume could land on. I tested removing them on all but the first commit
and reverted it — that would have broken resume. Content addressing already
dedupes the repeats; the growth was distinct content, not re-copies.

### Rejected, with reasons

- **`torch.mps.empty_cache` is a no-op, so the residency comments are wrong.**
  The premise is right — measured on this build (torch 2.8.0), a 2 GiB transient
  returns to base at `del` and `empty_cache` releases 0 further bytes, because
  MPS does not pool freed blocks the way the CUDA caching allocator does. The
  comments credited it with the 0.71 GiB win; they now credit `block.to('cpu')`,
  which is the actual mechanism, and say why the call is retained anyway.
- **`cached_inputs` is the only non-atomic writer.** True and worth fixing, but
  the specific failure cited was reproduced from *my own* harness: I ran `rm -rf`
  on the output directory while a previous process was still writing it. The
  `mkdir(parents=True, exist_ok=True)` guard is correct; the race was external.
- **"A real GSQ run is training on MPS" is refuted by its own log.** The audit
  read a stale log file from a killed attempt. The run reached **block 0 fully
  trained and block 1 sequence 19** before I stopped it deliberately to apply the
  fixes above; block 0's candidates are preserved under
  `runs/native-mps/evidence/block-0-trained/`.

### Fixed: two more defects found while fixing the first four

**The cache writer was the only non-atomic one.** `cached_inputs` wrote each
activation pair with a bare `torch.save` to its final path. A process dying
mid-write leaves a truncated `.pt`, and the `path.exists()` check at the top of
the loop would then **skip it forever** — silently training on a partially
written activation pair, which is precisely the class of silent corruption this
port is supposed to be verifiable against. It now writes `.pending` and
`os.replace`s into place, matching `atomic_json`, `archive_directory` and
`save_candidate`. Verified on both paths: after a successful write and after a
forced `torch.save` failure, no `.pending` survives.

**The runtime digest could not see a checkpoint-format change.**
`runtime_digest` hashed `solver/*.py` only, but `checkpoints.py` lives at the
package root and is where the port added the MPS RNG capture, the restore branch,
the manifest schema and the snapshot naming. A checkpoint written before that
change would have been admitted by an identity check structurally unable to
notice it. The digest now covers `checkpoints.py` and its siblings; verified by
perturbing the file and confirming the digest moves, then restoring it.

### Fixed: a retry could not reuse its own stage's store

This one cost most of a session, and the failure it produced looks like
anything but what it is.

Snapshot names are built from `stage + cursor + publish_ordinal`, and
`publish_ordinal` restarts at 0 in every process. The local store refuses to
overwrite an existing name with different bytes — correctly, because that is what
stops a resume reading a payload other than the one it was promised. Together
those mean a **second attempt at the same stage writes the same snapshot names
with different state and dies mid-run**:

```
CheckpointError: Immutable object conflict: smoke_gsq-b000-s00000001-e000-q00001-p00000.json
```

The stage is running the real model, so the death lands during a block and reads
like a numerical failure. It masked a *passing* block-3 smoke result across three
attempts: the reported `FloatingPointError: Actual hybrid GSQ backward failed`
was the wrong error entirely. Block 3 measured `loss=8.94e-02`, `gnorm=7.24e-01`,
all gradients finite, both in isolation and inside the full smoke.

`make_store` now scopes the local write root by `config['attempt_id']`, so no
attempt can be poisoned by a predecessor and the collision is impossible rather
than procedural. A resume still reads the original prefix through
`resume_checkpoint`, and a config without `attempt_id` keeps the previous
single-root behaviour, so nothing existing changes meaning.

### The block-3 failure: diagnosed, and made non-fatal

The relaunched run died entering block 3 with
`FloatingPointError: Nonfinite/missing GSQ gradient` on its first step, at 3.43 s
against a normal 4.5-6 s. What that is *not*:

- **Not the port's maths.** Block 3 was replayed from the run's own checkpoint and
  cache and trained **all 87 steps with zero non-finite gradients**, with the loss
  *recovering*: `509 -> 553 -> 455 -> 110 -> 37.9`. GSQ corrects the divergence it
  is trained on; that is the method working.
- **Not the learning rate.** Tested at 1e-3, 1e-4 and 1e-5: gradient norm 93/92/86,
  no non-finite gradients at any of them.
- **Not the warmstart.** Block 3 passes with and without it, and its scales are
  sane (0.02-0.34), well inside the FP16 grid.
- **Not my edits.** They touch `extras` (checkpoint payloads), the store root, the
  policy device and the digest -- none in the training path.
- **Not the store collision** that masked an earlier smoke, though that was real
  and is separately fixed.

What it *is*: a transient. The one state I could not reproduce is 261 accumulated
steps of MPS process state, and free memory measured as low as **4.0 GB** during
that run against 46.3 of 47.1 GB of swap in use. On a unified-memory device with
no allocator cap, a bad value on one step is a plausible outcome; a wrong gradient
19 minutes in is not a reason to discard three completed blocks.

So the check now **counts** a non-finite gradient instead of failing immediately:
drop the poisoned gradients, advance the cursor past the step so a resume does not
replay it forever, and **do not count it as an update** (no optimizer step
happened, so the LR schedule must not move). It logs
`nonfinite_gradient_skipped` so the event is visible rather than silent, and a
`nonfinite_gradient_budget` (default 16) still fails the run when the condition is
repeated, which is what distinguishes a transient from a real numerical fault.
Verified both ways: 16 skips allowed, the 17th raises.

### A note on divergence in the student stream

While chasing the above I measured something worth recording, because it looks
alarming and is not. Propagating the cached student stream through the **trained**
blocks grows it about 4x per block:

```
embed 1.6e-02 -> block 0 5.6e+01 -> block 1 2.2e+02 -> block 2 1.4e+03
```

with the teacher held at ~75. The same chain through **RTN** candidates is flat
(68 -> 71 -> 70 -> 86), and block 3 then trains that input down by an order of
magnitude. So the training objective is doing its job: the divergence is the thing
each block is asked to reduce, and it does reduce it. It is recorded here only so
the number is not mistaken for a defect later.

### The block-3 failure is device state, not data

The evidence is now conclusive, and it took a restore to get it.

Two runs entered block 3 in the same state and behaved differently:

| | long-lived run (261 steps in one process) | fresh process, restored from the run's own `gsq-b003` commit |
|---|---|---|
| `loss` | **non-finite, every step** | **519.3, finite** |
| steps | 17 skipped, then budget exhausted | trains normally |

Same checkpoint, same code, same `cache-3` (student `|max|` 1456 vs 1448,
teacher 75.0 in both, all finite), same warmstart. The only variable is **MPS
device state accumulated over 261 steps**. A fresh process replaying the exact
failing step produces a finite loss.

So block 3 is not broken, the cache is not corrupt, and the divergence I measured
earlier (student growing ~4x per trained block) is *not* the failure -- that
growth is what each block is trained to reduce, and block 3 reduces it from 519 to
38 when it runs.

What the run actually hit is a transient that a long-lived MPS process can
produce, on a machine where free memory was measured at 4.0 GB against 46.3 of
47.1 GB of swap. The guard added for it does the right thing: it logs
`nonfinite_gradient_skipped`, advances the cursor, and refuses to count the step
as an update so the LR schedule does not move. With the budget raised to 256 (a
per-block allowance) a cluster of these cannot end a 15-hour run, while a block
that fails every single step still will.

### The budget design

The guard's budget is deliberately per-run rather than per-step: a transient
arrives in clusters (17 consecutive here), so a small budget converts a live
process into a dead run. A large budget cannot hide a real defect either, because
a genuinely broken block produces non-finite loss on *every* step and 256 of those
exhausts any block's 87 steps several times over.

### A guard that was added, measured, and removed

Worth recording as a mistake, because the reasoning that produced it is
plausible and the measurement that refutes it is cheap.

Block 3 kept failing with a non-finite gradient while a fresh process replaying
the run's *own* block-3 checkpoint produced `loss=519.3`, finite, and trained
normally. That suggested a transient, so a skip-and-continue guard went in: count
the non-finite steps, drop the poisoned gradients, carry on, and fail only past a
budget. It worked -- the run reached block 5, past where two previous attempts
had died.

It was wrong. Every step of block 3 had been skipped, and `export` then wrote the
loaded warmstart straight back out as the block's candidate. Verified directly:

```
export == warmstart hard weights: True   (all 7 projections)
```

So the run recorded block 3 as **calibrated** while never having taken a single
optimizer step. Nothing downstream could detect it: the export is a valid Q1
candidate with the correct shape, correct dtype, and a plausible scale grid. That
is silent corruption of exactly the kind this port's identity checks exist to
prevent, produced by a guard whose stated purpose was to make the run more
robust.

The guard is **removed**. A non-finite gradient now fails immediately with the
block and sequence named, and the message says no optimizer step was taken so the
operator restarts from the last committed checkpoint. A crash that names the
block is strictly better than a run that finishes.

Two checks replace it, both in `gsq_run`:

1. `steps_in_block == 0` at block completion raises. A block that took no
   optimizer step is not calibrated, whatever its export looks like.
2. Every exported candidate is compared against its trained quantizer's hard
   weights and must match exactly.

Check 2 is the general form of the trap: it binds *what was persisted* to *what
was optimized*, which is the property a skip-and-continue guard silently breaks.

### Per-block restart: proven not to change the result

The workaround for the block-3 failure is one block per OS process. The obvious
objection is that it changes the science, so it was measured rather than argued.
The next block's phase-start anchor in a per-block run is **bit-identical** to the
same block's entry state in the single-process run:

```
single-process  gsq-b001-s00000087-e000-q00000-p00002
anchor          gsq-b001-s00000087-e000-q00000-p00002
  solver     identical=True
  optimizer  identical=True
  rng        identical=True
  scheduler  identical=True
  progress   identical=False      (global_step differs; expected)
```

Only the cumulative `global_step` counter differs, which is a reporting field, not
optimizer state.

An independent review raised the LR schedule as the risk: each block builds a
fresh `CosineAnnealingLR`, so a resumed block might follow a different curve. The
commits refute that. Every block in the single-process run shows the same pattern,
and the anchor reproduces it:

```
gsq-b000-...-e000-q00000  last_epoch=  0  lr=[0.001]     # block entry
gsq-b000-...-e001-q00000  last_epoch= 87  lr=[0.0]       # block finished
gsq-b001-...-e000-q00000  last_epoch=  0  lr=[0.001]     # next entry, fresh
```

Each block owns a full cosine anneal over its 87 steps; that is the intended
schedule, not a restart artifact.

## Still not proven

- **No production stage has run end to end.** Everything above validates layers of
  the stack in isolation.
- **RCO loss/backward/update** at full scale on MPS.
- **Head stage** (full-vocabulary KL, 248,320 rows).
- **Cold checkpoint recovery** on MPS after the RNG fix.
- **Cache propagation** cost — 64 extra full-corpus forwards.

### Compression, measured with the codecs that would actually be used

An earlier pass recorded compression as INVALID on a zlib level-1 measurement (40 MB/s,
~1.1x). That verdict was about the wrong codec. Real 256 MiB slices of this run's
payloads, on this machine:

```
payload               raw     zstd -1      zstd -3      zstd -9     lz4 -1   lz4 -9 hc     xz -1    zlib -1
warmstart_block_0   256 MiB  1.10x 183   1.10x 171    1.10x  53   1.00x    1.00x     1.16x   6   1.10x
solver              256 MiB  1.09x 192   1.09x 187    1.10x  67   1.00x    1.00x     1.15x   7   1.10x
optimizer           256 MiB  1.45x 187   1.86x 143    2.11x  32   1.14x    1.65x     2.33x  13   1.70x
cache_block_3       256 MiB  1.33x 159   1.33x 178    1.36x  44   1.00x    1.04x     1.47x   9   1.29x
candidate_block_0    51 MiB  1.03x 177   1.03x 185    1.03x 147   1.00x    1.01x     1.04x       1.03x
mean                        1.200x       1.283x       1.340x       1.029x   1.140x    1.428x       1.243x
```

(ratio, then MiB/s single-threaded)

- **zstd is practical; lz4 is not.** These payloads are float tensors, so lz4's speed buys
  nothing at all (1.00-1.14x). zstd -1 at ~180 MiB/s puts a 16.6 GB publish at ~90 s
  against ~52 s today -- affordable. zlib was four times slower *and* worse, which is
  where the "invalid" verdict came from.
- **The optimizer state is the compressible payload** (1.45x at zstd -1, 2.11x at -9,
  2.33x at xz -1). Warmstarts, solver and candidates sit at 1.03-1.10x and set the
  ceiling. xz wins on ratio and loses hopelessly on time (6-13 MiB/s).
- Weighted by the bytes a block actually writes, zstd -1 gives ~**1.46x** on the
  incremental store and ~1.11x on the per-attempt invariant bulk: a 64-block run goes
  from ~1.6 TB to ~1.1 TB. A real 30%, and nowhere near enough on its own -- which is
  why the structural levers above (one root, prune, and above all a single process) are
  the ones that decide whether the sweep fits.

## Correction: the disk projection above is superseded — the sweep does not fit

The entry under "Fixed: the checkpoint store would have exhausted the disk" records
**693 GB against 902 GB free**, from a measured **18.34 GB per commit** at a cadence
of two commits per block. Two things have changed since, and neither was re-derived
when the change landed.

First, free space: `df -h` now reads **588 GiB**, not 902 GB, and the store itself
reaches **335 GB** across twelve attempt roots.

Second, and this is the substantive one: that projection assumed **one** store root,
where content addressing dedupes the repeated payloads. The write root is now scoped
per `attempt_id` (`run.py`, `make_store`) to make the snapshot-name collision that
cost this project three attempts *structurally* impossible — and a per-block sweep
gives every block a **different** attempt id. Dedup is per root only
(`checkpoints.py`, `_upload` copies, `_install` hardlinks; every object in every root
has `st_nlink == 1`), so each attempt re-stores the payloads that never change.

Measured, not inferred, by comparing the block-1 anchor in `20260920-075700-ba75`
with the block-1 phase-start in `20260920-080955-41f6` (same cursor, different
`publish_ordinal`): `warmstart_block_0`, `warmstart_block_3`, `initial_candidates`,
`packing_cost_manifest`, **`solver`** and **`cache_block_1`** are byte-identical —
**16,605,521,521 B re-stored for one attempt**. Per block the store therefore grows
by 16.606 GB (phase-start) + 4.689 GB (block-end: optimizer 3.089739 + solver
1.544868 + candidate 0.053903) + 3.686 GB (anchor) = **24.980 GB**, and 28.070 GB for
blocks 0 and 2, which also carry fresh invariants:

```
64 blocks = 2 x 28.070 + 62 x 24.980 = 1,604.9 GB = 1,494.7 GiB
free today: 588 GiB          the store alone fills the disk around block 25
```

Add the per-run output scratch that is never deleted — `restored-<snapshot>/` is
16.61 GB and survives the run, 9.14 GB of it (the warmstart copies) never read on the
gsq path — and the combined marginal cost is ~49.4 GiB/block, which exhausts the disk
**inside block 12**. The recorded 693 GB undercounts by ~2.3x because it counts
solver+optimizer once per publish and omits the per-attempt invariant set.

Measured levers, largest first:

| lever | saved over 64 blocks | risk |
|---|---|---|
| delete `restored-<snapshot>/` after `restore_state` | 1.05 TB | none: nothing reads it afterwards on this path |
| one shared store root, attempt discriminator moved into the snapshot name | 1.05 TB | must also fix the name, else a retried block republishes a name with different bytes |
| delete the ten abandoned attempt roots (no code change) | 288 GB | none if the resume chain in the config is kept; **these roots are also the evidence for the comparisons in this document** |
| drop the never-read invariants (warmstarts, initial_candidates) from later commits | 814 GB | breaks cold-host/GCS resume; the re-add exists for that reason |

Measured per-attempt, from one attempt's own manifests (`20260920-100739-35a4`, three
commits):

```
phase-start   13 objects  16.71 GB   -> 1.54 GB reclaimable once the next commit lands
block-end     14 objects  19.86 GB   -> 6.78 GB reclaimable
anchor        14 objects  19.72 GB   -> keep
store on disk 28.05 GB; referenced by the latest commit 19.72 GB; orphan objects 0
```

So a chain-aware prune reclaims **8.32 GB per attempt, 30% of the store**, and the
publish path leaves no orphans. That tool now exists: `solver/tools/prune_store.py`
(dry run by default, `--apply` to delete). It deletes only objects no retained commit
references and keeps every manifest, so the store's immutability contract survives --
re-publishing a pruned snapshot re-uploads what it needs, and an existing name with
different bytes still fails. Verified on a hardlinked copy of `20260920-100739-35a4`:
10 objects / 8.32 GB deleted, the retained anchor re-verified as restorable afterwards,
the original store untouched. A pruned commit is no longer restorable, which is the
intended trade -- a per-block sweep only ever resumes from the newest anchor.

### If the per-block driver has to stay: one root, attempt-scoped names

Fallback for the case where one process cannot carry all 64 blocks. The store scopes its
WRITE root per `attempt_id` (`make_store`) so two attempts cannot publish the same
snapshot name with different bytes, while the READ path is deliberately unscoped so a
resume reads the prefix it was told to read. Sharing one root across the sweep stores the
invariant payloads once -- 16.6 GB per attempt today, measured by comparing two roots'
byte-identical solver/cache/warmstart/initial-candidates payloads -- and saves ~1.05 TB,
but it re-opens exactly the collision the scoping closed.

The fix is to make the *name* unique instead of the root. The snapshot name already
carries the stage, block, cursor, epoch, sequence and a per-process publish ordinal:

    gsq-b003-s00000261-e000-q00000-p00002

Adding the attempt discriminator leaves immutability intact in a shared root, which is
what lets the root be shared. `identity` deliberately excludes `attempt_id`, so manifests
stay comparable across attempts. That change plus `prune_store.py` is what lands the
~450 GB figure; without the single process, do both.

Putting the levers together, for 64 blocks:

| configuration | store at the end |
|---|---|
| as driven today (one root per attempt, no prune) | ~1.6 TB |
| + one shared root (invariants stored once) | ~790 GB |
| + chain-aware prune | **~450 GB** |
| one process for all 64 blocks, pruned | **~25-30 GB** |
| any of the above, zstd -1 | -30% on the store, -0% on the duplication |

Confirmed against a live 2-block run of the real model (the verification run described
below), which also validates the per-attempt model end to end:

```
block 0 store, 3 commits        26.1 GiB   (predicted 28.07 GB = 26.1 GiB)
block 0 output scratch           6.1 GiB   fresh attempt
resumed attempt output          29.0 GiB   of which restored-<snapshot> 16.61 GB
store total after one block       377 GB    (335 GB before the run)
```

It fits only with the first three, landing at ~520 GiB of store with ~47 GB of
transient peak — but note what the live numbers add: store growth is only half the
cost. A resumed block costs ~23 GiB of store *plus* ~29 GiB of output scratch that
nothing deletes, so ~52 GiB per block puts exhaustion at roughly **block 6** against
the 511 GiB free at the time of writing, not block 25. The restore reclaim is
therefore not optional housekeeping; it is the difference between a sweep that
finishes and one that dies in its first hour.

One more cost comes from the retention change itself: a restored
`candidate_block_<n>` is materialised three times per attempt — the payload inside
`restored-<snapshot>/`, the copy `restore_candidates` makes as
`output/candidate_block_<n>.tar`, and the directory it extracts into — and all three
live for the run. At 53,903,360 B per completed block that is ~162 MB per block per
attempt, ~10 GB in the 64th attempt's output alone and ~330 GB summed over a retained
sweep. The reclaim above removes the first copy; the other two are payload and
database targets, so they stay.

## Correction: what the per-block restart comparison does and does not prove

The table under "Per-block restart: proven not to change the result" compares
`solver`, `optimizer`, `rng` and `scheduler`, and notes `progress` differs. It does
not compare **`cache_block_1`**, which is the actual input the restarted block
trains on — and that payload is *not* invariant: all **six** stores that contain
`gsq-b001-s00000087-e000-q00000-p00002` carry a different `cache_block_1` hash
(`8a2716f5…`, `797611b8…`, `be8ed9ec…`, `3392f518…`, `0543be94…`, `73fe05cc…`),
because each came from a different attempt at block 0.

That distinction matters, and it is worth stating precisely rather than leaving the
table to imply more than it shows:

- **Proven, and now more strongly than before:** the resume *point* is
  process-independent. Across those same six independent attempts, the anchor's
  `solver` (`5f9ea857…`), `optimizer` (`aea89b95…`, 1593 B), `rng` (`bc5753aa…`) and
  `scheduler` (`3dc1fc85…`) are byte-identical. Six processes that each trained
  block 0 differently produced the same next-block entry state. The later
  `already_complete` change rests on exactly this.
- **Not proven by that table:** that a restarted block trains on the same inputs.
  That needs `cache_block_1`, and the per-block chain is what supplies it — the
  anchor's cache is propagated by the process that just trained the block
  (`run.py`, `cache_propagation` → `cache_block_{block_index+1}`), so a sweep that
  resumes the *same* chain is consistent. Comparing runs from *different* block-0
  attempts is meaningless for the trained result, by construction.

## The per-block sweep had two silent-failure defects; both are fixed

The workaround routes around the block-3 failure, but the driver implementing it and
the anchor supporting it would have produced a fully "successful" 64-block sweep
with an unusable artifact. Both defects are fixed, both have a reproduction, and
three independent adversarial reviews found the first two independently.

### A resumed process discarded every previously-trained block

`gsq_run` deleted each completed block's archive from `extras` once the next block
started, and the block-boundary anchor filtered `candidate_block_*` out of its
payload entirely. On resume `database` is rebuilt by `restore_candidates` from the
archives the snapshot carries, and the block loop re-exports only the block it is
training — so every earlier block fell back to its untrained RTN
`initial_candidates` while the run reported `completed`. The deleted comment claimed
the opposite ("cumulative in `database` and are re-exported on resume"); no
re-export path has ever existed. The artifact side confirms it: the anchors already
on disk carry no `candidate_block_*` at all
(`runs/native-mps/gsq-anchor1/restored-gsq-b001-…`).

Reproduced with the real `gsq_run` on a two-layer CPU model, two processes with
`gsq_max_blocks=1`, against a single-process reference of the same work — the
per-layer split is what makes the failure legible:

```
                      layer 0                layer 1
before the fix    (0 same, 8 DIFFERENT)   (7 same, 0 different)
after  the fix    (8 same, 0 different)   (7 same, 0 different)
```

Note the second column: the block the final process actually trained reproduced the
reference *exactly*, so the loss is purely the hand-off, not drift.

Fix: keep every completed block's archive in `extras` and in the anchor payload.
The store is content addressed, so retention costs one stored copy per block (54 MB
measured) plus a re-link per publish; `archive_directory`'s
"existing archive differs" re-verification is what makes re-shipping it safe.

### The driver could not run, and would have called a mid-block stop "complete"

The heredoc that was supposed to give each iteration a fresh `attempt_id`, a fresh
output directory, `gsq_max_blocks=1` and a `resume_checkpoint` built all four in
memory and **never wrote them back**. Every iteration therefore ran the stale config
on disk: iteration 1 dies immediately on the existing non-empty `output` (the
`FileExistsError` guard is by design), and any iteration that did run would have
written all 64 blocks into one store root under one attempt id — the exact collision
arrangement the per-attempt scoping exists to prevent. The handoff called this fix
"compiled but unverified"; it was strictly worse than unverified, because the write
was absent.

Separately, the driver derived the next block from
`report.get("next_block", -1)` and read `-1` as "all blocks complete".
`checkpointed_stop` — the status returned by a signal, a passed deadline, or
`--max-steps` — carries no `next_block`, so a deadline stop would have ended the
sweep with exit 0 and 62 blocks untrained. This is not hypothetical: the report at
`runs/native-mps/gsq-anchor1/gsq-report.json` is exactly that shape
(`status=checkpointed_stop`, block 1, sequence 47 of 87, no `next_block`), and the
config of record carries a `deadline_unix`.

The driver is rewritten against a stated contract:

- the **config of record is never mutated**; each iteration writes
  `pb-config-<stamp>.json` beside it, and the identity hash excludes
  `attempt_id`/`output`/`resume_checkpoint`, so a per-iteration config stays
  identity-stable across the sweep
- it advances **only** on `blocked_stop`; `completed` ends the sweep; a mid-block
  `checkpointed_stop` resumes the same block from its own snapshot **without
  advancing the block counter**; any other status, nonzero exit, passed deadline, or
  a republished snapshot stops it loudly with both stores left in place
- a passed `deadline_unix` is checked *before* starting a block, because otherwise
  every resume stops again after one step and "progress" never ends

### The workaround is verified end to end: two blocks, one process each

`runs/native-mps/run-gsq-per-block.sh 2` on the real 52 GiB model, 25.2 minutes:

```
=== block 0 -> runs/native-mps/gsq-pb-20260920-083644-584e (fresh run) ===
=== block 0: status=blocked_stop snapshot=gsq-b001-s00000087-e000-q00000-p00002 next_block=1 ===
=== block 1 -> runs/native-mps/gsq-pb-20260920-084920-7bd4 (resume gsq-b001-s00000087-e000-q00000-p00002 from .../20260920-083644-584e) ===
=== block 1: status=blocked_stop snapshot=gsq-b002-s00000174-e000-q00000-p00002 next_block=2 ===
=== reached the block limit (2) ===
```

Block 0 published its successor's phase-start anchor, and a *separate process* resumed
from exactly that snapshot and trained block 1 through 87/87 steps (loss 0.1135, from
0.0505 at block 0's start). This is the §3.1 dry run the handoff asked for and the
first time the driver has ever completed two blocks.

The hand-off of trained state is byte-exact, not merely nominal: `candidate_block_0`'s
sha256 is identical in iteration 1's anchor and in iteration 2's (`be6006a6…`), and
iteration 2's anchor carries `candidate_block_0` **and** `candidate_block_1` — the
cumulative retention working on the real model. Iteration 2's own
`candidate-database.json` covers layers 0 and 1, with their trained files extracted
into its own `block-0/` and `block-1/`.

Costs from the same run: model load ~2 s, 87 steps at ~7.5 s/step, cache propagation
74.4 s, ~50 s per publish, 26.1 GiB of store for block 0, and 6.1 GiB of output for
the fresh attempt against 29.0 GiB for the resumed one (16.61 GB of it the restored
directory). That run predates the driver's reclaim of `restored-*`, so the reclaimed
path is verified against the stub harness rather than on the real model; the leftover
directory is still on disk at `runs/native-mps/gsq-pb-20260920-084920-7bd4/`.

The rewritten driver's control flow is covered by a stub-runner harness (15 checks
over a `/tmp` copy of the script driving a stub `solver.run`): advance on
`blocked_stop`, end on `completed`, stop loudly on a nonzero exit or an unknown status,
refuse past a deadline, refuse a bad `max_blocks`, resume a mid-block stop without
advancing the block counter, bound repeated mid-block stops, reclaim the restored
scratch, and never mutate the config of record. It found three live defects in the
first version of this driver — no reclaim, no bound on repeated mid-block stops, and a
`max_blocks` typo that exited 0 having launched nothing.

### The zero-step guard rejected a legitimate resume

The guard added earlier in this work — `steps_in_block == 0` raises, so an untrained
block can never be recorded as calibrated — also fired when a resume landed on a
block's **own end state**, where the epoch loop correctly does no work and only the
export remains. It broke `test_final_block_and_next_phase_checkpoint_resume`, and it
would have wedged a sweep whose stop landed on a block's final update: every later
resume of that snapshot fails identically. The guard now exempts a block that was
already complete on entry (`epoch >= epochs`, or the final epoch with
`sequence >= len(records)`); a *freshly entered* block with zero steps still raises.
The export round-trip check is unchanged and still runs on the exempted path.

### The tree arrived with two failing tests

`526 passed, 3 skipped` now. It was `2 failed, 524 passed, 3 skipped` before this
session's fixes, so the handoff's "349 passed, 3 skipped" was already stale and
nothing had run the suite against the tree as left. One failure was the guard above;
the other was an assertion pinning an incidental dict shape
(`result['cuda'] == {'available': False}`) that the deliberate addition of
`device_type` invalidated — it now asserts the property it was written to protect: a
CPU fixture must not be labelled as a GPU. One thing is noted but not fixed:
two near-duplicate copies of the port exist under
`tools/calibration/cuda_runtime/.context/application/solver/` and have **already
diverged**; no script in `tools/calibration` generates them, so they are a stale fork
that a future reader could mistake for the live tree.

## Process: adversarial review as a build step, not a postscript

Four failure classes in this work were all reviewable, and none were caught by
reading code:

1. **A guard meant to add robustness silently corrupted five blocks** and was only
   caught by measurement (below, "A guard that was added, measured, and removed").
2. **The driver's config write was simply absent**, while the handoff described it as
   "compiled but unverified" — a status claim with no runnable check behind it.
3. **Two tests were failing at handoff time** while the handoff recorded a clean
   suite.
4. **A recorded measurement was invalidated by a later change**: the 693 GB disk
   projection assumed one store root, and per-attempt scoping removed that premise.
   Nothing re-derived it, and the number was still being quoted.

The practice that follows from these, and that this session ran:

- **Reviews are artifacts, in the repo, with a disposition.** Every review this
  session produced `<reviewer-id>.json` with `confidence`, `file:line` evidence, and
  a CONFIRMED/REFUTED/UNDETERMINED verdict per claim. Ephemeral coordination (a hub
  job id) is not a record — the previous session ended with two reviews that nothing
  could recover.
- **Reviewers are handed a verified scope block**, not a paraphrase: absolute repo
  root, exact files, the measured facts they may rely on, and the claims to *falsify*.
  An earlier review round was pointed at an empty clone of the wrong path and wasted
  its whole budget.
- **Line numbers come from a tool, never from a summary.** Handing reviewers the
  line numbers from a structural summary cost one round trip this session: the
  numbers were ~70 lines off because they came from a different file (a vendored
  copy of the same module).
- **Measurements record the configuration they were measured under**, and a change
  that alters that configuration re-derives or marks them stale.
- **No status claim without the command that produces it.** "Tests pass" is the
  output of the suite command; "the driver works" is a completed 2-block run, with
  the log named.
- **Reproduce before fixing, and keep the reproduction.** The candidate-loss defect
  above is a 30-line CPU script; the same defect reached the point of being written
  into a handoff as "unverified but compiled" without anyone running it.

## The block-3 failure is not process state, and the per-block restart does not route around it

This supersedes "The block-3 failure is device state, not data" above. That
conclusion rested on one experiment: a fresh process restoring the run's own
`gsq-b003` commit produced a finite loss (519.3) where the long-lived process had
produced a non-finite one. Re-measured today, from one checkpoint, three times:

```
10:20  block 3 resumed from gsq-b003-s00000261-e000-q00000-p00000  -> loss=nonfinite at sequence 0
10:24  same snapshot, fresh process, seeded driver resume          -> loss=nonfinite at sequence 0
```

Both fresh processes, both failing on the FIRST update, both leaving no optimizer
step behind. The inputs are not the problem and neither is the process:

- **The whole cache is finite.** All 87 files of the block-3 input cache were loaded
  and checked: zero non-finite entries, worst `|value|` 1440. (The earlier note
  sampled and compared maxima; this scans every file.)
- **The restore verified.** The snapshot restored under its identity, and the new
  read-side check confirmed it carries `candidate_block_0/1/2`, so blocks 0-2 are
  genuinely trained in it.
- **The failure is deterministic per input.** Same checkpoint, same result, twice.
  A device-state transient would not reproduce bit-for-bit on the first update.

What actually varies is the *student stream*, and it is compounding. Measured from
the archived caches of this sweep (student max / teacher max at the longest
invocation):

```
input to block 1   student     56.0   teacher   37.5   ratio  1.49
input to block 2   student    224.0   teacher   54.0   ratio  4.15
input to block 3   student   1440.0   teacher   75.0   ratio 19.20
```

**~4x per trained block**, while the teacher stays flat at 37-75. Each trained
quantized block amplifies its input, so the calibration error compounds
multiplicatively instead of being held. By block 3 the student stream is 19x the
teacher's, and block 3 is the first `full_attention` block -- the first place that
stream goes through an attention softmax. Blocks 0-2 are `linear_attention`
(confirmed from the run's own event log), whose recurrence has no exponential in it,
which is exactly why they pass and this one does not.

That also explains why the earlier replay succeeded: the smoke builds its own input
from the FROZEN model (`smoke_run` runs the preceding blocks with `run_block`), so
its student stream is the teacher's scale, ~75, and never 19x. And the earlier
fresh-process replay used a *different* attempt's cache (`|max|` 1456/1448 against
1440 today) -- each attempt trains block 2 independently, so each propagates a
different student stream. The failure is deterministic per cache and intermittent
across attempts, because the port is sitting at the edge of the numerical range at
the first attention block.

Consequences, and what does not follow:

- **Per-block restarting cannot fix this.** It changes when the code runs, not what
  it is fed. The workaround's stated premise ("a fresh process trains block 3 fine")
  does not hold for a propagated cache, which is the only kind the sweep has.
- **The fix belongs in the student stream's scale**, not in process management or
  memory pressure. The 4x-per-block amplification is the thing to explain: it is the
  product of trained quantized blocks, not of the training in progress, so a
  per-block renorm (or finding why the quantized forward drifts) is the lever.
- **The `--stage smoke` discriminator is unavailable on this machine.** It refuses
  `gsq_memory_mode: block_cpu_offload` ("Block CPU offload is implemented only for
  GSQ") and would need `full` residency, which is the configuration that silently
  killed this machine earlier (50.1 GiB on the accelerator, swap exhausted) -- and
  swap was already at 46.1 of 47.1 GB when this was written. Do not reach for it
  without a quiet machine.

## Block 3 is not broken: the trigger is in `run.main`'s prologue

The section above recorded that block 3 fails deterministically in fresh processes with
a finite cache. That is true, and it is not the block's arithmetic. Bisected on this
machine, the failing step reproduces as a *finite* update in every reconstruction of
the computation:

```
frozen teacher forward, MPS                      finite   max 82
quantized loss, fresh trainer                    finite   486.151
quantized loss, restored solver+optimizer+sched  finite   487.858     (the run's exact state)
quantized loss, same params + process RNG        finite   490.233
quantized loss, deterministic_algorithms=True    finite   486.768     (0 warnings raised)
quantized loss after the 19.56 GB publish        finite   487.858     gradients 14, all finite
quantized student forward, MPS                   finite   max 1440
direct gsq_run, real config+corpus+model+state   finite   487.858     one update, clean stop
with the real 19.72 GB restore materialisation   finite   487.858     one update, clean stop
```

So the same update that the CLI refuses to take is finite when the same code is called
directly with the same state. The CLI itself reproduces it in **165 s**
(`solver.run.main` with a synthetic argv, `--stage gsq --resume
gsq-b003-s00000261-e000-q00000-p00002`), which is the reproduction to use from here
rather than a 12-minute block.

Bisecting `main`'s prologue -- the only work a direct `gsq_run` call skips -- settles it.
Stubbing the three module-level steps:

```
validate_restored_inputs(config)      -> lambda config: {}
bind_identity(config, manifest)       -> config['identity']
reproducibility.configure(seed, dev)  -> returns a dict
```

makes block 3 **train 44 steps** (global_step 261 -> 305, loss 408-470 and falling)
before this session's own cleanup deleted its restored directory out from under it --
which is a mistake worth recording: a 1.4% CPU reading on a process stepping every ~7 s
on MPS looks like a hang, and it is not. Do not reclaim a probe's scratch while its
process is alive.

Stubbing `configure` **alone** trains block 3 to completion. The run stepped all 87
updates (loss 487.858 -> 158.845, falling throughout) and `main` returned without
raising:

```
{"block": 3, "sequence": 85, "global_step": 346, "last_loss": 171.79324340820312}
{"block": 3, "sequence": 86, "global_step": 347, "last_loss": 162.86917114257812}
{"block": 3, "sequence": 87, "global_step": 348, "last_loss": 158.84535217285156}
VERDICT: main returned without raising (the stubbed steps are implicated)
```

It exported its candidates too (`block-3/` and `block-3.tar` in
`runs/native-mps/probe-bisect-20260920-105231/`), and `export` only runs after a block's
epoch loop finishes. So the block is intact, the block is trainable, and what stood in
the way was one prologue call. `configure` is what `run.main` calls before the model is
loaded, and on MPS it installs:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

`reproducibility.py`'s own docstring says that flag buys nothing on Metal -- "strict mode
would abort on the first unsupported op without buying any reproducibility" -- and the
module's `MPS_CAVEATS` already record that MPS arithmetic is not bit-reproducible. Note
that setting the flag *after* the model is loaded and the state restored does **not**
reproduce the failure (probe 5 above is finite): what matters is the flag being in
effect while the model is loaded and the block is prepared, which is a kernel-selection
effect rather than a property of the update itself.

Two diagnostics are kept in the tree so this does not have to be rediscovered:
`solver/tools/repro_block3_cli.py` (reproduces the failure through `main` in ~165 s)
and `solver/tools/bisect_main.py` (`STUB=configure` and it trains).

The ordering is the whole of it, and it reproduces in ten seconds with no run at all.
Same block, same input, same restored state, the flag the only variable:

```
flag installed AFTER the load   (load -> restore -> forward)   finite   486.768
flag installed BEFORE the load  (flag -> load -> restore -> ...)   finite=False  value=nan
```

`_configure_mps` no longer installs the flag; `run.main` installs it immediately after
`load_original`, so the recorded policy is unchanged (`deterministic_algorithms: true,
warn_only: true`) and the flag still does its diagnostic job during training. The test
suite passes unchanged (529 passed, 3 skipped) because `test_reproducibility.py` pins the
CUDA branch explicitly and never asserted the MPS install.

A consequence worth stating plainly: `identity` is bound to the solver source, so this
edit -- like every edit to `solver/*.py` -- severs the existing chain. The blocks
produced before it are not resumable under the new code ("a checkpoint is resumable
under the code that wrote it and nothing else"), which is the intended semantics but
means a 64-block sweep is ~12 hours of wall time with no edit landing inside it.

Consequences:

- **The per-block restart, the cache, the warmstart, the RNG, the memory pressure and
  the student-stream scale are all exonerated** by the table above. Nothing needs to be
  fixed in the arithmetic.
- **The next step is a one-line experiment**: run block 3 with the determinism flag
  never set (or set after `load_original`), confirm it trains to completion, then decide
  whether the flag stays on MPS at all. The `execution-policy-report.json` records
  `deterministic_algorithms: true, warn_only: true, exact_arithmetic_replay: false`
  today, so turning it off changes a recorded field, not a scientific guarantee that was
  ever valid on this backend.
- **A 64-block sweep is therefore reachable** once that is settled, which is what makes
  the disk levers relevant again.

### The identity includes run controls, so two modes cannot share a chain

`bind_identity` hashes every config key except a listed few, and that list covers
*plumbing* -- `output`, `attempt_id`, `checkpoint`, `resume_checkpoint`, `deadline_unix`
and the input paths -- but not *operational controls*. `gsq_max_blocks` is therefore part
of the checkpoint identity, and by the same logic so are `checkpoint_seconds` and
`target_bytes`.

The consequence was hit directly tonight: the per-block driver sets `gsq_max_blocks=1`,
so a single-process run (which must leave it unset) **cannot resume a driver-produced
chain** -- the identity differs and `restore` refuses it. The two modes cannot share a
chain, and changing the checkpoint cadence mid-sweep would invalidate everything
published before the change.

"How many blocks one process runs" and "how many seconds between checkpoints" say nothing
about what the calibration *means*. The scientific surface is the learning rate, epochs,
seed, memory mode, execution device and dtype. Excluding the controls would let the modes
share a chain and let cadence be tuned during a 12-hour run instead of being frozen for
its duration. It is a one-line change to the exclusion set, and it severs whatever chain
exists when it lands, so it belongs before a production sweep rather than during one.

## The runtime pin is a pin, not a check

`runtime_sha256` is meant to identify the code that wrote a checkpoint. It is
computed from the source itself — every `solver/*.py`, the checkpoint-format modules
beside the package, and the pinned upstream revisions
(`solver/upstream_paths.py`, `runtime_digest`) — but it is computed by exactly **one**
caller, `solver/tools/make_run_configs.py`, which writes the value into a run config.
Nothing recomputes or compares it at run time: `run.py` reads
`config['runtime_sha256']` and stamps it into the identity (`bind_identity`), and the
store enforces equality of that identity on restore.

The pin is therefore only as truthful as the last config regeneration. This session's
fixes to `solver/run.py` and `checkpoints.py` changed the code that digest covers,
while the config on disk still carries `b0e32346…` from before them. Every snapshot
written since — including the ones this session's verification run is writing —
records a runtime identity that no longer describes the tree. The chain is still
self-consistent, because every process in it reads the same pinned value; that is
precisely why nothing complained.

Before a production sweep: regenerate the config with
`solver/tools/make_run_configs.py`, and expect the new pin to invalidate the existing
anchor chain, since `restore` verifies the identity it was promised. The sweep then
starts from block 0. That matches the intended semantics — "a checkpoint is resumable
under the code that wrote it and nothing else" — but it should be a deliberate step
rather than a surprise at block 40. The durable fix is for `run.py` to recompute the
digest at startup and refuse a config whose pin disagrees with the tree; that is a
deliberate behaviour change and should land together with the regenerated config, not
in the middle of a running sweep.