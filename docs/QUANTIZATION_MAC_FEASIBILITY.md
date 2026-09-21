# Qwen3.8-27B GSQ-RCO on this Mac: feasibility and plan

Written after direct measurement, not reading. Corrections to
`QUANTIZATION_MAC_OVERNIGHT_HANDOFF.md` are called out where the code disagrees
with it.

---

## Verdict

**A completed 27B GSQ-RCO model tonight is unlikely but no longer impossible.**
The data blocker is retracted (see Blocker 1) and measured compute is ~2.4 h per
GSQ epoch over all 64 blocks. What remains is the device port, which is real but
bounded and partly landed. Whether it finishes tonight depends on how much of the
port completes and how the first real block behaves.

| | Status |
|---|---|
| Disk | **985 GiB free** — the handoff's "28 GiB" is stale. Not a constraint. |
| Memory | 128 GiB unified. Not the binding constraint. |
| MPS can run the hybrid block | **Yes — measured.** Both block types, forward *and* backward, at full production width. |
| Native MPS training path | **Does not exist.** Confirmed by source audit. |
| Calibration data supports GSQ | **Yes — 104,557 token positions. My earlier objection was wrong.** |

---

## Blocker 1: RETRACTED — the corpus is adequate

I previously reported that the 87-invocation corpus was a stratified sample too
thin to train on. **That was wrong**, and the correction matters because it
removes the only fatal blocker.

What I misread: the 32-row `x` tensors per projection are **diagnostics**. The
inputs GSQ actually trains on are generated at run time from token IDs:

```python
# run.py:321-335  cached_inputs
ids = torch.tensor([tokens], device=device)
reference = model.model.embed_tokens(ids)
student = gather_candidate(embedding_candidate, ids, model.model.embed_tokens.weight)
torch.save({'teacher': reference.cpu(), 'student': student.cpu()}, path)
```

```python
# run.py:411-412  per-block propagation in gsq_run
teacher = run_block(model, block_index, tensors['teacher'].to(device))
student = trainer.hard_forward(tensors['student'].to(device))
```

So the effective training set is **104,557 token positions across 87 invocations**,
including the full 4,673-token longest sequence — not 87 × 32.

**And the student stream could not have been captured ahead of time at all:** it
depends on the learned quantized prefix. An all-layer capture of original-model
hidden states would yield the teacher stream and nothing else. The existing CUDA
run is the proof — it completed blocks 0/1 and durably stored the block-2 cache
this way.

Everything required is local:

| | |
|---|---|
| Token IDs + provenance | captured, integrity-checked by `corpus_inputs` |
| Original BF16 weights | `runs/qwen3.8-27b-original/`, ~51.8 GiB |
| Reference `lm_head` | `model-00018-of-00018.safetensors`; promoted to FP32 by the objective |
| Reference log-sum-exp | computed by `FullVocabularyKL`, not captured |

**No new capture is required.** See `docs/CAPTURE_REQUEST.md`, which is now a
retraction rather than a request.

### Caveats carried forward

- The "4,096-activation knee" I cited earlier came from a **different pipeline**
  (per-layer `gsq_bits` scalar quantization). It has not been verified for this
  objective; adequacy here should be judged on held-out Zimfo behaviour.
- `first_block_input` is `embed_tokens(token_array)` — **before** block 0's input
  layer norm. `EmbeddingTrainer` does not need it as a training argument.
- Capture ran on **MLX 0.32.2 / mlx-lm 0.31.3** with `use_kernel=not self.training`.
  A differentiable training path may use a different recurrent implementation, so
  parity must be measured on the port, not assumed from a version string.
- My earlier citation of `gsq-rco-mlx/docs/PARITY.md` has narrower scope than I
  implied: it covers **upstream's Gumbel quantizers** on CPU, not this solver's Q1
  candidates, full-vocabulary KL, or RCO. Sound basis for the noise design; not a
  validation of this pipeline.

## Blocker 2 (real, but bounded): no MPS path, and four specific gates

Source audit, with the code that must change:

| # | Site | What it does | Class |
|---|---|---|---|
| 1 | `qwen.py:28,70` | `from accelerate import init_empty_weights` — **twice** | BLOCKER. `accelerate` is absent from the venv; this is the *first* failure, before any device question. |
| 2 | `run.py:610` | `if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise` | BLOCKER |
| 3 | `run.py:612-613` | `compute_device = ... torch.device('cuda')`; `if compute_device.type != 'cuda': raise` | BLOCKER |
| 4 | `gsq_residency.py:31-32` | `if device.type not in ('cpu','cuda'): raise ValueError` | BLOCKER. `test_gsq_residency.py:85-87` *pins* MPS as rejected. |
| 5 | upstream `gumbel_quantizer_1bit.py:65,75-79` | `torch.cuda.get_rng_state` + `fork_rng(devices=[device])` | BLOCKER. `fork_rng` will not accept an MPS device list. |
| 6 | `rco.py:41` | float64 counts built on device | BLOCKER — **MPS has no float64** |
| 7 | `checkpoints.py:187-196` | RNG capture: python/numpy/torch_cpu/torch_cuda only | DEGRADES — exact resume unavailable on MPS |
| 8 | `performance.py:17-19` | synchronize only when `device.type=='cuda'` | DEGRADES — unsynchronized MPS timing |
| 9 | `smoke.py:36-37`, `gsq24.py:39-46` | require CUDA / a 20–25 GB GPU | BLOCKER for those stages only |

Also: `validate_restored_inputs` (`run.py:81-104`) hard-requires the cloud layout
`inputs/{model,calibration,restore-validation.json}`, and `input_commit_sha256`
is **inside the checkpoint identity** (`run.py:130-134`), so a local adapter must
produce a stable receipt frozen before the first stage. `corpus_inputs`
(`run.py:58-79`) is already local-compatible and needs no change.

**Correction to the handoff's stage order.** It says "initialization → embedding
→ GSQ blocks → head → RCO". The code is
`job.py:161`: `('initialize','smoke_gsq','smoke_rco','embedding','gsq','head','rco')`
— note `smoke` runs before `embedding`, and `head` runs *after* all 64 GSQ blocks.

**Correction to what `gsq24.py` is.** It is not a 24 GB implementation of GSQ; it
is a diagnostic harness that re-runs the *unchanged* `BlockTrainer` math under
three residency schedules on one 24 GB L4, to prove the schedules agree.

## What is *not* a blocker: MPS itself

This was the biggest unknown and it is resolved by measurement, not inference:

```
linear_attention block  (full width, 385.9M params): fwd+bwd OK, 1229 ms
full_attention  block   (full width, 374.9M params): fwd+bwd OK, 1008 ms
```

Both at production `hidden_size=5120`, on MPS, with `use_cache=False`. The
gated-delta and causal-conv paths fall back to reference PyTorch (no
`flash-linear-attention`, no `causal_conv1d`) — correct, and slow, but working.
`use_cache=True` fails with a cache-type mismatch on a single-layer
linear-attention model; the solver trains without cache, so this is not on the
critical path but should be handled if it surfaces.

---

## Tonight's plan

Ordered. Each step is independently verifiable, and steps 1–4 are worth having
even if the full run is impossible on this corpus.

### Step 0 — decide the corpus question (5 minutes, blocks everything)

Is `runs/mac-pipeline-v1-20260919/activations/` the intended training corpus, or a
validation sample whose full counterpart is in GCS?

- **Full corpus exists** → fetch it (985 GiB free), proceed to Step 3 knowing a
  real run is downstream.
- **2,784 rows is all there is** → the GSQ objective is not well-posed on it;
  either recollect full activations, or accept that tonight produces a *pipeline
  validation*, not a model.

### Step 1 — install `accelerate`, measure the real checkpoint size (20 min)

`accelerate` is the first hard blocker and its absence is a one-line install. Then
run `test_objective.py`, `test_performance.py`, `test_reproducibility.py` — the
three test files with no GPU dependency and no missing upstream path. These are
pure-math and should pass unchanged; if they do not, the port is broken before
anything else starts.

### Step 2 — port and validate the device layer (2–4 h)

One new module owning device policy, and the four gates edited to route through
it:

- `execution_device` accepts `mps`
- `run.main`'s CUDA-BF16 guard becomes a capability check
- the Gumbel quantizer's RNG: reuse the port in `gsq-rco-mlx/src/gsq_mlx/rng.py`
  (drawn on the global CPU stream, `manual_seed`-reproducible, bit-exact against
  upstream — verified there with a 5-bit-width parity harness). This is exactly
  the `fork_rng`/MPS problem, already solved and tested.
- RCO's fp64 counts: compute them in fp64 on **CPU** and move the result. MPS has
  no fp64; the counts are small and off the hot path, so this is cheap.
- `performance.synchronize` and `checkpoints.capture_rng_state` gain an MPS branch

Validation: `test_gsq_residency.py` extended so MPS is *accepted*, and a tiny-model
`gsq_run` end-to-end on MPS — the same shape as the existing CPU canary in
`test_gsq24.py`, which already runs the real `gsq_run` on a 4-layer tiny model.

### Step 3 — one real block, one real invocation (1–2 h)

Full production width, block 0 (linear-attention) and block 3 (full-attention) of
the actual model, with a real 4,673-token invocation. Verify loss, every required
gradient, optimizer state and candidate export. **Record actual peak memory** and
the per-update wall clock.

This is where the honest number comes from: the measured 1.0–1.2 s per
forward+backward at 32 tokens, scaled to production, tells you the real ETA rather
than the handoff's borrowed L4 figures.

### Step 4 — attempt `gsq` with a deadline, and stop cleanly

57 blocks × 87 invocations is a multi-day job on MPS at best. But
`should_stop` (`run.py:32-33`) is built for exactly this: deadline or
`--max-steps` returns `{'status':'checkpointed_stop'}` with a committed snapshot.
Set a real deadline, run as many blocks as complete, and leave a restartable
checkpoint plus a measured rate.

That is a legitimate overnight outcome under the handoff's own rules — it asks for
*"a human-readable status with a committed checkpoint and exact restart command,
even if the whole job does not finish tonight."*

---

## Measured ETA (replaces the handoff's borrowed L4 figures)

Measured on this machine, production width, one block, `teacher fwd + student
fwd + student bwd` — exactly what `BlockTrainer.forward` does per update:

| Block type | Params | Per update |
|---|---|---|
| `linear_attention` | 383 M | **1758 ms** |
| `full_attention` | 372 M | **864 ms** |

Corpus: 87 invocations, 104,557 tokens, median 625, max 4673.

One epoch over the full corpus, per block:

| | per block |
|---|---|
| `linear_attention` | 2.5 min |
| `full_attention` | 1.3 min |

All 64 blocks (48 linear + 16 full) × the configured epoch count:

| `num_epochs` | total |
|---|---|
| 1 | **2.4 h** |
| 5 | **11.9 h** |
| 10 | **23.7 h** |

**This changes the conclusion materially.** The handoff estimated "tens of hours to
several days" by extrapolating L4 numbers; the measurement puts a *single-epoch*
GSQ pass at **2.4 hours**, and a 10-epoch pass at about a day. A meaningful GSQ
stage is therefore achievable overnight, not merely started.

Not yet measured, and must be added before trusting the total:
- **cache propagation** — a full forward over all 87 invocations per block, once
  per block (64 extra passes; the earlier full-model measurement suggests this is
  not free)
- **checkpoint serialization** at the 120 s cadence, on a state whose observed
  cloud payload was 12–17 GB
- **head stage** — full-vocabulary KL over 248,320 rows, a separate cost

The remaining blocker is therefore **the corpus shape (Blocker 1), not compute.**

## What I would not do

- **Not** fake the `restore-validation.json` receipt. It is inside the frozen
  identity; a fabricated one produces checkpoints that cannot be resumed or
  migrated.
- **Not** substitute `gsq-rco-mlx` for the production solver. It is a different
  design (per-layer `gsq_bits` scalar grids vs this solver's Q1+BF16 candidate
  pairs and full-vocabulary RCO objective), and its own docs record that it
  cannot load this model. It is a *reference for what works on MPS* — the RNG
  port and the device-policy shape are directly reusable — not a drop-in.
- **Not** run the 52 GiB model on MPS "to see if it fits" without Step 2's device
  layer. It will fail in `load_original` on the `accelerate` import first anyway.
- **Not** change the objective to make it run. The handoff is explicit that
  sampled vocabulary or local projection MSE is not an equivalent RCO objective,
  and the same discipline applies to substituting a diagonal-moment
  approximation for `XᵀX`.

---

## The one-paragraph version

The MPS port is real work but tractable — four flagged gates, one of which
(the Gumbel RNG) I have already solved and tested in a sibling repo, measured
here at 1 s per block forward+backward at full width. What makes tonight
infeasible is not the port: it is that the calibration traces contain 32 sampled
rows per projection with only diagonal moments, and GSQ's objective needs `XᵀX`
plus a differentiable block forward. Answer the corpus question first. If the
full activations exist, tonight yields a working MPS port with a measured rate and
a restartable partial run. If they do not, tonight yields a validated pipeline and
the honest conclusion that this corpus cannot train this model.