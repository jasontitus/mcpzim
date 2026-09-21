# RCO on Apple Silicon: handoff

**Date**: 2026-09-20
**Project root**: `/Users/jasontitus/experiments/mcpzim`
**Solver**: `tools/calibration/solver/`
**Log of record**: `docs/QUANTIZATION_QUALITY_PIPELINE_2026-09-20.md`
**Related**: `HANDOFF_GSQ_MPS.md` (the GSQ half, which works)

Run everything from `tools/calibration` with `PYTHONPATH=.` and
`tools/calibration/.venv/bin/python`.

---

## 1. The task

The GSQ half of this quantizer is ported and working on Apple Silicon. **RCO is
not.** The problem is not that RCO is wrong — it is that the RCO stage **cannot
complete a single update on this machine**: PyTorch's MPS backend freezes
permanently inside Metal's ahead-of-time compiler. Nothing in this document is a
guess about RCO's mathematics; it is a hang in the Metal toolchain on one
specific Python graph, and the exact graph still needs to be isolated.

Your goal: **make the RCO stage run to completion on MPS.** If it cannot, say so
with the same evidence standard used below, and the fallback is a CUDA GPU for
this one stage (see §8).

## 2. Reproduce in one command

```sh
cd tools/calibration
rm -rf runs/native-mps/validate-new-objective/rco-hang-probe
PYTHONPATH=. .venv/bin/python -m solver.run \
  --config runs/native-mps/validate-new-objective/rco/frozen-config.json \
  --stage rco --max-steps 1
```

Expected (and what a working port produces): the run writes
`runs/native-mps/validate-new-objective/rco-hang-probe/allocation.json` and
prints a solver timing event for the update.

Actual: `model_load` completes (18–30 s, logged), `transformers` prints two
fallback notices, and then the process **freezes forever**. No exception, no
traceback, no exit.

## 3. The failure's exact signature

These four properties are what make this diagnosable. Reproduce them before
changing anything:

1. **~1 s of CPU total, frozen.** `ps -o time= -p <pid>` reads the same value
   over a 30 s window, after `model_load` has already been logged. It is not
   slow — it is stopped.
2. **It survives `faulthandler`.** `python -X faulthandler -c "import
   faulthandler; faulthandler.dump_traceback_later(180, exit=True);
   runpy.run_module('solver.run', ...)"` never fires. The watchdog thread cannot
   get the GIL, because the main thread is inside a C call.
3. **It survives SIGTERM.** `timeout 400 ...` does not end it. `kill -9` does.
4. **`sample <pid> 3` (macOS, no sudo needed) shows Metal's MLIR compiler:**

```
mlir::PassManager::run
mlir::RewritePatternSet::add<mlir::mps::ReductionVarianceOp>
mlir::RegisteredOperationName::Model<mlir::mps::ReductionVarianceOp>::getCanonicalizationPatterns
```

`mps.reduction_variance` is the `var`/`std` reduction. The model emits it
implicitly: `Qwen3_5RMSNorm` and its gated variant compute
`x.pow(2).mean(-1)`, and the config field is named `variance_epsilon`
(`modeling_qwen3_5.py:222-230`, and the gated form at `:848`).

`sample` cannot show Python frames (they live in the code object, not the C
stack), and `py-spy` needs root on macOS — `sudo -n true` fails on this machine,
so **`py-spy dump` was never obtained.** Getting it with a password is the single
highest-value first step, because it names the Python line directly.

## 4. Already exonerated — do not redo these

Each was tested directly on MPS, most in dedicated processes so a hang was
attributable. All passed.

| candidate | how it was tested | result |
|---|---|---|
| `torch.var/std/var_mean/std_mean`, `Tensor.var/std`, `F.layer_norm/batch_norm/group_norm/instance_norm/rms_norm`, `nn.LayerNorm/GroupNorm.forward` | all wrapped at runtime; each prints its caller before returning | **0 calls** before the freeze |
| `l2norm` (the FLA-aligned form at `modeling_qwen3_5.py:296`) | called directly, T=512/1337/2048/2049 | passes |
| the RMS reduction in four written forms: `pow(2).mean(-1)`, `(x*x).mean(-1)`, `(x*x).sum(-1)/n`, `linalg.vector_norm` | each at T=64/1337 | all four pass |
| `torch_chunk_gated_delta_rule` (the `fla` fallback, `modeling_qwen3_5.py:301`) | called with the model's real dims (48/48 heads, 128 dim), T=129/1337/2048, `chunk_size=64`, `use_qk_l2norm_in_kernel=True` | passes |
| a bare `x.std()` on MPS | trivial | passes (0.9888) |

The probe scripts are in `/tmp` (`rco_var_probe2.py`, `l2norm_probe.py`,
`rmsforms_probe.py`, `deltarule_probe.py`); re-create them from this description
if `/tmp` has been cleared. **The model is not broken at the layer level**: see
§5.

## 5. Why the layers are known-good

The config's schedule is `full_attention_interval=4`:

```
[linear, linear, linear, full, linear, linear, linear, full, ...]
```

So blocks 0, 1 and 2 — **the blocks this port has already trained
successfully** — are `linear_attention` layers. GSQ runs the gated delta rule on
MPS, including `l2norm`, and trains normally. Its logs contain the same two
`transformers` fallback notices. **The 64-block chain is not at risk from this
bug.**

## 6. The surviving hypothesis (inference, not yet proven)

Every component passes in isolation; GSQ trains one layer at a time; the capture
forwards the whole model under `no_grad`. RCO does neither — `RCOTrainer.step`
(`solver/rco.py:149`) runs:

```python
with torch.no_grad():
    reference = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
for _ in range(n_gumbel_samples):          # n_gumbel_samples = 4
    ...
    hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
    loss = full_kl(hidden[0,:-1], reference[0,:-1], self.model.lm_head.weight,
                   p, self.candidates['lm_head'], self.token_chunk, self.vocab_chunk)
    (loss/n_gumbel_samples).backward()
```

**The whole 64-layer model, forwarded with gradients enabled, four times per
step, in the same graph as that variance reduction.** That combination has never
run on this port. The hypothesis is that the MLIR pattern set explodes on the
*scale* of the grad-enabled graph, not on any single op.

This is testable and **should be tested before anything else**: build a *small*
grad-enabled Qwen3_5-architecture model (the solver has `tiny_model()` in
`solver/qwen.py`) and grow it layer by layer until it hangs. That would convert
this inference into a measurement and give a minimal reproducer — which is what
a PyTorch bug report needs anyway.

Also note `step`'s first statement:

```python
if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<2:
    raise ValueError('Each full invocation must contain >=2 tokens; no padding/truncation')
```

RCO deliberately feeds **raw, unpadded single invocations**, unlike GSQ's aligned
chunked inputs. Shape was tested for the individual ops (§4) but never for the
assembled model.

## 7. Not yet tried

- **`py-spy dump --pid <pid>` with sudo.** Names the Python line. If you have a
  password, do this first.
- **A minimal reproducer by growing `tiny_model()`** (§6). Highest-value
  experiment after the above.
- `PYTORCH_MPS_LOG_PROFILE_INFO=1` and `PYTORCH_MPS_TRACE_SIGNPOSTS=1` (the
  MPS backend reads both; neither was tried).
- `MTL_DEBUG_LAYER=1` (Metal's own validation layer).
- `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- A different torch build: this is **torch 2.8.0**; the MLIR MPS compiler has a
  history of reduction-op pattern explosions across versions.
- `PYTORCH_MPS_FAST_MATH`, `PYTORCH_MPS_LOW_WATERMARK_RATIO` (untried).
- Running `step` with `n_gumbel_samples=1` — note this is *not* expected to help,
  since the per-sample graph is identical and it is the compile that hangs, but
  it is one line and it halves the work if it does.

**Already tried and ineffective**: `PYTORCH_MPS_PREFER_METAL=0` (load completed
in 18.0 s, then froze at 00:00 CPU exactly as before).

**Already tried and closed off by design**: installing `flash-linear-attention`.
`transformers` gates that path on `is_torch_cuda_available() or
is_torch_xpu_available() or is_torch_mlu_available()`, which is false on Apple
Silicon by construction, and `fla`'s ops import `triton`, absent on macOS-arm.
Uninstall it if it is still present (`uv pip uninstall flash-linear-attention`);
it resolves but can never be selected.

## 8. If it cannot be fixed

The RCO stage needs the whole-model gradient; nothing else in the pipeline does.
Moving *only* RCO to a rented CUDA GPU is therefore the minimal deviation, and
this stage was already validated on CUDA. `solver/job.py` plus the existing GCS
machinery are the path. The chain, export and evaluation stay local.

## 9. Constraints a fix must not break

- **Upstream parity is load-bearing.** The port's standard is bit-identical
  where achievable — measured, not assumed: `solver/tests/test_block_parity.py`
  shows upstream-CPU vs port-CPU loss and student-input gradient at **0.0
  exactly**, cross-device loss 0.0, MPS 1 ulp. A fix that changes the RCO
  objective's arithmetic must demonstrate the same standard on the same axes.
- **Never load a second model while one is resident.** Two concurrent 52 GiB
  loads drove swap to 88.6 of 89 GiB and blocked *both* processes at ~0% CPU;
  the machine looks idle while it happens. Kill both and it recovers in ~20 s.
- **The store lock serialises publish/prune.** Do not add a second writer.

## 10. Environment as of this handoff

```
torch 2.8.0, MPS available; venv tools/calibration/.venv (uv-created, no pip module)
model: runs/local-inputs/model  (Qwen3_5ForConditionalGeneration, hidden 5120,
       16 key heads / 48 value heads, head dim 128, conv kernel 4,
       full_attention_interval 4, 64 layers)
config: runs/native-mps/validate-new-objective/rco/frozen-config.json
candidate db: runs/native-mps/validate-new-objective/stage-run-4/candidate-database.json
checkpoints: runs/native-mps/checkpoints/gsq/
546 GiB free on the volume
```

`causal_conv1d` and `flash-linear-attention` are **not** installed and, per §7,
cannot help.

## 11. What this branch contains, and what to regenerate

Branch **`gsq-rco-apple-silicon`** is the complete port. Verified by cloning it
clean and resolving upstream from the clone:

```sh
git clone --branch gsq-rco-apple-silicon <repo> /tmp/check
cd /tmp/check
python3 -c "
import importlib.util as u
s = u.spec_from_file_location('up','tools/calibration/solver/upstream_paths.py')
m = u.module_from_spec(s); s.loader.exec_module(m)
print(m.resolve())"
# -> .../tools/calibration/cuda_runtime/.context/sources
```

`resolve()` defaults to `verify=True`, so that call also hash-checks the sources
against the pinned manifest. If it returns a path under `/opt/upstream` instead,
the sources are missing and every stage will fail with a bare
`ModuleNotFoundError: No module named 'manifold'`.

**In the branch**: the solver and its tests; the calibration scripts; the
cloud-GPU and cloud-prep job machinery; `packing/`; the pinned upstream at
`cuda_runtime/.context/sources` (gsq 2.8 MB, rco 496 KB, plus the provenance
manifest) and the container build inputs; the stage configs (`gsq-run.json`,
`rco-config.json`, `initialize.json`, `smoke.json`, `gsq-single-process.json`,
`gsq-single-from-b005.json`); the per-block driver
`runs/native-mps/run-gsq-per-block.sh` and the reclaimer
`runs/native-mps/reclaim-superseded.sh` (which a 64-block sweep needs: see D8 -
each block writes ~15 GiB of store and up to 28 GiB of output, and without
reclaiming superseded roots the sweep stops on the driver's 60 GiB guard after
about five blocks); the gate wiring `runs/native-mps/gate-partial-export.sh`
(export -> control -> candidate -> record, for scoring an artifact against the
Bonsai control on identical held-out bytes); the RCO reproduction inputs named in
§2; the pinned Prism converter at `cuda_runtime/.context/prism`, checked in as
exactly the 2,973 files its source receipt covers so the receipt's size and
sha256 checks pass unchanged - `packing/export_qwen.py` inserts `<prism>/gguf-py`
and `<prism>` on `sys.path` and refuses to run unless `verify_prism_source`
passes, so without it no artifact can be exported at all;
and the documentation, including `docs/PORT_VALIDATION.md` (the port's log of
record) and the `QUANTIZATION_*` design documents.

**Not in the branch, and how to get it** — all of it is large or derived:

| missing | how to obtain |
|---|---|
| the model, 52 GB at `runs/local-inputs/model` | `tools/calibration/restore_inputs.py`, driven by the `input_commit_sha256` in the config |
| the corpus at `runs/local-inputs/calibration` | same restore; the directory is empty in the working tree |
| candidate archives and stage tars, ~6.2 GB under `runs/native-mps/validate-new-objective/` | regenerate with the chain, or restore from GCS |
| checkpoints and per-block outputs, ~536 GB under `runs/` | regenerate: `runs/native-mps/run-gsq-per-block.sh 64` |
| smoke warmstart states, 4.6 GB and 4.5 GB at `runs/native-mps/smoke/warmstart-block-{0,3}.pt` | regenerate via the smoke stage. **These are real inputs, not scratch** - deleting them killed a run once |
| per-iteration `runs/native-mps/pb-config-*.json` | the driver writes them itself |

`tools/calibration/.gitignore` excludes `runs/` deliberately. Anything under it
that is genuinely source rather than data (the driver, the stage configs) is
force-added, so add new ones the same way rather than assuming a `git add` will
take them.
