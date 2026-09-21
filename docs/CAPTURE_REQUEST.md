# Correction: no new capture is required

This file previously asked for a 64-layer block-input capture (~64 GiB). **That
request was wrong, and it is withdrawn.** The correct answer came from the session
that produced the original package, and I have verified it against the code.

## What I got wrong

I read `BlockTrainer.forward(student, teacher)` and concluded the solver needed
captured block inputs for every layer. It needs those *tensors* — but it builds
them itself:

```python
# run.py:321-335  cached_inputs
ids = torch.tensor([tokens], device=device)
reference = model.model.embed_tokens(ids)
student = gather_candidate(embedding_candidate, ids, model.model.embed_tokens.weight)
torch.save({'teacher': reference.cpu(), 'student': student.cpu()}, path)
```

`cached_inputs` constructs teacher **and** student streams from **token IDs and
weights**. Then `gsq_run` propagates them forward one block at a time:

```python
# run.py:411-412
teacher = run_block(model, block_index, tensors['teacher'].to(device))
student = trainer.hard_forward(tensors['student'].to(device))
```

So block *n* needs its *current* teacher/student inputs — not every later block's
inputs held simultaneously. Each block's output becomes the next block's input,
generated on the fly.

**The deeper reason this had to be wrong:** the student stream depends on the
*learned quantized prefix*. It cannot be captured ahead of time from the original
model at all. A one-time all-layer capture of original-model hidden states would
give the teacher stream and nothing else.

The existing CUDA run is the proof: it completed blocks 0 and 1 and durably stored
the block-2 cache using exactly this method.

## What is actually needed

| | |
|---|---|
| Token IDs + prompt provenance | **already captured**, and integrity-checked by `corpus_inputs` (`run.py:58-79`) |
| Original BF16 weights | **already local** (`runs/qwen3.8-27b-original/`, ~51.8 GiB) |
| Reference `lm_head` | **already local** — `model-00018-of-00018.safetensors`; `FullVocabularyKL` promotes row chunks to FP32 itself |
| Reference log-sum-exp | **computed by the objective**, not captured |
| Per-projection 32-row samples | diagnostics and a parity oracle — valuable, not training input |
| `sum_x` / `sum_x_squared` | exact over all rows; a first-moment and diagonal-Gram check, not off-diagonal |

**Nothing further needs to be captured before port work begins.**

## Corrections to specific things I asserted

- **"2,784 rows is all there is."** Wrong. The solver uses **104,557 token
  positions across 87 invocations**, including the full 4,673-token longest
  sequence. The 32-row figure counts only the diagnostic projection samples. My
  concern that the corpus was too thin was based on miscounting which artifact is
  the training input.
- **"All 64 layers must be captured."** Wrong, per the above.
- **"The 4,096-activation knee applies."** That number came from a *different*
  pipeline (per-layer `gsq_bits` scalar quantization). It has not been verified
  for this objective. Adequacy here should be measured on held-out Zimfo
  behaviour and coverage, not imported from a sibling project.
- **`first_block_input` semantics** — it is `body.embed_tokens(token_array)`,
  i.e. **before** block 0's input layer norm. `EmbeddingTrainer` does not require
  it as a training argument; it is a useful parity oracle.
- **Cache semantics** — capture calls the body with `cache=None` per invocation,
  and PyTorch block calls use `past_key_values=None, use_cache=False`. Both are
  fresh-state. The cache-type mismatch I hit on MPS is an **adapter** issue, not
  evidence the prompts need persistent cache objects.
- **Kernel provenance** — the capture ran on **MLX 0.32.2 / mlx-lm 0.31.3**, not
  PyTorch `flash-linear-attention` or `causal_conv1d`. MLX dispatches gated delta
  with `use_kernel=not self.training`, so inference capture may use a different
  recurrent implementation from a differentiable training path. Parity must be
  *measured* on the port, not inferred from a version string.

## What this changes about tonight

Nothing about the compute estimate (measured: 2.4 h per epoch over 64 blocks on
MPS) and nothing about the port plan. It removes the **data blocker** entirely:
the corpus, the weights and the head are all present, and the inputs GSQ trains on
are generated at run time.

The remaining work is the device port — `accelerate` removal, `execution_device`
admitting MPS, synchronized timing, checkpoint RNG, and the Gumbel quantizer's
CUDA RNG replay. Three of those are already landed or in flight.

## What I would still want verified, in the port's own tests

Not by capture, but by measurement on this machine:

1. **Teacher/student stream parity.** Regenerate successive teacher/student states
   locally for one invocation and compare against the captured `first_block_input`
   and `final_normalized_hidden`. This is what those two tensors are *for*.
2. **Both hybrid block types** at full production width, on the complete longest
   input — forward, gradients, optimizer state, candidate export.
3. **A full RCO loss/backward/update**, not just projection kernels.
4. **Cold native checkpoint recovery.**
5. **Numerical parity against the CPU/upstream path** for the ported Gumbel
   quantizer, with the exact revision, dtype, noise scheme, backward and
   scale-grid coverage stated — not a version string.

## Honest status of the parity claim I cited

I referenced `gsq-rco-mlx/docs/PARITY.md` when proposing the RNG port. Its scope is
narrower than "the port is validated": it covers **upstream's Gumbel quantizers**
(1/2/3/4-bit and ternary), on CPU, with upstream's CUDA RNG calls shimmed to CPU.
It does not cover this solver's Q1 candidate format, the full-vocabulary KL, or
RCO. It is a sound basis for reusing the noise design; it is not a validation of
this pipeline.
