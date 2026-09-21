# GSQ composition drift: deep review

**Date**: 2026-09-20/21
**Scope**: why the 1-bit GSQ chain's composed stream drifts away from the teacher, and what
to change before restarting training from scratch.

## 1. The measurement this explains

Per-block, from the sweep's own drift records (`runs/native-mps/drift-series-20260920.json`):

| block | 0 | 11 | 17 | 18 | 19 | 20 | 21 | 25 |
|---|---|---|---|---|---|---|---|---|
| norm ratio | 1.096 | 1.115 | 0.901 | 0.773 | 0.807 | 0.812 | 0.836 | 0.933 |
| cosine | 0.924 | 0.893 | 0.867 | 0.812 | 0.763 | 0.713 | 0.700 | 0.723 |

Magnitude holds near 0.9 throughout; **direction** decays to a plateau near 0.71. The
per-block final loss (recovered from the solver log, last sequence of each block) is
**0.000317 mean over blocks 12-17 and 0.001038 over blocks 18-25 - a 3.3x step up at block
18**, exactly where the cosine cliff sits. Every block believes it is fitting well while
the composition rotates.

End to end, an 18-block export scores **perplexity 50275** against an all-RTN floor of
**69784** and a **13.46** bar (Bonsai Q1_0, a 3.80 GB 1-bit model that matches bf16 at
13.397 on the same text). Training bought 1.39x.

## 2. Root cause: the objective cannot see drift

`solver/gsq.py:90-119`, `BlockTrainer.forward`:

```python
target = block(student, ...)      # unquantized block on the SAME drifted input
output = functional_call(block, quantized_weights, (student,), ...)
return (output.float() - target.float()).square().mean()
```

Both sides of the loss are evaluated on `student`, the drifted composed stream. The target
therefore **moves with the input**: a block is rewarded for reproducing the drift
faithfully, and its loss stays small however far the composed model walks from the
original. That is a *local fidelity* objective with no term that can oppose compounding
error - and it is what the measurements show, since the loss was small and falling
throughout the period in which cosine fell from 0.924 to 0.70.

This is not a port bug. **Upstream does the same thing.** `cuda_runtime/.context/sources/
gsq/src/models/base.py:444-453` computes `out_fp = forward_with_quantized(batch, None)` and
`out_q = forward_with_quantized(batch, quantized_weights)` on the same `batch`, and
`main.py:391-402` propagates the activation buffer through each *quantized* layer before
training the next, so at layer N the buffer holds the quantized prefix's outputs. Upstream
has **no** cross-block term: no anchor to the original activations, no end-to-end loss, no
regulariser, no rollback (`src/trainer.py:24`'s `self.min_loss` is dead code). It never
discusses compounding error anywhere in the tree, and offers no diagnostic that the
composition stays close to the teacher.

So drift is inherent to the method as published, and nothing in it mitigates drift.

## 3. The finding that matters most: 1 bit is outside the demonstrated regime

Upstream **never claims or reports 1-bit results**. `GumbelQuantizer1Bit` exists and is
reachable (`src/trainer.py:46-47`), but it is absent from the scheme table
(`README.md:76-82`), from the cheatsheet (`README.md:220`), from the converter's bit-width
note (`README.md:484`), and **no shipped config selects it** - every config uses 2, 3 or
`"ternary"`. The stray README lines that mention "1" are contradicted by all of those.

What upstream does report:

| arm | bits | result |
|---|---|---|
| Llama-3.1-8B | 2.13 bpp | 68.55 avg vs FP16 73.71 (**-5.2**) |
| Llama-3.1-70B | 2.13 bpp | 75.57 vs 78.99 (**-3.4**) |
| Llama-3.1-8B | 1.71 bpp (plot only) | ~63.5 (**-10**) |
| TL;DR (`README.md:14`) | - | "closes most of the accuracy gap ... at **2-3 bits**" |

Our target is Q1_0 at **~1.14 bpp** - **below every point upstream has ever published**, and
below the aggressive edge of its own Pareto curve by a wide margin, where the method is
already ~10 points down at 1.71 bpp. Expecting a 1-bit post-training quantization of this
port to match a natively 1-bit-trained model (Bonsai) is not supported by upstream's
evidence. The 50275-vs-13.46 gap is consistent with that, not evidence against the port.

## 4. Two concrete deviations from upstream, both fixable

Neither is the drift mechanism, but both are real and both plausibly contribute to how far
the chain falls short.

1. **Optimizer.** Upstream trains the quantizer logits with **Lion**, `betas=(0.9, 0.95)`,
   `lr1=0.0002`, `lr2=0.0001`, `weight_decay=1.0` (`src/trainer.py:70-97`, `configs/local/
   config.yaml:20-22`). The port uses **Adam** at `lr=0.001` for both. Lion's update is a
   sign-like step with `weight_decay=1.0` folded into it; the two do not behave alike.
2. **Steps per block.** Upstream: `num_epochs=10` over `num_samples=4096` at
   `batch_size=64` = **640 optimizer steps per layer** (`src/trainer.py:88-90`). The port:
   one pass over 87 invocations = **87 steps per block**, about 7x fewer. This also fixes a
   second, subtler deviation: the temperature/scale anneal is indexed by
   `fraction=(epoch*len(records)+index)/max(1,epochs*len(records)-1)`, so with one epoch the
   port annealed Gumbel temperature 2.0 -> 0.05 and logit scale 100 -> 500 over 87 steps
   instead of over the block's whole training run. Upstream anneals over
   `num_training_steps` (`src/trainer.py:136-137`). Both the number of steps and the
   horizon they are annealed across now match upstream.

Also noted while reading, not yet acted on: upstream quantizes attention with a GPTQ prior
(`src/prior/gptq.py`, 2000 epochs, Hessian-weighted Frobenius objective) and applies GSQ to
the MLP; the port's `gsq_bits`-equivalent is 1-bit for every eligible projection. And
`quantization.strength` is a *logit-initialisation* scale in the code, not a regulariser as
`README.md:227` labels it - so it is not a missing regulariser.

**A third mechanism, faithful to upstream but worth naming.** The block is *trained* on
**soft** Gumbel-sampled weights (`quantizer(temperature, scale)`, `solver/gsq.py`) while the
composition and the export both use the **hard** sign weights (`get_hard_weights()`), so the
drift this port measures is a property of the hard stream that training never directly
optimises. The temperature anneals 2.0 -> 0.05, which narrows the gap but does not
guarantee the hard and soft assignments agree at the end of a block; at 1 bit a wrong sign
is a large error rather than a small one. This is upstream's own design (the Gumbel-Softmax
relaxation), so it is not a deviation to fix blindly - but it is a plausible contributor,
and a straight-through hard forward is the obvious experiment if the validation shows the
drift persisting.

## 5. The fix set, and why each part is justified

1. **Keep upstream's objective shape but add the correction signal.** The change already
   made: input `student` (drifted, what the block sees at inference), target
   `block(teacher)` where `teacher` is the **clean** stream's input to this block. The
   teacher stream is genuinely unquantized - the cache writes
   `run_block(model, block_index, tensors['teacher'])` with the original weights - so the
   gradient now opposes drift instead of rewarding it. This is a **deliberate deviation
   from upstream**, and it should be measured, not assumed: upstream's own form has no
   mechanism that can correct drift, and we have a 25-block measurement of what that looks
   like.
2. **Adopt upstream's optimizer and step count** for the quantizer logits. This is the
   evidence-backed part: it is the only difference from a *demonstrated-working*
   configuration that we can see.
3. **Bound the direction in the guard** (`gsq_min_drift_cosine`, already added). The
   existing limits bound magnitude only, which is why a fifteen-block collapse in cosine
   tripped nothing.

## 6. What would falsify the plan

**The first criterion has already fired.** An A/B through the real code path on the tiny
model (identical model, seeds and records; three blocks; the fixed objective versus the old
target forced back by monkeypatching `BlockTrainer.forward`) compares the composed cosine
that `drift.json` records:

| block | fixed target (clean) | old target (same input) |
|---|---|---|
| 0 | 0.7462 | 0.7453 |
| 1 | 0.5952 | 0.5921 |
| 2 | 0.5820 | 0.5781 |

The correction signal moves drift in the right direction, consistently, but by ~0.4% - and
the drift **persists regardless**: cosine falls 0.746 -> 0.582 across three blocks with the
clean target in place, the same shape as the 25-block real run. The target mismatch is
therefore *not* the primary cause. What compounds is the **quantization error itself**: a
1-bit block cannot reproduce its input's transformation, its residual enters the next
block's input, and no choice of target inside a greedy per-block loop changes that. The
toy's drift is larger than the real run's at the same block index (random weights at 1 bit),
so the effect *size* does not transfer - but the *persistence* does, and that is the point.

- ~~If the validation run's cosine still decays with the correction signal in place, then
  drift is not merely an objective artefact and the target change is not the answer.~~ It
  decays. The target change is worth keeping - it is free and points the right way - but it
  is not the answer.
- If the loss under the clean target never falls - it cannot reach zero, since a drifted
  input cannot be perfectly corrected - that is *expected*, and the test is the **cosine
  trend**, not the loss level. Judging this fix by the loss would be the same mistake in
  reverse.
- **The question that actually decides this is the byte budget.** Upstream reports 2.13 bpp
  (-5.2 points on Llama-3.1-8B) and a 1.71 bpp plot point (-10). This port targets ~1.14
  bpp, below every published point, because the deliverable is a ~4 GB artifact for a 27B
  model - and a *narrower* budget leaves less room, not more: ternary at ~1.58 bpp is ~5.3 GB
  and 2.13 bpp is ~7.2 GB, both over the 4 GB target. If 4 GB is fixed and Bonsai is the
  bar, then the bar is being met by a model trained *natively* at 1 bit, and no
  post-training quantization of this kind reaches it. That is the finding to act on, and it
  is a change of method or of target, not another sweep.

## 7. Known issues left open, and what a production run needs

**A docstring I wrote repeats the misreading this review corrects.** `BlockTrainer.forward`
in `solver/gsq.py` says upstream "precomputes activations through the unquantized cascade
once and never updates them, so its blocks train on the clean stream with a clean target."
That is false, as section 2 shows: upstream propagates the buffer through each *quantized*
layer before training the next, so its blocks train on the drifted stream too. The *code* is
right; that sentence is wrong and must be corrected when `solver/*.py` is next editable - it
is frozen while a training run holds the checkpoint identity, because the identity hashes
that tree. The same misreading is what an earlier revision acted on when it moved the target
onto the student stream, so leaving it in place would invite the bug back. The parity test's
docstring carried the same error and is corrected.

**The warmstart artifacts are bound to the optimizer.**
`runs/native-mps/smoke/warmstart-block-{0,3}.pt` store an `optimizer` state dict beside the
quantizer state. That state was written by the old single-group Adam, so loading it against
the two-group Lion raises `ValueError: loaded state dict has a different number of parameter
groups` - which is how the first validation attempt died, at `run.py:534`, loudly and with
both stores intact. The quantizer half of those files is fine; only the moments are stale.

- The 3-block validation therefore runs with `warmstart_states: {}` and `warmstart_sha256:
  {}`. Since the moments cannot be used, starting without a warmstart is *equivalent* to
  starting from the RTN initialisation that every other block starts from - not a silent
  shortcut, but the same initial condition the file would supply with its optimizer half
  discarded. The original files are untouched on disk.
- **A production run must regenerate them** with the new optimizer - which is what the smoke
  stage is for (`--stage smoke`, which now uses Lion too) - or drop them as the validation
  does. Loading them unchanged is not possible.
