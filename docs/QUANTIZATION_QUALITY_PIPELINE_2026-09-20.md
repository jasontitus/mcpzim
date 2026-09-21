# Quantization for our use case: design, measurement, and the run plan

**Date**: 2026-09-20 · **Status**: design for review · **Supersedes**: the "just run 64 blocks"
plan, which would have spent ~12 h producing an unmeasured artifact under a non-GSQ objective.

## 1. Goal

Produce a **measurably good** low-bit quantization of Qwen3.8-27B for *this* product: an
offline assistant (MCPZim/Zimfo) that answers questions grounded in ZIM articles on-device
via llama.cpp/GGUF. "Good" means: **the app's own conversations still pass**, at a size and
speed the device can run. A number from the training loop is not evidence of that; the
suites in `eval/` are.

**The incumbent is Bonsai 27B Q1_0** (3.80 GB, `prism-ml/Bonsai-27B-gguf`). It is the
smallest model that currently works for us, and it is the thing to beat. The port's Q1
format is deliberately byte-compatible with it (`docs/QUANTIZATION_PACKING.md`: our all-Q1
layout lands +1,280 bytes on the published Bonsai), so **size parity is already achieved**
and the whole question is quality at that size.

**Target size is set by the phone, not by taste.** `DeviceProfile.modelFitsDevice`
(`ios/MCPZimChat/Common/DeviceProfile.swift:154-161`) admits at most
`min(physicalGB - 2.5, 5.5)` GB, and iOS caps a process at 6144 MB under
`increased-memory-limit` (`:92,:143`) - a cap the repo has already watched a 5.5 GB model
hit on device (the 2026-09-06 crash recorded at `:143`). The 7.2 GB ternary class is
therefore **not a deliverable**: `docs/BONSAI_27B_IPHONE_EVALUATION.md:23-24` states that
the 7.2 GB ternary build "does not fit the phone's practical app-memory budget", and the app
keeps that entry behind `#if os(macOS)` (`ios/MCPZimChat/Chat/ChatSession.swift:1829,1847`).

So the deliverable is the **3.8 GB Q1 class**, and any use of extra budget stays inside
~5.5 GB. That is exactly where RCO earns its place: the port's all-Q1 layout is
3,803,453,760 bytes, so a budget slightly above it - `make_run_configs.py`'s own default is
4,000,000,000 - describes a **mixed** allocation in which RCO decides which tensors keep 16
bits. Uniform Q1 is the floor of that range, not the whole of it.

Explicitly out of scope for now: matching the paper's accuracy at 2.13 bpp on
Llama-3.1-70B benchmarks. We are not chasing their number; we are measuring ours.

## 2. Measured current state (this machine, tonight)

### 2.0 The bar, measured (25 conversations / 66 turns, frozen protocol)

From `runs/mac-pipeline-v1-20260919/comparison-retry.json` (`status: compared_rubric_only`;
protocol `runs/comparison-v1-20260919/protocol.json`):

| arm | size | conversation-weighted pass rate | model-involved turns (17) |
|---|---:|---:|---:|
| Qwen3.8-27B bf16 (MLX) | 52 GiB | 0.7733 (51/66) | 0.7500 (13/17) |
| **Bonsai 27B Q1_0 (llama.cpp)** | **3.80 GB** | **0.7440 (48/66)** | **0.6458 (11/17)** |
| gap | 14x | +0.0293 | **+0.1042** |

**The same instrument, same text, three arms** (`llama-perplexity`, the held-out app text):

| arm | size | PPL |
|---|---:|---:|
| Qwen3.8-27B bf16 (F16 GGUF) | 54.6 GB | **13.397 ± 0.834** |
| Bonsai 27B Q1_0 | 3.80 GB | **7.065 ± 0.244** |
| our all-Q1 RTN control | 3.80 GB | **73,786 ± 2,141** |

Two conclusions, and the second is uncomfortable:

1. **Within our own family the instrument is decisive.** The base model scores 13.4 and our
   untrained 1-bit artifact scores 73,786 - a **5,507x** gap from the model we started from,
   measured on one binary, one text, one size class. That is the number the quantizer work
   has to close, and it is the only comparison of ours that is not confounded.
2. **Bonsai beats unquantized Qwen on this text** (7.07 against 13.40) by a margin no
   quantizer could manufacture, since quantization can only degrade a model's text
   prediction. The held-out text is the app's own rendered prompt and Bonsai is the model
   *adapted to that app*, so those prompts sit inside its training distribution and outside
   vanilla Qwen's. **This comparison is confounded by domain adaptation, not by
   quantization.**
   Consequence for the goal: "meet or beat Bonsai" **cannot be adjudicated by perplexity on
   app-shaped text**. The app suite is the instrument for that claim - where bf16 Qwen
   already edges Bonsai, 0.7733 to 0.7440 - which makes the suite's resolution problem
   (section 2.0.1) the thing to fix rather than a detail, and makes a never-scored
   confirmation set (D12) part of the deliverable rather than bookkeeping.

**The app suite is the instrument for the goal, and it has three properties this design has
to live with.** The perplexity numbers above rank *our* arms against each other, which is
what they are for; the use-case claim is adjudicated here.

1. **The ceiling is 0.77, not 1.0.** Unquantized Qwen fails roughly a quarter of these
   turns, so the metric saturates well below perfect, and "beat Bonsai" means crossing
   0.744 - a threshold only ~3 pp below where the 14x-larger model sits.
2. **The suite barely resolves the difference.** 21 of 25 conversations score *identically*
   for bf16 and Bonsai. The entire signal lives in 3 conversations bf16 wins
   (`country-facet-transfer`, `prepared_mongolia_religion_continuity`,
   `prepared_mongolia_topic_chat`) and 1 Bonsai wins
   (`prepared_mongolia_mongols_continuity`). A verdict built on 4 conversations flips on a
   single turn.
3. Therefore **a "better than Bonsai" claim is not supportable on the current suite** at
   any reasonable confidence. That is a property of the gate, not of the quantizer, and it
   is the first thing the build list has to fix.

**A rule for quoting any of it.** Adding a third arm must not move the pair's numbers:
`model_involved_app_turns` is a paired *union*, so a candidate that invokes a turn neither
other arm did would enter the reference-minus-bonsai denominator. Measured before the fix:
three candidate-only invocations moved bonsai 0.6458 -> 0.6667 and the delta +0.1042 ->
+0.0833. The pair row now unions over reference and bonsai only, the candidate has its own
`model_involved_app_turns_with_candidate` row, and a test asserts pair stability by running
before and after the mutation. `all_app_turns` is unaffected by a candidate and comparable
across protocols; the pair's model-involved numbers are not, unless the candidate provably
invokes no turn outside the pair's union.

The discriminating conversations share a shape: **facet transfer and cross-turn
continuity** - the same entity queried along a different facet, and facts carried across
turns - while single-turn factoid retrieval survives 14x compression intact. That is the
axis to grow the suite along, and it is also a testable hypothesis: our quantizer may
already hold those conversations and lose somewhere the suite does not look.

This is why the design carries a *continuous* metric (held-out reconstruction, D3) and a
depth curve (D4) next to the pass/fail suites: 66 binary turns cannot resolve a 3 pp
difference, but a reconstruction loss over 8 held-out invocations can rank arms cheaply and
reliably.

#### 2.0.1 Can this gate prove we beat Bonsai? No — it is a tripwire, not an instrument.

The paired analysis in `comparison-retry.json` is worse than the aggregate rates suggest.
Of the 66 turns in the frozen gate, **49 invoke no language model in either arm** and 33
belong to conversations that never invoke one; only 17 turns exercise the model at all.
The 0.7733-vs-0.7440 gap therefore mostly measures the deterministic answer presenter and
the retriever, which are identical code in both arms.

The entire answer-content signal the gate has ever produced fits in one table:

| conversation | turn | bf16 | Bonsai |
|---|---|---|---|
| `prepared_mongolia_topic_chat` | t3 | ok | missing grounded section `Modern history` |
| `prepared_mongolia_religion_continuity` | t3 | ok | missing section + grounded source `Mongolia` |
| `prepared_mongolia_mongols_continuity` | t3 | missing section | ok |
| `country-facet-transfer` | t4, t5 | ok | missing `2004` |

Four distinct failures, one of which is a single wrong year (`2004`) scored twice by a
duplicated "What year?" turn — and on those two turns the model was invoked in the Bonsai
arm only, so even that comparison is not like-for-like.

Paired per-turn outcomes, same 66 turns:

| | Bonsai pass | Bonsai fail |
|---|---:|---:|
| **bf16 pass** | 47 | 4 |
| **bf16 fail** | 1 | 14 |

Five discordant turns. Exact McNemar, two-sided: **p = 0.375**. The 52 GiB model and the
3.8 GB model are **not statistically distinguishable on our own suite**, and 61 of 66
turns carry no information about compression at all.

The minimum decisive result, computed exactly (paired wins / losses, alpha = 0.05):

| result | p | decisive |
|---|---:|---|
| 4 / 1 (bf16 vs Bonsai today) | 0.375 | no |
| 5 / 0 | 0.063 | no |
| **6 / 0** | **0.031** | **yes** |
| 7 / 1 | 0.070 | no |
| 8 / 1 | 0.039 | yes |

At Bonsai's observed 1.5% reversal rate, a model with bf16's observed 6.1% excess needs
**~200 turns** to reach p < 0.05 (150 turns gives p = 0.065). The suite is 66 turns.

**Consequences for the plan.**

1. **The app suite stays, as a non-regression tripwire, and is not the decision
   instrument.** It is the use-case measure and it must not regress, but a claim of the
   form "our Q1 beats Bonsai" needs either a decisive paired sweep (>= 8 wins against
   <= 1 loss) or a suite roughly 3x this size with far more model-invoked turns.
2. **The signal lives in retrieval-grounded, multi-turn turns.** Single-turn factoid
   turns tie across 14x compression; the failures above are all section/source grounding
   and cross-turn continuity. Growing the suite uniformly adds power to nothing.
3. **A continuous metric carries the ranking decision** (D3 in-loop, D10 held-out
   perplexity), with the suite as confirmation. This subsection is the quantitative
   justification for D10, not an argument for skipping the suites.
4. **A candidate role must be like-for-like.** Comparing our GGUF (llama.cpp) against the
   MLX bf16 reference confounds quantization with runtime, template and context policy.
   The candidate arm runs against Bonsai in the same runtime, with the same template, KV
   and context settings.

### 2.1 The objective is not GSQ's

Upstream (`cuda_runtime/.context/sources/gsq/main.py:204-218`, `src/trainer.py:300-347`)
captures each layer's calibration inputs by running the **original** model forward, with
every already-quantized layer parked on `meta` (`model.offload_to_meta(current_layer)`), and
then **reloads each trained layer's quantized weights and re-propagates the activations
through them** (`main.py:385-402`: "Reloading quantized weights from disk", then
"Propagating train/val activations through quantized layer"). The stream the next layer is
trained on is therefore **already composed of quantized layers** — an earlier revision of
this section claimed the opposite and was wrong. Its objective is the paper's layer-wise
form, **the same input on both sides**:

$$\hat w = \arg\min_{\tilde w}\ \lVert f(x;\tilde w) - f(x;w)\rVert_F^2$$

Our port instead trains `functional_call(block, quant_weights, (student_in,))` against
`block(teacher_in)`: **different inputs on the two sides**, and the quantized block is never
evaluated on the input it will actually see. Consequences, measured on tonight's
single-process run (`runs/native-mps/single-process.log`, blocks 0-7 complete):

| block | loss at first step | loss at last step | reduced |
|---|---|---|---|
| 0 | 0.1 | 0.04 | 0.67 |
| 1 | 0.2 | 0.14 | 0.71 |
| 2 | 26.3 | 7.2 | 0.27 |
| 3 | 495.7 | 159.9 | 0.32 |
| 4 | 3 757 | 2 874 | 0.77 |
| 5 | 16 229 | 7 834 | 0.48 |
| 6 | **449 790** | 398 295 | 0.89 |
| 7 | 88 683 | 67 782 | 0.76 |

The starting loss inflates ~4-10x per block because the student stream compounds
(student/teacher max ratio 1.49 -> 4.15 -> 19.20 over caches 1-3, measured from the
archived caches). Once the two sides see different inputs, the gradient is dominated by
**compensating the input mismatch**, not by reducing quantization error. The training
signal degrades exactly as the model gets worse.

### 2.2 Cost per block (measured)

```
87 steps x 5.22 s/step   = 465 s   training
cache propagation        =  76 s   exists only because of the student stream
publishing (2 forced)    = ~100 s
                           ~11 min per block, one process, ~8.3 GB store per block
```

### 2.3 What already exists (do not rebuild)

- **App-level quality gate**: `MCPZimEvalCLI --probe-discuss --gguf <m.gguf> --suite
  <eval/*.json> --report-json <out>` runs the real retrieval+grounding+generation path and
  reports per-turn latency, content failures, grounding sources and footprint (`eval/README.md`).
  Suites are written as anchors + `must_not_contain` / `must_not_suggest` gates.
- **Candidate -> GGUF export**: `packing/export_qwen.py --plan <plan.json>` with per-tensor
  `mode` entries. Every one of the 498 eligible 2D tensors must carry an explicit choice
  (`export_qwen.py:139-140` rejects any other set), so a partially quantized model is
  expressible only by listing all of them: `bf16` for the untouched ones, which preserves
  the original bits exactly. A plan generator is all that is missing, and it now exists
  (`solver/tools/export_plan.py`, with `--layers N` and a pinned-cost preflight).
- **Checkpoint store, prune** (`solver/tools/prune_store.py`), single-process mode (proven
  tonight: blocks 0-7 in one process, no `full_attention` failure).
- **Calibration inputs**: `runs/local-inputs/{model,calibration}`, 87 invocations,
  104 557 tokens.

### 2.4 What is missing

| capability | state |
|---|---|
| gate that can resolve bf16 (0.7733) from Bonsai (0.7440) | **MISSING** - 21/25 conversations tie |
| correct objective (same input both sides) | **MISSING** - solver change |
| held-out split so the metric is not on training data | **MISSING** - solver change |
| per-block quality number recorded in the report | **MISSING** |
| a candidate role in the two-role comparison protocol | **MISSING** - `prepare_comparison.py` freezes exactly `reference` + `bonsai` |
| export plan generator (candidate DB -> plan, with a layer cutoff) | **DONE** - `solver/tools/export_plan.py` |
| gate driver (export -> eval CLI -> parse -> record) | **MISSING** - wiring |
| budget chosen from a measured quality/cost curve | **MISSING** - this is the point |
| logit-scale (kappa) anneal in the training call | **MISSING** - one argument |
| GPTQ initialisation | MISSING - larger, separate |
| memory/storage plan for a long run | designed below |

### 2.5 The comparison pipeline: what is ready, and two traps

**Ready**: the frozen v1 protocol's evaluator binary is present and matches its pinned
sha256; the wikipedia ZIM has been moved out of the TCC-protected `~/Downloads` into
`zims/` and its **full sha256 verifies against the frozen pin** (52,690,706,555 bytes,
441a56d9e05b2d98…); the third `candidate` role exists, is like-for-like-enforced (same
runtime, per-turn chat template, effective sampler excluding the inactive topK, and
report context/KV policy), and leaves the two-role path byte-identical.

**Not ready**: the OSM streetzim the protocol pins
(`osm-california-2026-05-29.zim`, 3,495,655,515 bytes, sha256 6c210c3bfa18df36…) is not on
disk. Four of the 25 conversations declare `requires_streetzim`, so substituting another
region changes retrieval for those turns — which means re-capturing the reference and
bonsai arms and issuing a v2 protocol, i.e. moving the baseline rather than extending it.

**Trap 1 - the evaluator keys context/KV off the filename.** `ProbeE2ECLI.swift:733` derives
`isBonsai` from the lowercased GGUF path and `:818-819` sets 16384 / Q4_0 only on that
branch, so a candidate named "qwen…" captures 32768 / Q8_0 and `verify_like_for_like`
rejects it **after** a 3.8 GB capture. The freeze step now refuses such a path up front;
the real fix is an explicit n_ctx/KV switch in the evaluator, which changes the pinned
binary and therefore needs the v2 protocol anyway.

**Trap 2 - a rejected comparison overwrites its own output.** `compare_app_reports.py`
writes `{"status":"failed"}` to the output path on validation error, so a healthy result at
that path is destroyed by a later bad run. This already happened tonight: a failed run
overwrote `runs/mac-pipeline-v1-20260919/comparison.json` while `comparison-retry.json`
survived only because a later run used a different name. Write comparison outputs to fresh
paths, and treat a `status: failed` file as a tombstone rather than a result.

## 3. Design

**D0. Format axis: 1 bit, ternary, 2 bit.** The format is a variable now, not a fixed
target, because the goal is "small and working" and Q1 is not the only point on that axis.

- The codec is standalone (`packing/q1.py`; GGUF type 41 = 128-group, 18 bytes/group) and
  the exporter consumes whatever the solver emits, so a ternary or 4-level candidate needs
  a *codec* and a *runtime*, not a pipeline rewrite.
- Runtime support decides what can be scored at all. llama.cpp already loads Q1_0 and
  ternary Q2_0 GGUF files - Bonsai ships both, and `Ternary-Bonsai-27B-Q2_0.gguf` (7.16 GB)
  is on this machine. Our own type-41 Q1 is a Prism format whose in-app path is the shipped
  XCFramework. That path is better evidenced than this section first claimed and less
  evidenced than the goal needs: `packing/evidence/full-qwen-layout-proof.json` records the
  shipped XCFramework loading the **real** 851-tensor export and completing a 3-token
  prefill with 248,320 finite logits - but with `gpu_offload: false`, i.e. CPU only, because
  `runtime_smoke.c` pins CPU devices and disables KV/op offload by construction. So:
  full-model container and load are verified on CPU; the phone's **Metal offload path** and
  any **BF16-bearing mixed allocation** are unverified, and the port has **no ternary or
  2-bit codec at all** (`packing/gguf.py:13` carries only F32/F16/BF16/type 41, and
  `export_qwen` accepts only q1/f16/bf16/f32 modes).
- Upstream carries all three quantizers (`gumbel_quantizer_1bit.py`, `_2bit.py`,
  `_ternary.py`; `src/prior/quant.py` has ternary and an MSE-driven scale search), so the
  training side for ternary is ported, not invented.
- Ordering consequence: the first credible deliverable stays a **Q1** artifact, because
  size parity is proven and codec plus exporter exist. Ternary/Q2 opens as a second axis
  once the Q1 arm is measured, since it trades 3.4 GB of size for a quality and
  runtime-support question that D4's curve answers cheaply.

**D1. Objective: reconstruction on the original block, evaluated on the input the block
will actually see.** Train each block with the **student (propagated) stream as both input
and target**: the target is the *unquantized* block applied to that same input, so the
optimizer sees exactly the distribution the block meets at inference.

This is verified against the pinned source rather than inferred. Upstream's
`calculate_mse` (`src/models/base.py:444-453`) runs *both* forwards on the **same `batch`**
- `out_fp = self.forward_with_quantized(batch, None, ...)` with the original weights and
`out_q = self.forward_with_quantized(batch, quantized_weights, ...)` with the quantized
ones - and scores them with `self.loss_fn = torch.nn.MSELoss()`, a **mean** over the block
output. Same input on both sides, mean squared error, block output rather than logits. The
port's earlier target came from the *teacher* stream instead, which is a different input;
in code the change is one expression at the existing call site (`solver/gsq.py:75`):

```python
target = block(student, **block_kwargs(model, student, block_index))   # was block(teacher, ...)
```

Three things this changes from the current port, in order of weight:

- **The objective is the paper's.** Today the target comes from the teacher stream, so the
  block is trained to map a stream it will never be given. That is off-distribution in the
  direction that matters, and it is also what lets the loss shrink while the composed model
  degrades.
- **The propagation stays.** An earlier revision of this section proposed deleting it; that
  was wrong and would have made the objective *more* off-distribution, not less. Upstream
  reloads each trained layer's quantized weights and re-propagates (`main.py:385-402`), and
  that composed stream is the point of the method. The ~76 s/block cost is the price of
  training on the real distribution.
- **The teacher stream is kept, but as a measurement rather than a target.** It is already
  in the cache next to the student stream, and it is what makes D3.1 free.

**D3.1 Composed-stream drift: the per-block guard this design was missing.** Record, per
block, the **norm ratio and cosine of student against teacher at the next block's input**
(both tensors are already in the cache: `cache-<k+1>/<ordinal>.pt` holds `student` and
`teacher`), and **stop the chain at a block boundary** when the ratio grows past a stated
factor from the previous block.

Measured at the first block boundary available (`gsq-anchor0/cache-1`, 87
invocations): **ratio 2.94, cosine 0.934** - one quantized block already inflates the
stream ~3x, and `docs/PORT_VALIDATION.md` independently names the student stream's scale as
the lever and the 4x-per-block amplification as the thing to explain.

The discarded run's own log shows the same thing end to end: block 4 trains from a loss of
3,757 and block 8 from 60,800 - the incoming loss **roughly doubles every block** - and its
cache re-propagation costs 75.3 s per block over 87 sequences (104,557 tokens). Compounded
over the blocks that follow, that is the 93x / cos 0.56 the design review measured on the
last three trained blocks, and it is why those blocks are no better than the RTN-only arm
the repo already exported.

This metric costs one reduction over tensors that are already loaded, and it is the only
in-loop signal that can see the failure before block 64. A block boundary is the stop
point because a half-propagated chain cannot be resumed.

**D2. Held-out split — implemented, and it was missing until now.** Eight of the 87
invocations (`invocation-000000`, `-000011`, ..., stride 11) are reserved from training so no
score is measured on text the model was calibrated on. `corpus_inputs` used to admit **every**
sequence, while `quality_gate.py`'s docstring and this document both claimed the split
existed: the PPL bar and every arm were measured on eight invocations whose prompts were
still in the training corpus, so a *trained* arm's number would have been optimistic while
Bonsai's - trained outside this pipeline - was not. Now `run.is_held_out_invocation` /
`held_out_invocations` implement it, `main` passes
`config.get('held_out_stride', HELD_OUT_STRIDE)`, and `solver/tests/test_run.py` covers the
selection.

**What this does not invalidate, and why:** the two numbers in section 2.0 stand. RTN
requires no training, so the control export was never fitted to that text, and the bar is an
external model. The risk was entirely in comparing a *trained* arm against them - which is
exactly what the fix removes.

**D3. Per-block metric (in-loop, cheap).** After each block's training, with **hard**
weights and no noise, compute the reconstruction loss on the held-out slice and record it
in the block report alongside the training loss. Cost: 8 forward passes (~40 s). This is
the per-block gate: a block whose held-out loss is worse than the *incoming* RTN
candidate's is a regression and must be visible immediately, not at block 64.

**D4. Progressive artifact quality.** Every N blocks, emit an export plan that quantizes
layers `0..k` and leaves the rest f32, export with `packing.export_qwen.py`, and run the
eval suites against it. This gives a **quality-vs-depth curve** - the first end-to-end
signal this pipeline has ever had - and it is upstream's own `--max-layers N` idea. Cost
per point: export (CPU, minutes) + the eval CLI on the chosen cases.

**D5. Budget from measurement, not assumption.** Measured: 1 epoch = 87 updates = 465 s of
training per block; 64 blocks is therefore ~11-12 h at one epoch, ~19 h at two, ~32 h at
four. Matching upstream (10 epochs x 4096 samples) is ~weeks on one Mac and is not a
candidate. So: run the D4 curve at 1, 2 and 4 epochs over a **shallow subset (e.g. the
first 8 layers)**, pick the knee, then spend the long run there. The deliverable of the
sweep is a *decision*, not a model.

**D6. Schedules and initialisation noise.** Pass the logit scale (kappa) into the quantizer
call - it exists (`gumbel_mps.py`'s `forward(temperature, scale)`) but the trainer passes
only temperature, so kappa is pinned at 1.0 while upstream anneals 100 -> 500. Also compare
our temperature range (1.0 -> 0.1) against upstream's (2.0 -> 0.05) on the same
shallow-subset curve.

More first-order than kappa, and a one-line change at three call sites: our initialisation
noise is `std=0.2, strength=1` where upstream starts at `std=0.01, strength=6`, which flips
roughly **16% of the starting signs off the RTN initialisation** the block starts from. That
is not a schedule detail; it decides what the optimizer is even searching around.

Cadence is the same story one level up, and the review measured it as the **root cause**:
we take **87 updates per block**, where upstream takes **640**. At 87, bf16 saturation
zeroes the sign gradient after roughly step 30: measured zero-gradient fractions of
36%, 67% and 98% across the block, with a hard loss of 6.700e-2 at 87 steps against
6.138e-2 at 870. The two deviations that look like gratuitous infidelity are in fact
**compensations for that truncated budget** - one Adam learning rate of 1e-3 for both logits
and scales where upstream uses Lion at 1e-4/5e-5 with weight decay, and the initialisation's
sign-flip rate (10.8% against upstream's 1.8%) - so they should be revisited *together* with
the update budget, not one at a time.

Two corrections to earlier revisions of this section: upstream anneals **per optimizer
step**, not per epoch (`src/trainer.py:136-137`), which is what the port now does; and
blocks 0 and 3 load the smoke **warmstarts**, whose `load_state_dict` **overwrites** the
initialisation entirely - so any config that passes `warmstart_states` ignores the corrected
init for exactly those two blocks.

**D7. Initialisation.** RTN today; upstream's default is GPTQ (Hessian-based). **Second-order
until D6's noise and kappa are settled** - there is no point measuring a better initialiser
on top of a 16%-of-signs perturbation. It is the largest remaining lever after D6, and D5's
curve should tell us whether it is worth building at all.

**D8. Storage and memory.** Every number here is measured on this machine, and two of
them overturn an earlier draft of this section.

- **Disk was the binding constraint, and it has been reclaimed.** `df` reported **146 GiB
  available, 99% capacity** - not the 170-240 GiB this section first claimed. The store
  root held **534 GB across 21 attempt roots**, every one of them from today's porting
  effort (the solver edits changed the chain identity, so none of those chains is
  resumable in any case). Deleting the attempt stores - leaving outputs, calibration
  inputs, warmstart states and the RTN control artifact in place - returned the volume to
  **681 GiB free**. Measured growth in the live single-process store was
  (89.35 GB objects - 12.92 GB digest-shared)/8 blocks = **9.5 GB per block**, so the
  unpruned 550-600 GB figure was arithmetically sound and unsatisfiable at the old
  headroom: the store alone would have filled the disk near block 15.
- **Pruning is therefore mandatory, and it had a race that could end the run.** `publish`
  installs objects and only then writes the commit manifest - the order crash-safety
  requires - so in that window they are referenced by no commit, and a concurrent `prune`
  deletes everything no *retained* commit references. It would delete those objects
  (making the newest checkpoint permanently unrestorable) or unlink the in-flight
  `.upload-` temp that lives inside `objects/` and abort a 10-30 h run mid-block.
  **Fixed and tested**: `checkpoints.exclusive_store_lock` is held across publish's
  whole window and taken by `prune` for its read-then-delete, and prune no longer treats
  dotfile temporaries as garbage (`tests/test_checkpoints.py`).
- **Reclaim before the long run.** The abandoned roots must be deleted first, and the
  *current* attempt's store is 100+ GB by itself. Prune `--keep 3` during the run, and
  never let a skipped prune compound: one lapse is now a full disk, not a slow one.
- **Host RAM, not MPS allocation, is the resource that decides whether this machine
  survives.** The `0.71 GiB peak` this section first quoted is
  `torch.mps.current_allocated_memory()` sampled *inside* a block - not a peak, and the
  same mislabeling `docs/PORT_VALIDATION.md` already records as a high-severity
  provenance defect. `block_cpu_offload` requires the entire ~52 GiB baseline resident in
  host RAM (`gsq_residency.py` asserts every tensor is on CPU), the only guard
  (`require_baseline_fits`) returns immediately for that mode, and `total_memory_bytes`
  cannot decide at all on MPS. Measured now: 128 GiB installed, swap
  **43,952 of 45,056 MiB used (97.6%)**, 14.8 GiB in the compressor, 18.3 GiB pages free.
  The configuration already recorded `memory_current_allocated_bytes: 53.8 GB` with
  `memory_peak_available: false`.
- **Consequence for D4**: every curve point needs a *second* full model in memory
  (export plus the MLX reference arm), so curve points run only while the trainer is
  **stopped**. They are not concurrent with a block.
- **Never** materialise a whole 52 GiB model on the accelerator. `full` residency is the
  configuration that silently killed this machine before.
- Checkpoint cadence stays at 1200 s: mid-block publishes are throttled, so a block
  publishes twice, which is what bounds both disk and the loss on a crash.

**D9. The current run.** Discard its stores when it stops at 14:22 (its objective is
wrong). Keep the calibration inputs, which nothing in the run touches. Its lasting value
is the architecture proof: 8+ blocks in a single process with `full_attention` included.

**D10. Held-out perplexity: the instrument the ranking actually needs.** Section 2.0.1
shows the app suite cannot resolve a 3 pp difference on 66 turns, 49 of which invoke no
model at all. Upstream's own quality metric is WikiText2 perplexity (baseline plus every
six layers), and the equivalent for us is already installed:

```sh
/opt/homebrew/bin/llama-perplexity -m <model.gguf> -f <heldout.txt>   # Bonsai, ours, ternary
```

- **Text**: the held-out invocations' rendered prompts (`runs/local-inputs/calibration/
  invocation-*/input.json`, key `prompt`) - real app text with retrieved article prose,
  excluded from training by D2, identical bytes for every arm.
- **Why it works where the suite does not**: perplexity is continuous over every token, so
  it separates arms that tie on binary turns, and it needs no ZIM archives, no app
  runtime and no rubric.
- **The bar is measurable today**: Bonsai is a llama.cpp GGUF already on this machine, so
  its perplexity on exactly this text is one command away - before any artifact of ours
  exists. That number, not 0.7440, is what a Q1 candidate has to beat, with the bf16
  reference arm giving the ceiling via `mlx_lm`.
- **Cost**: minutes per arm on CPU/Metal, against 11 minutes per block of training. This
  is the cheapest high-resolution signal available and it should gate every D4 curve
  point alongside the tripwire suites.
- **The unquantized ceiling needs a different route, and it is being measured.** The bar and
  every GGUF arm are scored by one binary on one text, which is the comparison that matters.
  The bf16 ceiling cannot use that route cheaply: it requires converting the original model
  to F16, which produces a **54.6 GB** GGUF (~2 minutes to convert), and a 54.6 GB working
  set may exceed this machine's GPU working-set limit. Whether `llama-perplexity` evaluates
  it at a usable speed is **under measurement** - an earlier attempt was killed before it
  finished, by mistake, so nothing is claimed here either way yet. The bf16 reference from
  the app suite stands regardless (bf16 0.7733 against Bonsai 0.7440, section 2.0), and the
  `mlx_lm` number measured on this same text (9.61) is **not comparable** to the GGUF
  numbers: a different tokenizer and chunker put it *below* the 3.8 GB incumbent, which is
  the clearest available demonstration that cross-instrument perplexity comparisons are
  meaningless. Numbers from different instruments must never share a table.
- **Measure arms one at a time.** Two concurrent loads (and the long-running local
  `llama-server` instances: a ternary Bonsai on 8080 and a bge-m3 embedder) leave the
  loads blocked at ~0% CPU rather than failing, which is how a 60 s measurement becomes
  a 25 minute hang. The machine also carries 128 GiB with swap ~98% used, so the
  host-RAM budget in D8 is a shared budget, not this pipeline's alone.
- **One model resident at a time, measured the hard way.** Two concurrent 52 GiB loads -
  an RCO pricing run started while another was already loading - drove swap to
  **88.6 of 89 GiB** and blocked *both* processes: the survivor sat at 647 MB RSS with the
  model swapped out and 1% CPU while the machine showed as idle. Killing both returned
  swap to 62 GiB within 20 s and free memory to 94%. The failure is silent, symmetric
  (both lose) and looks like a hang rather than an error, so the rule is operational, not
  advisory: never start a second stage that loads the model until the first has exited.

**The RCO stage hangs on the one graph this port cannot compile.** The RCO update freezes
after `model_load` completes, consuming ~1 s of CPU total; the freeze outlives both
`faulthandler`'s watchdog and SIGTERM and dies only to SIGKILL. `sample` on the hung
process shows the main thread inside Metal's ahead-of-time compiler:

```
mlir::PassManager::run
mlir::RewritePatternSet::add<mlir::mps::ReductionVarianceOp>
mlir::RegisteredOperationName::Model<mlir::mps::ReductionVarianceOp>::getCanonicalizationPatterns
```

`mps.reduction_variance` is the `var`/`std` reduction, emitted for the model's RMS norms
(`Qwen3_5RMSNorm` and the gated variant both compute `x.pow(2).mean(-1)`, and the config
names the epsilon `variance_epsilon`). Every *component* of the path was then tested on MPS
in isolation, and each was exonerated:

| tested | result |
|---|---|
| `torch.var/std/var_mean/std_mean`, tensor methods, `layer_norm`/`batch_norm`/`group_norm`/`rms_norm` | wrapped at runtime: **0 calls** before the freeze |
| `l2norm` (FLA-aligned, `rsqrt((x*x).sum(-1)+eps)`) | passes at T=512/1337/2048/2049 |
| RMS reduction, four written forms (`pow(2).mean`, `(x*x).mean`, `sum/n`, `vector_norm`) | passes at T=64/1337, all four |
| `torch_chunk_gated_delta_rule` (the `fla` fallback actually used) | passes at T=129/1337/2048 |
| GSQ block training | trains normally, **including linear-attention blocks** |

That last row is the decisive one: the config's schedule is `full_attention_interval=4`
(`[linear, linear, linear, full, ...]`), so **blocks 0-2 - the blocks this port has already
trained - are the linear-attention layers**. The layers are not broken, and the chain is
not at risk from this.

What remains is the combination nothing here has ever run: **the whole 64-layer model
forwarded with gradients enabled**, four times per step (`n_gumbel_samples=4`), in the same
graph as that variance reduction. GSQ trains **one layer at a time**; the capture forwards
the whole model under `no_grad`. RCO does neither. That is the same conclusion RCO's
adversarial review reached from the paperwork - "RCO's loss/backward/update at full scale on
MPS is untested" - now confirmed empirically, with five candidate causes eliminated.

Installation cannot fix it: `transformers` gates the FLA path on
`is_torch_cuda_available() or is_torch_xpu_available() or is_torch_mlu_available()`, which is
false on Apple Silicon by construction, and `fla`'s ops import `triton`, which does not exist
for macOS-arm. The remaining MPS lever is `PYTORCH_MPS_PREFER_METAL` (the backend reads
`PYTORCH_MPS_FAST_MATH`, `PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO`, `PYTORCH_MPS_PREFER_METAL`,
`PYTORCH_MPS_LOG_PROFILE_INFO`, `PYTORCH_MPS_TRACE_SIGNPOSTS`). `PYTORCH_MPS_PREFER_METAL=0`
was tried: the load completed in 18.0 s and the process then froze at 00:00 CPU exactly as
before, so the lever does not reach this path. With every component exonerated, no MPS knob
effective, and no installable package closing the gap, the RCO stage moves to a rented CUDA
GPU - the one stage that needs the whole-model gradient, where it was already validated,
leaving the chain, export and evaluation local.

**D11. RCO is the second half of the method, not an allocation detail.** An earlier
revision of this section treated RCO as optional "when the target is not all-1-bit". That
was wrong, and it under-scoped the plan.

RCO is **Riemannian Constrained Optimization** (`cuda_runtime/.context/sources/rco`,
arXiv 2605.00649, *Model Compression with Exact Budget Constraints via Riemannian
Manifolds*): it solves budget-constrained discrete assignment - choose one of K options
per group to minimise a **non-decomposable** objective, subject to an **exact** equality
budget - by relaxing the assignment to per-group softmaxes on the manifold `C(alpha) = B`,
projecting each Adam step onto the budget's tangent plane, retracting with a binary search
so the budget holds to machine precision, and solving the discrete step with a
budget-constrained DP over an annealed Gumbel-STE. Per-group penalties are exactly what it
replaces: dynamic programming alone can only optimise per-group proxies, and
Lagrangian/penalty relaxations satisfy the budget only approximately.

Our `solver/rco.py` is that implementation, not a cost model: it imports
`manifold.project_gradient`, `retraction`, `vector_transport` and
`search.quant.budget_constrained_argmax` from the pinned tree, parameterises one
`alpha` per eligible tensor over K=2 options (Q1, BF16), and drives them with
`objective.full_kl` - the LM-level, non-decomposable loss the method requires. Byte costs
come from the **serialized** cost manifest when supplied
(`serialized_costs_verified`), so the budget it enforces is the one the export produces.

Why this is load-bearing for our target, in order:

1. **The paper's headline regime is an RCO result.** 2.13 bpp at near-bf16 accuracy is a
   *mixed* allocation under an exact budget, not a uniform-width model. Our Q2/ternary
   target (~2.1 bits/weight, ~7.2 GB, where Bonsai's own ternary Q2_0 sits) is precisely
   that regime, so RCO is what makes the target size good rather than merely small.
2. **It decides which tensors keep 16 bits.** Even at a fixed average budget, that choice
   is the quality lever - the same average bits allocated better is a better model.
3. **It needs the full candidate database.** `RCOTrainer` requires candidates covering
   every eligible projection **and the embedding and head**, so the `embedding` and `head`
   stages are RCO prerequisites, not merely separate stages.
4. **It cannot rescue bad candidates.** An allocator choosing between a 73,786-PPL Q1
   tensor and its BF16 original will simply choose BF16 everywhere, which is large rather
   than good. So GSQ quality comes first and RCO allocation second - but both are
   required, and neither substitutes for the other.

Known gaps, from the port's own record: RCO loss/backward/update at full scale on MPS is
**untested** (`docs/PORT_VALIDATION.md:117,730`), and three tests skip because they need
the external `manifold` RCO package (`:402`).

The independent review of this stage found the wiring **faithful and live** - tangent
projection, binary-search retraction, momentum transport after retraction, and annealed
Gumbel-STE over the budget DP in upstream's own order, over the non-decomposable
full-vocabulary KL on the propagated stream - and verified it empirically: `p=0` versus
`p=1` through `interpolated()` moves hidden states by 2.71, every group of the tiny model
receives nonzero gradient in one update, retraction lands 4.8e-7 bits/param from target with
`tol=1e-6`, and 200 fuzzed assignments never exceeded the weighted budget. Budget
exactness is verified where checked: the manifest's per-tensor Q1 cost is exactly 1.125
bits/param and BF16 exactly 16 for all 498 tensors, and the all-Q1 total equals the real
export's 3,803,453,760 bytes.

It also found the stage **not runnable as configured**, in four ways, and three are now
fixed:

- **Nothing consumed `allocation.json`.** A successful allocation was written and dropped,
  so the mechanism's second half could not reach an artifact. `solver/tools/export_plan.py`
  now takes `--allocation` and maps its per-tensor q1/bf16 choices onto the exporter's plan,
  validating that the allocation covers exactly the candidate database. Verified: a uniform
  allocation still reproduces the pinned 3,803,453,760-byte total, and a 60-tensor bf16 mix
  produces a 438/60 plan.
- **`hard_allocation` used a raw argmax** rather than the budget-constrained knapsack
  upstream's discrete forward runs, so the final answer was not the feasible assignment the
  method is defined on. Now uses the same DP solver as `step`.
- **A uniform Q1 allocation was accepted silently.** Measured: 0 bf16 groups at a 4.0e9-byte
  budget against 96+ at 4.15e9, with nothing in between warning - a model whose budget was
  never spent would have shipped looking like a considered allocation. `hard_allocation` now
  refuses a uniform answer whenever the budget affords at least one upgrade, and is allowed
  to return one at the floor where it is the only feasible answer. Both directions are
  tested, along with the retraction's budget exactness.
- **Still open**: `make_run_configs.py`'s `rco` config is refused at startup and lacks keys
  `rco_run` reads; the local full-scale RCO smoke cannot pass because the local store
  carries no commit generation (`smoke.py:134-135`), so `--stage rco --max-steps 1` is the
  substitute; three optimizer knobs differ from the pinned driver without recorded
  justification (temperature anneals to 0.1 rather than tau 1.0 -> 0.01 exponentially,
  `rco_lr` defaults to 0.01 against 0.1, and one Gumbel sample per update against
  `--n-gumbel-samples 4`); and the BF16 leg of the cost manifest has never been exercised by
  a measured export, since a bf16 choice runs `converter.modify_tensors` and writes FP32 for
  any output with `ndim <= 1` while the budget charged 2 bytes per element.

**Cost, measured**: one full-corpus forward is 75.8 s (linear attention) and 31.4 s (full
attention) for 104,557 tokens, so a full-model forward over the corpus is ~2080 s and an
**update epoch is ~3 h** - a bounded, affordable stage, but not a quick one. 53.8 GB of
weights stay resident because `block_cpu_offload` is refused for non-gsq stages; against
128 GiB the risk is a factor of two to three in MPS backward/checkpoint efficiency, not
correctness. **The mixed-allocation floor is around 4.15e9 bytes**, which sits inside the
phone budget - so a mixed Q1/bf16 arm under RCO is reachable, and a uniform-Q1 budget is a
budget that was never spent.

| stage | what it does | state |
|---|---|---|
| `initialize` | candidate database from RTN | exists |
| `gsq` | per-block 1-bit weights on the propagated stream (D1, D3.1) | fixed tonight, validating now |
| `embedding` | GSQ on `embed_tokens` over observed token rows | exists; RCO prerequisite |
| `head` | full-vocabulary KL on `lm_head` | exists; RCO prerequisite |
| `rco` | Riemannian constrained allocation across tensors under an exact byte budget | exists; untested at scale on MPS |

**D12. Promotion gate: the criteria that make an artifact shippable.** Nothing in D0-D11
adjudicates the goal, because every measurement in them is Mac-side: `llama-perplexity`
scores on this machine, and `MCPZimEvalCLI` ships as a plain macOS executable
(`ios/project.yml:246-248`). The repo already defines the device gate this plan was missing
(`docs/BONSAI_27B_IPHONE_EVALUATION.md:348-355`, `eval/README.md:103`), and the superseded
sibling plan carried numeric thresholds this revision dropped
(`QUANTIZATION_PLAN_2026-09-19.md:288,382`). An artifact is a shipping candidate only when
all of these hold:

1. **Held-out perplexity against a bar measured for its own size class** - not one 3.8 GB
   bar applied to every arm, and never measured on the invocations that gated training.
2. **The app tripwire suite** (`run_mac_pipeline.py` / `compare_app_reports.py`) with a
   like-for-like candidate role.
3. **Loads on device within budget** - `DeviceProfile.modelFitsDevice` and the 6144 MB
   process ceiling, with peak memory **measured** rather than estimated.
4. **Passes the physical-device gate** - end-to-end target conversations, prompt compaction
   under the 6,144-token rolling budget, peak Metal memory, a 10-minute thermal loop, p95
   TTFT within 10% of the shipping control, and no jetsam kill. The speed floor is the
   official MLX 1-bit result, ~11 tokens/s on iPhone 17 Pro Max; a GGUF provider materially
   slower than that is not a win.
5. **A pinned app entry** - filename, bytes, SHA-256, memory estimate and KV/context policy
   for the winning artifact.
6. **Tool calling stays strong.** It is most of what this model does: of the 87 calibration
   invocations, 42 are the planner call ("Return ONLY JSON: {question, need, time,
   subjects, queries}"), 37 are evidence-ID selection, and 8 are the assistant-with-tools
   prompt - every one mentions tools. The gate declares `expected_tool` on 18 of 66 frozen
   turns across six tools (`discuss_article` 7, `article_factoid` 5, `article_overview` 3,
   `locate`, `near_places`, `plan_driving_route`). Structured output is the first thing
   aggressive quantization breaks, so tool selection and JSON well-formedness are
   **acceptance criteria**, not diagnostics, and a candidate that degrades them is not shipped
   even if its perplexity improves.

**Never score a conversation you used to make a decision.** The 8-invocation held-out slice
gates every block (D3) *and* reports the final number, so it is held out from training but
not from selection: a run that picks its best epoch on it has already spent it. Keep a
separate confirmation set that no stage ever scores, and report final numbers on that.

**What a mixed arm is compared against.** The phone budget admits a mixed Q1/BF16 allocation
from the ~4.15e9-byte floor up to ~5.5 GB, and **no shipping model occupies that band**:
Bonsai ships 3.80 GB (Q1_0) and 7.16 GB (ternary Q2_0). Since the 7.16 GB build is outside
the phone budget (section 1), a mixed arm is held to the **harder** comparison - beat the
3.80 GB incumbent while spending at most 5.5 GB - rather than to a same-size bar that does
not exist. Bonsai's ternary Q2_0 therefore does **not** need measuring for this plan; it
would only matter if a 7.16 GB arm were a deliverable, which D0 and section 1 now rule out.
This supersedes the earlier "measure the ternary incumbent" item.

## 4. Build list (ordered)

Ordered by what unblocks what. Items marked DONE are finished and verified on this machine.

1. **DONE** `solver/tools/export_plan.py` - candidate database (+ `--layers N`, `--embedding`
   /`--head`) -> export plan. Verified: the all-Q1 arm reproduces the pinned
   `all_q1_serialized_bytes` exactly, and `--layers 4` quantizes 33 tensors with the
   boundary at layer 4.
2. **DONE** `bind_identity` no longer binds operational controls (`gsq_max_blocks`,
   `checkpoint_seconds`), so one chain resumes across a per-block sweep and a
   single-process run. `solver/tests/test_gsq_residency.py` covers both directions.
3. **DONE** store lock + dotfile exclusion in `prune_store.py`/`checkpoints.py`, closing the
   prune-vs-publish race that could corrupt the newest checkpoint or abort a run
   (`tests/test_checkpoints.py`).
4. **DONE** Reclaimed disk (D8): 534 GB of dead attempt stores deleted, taking the volume
   from 146 GiB free to 681 GiB. Outputs, calibration inputs, warmstart states and the RTN
   control artifact were left in place.
5. **Measure the bar: held-out perplexity (D10).** DONE for the bar itself - Bonsai 27B
   Q1_0 measures **7.0653 +/- 0.244** on the held-out invocations, recorded in the ledger
   by `solver/tools/quality_gate.py`. Pending: the bf16 ceiling via `mlx_lm`, which needs
   the 52 GiB model resident and therefore runs when the trainer is stopped.
   `solver/tools/quality_gate.py` is DONE for the perplexity arm (measure, refuse a
   duplicate label, rank against the bar, warn on mixed text); its suite half stays with
   `run_mac_pipeline.py` rather than being duplicated here.
6. **DONE** `solver/gsq.py` + `solver/run.py` + `solver/boundaries.py`: the same-input
   objective on the **propagated** stream (D1); upstream's initialisation
   (`std=0.01`, `strength=6`, applied to the normalized weight, in `create_quantizer`
   and `CPUOneBit`); and the logit-scale schedule (`scale` 100 -> 500 with temperature
   2 -> 0.05, both annealed per optimizer step as upstream's trainer does) passed
   through to the quantizers, including the embedding stage. The propagation is
   **kept** - an earlier revision of this design proposed deleting it, and the review
   refuted that against `main.py:385-402`. Verified by `solver/tests/test_gumbel_mps.py`,
   which asserts the new initialisation is **bit-identical** to upstream's quantizer at
   one seed, plus the existing objective, boundary and canary suites (85 + 18 passing).
7. **PARTLY DONE** `solver/run.py`: the composed-stream drift metric (D3.1) is implemented -
   recorded per block to `drift.json` with `gsq_max_drift_growth` stopping the chain at a
   block boundary. Pending: the held-out split (D2) and the in-loop held-out reconstruction
   metric (D3), which need a fixed slice reserved in `corpus_inputs`/`gsq_run`.
8. **Run the stages in order and keep the chain honest.** `initialize` -> `gsq` ->
   `embedding` -> `head` -> `rco`, each from `make_run_configs.py` with its own fresh
   output directory (`run.main` refuses a non-empty one). Verified by running it: `gsq`
   refuses to start without an `initial-database.json` from `initialize`, so the order is
   enforced in code rather than documented and hoped for.
9. **The embedding and head stages (D11).** RCO prerequisites, not polish: `RCOTrainer`
   raises unless `set(database)` covers every eligible projection *and* the embedding and
   the head, and `rco_run` refuses RTN-only boundary reports ("RTN-only embedding/head are
   smoke seeds, not final recipe"). So `embedding` -> `head` must run before `rco`, each
   from its own fresh output directory, carrying the `gsq` chain's candidate database.
10. **Get RCO working, in this order.** Three of the four blocking defects the review found
    are already fixed (the export link, the knapsack, the silent-uniform guard); the rest:

    a. **Make the stage launchable.** `make_run_configs.py`'s `rco` config is refused at
       startup and lacks keys `rco_run` reads - it needs `candidate_database` and
       `candidate_archives` from the `gsq` chain, and `boundary_reports` from the
       embedding and head reports. Diagnose the refusal precisely rather than guessing,
       then emit the whole set for the `rco` stage.
    b. **DONE - the BF16 leg agrees, all 498 tensors.** `solver/tools/verify_bf16_costs.py`
       (new, CPU-only, no GPU and no weights: meta tensors, 1.6 s to inspect) reuses the
       exporter's own converter and byte rule at pinned prism revision
       62061f91088281e65071cc38c5f69ee95c39f14e, and found every eligible 2-D source 1:1
       with `converter.modify_tensors`, ndim-preserving, and priced to the same
       53,786,705,920 bytes by the manifest's rule and the exporter's. Perturbing one
       `bf16_bytes` by 32 makes it exit 1 naming the tensor, its runtime name, the real
       cost and the budgeted cost - so the check can fail, which is the point. Residual gap
       the tool states: meta tensors carry no values or dtypes, so a *value-dependent*
       branch in `modify_tensors` would escape it, and the Q1 leg is not re-derived (it is
       already pinned by the measured 3,803,453,760-byte export).
    c. **Match the pinned driver's optimizer knobs**, each currently a silent deviation:
       temperature tau 1.0 -> 0.01 **exponentially** (we anneal linearly to 0.1), `rco_lr`
       0.1 (we default to 0.01), and `--n-gumbel-samples 4` averaged per update (we draw
       one).
    d. **Price one real update**: `--stage rco --max-steps 1` on the real corpus - the
       substitute for the local smoke, which cannot pass because the local store carries no
       commit generation (`solver/smoke.py:134-135`). This produces the liveness evidence
       the smoke would have: `loss`, `raw_gradient_norm`, `budget_bits`.
    **Requirement for this step**: the export plan's `max_gguf_bytes` must be the RCO target,
    not the 4 GiB the RTN plan uses - the exporter's gate is
    `estimate > plan['max_gguf_bytes']`, so a 4 GiB gate against a 4.0e9 target accepts an
    overshoot silently. `export_plan.py --max-gguf-bytes` is the knob.

    e. **Run the allocation at a phone-viable mixed budget** (>= ~4.15e9 bytes, <= ~5.5 GB)
       with `held_out_stride` active, then `export_plan.py --allocation` ->
       `export_qwen.py` -> a mixed GGUF.
    f. **Measure it as its own size class**: held-out PPL against a bar measured for that
       class (Bonsai's ternary Q2_0 is the incumbent there and has never been measured),
       the app tripwire suite, and the tool-calling gates - then D12's device gate.

    **Acceptance for "RCO works"**, from the review's own proof numbers: `budget_bits`
    within 1e-6 of `trainer.target`; a finite `raw_gradient_norm > 0`; per-update seconds
    that extrapolate to a bounded epoch; and an allocation whose exact byte sum is at most
    the target **with at least one bf16 group** - the failure signature being exactly zero
    groups, which is what this port produced at 4.0e9 bytes without saying a word.
11. A third comparison role in `prepare_comparison.py`, running the candidate in the *same*
    runtime as Bonsai with the same template/KV/context, so the head-to-head is
    like-for-like (D10, section 2.0.1).
12. `solver/tools/budget_curve.sh` or a small runner: the D5 sweep over epochs x depth.

## 5. Estimates, and how they will be checked

| quantity | estimate | basis |
|---|---|---|
| per block, 1 epoch | ~10.7 min | 465 s train + 76 s propagation + ~100 s publish, all measured; D1 keeps the propagation |
| full 64 blocks, 1 epoch | ~11.4 h | 64 x 10.7 min |
| full 64 blocks, 2 epochs | ~19 h | +465 s/block |
| full 64 blocks, 4 epochs | ~33 h | +3x465 s/block |
| shallow curve, 8 layers x {1,2,4} epochs | ~4 h | 24 block-equivalents |
| export per D4 point | minutes, CPU only | `export_qwen.py` docstring, no GPU |
| perplexity per arm | minutes | `llama-perplexity` on one 3.8 GB GGUF, CPU/Metal |
| RCO allocation stage | **unmeasured** | full-vocabulary KL per step; the port's record has no full-scale MPS number |
| storage, full run, pruned | ~8-10 GB per attempt live | prune verified at 8.32 GB reclaimable per attempt |
| storage, full run, unpruned | ~600 GB | 9.5 GB/block measured over 8 blocks |
| **free space available** | **681 GiB (91% used)** | `df` after reclaiming 534 GB of dead attempt stores |

The storage estimates and the free-space figure together are the binding constraint: an
unpruned run cannot finish, and the abandoned roots must be reclaimed first (D8). The
run must refuse to start when the free space cannot cover its remaining blocks plus the
export reserve.

**The reclamation is implemented** as `runs/native-mps/reclaim-superseded.sh`. It keeps
the newest two stores under `checkpoints/gsq` and the newest two `gsq-pb-*` output
directories - the live roots plus the ones the next block resumes from - and removes
older ones. Two is sufficient because the store is content addressed and every publish
carries every completed block's archive, so the newest store is a complete record of the
calibration; `run.py` explains that retention is load-bearing, not incidental. Measured
on the real chain, one superseded store is ~26 GiB and one superseded output directory
~5 GiB, against a block that writes ~15 GiB of store and up to 28 GiB of output, so
without this a 64-block sweep stops on the driver's own 60 GiB guard after about five
blocks. Only roots carrying the driver's `YYYYMMDD-HHMMSS-xxxx` stamp are ever eligible:
a deliberately named root such as `validate-gsq-20260920c` sorts *after* every timestamp
and a naive keep-the-newest rule would have deleted a 54 GiB validation store, which the
first dry run of this script did before the check was added.

**Every number here already includes the linear-attention fallback.** The model's 48
linear-attention layers run the reference PyTorch implementation, not an optimized kernel:
`transformers` reports `causal_conv1d_fn` and `chunk_gated_delta_rule` falling back because
`causal_conv1d` and `flash-linear-attention` are not installed. Both packages *resolve* on
this platform (`uv pip install --dry-run` finds `causal-conv1d 1.7.0` and
`flash-linear-attention 0.5.2`), but their accelerated paths are the CUDA/Triton ones
`transformers` is pointing at, so the fallback is expected on MPS. It is therefore not a
regression introduced anywhere - but it is a large, unexplored speedup: installing them and
timing one block is a cheap experiment for the next session, and if it works it moves every
row of this table.

Each of these is falsifiable cheaply: the per-block time is in every report, the export is
CPU-only and fast, and the eval reports its own timings. The budget-curve run is the
estimate's own check.

## 6. Open decisions for the owner

1. **Epochs.** D5 measures the curve; someone must say what quality is good enough to ship.
2. **Eval scope per point.** All suites every point is expensive; `conversational_qa_v1` plus
   one other is the proposal.
3. **GPTQ init (D7)** - worth building only if the curve shows init-bound quality.
4. **The 1-bit target itself.** `zimfo-q1-v1` is sign+scale at group 128. If the curve says
   1 bit cannot hold our conversations, the honest answer is more bits, not more epochs.
