# Answers to the native-port session's calibration questions

September 20, 2026. Checked against the current collector, GSQ stage runner,
embedding/head trainers, RCO step, objective, original weight index and retained
capture manifests. No new model capture or optimization was run for these answers.

## Main correction

The observation that `BlockTrainer` consumes block-input hidden states is correct.
The conclusion that all 64 layers must be captured before training can work is
not correct for this implementation. `solver/run.py::cached_inputs` constructs
full original-teacher and quantized-embedding student inputs from token IDs and
weights. `gsq_run` trains block n and then propagates both streams into the next
block's cache:

```python
teacher_next = run_block(model, n, teacher_input)
student_next = trainer.hard_forward(student_input)
```

Those are complete sequences, not the 32-row projection samples. Block n needs
its current teacher/student inputs, not every later block's inputs simultaneously.
Future student inputs depend on the learned quantized prefix and cannot be
supplied by a one-time original-model capture. Optional precomputed teacher-only
caches can save recomputation; they are not missing mandatory source data.

The current CUDA implementation already completed blocks 0/1 and durably stored
the block-2 cache using this method. That is evidence of the data flow, not proof
of native MPS/MLX compatibility.

## Answers, in the question order

1. **All 64 block inputs:** technically possible to extend the collector, but
   unnecessary for the existing streaming training algorithm. The suggested
   ~68.52 GB is the BF16 **teacher-only** hidden payload across all 64 blocks;
   it excludes student caches, serialization, weights and optimizer state.
   One full-corpus hidden stream at one boundary is 1,070,663,680 bytes; a
   teacher/student pair is about 2.14 GB. Preserve the sequential cache design
   rather than making 68 GB of new capture a prerequisite. If a different port
   deliberately requires a teacher cache, document that as its interface choice.

2. **Reference head and log-sum-exp:** already available/computable. The original
   index maps `lm_head.weight` to `model-00018-of-00018.safetensors` under
   `tools/calibration/runs/qwen3.8-27b-original/`. Shape is `[248320, 5120]`,
   stored in original BF16. `FullVocabularyKL` accepts that frozen BF16 tensor
   and promotes row chunks to FP32; it does not require a separate permanently
   FP32 full-head artifact. Its forward pass calculates teacher and student
   global log-sum-exp in vocabulary chunks and saves the normalizers for backward.
   They do not need prior capture. Current `RCOTrainer.step` runs both the frozen
   reference model and the differentiable interpolated student, not just one
   forward total. Frozen teacher hidden states/normalizers could be cached as an
   explicitly validated optimization, but cannot replace the student's graph.

3. **Machine and weights:** yes, both remain on this 128 GiB unified-memory Mac.
   Original model artifacts total 55,586,035,724 bytes (~51.77 GiB). The complete
   activation package is already local; paths are in the overnight handoff.
   A fresh disk check now reports about **985 GiB free**, superseding the earlier
   28 GiB observation. Recheck memory pressure, output retention and concurrent
   workloads before running; disk availability is not a memory-feasibility proof.

4. **Corpus size:** the solver already uses **104,557 token positions across
   87 invocations**, including the full 4,673-token longest sequence. It is not
   limited to 87×32 projection rows. The 32-row samples are diagnostics; they do
   not define GSQ's effective training token count. No recollection is needed to
   obtain those full sequences. We have not verified the proposed 4,096-activation
   threshold or its applicability; it does not establish this corpus's adequacy.
   Adequacy should be measured on held-out Zimfo behavior and coverage.

5. **Capture time and expansion:** retained activation replay/export took
   878.49 seconds (~14.6 minutes); independent validation took 39.39 seconds.
   Those are replay/export timings, not conversation generation or optimization
   timings. Existing collector limits included a 40 GiB export cap, 24 GiB disk
   reserve, 80 GiB MLX allocation budget and 16 GiB system reserve. An all-layer
   expansion would need a new disk estimate and policy. More invocations are
   not strictly better: redundant or unrepresentative material changes weighting
   and costs time. Start with the agreed complete corpus, then expand for measured
   coverage gaps. Do not mix the held-out comparison into calibration.

6. **Cache semantics:** capture calls
   `body(token_array, cache=None, input_embeddings=embeddings)` separately for
   each invocation. MLX's body turns `None` into one `None` per layer. No recurrent
   or KV state is carried across invocations. PyTorch block calls use
   `past_key_values=None, use_cache=False`; RCO full-model calls also disable
   cache. These are aligned fresh-state semantics, although their API types differ.
   A native port's cache-type mismatch is an adapter issue, not evidence that the
   recorded prompts require persistent cache objects.

7. **First-block boundary:** the collector explicitly saves
   `body.embed_tokens(token_array)` before calling the body. It is the embedding
   output **before block 0's input layer norm**. `EmbeddingTrainer` currently
   re-embeds the token IDs with original and trainable candidate rows and compares
   their block-0 outputs. It does not require captured block-0 inputs as an
   additional training argument. The captured boundary is a useful parity oracle.

8. **Moments:** yes, they cover **all** rows of each observed BF16 projection
   input, after conversion to FP32: `sum(float_x)` and `sum(float_x*float_x)`.
   They are not extrapolated from the 32 samples. “Exact” here means full-row
   coverage, not infinite-precision arithmetic or bitwise equality across different
   reduction kernels. They can check first moments and the diagonal of an
   uncentered Gram matrix, not its off-diagonal entries. Use appropriate numerical
   tolerances and the recorded row counts.

9. **Kernel provenance:** this was **MLX 0.32.2 / mlx-lm 0.31.3**, not the PyTorch
   flash-linear-attention or causal_conv1d packages. The recorded Qwen backend
   SHA-256 is `f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a`;
   the local `mlx_lm/models/qwen3_5.py` still matches it. It uses MLX `nn.Conv1d`
   and dispatches gated delta with `use_kernel=not self.training`; the gated-delta
   helper selects its custom Metal kernel for GPU inference and a reference path
   otherwise. The normal inference load path uses evaluation mode. This establishes
   the expected source path, not a retained per-kernel execution trace: the old
   export did not record exact Metal kernel dispatch variants. Validate output and
   gradient parity for the port rather than claiming kernel identity from a version
   string. In particular a differentiable MLX training path may use a different
   recurrent implementation from inference capture.

## Recommended next action

Proceed with the native device/autograd port and measured actual-block checks
using the existing corpus and weights. The session's proposed production-width
timing measurement is useful. The reported `gsq-rco-mlx/docs/PARITY.md` work was
not reviewed in this response; inspect its exact revision, dtype, noise, backward
and scale-grid coverage before inheriting a parity claim.

A one-invocation implementation check can regenerate successive teacher/student
states locally and compare the captured first/final boundaries. It does not
require a new all-layer capture deliverable before port work begins. Test both
hybrid block types, complete longest input, full RCO loss/backward/update and
cold native checkpoint recovery. Keep that distinct from shrinking the agreed
87-invocation training dataset.

For head training, use the **final propagated teacher/student cache**, with the
appropriate final norm, plus original reference-head weights. The original-only
captured final hidden states do not stand in for the quantized student's final
hidden states. This is the same essential distinction as the GSQ cache streams.

Source references: `tools/calibration/mac_collect.py` (`Sink.projection` and
capture loop), `solver/run.py` (`cached_inputs`, `gsq_run`, boundary stage),
`solver/boundaries.py` (`EmbeddingTrainer`, `HeadKLLoss`), `solver/rco.py`
(`RCOTrainer.step`), `solver/objective.py` (`FullVocabularyKL`). Solver paths are
under `tools/calibration/`. See [the overnight handoff](QUANTIZATION_MAC_OVERNIGHT_HANDOFF.md)
for data locations and immutable checkpoint references.
