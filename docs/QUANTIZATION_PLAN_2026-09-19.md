# A Zimfo-specific quantization experiment

Date: September 19, 2026. Status: Mac calibration conversations, BF16 activation export, independent artifact validation and paired app comparison completed. The report validator's inactive llama.cpp top-k mismatch was corrected and tested, and the paired report regenerated from retained outputs. Full CUDA solver integration and durable cloud resume remain unfinished.

## First complete Mac collection

The original-Qwen capture completed all 25 conversations/85 turns and produced
87 model invocations. Prefill export completed in about 14.6 minutes, and the
independent validator accepted all 87 sequences, 104,557 tokens and 497 projection
inputs per sequence. Tensor files total 23,436,670,016 bytes (21.83 GiB). This
verifies export integrity and coverage, not CUDA/MLX parity or solver feasibility.

Both models completed the 66-turn held-out app suite. Raw automated assertions
passed on 51 Qwen turns and 48 Bonsai turns; these are app regression checks,
not a general quality benchmark. Qwen gains four turns and regresses one relative
to Bonsai. Human answer review is still needed before interpreting the difference.
The report initially rejected Bonsai's recorded task-default `topK=40` despite
temperature zero. Source inspection confirmed `LlamaCppProvider` installs only
the greedy sampler in that branch, so top-k is inactive. The comparator now
records this as an explicit backend-specific warning; nonzero temperature and
MLX top-k mismatches remain rejected. All 32 comparison tests pass. No model
rerun is needed for this reporting correction; original failed evidence remains.

The regenerated report validates successfully. Among the paired union of 17
model-involved turns (eight conversations), Qwen passes 13 and Bonsai 11.
Conversation-weighted pass rates are 75.0% versus 64.6% for that subset and
77.3% versus 74.4% across all app turns. This is a small regression suite, with
closed-loop context differences, not proof of a general model advantage.
[Aggregate evidence](benchmarks/quantization-2026-09-19/mac-completed-results.json).
Local `status.json` records `completed_with_reporting_recovery` while retaining
the original failed report/stage and the corrected `comparison-retry.json`.

## Mac execution in progress

The full calibration trajectory run uses original Qwen, the local June Wikipedia
archive and California StreetZIM: 25 conversations and 85 turns. It records every
actual model invocation, including rendered prompts, token IDs and tool/evidence
context. Deterministic turns create no artificial model invocation. The separate
held-out comparison freezes the four existing suites: 25 conversations and 66
turns, with archive SHA-256s, evaluator binary, sampler and model manifests.

Local artifacts are under `tools/calibration/runs/` (ignored, never committed):
`qwen-calibration-v1-20260919` holds the running input capture, and
`comparison-v1-20260919/protocol.json` freezes the paired evaluation protocol.
The local pipeline records per-stage status/logs and runs Metal workloads
sequentially. Export failures do not silently become success; the independent
comparison may still finish and report the calibration failure.

`mac_collect.py` replays all captured invocations with original BF16 weights,
fresh state, no generation, and no truncation. It saves full first-block and
final normalized hidden states, 32 stratified input rows per projection/call,
and all-row first/diagonal-second moments for all 497 projection inputs. This
is a **replay package**, not a substitute for GSQ's nonlinear block objective
or RCO's full-vocabulary loss. Exact tokens and original weights remain required
to regenerate complete block inputs and optimize compressed embeddings.
Its disk preflight covers the entire dataset, with a 40 GiB export limit and
24 GiB free-space reserve; memory checks enforce an 80 GiB MLX budget and 16 GiB
system reserve. Limits fail explicitly rather than dropping data. See
[QUANTIZATION_CORPUS.md](QUANTIZATION_CORPUS.md) for weighting and review caveats.

The independent CPU validator checks copied provenance, exact projection
coverage, shapes/dtypes, hashes, finite values and row indices in bounded chunks.
The comparison reports all app turns separately from the paired union of turns
where either model was invoked, so deterministic routing cannot inflate claims
about model improvement. Existing content assertions are automated evidence;
critical answer-quality judgments and phone acceptance remain separate.

The local Linux/amd64 CUDA 13 image builds and its CPU objective/import tests pass;
this is not GPU or full-model feasibility evidence. Local checkpoint primitives
pass exact next-update resume tests but still need solver/GCS integration and a
real interruption/restart test. No paid GPU is running. See
[QUANTIZATION_CUDA_RUNTIME.md](QUANTIZATION_CUDA_RUNTIME.md) and
[QUANTIZATION_CHECKPOINTS.md](QUANTIZATION_CHECKPOINTS.md).

## Approved execution order

1. Measure original full-precision Qwen3.8-27B and shipping Bonsai on the Mac using the same frozen Zimfo tasks, evidence, tool cases and follow-ups.
2. Capture the complete selected set of real model inputs and collect original-Qwen reference data. Keep calibration and evaluation separate; no small-dataset preliminary optimization.
3. Verify packing, real Qwen block optimization and a complete memory-bounded RCO update before long optimization. Synthetic projection checks alone do not establish feasibility.
4. Quantize at multiple explicit byte budgets around Bonsai, including modestly larger candidates. Reuse validated calibration data and candidate weights.
5. Report compression loss against original Qwen, improvement over Bonsai, actual complete deployment size, memory, speed and critical regressions for every candidate.
6. Test the strongest candidates in the app on the phone, including offline tools, conversations, voice and stability, before selecting a replacement.

User approved this sequence and starting implementation. Paid GPU execution remains a separate decision after pricing; no GPU was rented during this implementation pass.

## Current launch discipline

Follow [QUANTIZATION_GPU_READINESS.md](QUANTIZATION_GPU_READINESS.md): prepare
the runtime and complete instrumented package before renting; stage inputs in
the compute region; run a short actual-model smoke, then continue useful
quantization with durable checkpoints; return completed artifacts to the Mac.
The old synthetic launcher is disabled for paid execution. No package installs
or planned source builds belong on the paid GPU. Durable cloud checkpoint/resume
and full solver integration remain unfinished, so no GPU start ETA is claimed.

## Decision

The fixed baseline and quantization source is **Qwen/Qwen3.8-27B, original full-precision BF16 weights**. Use that model for reference outputs, activations, GSQ reconstruction targets and RCO reference loss. Pin its exact revision and tokenizer before collection. **The objective is to produce a quantized Qwen3.8-27B that beats the currently shipping Bonsai on Zimfo's tasks preferably at the same or smaller total deployed model size, while considering a modest increase if the quality gain is substantial.** Bonsai supplies comparison measurements and the size anchor only; it supplies no teacher activations, reference logits or source weights. Optimize for Zimfo's grounded conversation and tool use, in its existing non-thinking mode. Do not use coding benchmarks as the calibration corpus or dequantize Bonsai to manufacture a baseline.

The current execution target is **Mac-first collection, GCP Spot GPU quantization**, with the original RTX 3090 research informing the memory feasibility work. The user confirmed this workflow and explicitly rejected a separate small-dataset collection gate. Run the full selected Zimfo calibration set and full-precision reference collection on the Mac without needing a phone. Validate shapes, coverage, precision and checksums during collection, then stage the complete versioned package for the selected GCP GPU before launch. Do not run a long optimization merely to validate collection, and do not wait for a CUDA pilot before collecting the rest of this modest dataset. Packing, numerical and memory checks remain implementation tests, not an additional dataset phase. Treat full RCO search as a subsequent engineering milestone, not a free operation on exported Hessians.

Bonsai's exact total deployed size is the **preferred size anchor, not an absolute acceptance ceiling**. Record file hashes and measured bytes for every required model file, scale, transform, embedding and output head. Report each candidate's absolute bytes and percentage difference versus Bonsai alongside its measured quality. The user explicitly accepts considering a modest increase for a substantial quality gain; no fixed percentage has been chosen. This does not make a 6–8 GB Mac result an accepted phone deliverable. Final selection uses the measured size/quality tradeoff plus phone memory and latency with the app, archives and voice services loaded.

This supersedes the imported coding-calibration directive. A dense-only MLX capture sketch was started before the clarification and removed after source review: it would not cover this hybrid architecture or supply the actual GSQ/RCO training contract. The initial research pass left app code unchanged; the implementation section below records the subsequent opt-in provider capture hooks. Model selection and installed apps remain unchanged. An ignored Python environment was prepared under `tools/calibration/.venv`; it is not a deliverable quantizer.

## Paired quality and size comparison

The first experiment measures **original Qwen3.8-27B BF16 versus shipping Bonsai** on identical frozen Zimfo tasks and source evidence. This establishes whether the selected original model actually has useful quality headroom. No quality measurements have been collected yet. Each subsequent quantized Qwen checkpoint adds a row, including disappointing results; never substitute a publisher benchmark for a Zimfo measurement.

For a higher-is-better score, report both **compression loss = full-precision Qwen score − candidate score** and **gain over Bonsai = candidate score − Bonsai score**, in percentage points. Negative compression loss is allowed if a candidate scores better on the finite test set. Do not define a percentage of headroom retained when the reference does not outperform Bonsai. Report per-category results and newly introduced critical failures so an aggregate cannot hide wrong-source answers. Use paired conversation-level uncertainty analysis before making a strong improvement claim; a small suite's raw score is not statistical proof.

Keep model-only fixed-evidence comparisons separate from app conversations where earlier model choices affect later retrieval. For the former, freeze logical messages, evidence, rubric, sampling policy and context policy; use each model's correct tokenizer/template and record their hashes. Identical token IDs are required for replay of a given model, not between different tokenizers. For the latter, freeze scenario and source snapshot, and retain every model invocation and tool outcome. Hold KV precision fixed within quantized-Qwen comparisons and record any unavoidable baseline runtime differences.

Track complete artifact bytes and percentage change versus Bonsai, peak runtime memory, TTFT and decode speed. Performance comparisons require the same machine and workload; a Mac BF16 timing is not a prediction of iPhone performance. Treat size estimates separately from measured packed file sizes. Keep calibration, development and final evaluation subjects/conversations disjoint. Failed or missing cases cannot disappear from the denominator. Any final suite used repeatedly to choose bit budgets becomes development data.

Implemented [paired comparison reporter](../tools/calibration/quality_frontier.py): checks protocol/case alignment, includes failed cases, reports per-category and critical regressions, and permits modest size increases without automatic promotion. It consumes normalized scored reports; adapting real Mac output and reviewed scoring into that contract remains unfinished. It has not produced real model quality measurements.

### Header-only encoding inventory (September 19)

Pinned original Qwen revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Reading 150,872 bytes of safetensors headers identifies 27,781,427,952 BF16 parameters: 26,895,998,464 text, 460,730,096 vision and 424,699,392 MTP. The full original checkpoint remains the reference; text-only deployment excludes unused vision/MTP explicitly. The public shipping Bonsai Q1 file is 3,803,452,480 bytes; verify the installed artifact hash and all deployment assets before the final comparison.

The pinned Prism Q1 layout stores 128 signs plus a half scale in 18 bytes. An optimistic text Q1 scenario is **3,792,459,776 bytes**; keeping recurrent `in_proj_a`/`in_proj_b` controls in F32 gives **3,883,513,856 bytes**, approximately **2.1% above Bonsai**. Both estimates include tensor alignment but exclude GGUF metadata/tokenizer and additional assets. Both also require one-bit embeddings and output head, which are not covered by the current GSQ/RCO integration. These are accounting scenarios, not packed models or evidence of answer quality. GSQ has a one-bit sign/scale quantizer, but optimizer-to-runtime packing and coverage still need validation. [Inventory](benchmarks/quantization-2026-09-19/encoding-inventory.json), [GSQ one-bit source](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/quantization/gumbel_quantizer_1bit.py), [pinned runtime layout](https://github.com/PrismML-Eng/llama.cpp/blob/62061f9/ggml/src/ggml-common.h).

## What the repository actually needs

The app already has deterministic routes and source-bound answers. Improving a language model cannot improve a turn that never invokes it. It also cannot fix Siri interpreting a question as a Contacts request before Zimfo runs.

The calibration unit must therefore be an **actual model invocation**, including its rendered instructions, tool definitions, selected passages, preceding turns and assistant prefix. A question string from a regression fixture is insufficient. Record the invocation purpose: tool selection, evidence selection, grounded generation, follow-up resolution or recovery. Count deterministic-only turns separately.

Evidence inspected:

- `ios/MCPZimChat/Chat/ChatSession.swift`: routing, transcript preparation, selected evidence and model registry.
- `ios/MCPZimChat/Providers/LlamaCppProvider.swift`: shipping runtime, prefill/cache behavior and model-specific settings.
- `ios/MCPZimChat/Providers/Gemma4Provider.swift`: existing MLX runtime.
- `ios/MCPZimEval/ProbeE2ECLI.swift`, `EvalHarness.swift`, and `eval/README.md`: model and application evaluation entry points.
- `tools/bonsai-ab/compare.sh` and `docs/BONSAI_MLX_VS_LLAMACPP.md`: existing runtime comparison. Historical results favor llama.cpp for the phone; they are not measurements of a new checkpoint.
- Existing conversation suites contain **25 conversations / 66 turns** in four files. Preserve these as a locked final regression set; create separate development cases.
- The locally parsed log corpus contains **43 conversations** and **93 recorded model prompt lengths**: minimum 385, median 810, maximum 7,330; 86 below 2,048, seven above 4,096. This is a biased, mixed-build sample, not a production traffic distribution. Full prompts cannot be reconstructed faithfully from these debug logs alone.

Aggregate inventory: [local-inventory.json](benchmarks/quantization-2026-09-19/local-inventory.json). No raw conversations or coordinates were copied into this document.

## What GSQ and RCO actually do

**GSQ** learns discrete weight assignments and group scales using a Gumbel-Softmax relaxation, with GPTQ initialization available. Its public implementation has binary, ternary and integer quantizers and supports a Qwen3.5/3.6 wrapper. It does not promise that arbitrary 27B models become good phone models at one bit. [GSQ paper](https://arxiv.org/abs/2604.18556), [repository](https://github.com/IST-DASLab/GSQ).

The reviewed trainer calls `calculate_mse`, which evaluates reference and quantized attention/MLP or block outputs and backpropagates reconstruction error. It carries hidden states forward between blocks. Thus per-linear `H = XᵀX` is useful for GPTQ initialization and linear reconstruction diagnostics, but **does not preserve the nonlinear block objective**. The reference path also has phase-specific handling of previously quantized attention; a new exporter must match that behavior explicitly. [Reference implementation](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/models/base.py), [training loop](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/trainer.py).

**RCO** chooses a precision option for each tensor/group under a budget. The implementation interpolates candidate weights and differentiates end-to-end loss with respect to allocation parameters. It needs a candidate database, token sequences, a scoring objective and a model forward/backward path. It cannot run from independent Hessians alone. CPU candidate deltas exist, but the model loader and search are CUDA-oriented; a full BF16 27B model cannot simply be placed on a 24 GB card. [RCO paper](https://arxiv.org/abs/2605.00649), [search code](https://github.com/IST-DASLab/RCO/blob/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/search/quant.py), [loader](https://github.com/IST-DASLab/RCO/blob/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/models.py).

For an isolated linear transform, with input rows X and weight error ΔW, reconstruction error is `tr(ΔW H ΔWᵀ)`. Store H as the **uncentered sum**, with sample count; normalization and damping belong to a documented solver step. Do not call it centered covariance or assume all solvers use the same scaling.

Two corrections to the proposed llama.cpp route:

- In GGML multiplication, `src[0]` holds weights and `src[1]` holds the input activation. Capture must honor tensor strides, device buffer transfer, and callback lifetime.
- Standard imatrix data records importance statistics; it is not a substitute for a full cross-feature Gram matrix or block training samples. The app currently consumes a binary XCFramework and has no imatrix collector wired into its provider. [Upstream collector](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp).

## Model and runtime selection

### What would establish feasibility before a long optimization?

The user wants evidence that the intended compression is achievable before committing substantial work. Treat the following as distinct decision gates; successful CUDA projection tests alone do not pass them:

1. **Size and representation, before paid compute:** inventory the exact original Qwen3.8 checkpoint and current Bonsai size anchor. Enumerate actual supported candidate encodings and their serialized costs for every tensor. Calculate the minimum supported size including mandatory overhead, and report how proposed budgets compare with Bonsai. For orientation, 27 billion weights at 1.58 bits already need approximately 5.33 GB before scales or metadata; a roughly 3.8 GB whole-model target allows only about 1.13 bits per nominal parameter. A uniformly ternary 27B model cannot meet this size target. We need a working predominantly one-bit or otherwise sufficiently compact representation, with embeddings/head included. Published 2.5-bpw GSQ/RCO artifacts are not evidence for this much tighter target. Establish optimizer-to-runtime packing/dequantization parity before spending on an unusable candidate grid. If the minimum supported size substantially exceeds Bonsai, report the mismatch before committing to long optimization; do not declare a slightly larger candidate a failure solely on bytes.
2. **Real GSQ candidate generation:** use original Qwen3.8 weights and the complete selected Zimfo calibration package. Run bounded initial optimization steps on actual full-attention, linear-attention and widest MLP blocks for the target encodings, validate the reconstruction objective/gradients and pack the resulting candidates. Measure memory and per-step time. Retain useful candidates and checkpoints. This is not a separate small calibration dataset or a full optimization merely to validate collection.
3. **Real whole-model RCO step:** once the exact candidate database exists, execute a complete 27B forward/backward and budget-constrained allocation update, including the output loss, every selected tensor and recurrent state, using bounded caches. Repeat enough steps to expose retention and record per-step time. Compare the streamed implementation's numerical behavior to its resident oracle where the latter fits. Failure, missing gradients, unbounded memory or impractical projected runtime is a stop/revise signal. The current synthetic harness cannot establish this result; candidate construction and the full adapter remain necessary work.
4. **Packed runtime and product check:** materialize whole-model allocations at explicit recorded byte budgets around the Bonsai size anchor and load it in the intended Zimfo backend. Verify packed-versus-optimizer output parity. Evaluate held-out grounded answers, tool calls, follow-ups, unsupported-answer handling, memory and latency against both original Qwen3.8 and current Bonsai. To claim the user's objective is achieved, the completed quantized model must improve on Bonsai at a size/quality tradeoff the user accepts. A short unfinished optimization cannot establish final achievable quality, and no hardware check can guarantee it.

Near-term recommendation: finish the size/encoding gate first, use the inexpensive CUDA harness to debug the necessary streaming primitive, then complete real GSQ/RCO execution gates before committing to the long optimization. Measure and report costs at each boundary. Do not keep adding infrastructure probes while presenting them as progress toward a demonstrated whole-model result.

| Candidate | Purpose | Constraint |
|---|---|---|
| Original Qwen3.8-27B BF16, exact revision pinned | Fixed full-precision baseline, teacher and quantization source | All reference data and optimization targets come from these original weights |
| Shipping Bonsai 27B Q1, existing pinned file/runtime | Existing-model quality comparison and exact deployed-size comparison anchor | Provides comparison results only, never teacher data or quantization source weights |
| Published Qwen3.8 GSQ/RCO GGUF | Ready-made method comparison on Mac | Smallest published release is 8.4 GB; not the target phone footprint |
| Published Ternary Bonsai 2 27B | Research context for achievable size/quality tradeoffs | Its 5.95 GB file exceeds our target; it is neither the baseline nor a successful deliverable |

Qwen3.8 uses the Qwen3.5 architectural family: 64 blocks with a mixture of Gated DeltaNet and full attention. This is substantially closer to our current runtime family than an unrelated coding model, but the exact model, tokenizer, template and kernel compatibility still need testing. [Official model card](https://huggingface.co/Qwen/Qwen3.8-27B).

The published GSQ/RCO Qwen3.8 release ranges from 8.4 GB at 2.50 whole-file bits/weight upward. Those publisher benchmarks do not measure our app, and the filenames are not a guarantee that every tensor has the named type. [Release card](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).

Bonsai 2 was announced September 17. Its model card lists approximately 5.95 GB dense-trit and 7.21 GB two-bit-slot packs and a required blockwise Hadamard activation transform. Our pinned `prism-b9591` framework must not be assumed to implement that newer representation. Its older `TQ1_0` enum is not proof of compatibility with `PTQ1_0`, nor are old and new Q2 layouts interchangeable. [Announcement](https://prismml.com/news/bonsai-2-27b), [model/format specification](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

**Go/no-go before training:** load a tiny packed candidate in the exact intended backend and compare dequantization and outputs against the optimization representation. If using a new runtime, isolate the upgrade and rerun existing Q1/Q2 model regression checks. Never replace the shipping framework merely to make an experimental file load.

The public GSQ/RCO repositories are not yet a verified reproduction recipe for the published mixed-GGUF release. Our source audit found separate GSQ compressed-tensors and RCO candidate/search paths. An open upstream issue identifies missing native-IQuant candidate generation, byte-cost modeling, allocation configuration and assembly tooling. Treat that report as corroborating evidence, not proof no implementation exists elsewhere. [Reproduction issue](https://github.com/IST-DASLab/GSQ/issues/9). Initial implementation must select a packing path we can inspect and test, not promise exact release reproduction.

## Calibration corpus

Collect **the complete selected calibration set, targeting roughly 60 Zimfo model-invocation examples**, in one Mac collection job. This is the intended working dataset, not a preliminary subset requiring another collection run. Some conversations produce several model invocations; actual counts must be recorded. Whether more examples improve quantization is a later empirical question, not an upfront requirement to generate hundreds more.

| Workload | Calibration examples | Include |
|---|---:|---|
| Grounded article discussion | 18 | Multi-turn history, source sections, concise evidence-based answers |
| Follow-up/reference resolution | 12 | Pronouns, topic changes, corrections, ambiguous dates/entities |
| Tool selection and structured arguments | 10 | Wikipedia, named-place search, nearby categories, route requests |
| Retrieved-data reasoning | 8 | Comparing places, selecting evidence, ranking candidates using supplied fields |
| Clarification and recovery | 8 | Ambiguous titles, empty results, stale source, missing coverage, unsupported current facts |
| General language robustness | 4 | Natural rephrasings, speech-recognition-style errors, instruction adherence |

These are proposed coverage quotas informed by known failures, not inferred production percentages. Include difficult unsupported-answer and wrong-subject cases, not only successful conversations. Use offline source material for answer provenance. Historical fine-tuning rows are useful scenario seeds, but their old Gemma template and sometimes synthetic tool payloads must not be copied as today's exact prompt.

Provisional length coverage: 30 short calls (roughly 384–1,536 tokens), 10 medium (1,536–4,096), 16 longer calls (4,096–8,192), and four stress cases up to the configured 16K app context. Adjust after capturing current invocations. Do not pad by repeating prose, truncate away a system/tool prefix, or concatenate unrelated conversations just to meet a token target.

Create an opt-in, evaluation-only **prompt capture boundary after template rendering and context trimming, before token evaluation**. Preserve exact rendered UTF-8 and token IDs when accessible, invocation purpose, template/tokenizer hashes, tools, source/archive edition and history. Record full logical prompts separately from newly-prefilled suffixes: prefix caching should not accidentally remove essential context from the calibration dataset. The supplied baseline must tokenize the captured prompt identically or the mismatch must be resolved first.

Separate three sets by conversation and subject/source family:

1. Calibration: the full selected set, targeting roughly 60 model invocations. Expand only if later held-out results justify additional data; no automatic 256/512-example phase.
2. Development: independent cases for hyperparameters, bit allocation and format decisions. Include targeted source-selection/citation failure checks.
3. Final regression: existing 66-turn suite plus newly frozen unrelated topics. Once used to select a candidate it becomes development data; refresh the final set before claiming an unbiased improvement.

Keep user transcripts local and excluded from Git. Use synthetic coordinates/places where private details add no calibration value. Deduplicate normalized text and source/topic families across splits. Preserve necessary references when replacing names. No remote model service is required for corpus construction.

## Execution architecture

### Mac collection, then native CUDA layer-streamed optimization

The public GSQ code already materializes one block at a time and supports activation-cache offloading. Reuse that structure on the 3090 and add an adapter for the Mac-exported Zimfo package rather than porting training to MLX. Original BF16 weights remain authoritative on disk; only the working block and microbatch need to fit the GPU. The package includes exact prompts/token IDs, reviewed scoring targets and loss masks, baseline/config/tokenizer identities, and the validated reference block samples. Full-precision source weights must also be available on the workstation, transferred or obtained by identical pinned revision and verified hashes. Activations alone cannot be quantized into model weights.

On the Mac, normal conversation execution may generate responses and subsequent tool calls. The separate reference-capture pass replays recorded tokens through the full-precision baseline without sampling further tokens. Keep generated responses distinct from reviewed desired outputs: an observed model mistake is useful as a scenario, not automatically a training target. Long collection runs should checkpoint completed examples, record failures and coverage, and obey a disk budget. A long run is not automatically representative; balance invocation types and preserve held-out cases.

On the 3090, exercise one full-attention block, one linear-attention block and the widest MLP in implementation memory checks. Use a microbatch of one and bounded CPU/disk activation buffers. Measure actual peak allocated/reserved VRAM and host memory before scaling. Aim below 20 GiB of the card's 24 GiB during these checks, leaving room for allocator and kernel workspaces. That is a proposed operating ceiling, not a demonstrated result. CPU RAM, free disk, driver and GPU availability on the workstation have not been inspected in this session.

For each block/phase, retain the input sequences, valid-token mask, positional/masking inputs and reference behavior required by the native solver. Keep only current/next activation buffers unless a measured reason justifies a teacher cache. Recompute later-block inputs as required by the sequential quantization method. Teacher activations from the unquantized path must remain clearly distinguished from inputs propagated through a partially quantized prefix.

This is still **prefill-only calibration**: replay token sequences with teacher forcing and do not sample new tokens. For answer-only scoring, reviewed assistant answers or tool-call tokens can be supplied in the input sequence with a loss mask; teacher-forced forward passes do not require autoregressive generation. Ordinary app acceptance tests will generate answers separately.

### Full-precision reference capture on the Mac

Use full-precision MLX on the 128 GB Mac for reference evaluation and teacher/block-cache export. Collect the full selected set, with schema, precision, invocation/layer coverage and checksum checks during the run. Validate numerical parity between MLX and PyTorch on identical token IDs and representative blocks, including recurrent attention, as an implementation test using the resulting package. This does not require a completed optimization or gate the remaining Mac collection behind a small export. Preserve resumable completed examples if a check exposes a defect.

Export **block-input samples plus explicit metadata**, optionally selected per-linear H diagnostics. Materialize MLX arrays at each bounded stage, release references, clear disposable caches and observe process footprint as well as MLX allocation metrics. A Metal allocator limit is not a hard total-process/Jetsam guarantee. Never keep every layer's lazy activation graph in a Python list. Do not promise that a static reference cache eliminates all later forward passes: the 3090 must recompute candidate-dependent or quantized-prefix activations when the solver requires them. This does not require another interactive phone collection session.

### Architecture coverage

Enumerate actual weights from the pinned checkpoint; do not assume seven similarly named projections in every block. Qwen hybrid blocks include fused `in_proj_qkv`, `in_proj_z`, `out_proj` and recurrent-control projections, convolution and state parameters. Full-attention blocks have separate q/k/v/o projections; MLPs have gate/up/down. The public Qwen GSQ wrapper treats some recurrent-path parameters separately and initializes some attention projections differently. Preserve and audit those choices. [Wrapper source](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/models/qwen35.py).

Embedding and LM-head storage must be included. The standard RCO layer finder excludes them. With Qwen's vocabulary and hidden width, leaving both untied matrices in BF16 costs roughly 5 GB before the transformer blocks—already over the current phone weight footprint. A low-bit transformer-only result is not a phone-sized complete model. Protect normalization and sensitive recurrent state first, and account for every byte; only compress additional components after separate sensitivity and packing tests.

### RCO comes after a working GSQ candidate

First compare uniform candidates and controlled mixed-precision candidates at the same serialized size. RCO can then search over those validated options with a cost table derived from actual packing: codes, scales, zeros, alignment, unchanged tensors and file metadata.

Full differentiable RCO on 27B requires explicit base-weight/candidate offloading and backward-memory engineering on a single 3090. A flag for CPU deltas alone does not prove feasibility. A Mac port also requires removing CUDA assumptions and validating autograd. Use short numerical and memory implementation checks before committing to a long 27B search; these do not gate full Mac dataset collection.

#### Bounded-memory RCO implementation options

Established training mechanisms exist; their availability does not establish a ready-made 27B RCO recipe:

- [PyTorch saved-tensor hooks and `save_on_cpu`](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) move tensors saved for backward to CPU or custom storage and restore them when needed. Combine with block recomputation and explicit weight residency management. Saved-tensor offload alone does not move persistent model weights or candidate buffers, and moving every saved tensor into RAM is not a bounded host-memory design.
- [DeepSpeed ZeRO-3 / ZeRO-Infinity](https://deepspeed.readthedocs.io/en/latest/zero3.html) supports parameter offload to CPU/NVMe during forward and backward, with live-parameter and prefetch limits. This is a genuine training-capable foundation, unlike relying on an inference dispatch map. However RCO's registered candidate buffers / plain CPU lists are not automatically managed as ZeRO parameters; integration must cover those explicitly. Optimizer-state-only offload offers little benefit because the trainable allocation logits are small.

Recommended engineering design: retain RCO's whole-model loss, Gumbel straight-through estimator, budget solver and optimizer, but replace eager weight interpolation with a streamed execution adapter. Store the exact candidate database on disk with a bounded host cache. For one calibration microbatch, traverse blocks in the forward direction, writing block-boundary activations to bounded CPU/disk storage. Traverse backwards, reload and recompute one block, calculate its allocation-logit gradients and the gradient for the preceding block, then release that block. This is whole-model backpropagation with recomputation, not independent block-loss optimization. GPU working memory should depend primarily on the largest live block/microbatch, head/loss buffers and bounded staging, rather than all blocks' weights and candidate differences. Actual fit still requires measurement, including attention/recurrent intermediates and allocator overhead.

Candidate interpolation also needs a custom backward: stream candidate differences or tiles and contract each with the local weight gradient, accumulating the small probability gradients. Do not let ordinary autograd retain every GPU candidate difference until backward. Preserve the upstream straight-through derivative even for candidates with zero hard forward probability. Keep allocation logits, sampled Gumbel noise, masks, position data and any recurrent state consistent across forward/recomputation; do not update logits until the complete intended gradient accumulation finishes. Chunk reference-cache reads and the output-head/loss calculation while preserving full-vocabulary normalization and the same token weighting.

Acceptance tests compare resident and streamed implementations under identical inputs/assignments: loss, hidden-state gradients, every allocation-logit gradient, one optimizer/retraction update and resulting budget feasibility, within explicit numerical tolerances. Cover full-attention, linear-attention and MLP blocks, and fail on missing gradients. Memory tests check peak live GPU tensors across increasing block counts at fixed microbatch, bound host-cache occupancy, and verify repeated iterations release storage. These are implementation tests, not a replacement for the agreed full Mac calibration collection or a preliminary long quantization run. Streaming trades memory for recomputation and PCIe/disk traffic; it supplies an implementable route, not an established runtime estimate.

If this is impractical on the available hardware, use measured layer sensitivity and discrete budget allocation as a **separate approximate baseline**, followed by whole-model evaluation. Do not label that algorithm RCO. A larger-GPU run is an optional later resource decision, not something this plan provisions automatically.

## Memory, storage and artifact contract

Verified local RAM: 128 GiB. Free local disk at inspection: approximately 157 GiB. Cache directory names for Qwen models exist, but the inspected Qwen3.8 entries did not establish a complete local BF16 checkpoint. Do not assume weights are already downloaded.

For 60 × 4,096 tokens and hidden width 5,120, one BF16 block-input buffer is **2.34 GiB**; two rolling buffers are 4.69 GiB. Saving all 64 such buffers would consume 150 GiB before weights. A single FP32 Gram matrix for width 17,408 is approximately **1.13 GiB**; all 64 down-projection Grams alone would consume over 72 GiB. Raw per-projection storage is larger still. Compute an exact inventory from the checkpoint before any export.

Recommendation: use block streaming and an output volume with at least **250 GiB free for a limited 27B experiment**, preferably 500 GiB if retaining several candidates. This is planning headroom, not a fixed solver requirement. A complete multi-bit fake-quant database may exceed it; budget that separately. Do not delete existing user data to make room.

Every complete artifact needs:

- Original model ID, immutable revision, weight/config/tokenizer hashes and precision provenance; reject quantized configurations/packed integer weights. FP16 dtype alone cannot prove a checkpoint was never dequantized.
- Runtime/solver commits, architecture, exact source and runtime layer-name mapping, all targeted and intentionally excluded tensors.
- Prompt/token hashes, sequence lengths, masks, ordering, chat-template version, thinking mode and dataset split/provenance.
- Tensor shapes, storage and accumulation dtypes, byte order, offsets, checksums, token/sample counts and statistic definition.
- Candidate grid, group size, scales/zero representation, initialization, damping, seeds, optimizer settings and actual serialized byte cost.
- Measured peak GPU/host memory, elapsed time and disk footprint. Write an incomplete manifest before work and mark complete only after coverage/checksum validation. Interrupted runs never masquerade as completed calibration.

Prefer native BF16 safetensors for block samples. FP32 accumulation is required for H. If a consumer cannot read BF16 and needs FP32 raw inputs, account for doubled storage explicitly; converting BF16 to FP16 can overflow values and is not an automatically lossless storage change.

## Tests and acceptance gates

1. **Capture fidelity:** a tiny full-precision model produces the same outputs with hooks enabled/disabled; compare captured inputs directly against PyTorch references. Quantized checkpoint rejection, fused-layer coverage, mask/padding exclusion, fresh state per independent prompt, truncation refusal and failed-run status are mandatory tests.
2. **Statistics:** compare streaming H against direct `X.T @ X`; exercise batch partitions, zero-variance features, nonfinite values and dtype conversion. Check exact feature dimension and orientation.
3. **Quantizer:** repeatable seeded pilot; gradients reach assignments/scales; hard candidate improves held-out reconstruction over initialization. Include real full-attention, linear-attention and widest-MLP shapes. A toy success alone does not establish 27B memory feasibility.
4. **Packing:** codes/scales round-trip through the exact GGUF type without an unintended second quantization. Compare dequantized weights and logits; verify exact group layout, dimensions, rotation metadata and whole-file bytes. Reject unsupported formats.
5. **Quality:** measure model-only tool JSON validity, tool/argument selection, source identity, unsupported-answer behavior and grounded follow-ups. Track full app results separately so deterministic routes cannot hide model regressions. Teacher accuracy and compression retention are separate metrics.
6. **Performance:** paired same-source/seed/sampler runs, cold and warm prefill, decode speed, TTFT, peak footprint and repeated long conversations. Treat KV precision as a separate ablation; hold it fixed during weight comparisons.
7. **Device:** signed Mac/iPhone gate from `docs/SIGNED_APP_BUILDS.md`, then offline use with full archives, model load/unload, speech playback and extended conversations. No new wrong-source/unsupported-fact regressions in the locked cases. Proposed promotion target: improve the model-dependent development score, preserve final regression outcomes, and keep p95 TTFT within 10% of the shipping control unless a consciously chosen quality tradeoff justifies it. No app crash/Jetsam in the stress run.

## Adversarial review of this plan

| Failure mode | Required response |
|---|---|
| Better generic perplexity but worse source fidelity | Select using grounded Zimfo cases and explicit wrong-source counts |
| Apparent gain comes from upgrading Qwen3.6 to Qwen3.8 | Compare both quantized and BF16 versions of the same new base; keep shipping control separately |
| Calibration duplicates the tests | Split by conversation, topic and source; freeze a separate final set |
| Most measured successes bypass the model | Require model-invocation traces and a model-only score |
| Full-precision tensors came from a low-bit checkpoint | Check provenance and file hashes in addition to dtype |
| Static H loses nonlinear/sequential behavior | Feed native GSQ block samples and recompute inputs as required |
| Hybrid/fused projections are silently missed | Inventory every weight and assert exact capture/quantization coverage |
| Nominal 1.58 bits is actually two-bit packing plus large FP16 heads | Enforce whole-file cost and device working-set gates |
| Single-GPU RCO assumes all weights fit | Verify gradient equivalence, then target-shape backward/offload peaks before the long search |
| Optimized weights are requantized during conversion | Preserve integer codes/scales and test packing round-trip |
| New Bonsai format loads with the wrong transform | Fail closed on unsupported rotation/packing; use isolated backend upgrade |
| Better model claimed to fix Siri Contacts routing | Keep system intent routing outside the quantization success criteria |

## Ordered work packages

### Early feasibility checks and stop conditions

These checks answer specific implementation questions without running a preliminary long optimization or collecting a separate small dataset. Full Zimfo Mac collection remains the agreed data workflow.

| Check | Evidence required | Status / consequence |
| --- | --- | --- |
| Streamed candidate derivative | Dense-autograd and finite-difference parity, including zero forward-probability candidates and gradients through earlier layers | **Passed CPU probe:** 9 tests passed; 1 Metal test skipped because Metal was unavailable to the executing environment |
| Reference/storage contract | Exact candidate tensors, dtype and metadata validated; disk cache reproduces identical values with bounded occupancy | Not implemented; do not claim host memory is bounded by the current probe |
| Whole-block recomputation | Full-attention and linear-attention blocks match resident loss and input/allocation gradients, with fixed random choices and recurrent state | Pending; a linear-layer probe does not establish hybrid-model correctness |
| RCO update parity | Upstream budget DP, temperature, gradient accumulation, projection/retraction and one complete optimizer update match with streamed execution | Pending; the probe verifies ordinary Adam parity only |
| 3090 capacity | Read-only GPU/driver, RAM and disk inventory; run exact largest-block shapes in BF16 at actual prompt lengths, including the observed 7,330-token tail if retained | Pending workstation access; target less than 20 GiB GPU allocated/reserved and explicit host-cache headroom |
| Memory lifetime | Repeated forward/backward steps and increasing block counts do not accumulate live GPU tensors; bounded staging and host cache; no disk-cache corruption after interruption | Pending full adapter; reduce tile/microbatch/prefetch bounds or fix retention on failure |
| Exact output loss | Chunked full-vocabulary KL and its gradients match dense loss, with identical masks and weighting | Pending; do not silently substitute top-k KL to obtain a fit |
| End-to-end readiness | All above pass; full package validated; actual device and file-format cost gates met | Only then authorize the implementation to launch the expensive full optimization under the user's existing scope |

Executed September 19: `PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python -m pytest tools/calibration/tests/test_streamed_linear_probe.py -q` → **9 passed, 1 skipped in 12.65 seconds**, using PyTorch 2.8.0. The [probe and tests](../tools/calibration/README.md) compare two connected nonlinear layers at three tile sizes, soft versus hard straight-through choices, loss, input gradients, allocation-logit gradients and an Adam step. A separate finite-difference gradcheck and an explicit zero-probability candidate check passed. This establishes the first-order streamed interpolation calculation on CPU; it does not establish full-model correctness, GPU capacity, BF16 numerical behavior, convergence or model quality.

Next implementation order: disk-backed candidate reads and block recomputation; resident-versus-streamed hybrid-block/update parity; exact head/loss chunking; target-shape CUDA memory and transfer-rate checks. Confirm workstation access in parallel. If a real 27B-shaped block cannot fit after tiling and recomputation, or transfer costs are impractical, revise the execution backend/hardware plan before the expensive run. Do not substitute an approximate search while labeling it RCO.

Workstation access check: the user identified **PC Gaming**. The saved `pcgaming` SSH alias resolves to `pcgaming.local` and an SSH server responded on September 19. The saved `jason` login failed in noninteractive mode with `Permission denied (publickey,password,keyboard-interactive)`. The host is reachable; authentication is the immediate blocker. No remote hardware inventory, file transfer, installation or GPU job occurred. Obtain a working saved SSH authentication method without putting credentials into this repository or chat logs.

### GCP Spot alternative

The user suggested GCP Spot as an alternative to PC Gaming. Read-only checks on September 19 confirmed this Mac has a working authenticated `gcloud` installation and can list projects. The existing `tiltastech-zimfo` project has billing enabled, but Compute Engine is disabled, so its regional GPU quota could not be inspected. The existing `mloptimization` project also did not list Compute Engine as enabled. No API was enabled, default project changed, quota requested, resource created or data uploaded.

Subsequent authorized action: the user asked to enable Compute Engine on Zimfo. `gcloud services enable compute.googleapis.com --project=tiltastech-zimfo --quiet` completed successfully on September 19. Compute project and regional reads now succeed. The initial disabled-API observation above is historical. No VM or GPU was launched.

Post-enablement quota inventory: `GPUS_ALL_REGIONS` is **0** (usage 0). Both `us-central1` and `us-west1` report **1** standard L4 and **1** preemptible L4, but **0** A100 40 GB and **0** A100 80 GB quotas, including their preemptible variants. The global limit is the immediate GPU prerequisite: request an increase to at least **1** before an L4 launch. [GCP allocation quota documentation](https://docs.cloud.google.com/compute/resource-usage) requires the global quota in addition to the GPU-specific regional quota. GPU machine families such as G2 use their GPU quota without a separate CPU-quota request. Quota is not a guarantee of Spot capacity. No quota-increase request has been submitted.

Latest status, September 19 at 19:28 UTC: at the user's explicit request, enabled the Cloud Quotas API and submitted preference `zimfo-global-gpu-one` for `GPUS-ALL-REGIONS-per-project`, requesting exactly 1. Google approved it: `grantedValue: 1`, `preferredValue: 1`, `stateDetail: Quota request approved to 1`. This supersedes the initial zero-quota inventory above. The request trace is `4f196b1f-9824-4206-971c-bb5766fccc2e`. No regional quota changes, VM creation or GPU rental occurred. L4 Spot capacity, current pricing and the runnable CUDA test package remain launch prerequisites.

Prepare the local CUDA test harness before starting paid compute. Proposed first rental is a single **24 GB L4**, for example `g2-standard-32` with 128 GB host RAM if required by the measured working set. Its GPU memory capacity is useful for testing the 3090-targeted residency design, but L4 kernel support, speed, bandwidth and allocator behavior do not establish identical RTX 3090 results. A smaller G2 host may suffice for isolated block checks once the host-cache bound is measured. For a larger-memory execution alternative, `a2-ultragpu-1g` provides one **80 GB A100**, 170 GB host RAM and 375 GiB Local SSD. These are documented machine configurations, not verified project quota or Spot capacity. [GCP GPU machine types](https://docs.cloud.google.com/compute/docs/gpus).

Before a paid launch, resolve project/API enablement and GPU quota, select a zone with Spot capacity, obtain the current whole-VM price including disk, and agree the spending ceiling. Use an explicit maximum run duration with termination, bounded retries, automatic deletion of scratch resources and external preservation of small test results. A budget alert alone is not a spending cap. Spot preemption can interrupt work; write results/checkpoints atomically and make reruns idempotent. Do not rely on a shutdown hook as the sole means of preserving results. [Spot VM documentation](https://docs.cloud.google.com/compute/docs/instances/spot).

Prepared implementation: `tools/calibration/gcp_preflight.py` defaults to building a local allowlisted archive and printing the launch command. `--execute` is the separate paid action, not run in this session. It requests one `g2-standard-8` in `us-central1-a`, the verified public image `common-cu129-ubuntu-2204-nvidia-580-v20260909`, one 100 GiB auto-delete balanced boot disk, no VM service account, SPOT provisioning, no restart, and API-side DELETE after one hour. No fallback VM or retry loop creates additional rentals. The guest installs pinned PyTorch 2.8.0 CUDA 12.8 wheels and runs the synthetic target profile under a 40-minute command timeout. The client retrieves small results, verifies CUDA/BF16/target coverage, deletes only its labeled VM, and verifies the boot disk is absent. Preemption can lose results; a missing report is failure, never success. [Harness runbook and limitations](../tools/calibration/README.md).

**Likely first-run price, September 19:** Google's directly fetched public Iowa Spot table lists `g2-standard-8` at **$0.512112/hour**, including the L4, 8 vCPUs and 32 GiB RAM. An older indexed copy lists $0.486396; use the more recent direct value for planning and recheck before launch. Add **$0.0136986/hour** for a 100 GiB balanced disk and **$0.0025/hour** for the ephemeral Spot IPv4 address: about **$0.53 for a full hour**, plus small result egress and any applicable tax. Plan **under $1** for one successful hour-limited synthetic preflight; a proposed **$2 approval envelope** is contingency, not a cloud-enforced dollar cap. Setup time is included in the hour. No instance has been launched. Sources: [Spot whole-VM prices](https://cloud.google.com/spot-vms/pricing), [balanced disk prices](https://cloud.google.com/compute/disks-image-pricing?hl=en), [IPv4/network prices](https://cloud.google.com/vpc/network-pricing). This is not an estimate for full GSQ candidate creation or whole-model RCO; that larger run needs its own measured schedule and storage budget.

Latest local checks: **20 tests passed, 1 Metal test skipped**, plus a completed CPU tiny-profile JSON report and CPU BF16 parity (relative L2 errors about 0.28% for input gradients and 0.29% for allocation gradients). These results supersede the earlier nine-test count. CUDA remains untested. The manual adversarial review is recorded in `docs/QUANTIZATION_PREFLIGHT_REVIEW_2026-09-19.md`.

### Published timing evidence

The [RCO paper, Appendix C](https://arxiv.org/html/2605.00649v1#A3), reports RTX 3090 allocation-search runs on **Qwen3-8B**, using 256 sequences of 2,048 tokens: **23, 62, 114 and 218 minutes** as Gumbel samples per step increase from 1 to 16 at 200 steps. These are search timings over already prepared quantized candidates, not a full GSQ-plus-RCO build. The main comparison table uses an RTX A6000; do not misattribute that table to the 3090.

The [GSQ paper, Appendix A.4](https://arxiv.org/html/2604.18556v1#A1.SS4), reports **10 hours for Llama-3.1-8B and 68 hours for Llama-3.1-70B on eight H200 GPUs**. Its Llama recipe uses 4,096 sequences of 4,096 tokens and 20 block-training epochs—far more data than our selected set. No directly comparable Qwen3.8-27B GSQ runtime on one 3090 was found in the inspected primary sources.

These measurements establish that search and candidate generation have different costs; they do not justify an exact 3090 ETA for our full job. Token volume, epochs, precision candidates, memory/offload behavior and architecture all differ. Measure per-block elapsed time during the actual full-data optimization, then project remaining full-attention/linear-attention blocks separately and add search/export time. This is progress estimation within the real run, not a new preliminary dataset or completed pilot optimization.

### Larger-model RCO on RTX 3090: evidence audit

Research on September 19 checked the RCO paper and repository, the GSQ reproduction issue, release discussion, model cards, and targeted searches for 27B/32B/70B RCO optimization on single and multiple RTX 3090s. **No verified successful larger-than-8B mixed-precision RCO optimization run on RTX 3090 hardware was found.** This is a limit of the evidence located, not proof that such a run is impossible.

The [27B release discussion](https://www.reddit.com/r/LocalLLaMA/comments/1w13vse/release_sota_ggufs_for_qwen3827b_gsqrco_at_25_to/) contains consumer-GPU inference reports and a question about reproducing optimization on a 96 GB RTX 6000 Blackwell. Neither supplies a completed 3090 optimization recipe. The [Lightje derivative card](https://huggingface.co/Lightje/Qwen3.8-27B-DavidAU-Turbo-GSQ-RCO) describes replicating the published allocation structure and an importance matrix collected on H100; its 24 GB compatibility claim concerns inference. It does not establish a fresh RCO search on a 3090. The [upstream reproduction issue](https://github.com/IST-DASLab/GSQ/issues/9) still identifies missing release search/configuration and candidate-generation details.

Keep 27B RCO on one 24 GB card marked **unverified**. Approximately 54 GB of BF16 base weights alone require offloading, before candidate deltas and backward activations. The public code's CPU-delta and device-placement options are possible building blocks, not evidence of a working 27B backward pass under that memory cap. Verify offload/autograd correctness and actual peak memory as implementation checks; do not replace the agreed full Mac collection with a separate pilot dataset or infer impossibility from the absence of reports.

### Concrete RCO memory hazards

The inspected [search entry point](https://github.com/IST-DASLab/RCO/blob/9a1e09c07d468109cbe60a1b87d5036034a79d10/rco_search_quant.py) and [weight interpolation code](https://github.com/IST-DASLab/RCO/blob/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/search/quant.py) expose more useful constraints than the absence of community reports:

- Search loads BF16 model storage and holds BF16 candidate differences for each searched tensor. With K candidate precisions and P searched parameters, persistent differences alone occupy approximately `2 * (K - 1) * P` bytes. Seven candidates over a hypothetical 27 billion searched parameters would mean 324 GB of differences before base weights and activations; actual P excludes unsearched tensors. This is an illustrative storage calculation, not a measured minimum or a claim that all of it must be in VRAM.
- `--cpu-deltas` moves persistent differences to host RAM and transfers them during weight interpolation. This trades VRAM for RAM and PCIe traffic. Differentiable interpolation may retain transferred tensors for backward; verify saved-tensor lifetimes under the chosen checkpointing/offload implementation. The current code skips its gradient sanity check in CPU-delta mode, so that flag cannot establish correct, bounded-memory differentiable RCO by itself.
- `--gradient-checkpointing` is available. Model-loading memory caps in the audited entry point are applied only when more than one GPU is visible. Fix or explicitly control the single-GPU placement path. A successful inference/offload forward pass is insufficient evidence that the full search backward pass fits.
- The [paper's Appendix C.9](https://arxiv.org/html/2605.00649v1#A3.SS9) reports approximately **159 GB host RAM** for cached reference log-probabilities at 256 calibration sequences. Our exact dense cache size is approximately `2 * scored_tokens * vocabulary_size` bytes for FP16, plus overhead. A bounded disk-backed cache can preserve the same objective; truncating to top-k probabilities changes it and needs separate numerical validation.

Using about 50–60 conversations reduces total token/cache volume and collection time; it does not reduce model storage or the candidate differences required per tensor. Confirm the 3090 workstation's host RAM and disk capacity, then budget base weights, differences, live backward tensors and reference-cache buffers independently. The evidence establishes real memory risks, but does not supply a validated 27B minimum-VRAM figure for a correctly streamed implementation.

1. **Baseline and prompt inventory:** pin original full-precision Qwen3.8-27B model/tokenizer/runtime artifacts, capture actual current-app invocations, freeze data splits, and measure its task-quality headroom against current Bonsai. Report lack of headroom as a project risk; do not silently replace the user-selected baseline.
2. **Format feasibility:** prove packing/load/output parity for one representative tensor and a runnable tiny checkpoint. Select a byte-realistic candidate grid for the phone deployment; use the Mac to measure the full-precision reference and experimental candidates.
3. **Full Mac collection and transfer:** implement exact Mac model-input recording and full-precision reference export; run the complete selected calibration set, validate the written artifacts, and transfer the complete package to the 3090. Perform inexpensive import/numerical/memory checks before the long optimization. No separate small-dataset capture-to-optimization phase. Stream exports within a planned disk budget rather than accumulating every projection's raw activations.
4. **First whole-model candidate:** quantize the original Qwen3.8-27B weights using a byte-feasible candidate grid. Evaluate on the Mac against full-precision Qwen and current Bonsai. Report its loss versus full-precision Qwen, gain versus Bonsai, and exact deployed size. Explore budgets around Bonsai to measure the quality/size frontier; a modest size increase is eligible for consideration, not automatic acceptance.
5. **Budget optimization:** add verified candidate costs and evaluate RCO feasibility. Retain a clearly named sensitivity-allocation fallback.
6. **Application/device acceptance:** full suites, held-out cases, memory/latency/offline/voice stress tests, documented result and rollback to the unchanged shipping model.

Full Mac collection is a modest-data job, not a planned multi-day run. Completion within an hour is an unmeasured target, not a guarantee; record actual invocation count, token volume, generation time and export overhead. The separate long-running 3090 optimization estimate requires measured solver throughput and memory. No full-precision model download, calibration pass, CUDA optimization, model installation or upload has run as part of this planning turn.

Reviewed source revisions: GSQ `03fc16484c369e3127225615d5e03e8d3a6043e3`; RCO `9a1e09c07d468109cbe60a1b87d5036034a79d10`. Upstream checkouts were read in temporary directories; no upstream code was copied into the app.

## Implementation started: model-input boundary and comparison (September 19)

- Added opt-in `--capture-model-inputs NEW_DIRECTORY` to the headless Mac discussion evaluator. Both llama.cpp and MLX record exact full rendered UTF-8, SHA-256, token IDs, effective sampler settings and stop strings before selecting a cache suffix/prefilling. CLI associates records with conversation and zero-based turn; several invocations may occur per turn. Deterministic-only turns produce no record.
- Capture is disabled by default and never activated by the shipped app. A fresh directory is required, records are atomic, and capture errors remain sticky so the chat layer cannot turn a partial capture into a successful manifest. A stopped/crashed run stays marked running, not completed. Completed means input capture completed, not valid BF16 calibration or successful answers.
- Captures currently carry source/model paths and template identity, explicitly marked as awaiting provenance verification. Artifact/source hashes, tokenizer/template hashes, build revision, invocation-purpose classification, raw logical messages and full-precision replay/activation export still need integration. Captured token IDs belong to their originating model; Bonsai prompts must be rendered/tokenized correctly for Qwen and never treated as Qwen activations.
- `quality_frontier.py` compares normalized scored runs with matching frozen protocol and case keys, conversation-weighted scores, category deltas, failure counts, complete model bytes and performance coverage. It rejects missing/duplicate/skipped cases, NaN, mismatched sources/inputs, deterministic results disguised as model-only and non-original/quantized references. It reports size increases rather than applying a hard Bonsai cutoff.
- Existing Mac MLX CLI setup still assumes Bonsai. Do not invoke it with a Qwen repo and call the resulting settings/provenance a valid Qwen baseline; that adapter and pinned full-precision loading remain next work.

Pinned Qwen configuration/tokenizer downloaded locally; hashes are recorded in [baseline-metadata-manifest.json](benchmarks/quantization-2026-09-19/baseline-metadata-manifest.json). Its config declares `qwen3_5` architecture and BF16 text weights. This is metadata only: the full weight shards have not been downloaded or run. The installed Bonsai file was verified at 3,803,452,480 bytes and its SHA-256 matches the published original: [local artifact verification](benchmarks/quantization-2026-09-19/bonsai-local-artifact.json). Additional required deployment assets still need inventory.

Real Mac capture integration completed using the installed Bonsai and local Wikipedia: two ad-hoc public-topic turns produced two model invocations (467 and 1,584 tokens), both associated with the second turn; the first turn was deterministic preparation. All captured prompt hashes verified, the manifest completed, and both turn answers were present. This is an implementation check, not a preliminary quantization/calibration subset or a scored quality comparison. Raw inputs/answers remain ignored locally. [Aggregate evidence](benchmarks/quantization-2026-09-19/mac-capture-integration.json).

## Current execution update

All 18 original Qwen weight shards are now downloaded and verified against
pinned public SHA-256 values and the exact BF16 tensor inventory. The first
original-BF16 Mac text load/generation check passed using a captured app prompt.
This supersedes the earlier preparation-only/download-pending notes above.
Full calibration/export and paired quality evaluation have not run. See the
[current readiness evidence](QUANTIZATION_GPU_READINESS.md#evidence-from-this-readiness-pass).

## Measured full-run ETA gate

Before a long paid run, report duration and cost using production measurements
for the complete recipe, including checkpoints, packing and Spot recovery.
Track coverage and uncertainty; smoke wall totals alone are insufficient. See
[the measured timing report](QUANTIZATION_TIMING_2026-09-19.md). The second GPU
attempt failed recovery equivalence before production; no full-run ETA is yet
validated.
