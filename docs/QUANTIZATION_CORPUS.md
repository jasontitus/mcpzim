# Zimfo calibration development corpus

`eval/quantization_calibration_v1.json` supplies public, synthetic app conversations for full-precision Qwen3.8-27B calibration. It is development data, not a quality benchmark. It imitates app interaction types without copying private transcripts, device coordinates, or answer content from the final comparison cases.

The corpus has **25 conversations / 85 user turns**: **23 conversations / 79 turns** need only Wikipedia, and **2 conversations / 6 turns** additionally require a verified StreetZIM archive. Every conversation starts fresh; turns inside it share discussion and clarification state.

## Coverage and invocation accounting

| Primary coverage tag | Conversations | User turns | Desired actual model-call coverage |
|---|---:|---:|---:|
| `article` | 6 | 24 | approximately 18 |
| `followup` | 4 | 16 | approximately 12 |
| `retrieval` | 4 | 12 | approximately 8 |
| `recovery` | 4 | 12 | approximately 8 |
| `robustness` | 2 | 6 | approximately 4 |
| `tool` | 5 | 15 | approximately 10 |

The last column is an aspiration, not a measured count or a quota already met. Prepared-discussion setup, clarification, and map routes may bypass the model. A turn may also invoke the model several times. The 79 Wikipedia turns deliberately leave room for deterministic preparation while aiming for roughly 60 actual calls. Tool scenarios may produce few or no router calls; do not count a deterministic tool execution as a captured model call or manufacture tool payloads to fill a quota.

Capture the entire selected corpus first. Count completed invocations from recorder artifacts, classify their actual prompt purposes, and report uncovered categories. If tool-routing or long-context coverage is absent, add public development scenarios exercising that real app path and record the corpus revision. Do not pad prompts with unrelated prose or repurpose comparison cases to fill a calibration quota. Keep natural prompt lengths, including short turns and any long retrieved contexts; do not force everything to 2,048 tokens.

## Separation from comparison data

The following existing suites remain the held-out comparison set, totaling 25 conversations and 66 user turns:

- `eval/conversational_qa_v1.json`
- `eval/event_followup_qa_v1.json`
- `eval/exploration_qa_v1.json`
- `eval/location_categories_qa_v1.json`

The calibration entities and intended source families exclude those suites' Russia/Putin, Mongolia/Mongols/Buddhism, Lithuania and its grand duchy, Bulgaria/NATO/Ukraine, Washington, Alamo, gravitational waves/LIGO, Apple, Santa Rosa/1906 earthquake, Seattle/San Francisco, 9 to 5, HP Garage, Salinas, and recorded nearby venue cases. The earlier printing-press probe is also excluded. Broad interaction types intentionally overlap: the purpose is to calibrate the tasks the app performs, not memorize evaluation facts.

No exact conversation IDs or user prompts overlap the four suites. This alone does **not** establish source-level independence: retrieval can find an unintended article. Before activation export, audit actual grounding titles, article identifiers, and substantive retrieved passages against comparison sources. Quarantine a leaking conversation and its follow-up state, replace it with another development topic, and report the change. Shared system instructions, schemas, and generic conversational wording are expected and are not factual answer leakage.

Do not use held-out answers or teacher traces as optimization inputs. Freeze the held-out set before judging candidates. Repeated candidate selection on these suites makes them a reused evaluation set; report that limitation and reserve fresh final device cases before making a strong generalization claim.

## Run procedure

1. Verify the local Wikipedia archive at `~/Downloads/wikipedia_en_all_nopic_2026-06.zim` and record its identity. Verify the original Qwen revision and unquantized weights. Use the real app prompt construction, tools, retrieval, conversation state, and model-specific template.
2. Run all 23 Wikipedia-only calibration conversations with a fresh capture directory and a JSON report. Do not reuse the historical MLX CLI's Bonsai identity or sampler for Qwen; the full-precision Qwen app backend must be qualified first. If an initial app-input capture uses Bonsai to drive conversations, label that trajectory producer explicitly; only a separate unquantized Qwen replay may supply calibration activations. Bonsai activations are never valid baseline activations.
3. If a local StreetZIM is verified to cover Sacramento and Fresno, run the two optional conversations with `--streetzim`. Their prompts name public cities, not a private origin. Record archive identity and whether the expected category/ordinal result exists. Otherwise record the cases as skipped, not passed or calibrated.
4. Validate and finalize every invocation record, then audit coverage, lengths, deterministic bypasses, errors, and source overlap. Preserve the complete captured prompts and model-generated trajectory for reproducibility. Do not splice outputs from different models into the same conversation without labeling that experiment.
5. Replay the approved captured calls through the original full-precision Qwen activation collector, or collect directly in that qualified backend. Preserve tokenizer/template identity, exact token IDs, effective truncation, layer coverage, dtype, and exported tensor/checkpoint provenance. Prompt capture alone is not activation collection and does not establish that the GSQ/RCO export contract is satisfied.
6. Separately run Qwen and Bonsai comparison suites without activation instrumentation. Use the same archive, cases, app settings and documented sampling policy. Preserve structured reports, answers, skipped/missing cases, and evidence. A closed-loop comparison allows later model inputs to diverge; use separately labeled frozen-evidence replay if isolating model quality from routing/retrieval differences.

List the corpus using the existing decoder, without loading a model:

```sh
DYLD_FRAMEWORK_PATH=ios/build-eval/Build/Products/Debug \
  ios/build-eval/Build/Products/Debug/MCPZimEvalCLI \
  --probe-discuss --suite eval/quantization_calibration_v1.json --list-suite
```

The run should select `--suite eval/quantization_calibration_v1.json`, the verified `--zim` path, the qualified original-Qwen backend, a new `--capture-model-inputs` directory and `--report-json` file. With no `--case`, all eligible cases run. Without `--streetzim`, the runner skips the optional map cases. Use `--case` only for retrying failed conversations; do not silently shrink the full collection into a small optimization pilot.

## Adversarial review and validation

- **Unsupported answer labels:** no answer anchors, section names, or expected tool payloads were invented. This is a calibration corpus. A generic harness report showing no expectation failures must not be presented as a measured quality pass.
- **Deterministic dominance:** first turns and map routes may bypass inference. The recorder's completed call count is the evidence; 85 turns does not mean 85 training examples.
- **Ambiguity ordering:** Mercury, Jaguar, and Crane replies name the desired meaning explicitly. They do not assume a stable ordinal ordering of search matches.
- **Missing evidence:** the deliberately fictitious title is labeled as fictitious in the next turn. It tests recovery, not permission to fabricate article contents. The echo question asks for an unavailable live count without asserting a factual answer.
- **Source drift:** ambiguity and cross-article follow-ups can retrieve unexpected sources. Post-capture family and passage audit is required; prompt-only overlap checks are insufficient.
- **Distribution bias:** the corpus covers discussion, biography, science, materials, art, retrieval, correction and map intents. It is intentionally synthetic and cannot claim to match measured production frequencies. Tool routing and long contexts need measurement after capture.
- **Unavailable maps:** the two StreetZIM cases remain conditional. Category scarcity or a missing second result is a genuine app condition to retain, not a reason to invent results.
- **Teacher identity:** full-precision Qwen supplies calibration activations. Bonsai supplies the comparison, and may be explicitly labeled as a trajectory producer only where needed to exercise the current app.

Validation completed: JSON field/type and nonempty-prompt checks; unique conversation IDs and prompts; zero exact ID/prompt overlap against all four comparison suites; successful `--list-suite` through the actual Swift QASuite decoder. No model workload was run to create this corpus, no article/source availability has yet been certified, and no quality score is claimed.

## Activation-export contract review (2026-09-19)

Read-only adversarial review of the initial `tools/calibration/mac_collect.py` implementation, conducted while the separate app-input run was active. Findings below describe the reviewed version; they are acceptance criteria for subsequent fixes, not a claim that the current implementation still has every issue. No full-model or GPU workload was launched for this review.

**The package can support replay, but it is not already GSQ/RCO optimization data.** Exact full prompt tokens, original pinned weights, architecture/configuration, tokenizer conventions, fresh-state masks/positions, and the complete first-block input are enough to regenerate the text model's block targets. GSQ still needs teacher block forwards and propagation through the evolving quantized prefix. Per-projection sampled X and diagonal second moments cannot substitute for the full nonlinear reconstruction objective, a full Gram matrix, or unsampled activations.

The complete final normalized hidden state plus the correctly mapped original output head can reconstruct full-vocabulary teacher scores in chunks. That only establishes the RCO teacher side after numerical parity validation; the student still needs differentiable full-model execution, candidate allocation, and backward/update. Chunked KL must preserve full-vocabulary normalization (for example, stable global log-sum-exp), the exact token mask, temperature and objective reduction. Independently normalizing each vocabulary chunk changes the objective. Tokens plus original weights remain the recovery path if stored MLX hidden states fail CUDA parity.

The reviewed collector correctly materializes each projection's samples and moments before saving, avoids full sequence-by-vocabulary logits, rejects repeated projection calls, rejects nonfinite input X, and requires all discovered projections to have exports. Its hybrid count of 497 is consistent with 48 linear-attention blocks (five attention projections plus three MLP projections), 16 full-attention blocks (four attention projections plus three MLP projections), and one output head. This is **linear projection** coverage, not every model parameter: embeddings, depthwise convolution, normalization and recurrent parameters also affect replay and must remain available from the original model. The complete first-block inputs do not by themselves calibrate embedding quantization because embedding lookup precedes that boundary; original token IDs and embedding weights must participate when testing a compressed embedding candidate.

Required checks before accepting an activation export:

1. Bind the baseline verification record to the model directory actually loaded, including current file identities/hashes and pinned configuration/tokenizer artifacts. A separate JSON saying `validated` does not prove an arbitrary `--model-dir` contains those same weights. Verify the pinned revision explicitly, and retain original-to-runtime weight mapping.
2. Check exact projection names, per-layer type and input/output shapes against the pinned architecture, not just `64 layers / 497 modules`. Validate X width against the corresponding weight input dimension. A same-count replacement can otherwise pass the count guard.
3. Verify contiguous unique input ordinals and expected schema, preserve source identity, and validate all exported hashes, tensor shapes/dtypes, token counts, and row-index bounds in a standalone consumer before CUDA startup. Missing or corrupt projection files, duplicate invocations, `running`/`failed` top-level status and truncated copies must fail closed. A completion manifest needs every required file to be independently verifiable.
4. Check the exported first and second moments themselves for finiteness. Finite X can still produce overflow in FP32 summation or squaring. Tests should include huge finite X, NaN/Inf, partial writes, disk-budget/reserve failure and duplicate interception.
5. Record collector source hash, MLX and mlx-lm versions, model configuration, and sanitizer semantics. The inspected Qwen backend shifts particular normalization weights by `+1` and transposes convolution weights during import. CUDA parity must account for these conventions and BF16 rounding, not only compare tokenization or file hashes.
6. Enforce a memory limit/headroom policy before loading and before the longest prefills. `mx.eval` at projection boundaries bounds lazy-graph retention, but does not alone bound forward temporaries, full hidden arrays, a wide MLP tensor, or the FP32 statistics temporary. Peak telemetry recorded after a completed invocation is not an enforcement mechanism. Keep the app generation and activation replay sequential on the Mac unless aggregate memory has been budgeted.

Recommended meaningful tests, without repeating the entire calibration dataset: a tiny hybrid model should match uninstrumented logits before/after wrapping; sampled rows must equal the indexed original X; moments must match an independent all-row sum; the reconstructed head must match native full-head logits and full-vocabulary KL; every planned linear/recurrent-attention/MLP projection must have the exact expected shape; and a saved package must reject corruption, omission, wrong source model and incomplete status. Existing tiny interception/moment tests are useful but do not replace cross-backend tests on the actual architecture.

Sampling limitations must remain visible in the manifest. One uniformly sampled row per stratum gives different inclusion probabilities when stratum lengths differ; equal samples per invocation also do not represent uniform weighting over the entire token corpus. If estimating a global Gram matrix from sampled X, weight each sampled outer product by its stratum size (and by any declared sequence weighting). The saved all-row moments cover all rows but repeated conversation prefixes are counted again on each full-prompt replay. This may be an intentional invocation-level calibration distribution; document it rather than implying independent uniformly sampled production tokens. Only prefill is captured, so final-turn generated answer tokens that never appear in a later prompt are not included. These are calibration design choices, not silently complete decode coverage.
