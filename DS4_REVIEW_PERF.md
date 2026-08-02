# DS4 sweep review (perf focus) — mcpzim

Exhaustive per-file pass: 376 code files across 37 batches.

## Findings

# Batch 1 — KokoroSwift MLX performance findings

## Findings

- [medium] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ConvWeighted.swift:100 — `let weight = weightNorm(weightV: weightV, weightG: weightG, dim: 0)` recomputes the L2-normalized conv weight on **every** forward call, and `bias = bias?.reshaped([1, 1, -1])` (line 101) re-reshapes the stored bias each call — weightV/weightG/bias never change after init (inference-only port), so the full norm reduction over the weight tensor (`computeNorm` → `MLX.sum(x * x)` over all weight elements) plus the elementwise normalize/scale is pure redundant recomputation — this runs per audio frame on every conv layer in the decoder/generator hot path (F0Conv, NConv, asrRes, decode.0-3, all Generator ups/resBlocks), adding a full-tensor reduction + reshape graph op per conv call — smallest safe fix: precompute `weight` and the reshaped bias once in `init` (weights are immutable in inference) and reuse the cached normalized weight in `callAsFunction`.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdainResBlk1d.swift:95-134 — `shortcut` and `residual` wrap every conv/upsample/pool in `MLX.swappedAxes(x, 2, 1)` round-trips (~10 transposes per residual+shortcut) — each transpose is a graph op that forces the frame tensor back and forth between (N,C,W)/(N,W,C) layouts on the per-frame decode path (Decoder.decode.0-3 and encode use this block), adding layout churn and non-contiguous memory access per frame — smallest safe fix: keep the decoder tensor in the layout conv1d expects and drop the per-op swappedAxes pairs (do one layout fix at the decoder boundary), or fold the transpose into the weight orientation so convs consume the native layout.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaLayerNorm.swift:25-26 — `MLX.mean(x, axes: [-1], keepDims: true)` and `MLX.variance(x, axes: [-1], keepDims: true)` are two separate full reductions over the frame tensor — a fused norm (MLXFast.layerNorm-style) computes mean+variance in a single pass, so this hand-composed norm costs 2 passes per frame per layer in the generator hot path — smallest safe fix: compute the sum and sum-of-squares in one pass (`MLX.sum(x)` and `MLX.sum(x*x)`) and derive mean/variance from them, or use the fused primitive where the gamma/beta scaling permits.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/InstanceNorm1d.swift:68-70 — `MLX.mean(input, axes: reduceDims, keepDims: true)` and `MLX.variance(input, axes: reduceDims, keepDims: true)` are two full reductions over the (N,C,W) frame tensor — InstanceNorm1d is invoked per frame per AdaIN norm in AdaIN1d on the decode path, so each call pays two passes where one pass computing sum and sum-of-squares suffices — smallest safe fix: derive mean/variance from a single `MLX.sum(input)` + `MLX.sum(input*input)` reduction, or use a fused instance-norm primitive.

## Coverage

- ios/LocalPackages/kokoro-ios/Package.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertEmbeddings.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertEncoder.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertIntermediate.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertLayer.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertLayerGroup.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertModelArgs.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertOutput.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfAttention.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfOutput.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/CustomAlbert.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaIN1d.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdainResBlk1d.swift — findings: 1
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaINResBlock1.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaLayerNorm.swift — findings: 1
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/Conv1dInference.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ConvWeighted.swift — findings: 1
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/InstanceNorm1d.swift — findings: 1
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/Interpolate.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LayerNormInference.swift — clean
# Performance findings — batch 2 (KokoroSwift MLX TTS)

## Findings

- [critical] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroTTS.swift:364 — `for frame in 0 ..< totalFrames { let phonemeIndex: Int = indices[frame].item() ... }` performs a `.item()` (full GPU→CPU eval+copy of the whole duration graph: BERT → durationEncoder LSTMs → predictorLSTM → durationProj) once per output frame inside `createAlignmentTarget`, on the synthesis hot path called from `predictDurations` in `generateAudio`. — totalFrames = sum of phoneme durations = thousands of frames for a long utterance (up to 510 tokens), so each synthesis pays thousands of full pipeline stalls (the skill's "per-frame `indices[frame].item()` building an alignment matrix — N stalls per synthesis"). — `let indicesArr = indices.asArray(Int.self)` once before the loop and index the native Swift array (`indicesArr[frame]`).

- [high] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroTTS.swift:353 — `durations.enumerated().map { ... let frameCount: Int = duration.item() ... return MLX.repeated(MLXArray([index]), count: frameCount) }` calls `.item()` per phoneme inside the `.map` (nPhonemes up to 510), forcing the entire duration graph to be evaluated and copied to host once per phoneme, and constructs a fresh `MLXArray([index])` + `MLX.repeated` tensor per phoneme (allocation churn). — per-utterance O(nPhonemes) pipeline stalls plus per-phoneme tensor construction; latency scales with token count. — bulk-extract durations once with `durations.asArray(Int.self)` before the map and build the `indices` array natively, then `MLXArray(indices)` once.

- [high] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TimestampPredictor.swift:57 — inside `for t in tokens` (loop grows with phoneme token count) the line `let tokenDuration: Float = predictionDuration[i..<j].sum().item()` (and the sibling `.item()` calls at lines 44, 45, 58) does a per-token slice+sum graph op plus a `.item()` host sync of the full duration tensor per token. — per-token GPU→CPU pipeline stalls on every timestamp prediction, latency scales with token count. — `let dur = predictionDuration.asArray(Float.self)` once before the loop; replace slice+`.sum().item()` with a native-array sum (`dur[i..<j].reduce(0, +)`).

- [medium] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LSTM.swift:88 — `var ifgo = xProj[0..., idx, 0...]` (same at line 136) is an MLXArray slice graph op created per sequence step inside the carried-state forward/backward loops; seqLen grows with phoneme count up to ~510 and the LSTM is invoked 3+ times per utterance (DurationEncoder blocks, predictorLSTM, TextEncoder.lstm). — O(seqLen) extra graph nodes per LSTM call (lazy slice construction + downstream matmul per token), adding graph-construction overhead that scales with token count. — pre-split `xProj` once before the loop (e.g. `MLX.split(xProj, into: seqLen, axis: -2)` or a native `asArray` per row) and index the prebuilt slices in the loop instead of slicing per step.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LSTM.swift:151 — `allCell.insert(currentCell, at: 0)` / `allHidden.insert(currentHidden, at: 0)` in the backward loop inserts at index 0, shifting the whole array every step — O(n) per step, O(n²) total for seqLen up to ~510. — quadratic element-move overhead in the hot LSTM path, worst on long utterances. — append in reverse order in the loop, then `reverse()` once before `MLX.stacked` (or use an index cursor).

- [medium] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/DurationEncoder.swift:123 — `let xPad = MLXArray.zeros([x.shape[0], x.shape[1], m.shape[m.shape.count - 1]])` then `xPad[0 ..< x.shape[0], 0 ..< x.shape[1], 0 ..< x.shape[2]] = x` allocates a zeros tensor and writes `x` into a slice of identical shape (m.shape[-1] == x.shape[2] == input seq_len here) per LSTM layer; the slice-assignment write forces a GPU evaluation of the LSTM output per layer. — redundant zeros-allocation + copy and a forced eval per LSTM layer on the duration-encoder hot path. — drop the padding copy when shapes already match (they do); keep only a conditional pad when seq_len differs.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/DurationEncoder.swift:109 — `x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(0, 2, 1), MLXArray.zeros(like: x), x)` re-applies the padding mask (and allocates `zeros(like: x)`) after every AdaLayerNorm layer; mask count scales with layer depth. — constant-factor mask pass + zeros allocation per alternating layer (depth bounded by config nLayer, so low severity). — apply the mask once before the layer stack; the layers here do not change padding positions, so re-masking per layer is redundant.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TextEncoder.swift:143 — `let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape[mask.shape.count - 1]])` then `xPad._updateInternal(x)` allocates a zeros tensor and copies `x` into it where the shapes are already identical (mask.shape[-1] == x.shape[2] == seq_len); `_updateInternal` forces evaluation. — redundant same-shape zeros allocation + copy + forced eval per utterance. — remove the padding step and return `MLX.where(mask, 0.0, x)` directly.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TextEncoder.swift:131 — `x = MLX.where(mask, 0.0, x)` re-applies the padding mask after every CNN layer in `for convBlock in cnn { for layer in convBlock { ... } }`; mask passes scale with `depth` (config nLayer). — constant-factor per-layer mask pass + zeros on the text-encoder path (depth config-bounded). — mask once before the CNN stack and again after the LSTM; the CNN layers preserve padding positions.

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Tokenizer.swift:20 — `text.map { vocab[String($0)] }` allocates a new `String` per character (each `Character`→`String` conversion) across three chained passes (`map`, `filter`, `map`) over the phonemized string, which grows with utterance length. — per-character allocation churn on the tokenization path; bounded by utterance length (max 510 tokens) so low severity. — iterate `unicodeScalars` once building keys (or reuse a single `Character`-keyed lookup via `vocab[Character]` / a prebuilt scalar→token map) and fuse the filter+unwrap into one pass.

## Coverage

- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LSTM.swift — findings: 2
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ReflectionPad1d.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/UpSample1d.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/Decoder.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/Generator.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/MLXSTFT.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/SineGen.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/SourceModuleHnNSF.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/eSpeakNGG2PProcessor.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/G2PFactory.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/G2PProcessor.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Language.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/MisakiG2PProcessor.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Tokenizer.swift — findings: 1
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/DurationEncoder.swift — findings: 2
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroConfig.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroTTS.swift — findings: 2
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/ProsodyPredictor.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TextEncoder.swift — findings: 2
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TimestampPredictor.swift — findings: 1
# Batch 3 — KokoroSwift utils/llama.cpp wrapper/MLXEmbedders performance findings

## Findings

No defensible performance findings in this batch.

**Why the batch is clean (zero-findings gate):** every in-scope loop iterates over a bounded axis, not user/row data.
- WeightLoader.swift:40-100 loops over model weight keys once at load time (bounded by model size); the `transposed` ops are one-time load preprocessing, not a hot path.
- AudioUtils.swift:65-67 is a plain float-copy loop into an AVAudioPCMBuffer, but it is `#if DEBUG`-only and runs once per debug file write — not shipped, not hot (per-skill DEBUG exemption).
- The MLXEmbedders forward passes (Bert/Gemma3/NomicBert/Qwen3) contain loops only over `layers` / `dense` bounded by the model config (numLayers/hiddenLayers), the MLX-skill "bounded model-config loop" exemption. No `.item()`/`asArray`/`eval()` syncs in loops (sync-point census found only doc-comment `asArray` examples, plus the README which is out of scope); no KV-cache concat growth (embedding calls pass `cache: nil`); attention uses fused `MLXFast.scaledDotProductAttention`; masks built once per forward, not per layer/token.
- BenchmarkHelpers/IntegrationTestHelpers loops are test/benchmark harnesses over fixed run counts (7–25 runs) and generation token counts capped by `maxTokens` (100–600); Swift `String +=` accumulation is amortized O(1), and these are test fixtures, not production hot paths.
- Config/CI/package manifests (pull_request.yml, .pre-commit-config.yaml, .spi.yml, Package.swift, LlamaCppSwift.swift, IntegrationTesting.swift, KokoroSwiftTests.swift, ToolCallIntegrationTests.swift) contain no runtime code.
- Fallback sweep run over the batch: cache grep (only `KVCache` doc comments, `BenchmarkHelpers` single-file temp cache, no unbounded registry) and for-loop inventory (top files NomicBert 38 / Qwen3 29 / Bert 24 / Gemma3 20 — all read end to end; every loop bounded by config or fixed counts). No external linters installed (Swift-only repo).

## Coverage

- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/WeightLoader.swift — clean
- ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Utils/AudioUtils.swift — clean
- ios/LocalPackages/kokoro-ios/Tests/KokoroSwiftTests/KokoroSwiftTests.swift — clean
- ios/LocalPackages/llama.cpp-swift/Package.swift — clean
- ios/LocalPackages/llama.cpp-swift/Sources/LlamaCppSwift/LlamaCppSwift.swift — clean
- ios/LocalPackages/mlx-swift-lm/.github/workflows/pull_request.yml — clean
- ios/LocalPackages/mlx-swift-lm/.pre-commit-config.yaml — clean
- ios/LocalPackages/mlx-swift-lm/.spi.yml — clean
- ios/LocalPackages/mlx-swift-lm/IntegrationTesting/IntegrationTesting/IntegrationTesting.swift — clean
- ios/LocalPackages/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/ToolCallIntegrationTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/BenchmarkHelpers/BenchmarkHelpers.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/EmbedderModelContainer.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/EmbeddingModel.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Extensions/MLXArray+Helper.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/ModelFactory.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Bert.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Gemma3.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Qwen3.swift — clean
# Batch 4 — MLX LLM performance findings

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma2.swift:72-78 — Gemma2Attention implements attention as a hand-composed chain (`scores = matmul(queries, keys.swappedAxes(-1, -2))`, `tanh(scores / logitSoftCap) * logitSoftCap`, then `softmax(scores, axis: -1, precise: true)` then `matmul(scores, values)`) instead of the fused `MLXFast.scaledDotProductAttention` used by every other model in this batch — the `precise: true` softmax forces the higher-precision, slower softmax path, and the raw per-head matmuls add graph ops (with an O(nKVHeads·repeats·L²) score tensor at prefill) — this runs per layer per token on the decode hot path (27+ layers for 9B/27B configs), so the fused SDPA kernel that natively handles Gemma2's soft-capping is strictly faster — smallest safe fix: route through `MLXFast.scaledDotProductAttention` with the soft-capping mask mode (as upstream mlx-lm does for gemma2) instead of manual matmul + precise softmax.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift:241-266 — `gatedDeltaOps` is a per-timestep loop: `for t in 0 ..< T` slices `q[0..., t]`, `k[0..., t]`, `v[0..., t]`, `g[0..., t]`, `beta[0..., t]` and calls `gatedDeltaStepOps` per step, accumulating into `ys` then `MLX.stacked(ys, axis: 1)` — with lazy MLX this builds one graph op per timestep and never evals inside the loop, so a T-token sequence (growth axis = sequence length; e.g. 4096-token prompt) builds ~T·10 graph nodes and a T-deep stack, ballooning lazy-graph memory and trace-construction time — this is the fallback reached by `gatedDeltaUpdate` (line 296) whenever the Metal kernel is unavailable — smallest safe fix: eval the carried `state` every N steps inside the loop (bounded lazy graph), or prefer the single-dispatch `gatedDeltaKernel` path and drop the per-token fallback for prefill.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LoraTrain.swift:46 — `LoRABatchIterator.next()` calls `tokenizer.encode(text: dataset[indices[$0]])` for every item of every batch, and `next()` is invoked once per training/validation step over the full dataset each epoch — because `indices.shuffle()` only reorders existing indices (lines 29-31, 39-41), the same strings are re-tokenized every epoch, so tokenization cost (the main CPU cost of LoRA training) is O(datasetSize × epochs) with no reuse — smallest safe fix: tokenize each `dataset[i]` once up-front into a `[[Int]]` cache keyed by index and index into it inside `next()`.

## Coverage

- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Pooling.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXHuggingFace/Macros.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Lora+Data.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LoraTrain.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/AfMoE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Apertus.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/BaichuanM1.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/BailingMoe.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Bitnet.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Cohere.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV3.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Ernie4_5.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Exaone4.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma2.swift — findings: 1
# Batch 5 — MLX LLM model implementations (performance-only)

Findings below are performance findings only, per the performance-review and
mlx-performance-review checklists. All paths are relative to the project root.

## Findings

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift:281 — Sequential per-token MLXArray slice/index loop in `ssmStep`: `for t in 0 ..< T { if let state = currentState { newState[0..., t] = fma(state, dtA[0..., t], newState[0..., t]) }; currentState = newState[0..., t] }` performs two element-slice graph ops (`newState[0..., t]` read and the in-place slice assignment) per sequence position inside a host loop over the full sequence length T. — During prompt processing T = prompt length (unbounded, thousands of tokens) and this runs once per Mamba layer (Jamba has ~32 layers, most of them Mamba), building O(T) sequential graph nodes per layer (≈2 ops × T × layers) that cannot overlap; prompt latency grows linearly with context and graph-construction cost explodes. During decode T=1 so it is cheap. — Replace the hand-written sequential scan with MLX's parallel associative-scan (`associative_scan`) or a chunked scan so the recurrence is computed in O(log T) parallel steps; at minimum, keep the scan on-device without per-position slice assignment (build `newState` by whole-array `takeAlong`/`putAlong` from precomputed deltas).

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift:601 — Host sync per sliding layer per decode step: `let offset = max(0, (cachePosition?.max().item() ?? 0) - effectiveSeqLen + 1)` forces a GPU→CPU `.item()` eval inside the per-layer decoder call, and `cachePosition` is loop-invariant across all layers of a step. — In the autoregressive loop this stalls once per sliding-attention layer per token (Gemma 3n has ~28 layers, most sliding), adding a full-graph eval sync per layer that is identical for every layer; latency grows with tokens × sliding layers. — Compute the offset scalar once from the already host-known `pastSeenTokens`/`h.dim(1)` in `Gemma3nLanguageModel.callAsFunction` (the language model already has `cachePosition` from host `pastSeenTokens`) and pass the host Int into the decoder layer instead of deriving it via `.max().item()` inside the layer loop.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift:435 — Loop-invariant weight transforms recomputed per token per layer: `predict` and `correct` call `predictionCoefs.weight.asType(.float32)` / `correctionCoefs.weight.asType(.float32)` and `clip(predictionWeight, min:, max:)` / `clip(correctionWeight, ...)` on the constant module weights every decode step for every layer. — Adds a re-cast + re-clip graph-op chain per token × layer on data that never changes, overhead in the autoregressive loop. — Cache the clipped/cast float32 weight as a lazily-computed `MLXArray` on the module (computed once on first call) instead of re-deriving it per `predict`/`correct` call.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Internlm2.swift:47 — Host sync per attention layer per token in the array-offset RoPE path: `let maxOffset = offset.max().item(Int.self)` forces a GPU→CPU eval on `offset` to compute the Dynamic-NTK base. — When `applyRotaryPosition` is invoked with a `BatchPositionedKVCache` (batch decode) this runs once per attention layer per decode step, adding a sync stall per layer; the scalar `maxOffset` is the same for all layers. — Compute the NTK base once from the host-side sequence offset (or pass the scalar offset as an `Int` from the cache, which is already host metadata) instead of `.max().item()` inside the per-layer call.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift:593 — Sliding-window attention mask rebuilt per sliding layer per token: `tril(MLXArray.ones(maskArray.shape, dtype: .bool), k: -slidingWindow)`, `MLX.where(...)`, and `take(...)` allocate a new slidingWindow×slidingWindow mask tensor per layer per decode step. — Allocation/op churn in the decode loop scaled by tokens × sliding layers (bounded mask size, so constant-factor waste). — Hoist the invariant `tril`/`where` mask construction out of the layer loop (it depends only on the sliding-window geometry) and slice it per token with the precomputed offset; only the `take` slice needs to run per call.

## Coverage

- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift — findings: 4
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3Text.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Granite.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Internlm2.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/LFM2.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/LFM2MoE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Lille130m.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Llama.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiMo.swift — clean
# Batch 6 — MLX LLM model implementations (performance-only)

Findings below are performance findings only, per the performance-review and
mlx-performance-review checklists. All paths are relative to the project root.
Hot-path context: these are LLM model classes; `callAsFunction`/attention/MLP
bodies run once per decoder token per layer in the autoregressive loop, so
per-call allocation and dtype round-trips there scale with tokens × layers.

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NanoChat.swift:17 — Hand-composed RMSNorm instead of the fused `MLXFast.rmsNorm` kernel, invoked in the hot decode path: `let meanSquares = mean(x.square(), axis: -1, keepDims: true); return x * (meanSquares + eps).rsqrt()` is called by `NanoChatAttention.callAsFunction` (lines 118-119) and `NanoChatBlock.callAsFunction` (lines 176, 178), i.e. every transformer layer and every attention/MLP norm per decoder token. — Each layer/token runs a full reduction (`x.square()` + `mean`) then a sqrt + divide/multiply, i.e. two reduction passes plus elementwise ops, instead of the single fused `rms_norm` kernel (which is also higher-precision internally); with NanoChat's ~32 layers the per-token decode cost is multiplied by layers × tokens, and the reduction is exactly the op the MLX fast kernels fuse. — Replace with the fused kernel: `MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)` (this norm applies no learned weight), which is a single fused kernel per layer/token.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Phi.swift:66 — Per-attention-layer dtype round-trip on the decode path: `attentionWithCacheUpdate(queries: queries.asType(.float32), ...)` then `.asType(values.dtype)` (line 73) re-casts queries to float32 and the attention output back to the value dtype once per layer per token. — Decode is bandwidth-bound; materializing queries as fp32 doubles their bandwidth (2× traffic) plus forces two extra cast graph ops per layer per token (Phi has ~32 layers), all of which could be avoided; the fused SDPA kernel internally handles the working precision. — Drop the explicit upcast: feed `queries` (fp16) directly to `attentionWithCacheUpdate` and only apply `.asType(values.dtype)` on the final output if the kernel result dtype differs, instead of round-tripping both ends per call.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift:82 — Per-call allocation of a constant `ones` weight inside the hot Mamba decode loop: `let identityWeight = MLXArray.ones([groupSize])` is built on every `NemotronHRMSNormGated.callAsFunction` invocation, which runs once per Mamba decoder layer per token (NemotronH is a hybrid with many Mamba layers, so this scales with tokens × Mamba layers). — Each token/layer allocates a fresh `[groupSize]` ones tensor (groupSize = intermediateSize/numGroups, e.g. thousands) that is always the same value, adding allocation/op churn to the bandwidth-bound SSM decode path. — Store the identity weight once on the module (lazily computed `MLXArray.ones([groupSize])`) and reuse it across calls instead of rebuilding it per call.

## Coverage

- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiMoV2Flash.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiniCPM.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiniMax.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Mistral3Text.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NanoChat.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Olmo2.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Olmo3.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OlmoE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OpenELM.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Phi.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Phi3.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/PhiMoE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen2.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MoE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3MoE.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3Next.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/SmolLM3.swift — clean
# Batch 7 — MLX LLM common adapters/chat/sampling (performance-only)

Findings below are performance findings only, per the performance-review and
mlx-performance-review checklists. All paths are relative to the project root.
Hot-path context: ChatSession.respond/streamMap and TokenIterator.next run once
per generated token; DoRA/LoRA forward layers run once per token per adapted
layer when adapters are used unfused; Evaluate.swift's sampling/penalty path runs
per token when configured.

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/DoRA+Layers.swift:22-24 — DoRA forward recomputes the full adapted weight matrix and its row-wise norm on every call: `let adapted = weight + matmul(scale * loraB.T, loraA.T)` then `let denom = norm(adapted, axis: 1)` (line 23) inside `forward(...)`, which `DoRALinear.callAsFunction` (line 110-118) and `QDoRALinear.callAsFunction` (line 171-181, passing `weight: dequantizedWeight` at line 177 — a full dequantization of the quantized weight per call) invoke once per token per adapted layer. — The unfused DoRA path (the `callAsFunction` forward, used while the adapter is loaded rather than `fused()`) performs an O(d_out×d_in) matmul (`loraB.T @ loraA.T`) plus an O(d_out×d_in) reduction to compute the row norm for a full-size weight matrix (e.g. 4096×4096) on every decoder token and every adapted layer, and QDoRA additionally dequantizes the whole quantized weight each call; since the base weight, loraA/loraB and magnitude are frozen during decode, this per-token O(d²) work is pure recomputation of a constant. — Cache the constant denominator once at load/freeze time (e.g. lazily compute `denom = norm(weight + matmul(scale*loraB.T, loraA.T), axis:1)` once after the adapter is loaded, and for QDoRALinear cache the dequantized weight), then have `forward` use the cached `magnitude/denom` instead of recomputing it per call.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift:264 — `output += chunk` accumulates the whole response string inside the per-token stream loop (`var output = ""` … `for try await chunk in streamResponse(...) { output += chunk }`, lines 260-266). — Each `+=` copies the entire previously-accumulated string, making a full-response build O(total_chars²); for a multi-thousand-token generation (tens of thousands of chars) this quadratic copy dominates the finalization of a response. — Append chunks into a `[String]` buffer and `joined()` once at the end, or use an accumulating buffer (e.g. `String.append` on a preallocated buffer) so each token is O(1)-amortized.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:444-448 — `FrequencyPenaltyContext.process` allocates a full-vocabulary histogram on every decoded token: `let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)` and `let histogram = MLXArray.zeros([vocabSize], type: Float32.self).at[validTokens.asType(.int32)].add(ones)` (line 444-446) then `logits - (histogram * frequencyPenalty)`. — When `frequencyPenalty` is configured, each token in the decode loop allocates a vocab-sized tensor (e.g. 49k float32 ≈ 196 KB) plus a ones array and scatter-add, adding per-token allocation/bandwidth churn on the sampling path. — Keep a persistent vocab-sized accumulator (allocated once in `init`) and scatter-add into it per token, clearing the touched positions each step, instead of reallocating `zeros([vocabSize])` per token.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:1889-1891 — `Generation.collect` (and `TokenGeneration.collect`, lines 1920-1926) builds a batch by array concatenation: `(batch ?? []) + [element]`. — Each appended element allocates a new array copying the whole running batch, making batch assembly O(n²) in the batch size (the reducers are documented for use with `throttle()` to gather elements); for a large throttled batch this is quadratic copy churn. — `mutating func collect` with `batch.append(element)` (or `batch?.append(element); return batch`) keeps appends O(1)-amortized.

## Coverage

- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/SSM.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Starcoder2.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/DoRA+Layers.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRA+Layers.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRAModel.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapter.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapterFactory.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapterTypeRegistry.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/BaseConfiguration.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Chat.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Downloader.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift — findings: 2
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Extensions/Encodable+toolResult.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Extensions/JSONDecoder+JSON5.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/GenerationConfigFile.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/InterpolationUtils.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/JSONDecodingTypes.swift — clean
# Batch 8 — MLX Swift LM common (MLXLMCommon) performance findings

Reviewed 20 files under mlx-performance-review + performance-review. Hot paths:
per-token autoregressive decode (KVCache.update/makeMask, RoPE layers, Gemma
RMSNorm/clipResidual), per-token streaming detokenization, and per-tool-call
parsers. No `.item()`/`eval()` host syncs occur inside the loops of the listed
files (the per-token argmax sync lives in Evaluate.swift, out of batch).

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tokenizer.swift:96 — NaiveStreamingDetokenizer.next() re-decodes the whole accumulated segment each token — `let newSegment = tokenizer.decode(tokenIds: segmentTokens)` decodes the full monotonically growing `segmentTokens` on every generation step, then takes the suffix; work is O(segment length) per token → O(n²) total within any non-newline chunk (e.g. a 2000-token paragraph: ~2M token decodes instead of 2k). Called per token from the generation loop (Evaluate.swift:1969-1970). Smallest safe fix: track the already-decoded prefix length and decode only `segmentTokens[newStart...]` (or cache the decoded prefix string), resetting on `startNewSegment()`.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift:36 — clipResidual does an fp16→fp32→fp16 dtype round-trip per residual add — `let xFloat32 = x.asType(.float32); let yFloat32 = y.asType(.float32); let result = xFloat32 + yFloat32; return clip(result, ...).asType(.float16)` materializes full tensors in fp32 (2× bandwidth) plus clip plus a cast back; called twice per layer per token in Gemma3Text/Gemma3/Gemma3-VLM (confirmed callers), so ~2×numLayers×(5 passes over hidden size) per token. Smallest safe fix: keep the residual stream in fp32 across the two residual adds and clip once at layer exit, or use a fused fp32 clip-add (MLXFast) so the upcast/downcast is not repeated per residual.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift:25 — RMSNorm recomputes `1.0 + self.weight` every call — `MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)` allocates an elementwise add over `dimensions` per norm per layer per token (Gemma3Text calls RMSNorm ~2× per layer per token; loop-invariant across tokens). Smallest safe fix: precompute `self.weight + 1` once in `init`.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEUtils.swift:144 — ProportionalRoPE.callAsFunction performs ~6 `split` + 4 `concatenated` graph ops per RoPE application — the partial-rotation path splits head/tail, left/right, leftParts/rightParts, concat, MLXFast.RoPE, then splits and re-concats three times; called for Q and K per layer per token in Gemma4Text (proportional rope with partial rotation), ~20 graph-op kernel launches per layer per token just for RoPE. Smallest safe fix: rotate only the `rotatedDims` slice in one MLXFast.RoPE and splice via a single concatenated per head (halve the split/concat count), or hoist the partial-rotation layout so only the rotated portion is re-split.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEUtils.swift:293 — YarnRoPE.callAsFunction copies x and applies `*= _mscale` per call before RoPE — `x = x[0..., .ellipsis]` forces a copy, then `x[.ellipsis, 0 ..< dimensions] *= _mscale` is a full elementwise pass per Q/K per layer per token for yarn models (GPTOSS); MLXFast.RoPE is then called with `scale: 1.0`. Smallest safe fix: fold `_mscale` into `MLXFast.RoPE(..., scale: _mscale)` (when the last axis == dimensions) to drop the copy + multiply pass.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SuScaledRoPE.swift:43 — SuScaledRoPE.callAsFunction copies x and applies `*= _scale` per call before RoPE — same copy+multiply pattern (`let x = x[0..., .ellipsis]; x[.ellipsis, 0 ..< dimensions] *= _scale`) per Q/K per layer per token for PhiMoE/Phi3 (Su-scaled rope), with RoPE called at scale 1.0. Smallest safe fix: pass `scale: _scale` to MLXFast.RoPE and drop the copy + elementwise multiply.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:671 — RotatingKVCache.makeMask single-token wrapped-window case builds and rolls an O(maxCacheSize) mask per call — when `offset >= windowSize && maxCacheSize > windowSize`, it allocates `MLXArray(0 ..< Int32(maskSize)) .>= ...` and `roll(mask, shift: currentIdx + 1)` once per layer per token in sliding-window decode after the window wraps (Qwen3Next/Gemma3nText windowed heads); host array build + roll copy per layer per token. Smallest safe fix: cache the rolled mask per (idx, windowSize) or build it symbolically once and index/roll lazily.

## Coverage

- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/LanguageModel.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelConfiguration.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelContainer.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelFactory.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift — findings: 2
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Module+Extensions.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/AbstractModelRegistry.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/ProcessorTypeRegistry.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEApplication.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEUtils.swift — findings: 2
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SuScaledRoPE.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tokenizer.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/TokenizerLoader.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GLM4ToolCallParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/JSONToolCallParser.swift — clean
# Batch 9 — Performance review findings (MLXLMCommon Tool/VLM, MLXVLM)

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift:143 — streaming tool-call buffering re-parses the entire accumulated buffer on every generated chunk — while in `collectingToolCall` state each incoming chunk does `toolCallBuffer += chunk` then `parser.parse(content: toolCallBuffer, tools: tools)` over the whole growing buffer (plus the O(buffer) `jsonBracesBalanced` scan at line 151/127). A tool call spanning N tokens runs N full JSON decodes/parses of an ever-larger string, so parse cost is O(N²) in the tool-call length in tokens; a long args payload (e.g. a large data/code argument) makes generation stall noticeably at the end of each tool call. — Only attempt a parse when the buffer can be complete (e.g. brace-balanced via `jsonBracesBalanced`, or when the end tag/`[ARGS]` delimiter is present), so the O(buffer) parse/scan runs once at completion instead of per token.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryUtils.swift:46 — `makeTokenIds` re-tokenizes the entire growing seed string each loop iteration — the loop does `chunk += seedText` then `tokenizer.encode(text: chunk)` over the whole accumulated string, so for a requested `tokenCount` of N the encoder processes ~N× the full string: O(N²) tokenizer work (e.g. thousands of tokens → thousands of re-encodes of a multi-KB string), which directly slows the one-time memory `tune()` measurement used to set wired budgets. — Encode `seedText` once and repeat/pad the resulting token IDs to `count` (or encode only the incremental delta), instead of re-encoding the growing string.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/MediaProcessing.swift:176 — `asMLXArray` calls `context.clearCaches()` on every render — in the video paths `_asProcessedSequence` maps every frame through `asMLXArray` (`ciImages.map { $0.asMLXArray() }` line 449), so clearing the shared CIContext cache per frame defeats GPU/texture cache reuse across the frames of a sequence; a multi-frame video pays a cache rebuild per frame on top of the render+RGBAf conversion. — Hoist `clearCaches()` out of the per-frame loop (e.g. once per `asProcessedSequence` call) or drop it, since the comment already notes it is "probably not strictly necessary".
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift:454 — `SEBlock.callAsFunction` constructs a new `AvgPool2d` module (kernel/stride sized by the runtime H,W) on every forward pass — the vision encoder runs per image, so this allocates a fresh pooling module plus its internal buffers each prefill; with video/`outputHiddenStates` paths it multiplies. — Hoist a fixed `AvgPool2d` (kernel = `(h,w)` is constant for a given image size) to an init-time stored property.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift:420 — `LayerNormChannel` hand-composes layer norm (mean, `pow(x-u,2)`, `sqrt(s+eps)`, div) as ~5 separate elementwise passes — the vision encoder forward runs per image, so this un-fused norm adds pass churn over the whole feature map on each image; `MLXFast.layer_norm` is fused and internally higher-precision. — Replace with `MLXFast.layer_norm(x, weight:..., bias:...)`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma3.swift:857 — `maskedScatter` forces a host sync and full host-side scan of the expanded image mask — `imageMaskExpandedFlattened.asArray(Bool.self)` copies the whole `[1, L, embedDim]` mask (e.g. 4096×2304 ≈ 9.4M elements) to the host and `enumerated().compactMap` builds the position list, then a host-side fancy-indexed scatter into `finalEmbeddingFlattened`; on a large prompt this blocks the GPU and does O(L×embedDim) host work per prefill (once per image). — Keep the mask on device (`argwhere`/`where` to gather positions) and do the scatter with `MLX` tensor ops instead of an `asArray` host round-trip; at minimum restrict the host copy to the non-expanded mask dimension before `repeated`.

## Coverage
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/KimiK2ToolCallParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/Llama3ToolCallParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM2ToolCallParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/MistralToolCallParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/ParserUtilities.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/PythonicToolCallParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/XMLFunctionParser.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Tool.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCall.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallFormat.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolParameter.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Value.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/UserInput.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryPolicies.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryUtils.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/MediaProcessing.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift — findings: 2
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma3.swift — findings: 1
# Batch 10 — Performance review findings (MLXVLM Models: Gemma4, GlmOcr, Idefics3)

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:60 — `gemma4MaskedScatter` forces a full host materialization + copy of the broadcast image mask and a host-side scan to build scatter indices — `let flattenedMask = mask.flattened().asArray(Bool.self)` is called with `imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)` (line 1714), so the whole `[batch, seq, hidden]` boolean mask (Gemma4 text hidden = 1536) is copied to the host and `enumerated().compactMap` (line 63) walks every element to build `targetIndices`, followed by a host-side fancy-index scatter `result[MLXArray(targetIndices,...)] = flattenedSource` (line 77). A 2048-token prompt ≈ 3.1M elements of host sync + copy + scan per image prefill (one GPU→CPU stall per image), on the prefill path that also runs the vision encoder. — Keep the mask on device (`argwhere`/`where` to gather positions) and scatter with MLX tensor ops instead of an `asArray` host round-trip; at minimum operate on the un-broadcast `[batch, seq]` image mask and derive positions without materializing `hidden`-dim copies.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:118 — `gemma4ApplyMultiDimensionalRoPE` multi-dim branch builds per-dimension cos/sin tables and partial results inside `for d in 0 ..< numDimensions` (each iteration does `cos`/`sin`, two `concatenated`+`asType`, `expandedDimensions`, plus a `gemma4RotateHalf` concatenation, then a final `concatenated(parts)` at line 139) — this runs on every vision attention layer (16 layers) per image prefill, so ~2×7 tensor ops × 16 layers of allocation/pass churn on the vision path even though the cos/sin tables for a fixed image are identical across layers. — Precompute the per-dim cos/sin tables once per image (or once per model) and reuse across layers instead of recomputing inside each layer's RoPE call.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:1569 — `patchPositions` builds the per-patch position table with host-side nested loops `for _ in 0 ..< batch { for y in 0 ..< patchesH { for x in 0 ..< patchesW { values.append(...) } } }` plus the padding loop, then constructs an `MLXArray(values, [batch, count, 2])` — per image prefill this is a host triple-loop over every patch (800×800 → 2500 patches × 2 values) and a host→GPU transfer; with multiple images (multi-image chat) it multiplies. — Vectorize on device (build `x,y` grids with `MLXArray(0 ..< patchesH)` broadcast/repeat) or at least hoist the position table for the fixed default 800×800 size since `patchPositions` is recomputed per image with identical geometry.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:1856 — token expansion allocates a fresh array per image token — `expandedTokens.append(contentsOf: Array(repeating: config.imageTokenId, count: config.imageSeqLength))` creates a new `[Int]` of `imageSeqLength` (280 default) elements inside the `for token in promptTokens` loop, plus `contentsOf:` copies, so a multi-image prompt (k image tokens) allocates k 280-element arrays and copies the whole expanded list repeatedly; one-time per request but wasteful on the prompt-construction path. — Build one `repeating` buffer once outside the loop and `append(contentsOf:)` the same array, or reserve `expandedTokens` up front.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift:941 — `getRopeIndex` builds 3D M-RoPE position IDs token-by-token on the host — `inputTokens[st...].firstIndex(of: Int32(imageTokenId))` scans the token slice per image (each `inputTokens[st...]` also allocates a new slice) and the triple-nested image-token loops `for t in 0 ..< llmGridT { for h ... { for w ... }}` append one `dimT/dimH/dimW` element each (lines 956-963), then `positionIds[0, batchIdx] = MLXArray(dimT)` (line 976) fancy-indexes back. For a prompt with many image tokens this is a full host-side enumeration of the whole sequence plus a host sync (`inputIds[batchIdx].asArray(Int32.self)` line 914) per image prefill. — Compute positions vectorized on-device from the image-grid THW (each image contributes a contiguous `[t,h,w]` block that can be built with broadcast/repeat/arange) instead of per-token host arrays; at minimum reuse the `dimT/H/W` builders across images.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift:105 — `applyMrope` does per-chunk element indexing — `chunk[i % 3]` inside `chunks.enumerated().map { ... }` creates a graph op per chunk, and `applyMrope` is invoked from `rotaryEmb(h, positionIds:)` on every decode token forward pass (position embeddings are recomputed per token), so 3 indexed-slice graph nodes plus `split`+`concatenated` per token on the decode hot path. — Replace the per-chunk `chunk[i % 3]` subscripts with a single vectorized `take_along_axis`/reshape gather across the 3 chunks.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift:509 — vision `Attention.callAsFunction` runs one `MLXFast.scaledDotProductAttention` per frame in `for i in 0 ..< (cuSeqlens.count - 1)` and `concatenated(attnOutputs, axis: 2)` (line 521) — for video with many frames this is one attention + concat graph op per frame per vision block (depth blocks), scaling O(frames)×depth on the vision prefill path (single-image is 1 chunk, so fine). — When `cuSeqlens.count == 2` (single image) call SDPA once directly, and for multi-frame use a batched single SDPA with per-frame masking instead of per-frame slicing+concat.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift:715 — `prepareInputsForMultimodal` slices one embedding row per image token then concatenates all segments — inside the chunk loops `segments.append(inputs_embeds[0, start_idx ..< pos])` and `segments.append(currentImage[i ..< i + 1])` create one MLXArray slice graph node per image token (64+ per image), and the final `concatenated(segments, axis: 0)` (line 724) joins potentially hundreds of segments; per image prefill this is O(imageTokens) graph nodes + one large concat on the multimodal prefill path. — Gather all image rows with one indexed `take_along_axis`/fancy-index and interleave text/image segments with a single reshape/concatenate of few segments instead of per-row slicing.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift:534 — vision `Encoder.callAsFunction` always collects per-layer hidden states (`encoderStates?.append(h)` each layer when `outputHiddenStates`) but `Idefics3.getInputEmbeddings` passes `outputHiddenStates: true` and discards the result (`let (pooler_output, _, _) = visionModel(pixelValues, outputHiddenStates: true)` line 664) — so every image prefill allocates an `[MLXArray]` of 13 references that is immediately dropped. — Pass `outputHiddenStates: false` from `getInputEmbeddings` (or change the default) so the encoder skips the append work.

## Coverage
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift — findings: 4
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift — findings: 3
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift — findings: 2
# Batch 11 — MLXVLM model files (performance)

Reviewed 5 MLX Swift VLM model files. The vision tower (image prefill) and
image-feature merge paths are per-image hot paths that grow with the number of
image patches; several build one lazy MLXArray graph node per patch or per
element, which cannot be fused. Language decode loops all use
`attentionWithCacheUpdate` (KV-cache) correctly — no per-token sync or
re-attention issues found in these files.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift:215-219 — per-element MLXArray assignment filling positional-embedding tail — `resultedPositionalEmbeddings[i, j] = resizedEmbeddings[0]` runs inside `for j in numPositions ..< maxLength`, each write is a separate lazy graph node (indexed write + scalar extract), so an image whose patch count is much smaller than `maxLength` (pixelValues.dim(1)) creates one graph op per filled position and can never be fused; growth axis = maxLength (patch count, up to tileSize² × tiles) — replace the fill loop with a single slice assignment broadcast, e.g. `resultedPositionalEmbeddings[i, numPositions...] = resizedEmbeddings[0].expandedDimensions(axis: 0)` broadcast along the tail.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift:726-735 — patch extraction builds one slice+flatten per patch — `imageArray[0, startH ..< (startH + patchSize), startW ..< (startW + patchSize), 0...].flattened()` inside nested `for ph`/`for pw` creates one lazy graph node + allocation per patch; for tileSize 512 / patchSize 16 that is 1024 slices per tile, × up to maxTiles 10 → ~10k graph ops per image; growth axis = totalPatchesH×totalPatchesW — reshape to `[numPatchesH, numPatchesW, patchSize, patchSize, C]` once and flatten per-patch with a single batched reshape/gather instead of per-patch subscripts.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift:150-164 — `unfold` (im2col) uses 4 nested loops with per-element indexing `x[0..., 0..., hIdx, wIdx]` — each subscript is a separate lazy graph op, so a 24×24 patch grid with kernelSize 2/stride 2 costs ~576 subscripts per image and the graph cannot fuse; growth axis = heightOut×widthOut×kernelSize² — implement with a single strided slice/reshape (or one `take_along_axis` gather) rather than per-element subscripts.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift:706-721 — per-patch split + interleave concatenation in `mergeInputIdsWithImageFeatures` — `MLX.split(imageFeatures, indices: splitIndices, axis: 1)` slices image features into `numImagePatches` (one per patch, e.g. 576 for Pixtral-sized images) separate arrays, then the zip loop concatenates 2×numImagePatches+1 pieces along axis 1; each slice and concat piece is a graph node — scatter the whole image-features block into the embedding in one assignment (like LFM2VL's merge) instead of per-patch split/concat.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift:854-856 — per-patch slicing in `mergeInputIdsWithImageFeatures` — `imageEmbeddings.append(imageFeatures[0..., i ..< (i + 1), 0...])` runs `for i in 0 ..< numImagePatches` creating one lazy slice per patch (576 for a 24×24 grid), then a 1153-piece concatenation; growth axis = numImagePatches — do a single bulk scatter of `imageFeatures` at the image-token positions (one assignment op) instead of per-patch slices.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift:224-241 — `generateBlockAttentionMask` builds an O(seqLen²) Swift `[Float]` array with three nested loops and then copies it into an MLXArray — for 576 patches that is 331k element writes (~1.3 MB) plus the O(seqLen²) loop per image; growth axis = seqLen² (total patches across the batch) — build the block-diagonal mask with MLX ops (e.g. per-image `ones` blocks assembled/broadcast) or fill only the on-diagonal blocks instead of materializing every off-diagonal element.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift:298-305 — vision attention uses hand-composed `MLX.matmul(queries, keys.transposed) * scale` + `softmax` + matmul instead of the fused `MLXFast.scaledDotProductAttention` — materializes the full (heads, seqLen, seqLen) attention-weight tensor (e.g. 32 heads × 576 × 576 ≈ 42 MB fp16 per image) instead of fusing and streaming it; growth axis = heads×seqLen² — use `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:)` which is fused and higher-precision.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift:550-564 — `attentionMask` allocates two dense boolean `full([1, sequenceLength, sequenceLength], values: false)` tensors (full + window masks) per image/video prefill, O(seqLen²) memory each; for a 1024-patch image that is ~1M elements per mask and the cost grows with video frame count; growth axis = sequenceLength² — build the causal block mask lazily/sparsely (mask only the attended on-diagonal blocks) or reuse a single mask when cuSeqlens are unchanged.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift — findings: 3
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift — findings: 1
# Batch 12 — MLXVLM model files + factory (performance)

Reviewed 9 MLX Swift VLM files. Language decode loops use `attentionWithCacheUpdate`
(KV-cache) correctly; the findings are in the linear-attention (gated delta net)
prefill scan, the per-token interleaved-mrope element loops, the once-per-prompt
rope-index host loop, the vision block mask, and prompt string builders.

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:100-119 — sequential delta-net scan over the whole prompt with no eval — `for t in 0 ..< T { let qT = q[0..., t] ... ys.append(y); state = newState }` runs one slice-subscript per step, carries `state` across T iterations and accumulates T lazy MLXArray nodes in `ys` before `MLX.stacked(ys, axis: 1)` at line 121, so a prefill of length T (thousands of tokens) builds ~6-10 graph ops × T (~24k nodes) with no intermediate eval and can never fuse; growth axis = prompt sequence length T (the linear layers are 3/4 of the layers given `fullAttentionInterval=4`) — implement a chunked/parallel-scan formulation (or `eval` the carried `state` every chunk and accumulate in a native array, building the tensor once at exit) instead of per-element subscripts.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:366-378 — per-element mrope indexing in the decode hot path — `for idx in 0 ..< dims { var slice = freqsT[0..., 0..., idx] ... slices.append(slice) }; return stacked(slices, axis: -1)` turns each rotary dimension into a separate lazy graph op (each `freqsT[0..., 0..., idx]` / `freqs[dim, 0..., 0..., idx]` is a subscript), then stacks dims arrays; with rotaryDim ≈ 48 this is ~48 subscripts + a stack per full-attention layer per decoded token, repeated over ~numHiddenLayers/4 layers → ~hundreds of un-fusable graph ops per token; growth axis = generated tokens × attention layers — replace the element loop with whole-array ops (e.g. a single interleaved gather via `take_along_axis`/reshape, or precompute the per-dim slice mask once) rather than one subscript per dimension.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:924-938 — same per-element interleaved-mrope loop in the decode hot path — `for idx in 0 ..< dims { var slice = freqs_t[0..., 0..., idx] ... slices.append(slice) }; return stacked(slices, axis: -1)` creates one lazy graph subscript + one stacked array per rotary dimension per attention layer per decoded token (headDim/2 ≈ 64 dims × numHiddenLayers layers ≈ thousands of un-fusable graph ops per token); growth axis = generated tokens × layers — build the interleave with a single whole-array gather/reshape (or precomputed per-dim masks) instead of per-dimension subscripts.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:1328-1510 — `getRopeIndex` runs host syncs and O(n) scans per prompt — `argMax(visionStartWeighted).item(Int.self)` (1369), `sum().item(Int.self)` (1375/1377), `batchInputIds.asArray(Int32.self).map { Int($0) }` (1380), then per image/video `inputTokens[st...].firstIndex(of:)` (1393/1400) and `lastArray.max().item(Int.self)` (1430) plus `llmPositions.max().item(Int.self)` (1497); each `.item()` is a blocking GPU→CPU eval+copy, and each `firstIndex` is an O(n) scan of the token suffix, so a long multimodal prompt with many images/videos pays multiple full evals and O(seq×images) host work once per new prompt; growth axis = prompt length × image/video count — compute the mrope indices with vectorized MLX ops (single `where`/gather over token positions) and one bulk `asArray` copy, hoisting the `.item()` syncs out of the per-image loop.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:542-550 — vision attention rebuilds a dense O(seqLen²) mask per block — `var mask = ones([1, sequenceLength, sequenceLength], ...); mask = mask * MLXArray(-1e9, ...)` allocates the full seq×seq mask (e.g. 784×784 ≈ 614k elements ≈ 2.4 MB fp32 per block) and `cuSeqlens.asArray(Int.self)` + a per-grid host loop with slice assignments (`mask[0..., start ..< end, start ..< end] = ...`) runs inside every `VisionBlock` (depth ~24 blocks), so image prefill pays depth × O(seqLen²) mask allocation and a host loop over grids; growth axis = patch sequence length² × vision depth — build the block-diagonal mask once outside the block loop (mask is identical across blocks for the same cuSeqlens) and reuse it, or construct it with MLX ops instead of a full dense ones/fill.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/SmolVLM2.swift:114-121 and 134-141 — string `+=` accumulation inside prompt-build loops — `textSplitFrames += (fakeToken + globalImageToken + String(repeating: imageToken, count: seqLen) + fakeToken)` per frame (114-121) and `textSplitImages += (fakeToken + "<row_\(h + 1)_col_\(w + 1)>" + String(repeating: imageToken, count: seqLen))` per tile (134-141) rebuild a seqLen-length repeated-token string each iteration and append onto a growing String (O(len²) copies); growth axis is bounded (frames ≤ maxVideoFrames 20, tiles ≤ image tiling), so this is prompt-time constant waste — hoist `String(repeating: imageToken, count: seqLen)` out of the loop and accumulate parts in a `[String]` + `joined()` instead of `+=`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35MoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift — findings: 3
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/QwenVL.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/SmolVLM2.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/VLMModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Package.swift — clean
# Batch 13 — Performance review findings

- [medium] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:19 — `static func load() async throws -> ZimfoRunner` re-opens every ZIM reader and rebuilds the whole tool stack from disk on every call, with no caching — `load()` lists the Documents directory, constructs a `LibzimReader` (a `ZimArchive` open that reads metadata, title/fulltext index, and routing-entry probes) for every `.zim`, resolves security-scoped bookmarks and opens those readers too, then builds `DefaultZimService`, `MCPToolAdapter.from(service:)`, and installs the host-state provider. It is called once per App Intent (`ZimfoRunner.load()` at ZimfoIntents.swift:61,157,174,193), so every Siri/Shortcuts invocation re-does identical disk scan + archive open + index/metadata reads + adapter construction for the same set of readers. With several multi-GB ZIMs this adds seconds of disk/index I/O per intent — concrete consequence: "how do I get to X" / "what's around here" intents stall on repeated full library reload instead of reusing an already-open archive. Smallest safe fix: memoize a lazy shared `ZimfoRunner` (or cache the opened readers by URL) and reuse it across intent calls, invalidating only when the library changes.

- [low] ios/MCPZimChat/AppIntents/LocationFetcher.swift:72 — `static func subscribe(_ cb: ...)` appends every callback to `subscribers` with no unsubscribe or dedup (`self.subscribers.append(cb)`), and `didUpdateLocations` fans out to all of them on the main actor (`for cb in self.subscribers { cb(loc.coordinate) }`). There is no removal path, so if any caller subscribes more than once (e.g. after a session/model reload re-registering) the array grows monotonically and each GPS fix (bounded ~1 Hz by `distanceFilter = 25`) invokes every accumulated closure on the main thread — concrete consequence: duplicate fan-out and monotonic memory growth that slow the main actor as subscribers accumulate. Smallest safe fix: dedup by closure identity or return an unsubscribe token, or assert single registration for the app-lifetime callers.

- [low] ios/MCPZimChat/AppIntents/ZimfoContext.swift:119 — `persist()` JSON-encodes and atomically writes the entire `Snapshot` (including the full `ActiveRoute.polyline`, which can be thousands of points) to disk on every `setActiveRoute` / `updateLastLocation` / `clearActiveRoute` (`JSONEncoder().encode(snap)` then `data.write(to: storeURL, options: [.atomic])`). `updateLastLocation` runs on each "how much longer"/"what's around here" intent even when only the coordinate changed, so each intent re-encodes + rewrites the whole route snapshot — concrete consequence: O(polyline)-size encode + full-file atomic write per intent for a one-field change, plus unnecessary disk writes. Smallest safe fix: persist `lastLocation` separately from the route (or only persist when the route changes), or throttle writes.

## Coverage
- ios/LocalPackages/mlx-swift-lm/scripts/verify-docs.sh — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/BaseConfigurationTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ChatSessionTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/EmbeddingPoolingTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/EvalTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/MediaProcessingTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/NemotronHTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ResolveTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/SampleTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/SpeculativeDecodingTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/TestTokenizer.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ToolTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/UserInputTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/WiredMemoryPolicyTests.swift — clean
- ios/MCPZimChat/App/MCPZimChatApp.swift — clean
- ios/MCPZimChat/AppIntents/LocationFetcher.swift — findings: 1
- ios/MCPZimChat/AppIntents/ZimfoContext.swift — findings: 1
- ios/MCPZimChat/AppIntents/ZimfoIntents.swift — clean
- ios/MCPZimChat/AppIntents/ZimfoRunner.swift — findings: 1
# Performance review — batch 14

File under review: `ios/MCPZimChat/Chat/ChatSession.swift` (6166 lines, read end-to-end).
Checklists applied: performance-review + ios-performance-review (Swift app/plugin code, @MainActor observable session).

## Findings

- [medium] ios/MCPZimChat/Chat/ChatSession.swift:394 — `print("[\(category)] \(decorated)")` inside `debug()`, which runs on the @MainActor hot path (every tool dispatch, every generation iter/stage of `runGenerationLoop`, every memory/background notification) — synchronous main-thread stdout writes accumulate during a turn and each call also computes `MemoryStats.formatted()` and enqueues a per-line disk append (`LogArchive.shared.append(...)` line 404); print in hot paths is synchronous/expensive per ios-performance-review — gate `print` behind a DEBUG flag and rate-limit/batch the archive appends.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:392 — `debugEntries.removeFirst(debugEntries.count - maxDebugEntries)` — `Array.removeFirst(k)` shifts the remaining elements, so once the pane is full (~500) every subsequent `debug()` call does an O(500) copy; `debug()` runs continuously across turns so this is steady churn on the main actor — use a ring buffer or drop in bulk less often.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:3074-3087 — exact-context guard re-renders the full prompt (`renderTranscript`) and re-tokenizes (`llama.promptTokenCount(prompt)`) inside a `while count > budget` loop, dropping one exchange at a time (up to ~10) — up to ~10 full renders+tokenizations of a ~10k-token prompt on the llama turn critical path whenever the budget is exceeded — drop to the watermark in one pass, then render+tokenize once.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:2997-3009 — `while turnsChars() > watermark` recomputes the whole transcript char sum (`turns.reduce(systemMessage.count + 2048) { $0 + $1.text.count + 16 }`) after every single dropped exchange — O(n²) over the bounded transcript on each turn that crosses the char budget — keep a running total and subtract the dropped turns.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:3824-3834 — `enrichSearchHits` reads the entire article content (`entry.reader.read(path: path)?.content`) for each hit, up to 3 hits, on every `search` tool dispatch, then `stripHTML`s it just to keep a 400-char preview — full ZIM reads (tens of KB each) plus HTML stripping on the search hot path — read only a bounded lead/prefix (cap bytes) instead of the whole article.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:4159-4185 — `prepareDiscussionEmbeddings` issues ~2 `SemanticReranker.shared.embedText` calls per section (title + contextual body) on the user-facing "preparing…" discussion path — a large article with ~100 sections ≈ 200 sequential on-device NLP embedding calls (~ms-10ms each) → seconds of latency before the discussion starts (re-run per pulled article at 4319/4354/4392) — cap sections, batch the embeddings, or embed lazily.

## Coverage
- ios/MCPZimChat/Chat/ChatSession.swift — findings: 6
# Performance review — batch 15

Files under review: `ios/MCPZimChat/Chat/DebugReport.swift`, `Chat/Message.swift`, `Common/DeviceProfile.swift`, `Common/LogArchive.swift`, `Common/SemanticReranker.swift`, `Common/ZimfoContext+Adapter.swift`, `Libzim/LibzimBridge.h`, `Libzim/LibzimBridge.mm`, `Libzim/LibzimReader.swift`, `Providers/FoundationModelsNativeTools.swift`, `Providers/FoundationModelsProvider.swift`, `Providers/Gemma4Provider.swift`.
Checklists applied: performance-review + ios-performance-review (Swift app/plugin code, MLX/tokenizer hot paths in Gemma4Provider).

## Findings

- [high] ios/MCPZimChat/Providers/Gemma4Provider.swift:780-782 — `let fullDecoded = tokenizer.decode(tokenIds: tokenIDs.map { Int($0) }, skipSpecialTokens: false)` re-decodes the ENTIRE accumulated token sequence on every generated token (comment at 756-762 calls this "incremental detokenisation" but the code re-decodes the full `tokenIDs` list each token) — O(n²) CPU: a maxReplyTokens turn (~384-512 tokens, DeviceProfile) runs ~1..n full sequence decodes ≈ 130k token decodes per turn on the on-device decode critical path, growing with reply length — decode only the newly-appended token(s) (e.g. keep the prior decode and decode just the incremental token range), or use the tokenizer's incremental-decode API if available; at minimum cache the last decode state instead of re-decoding from token 0.

- [medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:444-455 — download-progress `pollTask` calls `dirSizeBytes(at: repoRoot)` every 750 ms; `dirSizeBytes` (32-49) fully enumerates the recursive HF cache tree and calls `file.resourceValues` per file each tick — during a multi-minute 2.5 GB first-launch download this repeats a full-tree metadata scan (per-file resourceValues) hundreds of times, competing with the actual download I/O on the hot network path — poll at a slower cadence (e.g. 2-3 s) and/or track only the single dominant `model.safetensors` file size instead of re-summing the whole tree.

- [medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:142,338-352 — `stateStream()` appends every new `AsyncStream` continuation to `self.continuations` (inside `queue.sync`) and `set(_:)` (347-352) yields to ALL of them, with no removal when a subscriber deallocates — every `stateStream()` subscriber (model picker, status UI) is retained forever; `continuations` grows monotonically and `set` fan-out grows with it over the app lifetime — remove the continuation from the array when its stream terminates (`onTermination`) or use a per-subscriber token that gets removed.

- [medium] ios/MCPZimChat/Providers/FoundationModelsProvider.swift:75,238-252 — same unbounded-listener pattern: `stateStream()` appends `cont` to `self.continuations` inside `queue.sync` and `set(_:)` iterates all of them, with no removal on subscriber deallocation — every subscriber is retained and yielded-to forever; `continuations` grows monotonically and `set` fan-out grows across the app lifetime — remove the continuation on `onTermination` (stream finished) instead of appending without a cleanup path.

- [medium] ios/MCPZimChat/Common/SemanticReranker.swift:32,99 — `private var cache: [String: [Double]] = [:]` is a per-hit embedding cache written at `cache[key] = fresh` with NO eviction, TTL, or cap (unlike `embedTextCache` at 145 which is flushed when it exceeds 128 at 137-139) — a ZIM can hold tens/hundreds of thousands of articles; each distinct `zim:path` hit caches a ~512-dim `[Double]` (~4 KB), so an active search session can grow this into tens-to-hundreds of MB resident on a 4-8 GB phone, risking jetsam — cap the per-hit cache (e.g. flush wholesale when it exceeds a few thousand entries, mirroring `embedTextCache`) or bound by article count.

- [medium] ios/MCPZimChat/Common/ZimfoContext+Adapter.swift:24 — `polyline: r.polyline.map { .init(lat: $0.lat, lon: $0.lon) }` copies the ENTIRE route polyline point array into a fresh `[RouteSnapshot]` on every `mcpSnapshot()` call; `mcpSnapshot()` runs on every tool dispatch (the comment says "Called from tool dispatch"), and the polyline is unchanged between dispatches — a long route with thousands of points is re-allocated and re-copied per tool call across a multi-tool turn, wasting CPU/allocation on the tool hot path — keep the snapshot cached and rebuild only when `activeRoute` actually changes, or pass a lazily-backed/COW reference to the existing points.

- [low] ios/MCPZimChat/Chat/DebugReport.swift:103,144-171 — `emitDebugReport()` runs on the @MainActor (`extension ChatSession`) and does the full JSON encode (144-146), base64 of the whole payload (147), and the chunked `os.Logger` loop (160-171) synchronously on the main thread — for a long conversation with many tool-result messages + a full debug log, the serialization+encode+base64+chunk loop can exceed the 250 ms main-thread hang bar on the user's "send report" action — move the encode/base64/chunking off the main actor (utility QoS async) and yield only the final hash to the UI.

- [low] ios/MCPZimChat/Common/LogArchive.swift:84-101 — `previousSessionUncleanTail()` reads the ENTIRE previous-session log file with `String(contentsOf: prev, encoding: .utf8)` (line 88) and then `split`s it (91) — a long active session's log can grow large, and this whole-file read + line-split happens once at launch (likely main thread), paying memory proportional to the full log — read only the tail (e.g. seek to the last N bytes / use a bounded read) instead of loading the whole file.

## Coverage
- ios/MCPZimChat/Chat/DebugReport.swift — findings: 1
- ios/MCPZimChat/Chat/Message.swift — clean
- ios/MCPZimChat/Common/DeviceProfile.swift — clean
- ios/MCPZimChat/Common/LogArchive.swift — findings: 1
- ios/MCPZimChat/Common/SemanticReranker.swift — findings: 1
- ios/MCPZimChat/Common/ZimfoContext+Adapter.swift — findings: 1
- ios/MCPZimChat/Libzim/LibzimBridge.h — clean
- ios/MCPZimChat/Libzim/LibzimBridge.mm — clean
- ios/MCPZimChat/Libzim/LibzimReader.swift — clean
- ios/MCPZimChat/Providers/FoundationModelsNativeTools.swift — clean
- ios/MCPZimChat/Providers/FoundationModelsProvider.swift — findings: 1
- ios/MCPZimChat/Providers/Gemma4Provider.swift — findings: 3
# Batch 16 — iOS performance review (MCPZimChat Providers + Views)

## Findings

- [medium] ios/MCPZimChat/Providers/LlamaCppProvider.swift:1127 — decode-loop string accumulation `buffered += piece` (and, at 1142-1144, `parameters.stopSequences.contains(where: { buffered.contains($0) })`) makes the sampling loop O(n²) in output length — each sampled token copies the whole accumulated reply buffer and re-scans it for stop sequences. For a 512-token reply (~2 KB) this is small, but grounded/discuss answers with `replyTokensFloor` set high can run to thousands of tokens (8 KB+ buffer → ~64 MB+ of byte copies) on every reply, adding main-thread-independent but CPU-bound work to a latency-sensitive stream. Smallest safe fix: accumulate pieces into a `[Substring]`/array and `joined()` once at the end; for stop-sequence matching, match against the just-appended `piece` and the buffer tail only when the piece itself isn't a plain prose word (or check `buffered.hasSuffix` on a rolling tail window).

- [low] ios/MCPZimChat/Providers/LlamaCppProvider.swift:1117 — per-token allocation churn in the sampling loop: `var pieceBuf = [CChar](repeating: 0, count: 64)` plus `String(decoding: pieceBuf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)` allocates a 64-byte array, a mapped array, and a String per output token — allocation churn on the hottest loop (one iteration per generated token). Smallest safe fix: hoist a reused buffer (e.g. a class-scoped `[CChar]`/`UnsafeMutableRawPointer`) and decode the bytes into a `String` without the intermediate `map` (e.g. `String(bytes: pieceBuf.prefix(Int(n)), encoding: .utf8)`).

- [medium] ios/MCPZimChat/Views/ChatView.swift:229 — `showThinkingIndicator` calls `MessageRow.displayText(last.text, …)` on the still-streaming last assistant message, and then `MessageRow.body` (line 490) calls `displayText` again on the same message — so the full regex-stripping pipeline (4 closed-block `stringByReplacingMatches` + 4 stray-opener regexes + `range(of:)` + 4 `replacingOccurrences(of:)` passes) runs twice per 10 Hz streaming push over the growing reply text, on the main thread. Over a long grounded reply the double pass is meaningful main-thread work at 10 Hz. Smallest safe fix: memoize `displayText` keyed by message identity + text length/hash (one computation per push, shared by the indicator and the bubble), or gate `showThinkingIndicator` on a cheap raw-text check (`last.text.isEmpty` plus `contains("tool")`) instead of the full display pipeline.

- [medium] ios/MCPZimChat/Views/ChatView.swift:473 — `HeroMediaView(trace: trace)` is rendered for **every** article tool trace in every message (the `ForEach(message.toolCalls)` branch has no `isLatestAssistant` guard, unlike the `RouteWebView`/`PlacesWebView` branches above at 454/468). `HeroMediaView` instantiates a live `WKWebView` (`makeMediaWebView` → `WKWebView(frame:…)`) per resolved spec, and each webview holds ~300–500 MB of Metal buffers (the file's own comment at 442-447). Scrolling back through a long session with N article traces therefore resurrects N live webviews — the exact jetsam blowup the routes/places guard was written to prevent. Smallest safe fix: apply the same `isLatestAssistant` guard to the article branch (collapse older traces to a static placeholder, as `MapPlaceholder` does), or make `HeroMediaView` render a non-webview thumbnail for non-latest messages.

- [medium] ios/MCPZimChat/Views/DebugPane.swift:21 — `dateFormatter` is a computed property that constructs a fresh `DateFormatter` on every access, and it is accessed once per row inside `ForEach(session.debugEntries)` (line 48) over up to `maxDebugEntries = 500` entries. `DateFormatter` init is notoriously expensive (~5× reuse cost per the ios-performance-review checklist), so every body pass over the debug list constructs up to 500 formatters on the main thread — classic scroll jank. Smallest safe fix: hoist a single `static let` (or instance `let`) formatter with `en_US_POSIX`/fixed `dateFormat`, reused by all rows (and by `copyAll`).

- [medium] ios/MCPZimChat/Views/PastLogsView.swift:70 — `formattedDate` constructs a new `DateFormatter` (and line 78 a new `ByteCountFormatter`) per row inside `ForEach(files)`, over up to `maxFiles = 20` log files; additionally `modificationDate(url)`/`fileSize(url)` (lines 74, 78) re-fetch per-row metadata via `resourceValues` even though `LogArchive.allFiles()` already enumerated the directory with `includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]` (batched prefetch) — the values are fetched again, once per row, on the main thread. Smallest safe fix: return the already-batched `(url, date, size)` tuples from `allFiles()` so `PastLogsView` renders them without per-row formatter construction or per-row `resourceValues` calls (share one static formatter).

- [low] ios/MCPZimChat/Providers/ModelProvider.swift:195 — default `formatTranscript` accumulates with `out += "<|\(t.role.rawValue)|>\n\(t.text)\n"` over `turns` (the conversation transcript), which is O(n²) byte-copying for long transcripts. It is a protocol-extension fallback only used by providers that don't override it (MockProvider in this batch), so impact is bounded to dev/mock paths; the real LlamaCppProvider overrides via `template.renderTranscript`. Smallest safe fix: append pieces into an array and `joined()` once, or use an `NSMutableString`.

- [low] ios/MCPZimChat/Views/HeroMediaView.swift:129 — `reader.read(path: path)` loads the full article entry (up to ~2 MB of HTML per the file's comment) on the detached task, but only `raw.content.prefix(64 * 1024)` is scanned. Loading 2 MB to use 64 KB is a whole-file read where a partial/range read would suffice; it is off-main and once per trace (cold path), so low severity. Smallest safe fix: if the ZIM reader supports range reads, read only the first `heroScanBytes` bytes; otherwise accept the current cost.

## Coverage

- ios/MCPZimChat/Providers/LlamaCppProvider.swift — findings: 2
- ios/MCPZimChat/Providers/MockProvider.swift — clean
- ios/MCPZimChat/Providers/ModelProvider.swift — findings: 1
- ios/MCPZimChat/Views/ChatView.swift — findings: 2
- ios/MCPZimChat/Views/DebugPane.swift — findings: 1
- ios/MCPZimChat/Views/HeroMediaView.swift — findings: 1
- ios/MCPZimChat/Views/LibraryView.swift — clean
- ios/MCPZimChat/Views/ModelPickerView.swift — clean
- ios/MCPZimChat/Views/PastLogsView.swift — findings: 1
# Batch 17 — iOS performance review (MCPZimChat Views + Voice)

## Findings

- [medium] ios/MCPZimChat/Views/PlacesWebView.swift:78 — `payload` is a computed property (`{ parsePlaces(from: trace) }`) that re-parses `trace.rawResult` JSON on every access, and `body` (line 83-84) also calls `resolveSpec(userLocation:)`, which re-parses the same `trace.rawResult` again at 211-217. `body` re-evaluates on every SwiftUI tick (GPS location updates, chat streaming; the file's own comment at 511-512 confirms `updateUIView`/re-render fires on each tick), so each tick re-parses the places JSON ~5 times (payload accessed at 84, 113, 119, 128, and inside the sheet closures) — redundant recomputation of invariant data on a 2 Hz re-render path, growing with the number of returned places. Smallest safe fix: parse the payload once (`let payload = parsePlaces(from: trace)` at view init, or a `@State`/cached `PlacesPayload`) and pass it through; `resolveSpec` should read the already-parsed `zim` field instead of re-parsing `trace.rawResult`.

- [medium] ios/MCPZimChat/Views/RouteWebView.swift:70-109 — `routeEndpoints`, `turnByTurn`, and `resolveSpec` (245-319) are computed properties that each independently re-parse `trace.rawResult`/`trace.arguments` with `JSONSerialization` on every access, and `body` (111-175) accesses them multiple times per evaluation (`resolveSpec` at 112, `routeEndpoints` at 153/164, `turnByTurn` at 171/189/191). `body` re-evaluates on every SwiftUI tick — the file's comment at 698-700 states `updateUIView` fires on every GPS tick (~2 Hz) — so each tick re-parses the route JSON (which can carry a ~1500-point polyline) up to 4 times plus re-runs `Self.downsample(raw, target: 400)` and rebuilds the `geoJSONCoords` string (308-311) in `resolveSpec`. Smallest safe fix: parse the raw result once per trace into a cached `RouteSpec` (endpoints, turns, downsample, geoJSON string) and have the computed properties read from it.

- [low] ios/MCPZimChat/Voice/KokoroDownloader.swift:116-131 — `urlSession(_:downloadTask:didWriteData:)` spawns a `Task { @MainActor }` and writes a new `@Observable` `state` value on every progress callback (the delegate queue is already `.main`, so the Task hop is redundant), and for a ~360 MB download these callbacks fire many times per second — each drives Observation invalidation + UI re-render of the download progress view plus per-callback allocation. Smallest safe fix: throttle progress updates (e.g. coalesce to ~4-10 Hz by tracking last-write time / byte-delta, or update `cumulativeBytes`/`overall` only when the delta or time since last update exceeds a threshold).

- [low] ios/MCPZimChat/Voice/TTSService.swift:42-59 — `TTSPlaybackLevel.normalized` makes a second full pass and allocates a new `[Float]` via `samples.map { $0 * gain }` per synthesized chunk (called per chunk in `speak`/`speakChunk` at 417/468), and `speak`/`speakChunk` then copy that array into the `AVAudioPCMBuffer` at 425-428 — two array copies per chunk on the audio hot path, though bounded by the chunk cap (~400 chars / ~100k samples). Smallest safe fix: combine the RMS+peak scan and the gain application into one in-place pass over an `UnsafeMutableBufferPointer` (or reuse a single scratch array) so the normalized PCM is written straight into the buffer without an intermediate `map` allocation.

## Coverage

- ios/MCPZimChat/Views/PlacesWebView.swift — findings: 1
- ios/MCPZimChat/Views/RootView.swift — clean
- ios/MCPZimChat/Views/RouteWebView.swift — findings: 1
- ios/MCPZimChat/Views/VoiceChatView.swift — clean
- ios/MCPZimChat/Views/ZimURLSchemeHandler.swift — clean
- ios/MCPZimChat/Voice/KokoroAssets.swift — clean
- ios/MCPZimChat/Voice/KokoroDownloader.swift — findings: 1
- ios/MCPZimChat/Voice/ObjCExceptionWrapper.h — clean
- ios/MCPZimChat/Voice/ObjCExceptionWrapper.m — clean
- ios/MCPZimChat/Voice/SpeechRecognizerService.swift — clean
- ios/MCPZimChat/Voice/TTSService.swift — findings: 1
# batch-18 performance review

Files reviewed: VoiceChatController.swift, ConversationalEvalTests.swift, GemmaToolEmissionTests.swift, ZimfoIntentsTests.swift, EvalCLI.swift.

## Findings

- [medium] ios/MCPZimChat/Voice/VoiceChatController.swift:775 — `streamAssistantReply` re-runs `Self.sanitizeForSpeech(session.messages[idx].text)` on every 75 ms poll of the generation loop (`while !Task.isCancelled` → `try? await Task.sleep(nanoseconds: 75_000_000)` at line 866), even during polls where the assistant text is unchanged (`full.count == spokenUpTo` skips only the speakable-prefix branch, not the sanitize). `sanitizeForSpeech` performs 4 `replacingOccurrences(of:options:.regularExpression)` scans plus several `range(of:)` scans over the ENTIRE accumulated assistant reply, which grows throughout on-device generation. The sanitized result only changes when new text arrives, so this is pure redundant recomputation: cost scales as reply length × (generation duration / 75 ms), i.e. for a 10 s generation over a multi-thousand-char reply that is ~130 full-text regex passes instead of a handful — measurable CPU waste that competes with the on-device ML decode for the shared core. Smallest safe fix: track the last sanitized raw-text length (or a dirty flag set only when `session.messages[idx].text` grows) and recompute `full` only when the text changed; otherwise reuse the previous `full`.

## Coverage
- ios/MCPZimChat/Voice/VoiceChatController.swift — findings: 1
- ios/MCPZimChatMacTests/ConversationalEvalTests.swift — clean
- ios/MCPZimChatMacTests/GemmaToolEmissionTests.swift — clean
- ios/MCPZimChatMacTests/ZimfoIntentsTests.swift — clean
- ios/MCPZimEval/EvalCLI.swift — clean
# Batch 19 — Performance review

Reviewed for performance only, applying performance-review + ios-performance-review checklists. All four Swift files are headless CLI eval/test harnesses (not production hot paths), and `project.yml` is a build-config spec. Every loop found is over small, bounded data; none has an unbounded growth driver.

Findings: none.

Notes / dismissed candidates:
- `EvalHarness.swift:839` `scorecard.rows.filter` per scenario is O(V·S²), but the matrix is fixed at V≈14 variants × S≈17 scenarios (~238 rows max) — bounded eval, no growing input. Dismissed.
- `EvalHarness.swift:966` `session.messages.lastIndex { $0.role == .assistant }` and `:976` `debugEntries.suffix(40)` are O(messages)/O(n) per turn, but messages reset via `resetConversation()` between scenarios and debug entries are capped at `maxDebugEntries = 20_000`; bounded. Dismissed.
- `EvalHarness.swift:779` `out +=` string accumulation in `markdown()` over ≤238 rows — bounded accumulation, no unbounded loop. Dismissed.
- `LlamaCppProbeCLI.swift:159` `firstText += chunk` is guarded by `if firstText.count < 200` (bounded at 200 chars); `:207` `reply += chunk` is bounded by `maxTokens: 48`. Dismissed (bounded accumulation).
- `ProbeCompareCLI.swift` / `ProbeE2ECLI.swift` `args = args.dropFirst()` in `while` loops is the O(n²) ArraySlice pattern, but the arg array is a handful of CLI flags. Dismissed (bounded).
- `ProbeE2ECLI.swift:819-823` `PeakMem` sampler polls at 150 ms — deliberate peak-RSS measurement for the harness's jetsam metric, not a defect. Dismissed.
- `ProbeE2ECLI.swift:851` `session.messages.last`, `:857` `Array(Set(...)).sorted()` — bounded within a conversation. Dismissed.

## Coverage
- ios/MCPZimEval/EvalHarness.swift — clean
- ios/MCPZimEval/LlamaCppProbeCLI.swift — clean
- ios/MCPZimEval/ProbeCompareCLI.swift — clean
- ios/MCPZimEval/ProbeE2ECLI.swift — clean
- ios/project.yml — clean
# Batch 20 — performance review

Reviewed for PERFORMANCE ONLY (no security/style findings). Files reviewed:
Python MCP server (`mcpzim/*.py`, `pyproject.toml`), iOS dev shell scripts
(`ios/scripts/*.sh`, `ios/tools/eval.sh`), and Swift MCPZimKit sources
(`swift/Sources/MCPZimKit/*`, `swift/Package.swift`, `swift/Examples/...`).
Cross-file context traced: `server.py` search/get_article handlers call
`content.search_zim`/`fetch_article`; `content._hit_for` is the per-hit parse
site; `server.plan_driving_route`/`route_from_places` call `plan_route` →
`graph.nearest_node`.

## Findings

- [medium] mcpzim/routing.py:186 — `nearest_node` is a linear scan over every node (`for i in range(self.num_nodes): ... d = dlat*dlat + dlon*dlon`) and `plan_route` (routing.py:398-399) calls it twice per request. — Graph size grows with map extent (city street graphs are hundreds of thousands to millions of nodes); every `plan_driving_route` / `route_from_places` request does 2 O(n) full scans (2M nodes ≈ 4M float ops + cache misses per route), adding tens of ms on the hottest routing path. — Build a spatial index (uniform grid or k-d tree) once in `Graph.parse`, or on first `graph_for`, and query the bucket nearest to (lat, lon) instead of scanning all nodes; the `cos(lat)` hoist already done here is not enough to save an unbounded scan.

- [medium] mcpzim/content.py:304 — `_hit_for` does a full article fetch + full `html_to_text` BeautifulSoup parse per search hit (`snippet = _snippet(html_to_text(_decode(bytes(item.content))), query)`), and `search_zim` (content.py:279-290) calls it once per returned path, up to `limit`. — `search` (server.py:76-95) caps `limit` at 50 per ZIM, so one search can serially BS4-parse up to 50 full HTML articles (plus again per extra loaded ZIM); a Wikipedia search that hits 50 large pages turns a 1-result-latency search into many seconds of parse work. — Cap the number of hits that get a full snippet parse (e.g. parse only the top `min(limit, 8)` hits, or extract the snippet from a truncated prefix of the raw HTML rather than the whole article), and/or skip the snippet parse when the caller doesn't need it.

- [medium] swift/Sources/MCPZimKit/ArticleSections.swift:110 — section slicing uses `html.index(html.startIndex, offsetBy: marker.headingEnd)` / `html.index(html.startIndex, offsetBy: nextHeadingStart)` (and builds `Marker.headingStart/headingEnd` via `html.distance(from: html.startIndex, to:)` at lines 95-96), each of which walks the whole string from the start. — For a large article (≈400 KB, hundreds of `<h2>/<h3>` headings) the parse loop does O(m) index walks per section → O(m·n) character traversal per article parse, and parse runs on the article-fetch hot path (every `list_article_sections`/section retrieval). — Keep a running `String.Index` cursor while walking the matches instead of recomputing offsets from `startIndex` each time, or slice via `NSString`/`NSRange` (`ns.substring(with:)`) which is O(1) per substring.

- [low] mcpzim/content.py:118-121 — `html_to_text` runs ~30 `soup.select(sel)` CSS-select passes over the full DOM (`for sel in _STRIP_SELECTORS: for el in soup.select(sel): el.decompose()`), each O(tree). — Every article parse pays a ~30× full-tree traversal constant factor on the article-fetch hot path (fetch_article/fetch_main_page and each search-hit snippet). — Select all stripped nodes in one pass (e.g. a single combined selector or one `find_all` walk filtering by the `_STRIP_SELECTORS` classes/tags) instead of 30 separate tree walks.

## Coverage

- ios/scripts/mcp-crashes.sh — clean
- ios/scripts/mcp-deploy-verify.sh — clean
- ios/scripts/mcp-logs.sh — clean
- ios/scripts/mcp-report.sh — clean
- ios/tools/eval.sh — clean
- mcpzim/__init__.py — clean
- mcpzim/__main__.py — clean
- mcpzim/cli.py — clean
- mcpzim/content.py — findings: 2
- mcpzim/geocode.py — clean
- mcpzim/library.py — clean
- mcpzim/routing.py — findings: 1
- mcpzim/server.py — clean
- pyproject.toml — clean
- swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift — clean
- swift/Package.swift — clean
- swift/Sources/MCPZimKit/ArticleHeuristics.swift — clean
- swift/Sources/MCPZimKit/ArticleSections.swift — findings: 1
- swift/Sources/MCPZimKit/ChatToolCallParser.swift — clean
- swift/Sources/MCPZimKit/ChatTurn.swift — clean
# Batch 21 — performance review (swift/Sources/MCPZimKit)

Scope: PERFORMANCE ONLY. Specialized skill: ios-performance-review (Swift on-device kit; the
hot paths are the per-turn intent router and per-article drift/embedding extraction, all pure
CPU text/math — no UI, no GCD, no I/O on these files). Every finding names the input that
grows.

## Findings

- [high] swift/Sources/MCPZimKit/IntentRouter.swift:1231 — `match` compiles an `NSRegularExpression` on every call (`guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])`), and `matches` (line 1221) does the same; `classify` (line 84) invokes `match` ~15–20 times per user turn plus regex `replacingOccurrences` calls (lines 270–273, 414–416), and `extractFoundationFact`/`extractPlaceOriginFact` call `matches` per sentence in a loop (lines 1576–1585, 1640). — consequence: the fast-path router, which runs on every user turn to avoid an LLM prefill, pays a regex compilation (tens of µs each on-device) for every pattern attempt — measurable CPU/latency on each turn, and per-sentence compiles scale with article length. — smallest safe fix: precompile the ~30 constant patterns once (static `NSRegularExpression` constants or a `pattern -> regex` cache), and hoist the `hasVerb`/`hasYear`/`matches` regexes out of the sentence loops.

- [medium] swift/Sources/MCPZimKit/ConversationThreads.swift:610,656,701 — `WikiLinks.parseLinks` and `proseParagraphs` compile an `NSRegularExpression` per call (`try? NSRegularExpression(...)` inside each function), and `decodeAndStrip` compiles the `"<[^>]+>"` regex per link inside the `parseLinks` match loop (line 700–701 `replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)`). — consequence: per-article drift extraction pays a regex compile per call, and a regex compile per link in the hot loop (bounded to `max` 8 links, so constant-factor but repeated per article/turn). — smallest safe fix: hoist all three patterns to static `NSRegularExpression` constants; use the compiled regex in `decodeAndStrip` rather than `replacingOccurrences`.

- [low] swift/Sources/MCPZimKit/ConversationThreads.swift:664-666 — `proseParagraphs` accumulates `out += html[r]; out += " "` per paragraph — string `+=` accumulation is O(n²) bytes over the article HTML body. — consequence: for a long article body with many `<p>` blocks the concatenation copies the growing string each paragraph; bounded by article size so cold-path, but avoidable. — smallest safe fix: append pieces into a `[String]` and `joined(separator: " ")` once.

- [medium] swift/Sources/MCPZimKit/Gemma4PromptTemplate.swift:78-97 — `renderImpl` accumulates the transcript with `out += userTurnOpen + text + turnClose` and `out += formatSystemTurn(...)` per turn — each `+=` copies the whole accumulated prompt string. — consequence: as the conversation history grows (long walks append turns), rebuilding the prompt each turn is O(n²) in turn count on the per-turn prefill path. — smallest safe fix: build `out` from an array of rendered pieces and `joined()` once.

- [low] swift/Sources/MCPZimKit/Gemma3Template.swift:353-368 — `repairJSON` compiles three regular expressions per call (`replacingOccurrences(of: #",\s*([}\]])"#, ...options: .regularExpression)` etc.) and the `while out.contains(",,")` loop rescans the string each iteration. — consequence: per tool-call parse (each streamed/end-of-stream tool emission) a handful of regex compiles; bounded small JSON bodies so constant-factor. — smallest safe fix: hoist the three patterns to static regexes and replace the `contains(",,")` loop with one `replacingOccurrences`.

- [low] swift/Sources/MCPZimKit/Gemma3Template.swift:210-222 — `firstToolCall` scans the entire accumulated stream buffer with `buffer.range(of:)` for `<tool_call>` and then for each of 4 fence tags (5 full scans) per invocation. — consequence: if the host invokes this per streamed chunk while no tool call has appeared yet, the buffer is re-scanned each chunk — O(n²) over the stream; even a single call late in a long narration scans the whole buffer. — smallest safe fix: single-pass incremental scan tracking the last searched index.

- [low] swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift:34-58 — `firstCall` does `buffer.range(of: "<|tool_call>")` over the whole buffer per call (then a second scan for the close marker). — consequence: same streaming O(n²) risk as Gemma3Template.firstToolCall if invoked per streamed chunk; constant-factor full-buffer scan otherwise. — smallest safe fix: incremental search from the last scanned offset.

- [low] swift/Sources/MCPZimKit/Embeddings.swift:222 — `scores(for:against:)` does `for e in entries where keys.contains(e.key)` — array membership test `keys.contains` is O(m) inside a loop over up to `maxEntries` (2000) entries. — consequence: hidden O(n·m) per kNN re-rank query (n=entries ≤2000, m=drift-thread keys); with ≤4 keys it's ~8000 string comparisons per query — modest but avoidable. — smallest safe fix: `let wanted = Set(keys)` once before the loop, then `wanted.contains(e.key)`.

- [low] swift/Sources/MCPZimKit/Geocoder.swift:164-193 — `rank(records:query:limit:)` calls `name.lowercased()` (allocation) and `lower.range(of: q)` (O(name·q) substring scan) per record over a decoded `search-data/{prefix}.json` chunk, then sorts all scored records. — consequence: on the substring-search fallback path, work scales with chunk record count × query length (a 256-leaf chunk can hold many records), per query; it's the documented fallback so bounded by chunk size. — smallest safe fix: keep the scan but short-circuit `kinds` mismatch before the lowercase/range work (already ordered) and avoid materialising the full scored array when `limit` is small (top-k instead of full sort).

- [low] swift/Sources/MCPZimKit/Geocoder.swift:133 — `prioritizeSubChunkLeaves` computes `leaf == branch || leaf.hasPrefix(branch + "-")` — the `branch + "-"` string concatenation is recomputed for every leaf in the inner loop. — consequence: loop-invariant string build repeated per leaf (up to 256 leaves × up to 4 variants); constant-factor. — smallest safe fix: hoist `let branchPrefix = branch + "-"` out of the leaf loop.

## Coverage

- swift/Sources/MCPZimKit/ConversationFocus.swift — clean
- swift/Sources/MCPZimKit/ConversationThreads.swift — findings: 3
- swift/Sources/MCPZimKit/Embeddings.swift — findings: 1
- swift/Sources/MCPZimKit/Gemma3Template.swift — findings: 2
- swift/Sources/MCPZimKit/Gemma4PromptTemplate.swift — findings: 1
- swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift — findings: 1
- swift/Sources/MCPZimKit/Gemma4ToolFormat.swift — clean
- swift/Sources/MCPZimKit/Geocoder.swift — findings: 2
- swift/Sources/MCPZimKit/GeoMath.swift — clean
- swift/Sources/MCPZimKit/IntentRouter.swift — findings: 1
# batch-22 performance findings (Swift: MCPZimKit templates + tool adapter)

- [medium] swift/Sources/MCPZimKit/QwenChatMLTemplate.swift:149 — `firstToolCall` rescans the whole accumulated streamed buffer on every decode token — `guard let openMarker = buffer.range(of: "<tool_call>") else { return nil }` scans from index 0 of `buffer` each call; on the per-token streaming hot path (ChatSession.swift:3184 calls `extractToolCall(in: buffer)` inside `for try await chunk in selectedModel.generate(...)` with `buffer += chunk` per chunk) this is O(n²) over the growing buffer — a 512-token generation scans the whole prefix ~512 times, and the cost grows quadratically with stream length / maxTokens (worse on macOS/LlamaCpp paths with larger budgets). Smallest safe fix: keep a scan cursor (start `range(of:options:range:)` at the last scanned offset) and set a `seenMarker` flag once `<tool_call>` is found, so only the newly-appended tail is searched per token.

- [low] swift/Sources/MCPZimKit/MemoryProbe.swift:83 — continuous sampler appends to `samples` with no cap — `startContinuous` loops `await self.sample("\(tagPrefix).#\(i)")` every `intervalMs` (default 100 ms) for the probe's lifetime, and each `sample()` does `samples.append(s)`; the array grows monotonically with wall-clock run duration (an hour of continuous sampling ≈ 36k `MemorySample` structs, unbounded on a long soak) — memory grows linearly with time until `stop()`. Smallest safe fix: bound the continuous path (rolling-window cap / coarse downsampling after N samples) or document it as eval-only; the one-shot `sample(_:)` API is fine.

## Notes / dismissed false positives
- `out += …` / `pendingSys += …` string accumulation in `renderTranscript` loops (LFM25Template.swift:62-96, QwenChatMLTemplate.swift:86-114) — dismissed: `out`/`pendingSys` are uniquely-referenced local `var String` buffers whose `+=` append is amortized O(1) via geometric capacity growth in Swift (unlike Python/Java immutable-string `+`), so turns-count growth is not quadratic. No finding.
- MCPToolAdapter.swift loops are all bounded by explicit caps/limits (fetchWikiExcerpts cap 10, buildNearbyStories maxStories*2, didYouMeanTitles limit 3-6, compare_articles ≤4 titles via withThrowingTaskGroup, dynamic schemas built once at registry construction), so no unbounded/N+1/per-item-I/O finding.

## Coverage
swift/Sources/MCPZimKit/LFM25Template.swift — clean
swift/Sources/MCPZimKit/MCPToolAdapter.swift — clean
swift/Sources/MCPZimKit/MemoryProbe.swift — findings: 1
swift/Sources/MCPZimKit/MemoryStats.swift — clean
swift/Sources/MCPZimKit/ModelTemplate.swift — clean
swift/Sources/MCPZimKit/PlacesPayload.swift — clean
swift/Sources/MCPZimKit/QueryComplexity.swift — clean
swift/Sources/MCPZimKit/QwenChatMLTemplate.swift — findings: 1
# Batch 23 — performance findings (Swift MCPZimKit)

- [medium] swift/Sources/MCPZimKit/SZRGGraph.swift:368 — `nearestNode(lat:lon:)` does a linear scan over every node and calls full `haversineMeters` (sin/cos/asin/sqrt) per node: `for i in 0..<numNodes { let d = haversineMeters(lat, lon, self.lat[i], self.lon[i]) ... }` — called twice per driving-route request from ZimService.swift:572-573. At country-scale monolithic graph node counts (millions), each lookup is an O(n) trig scan costing tens of ms, adding tens of ms to every route request. Fix: use the squared-euclidean e7 distance scan like `nearestNodeSpatial` (Router.swift:206) or add a spatial index / k-d tree; the comment at line 366-367 already flags this as a potential hotspot.

- [low] swift/Sources/MCPZimKit/Router.swift:196 — `nearestNodeSpatial(index:lat:lon:)` linear-scans the full eager `nodesScaled` table (`while i + 1 < n` over `nodes.count`) for each origin/destination of a spatial route (ZimService.swift:563-564). For continent-scale spatial ZIMs with millions of eager nodes this is an O(n) scan per routing request, though the squared-e7 math is cheap per node. Fix: a lat/lon bucketed or k-d spatial index over the eager node table; the JS viewer's linear scan is the only reason it mirrors this.

- [low] swift/Sources/MCPZimKit/Router.swift:76 — monolithic `aStar(graph:origin:goal:)` allocates three whole-graph-sized arrays per routing request (`gScore = [Double](repeating: .infinity, count: n)`, `cameFromPrev`, `cameFromEdge` sized by `graph.numNodes`), so a country-scale graph (millions of nodes) allocates ~16 MB of node-state per request on a hot path, plus the queue. It's the documented array-vs-dict tradeoff for cache locality, but the O(n) per-call allocation is still churn. Fix: reuse a caller-owned scratch buffer sized to the visited frontier, or reset only the touched indices across calls.

- [low] swift/Sources/MCPZimKit/ReferenceResolver.swift:589 — `firstMatch(_:pattern:)` compiles an `NSRegularExpression(pattern:)` on every call (`guard let re = try? NSRegularExpression(pattern: pattern)`), and `resolve` invokes it a handful of times per user turn (list-selection "number N"/"#N", the attribute-question regex). Regex compilation is far more expensive than the match and repeats the same patterns each turn. Fix: hoist the patterns to `static let` compiled constants.

## Coverage
swift/Sources/MCPZimKit/ReferenceResolver.swift — findings: 1
swift/Sources/MCPZimKit/Router.swift — findings: 2
swift/Sources/MCPZimKit/StreamingSpeechPolicy.swift — clean
swift/Sources/MCPZimKit/StubZimService.swift — clean
swift/Sources/MCPZimKit/SZRGChunked.swift — clean
swift/Sources/MCPZimKit/SZRGEncoder.swift — clean
swift/Sources/MCPZimKit/SZRGGraph.swift — findings: 1
swift/Sources/MCPZimKit/SZRGSpatial.swift — clean
swift/Sources/MCPZimKit/ToolLoopGuard.swift — clean
swift/Sources/MCPZimKit/ZimReader.swift — clean
# Batch 24 — performance findings (Swift MCPZimKit)

- [high] swift/Sources/MCPZimKit/ZimService.swift:896 — the `nearPlaces` full-scan fallback calls `loadChunk(pair: pair, prefix: prefix)` with the default `cache: true`, pinning every prefix chunk it touches into the actor's `chunks` dict (`chunks[pair.name]?[prefix]` is written at loadChunk.swift:1632-1636 with no eviction/TTL/max). The single-leaf geocode path at line 681 (`matching = try loadChunk(pair: pair, prefix: leaf)`) likewise caches. The code's own comment at 674-675 and 686-687 acknowledges multi-MB shards pinning hundreds of MB as the jetsam trigger, and deliberately uses `cache: false` for fan-out leaf scans — but these two cache-true sites bypass that. Consequence: across a session's repeated `near_places`/`geocode` calls on a country-scale ZIM, every distinct prefix ever loaded stays resident forever; monotonic multi-GB growth that the code already observed jetsamming the app at 5.4 GB RSS. Fix: pass `cache: false` at lines 896 and 681 (like the fan-out scans), or add an LRU/max-size cap on `chunks`.

- [medium] swift/Sources/MCPZimKit/ZimService.swift:915 — `summarize(hits:limit:)` sorts the entire in-radius hit array (`let sorted = hits.sorted { $0.1 < $1.1 }`) and builds a string dedup key per hit (`"\(p.name.lowercased())|\(cellLat)|\(cellLon)"`, lines 919-922) before returning only `deduped.prefix(max(1, limit))`. `scanRecords` (lines 937-1039) appends every record inside the radius with no cap, so a generic "what's around me" query on dense urban data (up to the 500k-record scan cap at line 886) can collect tens of thousands of hits, making each request O(n log n) sort + per-item string allocation even though the caller asked for a tiny top-N. Fix: keep a bounded top-K during `scanRecords` (e.g. a small nearest priority queue sized ~limit*4, which still lets the dedup collapse duplicates) instead of sorting the full set.

- [medium] swift/Sources/MCPZimKit/ZimService.swift:363 — `leadSnippet(from:path:maxChars:)` does a full article read (`pair.reader.read(path:)` → whole `entry.content` decoded to String) and a full `ArticleSections.parse(html:)` of the entire article body just to return the first section's first 220 chars. It is called per candidate hit inside `search` (lines 257 and 272), over `overfetch = max(limit*2, 10)` title hits plus the same per variant. Consequence: each search request performs up to ~limit*2×variants full-article reads+parses (each hundreds of KB on real Wikipedia ZIMs) to produce a short snippet — per-item I/O on the model's hot search path. Fix: extract only the lead — parse just the first `<p>`/lead section and truncate the read — rather than a full `ArticleSections.parse` over the whole body.

## Coverage
swift/Sources/MCPZimKit/ZimService.swift — findings: 3
swift/Tests/MCPZimKitTests/ArticleFactoidTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleHeuristicsCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleSpeechCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/BundledArticleTests.swift — clean
swift/Tests/MCPZimKitTests/ChatToolCallParserTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationContinuationTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationFocusTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationThreadsTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussArticleLinkTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussionRetrievalTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean
# Batch 25 — performance findings (Swift test files, anchored to production hot paths they exercise)

Format: `- [severity] path:line — issue — concrete consequence — smallest safe fix`
Anchors are test files in this batch; the offending production statement is cited by `Sources/...:line`.

- [medium] swift/Tests/MCPZimKitTests/IntentRouterTests.swift:19 — `IntentRouter.match`/`matches` compile a new `NSRegularExpression` on every call (`guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])`, IntentRouter.swift:1232, and :1221), and `classify` invokes them ~15-20 times per user turn (each with a distinct literal pattern); `ReferenceResolver.firstMatch` (ReferenceResolver.swift:590) does the same per turn — growing input: user turns, each paying tens of regex compilations — consequence: 1-3 ms of regex-compile-only work per on-device voice turn on the classification hot path, plus allocation churn — fix: hoist the literal patterns to `static let` compiled `NSRegularExpression`s (or use Swift `range(of:options:.regularExpression)` which uses Swift's engine and doesn't compile NSRegex) and reuse them.
- [medium] swift/Tests/MCPZimKitTests/StreamingSpeechPolicyTests.swift:138 — `StreamingSpeechPolicy.takeSpeakablePrefix` does `let chars = Array(text)` (StreamingSpeechPolicy.swift:44), materialising a Character array of the entire growing reply buffer on every call; the policy is invoked once per streamed token for streaming TTS, so a reply of length N characters is re-copied into an array N times → O(n²) allocation over the reply — consequence: CPU churn proportional to (reply length)² on the TTS policy path, noticeable on long answers — fix: track the consumed index and operate on the buffer via indices/substrings instead of rebuilding `Array(text)` per call.
- [low] swift/Tests/MCPZimKitTests/QwenClippedToolCallTests.swift:61 — the streaming tool-call parser re-scans the whole growing buffer every token: `QwenChatMLTemplate.firstToolCall` calls `buffer.range(of: "<tool_call>")` then `buffer.range(of: "</tool_call>")` (QwenChatMLTemplate.swift:149-152), and when it returns nil `ChatSession.extractToolCall` (ChatSession.swift:6158-6163) falls through to `ChatToolCallParser.firstCall`, which scans three openers + a brace walk over the whole buffer (ChatToolCallParser.swift:45-61,152) — growing input: the reply buffer, re-scanned once per streamed token → O(n²) over a long reply — consequence: quadratic string scans during generation; for 500-token replies this is ~9M char ops, low on CPU but wasteful and grows with maxTokens — fix: keep a cursor/offset of the last scanned position and only scan appended suffixes (incremental parse state).
- [medium] swift/Tests/MCPZimKitTests/SZRGGraphTests.swift:66 — `SZRGGraph.nearestNode` (SZRGGraph.swift:368-375) is a linear scan computing `haversineMeters` for every graph node; `planDrivingRoute` calls it twice per route (origin+goal) — growing input: graph node count, which is millions on country-scale ZIMs — consequence: ~2×N trig-heavy haversine computations per driving route (e.g. 2M nodes → ~4M trig calls), a visible per-route latency on large ZIMs — fix: build a k-d tree / spatial hash over `(lat, lon)` at parse time (comment at SZRGGraph.swift:365 already flags this).
- [medium] swift/Tests/MCPZimKitTests/SZRGSpatialTests.swift:313 — `nearestNodeSpatial` (Router.swift:196-212) linearly scans the full `nodesScaled` table (millions of entries on a spatial-chunked country ZIM) for origin and goal per route — growing input: node count — consequence: O(n) squared-distance scan per route, dominating route latency on large spatial ZIMs — fix: bucket nodes into the existing SZCI cell grid and search only nearby cells, or build a k-d tree.
- [medium] swift/Tests/MCPZimKitTests/EmbeddingsTests.swift:97 — `EmbeddingIndex.nearest` (Embeddings.swift:195-206) materialises a `Hit` for every entry and sorts the entire scored array (`scored.sorted { $0.score > $1.score }`) just to return `prefix(k)`; on the per-follow-up semantic-recall path with maxEntries=2000 this allocates 2000 Hit structs and does an O(n log n) sort per query — growing input: index size up to maxEntries (2000) × recall queries — consequence: needless allocation + full-sort latency per follow-up; use a partial top-k selection (e.g. a k-sized heap) over the dot-product loop instead of sort-all-then-prefix.
- [low] swift/Tests/MCPZimKitTests/EmbeddingsTests.swift:110 — `EmbeddingIndex.scores` does an array membership test inside a loop over every entry: `for e in entries where keys.contains(e.key)` (Embeddings.swift:222) where `keys` is `[String]` — growing input: entries (≤2000) × requested keys, making this O(n·k) with a linear `contains` per entry; the sibling `centroid(of:)` (Embeddings.swift:211) already builds `Set(keys)` — consequence: O(n·k) membership scans on the drift-thread re-rank path — fix: `let set = Set(keys)` once, then `set.contains(e.key)`.
- [low] swift/Tests/MCPZimKitTests/EmbeddingsTests.swift:38 — `HashingEmbedder.embed` (Embeddings.swift:101-121) allocates per token: `Array("<" + token + ">")`, then a new `String(padded[j ..< j+3])` per character window plus a `"n:" + …` concatenation and two `fnv1a` passes per feature — growing input: tokens per title/query (embedding runs for every article/thread touched during a walk) — consequence: per-token allocation churn proportional to token length; bounded by article count but wasteful — fix: walk the UTF-8 bytes of the token directly (or use a single reusable buffer) and hash subwords without building `String` slices.
- [medium] swift/Tests/MCPZimKitTests/HotSplitGeocoderTests.swift:90 — `DefaultZimService.geocodeResolved` (ZimService.swift:679-717) reads every sub-chunk leaf of a hot prefix when the query has no exact match: `prioritizeSubChunkLeaves` retains all leaves and the loop `loadChunk(..., cache: false)` runs over all of them (256 on a full `st` fan-out), with the early-exit only firing on a literal exact-name match and the `matching.count >= max(200, limit*8)` cap only stopping once 200+ substring matches accumulate — growing input: hot-prefix leaf count × miss queries — consequence: a substring/no-exact geocode can issue up to 256 uncached multi-MB chunk reads per query per streetzim, adding per-query I/O latency — fix: stop reading further leaves once `matching` reaches the cap even for substring queries (the cap check already exists but only applies post-exact-match), or cache leaf parses keyed by chunk name.
- [low] swift/Tests/MCPZimKitTests/NearPlacesChipIndexTests.swift:90 — `nearPlaces` collects every in-radius record into `hits` and `summarize` (ZimService.swift:908-935) then sorts all of them and builds a dedup string key (`"\(p.name.lowercased())|\(cellLat)|\(cellLon)"`, allocating a lowercased copy per hit) — growing input: in-radius records, which for a generic "what's around me" query on a state-scale ZIM (under `maxFullScanRecords`) can be tens of thousands — consequence: per-query sort + per-hit string allocations over all in-radius rows, not just the requested `limit` — fix: cap the in-radius accumulation to a bound (e.g. limit×N) during `scanRecords`, or dedup with a packed numeric key instead of a lowercased-string key.

## Coverage
- swift/Tests/MCPZimKitTests/EmbeddingsTests.swift — findings: 3
- swift/Tests/MCPZimKitTests/Gemma4PromptTemplateTests.swift — clean
- swift/Tests/MCPZimKitTests/Gemma4ToolFormatTests.swift — clean
- swift/Tests/MCPZimKitTests/HotSplitGeocoderTests.swift — findings: 1
- swift/Tests/MCPZimKitTests/IntentRouterTests.swift — findings: 1
- swift/Tests/MCPZimKitTests/LFM25TemplateTests.swift — clean
- swift/Tests/MCPZimKitTests/LocateToolTests.swift — clean
- swift/Tests/MCPZimKitTests/NearPlacesCenterResolutionTests.swift — clean
- swift/Tests/MCPZimKitTests/NearPlacesChipIndexTests.swift — findings: 1
- swift/Tests/MCPZimKitTests/NearPlacesWikiEnrichmentTests.swift — clean
- swift/Tests/MCPZimKitTests/PlacesPayloadTests.swift — clean
- swift/Tests/MCPZimKitTests/QueryComplexityTests.swift — clean
- swift/Tests/MCPZimKitTests/QwenClippedToolCallTests.swift — findings: 1
- swift/Tests/MCPZimKitTests/ReferenceResolverTests.swift — clean
- swift/Tests/MCPZimKitTests/SanitizedWikiTagTests.swift — clean
- swift/Tests/MCPZimKitTests/SanitizeZimArgTests.swift — clean
- swift/Tests/MCPZimKitTests/StreamingSpeechPolicyTests.swift — findings: 1
- swift/Tests/MCPZimKitTests/SZRGGraphTests.swift — findings: 1
- swift/Tests/MCPZimKitTests/SZRGSpatialTests.swift — findings: 1
# Batch-26 performance review findings

No findings. All 15 listed files were read in full. They are one-off batch / fine-tune /
benchmark / unit-test scripts, not request hot paths, so per the performance-review skill
the CLI/one-off-script exemption applies to most anti-patterns. Candidates examined and
dismissed:

- `tools/fine-tune/generate.py:880`, `generate_chains.py:236`, `generate_places_diverse.py:403`,
  `generate_chains3.py:938,942` — `out_fh.flush()` / `fail_fh.flush()` per row. Deliberate
  checkpointing: every generator counts existing lines on disk to resume (`sum(1 for _ in fh)`
  before generation), so flush-per-row keeps the resume count accurate on crash. Skill
  false-positive bullet (deliberate checkpoint/restart semantics) applies; also generation is
  ~1 row/s so a flush syscall per row is negligible. Dismissed.
- `generate_places_diverse.py:289` — `hits = sum(1 for n in names if n and n in low)`: O(names ×
  reply_len) substring scans, but names ≤200 (bucket capped) and reply ≤600 chars; runs once per
  generated example (n=400), bounded one-off batch validator. Dismissed (bounded input).
- `generate.py:889`, `generate_chains.py:245`, `generate_places_diverse.py:412`,
  `generate_chains3.py:960` — `asyncio.gather`/`create_task` over the query/seed list, but every
  generator bounds actual concurrency with `asyncio.Semaphore(concurrency)` (default 4) and the
  list is bounded by the CLI `--n` (100–800). No unbounded fan-out. Dismissed.
- `finetune_lfm2.sh:157` — `new_tensors[nk] = t[xid].clone()` per expert during the one-off
  fused-HF→per-expert conversion. The original stacked expert tensor is released each loop
  iteration (local reassign), so peak RSS ≈ shard size + one stacked expert tensor; inherent to
  the safetensors per-expert format and a one-off conversion. Dismissed.
- `eval_ft_pcgaming.sh`, `compare.sh` — subprocess-per-cell/model runs are deliberate
  benchmarking (each cell is a full model load+eval); the script documents the CPU-vs-CUDA
  throughput tradeoff. Dismissed (inherent workload).
- `retry_lfm2_train.sh` — `highest_ckpt()` pipeline and `sleep 5` retry loop are bounded by
  `MAX_TRIES=8`. Dismissed (bounded).
- 95/5-split blocks in finetune_cuda.sh / finetune_unsloth.sh / finetune.sh / finetune_lfm2.sh /
  retry_lfm2_train.sh load all rows into memory once (`rows = [l for l in open(...)]`) and
  shuffle — one-off, bounded by training-file size. Dismissed.
- Swift test files (`SZRGv5AndChunkedTests.swift`, `ToolLoopGuardTests.swift`) are test code
  building small fixed buffers; not hot paths.

## Coverage
- swift/Tests/MCPZimKitTests/SZRGv5AndChunkedTests.swift — clean
- swift/Tests/MCPZimKitTests/ToolLoopGuardTests.swift — clean
- tools/bonsai-ab/compare.sh — clean
- tools/fine-tune/convert_to_lfm2_native.py — clean
- tools/fine-tune/eval_ft_pcgaming.sh — clean
- tools/fine-tune/finetune_cuda.py — clean
- tools/fine-tune/finetune_cuda.sh — clean
- tools/fine-tune/finetune_lfm2.sh — clean
- tools/fine-tune/finetune_unsloth.py — clean
- tools/fine-tune/finetune_unsloth.sh — clean
- tools/fine-tune/finetune.sh — clean
- tools/fine-tune/generate_chains.py — clean
- tools/fine-tune/generate_chains3.py — clean
- tools/fine-tune/generate_places_diverse.py — clean
- tools/fine-tune/generate.py — clean
- tools/fine-tune/retry_lfm2_train.sh — clean
# Batch 27 — performance findings

All files in this batch are CLI/dev-tool scripts (fine-tune pipelines, smoke-test harnesses, eval grids). Reviewed each for the performance-review checklist (N+1, hidden-O(n), allocation churn, unbounded caches, per-item I/O); the MLX and Swift lenses were applied to the MLX/Swift files. These are one-shot developer tools, so severity is calibrated to their actual (bounded) inputs rather than production request paths.

- [medium] tools/llm-smoke/bench_memory_gemma4.py:40 — O(n²) preamble build: `build_preamble` re-encodes the entire accumulated string on every loop iteration — `ids = tokenizer.encode("".join(chunks))` then `chunks.append(...)` then next iteration re-encodes all prior paragraphs — and `"".join(chunks)` rebuilds the whole string each pass. With default `--sizes 7000,20000,40000` the final target builds ~40k tokens by ~1k+ iterations, each re-encoding an average of ~20k tokens plus a full string copy: roughly O(target²) ≈ tens of millions of tokenizer ops and quadratic string churn, so the memory bench spends most of its wall time just constructing the preamble it measures. — Smallest safe fix: encode incrementally — keep a running list of accumulated token ids, append only the newly added paragraph's encode per iteration, track cumulative length against target, and decode once at the end (or use a chars/4 token estimate like split_chain_rows.py does).

- [low] tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift:533 — per-token full re-decode of the whole accumulated token array inside the generation stream loop: `decodedSoFar = context.tokenizer.decode(tokens: tokenIDs.map { Int($0) })` on every `.token` event re-decodes all tokens so far (O(n) per token → O(n²) decode work per generation) and `tokenIDs.map { Int($0) }` allocates a fresh [Int] array per token. Growth axis is generated tokens (bounded by maxTokens 60–120 in this harness, so impact is low), but it is the same hot-streaming anti-pattern this smoke tool exists to measure. Same pattern at line 259 (runBehaviorTest) and line 662 (CacheSim.runGenerate, marked UNUSED). — Smallest safe fix: decode incrementally (keep the last emitted text and decode only the appended token/span), or decode once at the end; drop the per-token `map` by decoding from the Int32 array/slice directly.

## Coverage
- tools/fine-tune/split_chain_rows.py — clean
- tools/fine-tune/train_all_cuda.sh — clean
- tools/fine-tune/train_all.sh — clean
- tools/fine-tune/v7_eval_and_memsweep.sh — clean
- tools/gemma-smoke/Package.swift — clean
- tools/gemma-smoke/Sources/GemmaSmoke/main.swift — clean
- tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — findings: 1
- tools/llama-smoke/bench.py — clean
- tools/llama-smoke/eval.py — clean
- tools/llama-smoke/grid.py — clean
- tools/llama-smoke/sweep.sh — clean
- tools/llm-smoke/bench_kv.py — clean
- tools/llm-smoke/bench_memory_gemma4.py — findings: 1
# Batch 28 — tools/llm-smoke/* (MLX LLM bench/eval smoke scripts)

## Findings

- [medium] tools/llm-smoke/bench_memory.py:100 — `build_preamble` re-tokenizes the entire growing preamble every loop iteration (`ids = tokenizer.encode("".join(chunks))`), an O(n²) BPE workload — for target_tokens=20000 the loop runs ~200 iterations, each re-encoding the whole accumulated string (~2M tokens of tokenizer work vs the ~20k actually needed, ~100x waste); `run()` then re-encodes the decoded preamble a second time at line 144, so the final preamble is tokenized twice per (model, variant, size) across the ~30 bench runs in `main()` — hoist/accumulate token ids incrementally (encode only the newly appended paragraph and keep a running length) and return the ids from `build_preamble` to avoid the second full encode.

- [low] tools/llm-smoke/eval_gemma4_native.py:63 — `buffer += text` accumulates the decoded output as a string inside the decode loop (O(n²) bytes for long outputs) while the `decoded` list already holds the same text, and each token runs a substring scan `any(m in buffer for m in STOP_MARKERS)` at line 66 — negligible at the 300-token default, but with a large `--max-tokens` the `+=` copies grow quadratically and `buffer` duplicates `decoded`; drop `buffer` (or join at checkpoints) and only keep a small rolling tail for marker detection.

- [low] tools/llm-smoke/eval.py:417 — `json.dumps(TOOLS, ensure_ascii=False)` plus the whole `sys_with_tools` schema string are rebuilt inside `build_prompt()`, which is called once per case inside the case loop (9 cases) even though TOOLS/SYSTEM are module-level constants — negligible for 11 tools/9 cases, but hoist the serialized schema to module scope since it is invariant constant data re-computed per case.

## Coverage
tools/llm-smoke/bench_memory.py — findings: 1
tools/llm-smoke/bench.py — clean
tools/llm-smoke/eval_gemma4_native.py — findings: 1
tools/llm-smoke/eval_gemma4.py — clean
tools/llm-smoke/eval.py — findings: 1
tools/llm-smoke/gemma4_format.py — clean
# Batch 29 — Performance review (tests/)

Files reviewed are test modules. Per the performance-review skill, tests are not hot paths; the only loops present iterate over bounded, fixed test fixtures (4-node grid graphs, 3-point polylines), so there are no defensible performance findings. No clients, caches, unbounded collections, or per-item I/O appear in any file.

## Findings

## Coverage
tests/__init__.py — clean
tests/test_content.py — clean
tests/test_geocode.py — clean
tests/test_library.py — clean
tests/test_routing.py — clean
# Batch 30 — performance review findings

File reviewed: `swift/Sources/MCPZimKit/MCPToolAdapter.swift` (2859 lines, Swift, MCP tool dispatch/registry). Applied `performance-review` checklist plus `ios-performance-review` (Swift platform code; no iOS-specific UI/thread issues found — the adapter is pure Swift with an actor, no MainActor/DispatchQueue misuse).

- [medium] swift/Sources/MCPZimKit/MCPToolAdapter.swift:113 — O(n) nearest-polyline haversine scan on every `route_status` call, with the fixed GPS origin's trig recomputed per vertex — `remaining(at:)` loops `for (i, p) in polyline.enumerated() { let d = Self.haversineMetersApprox(current.lat, current.lon, p.lat, p.lon) ... }` and `haversineMetersApprox` (lines 126–137) recomputes `rlat1`, `cos(rlat1)` for the constant `current` point on every vertex. `dispatchRouteStatus` (line 2157) calls `route.remaining(at:)` each dispatch, and `route_status` is the "how much longer?" tool fired repeatedly during navigation. A city drive polyline (hundreds–thousands of vertices from the SZRG graph) means each status check runs thousands of sin/cos/asin ops — ms-level latency per turn, no caching. Fix: exploit monotonic forward progress — keep a cached last-best index in the adapter keyed to the route and search forward from it, and hoist `rlat1`/`cos(rlat1)` out of the loop (a specialized distance-to-vertex function taking the fixed origin trig as arguments).
- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:246 — `registry` computed property re-encodes dynamic schemas on every access — each `registry` read rebuilds the full `[MCPTool]` array and calls `nearPlacesSchema(vocabulary:)` (line 388) and `nearbyStoriesSchema(vocabulary:)` (line 430), which funnel into `schemaJSON` → `JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])` (line 2751). Hosts feed `toolList` to the model every LLM turn, so this is full JSON re-serialization (sortedKeys is O(n log n)) + array/string rebuild of immutable data per turn. Fix: compute the registry once into a stored/lazy property — `hasStreetzim`, `surface`, and `categoryVocabulary` are immutable after init.
- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:1047 — `article_overview` re-reads the full article body twice after `sectionsByTitle` already fetched it — `relatedLinks` (line 1047, body at line 1442 `let article = try await service.article(path: path, zim: zim)`) and `disambiguationAlternates` (line 1071, body at line 1427 `let page = try await service.article(path: path, zim: zim)`) each call `service.article`, which decodes the whole article HTML (`String(data: entry.content, encoding: .utf8)`, ZimService.swift:388). For a large article this adds 2 full-body reads (plus parse) per overview call, dominating I/O on a hot path. Fix: fetch `article.text` once and pass it into both helpers, or reuse the already-fetched section data.
- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:2253 — `fetchWikiExcerpts` iterates `result.results` twice and allocates a filtered array just to count — the candidates loop (lines 2241–2247) scans the result set, then `let totalWikiTagged = result.results.filter { !($0.place.wiki?.isEmpty ?? true) }.count` (line 2253) scans it again and allocates a whole new array for a count. For a 50+ hit result set this is redundant allocation/iteration on the enrichment path. Fix: count wiki-tagged rows inside the single candidates loop (or use `reduce`).

## Coverage
swift/Sources/MCPZimKit/MCPToolAdapter.swift — findings: 4
# Batch 31 performance review

## Findings

- [low] ios/MCPZimEval/EvalHarness.swift:886 — `session.maxDebugEntries = 20_000` retains a 20k-entry debug ring per variant session, but the harness only ever prints `session.debugEntries.suffix(40)` per turn (lines 976-978). Each `DebugEntry` carries a `UUID`, `Date`, and string fields (`DebugReport.swift`/`ChatSession.swift:133-137`), and `resetConversation()` (ChatSession.swift:2445) does NOT clear `debugEntries`, so the ring accumulates across every scenario in a variant run (15 scenarios × 3-4 turns × 20 variants), holding tens of MB of logs the eval never uses. — In a harness whose whole point is measuring peak RSS, the 20k-entry log ring (40× the interactive cap of 500, and the eval only reads the last 40 entries) inflates harness memory and wastes allocation that the memory probe will attribute to the run. — Lower `maxDebugEntries` to ~500-1000 (matching the debug-pane cap) since only `suffix(40)` is ever consumed.

- [medium] tools/llama-smoke/eval.py:738-742 — `MemoryProbe.start()` creates a fresh `psutil.Process()` inside the sampling loop: `proc = psutil.Process()` then `proc.memory_info().rss` every `interval_s` (0.1s) for the whole eval wall-clock. Constructing a `Process` and calling `memory_info()` each incur a syscall (`proc_pidinfo` on macOS), so the daemon does ~10 syscalls/s and competes for the GIL with the main thread — the overhead lands directly on the measured wall_s/timing of the eval. — Every 100ms sample does 2+ syscalls of measurement bookkeeping that inflates the reported `wall_s` and adds CPU contention while llama.cpp decodes; the harness is specifically timing/scoring those runs. — Reuse one `psutil.Process()` instance across all samples (create it once in `start`), or sample via `resource.getrusage(RUSAGE_SELF)` like the existing `rss_mb()` helper instead of per-sample Process construction.

- [low] tools/llama-smoke/eval.py:1396-1400 — `_lfm2_render()` re-serialises the constant `TOOLS_SCHEMA` with `json.dumps(tools_schema)` on every call, and `run_scenario` calls `_lfm2_render` on every tool-loop iteration (up to `TOOL_ITER_BUDGET=8` per turn × turns), so the ~10-schema JSON block is re-built each iteration even though it never changes. — Repeated O(schema-size) JSON stringification + re-render per iteration is pure redundant work in the prompt-build hot loop for LFM2 runs; with 8 iterations × 3 turns × several scenarios it adds up to dozens of needless full serializations. — Serialise the tool block once before the turn loop (or once per scenario) and pass the pre-rendered string into `_lfm2_render`.

## Coverage

ios/MCPZimEval/EvalHarness.swift — findings: 1
tools/llama-smoke/eval.py — findings: 2
# Batch 32 — performance review findings

- [low] tools/fine-tune/generate_chains3.py:941-942 — per-row `out_fh.write(json.dumps(row) + "\n"); out_fh.flush()` flushes the append buffer after every generated row — each row forces a write syscall, defeating buffering across the whole run — with `--n 800` (or higher) this is ~n write syscalls where one flush at a checkpoint would suffice; negligible next to the seconds-per-row teacher LLM call, but it scales with row count and is pure syscall overhead — flush every N rows (e.g. every 10) instead of per row, and rely on the resume-by-line-count logic (line 873-874) still seeing partial rows; keep a final flush before close.

## Coverage

swift/Sources/MCPZimKit/ArticleHeuristics.swift — clean
ios/MCPZimEval/ProbeE2ECLI.swift — clean
tools/fine-tune/generate_chains3.py — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift — clean
# Batch 33 — performance review findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift:245-246 — vision rotary `cos = cos(freqs)` / `sin = sin(freqs)` (plus `expandedDimensions`/`tiled` at 248-254) are recomputed inside `applyMultimodalRotaryPositionEmbedding(_ tensor:freqs:)`, which is called from `Vision.Attention.callAsFunction` (lines 316-317) for every vision block — `VisionModel.callAsFunction` (lines 466-470) runs `for block in blocks { block(hiddenStates, frames: frames, rotaryPositionEmbedding: rotaryPositionEmbedding) }`, so the cos/sin trig + expansion + tile work is re-executed `depth` times per image even though `freqs` is identical across all blocks — each image forward pays depth× redundant elementwise/graph work on the GPU (Qwen2-VL depth is 24+), compounding with every image/video frame in a turn; negligible next to the attention matmuls but it is loop-invariant work on the vision hot path — compute cos/sin (and the expanded/tiled forms) once per image in `VisionModel.callAsFunction` and pass the prepared cos/sin arrays into the attention/blocks instead of recomputing them per block.

## Coverage
tools/fine-tune/generate.py — clean
ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ToolTests.swift — clean
# batch-34 performance review findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift:687 — per-decode-step fancy indexing to build per-layer PLE inputs: `perLayerInputs = (0 ..< config.numHiddenLayers).map { i in combined[.ellipsis, i, 0...] }` creates one strided-slice graph op per layer (35 for this config) on every `model.callAsFunction` invocation, i.e. every generated token. Each slice is a graph op (MLX per-subscript overhead) and each `perLayerInputs[idx]` is consumed once, so 35 slice nodes are constructed per token on the decode hot path. Consequence: ~35 extra graph nodes per token; small vs the layer matmuls but pure per-token churn that grows linearly with `numHiddenLayers`. Smallest safe fix: reshape `combined` once into per-layer contiguous chunks (`combined.reshaped(B, L, config.numHiddenLayers * config.hiddenSizePerLayerInput)` then slice each layer's contiguous `PLE` block), or hoist the PLE slices to the caller and pass them in, avoiding a `.map` of dynamic-range subscripts per token.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift:411 — MoE expert combination silently promotes fp16 activations to fp32: `y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)`. `scores` is fp32 (from `sigmoid(hiddenStates.asType(.float32))` at line 350 and the takeAlong/normalize chain), while `y` (SwitchGLU output) is fp16 — the `.asType(y.dtype)` cast at the end is only needed because the broadcast multiply already promoted `y` to fp32. The elementwise `y * scores[...]` over [B, L, numExpertsPerTok, hiddenSize] then runs in fp32 (2× bandwidth) and the sum result is cast back to fp16. Consequence: every MoE layer, every decode token pays fp32 elementwise bandwidth on the full combination tensor plus a dtype round-trip. Smallest safe fix: cast the (small) scores vector to y's dtype before the broadcast multiply, e.g. `y * scores.asType(y.dtype)[.ellipsis, .newAxis]`, keeping the multiply+sum in fp16.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift:552 — vision encoder collects all hidden states that the caller discards: `let (hiddenState, _, _) = self.visionModel(pixelValues..., outputHiddenStates: true)` uses only the first returned value (the pooler output); the second and third tuple elements are dropped with `_`. With `outputHiddenStates: true`, `Encoder.callAsFunction` (lines 315-317) appends every one of the ~27 layer outputs into a `[MLXArray]` and `SigLipVisionModel` returns `hiddenStates?.last` — all of which are unused. Consequence: per image forward, a 27-element Swift array plus 27 extra MLX graph nodes/retained references are built for zero benefit (the pooler comes from `h = x[0]` of the last layer, not from the collected states). Smallest safe fix: pass `outputHiddenStates: false` (the pooler output is returned independently), eliminating the per-layer `encoderStates?.append(x)` and the `hiddenStates?.last` work.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift:658 — per-decode-step dead array allocation feeding an always-nil mask: `let attnMask: MLXArray? = createAttentionMask(h: h, cache: cache[0]?[1] != nil ? [cache[0]![1]] : nil)` builds a one-element `[KVCache]` array every forward (every generated token) even though `createAttentionMask` (lines 611-620) unconditionally returns `nil` ("Will be handled by SDPA internally when nil"). The `cache[0]?[1] != nil ? [cache[0]![1]] : nil` ternary and its array allocation are pure per-token waste on the decode loop. Consequence: one small heap allocation per token in the hot path; negligible vs the matmuls but trivially removable. Smallest safe fix: call `createAttentionMask(h: h, cache: nil)` (or drop the attnMask argument entirely) since the function ignores its cache argument and returns nil.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift — findings: 1
# batch-35 performance findings

No findings. All four listed files reviewed end-to-end for performance (general
performance-review checklist plus the matching specialized stack checklists:
mlx-performance-review for the two MLX model files, ios-performance-review /
web-performance-review lens for the SwiftUI view, and general checklist for the
test harness).

## Dismissed candidates (documented, not findings)

- GraniteMoeHybrid.swift:535-581 & Qwen35.swift:589-649 `sanitize` — loops over
  every weight key with `key.contains("conv1d.weight")` / `normKeys.contains(where:)`
  and builds lazy graph slices (`inputWeight[0..., ..<halfHidden, 0...]`). These
  are one-time model-load paths iterating over config-bounded weight dicts;
  `.dim`/`.ndim` are shape-metadata reads that do NOT force GPU eval. Not hot-path.
- GraniteMoeHybrid.swift:122-128 / Qwen35.swift:252-255 — per-token conv-state
  `concatenated([convState, input])` and slice-back into the cache. The state is
  exactly (convKernelSize-1) rows (~3), so the concat is O(convDim*convKernelSize)
  constant per token — the standard Mamba rolling-conv pattern, not O(t²).
- Qwen35.swift:529-531 — `Array(repeating: nil, count: layers.count)` per call
  when `cache` is nil. Decode always supplies a cache; the branch is cold and the
  allocation is a ~32-element nil array (trivial). Bounded.
- GraniteMoeHybrid.swift:464-475 / Qwen35.swift:534-535 — attention/SSM masks
  created once per `callAsFunction` (hoisted out of the layer loop). Correct.
- LibraryView.swift:14, 347-359 — `enabledCount` filter and `registryCapabilities()`
  `reduce(into:)` with `acc.contains(x)` recomputed on body render. The input
  (`session.library`) is bounded to a user's handful of loaded ZIMs and the
  capability set is at most 4 distinct strings — no growing input. Dismissed per
  bounded-collection false-positive rule.
- LibraryView.swift:25 — `TimelineView(.periodic(by: 1.0))` re-renders the small
  model-status HStack every second; deliberate and commented for download progress
  honesty. Tiny subtree, not a re-render storm.
- LibraryView.swift:532-537 — `formatBytes` builds a fresh `ByteCountFormatter`
  per call, but only ~4 invocations per settings-pane render (not a loop). Low
  constant-factor waste, not defensible.
- ConversationalEvalTests.swift:434-454, 393-431 — `runTurn`/`makeOrReuseSession`
  poll `isGenerating`/`modelState` at 200/500 ms with deadlines; `toolsCalled`
  scans only entries added since the per-turn `mark` (bounded by that turn's
  output, not all 20k debug entries). Test harness, not a hot path; polling
  intervals are not tight (<100ms). Session is shared/reused across tests so the
  ~5s model load happens once.

## MLX sync-point census (mandatory inventory)

No `.item()`, `.asArray()`, `.tolist()`, `.eval()`, or `asyncEval` in any listed
file. The decode loops (`for (i, layer) in layers.enumerated()`) are bounded over
model layers (config-sized, not token-sized) and contain no host branches on
array values; eval cadence is left to the external generation driver. KV-cache
threading is correct (per-layer `cache?[i]`, MambaCache/KVCacheSimple, state
stored back into cache per token). No forced GPU→CPU sync in any decode path.

## Coverage

- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift — clean
- ios/MCPZimChat/Views/LibraryView.swift — clean
- ios/MCPZimChatMacTests/ConversationalEvalTests.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift — clean
# Batch 36 — performance review

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3Next.swift:272 — loop-invariant scalar tensors rebuilt per decode step in the linear (SSM) layer hot path — `let invScale = pow(Float(headKDim), -0.5)` then `MLXArray(invScale * invScale).asType(dtype)` (line 272) and `MLXArray(invScale).asType(dtype)` (line 275) recompute two scalar MLXArray tensors plus a pow() plus a dtype cast on every token for every linear layer (linear layers are the majority: every layer where `(layerIdx+1) % fullAttentionInterval != 0`, default interval 4). Growth axis: decode tokens × linear layers. Consequence: redundant scalar allocation/cast and pow() work per token in the hottest decode path — constant-factor waste that scales linearly with generated-token count; for a 30-layer model with ~75% linear layers this is ~30 scalar tensor constructions per generated token. Smallest safe fix: hoist `invScale` (and its square) to stored Float properties computed once at init, and construct/cast the scalar MLXArray once per dtype (e.g. lazily memoize `MLXArray(invScale * invScale).asType(dtype)` keyed on dtype) instead of rebuilding it inside `callAsFunction`.

## Coverage
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3Next.swift — findings: 1
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Bert.swift — clean
- swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean
# Batch 37 — SZRGSpatial.swift performance findings

## Findings

- [medium] swift/Sources/MCPZimKit/SZRGSpatial.swift:248 — `edgesOfNode` materializes a fresh `[SpatialEdge]` array (`var out: [SpatialEdge] = []` + `out.reserveCapacity(eEnd - eStart)` + `out.append(...)` per edge, lines 248-259) plus an actor hop per call, on the A* hot path — `Router.aStarSpatial` pops up to `popLimit` (200,000) nodes and calls `graph.edgesOfNode(current)` once per pop (Router.swift:268), so a long route does ~200k array allocations of degree-sized structs plus 200k actor-executor hops, and the cell cache comment itself notes A* "hammers the same cells thousands of times per route" — smallest safe fix: since cells are immutable and cached, expose a non-allocating accessor that returns the cell-local adjacency range/offset into the immutable stride-5 `cell.edges` flat array (or returns the `SZRCCell`/a slice) so the A* loop iterates the flat array directly without materializing a `SpatialEdge` array per expansion; the per-call `cellForNode` dict lookup + `localIdx` binary search stay but the array+hop churn drops.

- [medium] swift/Sources/MCPZimKit/SZRGSpatial.swift:292 — `let task = Task { @Sendable in ... }` created inside actor-isolated `ensureCell` inherits the enclosing actor's executor, so every cell fetch (`self.fetch(cid)` → libzim read / HTTP) runs serially on the single `SpatialGraph` actor executor — concurrent routing across *different* cells (the comment's "Callers that need to route multiple concurrent requests") can't overlap fetch I/O, so N concurrent routes waiting on N different cells pay the sum of fetch latencies instead of the max; `inFlight` dedups same-cell but does nothing for cross-cell parallelism — smallest safe fix: run the fetch on a nonisolated executor (`Task.detached` with `@Sendable`, or a separate fetch actor / `nonisolated` helper) so independent cell fetches overlap while `inFlight` still dedups concurrent same-cell loads.

- [low] swift/Sources/MCPZimKit/SZRGSpatial.swift:410-413 — `SZCI.parse` copies the entire `namesBlob` (`namesBlob.append(raw[off + i])` for all `namesBytes`) into a `[UInt8]` array, and SZRCCell.parse (lines 487-489) copies the entire `geomBlob` the same way — for the v2 continent-scale layout (the file notes a country's node table is too big to inline), a cell's geom blob can be large, so each cached cell/index holds a full second copy of the blob bytes, doubling memory for every cached cell; parse runs once per cell so it is cold-path, but the memory overhead is proportional to total cached geom/name bytes — smallest safe fix: retain the original `Data` and index it via `withUnsafeBytes` in `decodeGeom`/`name` instead of copying the blob into a `[UInt8]` array (cells are immutable/Sendable, so holding `Data` is still actor-safe).

## Coverage

- swift/Sources/MCPZimKit/SZRGSpatial.swift — findings: 3

## Run stats

Engine throughput (weighted across batches): prefill 488476 tok @ 912 t/s, generated 152460 tok @ 23.0 t/s (37 batches)
