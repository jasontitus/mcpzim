# Pi sweep review (perf focus) — mcpzim-ec2c203f

Exhaustive per-file pass: 434 code files across 41 batches.

## Findings

# Performance review — batch-1 (LocalSwarmEngine)

Reviewed all 21 listed files against performance-review + ios-performance-review checklists. The engine is deliberately well-optimized (precomputed chunk lengths, incremental bitfield packing, rate-limited snapshots, windowed in-flight caps, autoreleasepool'd hashing, slice-based ByteReader), so the majority of files are clean. Findings below are cold-path/cache-growth items — no N+1 or O(n²) defect exists on the per-chunk data-transfer hot path (SwarmSession.pump is bounded by the 256 MiB in-flight cap + per-peer windows).

## Findings

- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/AuthToken.swift:11 — swarmAuthToken() encodes a 32-byte SHA-256 digest with per-byte `String(format: "%02x", $0)` — the exact slow pattern `Hashing.hex` was written to replace ("an order of magnitude slower", Hashing.swift). Impact: per auth-token computation; only reached once per PIN-protected session/fetch, so a cold path, but it is pure constant-factor waste (32 formatter calls) on an operation already done. Smallest safe fix: reuse `Hashing.sha256Hex(digest)` (same lowercase-hex output) instead of re-implementing the slow hex.

- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/TransferLogger.swift:83 — `iso()` constructs a fresh `ISO8601DateFormatter()` on every `record()` call, which fires once per status sample (~1/s per active transfer) plus start/complete/pause/cancel events. Formatter init is the ~5×-reuse-cost pattern the iOS checklist calls out (QA1480). Impact: redundant formatter construction on the status-sampling path; small but repeated, and cheap to eliminate. Smallest safe fix: cache a static `ISO8601DateFormatter` (module-level) rather than building one per record.

- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmManager.swift:576 — `runBenchmark` computes the selected total with `indices.reduce { $0 + manifest.length(ofChunk: $1) }` where `length(ofChunk:)` is a documented O(files) linear scan per call, making this O(chunks × files). SwarmSession already precomputes the identical `chunkLengths` array precisely to avoid this pattern. Impact: for a 100 GiB swarm (100k chunks) with thousands of files this sums to tens of millions of file-range comparisons per benchmark leg — a several-second stall before each leg, cold path but user-visible. Smallest safe fix: sum over `manifest.chunkLayout()`'s lengths (one O(chunks) pass) instead of calling `length(ofChunk:)` per index.

- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/ManifestCache.swift:63 — `store()` writes one JSON file per cache key (content-addressed via path+size+mtime+name) into Application Support with no eviction: no TTL, no max entry count, no directory sweep (only a manual `clear()`). Every distinct share layout/name/chunk-size generates a permanently retained entry; a large swarm's manifest JSON can be ~MBs. Impact: Application Support disk usage grows monotonically with the number of distinct shares ever hosted on a device, with no automatic bound. Smallest safe fix: cap/age the cache directory (e.g. keep N most-recent entries or sweep files older than a TTL on access).

- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmManager.swift:426 — `downloadManifests[manifest.swarmID] = manifest` is written in `startDownload` and only read (line 107) / never removed for the life of the session. Each entry retains a full `SwarmManifest` (including an unbounded `chunkHashes` string array — tens of thousands of entries for a large swarm). Impact: in-memory growth proportional to the cumulative number of distinct large downloads started in one app session; bounded by user actions but unreleased. Smallest safe fix: drop entries when a download is canceled/removed (alongside `downloadParams` removal) or cap the dictionary.

## Coverage
ios/LocalPackages/LocalSwarm/Package.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/AuthToken.swift — findings: 1
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Benchmark.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/ChunkStore.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Chunker.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Diag.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Discovery.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/EmbeddedIdentity.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Hashing.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/InterfaceTracker.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Manifest.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/ManifestCache.swift — findings: 1
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/PeerConnection.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/PeerSecurity.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmManager.swift — findings: 2
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmModels.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmSession.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Telemetry.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/TransferLogger.swift — findings: 1
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/WireProtocol.swift — clean
# Batch 2 — performance review

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfAttention.swift:76-85 — hand-composed multi-head attention chain (separate QK matmul, scaling, optional additive mask, softmax, PV matmul, plus the per-layer reshape→transpose(0,2,1,3)→reshape round trip in transposeForScores) instead of the fused `MLXFast.scaledDotProductAttention` — on every encoder forward pass each of the numHiddenLayers layers allocates extra lazy-graph nodes and executes unfused QK-product, softmax, and PV-product kernels (and the transpose/residual-reshape layout churn), adding constant-factor latency and a bit of extra precision loss to the text-encoding step of each TTS utterance; impact is bounded because Kokoro's PLBERT encoder is a single synchronous bidirectional pass per utterance (not the autoregressive decode loop), hence low — smallest safe fix: replace the matmul/softmax/matmul and the reshape-transpose-reshape round trip with `MLXFast.scaledDotProductAttention(queries:keys:values:..., scale: 1/sqrt(headSize))`, which fuses the kernels and is higher-precision (MLXFast is already imported elsewhere in the package, e.g. LayerNormInference.swift:20).

## Coverage
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/AuthTokenTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/BenchmarkMetricsTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/ChunkerMemoryTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/ConformanceTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/DiscoveryTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/EngineCoreTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/FolderTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/HostingTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/ManifestCacheTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/QUICLoopbackTests.swift — clean
ios/LocalPackages/LocalSwarm/Tests/LocalSwarmEngineTests/SecurityTests.swift — clean
ios/LocalPackages/kokoro-ios/Package.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertEmbeddings.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertEncoder.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertIntermediate.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertLayer.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertLayerGroup.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertModelArgs.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertOutput.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfAttention.swift — findings: 1
# Pi sweep — performance review — batch-3

## Findings

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ConvWeighted.swift:130,158 — the faster-oriented `weight.transposed()` is recomputed on every forward call whenever `x.shape.last != weight.shape.last` (for time-axis conv1d the spatial frame length almost always differs from the fixed kernel width, so the else branch is the common path). — a full kernel-weight transpose + allocation is re-executed per forward pass across every conv in the decoder/generator hot path during each synthesis, even though the transposed orientation is a loop-invariant fixed at init. — precompute and store both the forward and transposed orientations once at `init` and select with the same shape test, removing the per-call transpose.
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/DurationEncoder.swift:92,109 — `MLXArray.zeros(like: x)` (a full `[batch, features, seq]` tensor) is allocated for every masking application: once up front plus once after every AdaLayerNorm layer, and only ever used as the constant fill value in `MLX.where(mask, zeros, x)`. — with `nLayer` alternating AdaLN blocks this is nLayer+1 redundant full-tensor allocations per synthesis; since the fill value is a constant zero it can be a single shared tensor. — hoist one `zeros(like: x)` (or reuse a single zero scalar/array broadcast) allocated once before the layer loop and reference it at every masking site.

## Coverage
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfOutput.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/CustomAlbert.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaIN1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaINResBlock1.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaLayerNorm.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdainResBlk1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/Conv1dInference.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ConvWeighted.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/InstanceNorm1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/Interpolate.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LSTM.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LayerNormInference.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ReflectionPad1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/UpSample1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/Decoder.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/Generator.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/MLXSTFT.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/SineGen.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/SourceModuleHnNSF.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/DurationEncoder.swift — findings: 1
# Batch 4 — Performance review (deepseek-v4-flash-0731)

## Findings

- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroConfig.swift:155 — `loadConfig()` re-reads the JSON file from disk and re-runs `JSONDecoder().decode(...)` on every invocation even though it owns a static `config` cache and claims to be "cached after first load"; the cache is only written, never consulted as a short-circuit. Every `KokoroTTS.init` (KokoroTTS.swift:75) re-parses the config from scratch. With a single long-lived engine the cost is a one-time ~disk read + JSON decode, but any app that constructs/reconstructs engines (per-session or per-voice) redoes the full file read + decode each time for identical data. — Add an early `if let cached = KokoroConfig.config { return cached }` (or use a `loadOnce`/static-let reader) so the file is read and decoded at most once per process.
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Tokenizer.swift:22 — per-character `String(ch)` allocation inside the loop builds a fresh Swift `String` for every Character just to look it up in the `vocab` dictionary. On long utterances (up to `Constants.maxTokenCount` = 510 chars) this allocates ~one small String per token on the synthesis hot path. — Constant-factor allocation churn on an already short hot loop; acceptable at current max length. If this loop is ever widened to longer inputs, index `vocab` via `Character`/substring keys or build a `[UnicodeScalar: Int]`/Character-keyed lookup to avoid the per-char String boxes.

## Coverage
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroConfig.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroTTS.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/ProsodyPredictor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TextEncoder.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TimestampPredictor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/WeightLoader.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/G2PFactory.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/G2PProcessor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Language.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/MisakiG2PProcessor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Tokenizer.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/eSpeakNGG2PProcessor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Utils/AudioUtils.swift — clean
ios/LocalPackages/kokoro-ios/Tests/KokoroSwiftTests/KokoroSwiftTests.swift — clean
ios/LocalPackages/llama.cpp-swift/Package.swift — clean
ios/LocalPackages/llama.cpp-swift/Sources/LlamaCppSwift/LlamaCppSwift.swift — clean
ios/LocalPackages/mlx-swift-lm/.github/workflows/pull_request.yml — clean
ios/LocalPackages/mlx-swift-lm/.pre-commit-config.yaml — clean
ios/LocalPackages/mlx-swift-lm/.spi.yml — clean
ios/LocalPackages/mlx-swift-lm/IntegrationTesting/IntegrationTesting/IntegrationTesting.swift — clean
# Pi sweep — batch-5 — performance-only review

Reviewed each file below against the `performance-review` checklist plus the
`mlx-performance-review` and `ios-performance-review` specializations (all files
are Swift, in or around `mlx-swift-lm`).

Key context established during review: none of the MLXEmbedders models are
autoregressive — they are single-forward-pass batch encoders (token IDs ->
embeddings), so the dominant MLX decode-loop defect classes (per-step
`.item()`/`asArray` sync stalls, per-token KV-cache `concatenate` growth, missing
KV cache, per-step host branches) do not apply. The only `eval()` on a hot path
is `LLMModel.prepare`'s `eval(cache)` per prefill chunk, which is the canonical
chunked-prefill pattern that bounds lazy-graph growth (the fix the MLX skill
prescribes, not a bug). Encoder layer loops (Bert/Nomic/Gemma3/Qwen3) iterate
over `config.numLayers` (bounded 12-26). `sanitize` renames weight keys via a
bounded number of `replacingOccurrences` at load time only (once per model load,
~hundreds of keys) — not a hot path. No DB queries, no per-item network/file I/O,
no unbounded caches, no per-request client construction exist in this batch.

No findings.

## Coverage
ios/LocalPackages/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/ToolCallIntegrationTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/BenchmarkHelpers/BenchmarkHelpers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/EmbedderModelContainer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/EmbeddingModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Extensions/MLXArray+Helper.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/ModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Bert.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Gemma3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Qwen3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Pooling.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXHuggingFace/Macros.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift — clean
# Pi sweep — batch-6 (performance only) — MLX LLM models

Model files are MLX forward-pass graph construction (lazy/async); the standard
decode-path checklist (per-token forced sync, KV-cache concatenation, missing
cache, per-layer eval) is clean across all files: every KV cache goes through
`KVCache`/`KVCacheSimple`/`RotatingKVCache.update`, there are no `.item()`/`asArray`
calls inside per-token loops, and attention uses `scaledDotProductAttention`.
The one real defect class is fp32 silent promotion during the MoE expert-combine
broadcast-multiply, present in two files and already fixed in the sibling
`GLM4MOELite.swift` (whose inline comment documents the exact fix), confirming it
is a genuine, recognized hot-path issue.

## Findings
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOE.swift:363 — `y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)` — `scores` is fp32 (gate does `sigmoid(hiddenStates.asType(.float32))`); multiplying bf16 `y` (`switchMLP` output, shape [B,L,topK,hidden]) by fp32 scores silently promotes the whole expert-combined tensor to fp32, then casts back to bf16 only after the multiply — 2x memory/bandwidth on the combine reduction for every token through every MoE layer — cast scores to `y.dtype` before the broadcast multiply: `y = (y * scores.asType(y.dtype)[.ellipsis, .newAxis]).sum(axis: -2)` (mirrors the already-applied fix in GLM4MOELite.swift).
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/AfMoE.swift:472 — `y = (y * selectedScores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)` — `selectedScores` is fp32 (built from `scores = sigmoid(gates.asType(.float32))`), so the bf16 expert output tensor is silently promoted to fp32 for the combine multiply and only cast back after the sum — doubles bandwidth on the expert-combine reduction per token per MoE layer — cast `selectedScores` to `y.dtype` before the broadcast multiply: `y = (y * selectedScores.asType(y.dtype)[.ellipsis, .newAxis]).sum(axis: -2)`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Lora+Data.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LoraTrain.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/AfMoE.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Apertus.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/BaichuanM1.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/BailingMoe.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Bitnet.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Cohere.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Ernie4_5.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Exaone4.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOE.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma2.swift — clean
# Pi sweep — performance review — batch-7 (MLXLLM Models)

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift:281 — per-timestep Swift loop in `ssmStep` over the sequence (`for t in 0 ..< T`) that slices `newState[0..., t]` / `dtA[0..., t]` and performs a scatter assignment each iteration — builds an MLX lazy op graph with O(T) nodes and per-iteration array slice/allocation during prompt processing, and materializes each state subscript; at T = a long prompt (hundreds–thousands of tokens) the prefill pays a Python-style sequential recurrence instead of a single vectorized scan — the loop is the SSM recurrence, but the per-t scatter/subscript form is materially slower than a fused/`scan`-based formulation; consider expressing the recurrent update with a vectorized cumulative/slice-based pass or an MLX scan so the prefill cost does not scale as O(T) Swift-level MLX calls.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift:595 — in `Gemma3nDecoderLayer.callAsFunction`, every sliding layer builds a full boolean attention-shape tensor each forward: `tril(MLXArray.ones(maskArray.shape, dtype: .bool), k: -slidingWindow)` then `MLX.where(slidingWindowMask, minDtype, maskArray)` allocates O(L²) bytes and runs a full elementwise pass on the already-created attention mask — this repeats per sliding layer (N_sliding layers) on every prompt/decode step and its content depends only on sequence length, so it is re-allocated and recomputed N_sliding × per step with no hoisting — during long-context prefill with L on the order of thousands and many sliding layers this adds a full extra O(L²) mask pass per layer; hoist the tril/where slice (it depends only on `maskArray.shape`, `slidingWindow`, and `cachePosition`) above the layer loop and slice per layer.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3Text.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Granite.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Internlm2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/LFM2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/LFM2MoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Lille130m.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Llama.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiMo.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiMoV2Flash.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiniCPM.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/MiniMax.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Mistral3Text.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NanoChat.swift — clean
# Performance review — batch 8 (MLX LLM model ports + LoRA/DoRA adapters)

## Findings
- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/DoRA+Layers.swift:22-24 — `forward(...)` reconstructs the full adapted weight `adapted = weight + matmul(scale * loraB.T, loraA.T)` and its row norm `denom = norm(adapted, axis: 1)` inside `callAsFunction`, i.e. once **per generated token per DoRA layer** — but `weight`, `loraA`, `loraB` are all frozen at inference, so `adapted`/`denom` are loop-invariant across the whole decode loop and are needlessly recomputed on every token. Consequence: decode throughput is cut roughly in half for DoRA-adapter inference — each forward adds O(outDim·inDim) work equal to the base matmul itself (x is length-1 in decode), multiplied by the ~numLayers adapted layers and the full sequence length; and for `QDoRALinear` the `weight` parameter is `dequantizedWeight`, so each token also forces a full 4-bit dequantization of the base weights just to recompute a constant scaling vector. Smallest safe fix: compute the constant scaling term once (e.g. precompute `magnitude / norm(weight + matmul(scale*loraB.T, loraA.T), axis:1)` at load/freeze time, as `fused()` already does) and reuse it in `callAsFunction`, leaving only the low-rank `x·A·B` path per token.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Olmo2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Olmo3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OlmoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OpenELM.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Phi.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Phi3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/PhiMoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3MoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3Next.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/SSM.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/SmolLM3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Starcoder2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/DoRA+Layers.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRA+Layers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift — clean
# Pi sweep — batch-9 (MLXLMCommon adapters/inference/chat)

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:443 — FrequencyPenaltyContext.process allocates a fresh vocab-sized `MLXArray.zeros([vocabSize])` histogram plus a ones tensor and does a scatter_add + full-vocab subtract on every call — Vocab sized 32k–150k, this function runs once per generated token in the decode loop when `frequencyPenalty` is enabled, so a 1k-token generation rebuilds the full-vocabulary histogram and pays O(vocab) GPU allocation/ops 1000× instead of reusing one buffer — preallocate a persistent histogram of `vocabSize` once and clear/reuse it each step (or keep the ring as a pre-sorted index) instead of `zeros`+scatter fresh every token.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRAModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapter.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapterFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapterTypeRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/BaseConfiguration.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Chat.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Downloader.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Extensions/Encodable+toolResult.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Extensions/JSONDecoder+JSON5.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/GenerationConfigFile.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/InterpolationUtils.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/JSONDecodingTypes.swift — clean
# Pi perf review — batch-10 (mlx-swift-lm MLXLMCommon)

Perf-only review of vendored mlx-swift-lm sources per performance-review + mlx-performance-review + simd-accelerate-review.

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tokenizer.swift:97 — `NaiveStreamingDetokenizer.next()` re-decodes the entire accumulated `segmentTokens` array on every generated token, and `startNewSegment()` (line 90) decodes the full segment again. This fires on the per-token generation hot path (Evaluate.swift:1969). Within one paragraph `segmentTokens` grows 1..n, so total decode work is O(n²) token-decode operations for an n-token paragraph (e.g. a 500-token answer with no newline = ~125k token decodes instead of 500). — Smallest safe fix: decode only the newly appended token and diff/splice incrementally (the old `newSegment.suffix(...)` trick already assumes prefix stability), or cap/refresh `segment` via a sliding window; the O(n²) growth comes from decode scanning all prior tokens every call.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift:34 — `Gemma.clipResidual` upcasts both full fp16 hidden states `x` and `y` to `.float32`, adds, clips, and casts back to fp16 on every residual connection. It is called twice per transformer layer per generated token (Gemma3Text.swift:275/279, Gemma3 embedders) for fp16 Gemma/Gemma3 models, so it allocates and moves 2x the hidden state in fp32 for every layer-token — a constant-factor bandwidth/allocation churn on the autoregressive loop that scales with hidden_size × layers × tokens. — Smallest safe fix: add and clip in an accumulator using one fused fp32 pass (allocate a single fp32 workspace reused across layers) instead of per-call upcast of both operands, or gate the upcast so non-fp16 models (the common path) skip it entirely.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift:21 — `Gemma.RMSNorm.callAsFunction` recomputes `1.0 + self.weight` (a full `dimensions`-length tensor add + allocation) on every forward call, i.e. per layer per token. — Smallest safe fix: precompute `1 + weight` once in `init` and store it, so the add and its temporary are hoisted out of the hot loop.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/LanguageModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelConfiguration.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelContainer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Module+Extensions.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/AbstractModelRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/ProcessorTypeRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEApplication.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEUtils.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SuScaledRoPE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tokenizer.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/TokenizerLoader.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GLM4ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/JSONToolCallParser.swift — clean
# Pi review — batch-11 (performance deep review)

## Summary
Reviewed 20 Swift files in the mlx-swift-lm Tool parsers, Tool support types,
Utilities, MLXVLM media/manifest handling, and FastVLM/Gemma3 VLM model code.
One medium finding: the streaming ToolCallProcessor re-scans and re-parses the
entire tool-call buffer on every generated chunk, giving O(n²) work on the
streaming generation hot path for long tool-call arguments. The parsers and ML
model prepare() paths are bounded (tools count, parameter count, images per
request are all small/fixed) and were dismissed.

## Findings
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift:141 — ToolCallProcessor accumulates the partial tool-call buffer with `toolCallBuffer += chunk` and then runs a full `parser.parse(content: toolCallBuffer)` (complete JSONSerialization of the whole buffer, e.g. `.json` format) plus a full `jsonBracesBalanced(toolCallBuffer)` linear scan on every incoming streamed chunk, and in `processTaggedChunk` does `toolCallBuffer.contains(endTag)` (full scan) per chunk while collecting. This makes the token-streaming hot path quadratic in the tool-call length: for a tool call with large arguments streamed across n chunks, each chunk re-copies the whole growing string (String += ) and re-serializes/re-scans it, so total work is ~O(n²) regardless of which format is used — generation latency and decode throughput degrade as models emit longer tool arguments (e.g. code blocks written to file tools). — Only re-check the delta: keep an incremental brace/end-tag balance count and only attempt `parser.parse` when the buffer is complete (braces balanced / end tag present), and append chunks to the buffer instead of re-copying (or window the completion checks to the newly appended tail). This turns per-chunk work from O(buffer) to O(chunk).

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/KimiK2ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/Llama3ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM2ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/MistralToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/ParserUtilities.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/PythonicToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/XMLFunctionParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Tool.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCall.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallFormat.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolParameter.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Value.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/UserInput.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryPolicies.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryUtils.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/MediaProcessing.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma3.swift — clean
# Batch 12 — Performance review (MLXVLM models)

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:56 — `gemma4MaskedScatter` (called from `getInputEmbeddings` at line 1715) flattens the entire `inputsEmbeds` tensor AND its broadcast image mask and forces the mask to host memory with `mask.flattened().asArray(Bool.self)`, then builds a Swift `targetIndices` array via `compactMap` and scatters back on GPU — a full GPU→host→GPU round trip of a (batch × seq × hiddenSize) boolean mask plus the full embedding tensor on every image prefill. Consequence: prefill latency stalls on a host sync that scales with seq×hidden (e.g. 2048×1536 ≈ 3M elements copied and indexed per request). — Smallest safe fix: perform the scatter entirely on GPU (e.g. derive target indices with an on-GPU argwhere/nonzero and use `MLX` masked/gather-put, or assign via a boolean-mask `where`), avoiding `.asArray(Bool.self)` and the Swift index array.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:1489 — `Gemma4VisionPooler.callAsFunction` forces a GPU→host sync with `actualPositions.max().item(Int32.self)` to derive the kernel/divisor; similarly line 1706 (`imageMask.sum().item(Int.self)`) syncs to count image tokens. Each is a blocking device sync inside the prefill compute path; with multiple images per request they serialize. — Smallest safe fix: compute the count and max with on-device reductions/`argmax` and pass scalar results via the existing `validCount`/loop variables rather than calling `.item()` in the tensor path.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift:509 — `Vision.Attention` runs one `scaledDotProductAttention` kernel launch per image/frame in a `for i in 0 ..< (cuSeqlens.count - 1)` loop, then concatenates; with many OCR crops or a video (t frames) this launches O(frames) small SDPA calls per block × depth on every prefill, multiplying kernel-launch overhead. — Smallest safe fix: fuse into a single batched SDPA using a per-image block-diagonal causal mask (or pad to a max frame and mask), keeping GPU-attention as one kernel.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift:914 — `getRopeIndex` copies the whole token row to host (`inputIds[batchIdx].asArray(Int32.self)`) and then builds the three M-RoPE position arrays with per-token Swift `append` loops (nested t/h/w loops for image tokens) on every prefill, an unnecessary host round trip and per-element allocation for already-vectorizable positions. — Smallest safe fix: compute the 3×batch×seq position IDs with vectorized MLX ops/`arange`+`repeat` instead of a host element loop.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift:691 — `prepareInputsForMultimodal` forces the full token sequence to host (`inputIds.asArray(Int.self)`) just to locate image-token indices, then builds one small slice MLXArray per segment and finally `concatenate`s them all, on every image prefill. — Smallest safe fix: locate image positions with an on-GPU comparison/argwhere and use a vectorized (single gather + scatter) construction instead of per-segment host slices and appends.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift — findings: 1
# Batch 13 — VLM model performance review (MLX/Swift)

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift:218 — `generateBlockAttentionMask` materializes a dense N×N `[Float]` mask in Swift then reshapes/broadcasts it — a `.mask` array needs only block-diagonal `0`s, but the code allocates `seqLen * seqLen` Floats filled with `-1e9` for every vision forward. At the default longestEdge 1540 with patchSize 14 a near-square image yields ~110×110≈12k patches, so seqLen² ≈ 1.5e8 Floats ≈ 585 MB allocated and walked by the nested row/col loops on every `prepare`/`getInputEmbeddings` call — a per-image GP of seconds of allocation + a memory spike proportional to patch count². — Build the block mask as a sparse/region-based structure (construct only the diagonal blocks, e.g. via block-diagonal concat of per-image all-zero squares, or a `masked_fill` from an index matrix) instead of a full dense N×N Swift array.
- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift:150 — `unfold` (im2col) is a 4-level nested loop that, for every output position, appends `kernelSize²` per-element slice ops (`x[0..., 0..., hIdx, wIdx]`) into a `block` array, stacks+transposes them, then does a final `MLX.stacked(blocks, axis: -1)` over all positions. Each subscript is a separate lazy graph op, so the cost scales ~O(H_out·W_out·k²) in Python-loop slice ops — for a ~110×110-patch image with k=2 that is ~4900 positions × 4 slices + 4900 stack/transpose ops, all rebuilt on every patch-merger call during the vision forward. — Implement `unfold` with whole-array gather (`take_along_axis`) / stride-slicing or a single `padded`+reshape trick instead of per-position Swift slices and stacking.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift:727 — `splitIntoPatchesAndPreprocess` extracts one slice per patch in a double nested loop, calls `.flattened()` per patch, and appends each to an array, then does a single `stacked(patches, axis: 0)`. With maxTiles 10 and tileSize 512/patchSize 16 this is up to 100 tiles × 1024 patches ≈ 102,400 separate slice+flatten+append graph ops per image during `prepare` — heavy allocation churn on the preprocessing hot path. — Slice once into a 4-D [H/n, W/n, n, n, C] view and reshape/flatten in a few whole-array ops, or build rows with `split`/strided slicing instead of per-patch subscripts.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift:854 — `mergeInputIdsWithImageFeatures` iterates every image patch (`for i in 0..<numImagePatches`) and takes a single-patch slice `imageFeatures[0..., i..<(i+1), 0...]` appended to list, then one final `concatenated`. For a large image (~12k patches) that is ~12k separate graph-op slices per forward; the sibling Mistral3 path uses `MLX.split(imageFeatures, indices:, axis:1)` once instead. — Use `MLX.split` (or a single gathered assignment like LFM2's index-assign) rather than per-patch subscripting.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift:216 — `resizePositionalEmbeddings` pads each image's positional embeddings with a Swift loop of per-position indexed assignments `resultedPositionalEmbeddings[i, j] = resizedEmbeddings[0]`; every iteration is a separate gather/scatter graph op (up to `maxLength − numPositions` ops when the target grid is much smaller than max patches). — Replace the loop with a single whole-row slice assignment (`resultedPositionalEmbeddings[i, numPositions...] = MLX.repeated(resizedEmbeddings[0], ...)`) or a masked `where`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift — findings: 2
# Batch 14 — MLX VLM model performance review

Reviewed the 5 listed Qwen VLM model files under the general performance-review and
mlx-performance-review checklists (GPU→CPU sync stalls, per-token/per-loop eval &
graph construction, mask re-allocation, host loops over sequences).

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift:513,525 — Vision `getWindowIndex` forces a GPU→CPU sync per video/image frame — per frame it calls `indexFlattened.asArray(Int.self)` (full per-frame index array copied to host) at 513 and `cuSeqlensTmp.asArray(Int.self)` at 525, plus a Swift `enumerated().filter{...}` host scan over the whole frame. The loop iterates once per THW frame, so multi-frame video prefill (2 FPS extraction → ~120 frames/min of footage) pays 2 full pipeline stalls + a host scan per frame. — Smallest safe fix: compute the `!= -100` filtering and the cumsum on-device (mask + `nonZero`/`cumsum`), or hoist a single `asArray` out of the frame loop, doing all frame processing on host before the loop.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:100-121 — `gatedDeltaOps` runs a per-token eager Swift loop over the whole sequence length T (the recurrent linear-attention/SSM scan), slicing q/k/v/g/beta one token at a time, executing `gatedDeltaStepOps` (several MLX ops incl. `exp`, `softplus`, broadcasted multiply, and a `where` mask) per token, collecting T tensors into `ys`, then `MLX.stacked(ys, axis: 1)`. Growth axis = token sequence length; every token layer forward dispatches ~6-8 async GPU ops plus a host loop iteration and T accumulated tensors, so long-context prompts on this hybrid model pay O(T) host dispatches and allocations per layer. — Smallest safe fix: replace the naive step loop with a chunked / fused associative scan over the recurrence (run on-device), or at minimum drop the per-token `where` mask construction when `mask == nil` and avoid building the intermediate tensor list.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:366 (and Qwen3VL.swift:924) — `RotaryEmbedding.applyInterleavedMRope` rebuilds the mRoPE interleave by a host loop over `dims` (`for idx in 0 ..< dims`) that emits one per-index MLX subscript/slice op per element and then `stacked(slices, axis: -1)`. This runs inside `rotaryEmbedding(...)` which is called on the hot path every attention layer on every decode/prefill step (rotaryDim ≈ headDim/2 ≈ tens of elements), so each forward constructs O(dims) tiny graph nodes that queue behind the big matmuls and adds per-layer host overhead scaled by token count × layer count. The interleave pattern is static for a fixed `mrope_section`, so it needn't be recomputed per call. — Smallest safe fix: precompute the interleave index permutation once (per model config) and apply it as a single whole-array gather (`take_along_axis` / one reindex) instead of per-element slices + stack.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:542-550 — vision `Attention` re-allocates and rebuilds the full block-diagonal attention mask `ones([1, sequenceLength, sequenceLength]) * -1e9` (an O(L²) tensor) and then does a per-segment `mask[..., start..<end, start..<end] = 0` scatter for every vision block, even though `cuSeqlens` is identical across blocks. Vision sequence length L is thousands of patches (image × frames), so with `depth` vision blocks this is O(depth × L²) allocation + scatter per image encode. (Contrast Qwen25VL, which builds full/window masks once and reuses them across blocks.) — Smallest safe fix: build the mask once in `VisionModel.callAsFunction` and pass `cuSeqlens`/mask into each block, exactly as Qwen25VL does.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:1430,1470 (shared `getRopeIndex`, also used by Qwen35.swift:959) — `getRopeIndex` runs one GPU→CPU sync — `lastArray.max().item(Int.self)` (an O(seq) reduction + `.item()` stall) — per image/video token inside the `for _ in 0 ..< (imageNums + videoNums)` loop, plus re-scans `inputTokens[st...].firstIndex(of:)` (O(seq) host scan) at 1393/1400 for each visual token. Growth axis = number of image/video tokens (scales with video length / image count). Prefill of a video with many frames therefore pays multiple `.item()` pipeline stalls and O(visualTokens × seqLen) host scanning to derive position IDs. — Smallest safe fix: track the running max/st cursor incrementally in host Int state instead of recomputing `lastArray.max().item()` per token, and advance `st` so the `firstIndex(of:)` scan only covers the inter-token gap rather than restarting from `st` each time.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35MoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift — findings: 3
# Pi sweep — perf — batch-15

## Summary
Reviewed 19 files in the mlx-swift-lm package: VLM model utilities (QwenVL, SmolVLM2), a protocol, the VLM model factory, Package.swift, a CI doc script, and 10 test files. The only application code in this batch is image/video preprocessing for VLM models, which operates on bounded inputs (a handful of images, video frames capped at 20, prefill token sequences). The autoregressive decode loop lives in MLXLLM/MLXLMCommon, not in this batch. No defensible performance findings: all loops in scope iterate over small, bounded collections or run at one-time model-load/prefill boundaries.

## Findings
(no findings)

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/QwenVL.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/SmolVLM2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/VLMModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Package.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/BaseConfigurationTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ChatSessionTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/EmbeddingPoolingTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/EvalTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/MediaProcessingTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/NemotronHTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ResolveTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/SampleTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/SpeculativeDecodingTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/TestTokenizer.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ToolTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/UserInputTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/WiredMemoryPolicyTests.swift — clean
ios/LocalPackages/mlx-swift-lm/scripts/verify-docs.sh — clean
# Batch 16 — performance review

- [medium] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:124-139 — Redundant recomputation of the driving route on the AppIntent hot path: `service.planDrivingRoute(...)` is called, its result stored in `route`, then immediately discarded (`_ = route` at line 139) while `adapter.dispatchLocal_plan(...)` (line 131) recomputes the same route through the tool adapter and that result is what is returned. Every StartRouteIntent pays the street-graph routing cost twice. — With routing being the most expensive step (pathfinding over street ZIM data, likely hundreds of ms), each Siri "get directions" invocation does the work twice, roughly doubling intent latency. — Drop the `service.planDrivingRoute` call (and `_ = route`) and build the body purely from `dispatchLocal_plan`, or format the body from the already-returned `route` and delete the redundant dispatch; do not run both.

## Coverage
ios/MCPZimChat/App/MCPZimChatApp.swift — clean
ios/MCPZimChat/AppIntents/LocationFetcher.swift — clean
ios/MCPZimChat/AppIntents/ZimfoContext.swift — clean
ios/MCPZimChat/AppIntents/ZimfoIntents.swift — clean
ios/MCPZimChat/AppIntents/ZimfoRunner.swift — findings: 1
ios/MCPZimChat/Chat/AppTelemetry.swift — clean
# Batch 17 — Performance review

## Findings

- [medium] ios/MCPZimChat/Chat/ChatSession.swift:3492,3985 — `enrichSearchHits` performs up to 3 synchronous libzim `entry.reader.read(path:)` calls (each decompressing up to 64 KB of article HTML) on the @MainActor while handling a `search` tool result inside the LLM tool loop. The whole `runGenerationLoop`/`send` chain runs on the main actor, so a cold-ZIM or first-search read stalls the UI (frozen chat header/spinner) for the read+decompress duration and serializes with token streaming. Consequence: main-thread jank proportional to ZIM seek/decompress latency on every search-augmented turn. Fix: run the per-hit `reader.read` on a background executor (e.g. `Task.detached`/a serial utility queue) or batch the three reads e.g. via `func withCheckedContinuation` from a `DispatchQueue.global()`, then join the results on the main actor.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:412 — `debug()` prefixes every line with `MemoryStats.formatted()`, which calls `physFootprintBytes()` → a `task_info` Mach syscall plus a `String(format:)` allocation on every call. `debug` is invoked once per tool dispatch, per generation stage, and inside the streaming path, all on the @MainActor. Consequence: a Mach syscall + allocation tax on the main thread multiplied across a multi-iteration tool turn (several dozen calls), measurable on lower-end devices during an active generation. Fix: cache the footprint decoration or append the log line without re-reading task_info when the pane's "mem=" prefix isn't needed (e.g. only decorate for the debug pane, and let the persistent/OSLog rows be plain).

- [low] ios/MCPZimChat/Chat/ChatSession.swift:3340-3342 — once `sawMarkerChar` is set by ANY "<" or "`" byte in the streamed reply, every subsequent chunk runs `self.extractToolCall(in: buffer)` over the whole accumulated buffer. For a long prose reply that contains a stray "<" or "`" early (code, "<3", quoted text), this degenerates to an O(n²) whole-buffer regex/JSON scan per chunk on the main actor for the rest of the generation. Consequence: growing per-chunk main-actor CPU cost as the reply lengthens, delaying UI pushes. Fix: gate the scan on an actual tool-call opener (e.g. only scan once a chunk contains "<tool" / template-specific marker, or bound the scan to a trailing window of the buffer) rather than any "<"/"`" byte.

## Coverage
- ios/MCPZimChat/Chat/ChatSession.swift — findings: 3
# Batch 18 — iOS performance findings

Findings for files in this batch (performance-only review; stack skill:
ios-performance-review + swift-review language lens + performance-review).

- [medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:801 — Per-token full re-decode of the entire accumulated token list. Inside the `chunkLoop: for await event in tokenStream` loop, every generated token appends to `tokenIDsInt` and then calls `tokenizer.decode(tokenIds: tokenIDsInt, skipSpecialTokens: false)`, which re-decodes all N tokens emitted so far. Decoded-string work is O(N) per token → O(N²) over the turn. The surrounding comment claims "Incremental detokenisation… the old `tokenIDs.map { Int($0) }` allocated a fresh n-element array on every generated token", but that only removed the array-copy; the decode itself still re-runs over the whole list. A 512-token reply (DeviceProfile `maxReplyTokens` upper bound) pays ~131k token-decodes and repeatedly rebuilds a growing NSString/UTF-8 string, adding latency to the already hot per-token decode path and pushing more work into the model-turn hot loop. — Smallest safe fix: decode incrementally — decode only the newest appended token (e.g. `tokenizer.decode(tokenIds: [last], skipSpecialTokens: true)`) and concatenate, re-decoding the full list only as a rare correction path when `fullDecoded.hasPrefix(decodedSoFar)` fails (the code already keeps that fallback for BPE span rewrites).
- [low] ios/MCPZimChat/Chat/DebugReport.swift:146 — Synchronous JSON encoding of the entire conversation on the main thread inside a `@MainActor` method. `emitDebugReport()` builds `SerializedDebugReport` (messages × tool payloads, each tool carries full `rawResult`-class text and a 240-char `resultPreview`, plus all debug entries) and calls `encoder.encode(report)` inline on main before handing off to the detached task. On a long session with large tool payloads this encode is unbounded by conversation size and runs on the main thread against the 250 ms hang bar (the code comment even acknowledges this tradeoff). — Impact: the "send report" tap can freeze the UI (scroll/paint) for the duration of serializing a large transcript. — Smallest safe fix: build the report and encode it inside the existing `Task.detached(priority: .utility)` block, then compute the show-to-user hash off-main (or compute the hash from the smallest cheap prefix) so only the hash comparison stays on main.

## Coverage
ios/MCPZimChat/Chat/DebugReport.swift — findings: 1
ios/MCPZimChat/Chat/Message.swift — clean
ios/MCPZimChat/Common/DeviceProfile.swift — clean
ios/MCPZimChat/Common/DiagnosticsUploader.swift — clean
ios/MCPZimChat/Common/LogArchive.swift — clean
ios/MCPZimChat/Common/SemanticReranker.swift — clean
ios/MCPZimChat/Common/ZimfoContext+Adapter.swift — clean
ios/MCPZimChat/Libzim/LibzimBridge.h — clean
ios/MCPZimChat/Libzim/LibzimBridge.mm — clean
ios/MCPZimChat/Libzim/LibzimReader.swift — clean
ios/MCPZimChat/Providers/FoundationModelsNativeTools.swift — clean
ios/MCPZimChat/Providers/FoundationModelsProvider.swift — clean
ios/MCPZimChat/Providers/Gemma4Provider.swift — findings: 1
# Pi sweep perf — batch-19

- [low] ios/MCPZimChat/Providers/ModelProvider.swift:195-198 — `formatTranscript` default accumulates the whole transcript with `out += "…"` per turn inside `for t in turns` — each `+=` copies every previously-appended byte, so cost is O(turns² × avg byte len) in string-copy work as the conversation grows — for a long grounded/agentic session with many large turns this re-copies the growing transcript on every generation; build an array of parts and `joined()` once (or reserve capacity).

- [low] ios/MCPZimChat/Sharing/ZimDownloadManager.swift:270 — `progress()` calls `updateSleepBlocker()` on every `didWriteData` delegate callback (one MainActor hop + `UIApplication.shared.isIdleTimerDisabled` write + `Set` mutation via `SleepBlocker.apply`) for the entire duration of a multi-GB download — with tens/hundreds of thousands of byte-progress callbacks this replays an unnecessary main-thread UIApplication property write and Set rebuild per chunk; compute `hasActiveDownloads` and call `updateSleepBlocker()` only on state transitions (start/pause/resume/finish) instead of per-progress, or early-return when the blocked state is unchanged.

## Coverage
ios/MCPZimChat/Providers/LlamaCppProvider.swift — clean
ios/MCPZimChat/Providers/MockProvider.swift — clean
ios/MCPZimChat/Providers/ModelProvider.swift — findings: 1
ios/MCPZimChat/Sharing/ChatSession+ModelSharing.swift — clean
ios/MCPZimChat/Sharing/ZimCatalog.swift — clean
ios/MCPZimChat/Sharing/ZimDownloadManager.swift — findings: 1
ios/MCPZimChat/Sharing/ZimSwarmController.swift — clean
# Perf review — batch-20

- [medium] ios/MCPZimChat/Views/DownloadCatalogView.swift:197 — `status(of:)` calls `ZimDownloadManager.alreadyInLibrary(filename:)` which does a synchronous `FileManager.fileExists` stat per catalog row, and the whole List body re-evaluates it for every row on every render (each search keystroke, each selection toggle, each download-list change). StreetZIM/Wikipedia catalogs run to hundreds of rows, so each render fires hundreds of main-thread filesystem syscalls; typing in search re-issues them per keystroke. — Precompute a `Set<String>` of in-library filenames once (on catalog load / on download completion) and pass membership down, instead of a stat per row per render.
- [low] ios/MCPZimChat/Views/DownloadCatalogView.swift:119 — in `mapSections`, for each tier the entire `visible` array is re-filtered (`visible.filter { ($0.tier ?? "") == tier }`), so every body evaluation does O(Tiers × N) linear scans of the full catalog (tiers × hundreds of maps). — Group items into a `[String: [ZimCatalogItem]]` dictionary once per catalog load (or per filtered result) and iterate the groups, rather than filtering per tier.
- [medium] ios/MCPZimChat/Views/MarkdownMessageText.swift:12 — during streaming the latest assistant message's `source` changes on every ~10 Hz chunk, which re-runs `MarkdownMessageParser.parse(source)` over the *entire* accumulated message and then rebuilds an `AttributedString(markdown:)` (another full markdown parse) for every block, all on the main thread. Cost grows linearly with message length and block count each frame; long multi-KB generations (the app doubles reply-token budget) re-parse everything dozens of times a second — jank on the streaming hot path. — Memoize parse by text (hash,count) like `displayTextMemo`, or re-render only the delta appended since the last chunk; defer/offload inline attributed-string construction.
- [low] ios/MCPZimChat/Views/NearbyShareView.swift:168 — `shareableModelSizeLabel` runs a synchronous `FileManager.attributesOfItem` (and line 175's `shareableVoiceSizeLabel` a directory walk of voice-model assets, both on the main thread) on every body render of the browse screen, which re-evaluates frequently as the swarm browser updates. — Compute model/voice sizes once in `onAppear` off the main actor and cache the labels, refreshing only when the assets or selection change.

## Coverage
ios/MCPZimChat/Views/ChatView.swift — clean
ios/MCPZimChat/Views/DebugPane.swift — clean
ios/MCPZimChat/Views/DownloadCatalogView.swift — findings: 2
ios/MCPZimChat/Views/HeroMediaView.swift — clean
ios/MCPZimChat/Views/LibraryView.swift — clean
ios/MCPZimChat/Views/MarkdownMessageParser.swift — clean
ios/MCPZimChat/Views/MarkdownMessageText.swift — findings: 1
ios/MCPZimChat/Views/ModelPickerView.swift — clean
ios/MCPZimChat/Views/NearbyShareView.swift — findings: 1
ios/MCPZimChat/Views/OfflineContentSetupView.swift — clean
ios/MCPZimChat/Views/PastLogsView.swift — clean
# Pi perf review — batch-21 (iOS Views / Voice)

## Findings

- [low] ios/MCPZimChat/Views/RouteWebView.swift:712-713 — `reloadIfNeeded` evaluates a freshly-built `userDotOnlyJS` script via the WKScriptMessage bridge on *every* `updateUIView` tick whenever a user location is present, with no guard that the (lat,lon) actually changed — it re-pushes identical coordinates. The code already guards against re-running the full `frameRoute` injection precisely because `updateUIView` "fires on every GPS tick, ~2 Hz" and on each session-state tick (location updates, streaming chat), but the dot path was left ungated. Impact: redundant JS string interpolation + bridge round-trip into the WebContent process on every TickView re-render of the map bubble; during a streaming assistant turn with the map visible this is a repeated per-tick bridge cost plus the `waitForMap` polling chain (up to 120×100 ms) that each injection schedules while the map is still loading — stacking duplicate pollers under rapid ticks. Fix: remember the last-pushed (lat,lon) (e.g. in `RouteWebCoordinator`) and skip `evaluateJavaScript` when the coordinate equals the last one pushed.

- [low] ios/MCPZimChat/Views/PlacesWebView.swift:918-919 — same pattern as above: `reloadPlacesIfNeeded` runs `placesDotOnlyJS` via `evaluateJavaScript` on every `updateUIView` tick when `spec.userLocation` is present, re-setting identical me-dot coordinates with no change check, and each injection spawns its own `waitForMap` 120×100 ms polling chain if the map isn't loaded yet. Impact: per-tick bridge round-trip + JS script construction on each map-bubble re-render (GPS tick / streaming push), and stacked duplicate map-poll timers during the webview's load window. Fix: cache last-pushed coordinates (or reuse `lastFocusStamp`-style guard) in `PlacesWebCoordinator` and skip the round-trip when unchanged.

## Coverage
ios/MCPZimChat/Views/PlacesWebView.swift — findings: 1
ios/MCPZimChat/Views/RootView.swift — clean
ios/MCPZimChat/Views/RouteWebView.swift — findings: 1
ios/MCPZimChat/Views/VoiceChatView.swift — clean
ios/MCPZimChat/Views/ZimURLSchemeHandler.swift — clean
ios/MCPZimChat/Voice/KokoroAssets.swift — clean
ios/MCPZimChat/Voice/KokoroDownloader.swift — clean
ios/MCPZimChat/Voice/ObjCExceptionWrapper.h — clean
ios/MCPZimChat/Voice/ObjCExceptionWrapper.m — clean
ios/MCPZimChat/Voice/SpeechRecognizerService.swift — clean
ios/MCPZimChat/Voice/Supertonic3TTSService.swift — clean
# Pi sweep perf review — batch-22

- [medium] ios/MCPZimChat/Voice/TTSService.swift:508 (also 451) — `KokoroTTSService.speakChunk`/`speak` call `kokoro.generateAudio(...)` as a synchronous blocking call with no `await` before it, and this is invoked from the `@MainActor` `VoiceChatController.streamAssistantReply` (VoiceChatController.swift:907) which runs on the main actor. MLX synthesis for a ~400-char chunk takes seconds on-device (≈3.3× real-time on Mac, slower on iPhone), so the entire UI (orb animation, chat scroll, recognizer handling, the STT tap) freezes for the duration of every spoken chunk — that's a hard stall once per chunk per turn, and a double penalty because the per-chunk busy loop spins immediately after each synthesis. Fix: hop off the main actor for generation, e.g. wrap the per-chunk `generateAudio` in `Task.detached(priority: .utility)` (like `prepareForConversation` already does) and return to the caller only to schedule the small PCM buffer on `AVAudioPlayerNode`; keep `Memory.clearCache()` in the detached task.

## Coverage
ios/MCPZimChat/Voice/TTSService.swift — findings: 1
ios/MCPZimChat/Voice/VoiceChatController.swift — clean
ios/MCPZimChatMacTests/CatalogParsingTests.swift — clean
ios/MCPZimChatMacTests/ConversationalEvalTests.swift — clean
ios/MCPZimChatMacTests/GemmaToolEmissionTests.swift — clean
ios/MCPZimChatMacTests/ModelSharingTests.swift — clean
ios/MCPZimChatMacTests/ZimfoIntentsTests.swift — clean
ios/MCPZimEval/EvalCLI.swift — clean
# Pi sweep — performance — batch-23

## Summary
Reviewed the five listed files under `ios/` for PERFORMANCE ONLY (perf checklist from
`performance-review` + `ios-performance-review`). All are standalone developer-run macOS CLI
diagnostic/eval harnesses (`EvalHarness`, `LlamaCppProbeCLI`, `ProbeCompareCLI`, `ProbeE2ECLI`)
that load a model/ZIM, run a small fixed scenario/case set, print results, and exit; `project.yml`
is an XcodeGen build spec. Every loop iterates over fixed/bounded inputs (25 synthetic bars, static
`variants`/`scenarios`/`defaultCases` arrays, token streams capped by `maxTokens`, `suffix(40)`
debug tails) and runs a handful of times by a developer. No unbounded-growing-input hot paths exist
in these files; the only loops that poll (`while session.isGenerating … Task.sleep 100–200 ms`) are
acceptable CLI wait loops at sane cadences. Zero findings.

## Findings
(no findings)

## Coverage
ios/MCPZimEval/EvalHarness.swift — clean
ios/MCPZimEval/LlamaCppProbeCLI.swift — clean
ios/MCPZimEval/ProbeCompareCLI.swift — clean
ios/MCPZimEval/ProbeE2ECLI.swift — clean
ios/project.yml — clean
# Sweep findings — batch-24 (performance only)

- [medium] mcpzim/content.py:221 — `fetch_main_page` bypasses the LRU cache that `fetch_article` uses: every `get_main_page` call performs a fresh libzim read plus a full `html_to_text` (BeautifulSoup) parse of the main page, and `server.get_main_page` with no `zim` arg does this for *every* loaded ZIM on each call — an N-times re-parse of large HTML pages with no memoisation across requests. LLM agents that poll for a main page / overview pay disk read + full BS4 parse each time; a handful of ZIMs × repeated calls multiplies parse cost with no reuse — the article LRU already caches exactly this class of work. Smallest fix: route the main-page bytes through the same `_article_cache`-style keyed LRU (e.g. store under `("<main>", zim.path)`), or parse once and cache the `Article`.

- [low] mcpzim/routing.py:349 — the A* heuristic `h(node)` recomputes `haversine_m` on every call, and `haversine_m` re-runs `math.radians(goal_lat)` / `math.radians(goal_lon)` plus sin/cos/sqrt/asin each time even though the goal is fixed for the whole search. `h()` is invoked on every heap pop and on every neighbor relaxation in the inner loop, so for a city/region graph (thousands of expanded nodes) this is repeated trig + 4 radians conversions per expansion — measurable constant-factor CPU in the hottest routing loop. Smallest fix: precompute `goal_lat_r = math.radians(goal_lat)` / `goal_lon_r` and the `cos(goal_lat_r)` once outside `astar`, and give the heuristic a haversine variant that takes pre-radian goal coords (or drop the shared `haversine_m` overhead into a goal-relative helper).

- [low] mcpzim/geocode.py:111 — `Geocoder.search` linearly scans every record in the whole 2-char prefix chunk on every geocode call and sorts *all* substring-matching records before truncating to `limit`. Chunk size grows with the map (a dense country chunk can hold tens of thousands of address/place records), so a stream of geocode calls from an agent pays O(chunk) scan + O(m log m) sort per query with no early bound, even though only `limit` (default 10) results are returned. Smallest fix: since records are indexed by the same prefix, cap the scan (e.g. stop collecting once enough matches accumulate or the prefix no longer matches) and use `heapq.nsmallest(limit, ...)` instead of sorting the full match list, so cost stops growing with chunk size.

- [low] swift/Sources/MCPZimKit/ArticleHeuristics.swift:756 — `rankSectionsForQuestion` (and the multi-source variant ~line 983) call `embedder.embed(s.text)` and `embedder.embed(s.title)` for every section on every grounded question, and `keywordScore` inside it re-lowercases each full section body (`section.text.lowercased()`) then runs substring searches per term. On-device, every conversational follow-up re-tokenizes/hashes and re-copies the full text of all sections (an article can have 30+ sections of multi-KB prose), repeated on each warm turn with no per-article memoisation — device CPU spent re-doing the same section embeddings/score for the same article. Smallest fix: cache the per-section `embed` vector / lowercased body keyed by article+section identity (alongside the existing `ArticleCache`), or compute keyword evidence once per section per article and reuse it across questions in a turn.

## Coverage
ios/scripts/mcp-crashes.sh — clean
ios/scripts/mcp-deploy-verify.sh — clean
ios/scripts/mcp-logs.sh — clean
ios/scripts/mcp-report.sh — clean
ios/scripts/testflight-upload.sh — clean
ios/tools/eval.sh — clean
mcpzim/__init__.py — clean
mcpzim/__main__.py — clean
mcpzim/cli.py — clean
mcpzim/content.py — findings: 1
mcpzim/geocode.py — findings: 1
mcpzim/library.py — clean
mcpzim/routing.py — findings: 1
mcpzim/server.py — clean
pyproject.toml — clean
swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift — clean
swift/Package.swift — clean
swift/Sources/MCPZimKit/AnswerAttribution.swift — clean
swift/Sources/MCPZimKit/ArticleCache.swift — clean
swift/Sources/MCPZimKit/ArticleHeuristics.swift — findings: 1
# Pi sweep — performance — batch-25

Swift kit (MCPZimKit) sources. Reviewed with the performance-review checklist plus Swift-context notes
(Swift `String +=` builds strings in-place with geometric buffer growth, so repeated append inside a
transcript loop is amortized O(1) — NOT the Java/Kotlin O(n²) string-accumulation trap, and not flagged).
RegexCache is a compile-once cache keyed by static literal patterns, so it is bounded by distinct call
sites — not an unbounded leak.

## Findings
- [medium] swift/Sources/MCPZimKit/Geocoder.swift:169 — per-lookup linear scan of the entire decoded prefix chunk — every geocode query lowercases and substring-matches every record in `records` (the chunk the caller decoded; the file's own doc cites repackaged sub-buckets over chunks originally hundreds of MB / 100k+ records), so lookup latency and allocation grow linearly with chunk size on a conversational "near me" hot path — build a lowercased-name → record index once per decoded chunk (and/or a prefix index) and probe it instead of scanning, or at minimum keep a memoized lowercased name alongside each record so `lowercased()` isn't recomputed per query.
- [low] swift/Sources/MCPZimKit/Geocoder.swift:181 — `scored.sort` fully sorts every substring match before `.prefix(limit)` discards the tail — a broad substring query matching thousands of records pays O(m log m) for the whole match list when only `limit` (a small cap, e.g. 20) survive — replace the full sort with partial top-k selection (a min-heap capped at `limit`), turning it into O(m log limit).

## Coverage
swift/Sources/MCPZimKit/ArticleSections.swift — clean
swift/Sources/MCPZimKit/ChatToolCallParser.swift — clean
swift/Sources/MCPZimKit/ChatTurn.swift — clean
swift/Sources/MCPZimKit/ConversationFocus.swift — clean
swift/Sources/MCPZimKit/ConversationThreads.swift — clean
swift/Sources/MCPZimKit/Embeddings.swift — clean
swift/Sources/MCPZimKit/Gemma3Template.swift — clean
swift/Sources/MCPZimKit/Gemma4PromptTemplate.swift — clean
swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift — clean
swift/Sources/MCPZimKit/Gemma4ToolFormat.swift — clean
swift/Sources/MCPZimKit/GeoMath.swift — clean
swift/Sources/MCPZimKit/Geocoder.swift — findings: 2
# Batch 26 — performance review

- [medium] swift/Sources/MCPZimKit/IntentRouter.swift:1705 — `extractFoundationFact`'s nested `hasVerb`/`hasYear` helpers call `s.range(of: pattern, options: .regularExpression)` on every invocation, which compiles a fresh `NSRegularExpression` per sentence, and they run inside the `for i in sentences.indices` loop (1718-1724) over all sentences in the article text (`factoidSentences` over the full `rawText`, potentially hundreds of sentences). Each factoid command pays dozens-to-hundreds of regex compiles at ~tens of µs each — a few ms of pure compilation per query on device. — Hoist the two patterns to `RegexCache.shared.compiled(...)` (already used by `match`) so they compile once process-wide.

- [medium] swift/Sources/MCPZimKit/IntentRouter.swift:1763 — `extractPlaceOriginFact`'s `hasYear`/`matches` helpers also use `s.range(of: pattern, options: .regularExpression)`, recompiling each regex per call, and they are invoked in four full passes over `sentences` (1776/1785/1794/1808), each calling `hasYear` on every sentence. As the article text grows this multiplies regex compilations linearly per pass. — Compile the `hasYear` and each `matches` pattern once via `RegexCache.shared.compiled(...)` before the loops and reuse the compiled regex.

## Coverage
swift/Sources/MCPZimKit/IntentRouter.swift — findings: 2
swift/Sources/MCPZimKit/LFM25Template.swift — clean
# Pi perf review — batch 27

## Findings

- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:119 — `RouteSnapshot.remaining(at:)` does a full O(polyline) linear scan from index 0 on every `route_status` dispatch (the "how much longer?" tool, called repeatedly while driving). Near-route GPS advances monotonically along the route, so each call re-scans the entire cross-city polyline from the start instead of resuming near the previous nearest vertex. Since the prior nearest index is discarded between calls, the work is pure redundant recomputation on a repeat hot path (every GPS update / question). — Smallest safe fix: track/hold the last resolved `bestIdx` (or a window around it) on the adapter/progress state and only scan from that neighborhood; at minimum the comment's optimization is already correct but the per-call full re-scan remains. — Impact: grows with polyline size (thousands of vertices for long routes), repeated per route_status call.
- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:582 — `get_article_section` first fetches the requested section via `service.articleSection(...)` (a full ZIM article read+decompress), then calls `Self.relatedLinks(service:path:zim:)` with no `html`, which performs a SECOND whole-article `service.article(...)` read just to extract outbound wikilinks. `article_overview` (line 1076) fixed exactly this by sharing the already-fetched body via the `html:` parameter; these drill-in paths did not adopt it. — Smallest safe fix: pass the already-decoded article body into `relatedLinks(html:)` (or have `articleSection` return the body so one ZIM read serves both the section and the links). — Impact: one redundant full ZIM decompress per `get_article_section` call; ZIM article decompression is the dominant per-call cost.
- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:604 — same redundant whole-article re-read for the `related` enrichment in `get_article_by_title`: the article body is already fetched by `articleByTitle`, but `relatedLinks` (no `html`) triggers a second `service.article(...)` read. Applies to the factoid path at line 1357 as well. — Smallest safe fix: share the decoded body with `relatedLinks(html:)` as `article_overview` does. — Impact: duplicate full ZIM decompress per call.

## Coverage

swift/Sources/MCPZimKit/MCPToolAdapter.swift — findings: 3
swift/Sources/MCPZimKit/MemoryProbe.swift — clean
swift/Sources/MCPZimKit/MemoryStats.swift — clean
swift/Sources/MCPZimKit/ModelTemplate.swift — clean
swift/Sources/MCPZimKit/PlacesPayload.swift — clean
swift/Sources/MCPZimKit/QueryComplexity.swift — clean
swift/Sources/MCPZimKit/QwenChatMLTemplate.swift — clean
# Batch 28 — performance review (Swift/MCPZimKit)

## Findings

- [medium] swift/Sources/MCPZimKit/SZRGGraph.swift:400 — `nearestNode(lat:lon:)` is a full linear scan over the entire node table (`for i in 0..<numNodes`), and `ZimService.planDrivingRoute` calls it twice per route request (origin + goal, ZimService.swift:705-706). On a country-scale streetzim graph (millions of nodes) every driving route pays two O(N) sweeps over the whole node array + one full A* search, adding O(N) work proportional to the total node count on the request hot path. — As the in-code note says, a k-d tree / spatial bucketing index over node coords turns this into O(log N); smallest safe fix is to index nodes by a fixed grid and probe only the nearby cells, falling back only when sparse.
- [medium] swift/Sources/MCPZimKit/Router.swift:201 — `nearestNodeSpatial(index:lat:lon:)` does a linear scan over `index.nodesScaled` for every node in the (continent-scale) spatial index; `ZimService:696-697` calls it twice per spatial route request. Same O(N) cost over the full node table on a request path, and it defeats the purpose of spatial chunking (the point of SZCI/SZRC is to fetch only near cells). — Restrict the scan to nodes in the destination/start SZCI cells first (the index already has per-cell node ranges); a grid/nearest-cell probe bounds the scan to one cell instead of the whole table.
- [low] swift/Sources/MCPZimKit/StreamingSpeechPolicy.swift:44 — `let chars = Array(text)` copies the whole accumulated reply buffer to `[Character]` on every call, and the function re-scans from index 0 each time it is invoked. When the host calls `takeSpeakablePrefix` per streaming token over a growing reply, work is O(n) per call → O(n²) over a long reply plus repeated full-buffer allocation churn. — Keep the last-consumed offset and pass only the newly-appended tail, or scan `text`/`text.utf16` directly for punctuation without materialising a `[Character]` copy of the entire buffer each call.

## Coverage
swift/Sources/MCPZimKit/ReferenceResolver.swift — clean
swift/Sources/MCPZimKit/RegexCache.swift — clean
swift/Sources/MCPZimKit/Router.swift — findings: 1
swift/Sources/MCPZimKit/SZRGChunked.swift — clean
swift/Sources/MCPZimKit/SZRGEncoder.swift — clean
swift/Sources/MCPZimKit/SZRGGraph.swift — findings: 1
swift/Sources/MCPZimKit/SZRGSpatial.swift — clean
swift/Sources/MCPZimKit/StreamingSpeechPolicy.swift — findings: 1
swift/Sources/MCPZimKit/StubZimService.swift — clean
swift/Sources/MCPZimKit/ToolLoopGuard.swift — clean
swift/Sources/MCPZimKit/ZimReader.swift — clean
# Pi perf review — batch-29

## Findings

- [medium] swift/Sources/MCPZimKit/ZimService.swift:1561 — `summarize` sorts the ENTIRE `hits` array (`hits.sorted { $0.1 < $1.1 }`) and `scanRecords` accumulates every in-radius matching record into `hits` before any cap, so `nearPlaces` on a broad kind (`restaurant`, `cafe`, `food`) with a generous radius over a dense/country-scale ZIM retains and sorts tens of thousands of heavy `(Place, Double)` tuples to then return only `limit` results + a subtype count. — A "restaurants near me" query with a large radius on a big streetzim holds/swaps many MB of Place objects and pays O(n log n) with n = all in-radius records, adding latency and heap pressure on the hottest geospatial tool. — Cap memory up-front: keep only a top-`limit` bounded candidate set via an insertion/heap while scanning, and fold `totalInRadius`/`breakdown` counts into the same single pass instead of collecting every hit then sorting the whole array.

- [low] swift/Sources/MCPZimKit/ZimService.swift:815 — `renderLeadSnippet` calls `reader.read(path:)`, materializing the FULL article body (`entry.content`) to render a ≤220-char search snippet, then `leadPrefixHTML` only scans the first 64 KB of that already-decoded Data. Run per unique candidate hit on the search hot path (overfetch = max(limit*2,10) across title + each FTS variant, snippet cache capped at 64 entries). — Each search turn decompresses and allocates the entire body of every candidate article even though only the lead prefix is ever used, so payload size (article bytes) multiplies with candidate count per query. — Have the ZIM read expose a bounded/lead-only read (or raise/keep the snippet LRU large enough to avoid re-reads within a turn) so snippet generation never materializes a full multi-hundred-KB body for a 220-char snippet.

## Coverage
swift/Sources/MCPZimKit/ZimService.swift — findings: 2
swift/Tests/MCPZimKitTests/AnswerAttributionTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleFactoidTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleHeuristicsCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleSpeechCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/BundledArticleTests.swift — clean
swift/Tests/MCPZimKitTests/ChatToolCallParserTests.swift — clean
swift/Tests/MCPZimKitTests/ClarificationAndTitleCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationContinuationTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationFocusTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationThreadsTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussArticleLinkTests.swift — clean
# Pi sweep perf — batch-30

## Summary
Reviewed 16 Swift unit-test files under `swift/Tests/MCPZimKitTests/` for
performance only. All are XCTest suites exercising the production kit
(`ArticleHeuristics`, `IntentRouter`, `DefaultZimService.nearPlaces`,
`MCPToolAdapter`, `Gemma4ToolFormat`, `parsePlacesJSON`, `QueryComplexity`,
etc.) against small, bounded in-memory fixtures (a few `ArticleSection`s per
case, a handful of Place records per ZIM fixture, short prompt strings, and a
bounded 256-leaf hot-split fixture). Test code is not a runtime hot path and
none of these files allocate in unbounded loops, issue per-row queries,
construct per-request clients, or grow caches on a request path, so no real
production-scale performance defect is present. Zero findings.

## Findings

## Coverage
swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussionRetrievalTests.swift — clean
swift/Tests/MCPZimKitTests/EmbeddingsTests.swift — clean
swift/Tests/MCPZimKitTests/Gemma4PromptTemplateTests.swift — clean
swift/Tests/MCPZimKitTests/Gemma4ToolFormatTests.swift — clean
swift/Tests/MCPZimKitTests/GeocodeVariantsTests.swift — clean
swift/Tests/MCPZimKitTests/HotSplitGeocoderTests.swift — clean
swift/Tests/MCPZimKitTests/IntentRouterTests.swift — clean
swift/Tests/MCPZimKitTests/LFM25TemplateTests.swift — clean
swift/Tests/MCPZimKitTests/LocateToolTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesCenterResolutionTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesChipIndexTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesKindFallbackTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesWikiEnrichmentTests.swift — clean
swift/Tests/MCPZimKitTests/PlacesPayloadTests.swift — clean
swift/Tests/MCPZimKitTests/QueryComplexityTests.swift — clean

## Branch
(no branch created; read-only review)
# Pi sweep — performance — batch 31

## Summary
Reviewed 20 files: 10 Swift unit-test files (MCPZimKitTests) and 10 offline dev-tool files (fine-tune pipelines, A/B benchmark scripts, data-conversion script). No defensible performance findings. Every file is either a test harness operating on tiny in-memory fixtures (a 4-node graph, a 3-cell spatial grid, short chat strings) or a one-shot/offline batch job whose iteration counts are bounded by small fixed seed lists, the `--n`/`--iters` CLI argument, or the training batch size. Per the performance-review checklist's false-positive rules ("queries inside loops in ... one-off scripts, CLI admin tools, or test setup — not hot paths"; "nested loops over small, bounded collections"), none of these rise to findings. The SpatialGraph cell cache intentionally has an eviction limit (cacheLimit) and the generation script reuses a single AsyncOpenAI client with a bounded semaphore. No N+1, no unbounded caches, no per-item I/O on growing inputs, no main-thread-blocking work.

## Findings
(none)

## Coverage
swift/Tests/MCPZimKitTests/QwenClippedToolCallTests.swift — clean
swift/Tests/MCPZimKitTests/ReferenceResolverTests.swift — clean
swift/Tests/MCPZimKitTests/RouterBenchTests.swift — clean
swift/Tests/MCPZimKitTests/SZRGGraphTests.swift — clean
swift/Tests/MCPZimKitTests/SZRGSpatialTests.swift — clean
swift/Tests/MCPZimKitTests/SZRGv5AndChunkedTests.swift — clean
swift/Tests/MCPZimKitTests/SanitizeZimArgTests.swift — clean
swift/Tests/MCPZimKitTests/SanitizedWikiTagTests.swift — clean
swift/Tests/MCPZimKitTests/StreamingSpeechPolicyTests.swift — clean
swift/Tests/MCPZimKitTests/ToolLoopGuardTests.swift — clean
tools/bonsai-ab/compare.sh — clean
tools/fine-tune/convert_to_lfm2_native.py — clean
tools/fine-tune/eval_ft_pcgaming.sh — clean
tools/fine-tune/finetune.sh — clean
tools/fine-tune/finetune_cuda.py — clean
tools/fine-tune/finetune_cuda.sh — clean
tools/fine-tune/finetune_lfm2.sh — clean
tools/fine-tune/finetune_unsloth.py — clean
tools/fine-tune/finetune_unsloth.sh — clean
tools/fine-tune/generate.py — clean
# Pi perf review — batch-32

## Summary
Reviewed 13 files in tools/fine-tune, tools/gemma-smoke, and tools/llama-smoke.
Most are offline training-data generators / orchestration shell scripts that are
network- or GPU-bound with negligible CPU hot paths. Real finding: the Gemma
smoke-test harness re-decodes the entire accumulated token array on every
generated token (O(n^2)) inside the generation loops that the harness itself
wall-times, which inflates and adds variance to the very measurements the
experiments exist to report.

## Findings
- [low] tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift:533 — inside the `runOne` stream loop in `simulateRunInSinglePerform`, every generated token calls `context.tokenizer.decode(tokens: tokenIDs.map { Int($0) })` over the full accumulated buffer, and `runOne`'s wall time is returned and printed as the model's decode time. As generation grows past ~60-120 tokens this re-decode becomes O(n^2) string work per run (plus an allocation per token for the map), so the reported `iter N done in X.XXs hit=...` numbers include tokenizer overhead that grows with output length rather than the model's true throughput. — Decode incrementally: only decode the few newly appended tokens (keep a `pendingText` string and append each new token's decoded piece) and scan that tail for the stop markers, or run marker detection on raw IDs. Same pattern at line 259 in `runBehaviorTest`.

## Coverage
tools/fine-tune/generate_chains.py — clean
tools/fine-tune/generate_chains3.py — clean
tools/fine-tune/generate_places_diverse.py — clean
tools/fine-tune/retry_lfm2_train.sh — clean
tools/fine-tune/split_chain_rows.py — clean
tools/fine-tune/train_all.sh — clean
tools/fine-tune/train_all_cuda.sh — clean
tools/fine-tune/v7_eval_and_memsweep.sh — clean
tools/gemma-smoke/Package.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — findings: 1
tools/gemma-smoke/Sources/GemmaSmoke/main.swift — clean
tools/llama-smoke/bench.py — clean
# Pi performance review — batch-33

## Findings

- [low] tools/llama-smoke/eval.py:1592-1596 — `final_content.lower()` is recomputed on every keyword inside the scoring generators (`any(kw.lower() in final_content.lower() for kw in kw_any)` and again per keyword per group). For multi-group chain scenarios (e.g. putin_biography_chain: 3 groups × 3 keywords) the full assistant response (up to `max_turn_tokens` = 2048 tokens) is lower-cased once per keyword, per turn, across every grid combo. Hoist `fc = final_content.lower()` once before the two checks and test against `fc`.
- [low] tools/llama-smoke/eval.py:181-186 — inside `bars_heavy_fixture`'s `for i in range(n)` loop the `name_pool` list (`BAR_NAMES + [f"Bar {i}", ...]`) is rebuilt on every iteration even though only six elements vary with `i`; the loop only ever indexes `name_pool[i % len(name_pool)]`. This is allocation churn in a loop of size `n` (default 500, up to 1000 in the heavy path), each iteration allocating and GC-ing a ~31-element list. Split the 25 invariant `BAR_NAMES` out once and pick the per-`i` variant only when `i` indexes past the invariant prefix.

## Coverage
tools/llama-smoke/eval.py — findings: 2
tools/llama-smoke/grid.py — clean
tools/llama-smoke/sweep.sh — clean
tools/llm-smoke/bench.py — clean
tools/llm-smoke/bench_kv.py — clean
tools/llm-smoke/bench_memory.py — clean
tools/llm-smoke/bench_memory_gemma4.py — clean
tools/llm-smoke/eval.py — clean
tools/llm-smoke/eval_gemma4.py — clean
tools/llm-smoke/eval_gemma4_native.py — clean
tools/llm-smoke/gemma4_format.py — clean
tools/logpipe/ingest.sh — clean
tools/logpipe/parse_log.py — clean
tools/logpipe/prep_judge.py — clean
tools/logpipe/report.py — clean
# Pi sweep — performance review — batch-34

## Findings

No findings. All five files are test modules exercising tiny synthetic inputs; none sit on a production hot path, so no N+1, allocation-churn, unbounded-collection, or algorithmic performance defects are exhibited by the code.

## Coverage
tests/__init__.py — clean
tests/test_content.py — clean
tests/test_geocode.py — clean
tests/test_library.py — clean
tests/test_routing.py — clean
# Pi perf review — batch-35

## Summary
Reviewed the MLX/llama.cpp eval harness and the llama.cpp GGUF provider for performance. The llama.cpp provider is already heavily micro-optimized (reused detokenize scratch, bounded rolling stop-token tail, KV prefix reuse); only one small per-token allocation churn remains in its hottest decode/prefill loop. The eval harness is a headless benchmark driver with no production hot path — its loops are bounded by small scenario/turn counts, so no defensible findings.

## Findings
- [low] ios/MCPZimChat/Providers/LlamaCppProvider.swift:1223 (and 1091 prefill) — `Self.batchAdd(&batch, ..., seqIds: [0], ...)` allocates a fresh one-element `[llama_seq_id]` array literal on every token in both the prefill loop and the per-token autoregressive decode loop. Unlike the detokenize scratch buffer (which was deliberately hoisted to avoid churn), this literal is re-created per token on the hottest path, so a 32k-token prefill + long decode issues tens of thousands of small heap allocations. — Consequence: needless heap churn on the generation-critical loop; grows linearly with context/decode length and is the one remaining per-token allocation the authors already eliminated elsewhere. — Smallest safe fix: hoist a file-scope constant, e.g. `private let zeroSeq: [llama_seq_id] = [0]`, and pass `seqIds: zeroSeq` in `batchAdd` calls (the seq_ids array is only iterated for its count and reused slot, never mutated by the helper).

## Coverage
ios/MCPZimEval/EvalHarness.swift — clean
ios/MCPZimChat/Providers/LlamaCppProvider.swift — findings: 1
# Pi sweep — performance — batch-36

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:1154 — RotatingKVCache.makeMask single-token branch constructs and rolls a full boolean mask array (`MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)` then `roll(mask, shift:)`) on every single-token decode step whenever the sliding window has wrapped (`offset >= windowSize && maxCacheSize > windowSize`) — per decode token, as the session runs beyond the window, this allocates a mask array of size up to maxCacheSize (e.g. 8 k booleans) and issues an extra roll kernel launch on top of the attention, adding constant allocation/GPU churn each step in the hottest loop of sliding-window decode — winnow the allocation by representing the wrapped-window mask symbolically (partition the always-true keep region from the rolled edge) or cache/reuse the rolled mask across steps and only rebuild on window boundary crossings, instead of rebuilding the array every token.

## Coverage
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift — findings: 1
- ios/MCPZimChat/Voice/VoiceChatController.swift — clean
- swift/Tests/MCPZimKitTests/IntentRouterTests.swift — clean
# Batch 37 — Performance review

## Findings

- [low] ios/MCPZimChat/Views/ChatView.swift:738-772 — Streaming assistant text is re-stripped from scratch (full multi-regex `stringByReplacingMatches`/`firstMatch` pipeline plus several `replacingOccurrences` passes) over the ENTIRE growing buffer on every streaming push on the main thread. The `displayTextMemo` only dedupes within a single UI tick (keys on `count`, which increments every push), so each 10 Hz tick re-scans and re-allocates the whole response so far across ~8 regex/literal passes. Over an n-token answer this is O(n²) regex scanning + string copies on the main thread — grows with answer length and can add main-thread work/layout jank on long generations (e.g. running a ~10 KB answer generates ~4000 ticks × avg 5 KB × 8 passes of regex work). Smallest safe fix: strip only the newly-appended delta each tick (carry the already-stripped prefix + a stable prefix-length marker), or bound/throttle how often the strip runs during streaming.

## Coverage

- ios/MCPZimChat/Views/ChatView.swift — findings: 1
- ios/MCPZimEval/ProbeE2ECLI.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift — clean
- ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma3.swift — clean
# Pi sweep — performance — batch-38

## Summary
Reviewed four files: two fine-tune data-generation scripts (offline batch
LLM-callers, not hot paths) and two shipped runtime files (the NomicBERT MLX
embedder used at inference, and the Swift reference resolver run per user
turn). The embedder is a clean, standard MLX forward pass with no hot-path
allocation or redundant-computation defect; RoPE base frequency is computed
once at init, and the regex cache used by the resolver is bounded and
compiled-once. The only defensible findings are low-severity unbounded
async fan-out in the two batch generators, which eagerly schedule the full
workload as coroutines even though a small semaphore caps execution.

## Findings
- [low] tools/fine-tune/generate_chains3.py:517 — `tasks = [asyncio.create_task(one_with_sem(template)) for _ in range(target)]` eagerly creates one coroutine task per row of the whole per-template target (default `--n` 800, × up to 6 templates) up front, even though only `args.concurrency` (default 4) can run at once; all the rest sit parked on the semaphore. For large `--n` this is thousands of concurrently scheduled coroutines and a wasted max-concurrency schedule snapshot — at scale (e.g. --n 5000+) memory and scheduler churn grow linearly with n although the machine can only run 4 requests. Smallest safe fix: drain a bounded worker pool (max `concurrency` workers pulling row indices from an `asyncio.Queue`) instead of pre-creating one task per row.
- [low] tools/fine-tune/generate.py:641 — `asyncio.gather(*[worker(q) for q in queries])` (with `queries = [q for ...][done:]`, length up to `--n`) pre-spawns one coroutine per remaining query, all but `concurrency` parked on the semaphore; same unbounded fan-out as generate_chains3, just before a finally that closes the shared append handle. Impact is minor for the default `--n 100` but grows linearly with n on larger runs. Smallest safe fix: bounded worker pool (Queue/`asyncio.Semaphore`-guarded worker loop) sized `concurrency`.
## Branch
(no branch changes made)

## Coverage
tools/fine-tune/generate_chains3.py — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift — clean
tools/fine-tune/generate.py — findings: 1
swift/Sources/MCPZimKit/ReferenceResolver.swift — clean
# Batch 39 — Performance review (Swift / MLX model ports + test helpers)

Reviewed with the `performance-review` checklist plus the `mlx-performance-review` stack skill. The four listed files are vendored MLX model implementations (compute kernels) and test support code; almost all of the work is vectorized MLX ops, so the standard host-side perf anti-patterns (N+1, O(n^2) app loops, unbounded caches, per-item I/O) are absent. The one real finding is a hot-path constant reallocated per token in the NemotronH mamba mixer.

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift:82 — `MLXArray.ones([groupSize])` identity weight is rebuilt on every call of `NemotronHRMSNormGated.callAsFunction`, which runs once per token per mamba layer during autoregressive generation (and once per row × head-group during prompt processing). Each call allocates a fresh `[groupSize]` MLX constant buffer and dispatches an extra GPU op (`rmsNorm` against a freshly-built ones tensor + the constant multiply) purely to express "normalize without a weight". With a 30B-class hybrid model (dozens of mamba layers) over thousands of generated tokens, this adds thousands of small allocations and buffer rebuilds to the single hottest loop in the app. — Hoist the identity weight to a stored property computed once in `init` (it is a constant `ones([groupSize])`), and reuse it across calls instead of allocating per call.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift — clean
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ToolTests.swift — clean
# Batch 40 — performance findings

- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmSession.swift:553 — pump() rescans `neededHead..<neededOrder.count` per peer that has any available slot, and pump() runs on the serial `netQueue` on every chunk receive (+ handshake/bitfield/have). When a peer's `has(index)` is true only for a sparse subset of the still-needed chunks (common at transfer start, on peer loss, or when peers hold partially-different chunks), the inner loop scans the entire remaining `neededOrder` just to fill that peer's 48 slots, and a peer that holds nothing you still need scans the whole list without requesting. With a 127k-chunk file (the code's own cited scale) and several such peers, each received chunk triggers ~chunks_remaining × peers Set-lookups (`neededSet.contains` / `peer.has` / `globalInFlight.contains`) on the single netQueue, which can add real per-chunk CPU on a memory-constrained device and delay the queue's other work (message dispatch, timers). The `neededHead` cursor and compaction only bound the *dead-prefix*; they don't stop a per-peer full scan for peers lacking front-ordered chunks. — Smallest safe fix: keep a per-peer cursor (advance it monotonically past indices that peer can't use / already has in flight, reset on bitfield/have updates) so each pump continues from where the peer's last scan stopped instead of re-walking from `neededHead`, or pre-index remaining chunks by availability per peer.
- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmSession.swift:415 — `peer.serveQueue.removeFirst()` is O(n) on Swift `Array`, and `serveQueue` is appended to on every `chunkRequest` (`serveChunk`) with no upper bound — only `serveWindow` (16) in-flight serves are gated, while the FIFO queue itself grows with the incoming request rate when requests arrive faster than the ioQueue read/send completes. Each serviced chunk then pays an O(queue-size) element shift on the serial queue, and an unbounded `serveQueue` also lets one request-hungry peer accumulate arbitrary memory on the seeder. — Smallest safe fix: cap `serveQueue` length (drop/ignore requests beyond a bound, or signal the requester), and use a ring/index-cursor or `ContiguousArray` deque so serving pops O(1) instead of shifting the head on every chunk.

## Coverage
ios/MCPZimChat/Sharing/ZimCatalog.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift — clean
swift/Sources/MCPZimKit/ConversationThreads.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmSession.swift — findings: 2
# Perf review batch-41

- [low] ios/MCPZimChat/Views/LibraryView.swift:480,506 — `VoiceModelSection.backendControls` calls `formatBytes(Supertonic3Assets.currentBytesOnDisk)` and `formatBytes(KokoroAssets.currentBytesOnDisk)` directly inside the view `body`. Both accessors perform synchronous filesystem I/O on the caller (main) thread: Supertonic walks the whole `supertonic_3` asset directory via `FileManager.enumerator(at:)` statting every file, and Kokoro stats each download with `attributesOfItem(atPath:)`. Every SwiftUI re-evaluation of this section (state change, engine/voice picker change, appear) re-runs the walk, so navigating/rendering the "Voice chat" section blocks the main thread and can stutter the UI as the on-disk voice corpus grows. — Smallest safe fix: compute the byte counts once off the main thread (e.g. `Task.detached` into an `@State` value, or compute on `onAppear`/refresh) and bind the `Text` to the cached value instead of recomputing in `body`.
- [low] ios/MCPZimChat/Views/LibraryView.swift:478-506 — `formatBytes(_:)` allocates a fresh `ByteCountFormatter` (and sets its properties) on every render of the Voice section just to format one byte count. Formatter construction is non-trivial and, combined with the disk-stat above, is done repeatedly on the main thread during `body` evaluation. — Smallest safe fix: hoist the formatter to a `static let`/module-level constant and reuse it, or format lazily only when the cached byte value changes.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift — clean
ios/MCPZimChat/Views/LibraryView.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift — clean

## Run stats

input 2698516 tok (+15935744 cached), output 295077 tok, cost $0.58 — 458 files in 29m (938.4 files/h, 0.7 min/batch)
