# mcpzim — bug / correctness review (2026-08-09)

DeepSeek V4 Flash-0731 (HIGH thinking) via DeepInfra, per-file sweep (55 batches).
Recovered from the raw per-batch sweep logs — the perf pass overwrote the committed
file. Contains refeed duplicates (~228 raw finding lines, ~114 unique). NOTE: findings
in `ios/LocalPackages/mlx-swift-lm/` are a VENDORED model-port library, not first-party.

# Pi sweep review — batch-1 (LocalSwarmEngine)

## Findings

- [medium] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmManager.swift:380-395 — `receive(_:)` (the one-call "download everything" path) never supplies a PIN: it calls `fetchManifest(for: swarm)` and `startDownload(manifest:selecting:from:)` without the `pin:` argument. The engine's seeder gates a locked share on `.auth` before serving `.manifestRequest` or any `.chunkRequest` (SwarmSession.handle), so a PIN-protected (locked) share received via `receive()` is never authenticated: the control fetch is closed and the whole receive fails silently. `receive()` also has no `pin` parameter, so the UI cannot pass it. — Add an optional `pin` parameter to `receive(_:)` and thread it into both `fetchManifest(for:pin:)` and `startDownload(..., pin:)`, exactly as a direct `startDownload` already allows.
- [medium] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmManager.swift:533-545 — `resumeDownload(swarmID:)` resumes a paused transfer without a PIN, and `DownloadParams` has no field to retain it (`startDownload(manifest:selecting:from:params.swarm)` at line 543 passes none). For a locked share, the resumed bulk `SwarmSession` has `authTokenToSend == nil`, so every reconnected source is closed before it serves a chunk and the download stalls permanently with no way for the user to supply the PIN. Store `pin` in `DownloadParams` and pass it through to `startDownload` on resume.
- [low] ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/EmbeddedIdentity.swift:160-170 — a fixed, app-wide self-signed P-256 private key plus its PKCS#12 password (and the matching `PeerSecurity.sharedPasscode = "localswarm-v1"`) are compiled into every shipped binary. Because every client shares the same static key/cert/passcode, any attacker who unpacks one app binary can present the exact pinned cert + key, so the TLS PSK and the QUIC cert-pinning verify block provide no real peer authentication or secrecy against a motivated LAN attacker (only "accidental/naive-script" gating), and the key cannot be rotated short of shipping a new build to every client. — This is a documented v1 trust model, but note in code that it grants no per-device authentication and plan for an ephemeral per-pairing identity (the comment already advertises this as future work); do not present the current pinning/PSK as authentication in any security-sensitive consumer.

## Coverage
ios/LocalPackages/LocalSwarm/Package.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/AuthToken.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Benchmark.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/ChunkStore.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Chunker.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Diag.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Discovery.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/EmbeddedIdentity.swift — findings: 1
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Hashing.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/InterfaceTracker.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Manifest.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/ManifestCache.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/PeerConnection.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/PeerSecurity.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmManager.swift — findings: 2
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmModels.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmSession.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/Telemetry.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/TransferLogger.swift — clean
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/WireProtocol.swift — clean

# Pi sweep review — batch-4

## Summary
Reviewed the Kokoro iOS TTS engine (config/tokenizer/g2p/weight-loader/prosody/text-encoder/timestamp), its tests, the llama.cpp-swift wrapper package, and the mlx-swift-lm CI/pre-commit/SPI/IntegrationTesting files. The TTS engine code is largely clean (careful MLX array handling, a synthesis lock already serializing shared G2P/language state, and bulk host-syncs replacing per-token `.item()` stalls). Findings are concentrated in the GitHub Actions workflow (mutable action tags), one thread-safety gap on the shared config cache, an unguarded negative-index path on empty input, and an empty placeholder test in an otherwise testless package.

## Findings
- [high] ios/LocalPackages/mlx-swift-lm/.github/workflows/pull_request.yml:20 — third-party action `astral-sh/setup-uv@v6` pinned to a mutable major-version tag, not a full 40-char commit SHA — a retagged `@v6` changes what executes on every run without review (supply-chain / credential-scope risk) — pin to the immutable commit SHA referenced by the tag (`uses: astral-sh/setup-uv@<sha>`).
- [medium] ios/LocalPackages/mlx-swift-lm/.github/workflows/pull_request.yml:15,41,77,89,113 — first-party actions (`actions/checkout@v6`, `actions/cache@v4`, `actions/upload-artifact@v4`) pinned to mutable tags rather than SHAs — a repointed tag on a shared action runs arbitrary code in CI — pin each to its full 40-char commit SHA.
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroConfig.swift:18 — `nonisolated(unsafe) static var config` is written by `loadConfig()` (line ~149) without any synchronization while `Tokenizer.tokenize` reads `KokoroConfig.config?.vocab` from arbitrary threads; two `KokoroTTS` instances constructed concurrently (or a read racing the first write) is a data race on shared global state — wrap the cache in a `NSLock` (or use `nonisolated(unsafe) let` filled once under a lock) so a single assignment is guaranteed before any reader.
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroTTS.swift:301 — `voice[tokenCount - 1, 0 ... 1, 0...]` is indexed with `tokenCount = inputIds.count`, so empty phonemized text (empty or whitespace-only input) yields tokenCount==0 and a negative index `voice[-1]` → invalid access / crash on the voice tensor; there is no guard rejecting empty phonemes before this point — reject empty input earlier (or guard `tokenCount > 0`) and surface a `.tooManyTokens`-style error instead of indexing with -1.
- [low] ios/LocalPackages/kokoro-ios/Tests/KokoroSwiftTests/KokoroSwiftTests.swift:4-5 — the entire `@Test func exampleTest() async throws {}` is an empty body with no assertions; it can never fail, so it licenses changes to the MLX/G2P/tokenizer pipeline with zero regression signal — remove it or replace with tests that actually exercise `KokoroTTS`/`Tokenizer`/G2P and assert on real output.

## Coverage
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroConfig.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/KokoroTTS.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/ProsodyPredictor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TextEncoder.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/TimestampPredictor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/WeightLoader.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/G2PFactory.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/G2PProcessor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Language.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/MisakiG2PProcessor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/Tokenizer.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TextProcessing/eSpeakNGG2PProcessor.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Utils/AudioUtils.swift — clean
ios/LocalPackages/kokoro-ios/Tests/KokoroSwiftTests/KokoroSwiftTests.swift — findings: 1
ios/LocalPackages/llama.cpp-swift/Package.swift — clean
ios/LocalPackages/llama.cpp-swift/Sources/LlamaCppSwift/LlamaCppSwift.swift — clean
ios/LocalPackages/mlx-swift-lm/.github/workflows/pull_request.yml — findings: 2
ios/LocalPackages/mlx-swift-lm/.pre-commit-config.yaml — clean
ios/LocalPackages/mlx-swift-lm/.spi.yml — clean
ios/LocalPackages/mlx-swift-lm/IntegrationTesting/IntegrationTesting/IntegrationTesting.swift — clean

# Pi sweep review — batch-5

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Gemma3.swift:442 — mean pooling divides by `nonMasked` (sum of the attention/padding mask) without guarding the zero case; a sequence row whose tokens are all padding yields `sum / 0` = NaN/inf — silently emits a NaN embedding that corrupts cosine-similarity and poisons any downstream vector store — clamp the denominator (e.g. `MLX.maximum(nonMasked, 1)` or `MLXArray(1)`) before dividing, returning a zero vector instead of NaN.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Pooling.swift:142 — the `.mean` case also divides by `sum(_mask, axis: -1)` unguarded; a row where the mask is all zeros produces a NaN/inf pooled vector even though `.last`/`.first`/`.cls` clamp their index — same silent-NaN embedding consequence — use `MLX.maximum(sum(_mask,...), 1)` (or the mask of `_mask.sum(...) > 0`) before dividing.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Bert.swift:55 — `BertEmbedding` always allocates and applies `position_embeddings` with `embeddingCount: config.maxPositionEmbeddings`, but `BertConfiguration` decodes `max_position_embeddings` with default `0` (line 485); for a BERT-family config.json that omits `max_position_embeddings` (NomicBert guards this with `if config.maxPositionEmbeddings > 0`, Bert does not) the embedding table has 0 elements and line 76 `positionEmbeddings(posIds)` indexes out of range → model init/run crash on an otherwise valid model — guard `positionEmbeddings` creation and application behind `maxPositionEmbeddings > 0` exactly as NomicBert does.

## Coverage
ios/LocalPackages/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/ToolCallIntegrationTests.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/BenchmarkHelpers/BenchmarkHelpers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/EmbedderModelContainer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/EmbeddingModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Extensions/MLXArray+Helper.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/ModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Bert.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Gemma3.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/Qwen3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Pooling.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXHuggingFace/Macros.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift — clean

# Pi sweep — batch 7 (mlx-swift-lm Models)

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift:54 — `IntOrArray.subscript` returns `values[layerIdx]` with no bounds check when the config ships a per-layer `intermediate_size` array shorter than `num_hidden_layers`. A truncated/malformed `intermediate_size` array makes `Gemma3nMLP.init` (via `config.intermediateSize[layerIdx]`) index out of range and crash at model load — the crash is a runtime array bound violation on config input rather than a graceful decode error. — Guard the subscript to fall back to the single/scalar value (or clip to the last element) when `layerIdx >= values.count`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift:447-448 — `self.attnIdx = layersBlockType.firstIndex(of: "attention")!` and `ssmIdx = ...firstIndex(of: "mamba")!` force-unwrap on config-derived `layersBlockType`. A `layers_block_type` config containing only one of the two layer kinds (no attention layers, or no mamba layers) crashes at init; the same applies to `layersBlockType!` (line 439) and `mambaDtRank!` (line 214), which is set only in the post-init block and would be nil for a decoder path that never runs it. — Replace the force-unwraps with guarded defaults (nilable indices used with `?? 0` / `?? layers.count`) and fail gracefully with a DecodingError instead of trapping.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3Text.swift:323 — `createAttentionMask(h: h, cache: cache?[config.slidingWindowPattern - 1])` subscripts the layer cache with `slidingWindowPattern - 1` unconditionally. If a config has `hiddenLayers < slidingWindowPattern` (no global layer exists at `pattern - 1`) the subscript on the `[KVCache?]` array traps with an index-out-of-range crash at prompt time. — Clamp the index (e.g. `min(config.slidingWindowPattern - 1, (cache?.count ?? 1) - 1)`) or guard on `hiddenLayers >= slidingWindowPattern`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift:281-286 — `ssmStep` runs the SSM selective-scan recurrence as a per-timestep Swift loop over `0 ..< T` with per-step slice reads/writes (`newState[0..., t] = fma(...)`). During prompt prefill (T = prompt length, thousands of tokens) this is a serial, non-vectorized scan executed on the host across all mamba layers, a significant latency path versus a fused scan; the code mutates slice views of a tensor that is also being read, which relies on MLX slice-sharing semantics to observe prior writes. — Prefer a vectorized/scan-based formulation (or process chunks) and avoid in-place mutation of the same array being indexed; at minimum comment the recurring-timestep dependency.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3Text.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma3nText.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Granite.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Internlm2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Jamba.swift — findings: 2
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

# Pi review — batch 8 (MLX model ports + LoRA/DoRA adapters)

## Summary
Reviewed 17 mlx-swift-lm model files (dense, MoE, and hybrid SSM/attention transformers) plus the 3 MLXLMCommon LoRA/DoRA adapter files and the shared SSM kernel helper. Most files are faithful ports of their mlx-lm Python counterparts (attention/MLP shapes, RoPE init, residual add-norm ordering, KV cache indexing, expert stacking in sanitize(), and hybrid mamba/full-attention cache/mask indexing all check out against the shared KVCache/AttentionUtils contracts). Two real defect classes were found: an inverted top-k group-selection slice in the Nemotron-H MoE gate that silently selects the wrong expert groups, and a configuration-decoder division-by-zero on single-layer OpenELM configs. The LoRA/DoRA adapter math (forward vs fused consistency, magnitude/denom row scaling, freeze-key filtering) is consistent.

## Findings
- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift:381 — `topKGroupScores = sorted(groupScores, axis: -1)[.ellipsis, ..<2]` takes the two *smallest* expert gate values per group, but the reference (`mx.sort(..., axis=-1)[..., -2:]`) sums the two *largest*. MLX `sort` is ascending, so `..<2` is the wrong end of the sorted axis — this corrupts the per-group scoring that feeds `argPartition`/group selection (and the expert scores ultimately picked). — Nemotron-H MoE routers pick the wrong expert(groups), producing silently degraded/garbage logits for the whole MoE model. — Use the tail slice: `sorted(groupScores, axis: -1)[.ellipsis, -2...].sum(axis: -1, keepDims: true)` (or negate then take `..<2`) to match the Python `[-2:]`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OpenELM.swift:268,287 — `stride(from:through:by:)` divides by `Float(numTransformerLayers - 1)` to build `qkvMultipliers` and `ffnMultipliers`. For a config with `num_transformer_layers == 1` the divisor is 0, producing NaN multipliers that propagate into `makeDivisible` → `Int(NaN)` traps at decode time (crash). — Decoding a single-layer OpenELM config crashes. — Guard `numTransformerLayers > 1` before computing the multiplier strides (fall back to `[0.5, 1.0]`), matching the reference's linspace handling.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Olmo2.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Olmo3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OlmoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/OpenELM.swift — findings: 1
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
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/DoRA+Layers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRA+Layers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift — clean

# Batch 10 review — MLXLMCommon

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift:23 — `FileManager.default.enumerator(at:includingPropertiesForKeys:)` result is force-unwrapped with `!`. If the model directory does not exist or is not readable (e.g. a user-supplied `ModelConfiguration(directory:)` pointing at a missing path, or a failed/partial download), `enumerator` returns `nil` and `loadWeights` traps with a crash instead of surfacing a recoverable error up through the `throws` chain. — `throw` the error instead: `guard let enumerator = FileManager.default.enumerator(at: modelDirectory, includingPropertiesForKeys: nil) else { throw /* error */ }`.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift:65 — In the non-escaped branch the value is cut at the first `,` (`firstIndex(of: ",")`). A structured argument rendered as raw JSON that contains a comma inside an object/array (e.g. `call:foo{args:{"a":1,"b":2}}`, which the model may emit without `<escape>` markers) is truncated at that first comma, yielding a malformed key/value pair on the next iteration and corrupting the emitted `ToolCall.arguments`. — Scan for the matching closing brace for JSON objects/arrays before falling back to comma splitting, or require/verify `<escape>` markers around any comma-bearing value.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/LanguageModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelConfiguration.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelContainer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Models/Gemma.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Module+Extensions.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/AbstractModelRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Registries/ProcessorTypeRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEApplication.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/RoPEUtils.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SuScaledRoPE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tokenizer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/TokenizerLoader.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GLM4ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/JSONToolCallParser.swift — clean

# Pi review — batch 11 (mlx-swift-lm: Tool parsers, Tool types, WiredMemory, MediaProcessing, FastVLM, Gemma3)

## Summary
Reviewed 21 files covering the MLXLMCommon Tool layer (parsers, ToolCall/Tool/value types, ToolCallProcessor, UserInput, SerialAccessContainer, wired-memory helpers) and the MLXVLM media-processing + FastVLM + Gemma3 models. The streaming/tool pipeline is generally sound and null-safe. One concrete correctness bug found in MediaProcessing size math (aspect-ratio distortion); one low-severity parsing truncation edge in the Pythonic multi-call path.

## Findings
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/MediaProcessing.swift:251 — `fitIn(_:longestEdge:)` clamps `newLong = floatLongestEdge` before computing `newShort = floatLongestEdge * newShort / newLong`, so the denominator is already the clamped value and the expression collapses to `newShort = newShort` (no reduction). The shorter edge is never shrunk, so any image whose longest edge exceeds the target comes back as `(newShort, floatLongestEdge)` with the wrong aspect ratio (e.g. 1920×1080 with longestEdge=512 yields 1080×512 instead of 288×512). Public API used for image resizing/distortion. — Fix: compute the reduced side first, e.g. `let origNewLong = newLong; newLong = floatLongestEdge; newShort = floatLongestEdge * newShort / origNewLong`, or `newShort = newShort * (floatLongestEdge / newLong)` before reassigning `newLong`.

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/PythonicToolCallParser.swift:118 — `parseMultiple` (used on the EOS path for LFM2/multiple-call formats) matches arguments with the naive regex `(\w+)\((.*?)\)`, whose non-greedy `(.*?)` stops at the first `)` anywhere. Any argument value containing a `)` (nested call, or a `)` inside an otherwise valid string/nested container) is truncated, corrupting the argument and the whole call's JSON. The single-call `parse` path handles this correctly via back-tracking on the trailing `\]`, but the multi-call path does not. — Fix: parse each candidate with the same bracket-aware logic as `parse` (iterate over `match(range(at:0))` positions with balanced-paren scanning) or reuse a shared argument extractor.

## Branch
review branch (no changes made; findings only)

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/KimiK2ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/Llama3ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM2ToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/MistralToolCallParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/ParserUtilities.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/PythonicToolCallParser.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/XMLFunctionParser.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Tool.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCall.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallFormat.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolParameter.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Value.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/UserInput.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryPolicies.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/WiredMemoryUtils.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/MediaProcessing.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma3.swift — clean

# Pi sweep review — batch-12 (VLM model ports)

## Findings

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift:703 — Image features are never merged into the embeddings for any prompt. `prepareInputsForMultimodal` computes `chunkSize = imageFeatures.dim(1)` (the per-image feature-token count, e.g. 144 for a 384px image) and `chunkCount = imagePositions.count / chunkSize`, but the `Idefics3Processor.prepare` inserts exactly ONE image token (`promptTokens.insert(imageTokenId, at: ...)`, line 858), so `imagePositions.count == 1` and `chunkCount = 1 / 144 = 0`. The merge loop iterates zero times and the fallback `if start_idx < inputs_embeds.dim(1)` appends the whole text embedding (with the image token embedded as an ordinary vocabulary token). Consequence: for every image prompt the connector/vision output is discarded and the language model sees no image content, producing text-only/garbage answers while still accepting an image. Fix: iterate over `imageFeatures.dim(0)` images and for each single image-token position splice in all `chunkSize` feature rows at that position (e.g. drop the chunk grouping and do `segments.append(imageFeatures[i ... i + chunkSize])` at the one `pos`), keeping the token/feature counts consistent with the single-token processor.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:1706 — Multi-image input is guaranteed to throw `imageTokenCountMismatch`. The processor (`Gemma4Processor.prepare`) expands each of N image placeholders into `imageSeqLength` (default 280) image tokens, so the prompt contains `N*280` image tokens; but `visionTower` batches all N images and the pooler returns shape `(N, defaultOutputLength, d)` with `imageFeatures.dim(1) == 280` regardless of N. The check `expectedImageTokens != imageFeatures.dim(1)` (line 1708) therefore fails for any N > 1 and the call throws. The message generator accepts multiple images, so this is a guaranteed runtime error on a supported input path (not a silent corruption). Fix: either concatenate per-image features along the sequence axis so `dim(1)` sums to `N*280`, or pre-process the image batch per-image and only support a single image (reject N>1 explicitly with a clear error).

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift:330 — M-RoPE generation state (`_positionIds` / `_ropeDeltas`) is stored as mutable instance vars on the shared `LanguageModel` module instead of in the per-generation KV cache. `inputEmbeddings` overwrites them on every prefill (lines 1012-1013) and nulls them on text-only calls (lines 991-992), while `LanguageModel.callAsFunction` (lines 356-374) reads the stale value during autoregressive decoding. Two interleaved or concurrent generations on the same model instance will read/write each other's position ids, producing wrong position embeddings/incorrect tokens (a data race if concurrent, or a cross-call leak if sequential-but-reentrant). Fix: carry `positionIds`/`ropeDeltas` in the KV cache or as an explicit argument threaded through `prepare`/`callAsFunction` rather than mutating shared module state.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/GlmOcr.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Idefics3.swift — findings: 1

# Pi sweep batch-13 — MLX VLM model ports (LFM2VL, Mistral3, PaliGemma, Pixtral)

## Summary
Reviewed four MLX VLM model/processor implementations (ports of mlx-vlm). All are
largely faithful ports of the reference models, and I traced the cross-file contracts
(vision patch counts vs processor token counts, projector shapes, cache types) — these
are internally consistent, including the Mistral3/Pixtral image-token merging and the
LFM2 tile/downsample bookkeeping. The defensible issues are robustness and performance
items: one missing validation guard in Pixtral that the equivalent Mistral3 path has,
one hardcoded image-token-id coupling in LFM2's processor, and an O(N) per-element
MLXArray fill loop in LFM2's positional-embedding resizing. No secrets or committed-key
issues found (these are source model files, no credentials).

## Findings
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift:860 — `mergeInputIdsWithImageFeatures` interleaves `textSegments` with `imageEmbeddings` via `zip` and `imagePositions` with no validation that the number of image tokens in the prompt equals the number of vision patches. The byte-for-byte-analogous path in Mistral3 (Mistral3.swift `mergeInputIdsWithImageFeatures`) adds an explicit `guard imagePositions.count == numImagePatches else { fatalError(...) }`. — If the chat template / tokenizer inserts a number of image tokens that doesn't match what the vision tower produces (e.g. an image token that tokenizes to multiple ids, or a processor/config mismatch), `zip` silently truncates: excess text segments (with the trailing text) or excess image patches are dropped, producing a misaligned embedding that yields garbage model output with no error. — Add the same count guard as Mistral3: `guard imagePositions.count == numImagePatches else { fatalError("Image token count (...) does not match image patches (...)") }`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift:785 — The processor hardcodes `let imageTokenId = 396` for the image placeholder instead of deriving it from the config/tokenizer (the model's `LFM2VLConfiguration.imageTokenIndex` reads `image_token_id` from config, default 396, but the processor is a separate config with no such field). — If a LFM2 VL model is released with a different `image_token_id`, the placeholder-expansion loop never matches, the prompt keeps its original token count, and `LFM2VL.mergeInputIdsWithImageFeatures` hits its `fatalError` (image positions ≠ image features) at runtime. — Resolve the image token id from the tokenizer/config (e.g. `tokenizer.convertTokenToId`) with the 396 default as fallback, mirroring the Pixtral/Mistral3 processor pattern.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift:216 — In `VisionEmbeddings.resizePositionalEmbeddings`, when a resized image has fewer positions than `maxLength`, the remainder is filled with `resultedPositionalEmbeddings[i, j] = resizedEmbeddings[0]` inside a per-position loop. — Each iteration is a separate scalar MLXArray store/node, so for large `maxLength` (multiple tiles → thousands of positions) and multiple images this becomes O(maxLength) scalar ops per image just to tile one vector into a fixed buffer, needlessly increasing graph size and CPU copy cost on every image encode. — Fill the padding once with a broadcast/`expand` (e.g. `resultedPositionalEmbeddings[i, numPositions..<maxLength] = expandedDimensions(resizedEmbeddings[0], axis: 0)` in one slice or a tiled copy) instead of the loop.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/LFM2VL.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Pixtral.swift — findings: 1

# Pi sweep — batch 14 (MLXVLM model implementations)

## Findings

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift:324 — The vision `Attention.callAsFunction` receives `attentionMask` (threaded from `VisionModel.callAsFunction`, which builds `fullAttentionMask`/`windowAttentionMask` at 586-587 and selects per-block at 604) but discards it by calling `MLXFast.scaledDotProductAttention(..., mask: .none)`. The windowed-attention semantics the encoder was trained with (most blocks attend only within their `windowSize` window, a few use full attention) are never applied, so every token attends to all other tokens, producing incorrect vision features and making the entire mask-construction path dead work. — Fix: apply the mask, e.g. `mask: .array(attentionMask)` (converting/additive as `scaledDotProductAttention` expects), so the per-block window/full masks actually gate attention.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift:79 — `Language.Attention.init` computes `self.mropeSection` and `fatalError`s when `rope_scaling["mrope_section"]` is absent, yet `mropeSection` (and the whole `Language.applyMultimodalRotaryPositionEmbedding` helper at line 14) is never used: `Attention.callAsFunction` applies plain `rotaryEmbedding(queries/keys, offset:)` at 102-103. A Qwen2.5VL checkpoint whose `rope_scaling` lacks `mrope_section` (a valid configuration, since the language tower here uses 1-D offsets) crashes at init even though the value is never consumed. — Fix: remove the `fatalError`/dead mrope computation (or make it non-fatal), keeping only what the actual attention path reads.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift:692 — `preprocess(images:processing:)` reads `images[0].extent.size` with no empty-array guard (Qwen3VL's equivalent at Qwen3VL.swift added a `guard let first = processed.first else { throw ... }`). Calling the public `preprocess(images: [])` traps with an out-of-range crash. — Fix: guard against `images.isEmpty` (e.g. reuse the `guard let first` pattern from Qwen3VL) and throw `VLMError.imageProcessingFailure`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift:544 — Same missing empty-array guard: `preprocess(images:processing:)` indexes `images[0]` directly, crashing on an empty input array (Qwen3VL guards this case). — Fix: guard `images.isEmpty` first and throw a processing error.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift — findings: 3
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen2VL.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35MoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift — clean

# Pi sweep — batch-15

## Summary
Reviewed the MLX Swift VLM models (QwenVL, SmolVLM2), the VLM model/protocol/factory layer, the SwiftPM manifest, the MLXLMTests suite, and the docs-verification script. Most of the MLX/VLM production code is a clean port of the upstream mlx-vlm reference code with sound error handling and validation. The findings concentrate in SmolVLM2.swift (an unbounded video-frame sampling path and dead code). No security, crypto, or hardcoded-secret issues found (this is model-inference library and test code).

## Findings
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/SmolVLM2.swift:94 — `maxVideoFrames` is computed (`{ 20 }`) but never passed to or enforced in the video branch of `prepare`, so the frame count is bounded only by `targetFPS = max((10 - 0.9 * duration.seconds) * targetVideoFPS, 1)` sampled over the whole duration. A long video yields roughly one full-resolution frame per second with no cap (e.g. a 10-minute clip → ~600 normalized frames concatenated into one pixel tensor), so memory and prompt-token growth are unbounded and can OOM the device. Fix: thread `maxVideoFrames` into the sampling (e.g. cap the number of samples per duration in the `asProcessedSequence(targetFPS:)` closure) so long videos are capped at 20 frames.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/SmolVLM2.swift:86 — `let imageTokenId = 49190` is declared but never referenced anywhere in the struct (the image placeholder is keyed by the string `imageToken = "<image>"`), so it is dead code that implies an unused numeric token id. Consequence: misleading for maintainers who may assume an integer image-token pipeline exists. Fix: delete the unused `imageTokenId` property.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/QwenVL.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/SmolVLM2.swift — findings: 2
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

# Pi sweep — batch-16

## Findings

- [medium] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:104 — `Dictionary(uniqueKeysWithValues:)` traps with a fatal error if any two readers share a `lastPathComponent`. Document-folder ZIMs (line 70) and external security-scoped bookmarks (line 92) are both keyed by bare filename, so a bookmarked `wikipedia.zim` that collides with a file also dropped in Documents — or two bookmarks with the same basename from different folders — crashes the process on the very first Siri/Shortcuts intent. — Concrete consequence: process crash in `ZimfoRunner.load()`/instantiation, making the intents unusable whenever a same-named file is loaded from two sources. — Smallest safe fix: build the map with `Dictionary(dictionary: ...)` after deduping or use `Dictionary(uniqueKeysWithValues: ..., uniquingKeysWith: { first, _ in first })` (or build with a `reduce(into:)`) so a duplicate filename keeps the first reader instead of trapping.

- [low] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:90 — `url.startAccessingSecurityScopedResource()` (macOS bookmark path) is called but the matching `stopAccessingSecurityScopedResource()` is never called; the runner (and therefore the scope) is cached for the whole process in `cached`, so the security-scope extension is held open indefinitely rather than released when the reader is no longer active. — Concrete consequence: leaked/repeatedly-held security-scoped sandbox extension on macOS every time a fresh runner is built, exhausting the per-app extension budget and keeping the external directory mounted. — Smallest safe fix: mirror the stored flag from `startAccessingSecurityScopedResource()` and call `stopAccessingSecurityScopedResource()` when the runner is evicted, e.g. in a `deinit` on the `ZimfoRunner` before `cached` is replaced, or balance the call when building a new cache entry.

## Coverage
ios/MCPZimChat/App/MCPZimChatApp.swift — clean
ios/MCPZimChat/AppIntents/LocationFetcher.swift — clean
ios/MCPZimChat/AppIntents/ZimfoContext.swift — clean
ios/MCPZimChat/AppIntents/ZimfoIntents.swift — clean
ios/MCPZimChat/AppIntents/ZimfoRunner.swift — findings: 2
ios/MCPZimChat/Chat/AppTelemetry.swift — clean

# Pi sweep — batch-17

## Findings

- [low] ios/MCPZimChat/Chat/ChatSession.swift:1384-1386 — `substituteCurrentLocation` ends with an unconditional fallback (`if out["origin"] == nil, out["origin_lat"] == nil, out["origin_lon"] == nil { out["origin"] = coord }`) that runs for EVERY dispatched tool call whenever a GPS fix exists. The function's own comment labels it "if the tool is route_from_places", but no tool-name check is made, so `origin:"lat,lon"` is injected into non-routing tools (`search`, `article_overview`, `get_article_section`, `compare_articles`, `discuss_article`, `narrate_article`) that have no `origin`/`origin_lat`/`origin_lon` key, and it also defeats the function's earlier (step-2) guard + comment that deliberately skips location injection when a meaningful named `place`/`destination` is present — the exact case the code warns "would make the tool adapter prefer them over geocoding the place" (i.e. silently searching near the user's couch instead of the named place). Both call sites (`runGenerationLoop` and `executeDirectIntent`) route through this, so it fires on every GPS-enabled turn. — Gate the fallback to the actual routing/proximity tool names (e.g. a `routingTools`/`proximityTools` set) and skip it when a meaningful `place` or `destination` string is present.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:434,440,2580 — the full conversation (every user `send(_:)` text logged under "User" at 2580, plus assistant replies) is emitted to OSLog with `privacy: .public` at line 434 and appended verbatim to the persistent plaintext `LogArchive` at line 440. For a chat app whose surface includes clinical mdwiki queries, raw health/medical searches and personal chat history are written unredacted to the unified system log and stored in cleartext on disk in Application Support (surviving crashes and backup/forensics), not just when the user opts into the debug pane. — Mark the message-bearing portions `.private` (or redact/omit message bodies) in the OSLog line, and make the LogArchive persistence opt-in or length/retention-capped rather than implicit and unbounded.

## Coverage
ios/MCPZimChat/Chat/ChatSession.swift — findings: 2

# Pi sweep — batch-18 (iOS SDK Swift) findings

## Summary
Reviewed 13 files in the iOS target covering debug/telemetry plumbing (DebugReport, DiagnosticsUploader, LogArchive, SemanticReranker), the libzim Obj-C++ bridge, and three ML/model providers (Gemma4Provider, FoundationModelsProvider, FoundationModelsNativeTools). The bridge and reranker code is sound (zero-copy NSData-from-Blob lifetime handling and the actor-isolated reranker are correct). The real defects are concurrency races on provider/mutable‑state written from multiple threads, plus a couple of low-severity ledger/comment issues.

## Findings
- [medium] ios/MCPZimChat/Providers/FoundationModelsProvider.swift:45,61,73 — Mutable provider state (`nativeTools`, `warmSession`, `nativeInstructions`, `session`) is read/written from several public methods (`installNativeTools`, `setNativeInstructions`, `prewarmIfIdle`, `generateNativeTurn`, `generate`) that are nonisolated on a class declared `@unchecked Sendable` with no lock or `@MainActor`, and `generate()`/`generateNativeTurn()` touch them from inside detached `Task`s. `installNativeTools` (line ~97) replaces the `nativeTools` array and `ensureWarmSession` reads it concurrently → torn/raced array access and un-synchronized `warmSession` swap. Consequence: a data race that can crash or produce a session built from a partially-installed tool set when a tool install lands while a turn is streaming. Fix: route all state through a single serial actor/dispatch queue (wrap access in `queue.sync`) like the existing `state`/`continuations` fields, or annotate the class `@MainActor` and ensure callers hop to it.
- [medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:165-167,893-894 — The KV prompt-cache mirror (`promptKVCache`, `cachedTokens`, `generatedTokensThisTurn`) is mutated by the streaming `generate()` Task (reads at 615-616, writes `cachedTokens.append` at 893-894) while `resetPromptCache()` (185-190) and `unload()` (511-513) clear them — and `resetPromptCache` is explicitly documented to be called from memory-warning/backgrounded notifications (typically main thread) that can fire mid-generation. Race: array mutation while the Task appends → crash or a torn token mirror that silently diverges LCP and forces a full prefill (or worse, feeds a mismatched cache into `generateTokens`). Fix: guard the four fields with a lock (or move cache mutation into a single actor), and make `resetPromptCache`/`unload` synchronize against the in-flight generate.
- [low] ios/MCPZimChat/Common/DiagnosticsUploader.swift:43,71 — The `uploaded` ledger is a read-modify-write (read the `Set`, insert, write back) performed inside each `putFile` completion handler, which URLSession invokes concurrently on arbitrary queues. Two completions can read the same base set and clobber each other's insert. Consequence: a successfully-uploaded session log is dropped from the ledger and re-uploaded on the next background pass (duplicate entries in the off-device corpus). Fix: serialize the ledger update on a dedicated queue/actor, or build the pending list and mark entries as in-flight before starting uploads so re-backgrounding / concurrent completions can't double-send.
- [nit] ios/MCPZimChat/Providers/FoundationModelsProvider.swift:13-23 — The class-level design comment describes the native-tools path reusing a single warmed `LanguageModelSession` across turns, but `generate()` (line ~370+) actually creates a fresh `perCall` session every call and the code block even documents that warm-session reuse was dropped. The stale doc misleads readers into trusting the (abandoned) reuse design. Fix: update the header comment to state that a fresh session is created per `generate()` while `prewarm()`/`ensureWarmSession()` only keep the weights resident.

## Coverage
ios/MCPZimChat/Chat/DebugReport.swift — clean
ios/MCPZimChat/Chat/Message.swift — clean
ios/MCPZimChat/Common/DeviceProfile.swift — clean
ios/MCPZimChat/Common/DiagnosticsUploader.swift — findings: 1
ios/MCPZimChat/Common/LogArchive.swift — clean
ios/MCPZimChat/Common/SemanticReranker.swift — clean
ios/MCPZimChat/Common/ZimfoContext+Adapter.swift — clean
ios/MCPZimChat/Libzim/LibzimBridge.h — clean
ios/MCPZimChat/Libzim/LibzimBridge.mm — clean
ios/MCPZimChat/Libzim/LibzimReader.swift — clean
ios/MCPZimChat/Providers/FoundationModelsNativeTools.swift — clean
ios/MCPZimChat/Providers/FoundationModelsProvider.swift — findings: 2
ios/MCPZimChat/Providers/Gemma4Provider.swift — findings: 1

# Pi sweep — batch-19 (iOS Providers + Sharing) findings

## Summary
Reviewed seven files: the three provider abstractions (ModelProvider / MockProvider / LlamaCppProvider) and the four Sharing files (ChatSession+ModelSharing, ZimCatalog, ZimDownloadManager, ZimSwarmController). The sharing stack is well-engineered: catalog parsing is side-effect-free with pinned offline fallbacks, swarm staging moves are prefix-checked against the staging root and route only known voice-tree names, and the background-downloader correctly moves files in `didFinishDownloadingTo`. The one defect is an unsynchronized cross-thread write/read of `lastGenerationStats` in LlamaCppProvider on a class declared `@unchecked Sendable`. MockProvider and ModelProvider are clean.

## Findings
- [low] ios/MCPZimChat/Providers/LlamaCppProvider.swift:1255 (declared at 137) — `lastGenerationStats` (a `GenerationStats` struct containing `String` + `Double` fields) is written on the `Task.detached` background thread inside `generateLocked()` (holding `modelLock`), while ChatSession reads it on the main actor right after the stream finishes, without ever taking `modelLock`/a shared lock. The header comment claims "same actor context … so no torn reads", but the writer runs on a detached background thread and the reader on the main actor — different threads, so the unsynchronized read of a multi-field struct is a data race (torn/potentially-observed stale or partially-updated stats in the debug pane/A-B harness). Fix: guard the read and write with `modelLock` (or store it as an atomic/`Mutex`-protected `Sendable` value read through a synchronized accessor), and correct the misleading comment.

## Coverage
ios/MCPZimChat/Providers/LlamaCppProvider.swift — findings: 1
ios/MCPZimChat/Providers/MockProvider.swift — clean
ios/MCPZimChat/Providers/ModelProvider.swift — clean
ios/MCPZimChat/Sharing/ChatSession+ModelSharing.swift — clean
ios/MCPZimChat/Sharing/ZimCatalog.swift — clean
ios/MCPZimChat/Sharing/ZimDownloadManager.swift — clean
ios/MCPZimChat/Sharing/ZimSwarmController.swift — clean

# Pi review — batch-20 (iOS Views)

## Findings

- [medium] ios/MCPZimChat/Views/ChatView.swift:750 — `computeDisplayText` unconditionally drops everything up to and including the last occurrence of `" response"` (`if let r = t.range(of: " response", options: .backwards) { t = String(t[r.upperBound...]) }`). The comment intends to strip a Qwen-injected `thinking...scratchpad response` closer, but the search is never guarded by a preceding `...thinking` opener, so any ordinary assistant reply whose text contains the common substring " response" (e.g. "Here is my response to your question…") silently loses the whole prefix and renders only the tail after the word "response" (`"to your question…"`). This mangles a large fraction of real replies after the first-stream pass. Fix: only run this truncation when the text actually contains a `...thinking` opener; e.g. `guard t.contains("...thinking")` (or match the full `stimeman\s*...\s*response` template) before dropping from the closer, instead of firing on any `" response"`.

- [medium] ios/MCPZimChat/Views/NearbyShareView.swift:57 — two `.alert` modifiers are stacked on the same `List` (line 57 "Protected share" PIN prompt and line 72 "Nearby sharing" error alert). SwiftUI only honors the last `.alert` on a view; the earlier PIN-prompt alert is shadowed, so a locked/protected share can never actually present its PIN `SecureField` — the user tapping a `locked` swarm gets no working prompt (the `.alert(...)` at line 72 wins). Since this is also the auth path for protected shares, lock/PIN shares become unusable. Fix: use distinct presentation — e.g. `.alert` for the error and a `.sheet`/`.confirmationDialog` (or a single alert whose content switches on which state is set) for the PIN prompt, or put the PIN prompt on a separate subview.

## Coverage
- ios/MCPZimChat/Views/ChatView.swift — findings: 1
- ios/MCPZimChat/Views/DebugPane.swift — clean
- ios/MCPZimChat/Views/DownloadCatalogView.swift — clean
- ios/MCPZimChat/Views/HeroMediaView.swift — clean
- ios/MCPZimChat/Views/LibraryView.swift — clean
- ios/MCPZimChat/Views/MarkdownMessageParser.swift — clean
- ios/MCPZimChat/Views/MarkdownMessageText.swift — clean
- ios/MCPZimChat/Views/ModelPickerView.swift — clean
- ios/MCPZimChat/Views/NearbyShareView.swift — findings: 1
- ios/MCPZimChat/Views/OfflineContentSetupView.swift — clean
- ios/MCPZimChat/Views/PastLogsView.swift — clean

# Sweep findings — batch-21

- [low] ios/MCPZimChat/Views/PlacesWebView.swift:745 (openExternalURL) — the non-phone branch passes any value containing `://` straight to `UIApplication.shared.open` with no scheme allowlist, so a place `website` value of `file://`, `smb://`, `platform://`, etc. (sourced from the ZIM/Overture data or whatever the tool payload carries) gets handed to the system and can open a local file in QuickLook or a non-web external app, not just a browser. — Concrete consequence: tapping a crafted "Website" button on a place popup opens an unexpected local/arbitrary-scheme URL instead of being limited to http(s). — Smallest safe fix: after the `mailto:`/`https://` prepend logic, require the final scheme to be in `["http","https","mailto"]` (mirroring `webView(_:decidePolicyFor:)`) before calling `open(_:)`, else log+drop.

- [low] ios/MCPZimChat/Views/PlacesWebView.swift:444 and :885 — `webView.isInspectable = true` is set unconditionally (not gated behind `#if DEBUG`) on both the article webview and the places map webview, so Release builds ship with Web Inspector enabled on production webviews serving offline ZIM content. — Concrete consequence: a debugging affordance left on in distributed builds enables inspection of in-app web content. — Smallest safe fix: gate both `webView.isInspectable = true` (and `config.preferences.isElementFullscreenEnabled`, already `#available`-guarded) behind `#if DEBUG` (or an internal flag).

- [low] ios/MCPZimChat/Views/RouteWebView.swift:668 — `webView.isInspectable = true` likewise set unconditionally in the routing map webview, enabling Web Inspector in Release builds. — Concrete consequence: production web content is inspectable. — Smallest safe fix: gate `isInspectable = true` behind `#if DEBUG`.

## Coverage
ios/MCPZimChat/Views/PlacesWebView.swift — findings: 2
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

# Pi review — batch-22 (voice + test files)

## Summary
Reviewed the VoIP/TTS voice controller and service, plus five test suites and the Eval CLI. The core streaming voice loop is carefully engineered (VAD, barge-in, TTS memory policy). One real robustness defect found in the TTS streaming loop: a failed `speakChunk` still advances the spoken-character cursor, silently and permanently dropping the unspoken fragment of the reply (and under persistent failure, the whole answer is never voiced although the turn logs as finished). The TTS service and all four test files / Eval CLI are clean.

## Findings
- [medium] ios/MCPZimChat/Voice/VoiceChatController.swift:907-953 — When `tts.speakChunk(toSpeak, boundary:)` at line 907 throws, the `catch` at 909 only logs the error, but `spokenUpTo += prefix.consumedCharacters` at line 953 still executes and `advancedThisPass = true`. The failed chunk's text is therefore marked as already-spoken and is never retried; subsequent polls compute `newFragment = full.suffix(full.count - spokenUpTo)` which skips past it. On a single transient failure the affected fragment is permanently dropped; if the backend fails repeatedly (MLX OOM / bad chunk) the entire reply is skipped and `streamAssistantReply` still logs "TTS done" and returns to listening without the user hearing the answer. — Fix: only run `spokenUpTo += prefix.consumedCharacters` and `advancedThisPass = true` inside the success path of the `do` block (advance in the `do` after the successful `speakChunk`, and break or retry-limit on failure so text is not silently consumed).

## Coverage
ios/MCPZimChat/Voice/TTSService.swift — clean
ios/MCPZimChat/Voice/VoiceChatController.swift — findings: 1
ios/MCPZimChatMacTests/CatalogParsingTests.swift — clean
ios/MCPZimChatMacTests/ConversationalEvalTests.swift — clean
ios/MCPZimChatMacTests/GemmaToolEmissionTests.swift — clean
ios/MCPZimChatMacTests/ModelSharingTests.swift — clean
ios/MCPZimChatMacTests/ZimfoIntentsTests.swift — clean
ios/MCPZimEval/EvalCLI.swift — clean

# Pi sweep — batch-23 (ios/MCPZimEval + project.yml)

## Findings

- [low] ios/MCPZimEval/ProbeE2ECLI.swift:443 — `--probe-discuss` defaults `gguf` to a hardcoded personal absolute path `/Users/jasontitus/experiments/.../lfm2.5-8b-a1b-ft.imx.IQ3_XS.gguf` — any run without an explicit `--gguf` (or on any machine other than that one developer's) fails the default LLM load / `attributesOfItem` path check, and the tool's documented "no download" convenience behavior is tied to one laptop; also embeds a personal home-directory path in a shared codebase. — Smallest safe fix: default `gguf` to an empty/nil value and require `--gguf` (or resolve via an env var like `MCPZIM_GGUF`), failing with a clear message instead of a silently broken fixed path.
- [low] ios/MCPZimEval/EvalHarness.swift:956 — the per-scenario memory ceiling is checked against `postTurnSample.rssMB`, a single instantaneous RSS sample taken after generation stops, not the peak observed during decode/prefill; `await probe.stop()` (whose return would be the continuous-interval peak) is discarded — the `maxPeakMB` docstring (line ~190) promises "the scenario fails if MLX peak memory exceeds this at any turn" — the phone-jetsam repro scenarios (`bars_sc_caltrain_chain` etc.) therefore can pass despite a transient >cap spike, which is the exact regression they were added to catch. — Smallest safe fix: compare the interval peak returned by `startContinuous`/`stop()` against `cap` instead of the lone post-turn sample.

## Coverage
ios/MCPZimEval/EvalHarness.swift — findings: 1
ios/MCPZimEval/LlamaCppProbeCLI.swift — clean
ios/MCPZimEval/ProbeCompareCLI.swift — clean
ios/MCPZimEval/ProbeE2ECLI.swift — findings: 1
ios/project.yml — clean

# Pi sweep batch-24 — mcpzim / MCPZimKit

## Findings

- [low] ios/tools/eval.sh:88 — the `xcodebuild test ... 2>&1 | tee "$TEST_LOG" > /dev/null || true` pipeline swallows the test process's exit status, and nothing after it sets a non-zero exit — so a scripted or CI invocation of this eval returns 0 even when `ConversationalEvalTests` fail — a regression run would report green and gate nothing; drop the `|| true` (or capture xcodebuild's status, then `exit` non-zero when the grepped results show failures).
- [low] swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift:83 — `name` and the serialized tool result (`pretty`/`error`) are interpolated verbatim into `<tool_response name="\(name)">\(pretty)</tool_response>` with no escaping — a tool result or error string containing the `</tool_response>` sentinel (or a `name`/payload containing `"`) terminates/malforms the block and the rest of the payload is re-read as further tool calls, corrupting the injected transcript and the loop; escape XML/attribute characters or strip the sentinel from both the name and the result before embedding.
- [low] swift/Sources/MCPZimKit/AnswerAttribution.swift:146 — single-digit numbers (`w.count >= 2` guard) are dropped entirely from `weightedTokens`/`tokenSet`, so a sentence whose only content is a single-digit figure (e.g. "It took 5 days") never contributes a numeric token; it falls through the `tokens.count > 2 || hasNumeric` connective-prose guard (line 66) and is auto-marked `support: 1`/supported even when no passage contains that number — a hallucinated single-symbol count is never flagged, undercutting the module's stated "numbers are the strongest hallucination tell"; count single digits (length-1 numbers) at a lower weight instead of skipping them.

## Coverage
ios/scripts/mcp-crashes.sh — clean
ios/scripts/mcp-deploy-verify.sh — clean
ios/scripts/mcp-logs.sh — clean
ios/scripts/mcp-report.sh — clean
ios/scripts/testflight-upload.sh — clean
ios/tools/eval.sh — findings: 1
mcpzim/__init__.py — clean
mcpzim/__main__.py — clean
mcpzim/cli.py — clean
mcpzim/content.py — clean
mcpzim/geocode.py — clean
mcpzim/library.py — clean
mcpzim/routing.py — clean
mcpzim/server.py — clean
pyproject.toml — clean
swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift — findings: 1
swift/Package.swift — clean
swift/Sources/MCPZimKit/AnswerAttribution.swift — findings: 1
swift/Sources/MCPZimKit/ArticleCache.swift — clean
swift/Sources/MCPZimKit/ArticleHeuristics.swift — clean

# Pi sweep — batch 25

## Findings

- [low] swift/Sources/MCPZimKit/Gemma4ToolFormat.swift:30 — `formatValue` emits every string as `<|"|>\(s)<|"|>` without escaping or rejecting the `<|"|>` sentinel (and the parser `Gemma4ToolCallParser.takeQuotedString` / `impliedBodyEnd` reads until the first `<|"|>`). A string value containing that byte sequence — e.g. a place name or article snippet carried through `formatToolResponse` from ZIM/geocoder content — truncates at the first sentinel, silently dropping the rest of the value and corrupting the model context / a round-trip tool call. Fix: escape embedded sentinels (split the quoted payload) or strip/reject `<|"|>` inside string values in `formatValue` (and the description/name paths in `formatToolDeclaration`).
- [low] swift/Sources/MCPZimKit/ChatToolCallParser.swift:107 — in the strict path the closer set for the `<tool_call>`/`<|tool_call>` openers ends with a bare `">"`. If the model emits `<tool_call>{json}` with no `</tool_call>`/`<tool_call|>` yet and the very next non-whitespace stream byte is a `>` from prose that follows in the buffer, the call is treated as complete and ends at that stray `>`, producing a truncated dispatch range and leaving the real closer unremoved from the transcript. On the streaming hot path this can dispatch a broken call / mis-truncate. Fix: require the JSON to be followed by a whitespace-or-end boundary before accepting the bare `>` closer, or drop it for the generic opener.

## Coverage
swift/Sources/MCPZimKit/ArticleSections.swift — clean
swift/Sources/MCPZimKit/ChatToolCallParser.swift — findings: 1
swift/Sources/MCPZimKit/ChatTurn.swift — clean
swift/Sources/MCPZimKit/ConversationFocus.swift — clean
swift/Sources/MCPZimKit/ConversationThreads.swift — clean
swift/Sources/MCPZimKit/Embeddings.swift — clean
swift/Sources/MCPZimKit/Gemma3Template.swift — clean
swift/Sources/MCPZimKit/Gemma4PromptTemplate.swift — clean
swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift — clean
swift/Sources/MCPZimKit/Gemma4ToolFormat.swift — findings: 1
swift/Sources/MCPZimKit/GeoMath.swift — clean
swift/Sources/MCPZimKit/Geocoder.swift — clean

# Batch 26 review — swift/Sources/MCPZimKit

## Findings

- [medium] swift/Sources/MCPZimKit/LFM25Template.swift:246-262 — `stripReasoning` unconditionally truncates the assistant output at the LAST literal `" response"` substring in the already-cleaned text, even when no reasoning span was present or after a well-formed ` thinking… response` span was already removed by the `while` loop. The `while` loop strips every ` thinking … response` span, so the dangling-close branch then re-searches the *remaining* prose for `" response"` and keeps only everything *after* it. Since the branch runs regardless of whether any reasoning was detected, any legitimate answer containing the word "response" (e.g. "In response to your question, the Eiffel Tower…", "…I couldn't find the requested response") gets its entire lead chopped off, dropping the actual subject of the reply and mangling the spoken/displayed caption. This affects both final-message and streaming-buffer paths (the protocol doc says it runs on streaming buffers too). Consequence: the user hears/sees a truncated or wrong answer for normal, non-reasoning turns. Smallest safe fix: only apply the dangling-close truncation when a reasoner artifact is actually present — e.g. gate it on a flag set inside the `while` loop (a stripped span) OR, for the close-only case, require the leading segment before the candidate `" response"` to end in a scratchpad marker/opening tag and the trailing segment (after it) to be non-empty — and never cut output where the segment before `" response"` already contains sentence-final punctuation indicating a complete user-facing sentence. At minimum, require that the text before the last `" response"` does not contain a `.`/`?`/`!` immediately preceding a reasoning marker so plain prose is never collapsed.

## Coverage
swift/Sources/MCPZimKit/IntentRouter.swift — clean
swift/Sources/MCPZimKit/LFM25Template.swift — findings: 1

# Pi sweep — batch-27

## Findings

- [medium] swift/Sources/MCPZimKit/QwenChatMLTemplate.swift:337 — `stripReasoning`'s dangling-closer fallback treats ANY occurrence of the substring `" response"` (backwards search) as the end of a reasoning block and truncates everything before it, even when there was no ` thinking` opener / no reasoning span at all. The preceding `while` loop only removes matched ` thinking… response` pairs, so any remaining `" response"` in genuine assistant prose (e.g. "In response to economic pressure…") survives to this branch and gets the leading text deleted — the user-facing answer is truncated (possibly to nearly nothing). Fix: only take the dangling-close branch when a corresponding unmatched ` thinking` opener is actually present in `out` (e.g. track that some `" response"` exists WITH a preceding `" thinking"` that has no close); otherwise return `out` unchanged.

- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:760 — the `near_places` dispatch reads `limit = (args["limit"] as? Int) ?? 25` with no clamping, unlike every other bounded tool in the same file (`search` clamps to 1…50, `nearby_stories`/`narrate` to 1…10, `article_overview` to 1…10; the `search` case even documents guarding against an unclamped `Int.max`). A model-supplied unbounded `limit` (this adapter's whole purpose is trusting fallible model args) flows straight into `service.nearPlaces(limit:)` and bloats the `results` array serialized back into the model context (token/memory blowup, slow JSON) — the same `radius_km` is likewise unvalidated. The same unclamped `limit` exists in the `near_named_place` case at line 700. Fix: `let limit = max(1, min(50, (args["limit"] as? Int) ?? 25))` (and clamp `radius_km` to a sane positive range) in both places, matching the `search` guard.

- [low] swift/Sources/MCPZimKit/MemoryStats.swift:28 — `physFootprintMB()` is documented as "base-10, matching most memory UIs" but divides by 1_048_576 (base-2 MiB), not 1_000_000. So the reported value is actually MiB and ~5% lower than the true base-10 MB that UIs/displays and the memory-band thresholds elsewhere (MemoryProbe's ≥5/6/7 GB jetsam bands) imply, making cross-tool comparisons misleading. Fix: either divide by 1_000_000 to match the comment, or correct the comment to say base-2 / 1_048_576.

## Coverage
swift/Sources/MCPZimKit/MCPToolAdapter.swift — findings: 2
swift/Sources/MCPZimKit/MemoryProbe.swift — clean
swift/Sources/MCPZimKit/MemoryStats.swift — findings: 1
swift/Sources/MCPZimKit/ModelTemplate.swift — clean
swift/Sources/MCPZimKit/PlacesPayload.swift — clean
swift/Sources/MCPZimKit/QueryComplexity.swift — clean
swift/Sources/MCPZimKit/QwenChatMLTemplate.swift — findings: 1

# Pi review — batch-28

## Findings

- [medium] swift/Sources/MCPZimKit/SZRGGraph.swift:154 — count-driven `reserveCapacity` runs with zero per-field bounds validation — `SZRGGraph.parse` only checks `data.count >= 32` before `lat.reserveCapacity(numNodes)`, `edgeTargets.reserveCapacity(numEdges)`, `geoms.reserveCapacity(numGeoms)`, `names.reserveCapacity(numNames)`. A crafted/corrupt graph.bin header advertising e.g. `numNodes = 0xFFFFFFFF` forces a ~34 GB Double-array allocation (and similarly multi-GB for the edge arrays) that traps/OOM-kills the process before the first `readU32()` can throw `.truncated`. The same file's own DS4 comments treat graph.bin as untrusted and guard the offset tables, and the parallel spatial parser (`SZRGSpatial.requireBytes`) validates count×perEntry against remaining bytes *before* reserving — this path lacks that guard. — Crash/DoS on loading a malformed streetzim graph. — Add a `requireBytes`-style check (e.g. `numNodes*8 <= remaining` for lat/lon, `numEdges*4`/`*1`/`*8`/`*4` etc.) before each `reserveCapacity`, mirroring `SZRGSpatial.requireBytes`.

- [medium] swift/Sources/MCPZimKit/SZRGSpatial.swift:277 — CSR edge window from untrusted SZRC is not clamped to `edgeCount` — `SpatialGraph.edgesOfNode` (and the same loop in `Router.aStarSpatial`) reads `eStart = Int(cell.cellAdj[local])`, `eEnd = Int(cell.cellAdj[local+1])` and then indexes `cell.edges[base]` / `base+1...` where `base = ei*5`. `SZRCCell.parse` reads `cellAdj` values verbatim with no validation that they stay ≤ `edgeCount`, so a crafted cell with an oversized window drives `cell.edges[base]` out of bounds → Swift Array subscript trap (crash). The monolithic `Router.aStar` deliberately clamps the equivalent CSR window (`guard start >= 0, start <= end, end <= graph.numEdges`); the spatial path is missing the same clamp. — Crash/DoS on routing against a crafted SZRC cell. — Clamp/validate the window (`guard eStart >= 0, eStart <= eEnd, eEnd <= edgeCount` and skip targets past `numNodes`) before indexing `cell.edges`.

- [low] swift/Sources/MCPZimKit/SZRGSpatial.swift:198 — leading polyline coordinate read is not clamped to the declared geom window / blob — `SZRCCell.decodeGeom` reads `readInt32LE(base, at: start)` and `readInt32LE(base, at: start + 4)` (offsets `start..start+8`) guarded only by `start >= 0, end <= geomBlob.count`, not by `start + 8 <= geomBlob.count`. A short (<8 byte) geometry window ending at the blob tail drives the second `readInt32LE` past `geomBlob.count` (data-only OOB read of up to 4 bytes). The varint reads below correctly clamp to `end`; the initial `lon0`/`lat0` reads don't. — OOB/garbage data read on crafted or degenerate SZRC cell. — Add `guard start + 8 <= geomBlob.count` (or read via a cursor bounded by `end`/`geomBlob.count`) before the two `readInt32LE` calls.

## Coverage
swift/Sources/MCPZimKit/ReferenceResolver.swift — clean
swift/Sources/MCPZimKit/RegexCache.swift — clean
swift/Sources/MCPZimKit/Router.swift — clean
swift/Sources/MCPZimKit/SZRGChunked.swift — clean
swift/Sources/MCPZimKit/SZRGEncoder.swift — clean
swift/Sources/MCPZimKit/SZRGGraph.swift — findings: 1
swift/Sources/MCPZimKit/SZRGSpatial.swift — findings: 2
swift/Sources/MCPZimKit/StreamingSpeechPolicy.swift — clean
swift/Sources/MCPZimKit/StubZimService.swift — clean
swift/Sources/MCPZimKit/ToolLoopGuard.swift — clean
swift/Sources/MCPZimKit/ZimReader.swift — clean

# Pi sweep — batch-29

## Summary
Reviewed `ZimService.swift` (the in-process service implementation, ~2045 lines)
plus eleven test files covering attribution, factoids, article/speech cleanup,
bundled-article resolution, tool-call parsing, conversation focus/continuation/
threads, and discuss-article linking. `ZimService.swift` is well-engineered —
memory-bounded LRU caches, OOM guards on full scans, and defensive wiki-tag
sanitization are all present and correct. Test files are all concrete,
deterministic fixtures that assert real behavior. One low-severity heuristic
defect found in `articleByTitle`'s language-prefix stripping. No security,
correctness, or test-effectiveness defects rose to reportable severity.

## Findings
- [low] swift/Sources/MCPZimKit/ZimService.swift:1304-1320 — `articleByTitle` strips a leading "lang:" segment whenever the first colon-delimited token is 2–3 letters all-alphabetic, but applies this to ALL titles, not just OSM `en:`-style wiki tags. "FAQ: …", "US: …", "EU: …", "TV: …"-style real article titles have their first segment silently removed, so the direct-path probes and the fallback suggester look up only the fragment after the colon and return a wrong article or `notFound`. — Narrow the guard to a closed allowlist of known language codes (en, fr, de, es, it, pt, ru, zh, ja, …) instead of "any 2–3 letter word", or only strip when the prefix is immediately followed by a lowercased known-code match; leave genuinely coloned titles untouched.
- [low] swift/Sources/MCPZimKit/ZimService.swift:2019-2023 — `renderLeadSnippet`'s fallback re-computes `html.count` and materializes `String(html.prefix(64*1024))` per candidate on the search hot path when the lead-only fast path finds no prose before the first heading; `html.count` is O(n) over the full body, so a large article costs two full passes every snippet miss (the lead path also already capped at 64 KB). — Add the article-cache bytes to a per-(zim,path) cache or just index the fallback read with `html.utf8.prefix(cap)` once and reuse the count, avoiding the repeated full-length `count` traversal.

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

# Pi sweep — batch-30

## Summary
Reviewed 16 Swift XCTest unit-test files in `swift/Tests/MCPZimKitTests/`. They are regression/behavioural suites
for the MCPZimKit library (discuss-mode retrieval / ArticleHeuristics, Embeddings + hashing embedder + LRU index,
Gemma4/LFM25 prompt and tool-format templating, geocode variant ladder, hot-split geocode branch selection,
intent routing / reply synthesis, near-places chip/category/search-data layers + kind-synonym + name-search
fallback, places-payload parsing, and query complexity classification). Every test carries concrete assertions
against hard-coded expectations that a real regression would break. No skips, no `@available` gating, no
sleep/time/network patterns, no tautological assertions, no mocks of the thing under test (the in-memory
`MapReader`/`TrackingReader`/`StubZimService` fixtures are legitimate collaborator stubs). No findings.

## Findings
(none)

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

# Pi sweep — batch 31

## Summary
Reviewed 10 Swift (MCPZimKit) unit-test files and 10 tooling files
(fine-tune training scripts + an A/B comparison harness). The Swift
tests and benchmark files are thorough and defensible; no defects found
there. The PT-tooling findings are all in the SFT/training data path:
an SFT data-masking gap that trains on prompt tokens, an edge case
where trailing gradients are never applied, a machine-hardcoded sys.path
that silently degrades training-data generation on other hosts, a
resume-position bug in the data generator, and a fine-tune pipeline
shell script that assumes a pre-built llama.cpp tree it never clones or
builds.

## Findings
- [medium] tools/fine-tune/finetune_cuda.py:149 — `render_row` sets `enc["labels"] = list(enc["input_ids"])`, i.e. labels are the full chat-templated sequence including the system preamble and the user prompt; only padding is masked to -100 (collate line 157). The causal-LM loss then also trains the model to reproduce the user's query/system text verbatim instead of only the assistant completion, which biases SFT quality and wastes capacity on reproducing user input (the exact regression mode the companion unsloth path's `enable_thinking` handling exists to avoid). Fix: build labels in `render_row` using conversation boundaries, masking every non-assistant token to -100 (e.g. re-tokenize the assistant turns, or split the templated text at assistant role markers) before returning.
- [low] tools/fine-tune/finetune_cuda.py:327-334 — the training loop only calls `optimizer.step()`/`zero_grad()` inside `if accum >= args.grad_accum` and never flushes accumulated gradients after the loop. When `--grad-accum > 1` and `--iters` is not a multiple of it (or `--save-every`/crash interrupts before a full accumulation window), the final partial batch's gradients are discarded, so the model silently trains fewer effective steps than requested. Fix: after the loop, `if accum > 0: clip + step + zero`.
- [low] tools/fine-tune/finetune_lfm2.sh:184,191 — the conversion (step 4) and quantization (step 5) invoke `$LLAMA_CPP_SRC/convert_hf_to_gguf.py` and `$LLAMA_CPP_SRC/build/bin/llama-quantize` directly, but unlike `finetune.sh`/`finetune_cuda.sh`/`finetune_unsloth.sh` this script never clones `llama.cpp` into `LLAMA_CPP_SRC` nor builds `llama-quantize`. On a fresh checkout the pipeline runs all the way through a multi-hour fine-tune and then hard-fails at conversion/quantization with "command not found". Fix: add the same `git clone` + `cmake --build --target llama-quantize` bootstrap block the sibling scripts use, gated on the artifacts being absent.
- [low] tools/fine-tune/generate.py:672 — `_eval_preamble()` hardcodes `sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")`; on any other host the `from eval import SYSTEM_PREAMBLE, _build_tool_block` raises and the function silently falls back to a much weaker inline preamble. Because the generated training rows are supposed to byte-match the eval harness preamble, this silently changes the training distribution on non-author machines with no warning. Fix: make the llama-smoke path an env var / `--eval-path` arg and log (or raise) when the import fails instead of silently substituting.
- [low] tools/fine-tune/generate.py:858 — resume uses `queries = [q for (_c, q) in sampled][done:]` where `done` is the number of lines already in the output file, but a prior run's `generate_one` failures write no line (they are only logged). Any API failure during a prior run shifts the positional alignment, so a resumed run re-generates already-succeeded queries and permanently skips others, producing a non-deterministic final set that stops at `done + len(queries)` successes rather than the intended `n` unique seeds. Fix: persist a per-query completion marker (e.g. write the query into each line, or track a sentinel per attempt) and on resume skip by query identity, not byte/line count.

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
tools/fine-tune/finetune_cuda.py — findings: 2
tools/fine-tune/finetune_cuda.sh — clean
tools/fine-tune/finetune_lfm2.sh — findings: 1
tools/fine-tune/finetune_unsloth.py — clean
tools/fine-tune/finetune_unsloth.sh — clean
tools/fine-tune/generate.py — findings: 2

# Pi sweep review — batch-32

## Summary
Reviewed 13 files in `tools/fine-tune`, `tools/gemma-smoke`, and `tools/llama-smoke`. These are local training-data generators, fine-tune retry/shell scripts, and model smoke/bench harnesses. Most files are straightforward and raise no defensible defects. One real correctness defect found in the 3-turn chain data generator's resumable target accounting (it over-generates on resume and mis-accounts custom template weights). One low supply-chain note on an unpinned foreign Swift package dependency.

## Findings
- [medium] tools/fine-tune/generate_chains3.py:904-955 — `total_targets` is computed from the full `args.n * weight` and `todo = max(0, args.n - existing)` (line 908) only gates the early-exit; it is never used to reduce the per-template task counts at line 954-955. So a resumed run (`existing=400, n=800`) generates all 800 per-template rows again, ending with ~n+existing rows instead of topping up to n — the documented "Target row count (resumable)" behavior is broken. The same accounting also under-generates when `--templates` weights sum to <1 with no bare-name remainder (0.5 only yields 0.5·n rows), contradicting the "target n" contract. — Consequence: double LLM-API cost on every resumed run and/or a dataset that under/over-fills per template. — Fix: scale `total_targets` down by the remaining `todo` ratio (or subtract per-template counts proportional to weights from `existing`) before creating tasks, and normalize custom weights to sum to 1.
- [low] tools/gemma-smoke/Package.swift:16 — depends on an unpinned foreign GitHub repo `https://github.com/yejingyang8963-byte/Swift-gemma4-core.git` with a loose `from: "0.1.0"` range, resolved into the tool process that reads user messages/prompts. — Consequence: arbitrary/supply-chain code from an unaudited third-party source is fetched into and linked into the build, and any new 0.1.x tag is auto-pulled. — Fix: pin to an exact `exact:` version and commit a checksum/resolution (and prefer an audited mirror of the model core if one exists).

## Coverage
tools/fine-tune/generate_chains.py — clean
tools/fine-tune/generate_chains3.py — findings: 1
tools/fine-tune/generate_places_diverse.py — clean
tools/fine-tune/retry_lfm2_train.sh — clean
tools/fine-tune/split_chain_rows.py — clean
tools/fine-tune/train_all.sh — clean
tools/fine-tune/train_all_cuda.sh — clean
tools/fine-tune/v7_eval_and_memsweep.sh — clean
tools/gemma-smoke/Package.swift — findings: 1
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/main.swift — clean
tools/llama-smoke/bench.py — clean

## Branch
(batch review; no branch changes made)

# Pi review — batch-33

## Summary
Reviewed the llama.cpp / MLX model eval & benchmark harnesses (tools/llama-smoke,
tools/llm-smoke) and the Zimfo debug-log analytics pipeline (tools/logpipe). These
are local dev/eval tools, not network-exposed server code, so findings are
correctness/data-pipeline and tooling-hygiene defects rather than security
vulnerabilities. The main real bug is in the logpipe pipeline: conversations that
share a session id across `conversation reset` boundaries are silently dropped
from the judge pass because parse_log.py never implements the session-suffixing
that ingest.sh's comment claims.

## Findings
- [medium] tools/logpipe/parse_log.py:59 (with tools/logpipe/ingest.sh:74-75, tools/logpipe/prep_judge.py:31) — `flush_conv()` sets `"session": path.stem` for EVERY conversation, including the 2nd..Nth conversations a log splits into at `conversation reset`; ingest.sh:74-75's comment claims "later ones are suffixed so dedup stays stable" but no suffixing exists. prep_judge.py:31 filters on `conv.get("session") not in judged`, so once any verdict for that session id is recorded, every reset-split conversation sharing the id is excluded forever from `_to_judge.jsonl`. — Conversations after a reset are never re-emitted for judging, so the retrieval-vs-model failure analysis under-counts real failures and the corpus stats in report.py are silently biased. — Suffix reset conversations in parse_log.py (e.g. append `#1`, `#2` per reset) so each gets a unique stable session id, matching the ingest.sh comment; then ingest/prep_judge dedup works as documented.
- [low] tools/llama-smoke/sweep.sh:5,21 — `set -euo pipefail` combined with `eval.py ... | tee -a "$OUT" | grep ... | head -10`: once `head` has consumed 10 matching lines it closes the pipe, sending SIGPIPE back through `grep`/`tee`/`eval.py`; because the script uses `pipefail` under `set -e`, the pipeline's non-zero (141) status aborts the whole script. — Multi-turn scenarios print several `final_content`/`RESULT`/`model=` lines that can exceed head's 10-line window, silently truncating the sequential quant sweep after the first qualifying model instead of running all four quants. — Drop `head -10` (grep already limits output), or guard with `|| true` / use `sed -n '1,10p'` which tolerates a closed stdout.
- [low] tools/llama-smoke/eval.py:117-145 — `_install_generation_noop_extension()` monkeypatches the process-global `jinja2.Environment.__init__` at import time (regardless of model), permanently altering every jinja2 Environment created in the process and registering a `_GenerationNoop` tag that silently drops `{% generation %}...{% endgeneration %}` blocks. — Any other library in the same process that renders jinja2 templates will silently discard content inside those tags; the patch also runs for Gemma/Qwen runs that never needed it. — Restrict the patching to the LFM2/pythonic path (only when the baked template is actually going to be parsed) or scope it to a private Environment instance instead of mutating the global class.

## Coverage
tools/llama-smoke/eval.py — findings: 1
tools/llama-smoke/grid.py — clean
tools/llama-smoke/sweep.sh — findings: 1
tools/llm-smoke/bench.py — clean
tools/llm-smoke/bench_kv.py — clean
tools/llm-smoke/bench_memory.py — clean
tools/llm-smoke/bench_memory_gemma4.py — clean
tools/llm-smoke/eval.py — clean
tools/llm-smoke/eval_gemma4.py — clean
tools/llm-smoke/eval_gemma4_native.py — clean
tools/llm-smoke/gemma4_format.py — clean
tools/logpipe/ingest.sh — findings: 1
tools/logpipe/parse_log.py — findings: 1
tools/logpipe/prep_judge.py — findings: 1
tools/logpipe/report.py — clean

# Pi review — batch-34

## Summary
Reviewed the five unit-test modules under `tests/` (content, geocode, library,
routing) together with the `mcpzim.{content,geocode,library,routing}` modules
they exercise. All five files are correctly written: the normalization, prefix
padding, ZIM classification, and SZRG graph codec/round-trip assertions match the
actual implementation and would pass. No test is broken, flaky, or a guaranteed
pass. The only finding is a single maintainability nit — a test whose name
contradicts what it actually asserts. No security, correctness, or concurrency
defects were found in these test files.

## Findings
- [nit] tests/test_content.py:33 (with mcpzim/content.py `_snippet`) — `test_snippet_empty_on_no_match` is misnamed: for a non-empty text with no query match `_snippet` returns a `text[:width]` prefix (the very next line asserts `out.startswith("nothing here")`, and the in-test comment concedes "Fall back to a prefix — not empty but bounded"), so the name "empty_on_no_match" communicates behavior that is the opposite of what the test verifies. — A future maintainer reading the name may "trust" that a no-match snippet is empty and misjudge downstream UI/snippet handling. — Rename to `test_snippet_falls_back_to_prefix_on_no_match` (and drop the misleading name), or assert on the documented fallback contract explicitly.

## Coverage
tests/__init__.py — clean
tests/test_content.py — findings: 1
tests/test_geocode.py — clean
tests/test_library.py — clean
tests/test_routing.py — clean

# Pi sweep review — batch-35

## Summary
Reviewed 11 LocalSwarm engine test files (AuthToken, BenchmarkMetrics, ChunkerMemory, Conformance, Discovery, EngineCore, Folder, Hosting, ManifestCache, QUICLoopback, Security), the KokoroSwift Package.swift, and the Albert MLX model sources. The test files validate engine behavior (wire framing, chunking memory bounds, path-traversal rejection, conformance vectors) and the model files are MLX ports of the ALBERT encoder. Findings concentrate in the Albert sources: three unused/dead classes that duplicate live logic or carry randomly-initialized weights, and pervasive force-unwrapped weight lookups that crash without diagnostics if a model key is missing.

## Findings
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertIntermediate.swift:5-20 — `AlbertIntermediate` is never referenced anywhere in the package (`grep` finds only its own declaration), and it duplicates the `Linear + gelu` FFN stage already implemented inline in `AlbertLayer.ffChunk`. It is dead code that has diverged from the live path. — None today (unused), but it stays untested and the duplicated logic can drift from `AlbertLayer`; if ever wired in it may mismatch. — Remove the class, or route `AlbertLayer.ffChunk` through it so only one implementation exists.
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertOutput.swift:3-26 — `AlbertOutput` is unreferenced dead code and is byte-identical to `AlbertSelfOutput` (same `dense` + `layerNorm(output + inputTensor)`), i.e. a copy-paste duplicate that has already become two copies to maintain. Its `dense = Linear(config.intermediateSize, config.hiddenSize)` is randomly initialized (no weights loaded), so if anything ever calls it it emits garbage. — None today (unused), but risks confusion/divergence and random-weight output if used. — Delete `AlbertOutput` (and the twin `AlbertSelfOutput`), keeping a single shared implementation.
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfAttention.swift:25-49 — Model weights are force-unwrapped (`weights["...query.weight"]!`, `.key.bias!`, `.dense.bias!`, LayerNorm weights/biases) in the init, while `value.bias` is deliberately passed as a non-forced optional — evidence that key absence is a real model variant. Any of the force-unwrapped keys missing from a loaded safetensors (version/model mismatch) crashes the process with no diagnostic, and the inconsistency is unguarded. — A model archive missing one of the forced keys aborts the app at load instead of surfacing a loadable error. — Replace the forced lookups with a validated `guard let`/dictionary lookup that raises a descriptive error for every required key (mirroring how `value.bias` is tolerated).

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
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertIntermediate.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertLayer.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertLayerGroup.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertModelArgs.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertOutput.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfAttention.swift — findings: 1

# Pi batch review — batch-37

## Findings

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV3.swift:436 — `public var kvHeads: [Int] = []` is declared empty and never assigned; DeepseekV3Model relies on the default `KVCacheDimensionProvider.newCache` extension (MLXLMCommon/LanguageModel.swift:219) where `numLayers = kvHeads.count`, so generation (`Evaluate.init` calls `model.newCache`) yields an empty `[KVCache]`. During decoding `callAsFunction` indexes `cache?[i]` for layer 0 of a non-nil empty array → "index out of range" crash (or, if treated as nil, O(seq²) full recompute with no KV caching), breaking DeepSeek-V3 generation entirely. Every other model in the repo assigns `kvHeads` (e.g. `Array(repeating: config.kvHeads, count: config.hiddenLayers)`). — Fix: assign `self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)` in init.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Cohere.swift:224 — `CohereConfiguration.init(from:)` declares `ropeTraditional` in `CodingKeys` but never decodes it; the stored property keeps its default `true` regardless of the model's `rope_traditional` value in config. `CohereAttention` initializes RoPE with `args.ropeTraditional`, so a config that sets `rope_traditional=false` is silently ignored and attention positions are encoded with the wrong convention → garbled outputs with no error. — Fix: add `self.ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true` in `init(from:)`.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV3.swift:530 — `sanitize` unconditionally drops every weight key under `model.layers.61.*` (`!key.starts(with: "model.layers.61")`). This is a hardcoded, model-specific quirk (DeepSeek V3 uses 61 layers 0…60, so it's a no-op there); for any larger variant the weights of layer 61 are silently discarded, corrupting that layer's output with no guard tied to the actual config. — Fix: gate the drop on `args.numHiddenLayers > 61` (or remove the hardcoded layer filter).
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Bitnet.swift:96 — the custom `bitlinear_matmul` Metal kernel assumes `in_features` is a multiple of `BLOCK*M` (128) and `out_features` a multiple of 4 with no runtime validation. The grid only spans `floor(out_features/4)` row-groups and the column loop covers only the first `ceil(in_features/128)*128` columns, so for any config where `hiddenSize`/`intermediateSize` is not divisible by 128 (or `out_features` not by 4) trailing columns/features are silently never computed, yielding wrong logits with no error (crashes/UB only if the divisor leaves a partial output row). — Fix: `precondition(inFeatures % 128 == 0 && outFeatures % 4 == 0)` in `BitLinear.init` (or handle remainder in the kernel).

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Lora+Data.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/LoraTrain.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/AfMoE.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Apertus.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/BaichuanM1.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/BailingMoe.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Bitnet.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Cohere.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV3.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Ernie4_5.swift — clean

# Pi sweep batch-39

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift:408 — the `restart: while !messages.isEmpty { ... continue restart }` tool-call loop has no iteration cap. Whenever `toolDispatch` is set, every generation pass that emits tool calls appends `.tool(...)` results and `continue restart`, re-preparing input and generating again. If the model keeps producing tool calls (e.g. an agentic tool returning content that drives further calls, or a tool that always triggers another), the loop runs without bound, repeatedly invoking the side-effecting `toolDispatch` closure and regenerating tokens. — unbounded tool-dispatch loop = runaway execution / resource exhaustion (unbounded number of tool invocations). — Cap the number of restart passes (e.g. a `var toolIterations = 0` guard that breaks out and finishes after, say, N tool rounds) and stop dispatching when the cap is exceeded.

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift:451-455 — in the tool loop the assistant's tool-call message is never appended to `messages`; only the `.tool(toolResult)` results are added before `continue restart`. The next pass re-preparses a chat that is just `[.system?, user, .tool, .tool, ...]` with no assistant `tool_calls` entry preceding the tool results. Tokenizer chat templates that expect an assistant tool_calls message before each tool-role block (the common tool-calling format) will render a malformed prompt on the restart pass, producing incoherent or mis-parsed subsequent generations. — malformed tool sequences across multi-step tool use, unless the KV cache alone happens to mask it. — Before `continue restart`, append a synthesized assistant message capturing the emitted tool calls (role `.assistant` with the tool-call content) along with the `.tool` results.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/BaseConfiguration.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Chat.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Downloader.swift — clean

# Pi review — batch-40 (mlx-swift-lm MLXLMCommon)

## Summary
Reviewed six files in `MLXLMCommon`: generation/sampling (`Evaluate.swift`), JSON helpers
(`JSONDecodingTypes.swift`, `GenerationConfigFile.swift`, `JSONDecoder+JSON5.swift`,
`Encodable+toolResult.swift`) and Metal interpolation kernels (`InterpolationUtils.swift`).
One defensible defect found in the interpolation threadgroup sizing: the final
`max(8, …)` clamp can push the threadgroup past the 1024-thread limit after the
size-reduction loop, causing a Metal dispatch failure at certain output dimensions.

## Findings
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/InterpolationUtils.swift:342 — In `getOptimalThreadgroup`, the `while width * height > maxThreadsPerGroup` loop reduces the threadgroup until it fits under 1024 threads, but the subsequent `width = max(8, width); height = max(8, height)` can *raise* a dimension back over the cap (e.g. outW small / outH large yields width=4, height=256 → `max(8, …)` → 8×256 = 2048 > 1024). — Passing an oversized threadgroup to `MLXFast.metalKernel` dispatch aborts / throws at launch, so `bicubicInterpolate`/`nearestInterpolate` crash for those output shapes (e.g. a tall-narrow or 1-wide resize). — Re-verify/shorten after the `max(8, …)` clamp: clamp 8 to `min(8, maxWidth)`/`min(8, maxHeight)` or re-run the `while width*height > maxThreadsPerGroup` reduction after the floor clamp so the final group always stays ≤ maxThreadsPerGroup.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Extensions/Encodable+toolResult.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Extensions/JSONDecoder+JSON5.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/GenerationConfigFile.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/InterpolationUtils.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/JSONDecodingTypes.swift — clean

# Pi review — batch-41 (KokoroSwift decoder/building-blocks)

## Summary
Reviewed 20 files of the kokoro-ios MLX inference port (Albert attention block, AdaIN building blocks, LSTM, decoder/generator, STFT, sine source, duration encoder). The port is faithful to the reference and shape-consistent given the shipped `config.json` (batch is fixed to 1 throughout the TTS pipeline, so several latent single-batch assumptions never mis-fire). Two minor findings: a redundant double transpose on the generator hot path and a hardcoded iSTFT window-overlap constant that is only correct because it happens to equal the shipped `gen_istft_n_fft / gen_istft_hop_size`.

## Findings
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdainResBlk1d.swift:122-124 — two consecutive identical `MLX.swappedAxes(x, 2, 1)` calls before `conv1` cancel to a net no-op, leaving the tensor in the same `[B, L, C]` orientation — every residual block in the Adain conv decoder/generator hot path performs an extra pair of full transposes of a ~[1, 1024, L] tensor per frame per kernel, adding measurable per-frame transpose work — delete one of the two `swappedAxes(x, 2, 1)` statements (keep a single transpose so orientation before `conv1` stays `[B, L, C]`).
- [low] ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/MLXSTFT.swift:136 — `let windowModLen = 20 / 5` hardcodes the iSTFT window-overlap count to 4 instead of deriving it; it only matches the shipped `config.json` (`gen_istft_n_fft: 20, gen_istft_hop_size: 5` → 20/5); if a checkpoint/config with a different `win_len / hop_len` ratio is ever loaded, the overlap-add loop (`.stride(from: i, by: windowModLen)`) silently produces distorted/garbage audio with no error — replace with `let windowModLen = max(1, winLen / hopLen)` (derived from the actual window & hop lengths).

## Coverage
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/AlbertSelfOutput.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Albert/CustomAlbert.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaIN1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaINResBlock1.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdaLayerNorm.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/AdainResBlk1d.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/Conv1dInference.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ConvWeighted.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/InstanceNorm1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/Interpolate.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LSTM.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/LayerNormInference.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/ReflectionPad1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/BuildingBlocks/UpSample1d.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/Decoder.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/Generator.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/MLXSTFT.swift — findings: 1
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/SineGen.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/Decoder/SourceModuleHnNSF.swift — clean
ios/LocalPackages/kokoro-ios/Sources/KokoroSwift/TTSEngine/DurationEncoder.swift — clean

# Pi review — batch 42 (MLX LLM model ports)

## Summary
Reviewed four Swift MLX model ports (Exaone4, FalconH1, GLM4, GLM4MOE) against their
mlx-lm Python references. EXAONE-4's `useRope` handling (`isLocal == nil || isLocal`)
matches the Python `use_rope = is_local is None or is_local`, and GLM4/GLM4MOE
attention/sanitize logic is faithful. FalconH1 has two real defects: it drops the causal
attention mask for multi-token prefill (SDPA with a nil mask does full bidirectional
attention in MLX), and its `kvHeads` array has its dimensions swapped.

## Findings
- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift:213-217 (FalconH1Attention.callAsFunction) + :684-690 (createAttentionMask) — `createAttentionMask` unconditionally returns `nil`, and the comment's claim that "SDPA will handle causal mask internally when nil" is false for MLX: `scaled_dot_product_attention` with an empty mask_mode/no array applies no masking (verified in mlx fast.cpp). During prefill the prompt is processed as a >1-token batch, so queries attend to future keys, leaking future tokens into every token's logits and degrading generated output, unlike the Python port which passes a `"causal"` mask (or a cache `make_mask`) for N>1. — smallest safe fix: in `createAttentionMask` return a real causal mask for N>1 (e.g. `createCausalAttentionMask`/the shared `createAttentionMask`-style mask or an explicit boolean causal array honoring the cache offset), not nil.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift:719 — `self.kvHeads = (0 ..< args.numKeyValueHeads).map { _ in args.numHiddenLayers }` swaps the two dimensions; for `KVCacheDimensionProvider` the array length must be the layer count (each element is the kv-head count for that layer), i.e. `(0..<numHiddenLayers).map { _ in numKeyValueHeads }`. As written it reports `numKeyValueHeads` layers each with `numHiddenLayers` kv-heads, poisoning any layer-count/memory estimation that reads `kvHeads.count` (the automatic `newCache` path in LanguageModel.swift:222 uses `kvHeads.count` as the layer count). — fix: swap the bounds and value to `(0 ..< args.numHiddenLayers).map { _ in args.numKeyValueHeads }`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Exaone4.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/FalconH1.swift — findings: 2
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOE.swift — clean

# Batch 43 findings

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift:276 — The MLA attention output dimension does not match the module applied to it. `queries` is built as `concatenated([qNope, qPe], axis: -1)` where `qNope` (after `embedQ`) has last dim `kvLoraRank` and `qPe` has last dim `qkRopeHeadDim`, so the combined SDPA output has last dim `kvLoraRank + qkRopeHeadDim`. That output is then passed to `callMultiLinear(unembedOut, output)` (line 288), but `unembedOut = MultiLinear(inputDims: kvLoraRank, outputDims: vHeadDim, ...)`, so its packed weight is `[numHeads, vHeadDim, kvLoraRank]` and the matmul requires the input's last dim to equal `kvLoraRank`. Whenever `qkRopeHeadDim > 0` (true for all real glm4-moe-lite configs, e.g. qk_rope_head_dim=64) the attention crashes / is dimensionally inconsistent; it also diverges from the python reference (mlx_lm/models/glm4_moe_lite.py) which runs the rope part through a separate `pe_scores` mask and only feeds a `kvLoraRank`-shaped tensor into `unembed_out` (decode path) or computes keys via `embed_q(kv_latent, transpose=False)`/`unembed_out(kv_latent)` (prefill path). Consequence: the model cannot run (or produces wrong/truncated MLA output) — no valid dimension is produced at this point. Fix: port the reference's two-branch design — for decode route `q_nope = embed_q(q_nope)`, `k = v = kv_latent`, apply `unembed_out` to the `kvLoraRank`-wide output; for prefill route compute `k = embed_q(kv_latent, transpose: true)` and `v = unembed_out(kv_latent)` and run rope attention separately via the pe_scores mask, instead of concatenating q_pe into the full query fed to SDPA and then expecting `unembedOut` to accept `kvLoraRank + qkRopeHeadDim`.
- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift:438 — `convertMoePackedTensors` reshapes to a single collapsed dimension using `prefixShape.count` (the number of leading dims) instead of the actual leading dimensions. The python reference does `out.reshape(*blocks.shape[:-2], -1)`; the Swift `out.reshaped(prefixShape.count, G * B * 2)` produces shape `[len(prefix), G*B*2]` and drops all prefix dims into one bucket, which either throws (total element count no longer matches) or, when the element count coincidentally matches, produces a corrupted MoE weight layout for the 4-bit packed GPT-OSS checkpoints this sanitize branch is designed for. Consequence: loading packed GPT-OSS checkpoints yields wrong-shaped/corrupt `gate_up_proj` weights (or a runtime reshape failure) every time the `gate_up_proj_scales` packing path executes. Fix: `out = out.reshaped(prefixShape + [G * B * 2])`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma2.swift — clean

# Pi sweep — batch-44

## Findings

No defensible defects found in this batch. Cross-file verification (LoRAContainer.swift, LoRA+Layers.swift, DoRA+Layers.swift) confirmed inputs are validated at their sources and invariants hold; the only non-deterministic element (`loraDefaultKeys` returning a `Set` as `Array`) is consumed via `keys.contains`, so ordering has no concrete negative consequence and does not warrant a finding.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/LoRA/LoRAModel.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapter.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapterFactory.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Adapters/ModelAdapterTypeRegistry.swift — clean
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift — clean

# Pi review — batch-45

## Summary
Reviewed `swift/Sources/MCPZimKit/IntentRouter.swift` (fast-path intent router + reply synthesis). The router is text-in/text-out and heavily exercised by tests and real-device captures; no injection, no hardcoded secrets, no obvious nil-paths in the common routing flow. Two low-severity robustness/correctness issues found around the possessive-facet stripping helpers, which mix character indices taken from a lowercased copy back into the original string.

## Findings
- [low] swift/Sources/MCPZimKit/IntentRouter.swift:856-863 — `stripPossessiveFacet(from:)` computes `lower.range(of: "'s ", options: .backwards)` on `subject.lowercased()` and then maps that range's offset back onto the original `subject` via `subject.index(subject.startIndex, offsetBy: lower.distance(...))`. This index arithmetic is only valid when the case fold preserves length; for input containing a character whose `lowercased()` form changes length (e.g. Turkish `İ` U+0130 → `i` + combining dot, 2 scalars) the offset read from `lower` can exceed the length of `subject`, making `subject.index(offsetBy:)` trap (crash), or silently mis-slice the title. — A user query such as "Tell me about İzmir's early life" (or any typed/ASR subject containing `İ` before an `'s` facet) can crash the router or produce a shifted title. — Instead of mapping indices across copies, slice on the same string the range came from: perform the match on `subject` itself (compare against a lowercased copy only for membership testing), or guard the offset with `if distance <= subject.count` before indexing; simplest robust fix is to match `"'s "` directly on the original `subject`.

- [low] swift/Sources/MCPZimKit/IntentRouter.swift:885-896 — `stripPossessiveFacetAggressive(from:)` builds its match on `subject.lowercased()` and returns the capture `m[0]` verbatim, so the emitted entity name is always all-lowercase ("Apple" → "apple", "İzmir" → "i̇zmir"). The conservative sibling `stripPossessiveFacet` preserves capitalization because it slices the original `subject`, but this aggressive retry discards it. — On the article-miss retry path the host re-queries with a lowercased title; for a case-sensitive title index or acronyms ("MIT"/"AT&T") the retry re-misses the very article the user meant. — Return the corresponding slice of the original `subject` (e.g. `String(subject.prefix(m[0].count))`) instead of the lowercased capture, keeping the original casing.

## Coverage
swift/Sources/MCPZimKit/IntentRouter.swift — findings: 2

# Pi sweep — batch-46

## Findings
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:842-907 — `SpeculativeTokenIterator.speculateRound` runs a full draft+verify pass and appends the final/bonus token to `pendingTokens` even when the remaining `maxTokens` budget (`remaining = maxTokens - tokenCount`) is smaller than the round's `numDraft + 1` tokens. When `tokenCount` reaches `maxTokens` mid-buffer, `next()` (line 938) drops the surplus token(s), but the draft tokens were already generated and the main model already ran a full verification forward pass including the discards bonus position. On short generations (e.g. `maxTokens == tokenCount + 1`, or any final round where `remaining < numDraft + 1`), this wastes a complete model forward pass that produces no emitted token. — Concrete consequence: every generation's final round performs an extra verify forward pass (and computes a bonus token) that is silently thrown away; with default `numDraftTokens=2`, single-token tails trigger a full round of wasted GPU compute. — Smallest safe fix: clamp the verification work to the actual remaining budget — e.g. set `numDraft = min(numDraftTokens, remaining)` and cap the number of verify positions (and bonus token) to `remaining` so no token beyond the budget is computed, or early-return when `remaining <= 1` on the tail.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift — findings: 1

# Pi sweep — batch-47

Reviewed `swift/Sources/MCPZimKit/ArticleHeuristics.swift` (full file, 1416 lines)
plus its cross-file dependencies (`ArticleSection`/`ArticleSections.parse`, the
`ZimService` protocol signatures, `RegexCache`, `HashingEmbedder`) to confirm
the invariants this file relies on (lead carries empty title, `bytes` is
computed, `articleSections`/`articleByTitle` tuple shapes match) and the
reachability of each helper from its callers.

The file is a pure-heuristics utility: section selection, prose cleaning,
sentence splitting, and RAG ranking. Reviewed for correctness, performance
(regex caching used throughout; embedding calls are cheap n-gram hashing),
and security (no user data sinks, no file/network I/O, no secret handling).
No defensible defects found — the sentence splitter's abbreviation/decimal
guards, the title-repetition stripper, the bounded regex/loop bounds, and the
keyword/stem handling are all consistent and bounded.

## Findings

(no findings)

## Coverage
swift/Sources/MCPZimKit/ArticleHeuristics.swift — clean

# Pi sweep review — batch-48

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:1218 — `LanguageModel.ropeDeltas` is mutable per-instance state mutated inside `callAsFunction` (`ropeDeltas = nil` at 1249, `ropeDeltas = deltas` at 1268, read at 1282/1288) on a class that is otherwise `Sendable` and shared via the `Qwen3VL` Module. Two concurrent `prepare`/decode streams on the same model instance race on this field, producing wrong mRoPE positional ids (garbage logits) or a crash. Fix: make `ropeDeltas` part of per-call state (derive it from `cache` offsets / recompute in prefill) instead of shared mutable ivar, or serialize inference on the instance.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:1357 — `getRopeIndex` is documented "assume batch=1" but still loops `0 ..< batchSize`: if batch>1 with multimodal input, `positionIds = newPositions` (line ~1450) overwrites the whole (3,B,S) array each iteration, so earlier batch rows get clobbered and produce incorrect position encodings (no guard rejects batch>1). Fix: either `assert(batchSize == 1)` for the multimodal path or accumulate per-row into a `[3,B,S]` array instead of reassigning the full tensor.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift:1187 — `applyDeepstack` calls `maskIndices(visualMask)` which does `mask.asType(.bool).asArray(Bool.self)` (a GPU→CPU host sync) on every model forward that carries a visual mask, i.e. once per decode token per deepstack layer for the Qwen3-VL-235B / deepstack variant. Unavoidable prefill cost is fine, but per-token sync adds latency. Fix: compute the visual token indices once in `prepare` (prefill) and thread them through, instead of re-deriving from the GPU array each step.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift — findings: 3

# Pi sweep — batch-49

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:1526 — `QuantizedKVCache.metaState` getter serializes `[step, offset, groupSize, bits]` (4 values) but the setter only restores `self.offset = Int(newValue[1]) ?? 0`, silently discarding `groupSize` and `bits`. When a prompt cache is saved with non-default quantization (e.g. `bits=4`/`groupSize=128`, as produced by `toQuantized(groupSize:bits:)` or `maybeQuantizeKVCache(kvBits:)`) and later loaded via `loadPromptCache` → `restoreCacheFromMetaState` → `QuantizedKVCache()` (defaults `groupSize=64, bits=8`), the restored cache dequantizes with the wrong group size/bit width, yielding garbage key/value states. Consequence: saved quantized prompt caches silently produce incorrect generation output after load whenever the saved params differ from the defaults. Smallest safe fix: have the setter parse and assign `groupSize = Int(newValue[2]) ?? 64` and `bits = Int(newValue[3]) ?? 8` before/with offset (and guard them against configured params).

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:1199 — `CacheList.trim(_:)` overwrites `result` inside the loop (`result = cache.trim(n)`), so it returns only the *last* child's trimmed count instead of the value across all children (Python's CacheList returns the min trimmed count). Callers using the return value to decide how many tokens were trimmed (e.g. to coordinate state) get an incorrect, arbitrary-layer count. Smallest safe fix: return `caches.map { $0.trim(n) }.min() ?? 0` (track the minimum), or accumulate.

- [nit] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:1327 — in `QuantizedKVCache.updateQuantized`, the values-cache is initialized with `dtype: keys.dtype` instead of `values.dtype` (`initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)`). If keys and values ever have differing dtypes the values cache would be created at the wrong precision and later assignment/concatenation would mismatch. Smallest safe fix: pass `dtype: values.dtype`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift — findings: 3

# Pi sweep — batch-50

## Findings
None. The single file in this batch is a high-quality behavioral test suite for the fast-path intent router. Every test exercises the real static functions in `IntentRouter`/`MCPToolAdapter`/`ArticleHeuristics` directly (no mocks, no patching), each assertion is falsifiable against concrete documented production captures, and there is no shared mutable state, sleeps, network dependence, or order-dependence. Spot-traced representative routes through `IntentRouter.classify` (category-in-place, nearest/nearby GPS binding, directions precedence, compare suffix expansion, wikipedia-source directive, correction re-route) and confirmed the test expectations match the actual patterns and singularize/expandSharedSuffix/didYouMeanTitles implementations. No tautological or never-failing assertions found.

## Coverage
swift/Tests/MCPZimKitTests/IntentRouterTests.swift — clean

# Pi sweep — batch-51

## Findings

- [medium] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:500 — `mergeInputIdsWithImageFeatures` / `nonZero` (lines 500-570) force several GPU→CPU synchronizations for every prompt: `specialMask.sum().item(Int.self)`, `maskExpanded.sum().item(Int.self)`, and `nonZero` doing `mask.asArray(Bool.self)` (line 561) plus materializing the full index `[UInt32]` in Swift. Each `.item()`/`.asArray` blocks until the GPU pipeline drains and copies the whole mask tensor back to the host, stalling MLX graph execution on the (typically large, sequence×hidden) mask. — This serializes GPU and CPU per forward pass, adding a multi-ms sync stall at each prompt/vision merge and duplicating the index data in Swift. — Replace the host synchronizations with MLX-side ops: compute the mismatch guard and the scatter index entirely with MLXArray (e.g. `argwhere`/`nonzero` on the mask, `.sum()` kept as a tensor or checked once via a single `.item()`), and write features with `flattenedEmbeds`/`flattenedFeatures` advanced indexing without round-tripping the whole mask through `asArray`. Keep at most one sync per merge instead of three plus the full-array copy.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift — findings: 1

# Batch 52 review

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift:822 — `GlobalPool2D` declares `let proj: MLXArray` and initializes it to `MLXArray.zeros([inDim, outDim])` instead of a `@ModuleInfo` parameter, so it is never populated from the checkpoint weights and `head(x)` always returns zeros. Today this is masked because `FastVLM.getInputEmbeddings` discards the first return value (`clsOut`), so the only symptom is a wasted zero `matmul(B, projectionDim)` per forward pass; but the classification/global-pool head can never produce a correct value and would silently emit all-zeros if `clsOut` were ever used. — Smallest safe fix: don't create the dead `head` at all (single-image VLM never uses `clsOut`), or make `proj` a loadable `@ModuleInfo(key:) var proj: Linear` initialized from `Linear(inDim, outDim)` and load its weights; likewise drop the dead zero-multiply output.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/FastVLM.swift — findings: 1

# Pi sweep — batch-54 (Mistral3.swift)

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift:681-683 — `inputIds[0].asArray(Int32.self)` force-materializes the entire token sequence to the host (synchronous GPU→CPU copy) on every image-bearing prompt just to locate image-token positions, and iterates the host array in Swift. Consequence: a per-prompt compute pipeline stall (blocking the Metal/MLX device) that grows with context length, and a redundant transfer of data already present on device. Smallest safe fix: find image positions on-device, e.g. `MLX.argwhere(inputIds[0] == Int32(imageTokenIndex))` / `MLX.nonzero`, and only read back the (small) index array; or compute positions in the processor where the token layout is already known.
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift:661,683,1056-1088 — the model merge counts prompt tokens equal to `config.imageTokenIndex` (`imageTokenIndex` field, default resolved as `image_token_index ?? image_token_id ?? 10`), while the processor inserts `imageTokenId` resolved from the tokenizer (`convertTokenToId(imageToken) ?? 10`). The two normally coincide, but if a saved config sets `image_token_index` explicitly to a value different from the tokenizer's image-token id (or the tokenizer returns a non-10 id with no `image_token_id` set and no numeric `image_token_index`), the prompt will contain `imageTokenId` tokens while the merge scans for `imageTokenIndex`, tripping the `fatalError("Image token count ... does not match image patches")` guard at runtime. Consequence: hard crash on a valid model/processor config rather than a graceful error. Smallest safe fix: make both sides use the exact same id — have the processor expose the resolved image-token id and have the merge key on that same value (validate/image_token_index == resolved id at init), instead of relying on the default-indirection chain.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Mistral3.swift — findings: 2

# Pi sweep — batch-55

Findings for ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift

- [high] ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift:303 — Dynamic NTK scaling thresholds on `maxPositionEmbeddings` (config default 0) instead of the decoded `maxTrainedPositions` (default 2048), which is documented as "the reference length (L_train) for Dynamic NTK scaling calculations" but never read anywhere. In `Attention.init` the dynamic path passes `maxPositionEmbeddings: config.maxPositionEmbeddings` (line 409); with the default 0, `seqLen > 0` is always true, so `factorAdjustment = Float(seqLen)/0 - 1 = +inf`, and `base *= pow(1+inf, dims/(dims-2))` yields `base = inf`. Whenever the NTK scaling path is configured (`rotaryScalingFactor != nil`) the model always applies infinite-frequency RoPE even on short inputs, producing NaN/Inf embeddings instead of the intended length-conditional scaling (and `maxTrainedPositions` is silently dead config). Fix: pass `maxPositionEmbeddings: config.maxTrainedPositions` into `DynamicNTKScalingRoPE` from `Attention.init` (and/or default `maxPositionEmbeddings` to the trained length), so the scaling activates only past the trained context window.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXEmbedders/Models/NomicBert.swift — findings: 1

# Pi sweep — batch-56

## Findings

- [low] swift/Sources/MCPZimKit/ReferenceResolver.swift:369 — In `resolveListSelection`, `lower.hasPrefix(word)` makes ANY turn whose first token is an ordinal word into a list selection whenever `focus.lastList` is non-empty. Because the `ordinals` dict maps "one","first","second","two","three","four","five" to indices, turns like "one more thing about X", "first of all, tell me about Y", or "two days ago ..." will bind to `lastList[idx]` and be rewritten as "tell me about <item>" instead of routing as the fresh query the user actually typed. This hijacks a stateless-patterns routing decision (wrong article fetch). The prefix test was meant only to catch bare "second"/"first" picks; it should require the word to be the *only* content token (e.g. `words.count == 1` or `words.count == 2 && words.last == "one"`) rather than a raw prefix — drop the `lower.hasPrefix(word)` clause (or gate it on `words.count <= 2`).
- [low] swift/Sources/MCPZimKit/ReferenceResolver.swift:349-353 — "the other one" for a 2-item list falls back to `list[1]` (hard-coded index 1) whenever `focus.primaryEntity?.matchKey` is nil. If the two items are [A, B] and either primaryKey doesn't match either item, or the relevant "other" is actually item 0, this picks an arbitrary (and possibly the wrong) entity. Consequence: `rewrite` substitutes a wrong article name into the follow-up. When primaryKey is nil, the two candidate items are symmetric, so there is no principled choice; return `.ambiguous` (or fall through) instead of silently picking `list[1]`.

## Coverage
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — clean
swift/Sources/MCPZimKit/ReferenceResolver.swift — findings: 2

# Batch 57 review

## Findings
- [low] ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift:248 — `embeddingContainer()` is the only container method in `IntegrationTestModels` that is not memoized with a cached `Task`; every call re-runs `EmbedderModelFactory.shared.loadContainer` and re-downloads/re-loads the embedding model. In a test run where more than one embedder test (or a repeated call) needs it, this duplicates a multi-hundred-MB model download/load, while all seven LLM/VLM containers are cached. — redundant network download and model load per call, slowing the suite and hitting the downloader repeatedly — cache the task (e.g. add a private `embeddingTask: Task<EmbeddingModelContainer, Error>?` and reuse it, mirroring the LLM container methods).

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift — findings: 1
ios/LocalPackages/mlx-swift-lm/Tests/MLXLMTests/ToolTests.swift — clean

# Batch 58 — ios/MCPZimChat/Sharing/ZimCatalog.swift, ios/MCPZimChat/Voice/TTSService.swift

- [medium] ios/MCPZimChat/Voice/TTSService.swift:476,485,544,685 — `stop()` during a pending `scheduleBuffer` completion continuation hangs the async function. `KokoroTTSService.speak` awaits the tail buffer's completion handler (line 476) and `awaitPlayback` awaits a marker buffer's completion (line 544), but `stop()` calls `AVAudioPlayerNode.stop()` (line 685), which does NOT fire the completion callbacks of queued, not-yet-played buffers. If `stop()` is invoked while the task is awaiting either continuation, the `CheckedContinuation` is never resumed: `speak` never returns and its `defer { isSpeaking = false }` (line 444) never runs, so the service is left stuck `isSpeaking == true` and the voice loop that blocks on `await awaitPlayback()` hangs permanently (leaked task + next synthesised sentence never scheduled). Fix: complete the continuations on stop — e.g. track the pending continuation in a field that `stop()` resumes (and nil-ifies) before/instead of calling `player.stop()`, or gate the await so stop also resumes it; alternatively use a cancellation flag checked alongside the resume.

- [low] ios/MCPZimChat/Voice/TTSService.swift:320,450,495,507,682-686 — data race on `stopFlag`/`isSpeaking` in `KokoroTTSService` (`@unchecked Sendable`). The synthesis tasks read/write `stopFlag` (read at 450, 507; set true only in `stop()` line 684), while `stop()` is called from the caller thread (UI) with no lock/serialization; `isSpeaking` is likewise written from `stop()` (686) and from the synthesis loop (444, 548). Concurrent unsynchronized Bool access is a data race that Swift Concurrency does not validate because the class is `@unchecked Sendable`; can manifest as a stale `stopFlag` read (one more chunk synthesized after stop) or torn state. Fix: confine mutations to a serial queue/actor or use a `ManagedCriticalState`/lock around `stopFlag` and `isSpeaking`.

## Coverage
ios/MCPZimChat/Sharing/ZimCatalog.swift — clean
ios/MCPZimChat/Voice/TTSService.swift — findings: 2

# Pi sweep — batch 59

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift:241-247 — KV-shared "consumer" layers always construct `kProj`, `vProj`, `kNorm`, `vNorm` (and a K-dedicated RoPE), but in the default config 20 of 35 layers are consumers that take K/V from a donor via `sharedKV` and return at line 372 before ever touching those modules (`vProj`/`vNorm` are only reachable in the non-shared donor path). The reference `gemma4_text.py` allocates K/V projections only when `has_kv` is true for exactly this reason. Concrete consequence: ~16–32 MB of dead parameter/allocation weight is held on a device that the surrounding code explicitly worries about for jetsam (the gemma4 shared-KV hand-off was designed to cut memory); if a consumer ever fell through the shared path it would read uninitialized random Linear weights with no cache. Smallest fix: skip allocating `kProj`/`vProj`/`kNorm`/`vNorm` when the layer is a KV-shared consumer (mirror the reference's `has_kv` gating), keeping `qProj`/`oProj`/`qNorm`/`rope`.

- [low] swift/Sources/MCPZimKit/ConversationThreads.swift:715-725 — `WikiLinks.decodeAndStrip` decodes only a fixed list of named entities plus `&#39;`/`&nbsp;`; general numeric character references (`&#NNN;`/`&#xHH;`) that appear in Wikipedia anchor text (e.g. `&#8217;` curly apostrophe, `&#8211;` en-dash) pass through literally. Concrete consequence: user-facing offers/chips and `offer(_:)` captions can display raw `&#8217;`-style codes instead of the intended characters. Smallest fix: after the named-entity pass, regex-replace `&#(\d+);` and `&#x([0-9a-fA-F]+);` with their Unicode scalars in `decodeAndStrip`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift — findings: 1
swift/Sources/MCPZimKit/ConversationThreads.swift — findings: 1

# Batch 60 review

## Findings

- [low] ios/MCPZimChat/Views/LibraryView.swift:219-227 — GitHub PAT (account credential) is stored and read in plaintext via `UserDefaults` (`DebugReportConfig.githubToken` writes `UserDefaults.standard` in DebugReport.swift:53-62), surfaced here through a `SecureField`/`Binding`. UserDefaults values are stored unencrypted in the app sandbox plist (readable from an unencrypted device backup / by any process with sandbox access), so a user's GitHub PAT with `gist` scope can leak at rest. — Concrete consequence: credential at rest is recoverable from backups/filesystem rather than protected by the Keychain; a leaked PAT grants gist-write (and any scopes the token carries) on the developer's GitHub account. — Smallest safe fix: store the token in the Keychain (e.g. `SecItemAdd`/generic password) with the contents exposed only in-memory, keeping the same `githubToken` accessor API used here and in DebugReport.swift.

## Coverage
ios/LocalPackages/LocalSwarm/Sources/LocalSwarmEngine/SwarmSession.swift — clean
ios/MCPZimChat/Views/LibraryView.swift — findings: 1

# Pi review — batch-61

## Summary
Reviewed 2 files: a faithful MLX port of the PaliGemma VLM (model + processor + config) and a Swift regression-test suite for discuss-mode passage retrieval. No defensible defects found. The Paligemma `Encoder` returning `h = x[0]` was examined (it collapses the batch dim, which is consistent with B=1 inference and the `.newAxis` re-add in `inputEmbeddings`); the `outputHiddenStates` append-and-drop issue was already addressed in code with a comment. The test suite has concrete, fail-able assertions with no flaky/mock-proving-nothing patterns.

## Findings
None.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift — clean
swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean

# Batch 62 review — GraniteMoeHybrid.swift

## Findings

- [low] ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift:20 — `createSSMMask(cache:)` is a hard-coded stub that always returns `nil` and never inspects the `MambaCache` (it is not even given `h`, so it cannot compute the sequence length `N`). Every sibling SSM port (`NemotronH.swift:674`, `FalconH1.swift:606`) builds the real mask via `mambaCache.makeMask(N:)`; here the returned mask is always dropped. In `GraniteMoeHybridMamba2Mixer` the nil mask then skips both the conv-input zeroing and the mask argument to `ssmAttn`, so any batched/chunked prefill where a prior token cache exists and `N > 1` will compute the SSM (state/conv) contribution over invalid padding tokens and produce wrong hidden states — the silent-loss failure path. Standard single-token autoregressive generation (empty start cache, `N == 1`) is unaffected, hence low severity — implement the mask like NemotronH/FalconH1: pass `h` (or read `N` from the cache) and return `mambaCache.makeMask(N:)` when the cache is a `MambaCache`, otherwise `nil`.

## Coverage
ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/GraniteMoeHybrid.swift — findings: 1

