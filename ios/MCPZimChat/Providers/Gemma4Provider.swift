// SPDX-License-Identifier: MIT
//
// Gemma 4 provider over Swift-Gemma4-Core
// (https://github.com/yejingyang8963-byte/Swift-gemma4-core).
//
// The Swift-Gemma4-Core dependency is *optional at compile time*: if you
// haven't added it to the project yet, the file still builds and exposes a
// provider that reports its load state as `.failed("Gemma 4 runtime not
// linked")`. Once you add the package dependency in Xcode, remove the
// `#if canImport(Gemma4SwiftCore)` fallback path and the real implementation
// takes over.

import Foundation
import MCPZimKit
import OSLog

private let log = Logger(subsystem: "org.mcpzim.MCPZimChat", category: "Gemma4")

/// Serialised fraction store for the disk-poll progress sibling.
/// `load()` spawns a Task that writes to this; the provider reads the
/// max between this and the Hub-reported fraction.
actor DiskFraction {
    private var value: Double = 0
    func set(_ f: Double) { value = max(value, f) }
    func maxFraction() -> Double { value }
}

/// Sum of regular-file sizes under `url`, recursive. Returns 0 if the
/// directory doesn't exist yet (download hasn't started laying bytes
/// down). Used as a "real" progress signal to sidestep
/// `swift-huggingface`'s coalesced `NSProgress.fractionCompleted`.
func dirSizeBytes(at url: URL) -> Int64 {
    let fm = FileManager.default
    guard fm.fileExists(atPath: url.path) else { return 0 }
    guard let enumerator = fm.enumerator(
        at: url,
        includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
        options: [.skipsHiddenFiles],
        errorHandler: nil
    ) else { return 0 }
    var total: Int64 = 0
    for case let file as URL in enumerator {
        let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        if values?.isRegularFile == true, let size = values?.fileSize {
            total += Int64(size)
        }
    }
    return total
}

#if canImport(MLXLLM) && canImport(MLXLMCommon)
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

public final class Gemma4Provider: ModelProvider, @unchecked Sendable {
    public let id: String
    public let displayName: String
    /// Approximate resident weights in MB. Displayed in the model picker
    /// so the user sees the real footprint before committing to a load.
    /// Generation adds KV cache (~125 MB at 4-bit quant, ~500 MB FP16)
    /// plus MLX's Metal pool (~400 MB).
    public let approximateMemoryMB: Int
    public let supportsToolCalls = true

    /// HF repo id + display metadata passed in at construction time.
    /// Known Gemma 4 E2B variants:
    ///
    /// - `mlx-community/gemma-4-e2b-it-4bit` — multimodal 4-bit, but
    ///   `Gemma4Model.sanitize(weights:)` discards vision/audio/MM
    ///   projector keys at load time so resident is text-only
    ///   (~2.6 GB). Already cached on most dev devices; safe default.
    /// - `mlx-community/Gemma4-E2B-IT-Text-int4` — pure text-only
    ///   `gemma4_text` 4-bit. Smaller (2.46 GB download, ~2.2 GB
    ///   resident) and routes straight to `Gemma4TextModel` without
    ///   the multimodal wrapper. Currently blocked on iOS by an
    ///   HF Xet CDN redirect issue (CFNetwork "retry(N) reason(1)
    ///   error [4:-5]" on `cas-bridge.xethub.hf.co`).
    private let modelConfiguration: ModelConfiguration

    /// Template override. Defaults to `Gemma4Template()` — set via init
    /// to a different template (e.g. `QwenChatMLTemplate()`) for Qwen
    /// or other model families that reuse this provider's MLX loading
    /// + streaming path but emit different prompt / tool-call formats.
    public let template: any ModelTemplate

    /// Per-provider floor on reply tokens. Small models (e.g. Qwen 3
    /// 1.7B) burn their default budget in reasoning before reaching
    /// the tool call; raising the minimum gives them headroom to
    /// finish. Weighs against the DeviceProfile cap by taking max of
    /// the two — see `ChatSession.effectiveMaxReplyTokens`.
    public let replyTokensFloor: Int?

    /// Publisher-recommended sampling recipe, same semantics as
    /// `LlamaCppProvider.samplingProfile` — installed so a cross-runtime
    /// A/B (Bonsai GGUF vs Bonsai MLX) samples identically on both sides.
    public let samplingProfile: GenerationSamplingProfile?

    /// Uniform cross-runtime stats for the last completed generate().
    /// Written at the same point as the `perf complete` log line.
    public private(set) var lastGenerationStats: GenerationStats?

    public init(
        id: String = "gemma4-e2b-it-4bit",
        displayName: String = "Gemma 4 E2B (4-bit · multimodal)",
        huggingFaceRepo: String = "mlx-community/gemma-4-e2b-it-4bit",
        approximateMemoryMB: Int = 2600,
        template: any ModelTemplate = Gemma4Template(),
        replyTokensFloor: Int? = nil,
        samplingProfile: GenerationSamplingProfile? = nil,
        localWeightsDirectory: URL? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.samplingProfile = samplingProfile
        // Honor a local directory override when provided — used for
        // models we've quantized locally via `mlx_lm.convert` that
        // aren't published to HF. The directory variant skips HubClient
        // entirely and loads straight from disk; `huggingFaceRepo` is
        // kept around for display / diagnostics.
        if let dir = localWeightsDirectory {
            self.modelConfiguration = ModelConfiguration(directory: dir)
        } else {
            self.modelConfiguration = ModelConfiguration(id: huggingFaceRepo)
        }
        self.approximateMemoryMB = approximateMemoryMB
        self.template = template
        self.replyTokensFloor = replyTokensFloor
        // MLX's Metal buffer pool defaults its cache limit to the system
        // memory limit (~128 GB on an M1 Ultra) — which means it happily
        // hoards tens of GB of intermediate tensors from past generations
        // "just in case". Cap the cache per device tier so 6 GB phones
        // don't jetsam while Macs and 12 GB Pros still get hot-reuse.
        // Budget: weights ~2.6 GB + active KV cache ~1 GB + pool cap.
        MLX.GPU.set(cacheLimit: DeviceProfile.current.mlxCacheLimitMB * 1024 * 1024)
    }
    private var container: ModelContainer?
    private var state: ModelLoadState = .notLoaded
    /// Keyed so `onTermination` can remove a dead subscriber — the old
    /// append-only array retained every continuation for the app lifetime
    /// and `set(_:)` fanned out to all of them forever.
    private var continuations: [UUID: AsyncStream<ModelLoadState>.Continuation] = [:]
    private let queue = DispatchQueue(label: "gemma4.state")

    // In-window debug log sink. Set by ChatSession when it constructs the
    // provider so logs bubble up into the UI debug pane.
    public var debugSink: (@Sendable (String) -> Void)?

    // MARK: - Prompt cache (KV reuse across `generate()` calls).
    //
    // Each tool-call round-trip rebuilds the full system + tools + user
    // + prior-tool-calls prompt, which is strictly append-only. We
    // keep the `[KVCache]` from the previous `generate()` alive so the
    // next call only has to prefill the NEW tokens (usually just the
    // tool response). Without caching a tool-heavy turn spends 3–4 s
    // on duplicate prefills.
    //
    // `cachedTokens` mirrors what the cache contains (prompt tokens
    // plus the tokens this provider emitted during the last generate).
    // We compare it to the newly-tokenised prompt on every call; the
    // longest common PREFIX determines how much we can keep.
    private var promptKVCache: [KVCache]?
    private var cachedTokens: [Int32] = []
    private var generatedTokensThisTurn: [Int32] = []

    /// True if the next `generate()` has a shot at an LCP cache hit
    /// (i.e. we've either streamed at least one turn or run primeCache).
    /// ChatSession uses this to debounce background prewarm requests:
    /// if the cache is already warm, don't re-prime.
    public var hasPromptKVCache: Bool {
        promptKVCache != nil && !cachedTokens.isEmpty
    }

    /// Drop the prompt cache — called from `ChatSession.resetConversation`
    /// and also on memory warning / app-backgrounded notifications.
    /// Only clears Swift-level references; the Metal buffer pool drain
    /// (`MLX.GPU.clearCache()`) used to live here but it races with
    /// Kokoro TTS's Metal inference — clearing the pool mid-utterance
    /// left AVAudioEngine in a state where the next `installTapOnBus`
    /// threw an NSException and crashed the app. The pool is drained
    /// naturally at the tail of every `generate()` anyway.
    public func resetPromptCache() {
        promptKVCache = nil
        cachedTokens = []
        generatedTokensThisTurn = []
        debug("prompt cache reset")
    }

    /// Populate `promptKVCache` / `cachedTokens` by running a *prefill-only*
    /// pass over `prompt`. Unlike `generate()`, no token is sampled and
    /// committed, so the cache state lines up 1:1 with `prompt`'s tokens —
    /// making the next real `generate()` turn a perfect prefix match for
    /// everything up through `prompt`. Use this (not `generate(maxTokens:1)`)
    /// for the Setup screen's preamble warmup so iter 0 doesn't diverge
    /// at the sampled-placeholder token.
    public func primeCache(prompt: String) async throws {
        guard let container else { throw ModelError.notLoaded }
        Stream.defaultStream(.gpu).synchronize()
        MLX.GPU.clearCache()
        // Encode OUTSIDE container.perform — reentering the container
        // actor from within `perform` can deadlock on some builds.
        debug("primeCache: tokenising prompt (\(prompt.count) chars)…")
        let tokens = await container.encode(prompt)
        let tokens32 = tokens.map { Int32($0) }
        debug("primeCache: encoded \(tokens32.count) tokens — prefilling…")
        try await container.perform { context in
            // On iPhone: quantize the post-prefill KV cache to 4-bit
            // groupwise. On Qwen, `Qwen3Attention` already goes through
            // `attentionWithCacheUpdate` which routes to the quantized
            // path when the cache is `QuantizedKVCache`. On Gemma 4,
            // our vendored mlx-swift-lm `Gemma4Text` patch handles the
            // same routing. Mac stays FP16 (Mac has the RAM; quant's
            // only win is on memory-constrained devices).
            //
            // Subtlety that bit us in v1 of this fix: `TokenIterator`'s
            // init takes `cache` BY VALUE (Swift array). Its internal
            // `maybeQuantizeKVCache(&self.cache, …)` replaces
            // `self.cache[i]` with a fresh `QuantizedKVCache`, but our
            // outer `kvCache` array still holds the ORIGINAL
            // `KVCacheSimple` element (a class reference that was
            // mutated in-place during prefill — so it's grown to full
            // FP16 state). When the iterator is discarded the quantized
            // version drops with it and we keep the FP16 blob, which
            // is the opposite of what we want. Fix: after `prepare()`
            // has filled `kvCache[*]` via the shared-class mutation,
            // explicitly run `maybeQuantizeKVCache` on our own array.
            let useQuant = DeviceProfile.current.useQuantizedKVCache
            let params = GenerateParameters(
                maxTokens: 0,                 // no generation
                kvBits: useQuant ? 4 : nil,
                kvGroupSize: 64,
                quantizedKVStart: 0,
                temperature: 0.0, topP: 1.0,
                prefillStepSize: 128
            )
            var kvCache = context.model.newCache(parameters: params)
            let input = LMInput(tokens: MLXArray(tokens32))
            // Constructing a TokenIterator runs the full prefill via
            // `model.prepare(...)` + one `step(...)` over the remaining
            // tail — at end of init the KV cache holds exactly
            // `tokens32.count` tokens' worth of state. We never call
            // `.next()`, so nothing beyond the prompt is committed.
            _ = try TokenIterator(
                input: input, model: context.model, cache: kvCache, parameters: params
            )
            if useQuant {
                // Swap each `KVCacheSimple` (full-attention layers) for
                // its `.toQuantized(...)` counterpart. `RotatingKVCache`
                // (sliding layers) isn't supported upstream yet and gets
                // left alone by the library's dispatch. Measures ~4×
                // smaller on iPhone for the prefilled KV.
                MLXLMCommon.maybeQuantizeKVCache(
                    cache: &kvCache,
                    kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0
                )
            }
            self.promptKVCache = kvCache
            self.cachedTokens = tokens32
            self.generatedTokensThisTurn = []
        }
        Stream.defaultStream(.gpu).synchronize()
        MLX.GPU.clearCache()
        debug("primeCache: done (\(self.cachedTokens.count) tokens live)")
    }

    // MARK: - Disk serialisation of the prompt cache.
    //
    // ChatSession's one-time setup screen calls these to skip the
    // multi-second prefill after the first launch. We persist both the
    // [KVCache] (via MLXLMCommon's safetensors writer) and the
    // cachedTokens mirror (base64-packed Int32 in the metadata map)
    // so the LCP match on the next turn still lines up.

    /// Serialise the current prompt cache to `url`. Caller is
    /// responsible for choosing a stable key — typically a hash of the
    /// static preamble + enabled ZIMs + modelId.
    public func savePromptCache(to url: URL, keyHint: String) async throws {
        guard let cache = promptKVCache, !cache.isEmpty else {
            throw NSError(
                domain: "Gemma4Provider", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No cache to save"]
            )
        }
        let tokenData = cachedTokens.withUnsafeBufferPointer { buf in
            Data(buffer: buf)
        }
        let metadata: [String: String] = [
            "keyHint": keyHint,
            "tokenCount": String(cachedTokens.count),
            "tokensBase64": tokenData.base64EncodedString(),
        ]
        // Ensure no Metal work is pending before we read the KV
        // arrays' storage — otherwise the writer races with the GPU.
        Stream.defaultStream(.gpu).synchronize()
        try MLXLMCommon.savePromptCache(url: url, cache: cache, metadata: metadata)
        debug("saved prompt cache (\(cachedTokens.count) tokens) → \(url.lastPathComponent)")
    }

    /// Restore a previously-saved cache. Populates `promptKVCache` and
    /// `cachedTokens` so the next `generate()` does an LCP hit on
    /// whatever prefix was cached.
    public func loadPromptCache(from url: URL) async throws {
        let (cache, metadata) = try MLXLMCommon.loadPromptCache(url: url)
        guard !cache.isEmpty else {
            throw NSError(
                domain: "Gemma4Provider", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Loaded empty cache"]
            )
        }
        guard let b64 = metadata["tokensBase64"],
              let blob = Data(base64Encoded: b64),
              blob.count % MemoryLayout<Int32>.size == 0
        else {
            throw NSError(
                domain: "Gemma4Provider", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Cache metadata missing token mirror"]
            )
        }
        let tokens: [Int32] = blob.withUnsafeBytes { raw in
            let buf = raw.bindMemory(to: Int32.self)
            return Array(buf)
        }
        self.promptKVCache = cache
        self.cachedTokens = tokens
        self.generatedTokensThisTurn = []
        debug("loaded prompt cache (\(tokens.count) tokens) ← \(url.lastPathComponent)")
    }


    private func debug(_ s: String) {
        debugSink?(s)
        // Tag with the model family — this class hosts both Gemma 4
        // and Qwen (different templates) so a hard-coded "[Gemma4]"
        // is actively misleading in the Qwen case.
        print("[\(template.logCategory)] \(s)")
    }

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            let id = UUID()
            queue.sync {
                cont.yield(self.state)
                self.continuations[id] = cont
            }
            // Async removal: onTermination can fire from any context,
            // including re-entrantly during the registration above.
            cont.onTermination = { [weak self] _ in
                guard let self else { return }
                self.queue.async { self.continuations[id] = nil }
            }
        }
    }

    private func set(_ s: ModelLoadState) {
        queue.sync {
            self.state = s
            self.continuations.values.forEach { $0.yield(s) }
        }
    }

    public func load() async throws {
        // Idempotency: if the container is already built, a second `load()`
        // call would allocate a *second* ~2.6 GB weight set and leak the
        // original — we saw this correlate with runaway resident memory.
        if container != nil {
            debug("load() no-op (container already ready)")
            return
        }
        debug("load() start; modelId=\(modelConfiguration.name)")
        log.notice("load() start; modelId=\(self.modelConfiguration.name, privacy: .public)")
        #if targetEnvironment(simulator)
        // MLX aborts in `mlx::core::metal::Device::Device()` inside the iOS
        // Simulator because the simulator's Metal driver (`MTLSimDriver`) lacks
        // the GPU features MLX requires. Fail loudly here instead of letting
        // the C++ side SIGABRT the whole app.
        let msg = "MLX / Gemma 4 cannot run in the iOS Simulator — attach a physical iPhone (A17 Pro / M-series iPad) or build the macOS/Catalyst target."
        set(.failed(msg))
        throw ModelError.simulatorUnsupported(msg)
        #else
        set(.downloading(0))
        do {
            // mlx-swift-lm 3.x no longer bakes a hub downloader + tokenizer
            // loader into `LLMModelFactory.shared.loadContainer`. The
            // composite `#huggingFaceLoadModelContainer` macro fails to
            // type-check when invoked from inside our async/`do` context
            // (internal macro diagnostic about an `@Sendable () async
            // throws -> String?` closure signature), so we call the two
            // underlying macros inline and invoke `loadContainer(from:
            // using: configuration:progressHandler:)` directly — same
            // effect, but Swift can typecheck the smaller fragments.
            let modelConfiguration = self.modelConfiguration
            debug("downloading weights from HF Hub…")
            log.notice("loadContainer: downloading Gemma-4 weights from HF hub")
            // HubClient's default URLSession uses
            // `URLSessionConfiguration.default`, which caps
            // `timeoutIntervalForRequest` at 60s. For a 2.5 GB
            // safetensors file over a cellular or slow wifi link
            // that's far too short — the download aborts on the
            // first chunk that takes longer than 60s to arrive, and
            // the app sees "download failed" without a useful reason.
            // Pass an explicit session with a 10-minute per-request
            // timeout + 1-hour per-resource timeout to make large
            // first-launch downloads survive normal network wobbles.
            let longDownloadSession: URLSession = {
                let cfg = URLSessionConfiguration.default
                cfg.timeoutIntervalForRequest = 600      // 10 minutes
                cfg.timeoutIntervalForResource = 3600    // 1 hour
                cfg.waitsForConnectivity = true
                return URLSession(configuration: cfg)
            }()
            let hub = HubClient(session: longDownloadSession)

            // `swift-huggingface`'s progressHandler coalesces byte-level
            // updates through an NSProgress tree — for a snapshot with
            // one dominant `model.safetensors` the parent's
            // `fractionCompleted` can sit near 0 for long stretches even
            // while the file is actively growing on disk. Observed on
            // 2026-04-23: UI stalled at 1% on a 2.5 GB download even
            // though bytes were flowing.
            //
            // Sidestep by polling the HF cache dir in parallel with the
            // Hub download: sum all file bytes under
            // `<cachesDir>/huggingface/hub/models--<org>--<repo>/` every
            // 750 ms and report `bytes / (approximateMemoryMB * MB)` as
            // the fraction. Whichever signal (Hub native or disk poll)
            // reports a higher fraction wins, so progress is monotonic.
            set(.downloading(0))

            // Expected cache root = default `swift-huggingface` location
            // on iOS / macOS: `<cachesDir>/huggingface/hub`. The repo
            // folder uses Python-compatible naming: `models--<org>--<repo>`.
            let hfHubRoot: URL = {
                let caches = FileManager.default.urls(
                    for: .cachesDirectory, in: .userDomainMask
                ).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
                return caches
                    .appendingPathComponent("huggingface")
                    .appendingPathComponent("hub")
            }()
            let repoDirName = "models--" + modelConfiguration.name
                .replacingOccurrences(of: "/", with: "--")
            let repoRoot = hfHubRoot.appendingPathComponent(repoDirName)
            // Approximate total bytes from provider metadata. Slightly
            // over-estimated on purpose (bumps MB→bytes with 1024×1024
            // vs the actual weight file size) so real progress caps
            // just below 100% rather than overshooting when blobs +
            // metadata together exceed approximateMemoryMB.
            let approxTotalBytes: Int64 = Int64(approximateMemoryMB) * 1024 * 1024

            let diskFrac = DiskFraction()
            let pollTask = Task { [weak self] in
                while !Task.isCancelled {
                    let bytes = dirSizeBytes(at: repoRoot)
                    if bytes > 0, approxTotalBytes > 0 {
                        let frac = min(0.99, Double(bytes) / Double(approxTotalBytes))
                        await diskFrac.set(frac)
                        self?.set(.downloading(await diskFrac.maxFraction()))
                    }
                    // 2.5 s cadence: each tick walks the whole HF cache tree
                    // with per-file resourceValues, so a 750 ms poll spent a
                    // multi-minute 2.5 GB download re-scanning metadata in
                    // competition with the download I/O itself. The Hub
                    // progressHandler below still delivers fine-grained
                    // updates between ticks.
                    try? await Task.sleep(nanoseconds: 2_500_000_000)
                }
            }
            defer { pollTask.cancel() }

            let container = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(hub),
                using: #huggingFaceTokenizerLoader(),
                configuration: modelConfiguration,
                progressHandler: { [weak self] progress in
                    let pct = progress.fractionCompleted
                    // diskFrac is an actor; we can't `await` from a sync
                    // closure. Instead update with the Hub value and let
                    // the poll task raise the floor on its next tick.
                    self?.set(.downloading(pct))
                    let intPct = Int(pct * 100)
                    if intPct == 1 || intPct % 5 == 0 {
                        self?.debug("download \(intPct)%")
                        log.notice("download progress: \(intPct)%")
                    }
                }
            )
            self.container = container
            debug("container ready")
            log.notice("loadContainer: ready")
            set(.ready)
        } catch {
            log.error("load() failed: \(String(describing: error), privacy: .public)")
            set(.failed(String(describing: error)))
            throw error
        }
        #endif
    }

    public func unload() async {
        // Aggressively drop every MLX-held reference so the next
        // picker choice can claim the memory the departing model
        // held. Without this, ARC releases the `container` lazily,
        // the KV-cache mirror keeps MLXArrays alive, and the Metal
        // buffer pool hoards intermediates up to `mlxCacheLimitMB` —
        // all of which compound on the NEW model's load-time peak
        // and can tip an iPhone into jetsam.
        container = nil
        promptKVCache = nil
        cachedTokens = []
        generatedTokensThisTurn = []
        // Drain GPU work then release pool buffers. Safe here
        // because by the time `unload()` is called we've already
        // bailed out of the tool-loop and nobody else is issuing
        // MLX work against this container.
        Stream.defaultStream(.gpu).synchronize()
        MLX.GPU.clearCache()
        set(.notLoaded)
        debug("unloaded + drained MLX pool")
    }

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream<String, Error> { (continuation: AsyncThrowingStream<String, Error>.Continuation) in
            guard let container else {
                continuation.finish(throwing: ModelError.notLoaded)
                return
            }
            Task {
                do {
                    self.debug("generate() prompt=\(prompt.count) chars, maxTokens=\(parameters.maxTokens)")
                    log.notice("generate() prompt=\(prompt.count) chars, maxTokens=\(parameters.maxTokens)")
                    // Start from a clean Metal baseline — anything the
                    // previous turn (or Kokoro TTS) left behind would
                    // stack with this generate's prefill spike and can
                    // push past the 6144 MB per-process cap on a long
                    // route reply.
                    Stream.defaultStream(.gpu).synchronize()
                    MLX.GPU.clearCache()
                    let t0 = Date()
                    let memoryStarted = MemoryStats.physFootprintMB()
                    let tokens = await container.encode(prompt)
                    let tokens32 = tokens.map { Int32($0) }
                    let tokenizeSeconds = Date().timeIntervalSince(t0)
                    self.debug(String(format: "encoded %d tokens in %.2fs", tokens.count, tokenizeSeconds))
                    // Same field layout as LlamaCppProvider's perf lines so a
                    // cross-runtime A/B diffs cleanly (lock is llama-only —
                    // MLX serializes through the container actor instead).
                    self.debug(String(format:
                        "perf start · prompt=%d chars · lock=0.000s · tokenize=%d tok/%.3fs · thermal=%@ · footprint=%.0f MB",
                        prompt.count, tokens.count, tokenizeSeconds,
                        Self.thermalStateLabel(), memoryStarted))

                    // KV-cache quantization, wired via the vendored
                    // mlx-swift-lm fork's patched `Gemma4Attention`.
                    // `maybeQuantizeKVCache` inside `TokenIterator`
                    // swaps every `KVCacheSimple` (full-attention)
                    // layer to `QuantizedKVCache` once
                    // `offset > quantizedKVStart`. Our patch branches
                    // on `QuantizedKVCacheProtocol` and routes through
                    // `quantizedScaledDotProductAttention` against the
                    // 4-bit groupwise state. Sliding (RotatingKVCache)
                    // layers stay FP16 — upstream MLX doesn't support
                    // rotating-quantized yet.
                    //
                    // `quantizedKVStart: 0`: swap immediately after the
                    // first step. Prefill itself stays FP16 because
                    // `model.prepare(...)` does chunked prefill without
                    // calling `maybeQuantizeKVCache` between chunks,
                    // so the peak-prefill spike is unchanged — but the
                    // steady-state cache (what lingers during and
                    // after generation) is ~4× smaller. Gated on
                    // `DeviceProfile.useQuantizedKVCache` (Mac stays
                    // FP16 — the memory win doesn't matter there).
                    let useQuant = DeviceProfile.current.useQuantizedKVCache
                    // Publisher sampling profile, same precedence rule as
                    // LlamaCppProvider: model profile wins when the caller
                    // opted in, else the task-level parameters stand.
                    let profile = parameters.useModelSamplingProfile
                        ? self.samplingProfile : nil
                    let temperature = profile?.temperature ?? parameters.temperature
                    let topP = profile?.topP ?? parameters.topP
                    let topK = profile?.topK ?? 0
                    let presencePenalty = profile?.presencePenalty ?? 0
                    self.debug(String(format:
                        "sampler · temp=%.2f · top-p=%.2f · top-k=%d · presence=%.2f",
                        temperature, topP, topK, presencePenalty))
                    var genParams = GenerateParameters(
                        maxTokens: parameters.maxTokens,
                        kvBits: useQuant ? 4 : nil,
                        kvGroupSize: 64,
                        quantizedKVStart: 0,
                        temperature: Float(temperature),
                        topP: Float(topP),
                        prefillStepSize: 128
                    )
                    genParams.topK = topK
                    if presencePenalty != 0 {
                        genParams.presencePenalty = Float(presencePenalty)
                    }

                    // Decide cache reuse. The cache stores the CUMULATIVE
                    // state of (prompt + generated tokens) from the
                    // previous turn. Gemma 4's architecture mixes
                    // RotatingKVCache with StandardKVCache, so
                    // `trimPromptCache` always refuses — meaning we
                    // can only reuse when the new prompt EXACTLY
                    // starts with `cachedTokens`. That's true for
                    // append-only transcripts (tool-call
                    // round-trips, follow-up user turns without
                    // conversation reset).
                    let prefillStarted = Date()
                    let (tokenStream, tokenizerBox, prefillInputCount, reusedCount, lcpCandidate) = try await container.perform { (context) -> (AsyncStream<TokenGeneration>, SendableTokenizer, Int, Int, Int) in
                        let existing = self.promptKVCache
                        let cached = self.cachedTokens
                        let common = Self.longestCommonPrefix(cached, tokens32)
                        let inputTokens: [Int32]
                        let kvCache: [KVCache]
                        let hit: Bool
                        // Hybrid-cache reuse guard — RETIRED 2026-07-19.
                        // The blanket "force full prefill when the cache
                        // contains a MambaCache" workaround dated from the
                        // mlx-swift-lm#157 era (`broadcast_shapes (…128,256)
                        // vs (…129,256)` SIGABRT on partial-prefix reuse,
                        // QWEN35_HYBRID_CACHE.md). The vendored library was
                        // refreshed for Bonsai and the current Qwen35
                        // classes derive RoPE/masks/conv state from cache
                        // offset correctly; verified empirically on this
                        // machine for BOTH the original crasher (Qwen 3.5
                        // 4B: 835-token reuse, no abort) and Bonsai 27B
                        // ternary (838/1071-token reuse, follow-up TTFT
                        // 4.8s → 1.0s, answers correct). If a hybrid family
                        // regresses, set `hasStaleScratchStateBug = true`
                        // on ITS template rather than reinstating the
                        // blanket guard — the per-eval-state kill-switch
                        // below still forces full prefill per family, and
                        // the pre-`generateTokens` canary logging remains
                        // to diagnose any MLX-side abort.
                        let cacheIsHybrid = false
                        // Template-declared flag for families whose MLX
                        // model carries per-eval mutable state that
                        // leaks across calls. Gemma 3 / Qwen 3.5 are
                        // the known offenders (mlx-swift-lm#157:
                        // `precomputedPositionIds` / `ropeDeltas` on
                        // the model class, not the cache). Currently
                        // all known affected families ALSO use hybrid
                        // attention, so the `cacheIsHybrid` guard
                        // above catches them; the flag is reserved
                        // for families where we'd otherwise reuse.
                        // Gemma 3 sets this to `false` on purpose
                        // (see `Gemma3Template.hasStaleScratchStateBug`)
                        // — the Mac EvalCLI repro never fired on
                        // device, and forcing full prefill every turn
                        // jetsams the phone at 6 GB. The diagnostic
                        // block below is the tripwire: if the stale-
                        // state bug DOES fire on device, the MLX
                        // SIGABRT truncates Swift logs, so we dump
                        // cache layer shapes + offsets + LCP state
                        // right before `generateTokens` so the OS
                        // log tail captures what we handed MLX.
                        let templateBugsReuse = self.template.hasStaleScratchStateBug
                        if common == cached.count, common > 0, let existing, !existing.isEmpty,
                           !cacheIsHybrid, !templateBugsReuse
                        {
                            // Cache is a clean prefix — feed just the
                            // new tokens and keep the cache going.
                            kvCache = existing
                            inputTokens = Array(tokens32[common...])
                            hit = true
                        } else {
                            // Anything else (mismatch, shrink, empty
                            // cache) → fresh prefill of the whole
                            // prompt.
                            kvCache = context.model.newCache(parameters: genParams)
                            inputTokens = tokens32
                            hit = false
                        }
                        self.promptKVCache = kvCache
                        // Seed cachedTokens with the prompt — we'll
                        // append the generated tokens as they stream
                        // in, so the next call can match against the
                        // FULL cache contents.
                        self.cachedTokens = tokens32
                        self.generatedTokensThisTurn = []
                        let cacheShape = Self.describeCache(kvCache)
                        if hit {
                            self.debug("cache HIT: reusing \(common)/\(tokens32.count) prompt tokens, prefilling \(inputTokens.count) new · cache=\(cacheShape)")
                        } else {
                            // Emit the LCP length so we can tell whether
                            // we're a byte off the end of the cache
                            // (tokenizer boundary issue) or missing
                            // because the prompt legitimately changed.
                            // Also tag WHY we missed — stale-state
                            // flag, hybrid cache, or LCP divergence —
                            // so the reason is obvious from the log
                            // line alone.
                            let reason: String
                            if templateBugsReuse {
                                reason = "forced by hasStaleScratchStateBug"
                            } else if cacheIsHybrid {
                                reason = "forced by hybrid MambaCache"
                            } else if existing == nil || (existing?.isEmpty ?? true) {
                                reason = "no existing cache"
                            } else if common == 0 {
                                reason = "LCP=0 (prompt reset)"
                            } else {
                                reason = "LCP<cached.count (prompt diverged)"
                            }
                            self.debug("cache MISS: full prefill \(inputTokens.count) tokens — \(reason) (LCP=\(common), cached.count=\(cached.count), prompt.count=\(tokens32.count)) · cache=\(cacheShape)")
                            if common > 0 && common + 1 >= cached.count && cached.count > 0 {
                                // Very close — log a few tokens around the divergence point
                                let divergeIdx = common
                                let before = Array(cached.suffix(from: max(0, divergeIdx - 5)).prefix(min(10, cached.count - max(0, divergeIdx - 5))))
                                let after = Array(tokens32.suffix(from: max(0, divergeIdx - 5)).prefix(min(10, tokens32.count - max(0, divergeIdx - 5))))
                                self.debug("cache diverge near idx \(divergeIdx): cached=[\(before.map(String.init).joined(separator: ","))] prompt=[\(after.map(String.init).joined(separator: ","))]")
                            }
                        }
                        // Canary: the MLX C++ layer can SIGABRT inside
                        // `generateTokens` before any Swift catch runs
                        // (stale-state shape mismatch, mutex lock, OOM
                        // from Metal pool). Log cache shape + input
                        // token count RIGHT before the call so the OS
                        // log tail captures the last known good state.
                        // If the process dies without a matching
                        // "stream opened" line, the crash was inside
                        // `generateTokens` with this payload.
                        self.debug("generateTokens ← input=\(inputTokens.count) tokens, cache=\(cacheShape), hit=\(hit)")
                        let input = LMInput(tokens: MLXArray(inputTokens))
                        let stream = try MLXLMCommon.generateTokens(
                            input: input, cache: kvCache,
                            parameters: genParams, context: context
                        )
                        return (stream, SendableTokenizer(context.tokenizer),
                                inputTokens.count, hit ? common : 0, cached.count)
                    }
                    // `generateTokens` runs the chunked prefill inside
                    // TokenIterator init, i.e. inside the perform block —
                    // its duration IS the prefill (plus actor hop noise).
                    let prefillSeconds = Date().timeIntervalSince(prefillStarted)
                    let prefillRate = prefillSeconds > 0
                        ? Double(prefillInputCount) / prefillSeconds : 0
                    let prefillMemory = MemoryStats.physFootprintMB()
                    self.debug(String(format:
                        "perf prefill · prompt=%d tok · LCP candidate=%d · reused=%d · mode=%@ · batches=%d · %.3fs · %.1f tok/s · footprint=%.0f MB (Δ%+.0f)",
                        tokens.count, lcpCandidate, reusedCount,
                        reusedCount > 0 ? "mlx-reuse" : "mlx-fresh",
                        (prefillInputCount + 127) / 128,
                        prefillSeconds, prefillRate, prefillMemory,
                        prefillMemory - memoryStarted))
                    let tokenizer = tokenizerBox.wrapped
                    self.debug("stream opened — awaiting first token…")
                    // Gemma's tokenizer encodes `<end_of_turn>` as byte-BPE, not
                    // as a dedicated stop token, so MLX won't halt on it — we
                    // watch the rolling text buffer and terminate on the first
                    // occurrence of either turn marker ourselves. `pending` holds
                    // the tail of the last chunk that *might* be the start of a
                    // marker (so we don't flush "<" right before we'd otherwise
                    // recognise "<end_of_turn>" and hide it).
                    var pending = ""
                    var firstChunkSeen = false
                    var chunkIdx = 0
                    var firstTokenAt: Date?
                    var lastTokenAt: Date?
                    var stopReason = "eos_or_max"
                    // Halt on a turn close (end of assistant) OR on a tool_call
                    // close (ready for dispatch). Also stop if we see a stray
                    // system-turn open, which should never appear in model output.
                    let stopMarkers = ["<turn|>", "<|turn>"]
                    let maxMarker = stopMarkers.map(\.count).max() ?? 0
                    let tFirstChunk = Date()
                    // Incremental detokenisation: keep a running list of
                    // generated token IDs and rolling decoded string. Each
                    // new token's contribution = full decode minus what we
                    // already emitted. Also accumulate the IDs onto
                    // `self.generatedTokensThisTurn` so the next call can
                    // reuse the cache if the new prompt includes our
                    // output verbatim.
                    var decodedSoFar = ""
                    var tokenIDs: [Int32] = []
                    // Parallel Int copy for the tokenizer API — the old
                    // `tokenIDs.map { Int($0) }` allocated a fresh n-element
                    // array on every generated token.
                    var tokenIDsInt: [Int] = []
                    // Cache the length of `tokenIDs` at the point where
                    // ChatSession will cut the assistant turn (either a
                    // `<tool_call|>` closing marker, or the stop marker
                    // if generation runs to completion). Tokens past
                    // that point are thrown away by ChatSession and
                    // must NOT end up in the KV-cache mirror — else
                    // the next turn's LCP match diverges by whatever
                    // extra tokens we captured and we pay a full
                    // prefill. Set once, the first time we see either
                    // boundary.
                    var tokensAtCutoff: Int? = nil
                    let toolCallClose = "<tool_call|>"
                    chunkLoop: for await event in tokenStream {
                        guard case .token(let id) = event else { continue }
                        tokenIDs.append(Int32(id))
                        tokenIDsInt.append(Int(id))
                        let fullDecoded = tokenizer.decode(
                            tokenIds: tokenIDsInt,
                            skipSpecialTokens: false)
                        let newText: String
                        if fullDecoded.hasPrefix(decodedSoFar) {
                            newText = String(fullDecoded.dropFirst(decodedSoFar.count))
                        } else {
                            // Tokenizer rewrote earlier spans (rare with
                            // BPE). Reset the diff anchor so we don't
                            // miss content.
                            newText = fullDecoded
                        }
                        decodedSoFar = fullDecoded
                        if newText.isEmpty { continue }
                        if !firstChunkSeen {
                            let now = Date()
                            firstTokenAt = now
                            self.debug(String(format: "first token after %.2fs (%d chars)", now.timeIntervalSince(tFirstChunk), newText.count))
                            self.debug(String(format:
                                "perf first token · TTFT=%.3fs · after-prefill=%.3fs",
                                now.timeIntervalSince(t0),
                                now.timeIntervalSince(prefillStarted) - prefillSeconds))
                            firstChunkSeen = true
                        }
                        lastTokenAt = Date()
                        chunkIdx += 1
                        if chunkIdx.isMultiple(of: 40) {
                            self.debug("streaming · \(chunkIdx) chunks")
                        }
                        pending += newText
                        // If we cross a `<tool_call|>` boundary and
                        // haven't recorded the cutoff yet, stamp it
                        // now — ChatSession will truncate the
                        // assistant turn right after this marker.
                        if tokensAtCutoff == nil,
                           Self.tailContains(decodedSoFar, marker: toolCallClose,
                                             newSuffixCount: newText.count)
                        {
                            tokensAtCutoff = tokenIDs.count
                            // Halt the producer the instant we see
                            // `<tool_call|>`. Without this we keep
                            // generating 1–2 more tokens after
                            // ChatSession has already broken its
                            // read loop; those trailing tokens land
                            // in `committed` / `cachedTokens` but
                            // never make it into ChatSession's
                            // `buffer`, so the next turn's LCP
                            // falls short by exactly those tokens
                            // and we pay a full prefill.
                            // Flush any queued chars first so
                            // ChatSession sees a complete
                            // `<tool_call|>` sequence.
                            if !pending.isEmpty {
                                continuation.yield(pending)
                                pending = ""
                            }
                            self.debug("tool_call close seen — halting stream at \(tokenIDs.count) tokens")
                            stopReason = "tool_call"
                            break chunkLoop
                        }
                        if let hit = stopMarkers.compactMap({ pending.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
                            let clean = String(pending[..<hit.lowerBound])
                            if !clean.isEmpty { continuation.yield(clean) }
                            pending = ""
                            // Same rationale — only stamp if we
                            // hadn't already (a tool-call cut is
                            // earlier and more precise).
                            if tokensAtCutoff == nil {
                                tokensAtCutoff = tokenIDs.count
                            }
                            self.debug("stop marker hit — halting stream")
                            stopReason = "stop_marker"
                            break chunkLoop
                        }
                        if pending.count > maxMarker {
                            let flushEnd = pending.index(pending.endIndex, offsetBy: -(maxMarker - 1))
                            continuation.yield(String(pending[..<flushEnd]))
                            pending = String(pending[flushEnd...])
                        }
                    }
                    if !pending.isEmpty { continuation.yield(pending) }
                    // Commit the generated IDs into the KV-cache mirror
                    // so the next call's LCP pass recognises the full
                    // prefix and can reuse the cache. Truncate at
                    // the tool-call / stop-marker cutoff so only the
                    // tokens ChatSession will actually re-feed get
                    // stored. Without this, a few trailing tokens
                    // (produced between the tool_call close and the
                    // turn-marker) land in the cache but NOT in the
                    // next turn's prompt, which breaks LCP.
                    let kept = tokensAtCutoff ?? tokenIDs.count
                    let committed = Array(tokenIDs.prefix(kept))
                    self.generatedTokensThisTurn = committed
                    self.cachedTokens.append(contentsOf: committed)
                    // Across multi-turn sessions the Metal buffer pool
                    // creeps well past the DeviceProfile cache cap even
                    // with prefillStepSize=128, and a long-form turn
                    // eventually crosses the 6144 MB jetsam highwater.
                    // Force completion of any pending Metal work before
                    // dropping the pool — this closes the race that tripped
                    // `check_error` in the 23:32 crash: the completion
                    // handler runs on MLX's own queue and reads
                    // `commandBuffer.error`, so we just need to make sure
                    // no command buffers are still in-flight when we
                    // signal the pool to release.
                    Stream.defaultStream(.gpu).synchronize()
                    MLX.GPU.clearCache()
                    // Uniform completion line + stats — mirror of
                    // LlamaCppProvider's `perf complete` so cross-runtime
                    // comparisons read straight out of the logs.
                    let requestFinished = Date()
                    let totalSeconds = requestFinished.timeIntervalSince(t0)
                    let ttft = firstTokenAt.map { $0.timeIntervalSince(t0) } ?? -1
                    let steadyDecodeTokens = max(0, tokenIDs.count - 1)
                    let steadyDecodeSeconds: Double
                    if let f = firstTokenAt, let l = lastTokenAt, l > f {
                        steadyDecodeSeconds = l.timeIntervalSince(f)
                    } else {
                        steadyDecodeSeconds = 0
                    }
                    let steadyDecodeRate = steadyDecodeSeconds > 0
                        ? Double(steadyDecodeTokens) / steadyDecodeSeconds : 0
                    let generationSeconds = firstTokenAt.map {
                        requestFinished.timeIntervalSince($0)
                    } ?? 0
                    let generationRate = generationSeconds > 0
                        ? Double(tokenIDs.count) / generationSeconds : 0
                    let memoryFinished = MemoryStats.physFootprintMB()
                    self.debug(String(format:
                        "perf complete · output=%d tok · generation=%.3fs/%.2f tok/s · steady-decode=%d tok/%.3fs/%.2f tok/s · TTFT=%.3fs · total=%.3fs · stop=%@ · thermal=%@ · footprint=%.0f MB (Δ%+.0f)",
                        tokenIDs.count, generationSeconds, generationRate,
                        steadyDecodeTokens, steadyDecodeSeconds, steadyDecodeRate,
                        ttft, totalSeconds, stopReason,
                        Self.thermalStateLabel(), memoryFinished,
                        memoryFinished - memoryStarted))
                    self.lastGenerationStats = GenerationStats(
                        runtime: "mlx", modelID: self.id,
                        promptTokens: tokens.count, reusedTokens: reusedCount,
                        prefillSeconds: prefillSeconds, ttftSeconds: ttft,
                        outputTokens: tokenIDs.count,
                        decodeTokensPerSecond: steadyDecodeRate,
                        totalSeconds: totalSeconds,
                        peakFootprintMB: max(prefillMemory, memoryFinished),
                        stopReason: stopReason)
                    self.debug(String(format: "generate() finished — %d chunks, %.2fs total", chunkIdx, totalSeconds))
                    log.notice("generate() finished")
                    continuation.finish()
                } catch {
                    self.debug("generate() threw: \(error)")
                    log.error("generate() failed: \(String(describing: error), privacy: .public)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func formatTranscript(systemPreamble: String, turns: [ChatTurn]) -> String {
        Gemma4PromptTemplate.render(systemPreamble: systemPreamble, turns: turns)
    }

    /// Same labels as `LlamaCppProvider.thermalStateLabel()` so the perf
    /// lines diff cleanly across runtimes.
    private static func thermalStateLabel() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func longestCommonPrefix(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    /// Bounded `contains` for streaming: a marker completed by this chunk
    /// must END inside the newly appended suffix, so only the last
    /// `newSuffixCount + marker.count - 1` characters need scanning — the
    /// old whole-buffer `contains` re-scanned the entire accumulated reply
    /// on every generated token. The induction holds because every appended
    /// suffix is checked exactly once.
    private static func tailContains(
        _ text: String, marker: String, newSuffixCount: Int
    ) -> Bool {
        let window = newSuffixCount + marker.count - 1
        guard window > 0, !text.isEmpty else { return false }
        let start = text.index(
            text.endIndex, offsetBy: -window, limitedBy: text.startIndex
        ) ?? text.startIndex
        return text.range(of: marker, range: start..<text.endIndex) != nil
    }

    /// Compact one-line dump of a KV cache: layer count, the set of
    /// underlying types (e.g. `Rotating/Standard/Mamba`), and the
    /// min/max/first layer offsets. Used as pre-`generateTokens`
    /// breadcrumb so an MLX SIGABRT leaves us something to diagnose.
    /// Cheap — just reads `.offset` and class names, no MLXArray
    /// traversal.
    private static func describeCache(_ cache: [KVCache]) -> String {
        guard !cache.isEmpty else { return "empty" }
        let offsets = cache.map { $0.offset }
        let typeNames = cache.map { String(describing: type(of: $0)).split(separator: ".").last.map(String.init) ?? "?" }
        let uniq = Array(Set(typeNames)).sorted().joined(separator: "/")
        let minO = offsets.min() ?? 0
        let maxO = offsets.max() ?? 0
        return "\(cache.count)L[\(uniq)] off=\(minO)…\(maxO)"
    }
}

/// `Tokenizer` is not `Sendable` — shim lets us hand the instance
/// back out of a `container.perform { ... }` closure so the caller
/// can detokenise incrementally outside the actor isolation. The
/// model container ensures the tokenizer isn't concurrently mutated
/// while generation is in flight.
///
/// In mlx-swift-lm 3.x the `Tokenizer` protocol lives in
/// `MLXLMCommon` directly (no longer imported from `Tokenizers`),
/// so the module import at the top of this file is enough.

private struct SendableTokenizer: @unchecked Sendable {
    // Both `MLXLMCommon` and `Tokenizers` (swift-transformers) declare
    // a `Tokenizer` protocol; we want MLX's narrower one so the shim
    // matches `context.tokenizer`'s type.
    let wrapped: any MLXLMCommon.Tokenizer
    init(_ t: any MLXLMCommon.Tokenizer) { self.wrapped = t }
}

#else

// Fallback used when the Swift-Gemma4-Core package hasn't been added yet.
// Keeps the app buildable and surfaces a clean error in the model picker.
public final class Gemma4Provider: ModelProvider, @unchecked Sendable {
    public let id = "gemma4-4b-it-4bit"
    public let displayName = "Gemma 4 4B (4-bit) — not linked"
    public let approximateMemoryMB = 400
    public let supportsToolCalls = true

    public init() {}

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            cont.yield(.failed("Add the Swift-Gemma4-Core package dependency (see ios/README.md)."))
            cont.finish()
        }
    }

    public func load() async throws { throw ModelError.notLinked }
    public func unload() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { cont in cont.finish(throwing: ModelError.notLinked) }
    }

    public func formatTranscript(systemPreamble: String, turns: [ChatTurn]) -> String {
        Gemma4PromptTemplate.render(systemPreamble: systemPreamble, turns: turns)
    }
}

#endif

public enum ModelError: Error, CustomStringConvertible {
    case notLoaded
    case notLinked
    case simulatorUnsupported(String)

    public var description: String {
        switch self {
        case .notLoaded: return "Model is not loaded yet."
        case .notLinked: return "Gemma 4 runtime is not linked. Add the Swift-Gemma4-Core Swift Package to this target."
        case .simulatorUnsupported(let msg): return msg
        }
    }
}
