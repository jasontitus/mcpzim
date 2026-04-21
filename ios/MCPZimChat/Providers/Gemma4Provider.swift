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

#if canImport(Gemma4SwiftCore) && canImport(MLXLMCommon)
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

public final class Gemma4Provider: ModelProvider, @unchecked Sendable {
    public let id = "gemma4-4b-it-4bit"
    public let displayName = "Gemma 4 4B (4-bit)"
    /// Resident weights ~2.6 GB once the 4-bit safetensors decompress.
    /// Displayed in the model picker so the user sees the real footprint
    /// before committing to a load. Generation itself adds KV cache
    /// (~125 MB with 4-bit KV quant, ~500 MB FP16 on long prompts) plus
    /// MLX's Metal pool (~400 MB default).
    public let approximateMemoryMB = 2600
    public let supportsToolCalls = true

    private let modelId = Gemma4SwiftCore.verifiedModelId
    private var container: ModelContainer?
    private var state: ModelLoadState = .notLoaded
    private var continuations: [AsyncStream<ModelLoadState>.Continuation] = []
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
            let params = GenerateParameters(
                maxTokens: 0,                 // no generation
                temperature: 0.0, topP: 1.0,
                prefillStepSize: 128
            )
            let kvCache = context.model.newCache(parameters: params)
            let input = LMInput(tokens: MLXArray(tokens32))
            // Constructing a TokenIterator runs the full prefill via
            // `model.prepare(...)` + one `step(...)` over the remaining
            // tail — at end of init the KV cache holds exactly
            // `tokens32.count` tokens' worth of state. We never call
            // `.next()`, so nothing beyond the prompt is committed.
            _ = try TokenIterator(
                input: input, model: context.model, cache: kvCache, parameters: params
            )
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

    public init() {
        // Register Gemma-4's model/tokenizer handler with MLXLMCommon. The
        // returned Task resolves once registration is idempotent-complete.
        Task { await Gemma4Registration.registerIfNeeded().value }

        // MLX's Metal buffer pool defaults its cache limit to the system
        // memory limit (~128 GB on an M1 Ultra) — which means it happily
        // hoards tens of GB of intermediate tensors from past generations
        // "just in case". Cap the cache per device tier so 6 GB phones
        // don't jetsam while Macs and 12 GB Pros still get hot-reuse.
        // Budget: weights ~2.6 GB + active KV cache ~1 GB + pool cap.
        MLX.GPU.set(cacheLimit: DeviceProfile.current.mlxCacheLimitMB * 1024 * 1024)
    }

    private func debug(_ s: String) {
        debugSink?(s)
        print("[Gemma4] \(s)")
    }

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            queue.sync {
                cont.yield(self.state)
                self.continuations.append(cont)
            }
        }
    }

    private func set(_ s: ModelLoadState) {
        queue.sync {
            self.state = s
            self.continuations.forEach { $0.yield(s) }
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
        debug("load() start; modelId=\(modelId)")
        log.notice("load() start; modelId=\(self.modelId, privacy: .public)")
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
            // mlx-swift-lm's Hub client accepts a `progressHandler` we can
            // surface as UI state; it also logs to `log` so a tail shows the
            // per-file download rate.
            debug("downloading weights from HF Hub…")
            log.notice("loadContainer: downloading Gemma-4 weights from HF hub")
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: ModelConfiguration(id: modelId),
                progressHandler: { [weak self] progress in
                    let pct = progress.fractionCompleted
                    self?.set(.downloading(pct))
                    if Int(pct * 100) % 5 == 0 {
                        self?.debug("download \(Int(pct * 100))%")
                        log.notice("download progress: \(Int(pct * 100))%")
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
        container = nil
        set(.notLoaded)
    }

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
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
                    let tokens = await container.encode(prompt)
                    let tokens32 = tokens.map { Int32($0) }
                    self.debug(String(format: "encoded %d tokens in %.2fs", tokens.count, Date().timeIntervalSince(t0)))

                    // KV-cache quantization knobs, wired via
                    // `DeviceProfile.useQuantizedKVCache`. `maybeQuantizeKVCache`
                    // inside MLXLMCommon.TokenIterator swaps every
                    // `KVCacheSimple` (full-attention) layer to a
                    // `QuantizedKVCache` once `offset > quantizedKVStart`.
                    // The vendored Gemma4-Core fork's attention forward
                    // branches on `QuantizedKVCacheProtocol` and runs
                    // `quantizedScaledDotProductAttention` against the
                    // 4-bit groupwise state. Sliding (RotatingKVCache)
                    // layers stay FP16 — upstream MLX doesn't yet
                    // support rotating-quantized. That's fine: the
                    // full-attention layers are the ones that grow
                    // unboundedly with prompt length, and those are
                    // what dominate KV memory on long prompts.
                    //
                    // quantizedKVStart=0: swap immediately after the
                    // first step. Prefill itself stays FP16 (the model's
                    // `prepare(...)` does chunked prefill without
                    // quantizing between chunks) so the peak prefill
                    // spike is unchanged — but the steady-state cache
                    // (what lingers during and after generation) is
                    // ~4× smaller.
                    let useQuant = DeviceProfile.current.useQuantizedKVCache
                    let genParams = GenerateParameters(
                        maxTokens: parameters.maxTokens,
                        kvBits: useQuant ? 4 : nil,
                        kvGroupSize: 64,
                        quantizedKVStart: 0,
                        temperature: Float(parameters.temperature),
                        topP: Float(parameters.topP),
                        prefillStepSize: 128
                    )

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
                    let (tokenStream, tokenizerBox) = try await container.perform { (context) -> (AsyncStream<TokenGeneration>, SendableTokenizer) in
                        let existing = self.promptKVCache
                        let cached = self.cachedTokens
                        let common = Self.longestCommonPrefix(cached, tokens32)
                        let inputTokens: [Int32]
                        let kvCache: [KVCache]
                        let hit: Bool
                        if common == cached.count, common > 0, let existing, !existing.isEmpty {
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
                        if hit {
                            self.debug("cache hit: reusing \(common)/\(tokens32.count) prompt tokens, prefilling \(inputTokens.count) new")
                        } else {
                            // Emit the LCP length so we can tell whether
                            // we're a byte off the end of the cache
                            // (tokenizer boundary issue) or missing
                            // because the prompt legitimately changed.
                            self.debug("cache miss: full prefill \(inputTokens.count) tokens (LCP=\(common), cached.count=\(cached.count), prompt.count=\(tokens32.count))")
                            if common > 0 && common + 1 >= cached.count && cached.count > 0 {
                                // Very close — log a few tokens around the divergence point
                                let divergeIdx = common
                                let before = Array(cached.suffix(from: max(0, divergeIdx - 5)).prefix(min(10, cached.count - max(0, divergeIdx - 5))))
                                let after = Array(tokens32.suffix(from: max(0, divergeIdx - 5)).prefix(min(10, tokens32.count - max(0, divergeIdx - 5))))
                                self.debug("cache diverge near idx \(divergeIdx): cached=[\(before.map(String.init).joined(separator: ","))] prompt=[\(after.map(String.init).joined(separator: ","))]")
                            }
                        }
                        let input = LMInput(tokens: MLXArray(inputTokens))
                        let stream = try MLXLMCommon.generateTokens(
                            input: input, cache: kvCache,
                            parameters: genParams, context: context
                        )
                        return (stream, SendableTokenizer(context.tokenizer))
                    }
                    let tStream = Date()
                    let tokenizer = tokenizerBox.wrapped
                    self.debug(String(format: "stream opened in %.2fs — awaiting first token…", Date().timeIntervalSince(tStream)))
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
                        let fullDecoded = tokenizer.decode(tokens: tokenIDs.map { Int($0) })
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
                            self.debug(String(format: "first token after %.2fs (%d chars)", Date().timeIntervalSince(tFirstChunk), newText.count))
                            firstChunkSeen = true
                        }
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
                           decodedSoFar.contains(toolCallClose)
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
                    self.debug(String(format: "generate() finished — %d chunks, %.2fs total", chunkIdx, Date().timeIntervalSince(t0)))
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

    private static func longestCommonPrefix(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }
}

/// `Tokenizer` is not `Sendable` — shim lets us hand the instance
/// back out of a `container.perform { ... }` closure so the caller
/// can detokenise incrementally outside the actor isolation. The
/// model container ensures the tokenizer isn't concurrently mutated
/// while generation is in flight.
import Tokenizers

private struct SendableTokenizer: @unchecked Sendable {
    let wrapped: any Tokenizer
    init(_ t: any Tokenizer) { self.wrapped = t }
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
