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
    public let approximateMemoryMB = 400        // per Swift-Gemma4-Core benchmarks.
    public let supportsToolCalls = true

    private let modelId = Gemma4SwiftCore.verifiedModelId
    private var container: ModelContainer?
    private var state: ModelLoadState = .notLoaded
    private var continuations: [AsyncStream<ModelLoadState>.Continuation] = []
    private let queue = DispatchQueue(label: "gemma4.state")

    // In-window debug log sink. Set by ChatSession when it constructs the
    // provider so logs bubble up into the UI debug pane.
    public var debugSink: (@Sendable (String) -> Void)?

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
                    // Tokenize the caller-supplied prompt verbatim — ChatSession
                    // has already applied Gemma-4's `<start_of_turn>…<end_of_turn>`
                    // template via `Gemma4PromptTemplate.render(…)`, so we must
                    // NOT apply `Gemma4PromptFormatter.userTurn(…)` a second time.
                    let tokens = await container.encode(prompt)
                    self.debug(String(format: "encoded %d tokens in %.2fs", tokens.count, Date().timeIntervalSince(t0)))
                    let input = LMInput(tokens: MLXArray(tokens))
                    let tStream = Date()
                    // `.chunk(String)` is the streaming-text event; `.info` carries
                    // timing/metrics and `.toolCall` is MLX's structured tool-call
                    // event. We parse tool calls out of the text stream ourselves
                    // (see `ChatToolCallParser`), so we ignore the other cases.
                    // NOTE: `kvBits: 4` triggers `QuantizedKVCache` in MLX,
                    // but Swift-Gemma4-Core 0.1.0's attention layer calls the
                    // non-quantized `cache.update(keys:values:)` codepath,
                    // which `QuantizedKVCache` rejects with a runtime trap
                    // (KVCache.swift:894). Keep the cache unquantized until
                    // Gemma4SwiftCore teaches its attention layer about
                    // `updateQuantized`. See ios/OPTIMIZATIONS.md for
                    // the watch-list entry.
                    // `prefillStepSize` caps how many prompt tokens MLX
                    // activations hold at once during prefill. The default
                    // (512) drives a ~3 GB transient spike on Gemma 4B
                    // which pushes past the 6144 MB `increased-memory-limit`
                    // cap on iPhone 17 Pro Max. 128 keeps the spike under
                    // ~1 GB while only costing a few % in prefill throughput.
                    let stream = try await container.generate(
                        input: input,
                        parameters: GenerateParameters(
                            maxTokens: parameters.maxTokens,
                            temperature: Float(parameters.temperature),
                            topP: Float(parameters.topP),
                            prefillStepSize: 128
                        )
                    )
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
                    chunkLoop: for await event in stream {
                        guard case .chunk(let s) = event else { continue }
                        if !firstChunkSeen {
                            self.debug(String(format: "first token after %.2fs (%d chars)", Date().timeIntervalSince(tFirstChunk), s.count))
                            firstChunkSeen = true
                        }
                        chunkIdx += 1
                        if chunkIdx.isMultiple(of: 40) {
                            self.debug("streaming · \(chunkIdx) chunks")
                        }
                        pending += s
                        if let hit = stopMarkers.compactMap({ pending.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
                            let clean = String(pending[..<hit.lowerBound])
                            if !clean.isEmpty { continuation.yield(clean) }
                            pending = ""
                            self.debug("stop marker hit — halting stream")
                            break chunkLoop
                        }
                        // Keep the last (maxMarker-1) chars buffered in case a
                        // marker is split across chunks.
                        if pending.count > maxMarker {
                            let flushEnd = pending.index(pending.endIndex, offsetBy: -(maxMarker - 1))
                            continuation.yield(String(pending[..<flushEnd]))
                            pending = String(pending[flushEnd...])
                        }
                    }
                    if !pending.isEmpty { continuation.yield(pending) }
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
