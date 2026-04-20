// SPDX-License-Identifier: MIT
//
// Apple Foundation Models (Apple Intelligence on-device LLM) provider.
//
// Design:
//   - Gate the real implementation on `#if canImport(FoundationModels)`
//     and `@available(macOS 26.0, iOS 19.0, *)` so the app still builds
//     on older SDKs / platforms (the stub reports `.failed`).
//   - Reuse the generic `<|role|>…` template inherited from
//     ModelProvider — Apple's framework does its own templating
//     internally, but we pass the already-rendered transcript so the
//     tool preamble and turn history survive intact.
//   - Parse `<tool_call>…</tool_call>` out of the streamed text ourselves
//     (ChatSession does this for every provider). We do NOT use Apple's
//     structured Tool protocol, because the MCPZim tool surface is
//     transport-agnostic and already shared with Gemma.

import Foundation
import MCPZimKit

#if canImport(FoundationModels)
import FoundationModels

@available(macOS 26.0, iOS 19.0, *)
public final class FoundationModelsProvider: ModelProvider, @unchecked Sendable {
    /// Two integration strategies live behind the same class. The
    /// text-loop flavor (`useNativeTools = false`) is the default and
    /// goes through ChatSession's `<|tool_call|{json}>` parsing — same
    /// path every other provider uses. The native-tools flavor plumbs
    /// Apple's structured `Tool` + `@Generable` protocol so a single
    /// warmed session can handle multiple tool iterations without
    /// restarting (ChatSession steps out of the way when it sees this
    /// provider already handled tools internally).
    public let useNativeTools: Bool
    public let id: String
    public let displayName: String
    /// Apple Intelligence runs in a system-managed inference process,
    /// so the app itself only pays a modest RSS cost for the session
    /// handles — not the model weights.
    public let approximateMemoryMB = 50
    public let supportsToolCalls = true

    /// Tools built from the live `ZimService` by ChatSession. Used
    /// only when `useNativeTools` is true.
    private var nativeTools: [any Tool] = []

    /// Instructions (system preamble) last installed on the native
    /// session. Tracked so we can detect when it changes and rebuild.
    private var nativeInstructions: String = ""

    /// Persistent `LanguageModelSession` reused across every turn of
    /// a conversation on the native-tools path. Apple's session is
    /// designed as a stateful conversation — pass `instructions:` and
    /// `tools:` once at construction, call `streamResponse(to:)` with
    /// ONLY the new user message each turn, and let the framework
    /// own the transcript. Tool calls round-trip through `Tool.call()`
    /// inside a single `streamResponse`, so we never break mid-stream.
    /// Fresh session per-turn collided with the 4,096-token ceiling
    /// (re-sending the rendered transcript doubled the context) and
    /// discarded the KV cache we paid for every call.
    private var warmSession: LanguageModelSession?

    /// We don't reuse a single `LanguageModelSession` across generate
    /// calls. Apple's session serializes all `respond(to:)` /
    /// `streamResponse(to:)` invocations and surfaces a `concurrentRequests`
    /// programmer-error if a second one fires before the first fully
    /// completes. Our ChatSession tool loop legitimately interrupts the
    /// stream when a `<tool_call>…` block arrives, so the safe pattern
    /// is a fresh session per turn. Context isn't lost because
    /// ChatSession re-renders the full transcript into every prompt.
    /// `session` stays nil except to signal "framework available" after
    /// load().
    private var session: LanguageModelSession?
    private var state: ModelLoadState = .notLoaded
    private var continuations: [AsyncStream<ModelLoadState>.Continuation] = []
    private let queue = DispatchQueue(label: "foundation.state")

    /// Debug pane sink, mirroring the Gemma provider pattern.
    public var debugSink: (@Sendable (String) -> Void)?

    public init(useNativeTools: Bool = false) {
        self.useNativeTools = useNativeTools
        self.id = useNativeTools ? "apple-foundation-models-native" : "apple-foundation-models"
        self.displayName = useNativeTools
            ? "Apple Foundation Models (native tools)"
            : "Apple Foundation Models"
    }

    /// Fire a best-effort `prewarm()` on the warm session. Called
    /// when the composer text field gains focus — we want the
    /// framework to start reloading model assets before the user
    /// finishes typing, since Apple's daemon tends to unload the
    /// model on idle. No-op when the session is currently responding
    /// (prewarm while busy is pointless and the framework would
    /// throw).
    public func prewarmIfIdle() {
        guard useNativeTools else { return }
        guard let warm = ensureWarmSession() else { return }
        if warm.isResponding { return }
        warm.prewarm()
        debug("prewarmIfIdle: dispatched prewarm() on warm session")
    }

    /// Called by ChatSession when the library (and therefore the
    /// service-backed tool implementations) is rebuilt. Replaces the
    /// current tool set and drops the warm session — Apple's session
    /// locks in its tool set at construction so any change needs a
    /// new session (and thus a new conversation).
    public func installNativeTools(_ tools: [any Tool]) {
        self.nativeTools = tools
        dropWarmSession(reason: "tools changed")
        debug("installed \(tools.count) native tool(s)")
    }

    /// Set the conversational preamble. Used as the session's
    /// `instructions:` so the framework caches those tokens once per
    /// session instead of re-prefilling them every turn. ChatSession
    /// calls this before each turn; if the instructions string
    /// matches the live session's we skip the rebuild, so typical
    /// calls are zero-cost.
    public func setNativeInstructions(_ text: String) {
        guard useNativeTools else { return }
        if text == nativeInstructions, warmSession != nil { return }
        nativeInstructions = text
        dropWarmSession(reason: "instructions changed")
    }

    /// Drop the warm session. The next turn will open a fresh one,
    /// losing accumulated transcript history — callers use this on
    /// conversation reset or tool/instructions changes.
    public func resetNativeConversation() {
        guard useNativeTools else { return }
        dropWarmSession(reason: "conversation reset")
    }

    private func dropWarmSession(reason: String) {
        if warmSession != nil {
            debug("dropping warm session — \(reason)")
        }
        warmSession = nil
    }

    /// Build (and prewarm) the warm session on demand. Returns nil
    /// when the framework isn't loaded or no tools are installed.
    private func ensureWarmSession() -> LanguageModelSession? {
        if let existing = warmSession { return existing }
        guard useNativeTools, session != nil, !nativeTools.isEmpty else { return nil }
        let built: LanguageModelSession
        if nativeInstructions.isEmpty {
            built = LanguageModelSession(tools: nativeTools)
        } else {
            built = LanguageModelSession(tools: nativeTools, instructions: nativeInstructions)
        }
        warmSession = built
        let t0 = Date()
        built.prewarm()
        debug(String(format: "warm session built + prewarm() dispatched in %.3fs (tools=%d, instructions=%d chars)",
                     Date().timeIntervalSince(t0),
                     nativeTools.count,
                     nativeInstructions.count))
        return built
    }

    /// Send ONE user turn to the persistent session and stream the
    /// response. Apple's framework dispatches any tool calls
    /// internally via `Tool.call()`, so the caller doesn't see a
    /// text-level tool loop — a single stream consumes the whole
    /// turn, including tool round-trips.
    public func generateNativeTurn(
        userMessage: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            guard let session = self.ensureWarmSession() else {
                continuation.finish(throwing: ModelError.notLoaded)
                return
            }
            let task = Task {
                do {
                    self.debug("generateNativeTurn userMessage=\(userMessage.count) chars")
                    let t0 = Date()
                    let options = GenerationOptions(
                        sampling: .greedy,
                        maximumResponseTokens: parameters.maxTokens
                    )
                    let stream = session.streamResponse(to: userMessage, options: options)
                    var lastText = ""
                    var firstChunkSeen = false
                    var chunkIdx = 0
                    for try await partial in stream {
                        try Task.checkCancellation()
                        let current = partial.content
                        if current == lastText { continue }
                        if !firstChunkSeen {
                            self.debug(String(format: "first token after %.2fs", Date().timeIntervalSince(t0)))
                            firstChunkSeen = true
                        }
                        chunkIdx += 1
                        if chunkIdx.isMultiple(of: 40) {
                            self.debug("streaming · \(chunkIdx) chunks")
                        }
                        let delta: String
                        if current.hasPrefix(lastText) {
                            delta = String(current.dropFirst(lastText.count))
                        } else {
                            delta = current
                        }
                        lastText = current
                        if !delta.isEmpty { continuation.yield(delta) }
                    }
                    self.debug(String(format: "generateNativeTurn finished — %d chunks, %.2fs total",
                                       chunkIdx, Date().timeIntervalSince(t0)))
                    continuation.finish()
                } catch is CancellationError {
                    self.debug("generateNativeTurn cancelled")
                    continuation.finish()
                } catch {
                    self.debug("generateNativeTurn threw: \(error)")
                    // A fatal session error (e.g. context overflow
                    // still leaking through, or schema mismatch) can
                    // poison the warm session; drop it so the next
                    // turn rebuilds clean.
                    self.dropWarmSession(reason: "error: \(error)")
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func debug(_ s: String) {
        debugSink?(s)
        print("[AppleFM] \(s)")
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
        if session != nil {
            debug("load() no-op (session already created)")
            return
        }
        set(.loading)
        debug("load() start — checking Apple Intelligence availability…")
        switch SystemLanguageModel.default.availability {
        case .available:
            // An empty instructions string is fine — ChatSession bakes
            // the system preamble into the rendered transcript we pass
            // to `generate`, and we don't want two system messages.
            session = LanguageModelSession()
            debug("session ready")
            set(.ready)
        case .unavailable(let reason):
            let msg = "Apple Foundation Models unavailable: \(String(describing: reason))"
            debug(msg)
            set(.failed(msg))
            throw ModelError.notLoaded
        @unknown default:
            let msg = "Apple Foundation Models: unknown availability case"
            debug(msg)
            set(.failed(msg))
            throw ModelError.notLoaded
        }
    }

    public func unload() async {
        session = nil
        warmSession = nil
        set(.notLoaded)
    }

    public func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            guard session != nil else {
                continuation.finish(throwing: ModelError.notLoaded)
                return
            }
            let task = Task {
                do {
                    self.debug("generate() prompt=\(prompt.count) chars, maxTokens=\(parameters.maxTokens)")
                    let t0 = Date()
                    // Greedy sampling + tight token budget: for
                    // tool-augmented conversations, deterministic
                    // greedy decode is both faster (no sampling
                    // arithmetic per token) and more predictable. The
                    // caller's `parameters.temperature` is ignored.
                    let options = GenerationOptions(
                        sampling: .greedy,
                        maximumResponseTokens: parameters.maxTokens
                    )
                    // Session selection:
                    //   • text-loop: fresh session per call to
                    //     sidestep `concurrentRequests` if ChatSession
                    //     breaks out mid-stream on a tool marker.
                    //   • native-tools: reuse the warm session the
                    //     provider prewarmed at install/load time.
                    //     The framework dispatches tools inside one
                    //     streamResponse call, so no break-out is
                    //     needed and the KV cache stays hot across
                    //     turns (~15–24 s TTFT → single-digit).
                    let perCall: LanguageModelSession
                    let tInit = Date()
                    // Fresh session per call on both paths. Warm-session
                    // reuse was tempting for the native-tools path
                    // but collided with Apple's 4,096-token ceiling:
                    // the session retains its own transcript across
                    // streamResponse calls, so re-sending the full
                    // rendered prompt on turn 2 doubled the context
                    // and tripped `exceededContextWindowSize`. We
                    // still benefit from the `prewarm()` fired at
                    // install time (model weights stay resident in
                    // the system daemon) — just not from KV-cache
                    // reuse. Properly solving that needs a prompt
                    // refactor that passes only the new user turn
                    // per call and uses `instructions` for the
                    // preamble; filed for later.
                    if self.useNativeTools {
                        perCall = LanguageModelSession(tools: self.nativeTools)
                        self.debug(String(format: "native-tools session init: %.2fs (tools=%d)",
                                           Date().timeIntervalSince(tInit),
                                           self.nativeTools.count))
                    } else {
                        perCall = LanguageModelSession()
                        self.debug(String(format: "session init: %.2fs",
                                           Date().timeIntervalSince(tInit)))
                    }
                    let tStreamOpen = Date()
                    let stream = perCall.streamResponse(
                        to: prompt,
                        options: options
                    )
                    self.debug(String(format: "streamResponse(to:) returned in %.2fs — awaiting first partial…",
                                       Date().timeIntervalSince(tStreamOpen)))
                    var lastText = ""
                    var firstChunkSeen = false
                    var chunkIdx = 0
                    for try await partial in stream {
                        try Task.checkCancellation()
                        // `partial` yields the full accumulated text so
                        // far. Diff against the last yield to produce
                        // incremental chunks for downstream tool-call
                        // parsing.
                        let current = partial.content
                        if current == lastText { continue }
                        if !firstChunkSeen {
                            self.debug(String(format: "first token after %.2fs", Date().timeIntervalSince(t0)))
                            firstChunkSeen = true
                        }
                        chunkIdx += 1
                        if chunkIdx.isMultiple(of: 40) {
                            self.debug("streaming · \(chunkIdx) chunks")
                        }
                        let delta: String
                        if current.hasPrefix(lastText) {
                            delta = String(current.dropFirst(lastText.count))
                        } else {
                            // Rare: framework emitted a revised fragment.
                            // Fall back to yielding the full current text
                            // as a correction — downstream consumers
                            // tolerate duplicates.
                            delta = current
                        }
                        lastText = current
                        if !delta.isEmpty { continuation.yield(delta) }
                    }
                    self.debug(String(format: "generate() finished — %d chunks, %.2fs total",
                                       chunkIdx, Date().timeIntervalSince(t0)))
                    continuation.finish()
                } catch is CancellationError {
                    self.debug("generate() cancelled")
                    continuation.finish()
                } catch {
                    self.debug("generate() threw: \(error)")
                    continuation.finish(throwing: error)
                }
            }
            // When ChatSession detects a tool call and stops consuming,
            // the continuation terminates — cancel the in-flight
            // respond() so Apple's framework releases the serial slot
            // before the next `generate()` tries to open one. Without
            // this we trip `concurrentRequests`.
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    // Inherits the generic `<|role|>…` template from ModelProvider —
    // Apple's framework handles its own native templating, but we rely
    // on the caller-rendered transcript to preserve tool declarations
    // and tool responses across turns.
}

#else

/// Fallback for SDKs without FoundationModels — keeps the app building.
public final class FoundationModelsProvider: ModelProvider, @unchecked Sendable {
    public let id = "apple-foundation-models"
    public let displayName = "Apple Foundation Models — not available"
    public let approximateMemoryMB = 0
    public let supportsToolCalls = false
    /// No-op sink so call sites share a type between SDKs. On the real
    /// impl it routes debug lines into ChatSession's pane.
    public var debugSink: (@Sendable (String) -> Void)?

    public init() {}

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            cont.yield(.failed("FoundationModels framework not available on this SDK."))
            cont.finish()
        }
    }
    public func load() async throws {
        throw ModelError.notLoaded
    }
    public func unload() async {}
    public func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { cont in
            cont.finish(throwing: ModelError.notLoaded)
        }
    }
}

#endif
