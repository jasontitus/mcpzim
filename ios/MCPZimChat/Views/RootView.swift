// SPDX-License-Identifier: MIT

import Foundation
import MCPZimKit
import SwiftUI
#if DEBUG
import WebKit
import LocalSwarmEngine
#endif
#if canImport(AVFoundation)
import AVFoundation
#endif

struct RootView: View {
    @Environment(ChatSession.self) private var session
    @EnvironmentObject private var swarm: ZimSwarmController
    @AppStorage("onboarding.didOfferOfflineContentV1")
    private var didOfferOfflineContent = false
    @State private var showOfflineSetup = false
    #if DEBUG
    @State private var didRunAppReviewProbe = false
    @State private var didRunSwarmProbe = false
    @State private var didRunLaunchQuestions = false
    @State private var didRunRawContextBenchmark = false
    @State private var didRunLatencyBenchmark = false
    @State private var didRunSupertonicBenchmark = false
    @State private var didRunSilentVoiceProbe = false
    #endif

    var body: some View {
        NavigationStack {
            ChatView()
                .navigationTitle("Zimfo")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        Button {
                            session.resetConversation()
                        } label: {
                            Image(systemName: "arrow.counterclockwise")
                        }
                        .accessibilityLabel("New conversation")
                        .disabled(session.messages.isEmpty || session.isGenerating)
                    }
                    ToolbarItem(placement: .primaryAction) {
                        NavigationLink { LibraryView() } label: {
                            Image(systemName: "gearshape")
                        }
                        .accessibilityLabel("Settings")
                    }
                }
                .overlay {
                    SetupOverlayView()
                }
                .task {
                    // Reconnect the background download session right away —
                    // a transfer started last session needs a live delegate
                    // to hand its finished file to, even if the user never
                    // opens the downloads UI this launch.
                    _ = ZimDownloadManager.shared
                    // Nearby sharing seeds the enabled library entries and
                    // imports received ZIMs through the normal library path
                    // (which also invalidates the prompt cache).
                    swarm.shareableFiles = { [weak session] in
                        session?.library.filter { $0.isEnabled }.map(\.url) ?? []
                    }
                    swarm.importFiles = { [weak session] urls in
                        await session?.addReaders(urls: urls)
                    }
                    // The AI model rides along in the share (single-file
                    // GGUF models only) and adopts straight into the
                    // provider's cache slot on receive.
                    swarm.shareableModelFiles = { [weak session] in
                        session?.shareableModelFiles() ?? []
                    }
                    swarm.importModelFile = { [weak session] url in
                        await session?.importSharedModelFile(at: url) ?? false
                    }
                    // Single idempotent entry point — SwiftUI can fire
                    // `.task` more than once as navigation reshapes the
                    // stack, and ChatSession.runLaunchSequence() guards
                    // against double-opening the library / double-warming
                    // the streetzim routing graph.
                    await session.runLaunchSequence()
                    if session.library.isEmpty, !didOfferOfflineContent {
                        didOfferOfflineContent = true
                        showOfflineSetup = true
                    }
                    #if DEBUG
                    #if canImport(FluidAudio)
                    await runSupertonicBenchmarkIfRequested()
                    #endif
                    if !didRunAppReviewProbe,
                       ProcessInfo.processInfo.environment["MCPZIM_APP_REVIEW_PROBE"] == "1" {
                        didRunAppReviewProbe = true
                        await AppReviewProbe.run(log: { session.debug($0, category: "AppReview") })
                    }
                    if !didRunSwarmProbe,
                       ProcessInfo.processInfo.environment["MCPZIM_SWARM_PROBE"] == "1" {
                        didRunSwarmProbe = true
                        await SwarmIntegrationProbe.run(log: { session.debug($0, category: "SwarmProbe") })
                    }
                    await runLaunchQuestionsIfRequested()
                    #if canImport(FluidAudio)
                    await runSilentVoiceProbeIfRequested()
                    #endif
                    await runRawContextBenchmarkIfRequested()
                    await runLatencyBenchmarkIfRequested()
                    #endif
                }
        }
        .sheet(isPresented: $showOfflineSetup) {
            OfflineContentSetupView()
                .environment(session)
                .environmentObject(swarm)
        }
        .onReceive(NotificationCenter.default
            .publisher(for: ZimDownloadManager.fileReadyNotification)) { note in
            // A catalog download finished while the app is frontmost — load
            // it now instead of waiting for the next launch's Documents scan.
            guard let url = note.userInfo?["url"] as? URL else { return }
            Task {
                while !session.libraryBootstrapComplete, !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                await session.addReaders(urls: [url], reloadExisting: true)
            }
        }
        .onOpenURL { url in
            guard url.pathExtension.lowercased() == "zim" else { return }
            Task {
                // A cold document-open can arrive while the normal Documents
                // scan is still running. Wait for that replace-style scan to
                // finish before appending the externally opened file.
                while !session.libraryBootstrapComplete, !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                await session.addReaders(urls: [url])
                showOfflineSetup = false
            }
        }
    }

    #if DEBUG
    #if canImport(FluidAudio)
    /// Run an established speech backend on the actual autorun answer,
    /// discarding PCM. No audio session, playback, or microphone is started.
    @MainActor
    private func runSilentVoiceProbeIfRequested() async {
        guard !didRunSilentVoiceProbe,
              ProcessInfo.processInfo.environment["MCPZIM_SILENT_TTS_PROBE"] == "1" else { return }
        didRunSilentVoiceProbe = true
        let useKokoro = ProcessInfo.processInfo.environment["MCPZIM_SILENT_TTS_BACKEND"] == "kokoro"
        let service: any TTSService
        do {
            #if canImport(KokoroSwift)
            service = useKokoro ? try KokoroTTSService(voice: "af_heart") : Supertonic3TTSService()
            #else
            service = Supertonic3TTSService()
            #endif
        } catch {
            session.debug("Silent voice initialization failed: \(error)", category: "TTSBench")
            return
        }
        guard var remaining = session.messages.first(where: {
                  $0.role == .assistant && !$0.text.isEmpty
              })?.text else {
            session.debug("Silent voice probe requires an autorun answer", category: "TTSBench")
            return
        }
        session.debug("Silent voice probe start · backend=\(service.displayName) · playback=disabled",
                      category: "TTSBench")
        let started = ProcessInfo.processInfo.systemUptime
        do {
            // Capture a single local article for parser regression analysis.
            // This optional export stays in the app cache, never an upload.
            if let articlePath = ProcessInfo.processInfo.environment["MCPZIM_EXPORT_ARTICLE_PATH"],
               let library = session.library.first(where: { $0.isEnabled && $0.kind == .wikipedia }),
               let entry = try library.reader.read(path: articlePath) {
                let output = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("article-probe.html")
                try entry.content.write(to: output, options: .atomic)
                session.debug("Article probe saved · path=\(entry.path) · bytes=\(entry.content.count)", category: "TTSBench")
            }
            try await service.prepareForConversation()
            session.debug(String(format: "Silent voice prepared · %.3fs", ProcessInfo.processInfo.systemUptime - started),
                          category: "TTSBench")
            var count = 0
            while !remaining.isEmpty, !Task.isCancelled {
                guard let prefix = StreamingSpeechPolicy.takeSpeakablePrefix(
                    remaining, generating: false, allowEarlyClause: true,
                    minimumClause: 56, maximumClause: service.preferredStreamingChunkCharacters ?? 112) else {
                    throw TTSError.synthesisFailed("Silent probe could not drain the answer.")
                }
                var spoken = prefix.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if prefix.boundary == .softWrap, let last = spoken.last, !",;:—–".contains(last) { spoken += "," }
                let chunkStarted = ProcessInfo.processInfo.systemUptime
                let result: (sampleCount: Int, audioSeconds: Double)
                if let supertonic = service as? Supertonic3TTSService {
                    result = try await supertonic.synthesizeWithoutPlayback(spoken)
                } else {
                    #if canImport(KokoroSwift)
                    guard let kokoro = service as? KokoroTTSService else {
                        throw TTSError.synthesisFailed("Unsupported silent probe backend.")
                    }
                    result = try await kokoro.synthesizeWithoutPlayback(spoken)
                    #else
                    throw TTSError.synthesisFailed("Unsupported silent probe backend.")
                    #endif
                }
                guard result.sampleCount > 0 else { throw TTSError.synthesisFailed("No PCM samples.") }
                count += 1
                session.debug(String(format:
                    "Silent voice chunk %d · chars=%d · synthesis=%.3fs · audio=%.3fs · since-start=%.3fs · samples=%d · mem=%.1f MB",
                    count, spoken.count, ProcessInfo.processInfo.systemUptime - chunkStarted,
                    result.audioSeconds, ProcessInfo.processInfo.systemUptime - started, result.sampleCount,
                    MemoryStats.physFootprintMB()), category: "TTSBench")
                remaining = String(remaining.dropFirst(prefix.consumedCharacters))
            }
            session.debug("Silent voice probe \(Task.isCancelled ? "canceled" : "complete") · chunks=\(count) · backend=\(service.displayName) · playback=disabled",
                          category: "TTSBench")
        } catch {
            session.debug("Silent voice probe failed: \(error)", category: "TTSBench")
        }
    }

    /// Cached cold-process TTS probe for physical-device lifecycle tuning.
    /// Set `MCPZIM_BENCH_SUPERTONIC=1` through devicectl; normal launches are
    /// unaffected. `speakChunk` returns when PCM is queued, separating model
    /// preparation/synthesis from playback time in the persisted log.
    @MainActor
    private func runSupertonicBenchmarkIfRequested() async {
        guard !didRunSupertonicBenchmark,
              ProcessInfo.processInfo.environment["MCPZIM_BENCH_SUPERTONIC"] == "1"
        else { return }
        didRunSupertonicBenchmark = true

        let service: any TTSService
        do {
            #if canImport(KokoroSwift)
            if ProcessInfo.processInfo.environment["MCPZIM_BENCH_TTS_BACKEND"] == "kokoro" {
                service = try KokoroTTSService(voice: "af_heart")
            } else { service = Supertonic3TTSService(voice: "F1") }
            #else
            service = Supertonic3TTSService(voice: "F1")
            #endif
        } catch {
            session.debug("Voice probe initialization failed: \(error)", category: "TTSBench")
            return
        }
        let started = ProcessInfo.processInfo.systemUptime
        session.debug(
            "Voice probe start · backend=\(service.displayName)",
            category: "TTSBench")
        do {
            #if os(iOS)
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .spokenAudio)
            try audioSession.setActive(true)
            #endif
            try await service.prepareForConversation()
            let prepared = ProcessInfo.processInfo.systemUptime
            session.debug(String(format:
                "Voice probe prepared · %.3fs · mem=%.1f MB",
                prepared - started, MemoryStats.physFootprintMB()), category: "TTSBench")

            var remaining = "George Washington (February 22, 1732 – December 14, 1799) was a Founding Father and the first president of the United States, serving from 1789 to 1797. As commander of the Continental Army, he led Patriot forces to victory in the American Revolutionary War against the British Empire."
            var chunkNumber = 0
            while !remaining.isEmpty {
                guard let prefix = StreamingSpeechPolicy.takeSpeakablePrefix(
                    remaining,
                    generating: false,
                    allowEarlyClause: true,
                    minimumClause: 56,
                    maximumClause: service.preferredStreamingChunkCharacters ?? 112)
                else { break }

                var spoken = prefix.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if prefix.boundary == .softWrap,
                   let last = spoken.last,
                   !",;:—–".contains(last) {
                    spoken += ","
                }
                let boundary: TTSChunkBoundary
                switch prefix.boundary {
                case .sentence: boundary = .sentence
                case .clause: boundary = .clause
                case .softWrap: boundary = .softWrap
                case .final: boundary = .final
                }
                chunkNumber += 1
                session.debug(
                    "Voice probe chunk \(chunkNumber) · \(prefix.boundary) · \"\(spoken)\"",
                    category: "TTSBench")
                let chunkStarted = ProcessInfo.processInfo.systemUptime
                try await service.speakChunk(spoken, boundary: boundary)
                if let metrics = service.takeStreamingChunkMetrics() {
                    session.debug(String(format: "Voice probe timing · chunk=%d · synthesis=%.3fs · gap=%.3fs · ahead=%.3fs · trimmed=%.3fs",
                        chunkNumber, ProcessInfo.processInfo.systemUptime - chunkStarted,
                        metrics.gapSeconds, metrics.queueAheadSeconds, metrics.trimmedSeconds), category: "TTSBench")
                }
                remaining = String(remaining.dropFirst(prefix.consumedCharacters))
            }
            let queued = ProcessInfo.processInfo.systemUptime
            session.debug(String(format:
                "Voice probe audio queued · chunks=%d · synthesis=%.3fs · total=%.3fs · mem=%.1f MB",
                chunkNumber, queued - prepared, queued - started, MemoryStats.physFootprintMB()),
                category: "TTSBench")

            await service.awaitPlayback()
            session.debug(String(format:
                "Voice probe playback complete · total=%.3fs · mem=%.1f MB",
                ProcessInfo.processInfo.systemUptime - started,
                MemoryStats.physFootprintMB()), category: "TTSBench")
        } catch {
            session.debug("Voice probe failed: \(error)", category: "TTSBench")
        }
    }
    #endif

    /// Deterministic physical-device performance hook. `devicectl` can pass
    /// `MCPZIM_AUTORUN_QUESTIONS` as a `||`-delimited environment value; after
    /// normal model/library setup, the app submits each question and waits for
    /// the complete response before continuing. Normal interactive launches
    /// do not set the variable, and Release builds omit this code entirely.
    @MainActor
    private func runLaunchQuestionsIfRequested() async {
        guard !didRunLaunchQuestions,
              let raw = ProcessInfo.processInfo.environment[
                "MCPZIM_AUTORUN_QUESTIONS"
              ]
        else { return }
        let questions = raw.components(separatedBy: "||")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !questions.isEmpty else { return }
        didRunLaunchQuestions = true
        #if os(iOS)
        let previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = previousIdleTimer }
        #endif
        session.debug("autorun starting \(questions.count) question(s)",
                      category: "Perf")
        for (index, question) in questions.enumerated() {
            guard !Task.isCancelled else { return }
            session.debug("autorun question \(index + 1)/\(questions.count): \(question)",
                          category: "Perf")
            session.send(question)
            var cancellationTask: Task<Void, Never>?
            if let rawDelay = ProcessInfo.processInfo.environment[
                "MCPZIM_BENCH_CANCEL_AFTER_MS"
            ], let delayMS = UInt64(rawDelay), delayMS > 0 {
                cancellationTask = Task { @MainActor in
                    do { try await Task.sleep(nanoseconds: min(delayMS, 300_000) * 1_000_000) }
                    catch { return }
                    if session.isGenerating { session.stopGeneration() }
                }
            }
            while session.isGenerating {
                do { try await Task.sleep(nanoseconds: 100_000_000) }
                catch {
                    cancellationTask?.cancel()
                    session.stopGeneration()
                    return
                }
            }
            cancellationTask?.cancel()
        }
        session.debug("autorun complete", category: "Perf")
    }

    /// One-shot physical-device context-capacity benchmark. The repeated word
    /// is deliberately tokenizer-friendly (roughly one token per repetition),
    /// allowing a devicectl launch to prove that a prompt larger than the old
    /// 8K window actually prefills and decodes. Example:
    /// MCPZIM_BENCH_RAW_PROMPT_TOKENS=9000
    @MainActor
    private func runRawContextBenchmarkIfRequested() async {
        guard !didRunRawContextBenchmark,
              let raw = ProcessInfo.processInfo.environment[
                "MCPZIM_BENCH_RAW_PROMPT_TOKENS"
              ],
              let requested = Int(raw), requested > 0,
              let provider = session.selectedModel as? LlamaCppProvider
        else { return }
        didRunRawContextBenchmark = true

        let evidence = String(repeating: " evidence", count: requested)
        let prompt = provider.formatTranscript(
            systemPreamble: "This is a synthetic context-capacity benchmark. Reply with OK.",
            turns: [ChatTurn(
                role: .user,
                text: "Read this synthetic evidence and reply with OK:\n\(evidence)")])
        let actual = provider.promptTokenCount(prompt) ?? -1
        session.debug(
            "raw context benchmark start · requested=\(requested) · actual=\(actual) tok · n_ctx=\(provider.contextTokens)",
            category: "Perf")
        guard actual > 0, actual < provider.contextTokens - 16 else {
            session.debug("raw context benchmark refused: prompt does not fit",
                          category: "Perf")
            return
        }

        let started = ProcessInfo.processInfo.systemUptime
        var output = ""
        do {
            for try await chunk in provider.generate(
                prompt: prompt,
                parameters: GenerationParameters(
                    maxTokens: 8, temperature: 0, topP: 1,
                    useModelSamplingProfile: false))
            {
                output += chunk
            }
            session.debug(String(format:
                "raw context benchmark complete · prompt=%d tok · output=%d chars · %.3fs · sample=%@",
                actual, output.count,
                ProcessInfo.processInfo.systemUptime - started,
                String(output.prefix(80))), category: "Perf")
        } catch {
            session.debug("raw context benchmark failed: \(error)",
                          category: "Perf")
        }
    }

    /// Silent, opt-in device probe. Uses the actual provider and fixed greedy
    /// prompts; never enters the microphone or speech synthesis paths.
    @MainActor
    private func runLatencyBenchmarkIfRequested() async {
        guard !didRunLatencyBenchmark,
              ProcessInfo.processInfo.environment["MCPZIM_BENCH_LATENCY"] == "1",
              let provider = session.selectedModel as? LlamaCppProvider
        else { return }
        didRunLatencyBenchmark = true
        #if os(iOS)
        let previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = previousIdleTimer }
        #endif
        let checkpoint = ProcessInfo.processInfo.environment["MCPZIM_LLAMA_RECOVERY_CHECKPOINT"] ?? "provider-default"
        session.debug("latency suite start · model=\(provider.id) · checkpoint=\(checkpoint)", category: "Perf")
        let system = "Answer from the supplied offline encyclopedia evidence. Be concise."
        let evidence = String(repeating:
            "Albert Einstein was born in Ulm in 1879. His parents were Hermann Einstein and Pauline Koch. He studied at the Swiss Federal Polytechnic in Zurich.\n", count: 32)
        let turns = [ChatTurn(role: .user,
            text: "Offline evidence:\n\(evidence)\nQuestion: Who was Albert Einstein?")]
        let prompt = provider.formatTranscript(systemPreamble: system, turns: turns)
        let params = GenerationParameters(maxTokens: 32, temperature: 0,
            topP: 1, useModelSamplingProfile: false)
        var interruptedAsExpected = false
        var referencesWereCold = true

        func generate(_ label: String, _ text: String, cancelAfterPieces: Int? = nil) async -> String? {
            let started = ProcessInfo.processInfo.systemUptime
            session.debug("latency case start · \(label)", category: "Perf")
            var output = ""
            var pieces = 0
            do {
                for try await chunk in provider.generate(prompt: text, parameters: params) {
                    output += chunk
                    pieces += 1
                    if pieces == cancelAfterPieces { provider.cancelGeneration() }
                }
                if let stats = provider.lastGenerationStats {
                    if label.hasSuffix("reference") {
                        referencesWereCold = referencesWereCold && stats.reusedTokens == 0
                    }
                    session.debug("latency case \(label) · \(stats.summaryLine)", category: "Perf")
                }
                return output.isEmpty ? nil : output
            } catch {
                if label == "interrupted", error is CancellationError,
                   pieces >= (cancelAfterPieces ?? Int.max) {
                    interruptedAsExpected = true
                }
                session.debug(String(format: "latency case %@ · stopped after %d pieces/%.3fs · %@",
                    label, pieces, ProcessInfo.processInfo.systemUptime - started,
                    String(describing: error)), category: "Perf")
                return nil
            }
        }

        provider.resetPromptCache()
        let cold = await generate("cold", prompt)
        let retry = await generate("retry", prompt)
        let followup = provider.formatTranscript(systemPreamble: system, turns: turns + [
            ChatTurn(role: .assistant, text: retry ?? ""),
            ChatTurn(role: .user, text: "What were his parents' names?")])
        let warm = await generate("append", followup)
        provider.resetPromptCache()
        let coldFollowup = await generate("append-reference", followup)
        provider.resetPromptCache()
        _ = await generate("interrupted", prompt, cancelAfterPieces: 8)
        let recovered = await generate("interrupted-retry", prompt)
        let edited = provider.formatTranscript(systemPreamble: system, turns: turns + [
            ChatTurn(role: .assistant, text: "Einstein was a physicist."),
            ChatTurn(role: .user, text: "What were his parents' names?")])
        let editedReply = await generate("edited-answer", edited)
        provider.resetPromptCache()
        let editedReference = await generate("edited-reference", edited)
        // A changed early evidence token must invalidate the saved state.
        let changedPrompt = prompt.replacingOccurrences(of: "1879", with: "1880")
        let changed = await generate("changed-evidence", changedPrompt)
        provider.resetPromptCache()
        let changedReference = await generate("changed-reference", changedPrompt)
        let passed = cold != nil && cold == retry && cold == recovered
            && warm != nil && warm == coldFollowup
            && editedReply != nil && editedReply == editedReference
            && changed != nil && changed == changedReference
            && interruptedAsExpected && referencesWereCold
        session.debug("latency correctness · retry=\(cold != nil && cold == retry) · interrupted=\(interruptedAsExpected && cold != nil && cold == recovered) · append=\(warm != nil && warm == coldFollowup) · edited=\(editedReply != nil && editedReply == editedReference) · changed-evidence=\(changed != nil && changed == changedReference) · references-cold=\(referencesWereCold) · pass=\(passed)", category: "Perf")
        if ProcessInfo.processInfo.environment["MCPZIM_BENCH_VERIFY_CURVE"] == "1" {
            do { try await provider.benchmarkVerificationWidths(prompt: prompt) }
            catch { session.debug("verify curve failed: \(error)", category: "Perf") }
        }
        provider.resetPromptCache()
        session.debug("latency suite complete", category: "Perf")
        // Optional bounded window for the next devicectl A/B launch. The
        // prior idle timer setting is restored on completion or cancellation.
        if let raw = ProcessInfo.processInfo.environment["MCPZIM_BENCH_HOLD_OPEN_SECONDS"],
           let seconds = UInt64(raw), seconds > 0 {
            try? await Task.sleep(nanoseconds: min(seconds, 1800) * 1_000_000_000)
        }
    }
    #endif
}

/// Blocking overlay shown while the one-time prompt-cache prewarm runs.
/// Dismisses itself once `session.setupState == .ready`.
struct SetupOverlayView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        Group {
            switch session.setupState {
            case .ready:
                EmptyView()
            case .failed(let msg):
                // Don't block forever on error — show a toast-style
                // banner and let the user proceed.
                VStack(spacing: 8) {
                    Text("Setup failed")
                        .font(.headline)
                    Text(msg)
                        .font(.caption)
                        .multilineTextAlignment(.center)
                    Button("Continue to settings") {
                        session.dismissSetupFailure()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(20)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                .padding(24)
            case .pending, .running:
                ZStack {
                    Color.black.opacity(0.35).ignoresSafeArea()
                    VStack(spacing: 14) {
                        if case .running(_, let progress) = session.setupState,
                           let progress {
                            ProgressView(value: progress)
                                .progressViewStyle(.linear)
                                .frame(width: 220)
                        } else {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .scaleEffect(1.3)
                        }
                        Text("Setting things up…")
                            .font(.headline)
                        Text(stageText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 260)
                    }
                    .padding(24)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
                .transition(.opacity)
            }
        }
        .animation(.default, value: stateDescription)
    }

    private var stageText: String {
        switch session.setupState {
        case .pending: return "Waiting for the model to load."
        case .running(let stage, _): return stage
        case .ready: return ""
        case .failed: return ""
        }
    }

    private var stateDescription: String {
        switch session.setupState {
        case .pending: return "pending"
        case .running(let s, _): return "running:\(s)"
        case .ready: return "ready"
        case .failed(let m): return "failed:\(m)"
        }
    }
}

#Preview {
    RootView()
        .environment(ChatSession())
        .environmentObject(ZimSwarmController())
}

#if DEBUG
/// Silent integration checks using hostile model output and archive HTML.
/// No downloads, microphone, speech or changes to the user's library.
@MainActor
private enum AppReviewProbe {
    static func run(log: (String) -> Void) async {
        var checks: [String: Bool] = [:]
        let poison = "UNSOURCED_ZEBRA_CLAIM"
        let source = "A duet is a musical composition for two performers."
        for mode in ["draft", "tool", "cancel"] {
            let reply = mode == "tool"
                ? poison + #" <tool_call>{"name":"get_article","arguments":{"path":"A/Duet"}}</tool_call>"#
                : String(repeating: poison + " ", count: mode == "cancel" ? 80 : 2)
            let model = MockProvider(scriptedResponse: reply)
            var fixture = StubZimService.Fixture()
            fixture.articleHTML[StubZimService.keyArticleSections(path: "A/Duet")] = "<p>" + source + "</p>"
            let adapter = await MCPToolAdapter.from(service: StubZimService(fixture: fixture))
            let chat = ChatSession.forTesting(providers: [model], adapter: adapter,
                                             discussionPreparationStrategy: .none)
            chat.send("Please reply.")
            var leaked = false
            let start = Date()
            var stopped = false
            while chat.isGenerating, Date().timeIntervalSince(start) < 12 {
                leaked = leaked || chat.messages.filter { $0.role == .assistant }.contains {
                    $0.text.contains(poison) || $0.modelText.contains(poison)
                    || $0.toolRoundTrips.contains { $0.assistantEmission.contains(poison) }
                }
                if mode == "cancel", model.generationCount > 0,
                   Date().timeIntervalSince(start) > 0.2, !stopped {
                    stopped = true
                    chat.stopGeneration()
                }
                try? await Task.sleep(nanoseconds: 5_000_000)
            }
            let answer = chat.messages.last
            checks["source_" + mode] = model.generationCount > 0 && !chat.isGenerating && !leaked
                && !(answer?.modelText.contains(poison) ?? true)
                && !(answer?.text.isEmpty ?? true)
                && (mode != "tool" || answer?.toolCalls.contains { $0.name == "get_article" && $0.succeeded } == true)
                && (mode != "cancel" || (stopped && answer?.text == "Stopped."))
            if chat.isGenerating { chat.stopGeneration() }
            log("source \(mode): \(checks["source_" + mode] == true ? "PASS" : "FAIL") · calls=\(model.generationCount) · answer=\(answer?.text ?? "<none>")")
        }

        let manager = ZimDownloadManager(restoringDownloads: false)
        let id = "review-" + UUID().uuidString
        let label = TaskLabel(id: id, title: "Probe", urlString: "https://example.com/probe.zim", expectedBytes: 80)
        manager.adoptRestoredTasks([(label, 9001, 0, 80, true)])
        let initiallyActive = manager.hasActiveDownloads
        manager.cancel(id: id)
        manager.progress(taskID: 9001, label: label, received: 50, expected: 80)
        manager.adoptRestoredTasks([(label, 9001, 50, 80, true)])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let old = directory.appendingPathComponent("old.zim")
            let staged = directory.appendingPathComponent("staged.partial")
            try Data("previous archive".utf8).write(to: old)
            try Data("late download".utf8).write(to: staged)
            manager.finished(taskID: 9001, label: label, staged: staged, destination: old)
            let preserved = try Data(contentsOf: old)
            checks["download_cancel"] = initiallyActive && manager.items.isEmpty && !manager.hasActiveDownloads
                && preserved == Data("previous archive".utf8)
                && !FileManager.default.fileExists(atPath: staged.path)
        } catch { checks["download_cancel"] = false }

        let reader = ReviewWebReader()
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        var invokedAction = false
        ZimUserActionBridge.install(on: configuration) { _ in invokedAction = true }
        let handler = ZimURLSchemeHandler(lookup: { _ in reader })
        configuration.setURLSchemeHandler(handler, forURLScheme: "zim")
        let web = WKWebView(frame: .zero, configuration: configuration)
        web.load(URLRequest(url: URL(string: "zim://probe.zim/index.html")!))
        var webResult: [String: Any] = [:]
        for _ in 0..<100 {
            try? await Task.sleep(nanoseconds: 50_000_000)
            if let value = try? await web.evaluateJavaScript("window.reviewProbe"),
               let result = value as? [String: Any] {
                webResult = result
                if result["local"] as? Bool == true, result["remote"] as? String == "blocked",
                   (result["violations"] as? Int ?? 0) >= 3 { break }
            }
        }
        checks["web_offline"] = webResult["local"] as? Bool == true
            && webResult["remote"] as? String == "blocked"
            && (webResult["violations"] as? Int ?? 0) >= 3
        checks["web_action_gate"] = webResult["bridgeHidden"] as? Bool == true && !invokedAction
        // Positive control: only the app can evaluate in this isolated
        // world. The page-world negative test above cannot reach its handler.
        _ = try? await web.callAsyncJavaScript(
            "window.webkit.messageHandlers.zimfoTrustedAction.postMessage({action:'directions'});",
            arguments: [:], in: nil, contentWorld: ZimUserActionBridge.world)
        try? await Task.sleep(nanoseconds: 50_000_000)
        checks["web_action_positive"] = invokedAction
        web.stopLoading()
        log("web policy: \(webResult)")
        log("checks: \(checks.sorted { $0.key < $1.key })")
        log("ALL CHECKS \(checks.count == 7 && checks.values.allSatisfy { $0 } ? "PASS" : "FAIL")")
    }
}

private final class ReviewWebReader: ZimReader, @unchecked Sendable {
    let metadata = ZimMetadata(title: "Offline policy fixture")
    let kind: ZimKind = .generic
    let hasFullTextIndex = false
    let hasTitleIndex = false
    let hasRoutingData = false
    func readMainPage() throws -> ZimEntry? { try read(path: "index.html") }
    func read(path: String) throws -> ZimEntry? {
        let html = """
        <!doctype html><html><head></head><body><script>
        window.reviewProbe={local:false,remote:'pending',violations:0};
        reviewProbe.bridgeHidden = !window.webkit?.messageHandlers?.zimfoTrustedAction;
        let b=document.createElement('button');
        b.setAttribute('data-zimfo-action',JSON.stringify({action:'directions',lat:0,lon:0}));
        document.body.append(b);b.click();
        document.addEventListener('securitypolicyviolation',()=>reviewProbe.violations++);
        fetch('zim://probe.zim/data').then(r=>r.text()).then(t=>reviewProbe.local=t==='local');
        fetch('https://example.com/zimfo-offline-probe').then(()=>reviewProbe.remote='escaped',()=>reviewProbe.remote='blocked');
        let i=new Image();i.src='https://example.com/zimfo-offline-image';document.body.append(i);
        let s=document.createElement('script');s.src='https://example.com/zimfo-offline-script';document.body.append(s);
        </script></body></html>
        """
        return ZimEntry(path: path, title: "Probe", mimetype: path == "data" ? "text/plain" : "text/html",
                        content: Data((path == "data" ? "local" : html).utf8))
    }
}
#endif

#if DEBUG
/// Exercises the same public manager used by Nearby Sharing, using only
/// generated temporary fixtures. Never shares the user's library or models.
@MainActor
private enum SwarmIntegrationProbe {
    static func until(seconds: Double = 30, _ condition: () -> Bool) async -> Bool {
        let end = Date().addingTimeInterval(seconds)
        while !condition(), Date() < end, !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        return condition()
    }

    static func run(log: (String) -> Void) async {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("zimfo-swarm-probe-" + UUID().uuidString)
        let host = SwarmManager(transport: .quic, downloadDirectory: root.appendingPathComponent("host"))
        var receivers: [SwarmManager] = []
        var checks: [String: Bool] = [:]
        SleepBlocker.set(true, reason: "swarm-probe")
        defer {
            host.stopHosting()
            host.stopDiscovery()
            for receiver in receivers {
                for transfer in receiver.transfers { receiver.cancelDownload(swarmID: transfer.swarmID) }
                receiver.stopDiscovery()
            }
            SleepBlocker.set(false, reason: "swarm-probe")
        }
        do {
            let sources = root.appendingPathComponent("source")
            let folder = sources.appendingPathComponent("voice-fixture")
            try fm.createDirectory(at: folder.appendingPathComponent("nested"), withIntermediateDirectories: true)
            let fixtures: [String: Data] = [
                "archive-fixture.bin": Data(repeating: 0x37, count: 3 * 1024 * 1024 + 17),
                "voice-fixture/voices.npz": Data(repeating: 0x91, count: 24577),
                "voice-fixture/nested/weights.bin": Data(repeating: 0xc3, count: 65539)
            ]
            for (relative, data) in fixtures { try data.write(to: sources.appendingPathComponent(relative)) }
            let pin = UUID().uuidString
            host.hostFiles(at: [sources.appendingPathComponent("archive-fixture.bin"), folder],
                           name: "Zimfo generated test fixture", pin: pin)
            guard await until({ !host.hostedManifests.isEmpty || host.lastError != nil }),
                  let manifest = host.hostedManifests.first else {
                throw NSError(domain: "SwarmProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: host.lastError ?? "Hosting timeout"])
            }
            checks["mixed_layout"] = Set(manifest.files.map(\.path)) == Set(fixtures.keys)
            log("host ready: \(manifest.files.count) files, \(manifest.totalBytes) bytes")
            for transport in [Transport.quic, .tcp] {
                let receiver = SwarmManager(transport: transport,
                    downloadDirectory: root.appendingPathComponent("receive-" + transport.rawValue))
                receivers.append(receiver)
                receiver.startDiscovery()
                guard await until({ receiver.discoveredSwarms.contains {
                    $0.swarmID == manifest.swarmID && $0.availableTransports.contains(transport)
                } }), let swarm = receiver.discoveredSwarms.first(where: { $0.swarmID == manifest.swarmID }) else {
                    throw NSError(domain: "SwarmProbe", code: 2, userInfo: [NSLocalizedDescriptionKey: "Discovery timeout: " + receiver.diagnostics.joined(separator: "; ")])
                }
                if transport == .quic {
                    checks["wrong_pin_rejected"] = await receiver.fetchManifest(for: swarm, pin: "wrong-" + pin) == nil
                    receiver.clearError()
                }
                var completed: [URL]?
                receiver.onDownloadComplete = { id, urls in
                    if id == manifest.swarmID { completed = urls }
                }
                let start = Date()
                if transport == .tcp {
                    // Pause before the queued store/session creation finishes;
                    // resume must retain the PIN and retire the old callbacks.
                    receiver.startDownload(manifest: manifest, selecting: [], from: swarm, pin: pin)
                    receiver.pauseDownload(swarmID: manifest.swarmID)
                    checks["startup_pause"] = receiver.transfers.first?.role == .paused
                    receiver.resumeDownload(swarmID: manifest.swarmID)
                } else {
                    receiver.receive(swarm, pin: pin)
                }
                let done = await until(seconds: 60, { completed != nil || receiver.lastError != nil })
                guard done, completed != nil else {
                    throw NSError(domain: "SwarmProbe", code: 3, userInfo: [NSLocalizedDescriptionKey:
                        "Transfer " + transport.rawValue + ": " + (receiver.lastError ?? "timeout")])
                }
                let destination = root.appendingPathComponent("receive-" + transport.rawValue)
                    .appendingPathComponent(manifest.swarmID)
                let identical = try fixtures.allSatisfy { relative, data in
                    try Data(contentsOf: destination.appendingPathComponent(relative)) == data
                }
                checks["transfer_" + transport.rawValue] = done && completed?.count == fixtures.count && identical
                log("\(transport.rawValue) transfer: \(checks["transfer_" + transport.rawValue] == true ? "PASS" : "FAIL") · \(String(format: "%.3f", Date().timeIntervalSince(start)))s · byte equality=\(identical)")
                receiver.cancelDownload(swarmID: manifest.swarmID)
                receiver.stopDiscovery()
            }
            host.stopHosting()
            // Let stopped sessions finish their disk checkpoint before cleanup.
            try? await Task.sleep(nanoseconds: 500_000_000)
            try fm.removeItem(at: root)
            checks["fixture_cleanup"] = !fm.fileExists(atPath: root.path)
        } catch { log("FAIL: \(error)") }
        for key in checks.keys.sorted() { log("CHECK \(key)=\(checks[key] == true ? "PASS" : "FAIL")") }
        log("RESULT \(checks.count == 6 && checks.values.allSatisfy { $0 } ? "PASS" : "FAIL") · \(checks.count) checks · no audio")
    }
}
#endif
