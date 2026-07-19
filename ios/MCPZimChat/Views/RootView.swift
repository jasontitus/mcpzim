// SPDX-License-Identifier: MIT

import Foundation
import MCPZimKit
import SwiftUI
#if canImport(AVFoundation)
import AVFoundation
#endif

struct RootView: View {
    @Environment(ChatSession.self) private var session
    @AppStorage("onboarding.didOfferOfflineContentV1")
    private var didOfferOfflineContent = false
    @State private var showOfflineSetup = false
    #if DEBUG
    @State private var didRunLaunchQuestions = false
    @State private var didRunRawContextBenchmark = false
    @State private var didRunSupertonicBenchmark = false
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
                    await runLaunchQuestionsIfRequested()
                    await runRawContextBenchmarkIfRequested()
                    #endif
                }
        }
        .sheet(isPresented: $showOfflineSetup) {
            OfflineContentSetupView()
                .environment(session)
        }
        .onOpenURL { url in
            guard url.pathExtension.lowercased() == "zim" else { return }
            Task {
                // A cold document-open can arrive while the normal Documents
                // scan is still running. Wait for that replace-style scan to
                // finish before appending the externally opened file.
                while !session.libraryBootstrapComplete {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                await session.addReaders(urls: [url])
                showOfflineSetup = false
            }
        }
    }

    #if DEBUG
    #if canImport(FluidAudio)
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

        let service = Supertonic3TTSService(voice: "F1")
        let started = ProcessInfo.processInfo.systemUptime
        session.debug(
            "Supertonic probe start · backend=\(service.displayName) · voice=\(service.voiceName)",
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
                "Supertonic probe prepared · %.3fs · mem=%.1f MB",
                prepared - started, MemoryStats.physFootprintMB()), category: "TTSBench")

            var remaining = "George Washington (February 22, 1732 – December 14, 1799) was a Founding Father and the first president of the United States, serving from 1789 to 1797. As commander of the Continental Army, he led Patriot forces to victory in the American Revolutionary War against the British Empire."
            var chunkNumber = 0
            while !remaining.isEmpty {
                guard let prefix = StreamingSpeechPolicy.takeSpeakablePrefix(
                    remaining,
                    generating: false,
                    allowEarlyClause: true,
                    minimumClause: 56,
                    maximumClause: service.preferredStreamingChunkCharacters ?? 94)
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
                    "Supertonic probe chunk \(chunkNumber) · \(prefix.boundary) · \"\(spoken)\"",
                    category: "TTSBench")
                try await service.speakChunk(spoken, boundary: boundary)
                remaining = String(remaining.dropFirst(prefix.consumedCharacters))
            }
            let queued = ProcessInfo.processInfo.systemUptime
            session.debug(String(format:
                "Supertonic probe audio queued · chunks=%d · synthesis=%.3fs · total=%.3fs · mem=%.1f MB",
                chunkNumber, queued - prepared, queued - started, MemoryStats.physFootprintMB()),
                category: "TTSBench")

            await service.awaitPlayback()
            session.debug(String(format:
                "Supertonic probe playback complete · total=%.3fs · mem=%.1f MB",
                ProcessInfo.processInfo.systemUptime - started,
                MemoryStats.physFootprintMB()), category: "TTSBench")
        } catch {
            session.debug("Supertonic probe failed: \(error)", category: "TTSBench")
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
        session.debug("autorun starting \(questions.count) question(s)",
                      category: "Perf")
        for (index, question) in questions.enumerated() {
            session.debug("autorun question \(index + 1)/\(questions.count): \(question)",
                          category: "Perf")
            session.send(question)
            if let rawDelay = ProcessInfo.processInfo.environment[
                "MCPZIM_BENCH_CANCEL_AFTER_MS"
            ], let delayMS = UInt64(rawDelay), delayMS > 0 {
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: delayMS * 1_000_000)
                    if session.isGenerating { session.stopGeneration() }
                }
            }
            while session.isGenerating {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
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
                    maxTokens: 8, temperature: 0, topP: 1))
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
    RootView().environment(ChatSession())
}
