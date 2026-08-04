// SPDX-License-Identifier: MIT
//
// Voice-chat coordinator. Wires three pieces together:
//
//   1. `AVAudioEngine` mic tap → `SpeechRecognizerService` (Apple's
//      best on-device STT, see `SpeechRecognizerService.swift`).
//   2. End-of-utterance detection (silence VAD) → `ChatSession.send(_:)`.
//   3. Stream stable assistant phrases into `TTSService` while the model is
//      generating, then loop back to listening after playback drains.
//
// State machine (also surfaced to the UI):
//
//     idle ─▶ listening ─▶ thinking ─▶ speaking ─▶ listening ─▶ …
//
// Pressing the mic toggles between idle and listening; the rest of
// the loop runs automatically until the user taps stop or the audio
// session is interrupted by another app.

import AVFoundation
import Foundation
import MCPZimKit
import Observation
#if canImport(UIKit)
import UIKit
#endif

@MainActor
@Observable
public final class VoiceChatController {
    // TCC can briefly continue reporting `.notDetermined` after its callback
    // says the user granted access. A newly-created voice sheet must reuse the
    // result from that callback instead of presenting the same system prompt
    // again. Keeping this process-wide also coalesces simultaneous starts.
    private static var microphonePermissionRequestTask: Task<Bool, Never>?
    private static var microphonePermissionResultThisProcess: Bool?

    public enum State: Equatable, Sendable {
        case idle
        case starting
        case listening
        case thinking
        case speaking
        case error(String)
    }

    public private(set) var state: State = .idle
    /// Live partial transcript shown to the user under the orb so they
    /// can see the recognizer is hearing them in real time.
    public private(set) var liveTranscript: String = ""
    /// Approximate microphone level, 0…1, smoothed for the orb's pulse
    /// animation. RMS over the most recent buffer.
    public private(set) var inputLevel: Float = 0

    public let stt: SpeechRecognizerService
    /// Creating Kokoro/Supertonic also creates a playback AVAudioEngine. On
    /// some Macs that graph changed the shared device format before capture
    /// had delivered its first buffer, leaving the microphone apparently
    /// started but silent for several seconds. Keep TTS genuinely lazy and
    /// construct it only after the user's utterance has been submitted.
    @ObservationIgnored private var ttsStorage: TTSService?
    public var tts: TTSService {
        if let ttsStorage { return ttsStorage }
        let service = TTSFactory.makeBest(
            voice: KokoroVoicePreference.current)
        ttsStorage = service
        log("TTS backend initialized after capture — \(service.displayName)")
        return service
    }
    /// Swap a too-heavy synthesis backend for the always-affordable system
    /// voice when the device can't spare its peak. Called once per reply,
    /// after `tts` is realized. On macOS `availableMemoryMB` is 0 (no jetsam
    /// cap) so the gate is skipped and the chosen backend stands.
    private func ensureAffordableTTS() {
        guard let current = ttsStorage, !(current is SystemTTSService) else { return }
        let available = Self.availableMemoryMB()
        guard available > 0 else { return }
        let needed = Double(current.peakSynthesisMemoryMB)
            + minimumEagerSpeechHeadroomMB
        guard available < needed else { return }
        log(String(format:
            "TTS backend %@ needs ~%d MB peak but only %.0f MB free — using system voice this session (avoids MLX synthesis abort)",
            current.displayName, current.peakSynthesisMemoryMB, available))
        ttsStorage = SystemTTSService()   // ARC frees the heavy backend
    }

    public let session: ChatSession

    /// How long a contiguous silence ends the utterance and submits.
    /// 2.0 s lets users pause mid-thought without being cut off; 1.0
    /// was too aggressive for real speech (submitted after ~3 words).
    public var endOfUtteranceSilence: TimeInterval = 2.0
    /// RMS threshold below which we consider the room "silent" for the
    /// VAD timer. 0.02 better ignores breathing/keyboard noise; a
    /// quiet office may need recalibration downward.
    public var silenceThreshold: Float = 0.02
    /// Minimum iOS jetsam headroom required to overlap Kokoro synthesis with
    /// model generation. The policy also reserves the TTS backend's reported
    /// steady-state cost plus 384 MB for synthesis/audio transients.
    public var minimumEagerSpeechHeadroomMB: Double = 700

    // This must be replaceable. On a first launch macOS can grant microphone
    // access after AVAudioEngine has already discovered an input node; that
    // engine can then remain alive but deliver no buffers until the process is
    // restarted. Rebuilding it after the grant gives the newly-authorized
    // process a fresh input-device probe.
    private var engine = AVAudioEngine()
    private var sttStream: AsyncThrowingStream<SpeechPartial, Error>?
    private var sttTask: Task<Void, Never>?
    private var generationWatcher: Task<Void, Never>?
    private var ttsPreparation: Task<(seconds: TimeInterval, error: String?), Never>?
    private var microphoneStartupWatchdog: Task<Void, Never>?
    private var noSpeechRestartTask: Task<Void, Never>?
    private var receivedMicrophoneBuffer = false
    private var microphoneRecoveryAttempts = 0
    private var consecutiveNoSpeechErrors = 0
    private var captureConverter: AVAudioConverter?
    private var captureInputFormat: AVAudioFormat?
    private var recognitionInputFormat: AVAudioFormat?
    private var voiceStartRequestedAt: TimeInterval = 0
    private var listeningCycleStartedAt: TimeInterval = 0
    private var firstNonSilentBufferAt: TimeInterval?
    private var loggedFirstNonSilentBuffer = false
    private var loggedFirstTranscript = false

    private var lastNonSilentAt: Date = .distantPast
    private var lastSubmittedTranscript: String = ""
    /// Set when our VAD calls `stt.finish()`. The recognizer emits its own
    /// `isFinal` on brief in-utterance pauses (Apple's on-device STT does
    /// this aggressively, ~0.6 s of silence), which cuts users off mid-
    /// sentence. Only treat `isFinal` as authoritative after we explicitly
    /// asked to finalize.
    private var weRequestedFinish: Bool = false
    /// Accumulated text from completed recognition tasks in the current
    /// turn. When the recognizer auto-finalises mid-utterance (~0.6 s
    /// silence), we push the partial into here and start a fresh task;
    /// the merged display = `carriedTranscript` + " " + new partial.
    private var carriedTranscript: String = ""
    /// Index in `session.messages` where we expect the assistant reply
    /// for the most recent submission to land. Used by the watcher to
    /// know which message to read out once generation finishes.
    private var pendingAssistantIndex: Int?

    public init(session: ChatSession,
                stt: SpeechRecognizerService = SpeechRecognizerFactory.makeBest(),
                tts: TTSService? = nil) {
        self.session = session
        self.stt = stt
        self.ttsStorage = tts
    }

    // MARK: - Public controls

    public func toggle() {
        switch state {
        case .idle, .error: Task { await start() }
        case .starting, .listening, .thinking, .speaking: stop()
        }
    }

    /// Push a line to the session's debug pane so voice events show up in
    /// the same scrollback the user already uses for chat/model logs.
    private func log(_ message: String) {
        session.debug(message, category: "Voice")
    }

    public func start() async {
        guard state == .idle || isErrorState else { return }
        voiceStartRequestedAt = ProcessInfo.processInfo.systemUptime
        state = .starting
        let ttsLabel = ttsStorage?.displayName
            ?? "\(TTSBackendPreference.current.displayName) (deferred)"
        log("start() requested — backend=\(stt.displayName), tts=\(ttsLabel)")

        // SFSpeechRecognizer authorization and microphone authorization are
        // separate TCC permissions. Previously we requested only the former,
        // then AVAudioEngine triggered the microphone sheet from inside
        // beginListening(). On macOS the engine could successfully start while
        // that sheet was still up, but never deliver input until an app restart.
        let microphoneAuth = await requestMicrophoneAuthorization()
        guard microphoneAuth.authorized else {
            log("microphone auth denied")
            state = .error("Microphone access is required for voice chat. Enable it in System Settings > Privacy & Security > Microphone.")
            return
        }

        let auth = await stt.requestAuthorization()
        guard auth == .authorized else {
            log("STT auth denied: \(auth)")
            state = .error(authMessage(auth))
            return
        }

        if microphoneAuth.wasPrompted {
            log("rebuilding audio engine after first microphone grant")
            engine = AVAudioEngine()
            // Let CoreAudio publish the newly-authorized default input before
            // querying inputNode.outputFormat. This is a yield, not a user-
            // visible startup delay.
            try? await Task.sleep(nanoseconds: 150_000_000)
        } else {
            #if os(macOS)
            // Reusing a stopped input engine is unreliable after Kokoro has
            // initialized its own 24 kHz playback graph: start() can succeed
            // while the old tap never receives a buffer. Give each new Mac
            // voice session a fresh device probe, with a small interval for
            // CoreAudio to release the previous graph.
            log("refreshing Mac capture engine before listening")
            engine.stop()
            engine.reset()
            engine = AVAudioEngine()
            try? await Task.sleep(nanoseconds: 250_000_000)
            #endif
        }
        do {
            try configureAudioSession()
            try beginListening()
        } catch {
            log("start() failed: \(error.localizedDescription)")
            state = .error(error.localizedDescription)
        }
    }

    public func stop() {
        log("stop() — tearing down session")
        if session.isGenerating { session.stopGeneration() }
        sttTask?.cancel()
        sttTask = nil
        microphoneStartupWatchdog?.cancel()
        microphoneStartupWatchdog = nil
        noSpeechRestartTask?.cancel()
        noSpeechRestartTask = nil
        generationWatcher?.cancel()
        generationWatcher = nil
        stt.cancel()
        ttsStorage?.stop()
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        deactivateAudioSession()
        liveTranscript = ""
        inputLevel = 0
        state = .idle
    }

    /// Barge in while the assistant is thinking or speaking. Stop playback,
    /// cancel model work at its next safe boundary, then re-arm the microphone
    /// as soon as ChatSession confirms the context is free.
    public func interruptAndListen() {
        guard state == .thinking || state == .speaking else { return }
        log("barge-in requested")
        ttsStorage?.stop()
        generationWatcher?.cancel()
        pendingAssistantIndex = nil
        if session.isGenerating { session.stopGeneration() }
        state = .thinking
        generationWatcher = Task { [weak self] in
            guard let self else { return }
            while self.session.isGenerating, !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
            guard !Task.isCancelled, self.state != .idle else { return }
            self.resumeListeningAfterCycle()
        }
    }

    // MARK: - Listening

    private func beginListening() throws {
        log("beginListening (silence=\(endOfUtteranceSilence)s, floor=\(silenceThreshold))")
        listeningCycleStartedAt = ProcessInfo.processInfo.systemUptime
        state = .starting
        liveTranscript = ""
        carriedTranscript = ""
        lastNonSilentAt = Date()
        weRequestedFinish = false
        let stream = try stt.start(locale: .current)
        sttStream = stream
        receivedMicrophoneBuffer = false
        microphoneRecoveryAttempts = 0
        consecutiveNoSpeechErrors = 0
        loggedFirstNonSilentBuffer = false
        firstNonSilentBufferAt = nil
        loggedFirstTranscript = false
        noSpeechRestartTask?.cancel()
        noSpeechRestartTask = nil
        try installMicTap()
        sttTask = Task { [weak self] in
            await self?.consume(stream: stream)
        }
        armMicrophoneStartupWatchdog()
    }

    private func installMicTap() throws {
        let input = engine.inputNode
        // Always remove any previous tap before installing — `installTap`
        // raises an NSException ("only one tap can be installed on any
        // bus at a time") if one is already bound, and that kills the
        // whole app because Obj-C exceptions can't be caught from Swift.
        input.removeTap(onBus: 0)
        // After TTS playback (especially Kokoro's MLX-based engine),
        // AVAudioEngine has occasionally been observed in a state
        // where `installTap` throws even with no tap present — for
        // example if the hardware format changed mid-session. Stop
        // the engine before re-arming to force a clean re-probe of
        // the input node's current format. Idempotent when already
        // stopped.
        if engine.isRunning { engine.stop() }
        let nativeFormat = input.outputFormat(forBus: 0)
        // `installTap` rejects sample-rate==0 formats which happen on
        // first launch before the audio session is fully primed.
        guard nativeFormat.sampleRate > 0,
              nativeFormat.channelCount > 0
        else {
            throw SpeechSTTError.audioSessionFailed("Microphone format not ready.")
        }
        guard let recognitionFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: stt.preferredSampleRate,
            channels: 1,
            interleaved: false),
              let converter = AVAudioConverter(
                from: nativeFormat, to: recognitionFormat)
        else {
            throw SpeechSTTError.audioSessionFailed(
                "Could not configure microphone conversion to 16 kHz mono.")
        }
        captureInputFormat = nativeFormat
        recognitionInputFormat = recognitionFormat
        captureConverter = converter
        // Pull in 100 ms slices so the VAD can react quickly without
        // pegging the CPU on tiny 5 ms callbacks.
        let frameCount = AVAudioFrameCount(nativeFormat.sampleRate * 0.1)
        // The NSException AVFAudio throws from `installTapOnBus:` has
        // to be caught inside an Obj-C @try that's on the stack WHEN
        // the throw happens — not wrapped around a Swift closure that
        // calls installTap. If the throw unwinds through any Swift
        // frame first, Swift's exception personality aborts before
        // control can return to the @try. The dedicated Obj-C helper
        // does the installTap call itself, so the @try is live at
        // the moment of the throw and the exception is caught.
        let tapBlock: AVAudioNodeTapBlock = { [weak self] buf, _ in
            guard let self else { return }
            Task { @MainActor [weak self] in
                self?.handleCapturedAudio(buf)
            }
        }
        // On macOS, playback can change the shared device graph (for example
        // from 48 kHz to Kokoro's 24 kHz output) between listening cycles.
        // Passing the previously-observed format then raises AVFAudio's
        // "format mismatch" exception. A nil format tells AVAudioEngine to
        // negotiate the input node's current output format atomically while
        // installing the tap. iOS keeps the explicit native format because its
        // play-and-record AVAudioSession owns and stabilizes that graph.
        #if os(macOS)
        let tapFormat: AVAudioFormat? = nil
        #else
        let tapFormat: AVAudioFormat? = nativeFormat
        #endif
        if let reason = ObjCExceptionWrapper.installTap(
            on: input, bus: 0,
            bufferSize: frameCount, format: tapFormat,
            block: tapBlock
        ) {
            throw SpeechSTTError.audioSessionFailed(
                "AVAudioEngine installTap raised NSException: \(reason)")
        }
        engine.prepare()
        try engine.start()
        log("audio engine started · \(Int(nativeFormat.sampleRate)) Hz · channels=\(nativeFormat.channelCount)")
        log("capture normalization · \(Int(nativeFormat.sampleRate)) Hz/\(nativeFormat.channelCount)ch → \(Int(recognitionFormat.sampleRate)) Hz/1ch")
    }

    private func handleCapturedAudio(_ nativeBuffer: AVAudioPCMBuffer) {
        let buffer: AVAudioPCMBuffer
        do {
            buffer = try normalizedCaptureBuffer(nativeBuffer)
        } catch {
            log("microphone conversion failed: \(error.localizedDescription)")
            state = .error(error.localizedDescription)
            stop()
            return
        }
        handleAudio(buffer: buffer, level: Self.rms(buffer))
    }

    /// SFSpeechRecognizer is most reliable with the 16 kHz mono format its
    /// service advertises. A Mac input can initially appear as 48 kHz stereo;
    /// forwarding that buffer directly allowed Speech to listen only to a
    /// silent channel and made the first voice-sheet attempt look dead. Always
    /// downmix and resample before both recognition and VAD.
    private func normalizedCaptureBuffer(
        _ input: AVAudioPCMBuffer
    ) throws -> AVAudioPCMBuffer {
        guard let outputFormat = recognitionInputFormat else {
            throw SpeechSTTError.audioSessionFailed(
                "Microphone conversion format is unavailable.")
        }

        let inputFormat = input.format
        if captureConverter == nil
            || captureInputFormat?.sampleRate != inputFormat.sampleRate
            || captureInputFormat?.channelCount != inputFormat.channelCount
            || captureInputFormat?.commonFormat != inputFormat.commonFormat
        {
            guard let converter = AVAudioConverter(
                from: inputFormat, to: outputFormat)
            else {
                throw SpeechSTTError.audioSessionFailed(
                    "The microphone format changed and could not be converted.")
            }
            captureInputFormat = inputFormat
            captureConverter = converter
            log("capture format changed · \(Int(inputFormat.sampleRate)) Hz/\(inputFormat.channelCount)ch → \(Int(outputFormat.sampleRate)) Hz/1ch")
        }
        guard let converter = captureConverter else {
            throw SpeechSTTError.audioSessionFailed(
                "Microphone converter is unavailable.")
        }

        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let capacity = AVAudioFrameCount(
            max(1, Int(ceil(Double(input.frameLength) * ratio)) + 32))
        guard let output = AVAudioPCMBuffer(
            pcmFormat: outputFormat, frameCapacity: capacity)
        else {
            throw SpeechSTTError.audioSessionFailed(
                "Could not allocate a normalized microphone buffer.")
        }

        var suppliedInput = false
        var conversionError: NSError?
        let status = converter.convert(
            to: output, error: &conversionError
        ) { _, inputStatus in
            guard !suppliedInput else {
                inputStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error {
            throw SpeechSTTError.audioSessionFailed(
                conversionError?.localizedDescription
                    ?? "Microphone sample conversion failed.")
        }
        guard output.frameLength > 0 else {
            throw SpeechSTTError.audioSessionFailed(
                "Microphone conversion produced no samples.")
        }
        return output
    }

    private func handleAudio(buffer: AVAudioPCMBuffer, level: Float) {
        if !receivedMicrophoneBuffer {
            receivedMicrophoneBuffer = true
            microphoneStartupWatchdog?.cancel()
            microphoneStartupWatchdog = nil
            let now = ProcessInfo.processInfo.systemUptime
            log(String(format:
                "microphone audio flowing · frames=%d · level=%.4f · cycle=%.3fs · start=%.3fs",
                Int(buffer.frameLength), level,
                now - listeningCycleStartedAt,
                now - voiceStartRequestedAt))
        }
        // Smooth the meter: 1-pole low-pass so the orb pulse doesn't strobe.
        inputLevel = max(0, min(1, 0.7 * inputLevel + 0.3 * level))
        do {
            try stt.append(buffer: buffer)
            // `Listening` means more than AVAudioEngine having started: Speech
            // has now accepted live microphone audio. Until this succeeds the
            // UI remains in `Preparing microphone…`, avoiding a false-ready
            // state while retaining the very first buffer.
            if state == .starting { state = .listening }
        } catch {
            state = .error(error.localizedDescription)
            stop()
            return
        }
        if level >= silenceThreshold {
            if !loggedFirstNonSilentBuffer {
                loggedFirstNonSilentBuffer = true
                firstNonSilentBufferAt = ProcessInfo.processInfo.systemUptime
                log(String(format:
                    "microphone speech detected · level=%.4f · cycle=%.3fs",
                    level,
                    ProcessInfo.processInfo.systemUptime
                        - listeningCycleStartedAt))
            }
            consecutiveNoSpeechErrors = 0
            lastNonSilentAt = Date()
        } else if state == .listening,
                  !(liveTranscript.isEmpty && carriedTranscript.isEmpty),
                  !weRequestedFinish,
                  Date().timeIntervalSince(lastNonSilentAt) >= endOfUtteranceSilence {
            // End of turn: tell the recognizer to flush, the consumer
            // will pick the final result up and submit.
            log("VAD end-of-turn (silence=\(String(format: "%.2f", Date().timeIntervalSince(lastNonSilentAt)))s, carry=\"\(carriedTranscript)\", live=\"\(liveTranscript)\") — submitting")
            weRequestedFinish = true
            stt.finish()
            // Belt-and-braces: if the recognizer never emits another
            // isFinal (can happen after a previous restart left the
            // request stale), submit what we have after a short grace.
            let snapshot = Self.joined(carriedTranscript, liveTranscript)
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 800_000_000)
                guard let self else { return }
                guard self.state == .listening, self.weRequestedFinish else { return }
                log("finish() did not produce isFinal in 0.8s — force-submitting \"\(snapshot)\"")
                self.submitFinal(text: snapshot)
            }
        }
    }

    private func consume(stream: AsyncThrowingStream<SpeechPartial, Error>) async {
        do {
            for try await partial in stream {
                // Only advance `liveTranscript` from a NON-empty partial —
                // the backend often emits an empty `isFinal` after
                // `endAudio()` which would otherwise wipe the display and
                // our submission text.
                if !partial.text.isEmpty {
                    if !loggedFirstTranscript {
                        loggedFirstTranscript = true
                        log(String(format:
                            "recognizer first transcript · cycle=%.3fs · after-speech=%.3fs · text=\"%@\"",
                            ProcessInfo.processInfo.systemUptime
                                - listeningCycleStartedAt,
                            firstNonSilentBufferAt.map {
                                ProcessInfo.processInfo.systemUptime - $0
                            } ?? 0,
                            partial.text))
                    }
                    consecutiveNoSpeechErrors = 0
                    liveTranscript = Self.joined(carriedTranscript, partial.text)
                    lastNonSilentAt = Date()
                }
                // The recognizer emits its own `isFinal` after ~0.6 s of
                // intra-utterance silence. If our VAD hasn't fired yet,
                // fold this partial into `carriedTranscript` and spin up
                // a fresh recognition task so the user can keep talking.
                if partial.isFinal {
                    let merged = partial.text.isEmpty
                        ? liveTranscript
                        : Self.joined(carriedTranscript, partial.text)
                    if weRequestedFinish {
                        log("recognizer isFinal after our finish() — partial=\"\(partial.text)\", submitting \"\(merged)\"")
                        submitFinal(text: merged)
                    } else {
                        log("recognizer auto-final mid-utterance; carry=\"\(merged)\", restarting")
                        carriedTranscript = merged
                        liveTranscript = merged
                        do {
                            try restartRecognizer()
                        } catch {
                            log("restartRecognizer failed: \(error) — submitting what we have")
                            weRequestedFinish = true
                            submitFinal(text: merged)
                        }
                    }
                    return
                }
            }
            if weRequestedFinish {
                if !liveTranscript.isEmpty {
                    submitFinal(text: liveTranscript)
                } else {
                    resumeListeningAfterCycle()
                }
            } else if state == .listening || state == .starting {
                // Recognizer self-terminated before we asked it to. Keep
                // listening with a fresh task; don't double-restart if some
                // other code path already swapped us out.
                try? restartRecognizer()
            }
        } catch is CancellationError {
            // user-initiated stop; nothing to do.
        } catch {
            let ns = error as NSError
            log("recognizer stream error · \(ns.domain) \(ns.code) · \(error.localizedDescription)")
            if (state == .listening || state == .starting),
               !weRequestedFinish,
               Self.isRecoverableNoSpeechError(error) {
                scheduleRecognizerRestartAfterNoSpeech()
            } else {
                state = .error(error.localizedDescription)
            }
        }
    }

    /// SFSpeech may end an otherwise healthy silent session with error 1110.
    /// Restart it without surfacing a red error state, but never spin: stale
    /// callbacks or a persistently unavailable recognizer are bounded by an
    /// exponential delay (0.25, 0.5, 1, then 2 seconds).
    private func scheduleRecognizerRestartAfterNoSpeech() {
        guard noSpeechRestartTask == nil else {
            log("recognizer no-speech restart already scheduled — ignoring duplicate callback")
            return
        }
        consecutiveNoSpeechErrors += 1
        let exponent = min(consecutiveNoSpeechErrors - 1, 3)
        let delay = min(2.0, 0.25 * pow(2.0, Double(exponent)))
        state = receivedMicrophoneBuffer ? .listening : .starting
        log(String(format:
            "recognizer reported no speech — retrying in %.2fs (attempt %d)",
            delay, consecutiveNoSpeechErrors))

        noSpeechRestartTask = Task { [weak self] in
            do {
                try await Task.sleep(
                    nanoseconds: UInt64(delay * 1_000_000_000))
            } catch {
                return
            }
            guard let self,
                  self.state == .listening || self.state == .starting,
                  !self.weRequestedFinish
            else { return }
            self.noSpeechRestartTask = nil
            do {
                try self.restartRecognizer()
            } catch {
                self.log("recognizer restart after no-speech failed: \(error.localizedDescription)")
                self.state = .error(error.localizedDescription)
            }
        }
    }

    /// Start a fresh recognizer task in the current turn. The mic tap and
    /// `carriedTranscript` stay intact, so display is seamless and the
    /// final submission includes everything the user has said.
    private func restartRecognizer() throws {
        sttTask?.cancel()
        stt.cancel()
        let stream = try stt.start(locale: .current)
        sttStream = stream
        sttTask = Task { [weak self] in
            await self?.consume(stream: stream)
        }
    }

    /// Merge two transcript fragments with sensible whitespace.
    private static func joined(_ a: String, _ b: String) -> String {
        let left = a.trimmingCharacters(in: .whitespacesAndNewlines)
        let right = b.trimmingCharacters(in: .whitespacesAndNewlines)
        if left.isEmpty { return right }
        if right.isEmpty { return left }
        return left + " " + right
    }

    private func submitFinal(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != lastSubmittedTranscript else {
            log("submitFinal skipped (empty or duplicate): \"\(trimmed)\"")
            // Don't re-arm the mic mid-turn: a delayed duplicate `isFinal`
            // can land after the force-submit grace task already moved us to
            // `.thinking`, and resuming here would install a second recognizer
            // that captures the assistant's own TTS. Only resume while still
            // in a listening phase.
            if state != .thinking && state != .speaking {
                resumeListeningAfterCycle()
            }
            return
        }
        log("submitFinal → session.send(\"\(trimmed)\")")
        lastSubmittedTranscript = trimmed
        // Tear down the mic tap while the model is thinking — keeping
        // the engine running and continuing to feed the recognizer
        // would (a) waste CPU and (b) capture our own TTS output.
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        microphoneStartupWatchdog?.cancel()
        microphoneStartupWatchdog = nil
        state = .thinking
        // Recording uses `.voiceChat`, which is optimized for duplex speech
        // and can be dramatically quieter through the phone speaker. Once the
        // mic is stopped, move answers onto the normal media/spoken-audio path;
        // `resumeListeningAfterCycle()` restores `.playAndRecord` afterward.
        do {
            try configurePlaybackAudioSession()
        } catch {
            log("playback audio-session configuration failed: \(error.localizedDescription)")
        }
        // Capture and recognition now have exclusive priority for the whole
        // utterance. Warm TTS only after the mic is stopped; this still hides
        // most startup work under retrieval/model generation without delaying
        // or dropping the user's first words.
        // The assistant placeholder is appended by `ChatSession.send`,
        // so the index of the reply is the count *after* dispatch − 1.
        session.send(trimmed)
        beginTTSPreparationIfSafe()
        pendingAssistantIndex = max(0, session.messages.count - 1)
        watchForReply()
    }

    private func watchForReply() {
        generationWatcher?.cancel()
        generationWatcher = Task { [weak self] in
            guard let self else { return }
            await self.streamAssistantReply()
        }
    }

    /// Hide the first TTS graph/G2P execution under retrieval/model work,
    /// after capture is safely stopped. Preparation is skipped under the same
    /// memory rule used for overlapping synthesis and generation.
    private func beginTTSPreparationIfSafe() {
        guard ttsPreparation == nil else { return }
        let availableMB = Self.availableMemoryMB()
        guard StreamingSpeechPolicy.allowsEagerSynthesis(
            availableMemoryMB: availableMB,
            estimatedTTSMemoryMB: tts.peakSynthesisMemoryMB,
            minimumHeadroomMB: minimumEagerSpeechHeadroomMB)
        else {
            log(String(format:
                "TTS preparation skipped — %.0f MB available, peak estimate=%d MB",
                availableMB, tts.peakSynthesisMemoryMB))
            return
        }

        let backend = tts.displayName
        let service = tts
        let footprintBefore = MemoryStats.physFootprintMB()
        log("TTS preparation started — \(backend)")
        let task = Task.detached(priority: .utility) {
            () -> (seconds: TimeInterval, error: String?) in
            let started = ProcessInfo.processInfo.systemUptime
            do {
                try await service.prepareForConversation()
                return (ProcessInfo.processInfo.systemUptime - started, nil)
            } catch {
                return (ProcessInfo.processInfo.systemUptime - started,
                        error.localizedDescription)
            }
        }
        ttsPreparation = task
        Task { [weak self] in
            let result = await task.value
            guard let self else { return }
            if let error = result.error {
                self.log(String(format:
                    "TTS preparation failed after %.2fs — %@",
                    result.seconds, error))
            } else {
                self.log(String(format:
                    "TTS preparation ready in %.2fs · Δmem=%+.1f MB",
                    result.seconds,
                    MemoryStats.physFootprintMB() - footprintBefore))
            }
        }
    }

    private func awaitTTSPreparationIfNeeded() async {
        guard let task = ttsPreparation else { return }
        _ = await task.value
        ttsPreparation = nil
    }

    /// Stream stable prose into Kokoro as it arrives. Complete sentences are
    /// preferred, but a long first sentence can begin at a natural clause or
    /// a bounded word break instead of waiting for generation to finish.
    /// Under low jetsam headroom, eager synthesis is deferred until model
    /// generation ends so Kokoro's transient MLX work does not overlap decode.
    private func streamAssistantReply() async {
        guard let idx = pendingAssistantIndex,
              idx < session.messages.count,
              session.messages[idx].role == .assistant
        else {
            log("streamAssistantReply: no pending message — resuming listen")
            resumeListeningAfterCycle()
            return
        }
        pendingAssistantIndex = nil
        let maximumChunk = max(
            40, tts.preferredStreamingChunkCharacters ?? 112)
        let defaultMinimum = min(56, max(24, maximumChunk - 28))
        let requestedMinimum = tts.preferredStreamingMinimumCharacters
            ?? defaultMinimum
        let minimumChunk = min(
            max(24, requestedMinimum), max(24, maximumChunk - 16))
        log("TTS streaming start (adaptive sentence/clause polling · window=\(minimumChunk)–\(maximumChunk) chars)")
        // A heavy MLX backend (Kokoro ≈ 2.8 GB synthesis peak) cannot coexist
        // with a large resident LLM: the synthesis either defers forever
        // (silent turn) or, once generation frees the compute gate, aborts
        // inside MLX — EXC_BREAKPOINT in mlx_array_eval, real crash 2026-08-02
        // (Kokoro + Bonsai 27B). Downgrade to the ~5 MB system voice for the
        // rest of the session when the selected backend can't fit the current
        // jetsam headroom, so the user hears the answer instead of silence or
        // a crash.
        ensureAffordableTTS()
        let t0 = Date()
        // Keep the UI honest: we are still thinking until the first chunk is
        // actually handed to the audio backend.
        state = .thinking

        var spokenUpTo = 0   // characters already dispatched to TTS
        var sawAnyText = false
        var firstTextAt: Date?
        var queuedFirstChunk = false
        var memoryDeferralLogged = false
        // Turn-level pacing accounting, filled from the backend's per-chunk
        // metrics so the "TTS done" wall time can be split into real audio,
        // synthesis, and dead air between chunks.
        var chunkCount = 0
        var totalAudioSeconds = 0.0
        var totalGapSeconds = 0.0
        var totalSynthesisSeconds = 0.0
        // sanitizeForSpeech runs several regex passes over the ENTIRE
        // growing reply; at the 75 ms poll cadence most polls see
        // unchanged text (UI pushes are throttled to 10 Hz and decode is
        // slower still), so re-sanitize only when the raw text changed.
        // Length is a sufficient change signal here: streaming appends,
        // and the display-side scrubs that rewrite text always change
        // the length too.
        var lastRawCount = -1
        var lastSanitized = ""

        while !Task.isCancelled {
            guard idx < session.messages.count else { break }
            // When a chunk was dispatched this pass, loop again immediately —
            // more speakable text may already be buffered (always true while
            // draining a completed reply). Sleeping 75 ms per chunk here
            // added dead air to every multi-chunk drain.
            var advancedThisPass = false
            let raw = session.messages[idx].text
            if raw.count != lastRawCount {
                lastSanitized = Self.sanitizeForSpeech(raw)
                lastRawCount = raw.count
            }
            let full = lastSanitized
            if full.count > spokenUpTo {
                if !full.isEmpty {
                    sawAnyText = true
                    if firstTextAt == nil { firstTextAt = Date() }
                }

                let generating = session.isGenerating
                let availableMB = Self.availableMemoryMB()
                let eagerAllowed = !generating
                    || StreamingSpeechPolicy.allowsEagerSynthesis(
                        availableMemoryMB: availableMB,
                        estimatedTTSMemoryMB: tts.peakSynthesisMemoryMB,
                        minimumHeadroomMB: minimumEagerSpeechHeadroomMB)

                if !eagerAllowed {
                    if !memoryDeferralLogged {
                        log(String(format:
                            "TTS eager synthesis deferred — %.0f MB available, peak estimate=%d MB, footprint=%.0f MB",
                            availableMB, tts.peakSynthesisMemoryMB,
                            MemoryStats.physFootprintMB()))
                        memoryDeferralLogged = true
                    }
                } else {
                    if memoryDeferralLogged, generating {
                        log(String(format:
                            "TTS eager synthesis resumed — %.0f MB available",
                            availableMB))
                        memoryDeferralLogged = false
                    }
                    let newFragment = String(full.suffix(full.count - spokenUpTo))
                    if let prefix = StreamingSpeechPolicy.takeSpeakablePrefix(
                        newFragment,
                        generating: generating,
                        allowEarlyClause: true,
                        minimumClause: minimumChunk,
                        maximumClause: maximumChunk)
                    {
                        var toSpeak = prefix.text.trimmingCharacters(
                            in: .whitespacesAndNewlines)
                        // A forced word-boundary chunk should sound like a
                        // continuation, not a completed sentence.
                        if prefix.boundary == .softWrap,
                           let last = toSpeak.last,
                           !",;:—–".contains(last) {
                            toSpeak += ","
                        }
                        if !toSpeak.isEmpty {
                            if !queuedFirstChunk {
                                await awaitTTSPreparationIfNeeded()
                            }
                            state = .speaking
                            let synthesisStarted = Date()
                            do {
                                let ttsBoundary: TTSChunkBoundary
                                switch prefix.boundary {
                                case .sentence: ttsBoundary = .sentence
                                case .clause: ttsBoundary = .clause
                                case .softWrap: ttsBoundary = .softWrap
                                case .final: ttsBoundary = .final
                                }
                                try await tts.speakChunk(
                                    toSpeak, boundary: ttsBoundary)
                            } catch {
                                log("TTS chunk failed: \(error.localizedDescription)")
                            }
                            let synthesisSeconds =
                                Date().timeIntervalSince(synthesisStarted)
                            chunkCount += 1
                            totalSynthesisSeconds += synthesisSeconds
                            if !queuedFirstChunk {
                                queuedFirstChunk = true
                                let audioReady = Date()
                                let textWait = firstTextAt.map {
                                    synthesisStarted.timeIntervalSince($0)
                                } ?? 0
                                log(String(format:
                                    "TTS first audio ready · boundary=%@ · chars=%d · text-wait=%.2fs · synthesis=%.2fs · turn=%.2fs · available=%.0f MB",
                                    String(describing: prefix.boundary),
                                    prefix.consumedCharacters, textWait,
                                    audioReady.timeIntervalSince(synthesisStarted),
                                    audioReady.timeIntervalSince(t0), availableMB))
                            }
                            // Per-chunk pacing: `gap` is measured dead air
                            // (queue drained → this chunk's playback start);
                            // `ahead` is queued audio still unplayed when
                            // this chunk was scheduled. Proves whether long
                            // voice turns are silence or real audio length.
                            if let metrics = tts.takeStreamingChunkMetrics() {
                                totalAudioSeconds += metrics.audioSeconds
                                totalGapSeconds += metrics.gapSeconds
                                log(String(format:
                                    "TTS chunk %d · boundary=%@ · chars=%d · synthesis=%.2fs · audio=%.2fs (%.1f ch/s) · gap=%.2fs · ahead=%.2fs · trim=%.2fs",
                                    chunkCount,
                                    String(describing: prefix.boundary),
                                    prefix.consumedCharacters,
                                    synthesisSeconds,
                                    metrics.audioSeconds,
                                    Double(prefix.consumedCharacters)
                                        / max(0.001, metrics.audioSeconds),
                                    metrics.gapSeconds,
                                    metrics.queueAheadSeconds,
                                    metrics.trimmedSeconds))
                            }
                        }
                        // Advance by source characters, not the optional comma
                        // added only for soft-wrap prosody.
                        spokenUpTo += prefix.consumedCharacters
                        advancedThisPass = true
                    }
                }
            }
            if !session.isGenerating, spokenUpTo >= full.count {
                // A completed reply may require several backend-sized chunks.
                // Keep draining until all source characters are queued.
                break
            }
            // Only poll-wait when no chunk was consumed; after a dispatch the
            // next speakable prefix (if any) is already in the buffer.
            if !advancedThisPass {
                try? await Task.sleep(nanoseconds: 75_000_000)
            }
        }
        // Barge-in (interruptAndListen) cancels this task and spawns a fresh
        // watcher that owns the re-arm. Returning here avoids a second
        // resumeListeningAfterCycle() racing that watcher — which would
        // install a duplicate mic tap / STT stream.
        if Task.isCancelled { return }
        if !sawAnyText {
            log("streamAssistantReply: reply was empty")
            resumeListeningAfterCycle()
            return
        }
        await tts.awaitPlayback()
        // wall = generation overlap + audio + gaps + final drain. When
        // audio+gaps ≈ wall the turn length is real speech; large uncounted
        // remainder means waiting on text (generation-bound), not TTS.
        log(String(format:
            "TTS done in %.2fs · chunks=%d · audio=%.1fs · synthesis=%.1fs · gaps=%.1fs",
            Date().timeIntervalSince(t0), chunkCount,
            totalAudioSeconds, totalSynthesisSeconds, totalGapSeconds))
        resumeListeningAfterCycle()
    }

    /// `os_proc_available_memory` reflects the process's current iOS jetsam
    /// allowance, which is more useful here than physical device RAM or RSS.
    /// Other platforms return zero and the shared policy permits eager TTS.
    private static func availableMemoryMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / (1024 * 1024)
        #else
        return 0
        #endif
    }

    /// Apply the same display-time scrubbing the chat view uses, so
    /// half-emitted `<|tool_call>…` syntax doesn't get spoken before
    /// the parser strips it. Keeps in sync with
    /// `ChatView.MessageRow.displayText` — if that grows, grow this.
    private static func sanitizeForSpeech(_ raw: String) -> String {
        var t = raw
        let closed = [
            #"<\|tool_call\|?>[\s\S]*?<tool_call\|>"#,
            #"<tool_call>[\s\S]*?</tool_call>"#,
            #"<\|tool_response\|?>[\s\S]*?<tool_response\|>"#,
            // Reasoning blocks — the FT occasionally opens a <think>…</think>.
            // Strip closed ones so they're never spoken.
            #"<think>[\s\S]*?</think>"#,
        ]
        for pat in closed {
            t = t.replacingOccurrences(of: pat, with: "", options: .regularExpression)
        }
        // Qwen may inherit the opening <think> tag from the formatted prompt
        // and emit only `scratchpad</think>answer`. Speak only the answer.
        if let r = t.range(of: "</think>", options: .backwards) {
            t = String(t[r.upperBound...])
        }
        if let r = t.range(of: #"<\|?tool[_a-z]*"#, options: .regularExpression) {
            t = String(t[..<r.lowerBound])
        }
        if let r = t.range(of: #"<tool[_a-z]*"#, options: .regularExpression) {
            t = String(t[..<r.lowerBound])
        }
        // Mid-stream the closing </think> hasn't arrived yet — truncate at the
        // open <think> so the reasoning is never voiced before the parser
        // strips it. (The closed pattern above handles the post-close text.)
        if let r = t.range(of: "<think") { t = String(t[..<r.lowerBound]) }
        for lit in ["<tool_call|>", "<tool_response|>", "<|\"|>", "<|\""] {
            t = t.replacingOccurrences(of: lit, with: "")
        }
        return t
    }

    private func resumeListeningAfterCycle() {
        guard state != .idle else { return }
        do {
            try configureAudioSession()
            try beginListening()
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    // MARK: - Microphone permission and first-buffer recovery

    private struct MicrophoneAuthorization {
        let authorized: Bool
        let wasPrompted: Bool
    }

    private func requestMicrophoneAuthorization() async -> MicrophoneAuthorization {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        log("microphone auth status=\(microphoneAuthorizationName(status))")
        switch status {
        case .authorized:
            Self.microphonePermissionResultThisProcess = true
            return MicrophoneAuthorization(authorized: true, wasPrompted: false)
        case .notDetermined:
            if let granted = Self.microphonePermissionResultThisProcess {
                log("microphone auth status still notDetermined — reusing in-process prompt result · granted=\(granted)")
                return MicrophoneAuthorization(
                    authorized: granted, wasPrompted: false)
            }

            let initiatedPrompt: Bool
            let requestTask: Task<Bool, Never>
            if let inFlight = Self.microphonePermissionRequestTask {
                initiatedPrompt = false
                requestTask = inFlight
                log("microphone auth prompt already in flight — awaiting it")
            } else {
                initiatedPrompt = true
                let task = Task<Bool, Never> {
                    await withCheckedContinuation { continuation in
                        AVCaptureDevice.requestAccess(for: .audio) { granted in
                            continuation.resume(returning: granted)
                        }
                    }
                }
                Self.microphonePermissionRequestTask = task
                requestTask = task
            }

            let granted = await requestTask.value
            Self.microphonePermissionResultThisProcess = granted
            if initiatedPrompt {
                Self.microphonePermissionRequestTask = nil
            }
            log("microphone auth prompt completed · granted=\(granted)")
            return MicrophoneAuthorization(
                authorized: granted, wasPrompted: initiatedPrompt)
        case .denied, .restricted:
            Self.microphonePermissionResultThisProcess = false
            return MicrophoneAuthorization(authorized: false, wasPrompted: false)
        @unknown default:
            return MicrophoneAuthorization(authorized: false, wasPrompted: false)
        }
    }

    private func microphoneAuthorizationName(_ status: AVAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "notDetermined"
        case .restricted: return "restricted"
        case .denied: return "denied"
        case .authorized: return "authorized"
        @unknown default: return "unknown"
        }
    }

    /// AVAudioEngine can report that it started yet never invoke its tap after
    /// a first-run TCC transition. Detect that silent failure and rebuild the
    /// capture pipeline once, so the user does not have to restart the app.
    private func armMicrophoneStartupWatchdog() {
        microphoneStartupWatchdog?.cancel()
        #if os(macOS)
        // CoreAudio may need more than 0.5 s to restart after a 24 kHz Kokoro
        // graph has been active. Increase the grace period after each real
        // rebuild instead of repeatedly killing a slow-but-healthy start.
        let graceSeconds = min(
            3.0, 1.5 + (Double(microphoneRecoveryAttempts) * 0.75))
        #else
        let graceSeconds = 0.75
        #endif
        microphoneStartupWatchdog = Task { [weak self] in
            do {
                try await Task.sleep(
                    nanoseconds: UInt64(graceSeconds * 1_000_000_000))
            } catch {
                return
            }
            guard let self,
                  self.state == .starting,
                  !self.receivedMicrophoneBuffer
            else { return }
            await self.recoverMicrophoneCapture(graceSeconds: graceSeconds)
        }
    }

    private func recoverMicrophoneCapture(graceSeconds: Double) async {
        let maximumAttempts: Int
        #if os(macOS)
        maximumAttempts = 2
        #else
        maximumAttempts = 1
        #endif
        guard microphoneRecoveryAttempts < maximumAttempts else {
            log(String(format:
                "microphone startup failed — no audio buffers after %.2fs grace and %d engine rebuild(s)",
                graceSeconds, microphoneRecoveryAttempts))
            stt.cancel()
            if engine.isRunning {
                engine.inputNode.removeTap(onBus: 0)
                engine.stop()
            }
            state = .error("The microphone did not start. Check the selected input device and microphone permission, then tap the microphone again.")
            return
        }

        microphoneRecoveryAttempts += 1
        log(String(format:
            "no microphone buffers after %.2fs — rebuilding capture pipeline (attempt %d/%d)",
            graceSeconds, microphoneRecoveryAttempts, maximumAttempts))
        sttTask?.cancel()
        stt.cancel()
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        engine.reset()
        engine = AVAudioEngine()

        // Starting a replacement graph in the same CoreAudio callback cycle
        // can reproduce the no-buffer state. Let device teardown settle first.
        do {
            try await Task.sleep(nanoseconds: 350_000_000)
        } catch {
            return
        }
        guard state == .starting, !receivedMicrophoneBuffer else { return }

        do {
            let stream = try stt.start(locale: .current)
            sttStream = stream
            try installMicTap()
            sttTask = Task { [weak self] in
                await self?.consume(stream: stream)
            }
            armMicrophoneStartupWatchdog()
        } catch {
            log("microphone capture rebuild failed: \(error.localizedDescription)")
            state = .error(error.localizedDescription)
        }
    }

    // MARK: - AVAudioSession (iOS only)

    private func configureAudioSession() throws {
        #if canImport(UIKit)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord,
                                    mode: .voiceChat,
                                    options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            throw SpeechSTTError.audioSessionFailed(error.localizedDescription)
        }
        // Keep the display awake while voice chat is active — otherwise
        // the phone auto-locks mid-"thinking" and the user loses the mic
        // tap + Kokoro playback. Re-enabled in `stop()`.
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
    }

    private func configurePlaybackAudioSession() throws {
        #if canImport(UIKit)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(
                .playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
            let outputs = session.currentRoute.outputs
                .map { $0.portName }.joined(separator: ", ")
            log(String(format:
                "playback route ready · outputs=%@ · system-volume=%.2f",
                outputs.isEmpty ? "unknown" : outputs,
                session.outputVolume))
        } catch {
            throw SpeechSTTError.audioSessionFailed(error.localizedDescription)
        }
        #endif
    }

    private func deactivateAudioSession() {
        #if canImport(UIKit)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        UIApplication.shared.isIdleTimerDisabled = false
        #endif
    }

    // MARK: - Utilities

    private var isErrorState: Bool {
        if case .error = state { return true } else { return false }
    }

    private func authMessage(_ s: SpeechAuthState) -> String {
        switch s {
        case .denied(let r): return r
        default: return "Speech recognition not authorized."
        }
    }

    private static func isRecoverableNoSpeechError(_ error: Error) -> Bool {
        let ns = error as NSError
        let message = error.localizedDescription.lowercased()
        if message.contains("no speech") || message.contains("no voice") {
            return true
        }
        // SFSpeechRecognizer uses private assistant-domain errors for silence
        // and for a recognition task ending while its audio request is being
        // replaced. They are session-local; a fresh task is the correct
        // recovery and does not require rebuilding the microphone tap.
        return ns.domain == "kAFAssistantErrorDomain"
            && (ns.code == 203 || ns.code == 1101 || ns.code == 1110)
    }

    /// Root-mean-square across every Float32 channel. The normal capture path
    /// is mono, but considering every channel keeps diagnostics truthful if a
    /// device format changes before the converter can be rebuilt.
    private static func rms(_ buf: AVAudioPCMBuffer) -> Float {
        guard let channels = buf.floatChannelData else { return 0 }
        let n = Int(buf.frameLength)
        let channelCount = Int(buf.format.channelCount)
        guard n > 0, channelCount > 0 else { return 0 }
        var sumSquares: Float = 0
        for channel in 0..<channelCount {
            let samples = channels[channel]
            for i in 0..<n {
                let v = samples[i]
                sumSquares += v * v
            }
        }
        return (sumSquares / Float(n * channelCount)).squareRoot()
    }
}
