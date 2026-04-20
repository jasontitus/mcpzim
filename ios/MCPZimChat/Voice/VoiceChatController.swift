// SPDX-License-Identifier: MIT
//
// Voice-chat coordinator. Wires three pieces together:
//
//   1. `AVAudioEngine` mic tap → `SpeechRecognizerService` (Apple's
//      best on-device STT, see `SpeechRecognizerService.swift`).
//   2. End-of-utterance detection (silence VAD) → `ChatSession.send(_:)`.
//   3. Watch the assistant message stream until the model is no longer
//      generating, then hand the final reply to `TTSService` and loop
//      back to listening.
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
import Observation
#if canImport(UIKit)
import UIKit
#endif

@MainActor
@Observable
public final class VoiceChatController {
    public enum State: Equatable, Sendable {
        case idle
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
    public let tts: TTSService
    public let session: ChatSession

    /// How long a contiguous silence ends the utterance and submits.
    /// 1.0 s is the sweet spot for conversational use; tighter and the
    /// recognizer cuts the user off, looser and the model takes ages
    /// to respond.
    public var endOfUtteranceSilence: TimeInterval = 1.0
    /// RMS threshold below which we consider the room "silent" for the
    /// VAD timer. Empirically tuned for a quiet office; loud rooms
    /// will need calibration.
    public var silenceThreshold: Float = 0.012

    private let engine = AVAudioEngine()
    private var sttStream: AsyncThrowingStream<SpeechPartial, Error>?
    private var sttTask: Task<Void, Never>?
    private var generationWatcher: Task<Void, Never>?

    private var lastNonSilentAt: Date = .distantPast
    private var lastSubmittedTranscript: String = ""
    /// Index in `session.messages` where we expect the assistant reply
    /// for the most recent submission to land. Used by the watcher to
    /// know which message to read out once generation finishes.
    private var pendingAssistantIndex: Int?

    public init(session: ChatSession,
                stt: SpeechRecognizerService = SpeechRecognizerFactory.makeBest(),
                tts: TTSService = TTSFactory.makeBest()) {
        self.session = session
        self.stt = stt
        self.tts = tts
    }

    // MARK: - Public controls

    public func toggle() {
        switch state {
        case .idle, .error: Task { await start() }
        case .listening, .thinking, .speaking: stop()
        }
    }

    public func start() async {
        guard state == .idle || isErrorState else { return }
        let auth = await stt.requestAuthorization()
        guard auth == .authorized else {
            state = .error(authMessage(auth))
            return
        }
        do {
            try configureAudioSession()
            try beginListening()
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    public func stop() {
        sttTask?.cancel()
        sttTask = nil
        generationWatcher?.cancel()
        generationWatcher = nil
        stt.cancel()
        tts.stop()
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        deactivateAudioSession()
        liveTranscript = ""
        inputLevel = 0
        state = .idle
    }

    // MARK: - Listening

    private func beginListening() throws {
        liveTranscript = ""
        lastNonSilentAt = Date()
        let stream = try stt.start(locale: .current)
        sttStream = stream
        try installMicTap()
        state = .listening
        sttTask = Task { [weak self] in
            await self?.consume(stream: stream)
        }
    }

    private func installMicTap() throws {
        let input = engine.inputNode
        let nativeFormat = input.outputFormat(forBus: 0)
        // `installTap` rejects sample-rate==0 formats which happen on
        // first launch before the audio session is fully primed.
        guard nativeFormat.sampleRate > 0 else {
            throw SpeechSTTError.audioSessionFailed("Microphone format not ready.")
        }
        // Pull in 100 ms slices so the VAD can react quickly without
        // pegging the CPU on tiny 5 ms callbacks.
        let frameCount = AVAudioFrameCount(nativeFormat.sampleRate * 0.1)
        input.installTap(onBus: 0, bufferSize: frameCount, format: nativeFormat) { [weak self] buf, _ in
            guard let self else { return }
            let level = Self.rms(buf)
            Task { @MainActor [weak self] in
                self?.handleAudio(buffer: buf, level: level)
            }
        }
        engine.prepare()
        try engine.start()
    }

    private func handleAudio(buffer: AVAudioPCMBuffer, level: Float) {
        // Smooth the meter: 1-pole low-pass so the orb pulse doesn't strobe.
        inputLevel = max(0, min(1, 0.7 * inputLevel + 0.3 * level))
        do {
            try stt.append(buffer: buffer)
        } catch {
            state = .error(error.localizedDescription)
            stop()
            return
        }
        if level >= silenceThreshold {
            lastNonSilentAt = Date()
        } else if state == .listening,
                  !liveTranscript.isEmpty,
                  Date().timeIntervalSince(lastNonSilentAt) >= endOfUtteranceSilence {
            // End of turn: tell the recognizer to flush, the consumer
            // will pick the final result up and submit.
            stt.finish()
        }
    }

    private func consume(stream: AsyncThrowingStream<SpeechPartial, Error>) async {
        do {
            for try await partial in stream {
                liveTranscript = partial.text
                if partial.isFinal {
                    submitFinal(text: partial.text)
                    return
                }
            }
            // Stream ended without an explicit final (some backends do this
            // when `endAudio()` is called after a long silence) — treat the
            // last partial as the final.
            if !liveTranscript.isEmpty {
                submitFinal(text: liveTranscript)
            } else {
                // Nothing heard; just resume listening.
                resumeListeningAfterCycle()
            }
        } catch is CancellationError {
            // user-initiated stop; nothing to do.
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    private func submitFinal(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != lastSubmittedTranscript else {
            resumeListeningAfterCycle()
            return
        }
        lastSubmittedTranscript = trimmed
        // Tear down the mic tap while the model is thinking — keeping
        // the engine running and continuing to feed the recognizer
        // would (a) waste CPU and (b) capture our own TTS output.
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        state = .thinking
        // The assistant placeholder is appended by `ChatSession.send`,
        // so the index of the reply is the count *after* dispatch − 1.
        session.send(trimmed)
        pendingAssistantIndex = max(0, session.messages.count - 1)
        watchForReply()
    }

    private func watchForReply() {
        generationWatcher?.cancel()
        generationWatcher = Task { [weak self] in
            guard let self else { return }
            // Poll the observable session at 10 Hz until generation
            // settles. ChatSession flips `isGenerating = false` in a
            // `defer` at the end of `runGenerationLoop`, so this is a
            // reliable edge.
            while !Task.isCancelled {
                if !self.session.isGenerating { break }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if Task.isCancelled { return }
            await self.speakAssistantReply()
        }
    }

    private func speakAssistantReply() async {
        guard let idx = pendingAssistantIndex,
              idx < session.messages.count,
              session.messages[idx].role == .assistant
        else {
            resumeListeningAfterCycle()
            return
        }
        let reply = session.messages[idx].text.trimmingCharacters(in: .whitespacesAndNewlines)
        pendingAssistantIndex = nil
        guard !reply.isEmpty else {
            resumeListeningAfterCycle()
            return
        }
        state = .speaking
        do {
            try await tts.speak(reply)
        } catch {
            state = .error(error.localizedDescription)
            return
        }
        resumeListeningAfterCycle()
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
        #endif
    }

    private func deactivateAudioSession() {
        #if canImport(UIKit)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
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

    /// Root-mean-square of a Float32 mono buffer. Cheap (~one pass)
    /// and good enough for VAD + meter purposes.
    private static func rms(_ buf: AVAudioPCMBuffer) -> Float {
        guard let chan = buf.floatChannelData?[0] else { return 0 }
        let n = Int(buf.frameLength)
        guard n > 0 else { return 0 }
        var sumSquares: Float = 0
        for i in 0..<n {
            let v = chan[i]
            sumSquares += v * v
        }
        return (sumSquares / Float(n)).squareRoot()
    }
}
