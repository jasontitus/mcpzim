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
    /// 2.0 s lets users pause mid-thought without being cut off; 1.0
    /// was too aggressive for real speech (submitted after ~3 words).
    public var endOfUtteranceSilence: TimeInterval = 2.0
    /// RMS threshold below which we consider the room "silent" for the
    /// VAD timer. 0.02 better ignores breathing/keyboard noise; a
    /// quiet office may need recalibration downward.
    public var silenceThreshold: Float = 0.02

    private let engine = AVAudioEngine()
    private var sttStream: AsyncThrowingStream<SpeechPartial, Error>?
    private var sttTask: Task<Void, Never>?
    private var generationWatcher: Task<Void, Never>?

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
                tts: TTSService = TTSFactory.makeBest(voice: KokoroVoicePreference.current)) {
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

    /// Push a line to the session's debug pane so voice events show up in
    /// the same scrollback the user already uses for chat/model logs.
    private func log(_ message: String) {
        session.debug(message, category: "Voice")
    }

    public func start() async {
        guard state == .idle || isErrorState else { return }
        log("start() requested — backend=\(stt.displayName), tts=\(type(of: tts))")
        let auth = await stt.requestAuthorization()
        guard auth == .authorized else {
            log("STT auth denied: \(auth)")
            state = .error(authMessage(auth))
            return
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
        log("beginListening (silence=\(endOfUtteranceSilence)s, floor=\(silenceThreshold))")
        liveTranscript = ""
        carriedTranscript = ""
        lastNonSilentAt = Date()
        weRequestedFinish = false
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
        // Always remove any previous tap before installing — `installTap`
        // raises an NSException ("only one tap can be installed on any
        // bus at a time") if one is already bound, and that kills the
        // whole app because Obj-C exceptions can't be caught from Swift.
        input.removeTap(onBus: 0)
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
            } else if state == .listening {
                // Recognizer self-terminated before we asked it to. Keep
                // listening with a fresh task; don't double-restart if some
                // other code path already swapped us out.
                try? restartRecognizer()
            }
        } catch is CancellationError {
            // user-initiated stop; nothing to do.
        } catch {
            state = .error(error.localizedDescription)
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
            resumeListeningAfterCycle()
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
            await self.streamAssistantReply()
        }
    }

    /// Stream sentences into Kokoro as they arrive from the chat
    /// session, so playback begins ~2–4 s after the model starts
    /// replying instead of waiting for the full 15–40 s generation to
    /// finish. Sentences are cut at `.`, `!`, or `?` followed by a
    /// space/newline. Anything not yet cut gets flushed when
    /// `isGenerating` flips to false.
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
        log("TTS streaming start (polling for sentences)")
        let t0 = Date()
        state = .speaking

        var spokenUpTo = 0   // characters already dispatched to TTS
        var sawAnyText = false

        while !Task.isCancelled {
            guard idx < session.messages.count else { break }
            let full = Self.sanitizeForSpeech(session.messages[idx].text)
            if full.count > spokenUpTo {
                sawAnyText = true
                // Slice out the new suffix and extract complete sentences.
                let newFragment = String(full.suffix(full.count - spokenUpTo))
                if let (completeSoFar, consumed) = Self.takeCompleteSentences(
                    newFragment, generating: session.isGenerating)
                {
                    let toSpeak = completeSoFar.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !toSpeak.isEmpty {
                        do {
                            try await tts.speakChunk(toSpeak)
                        } catch {
                            log("TTS chunk failed: \(error.localizedDescription)")
                        }
                    }
                    // `consumed` is the number of characters we've
                    // committed to TTS — advance by exactly that so
                    // the next iteration's fragment starts right after
                    // the last sentence boundary. (Earlier version
                    // inverted this and replayed the same sentence.)
                    spokenUpTo += consumed
                }
            }
            if !session.isGenerating {
                // One more pass to catch anything that arrived after
                // the last iteration — then flush any trailing fragment.
                let finalText = Self.sanitizeForSpeech(session.messages[idx].text)
                if finalText.count > spokenUpTo {
                    let tail = String(finalText.suffix(finalText.count - spokenUpTo))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !tail.isEmpty {
                        try? await tts.speakChunk(tail)
                    }
                }
                break
            }
            try? await Task.sleep(nanoseconds: 150_000_000)
        }
        if !sawAnyText {
            log("streamAssistantReply: reply was empty")
            resumeListeningAfterCycle()
            return
        }
        await tts.awaitPlayback()
        log(String(format: "TTS done in %.2fs", Date().timeIntervalSince(t0)))
        resumeListeningAfterCycle()
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
        ]
        for pat in closed {
            t = t.replacingOccurrences(of: pat, with: "", options: .regularExpression)
        }
        if let r = t.range(of: #"<\|?tool[_a-z]*"#, options: .regularExpression) {
            t = String(t[..<r.lowerBound])
        }
        if let r = t.range(of: #"<tool[_a-z]*"#, options: .regularExpression) {
            t = String(t[..<r.lowerBound])
        }
        for lit in ["<tool_call|>", "<tool_response|>", "<|\"|>", "<|\""] {
            t = t.replacingOccurrences(of: lit, with: "")
        }
        return t
    }

    /// Split `text` into (prefix-of-complete-sentences, remainder-start-index).
    /// "Complete" = ends with `.!?` then whitespace / EOF. When the
    /// model is still generating, only fire sentences with a trailing
    /// terminator (keep buffering). When generation is done, treat the
    /// whole thing as complete. Returns nil if no sentence boundary.
    private static func takeCompleteSentences(
        _ text: String, generating: Bool
    ) -> (complete: String, leftoverStart: Int)? {
        if !generating, !text.isEmpty {
            return (text, text.count)
        }
        // Walk from the end looking for the last sentence boundary.
        var lastBoundary: Int? = nil
        let chars = Array(text)
        for i in 0..<chars.count {
            let c = chars[i]
            if c == "." || c == "!" || c == "?" {
                // Require a following whitespace or end-of-buffer —
                // avoids splitting "e.g." and decimal numbers.
                if i + 1 >= chars.count || chars[i + 1].isWhitespace {
                    lastBoundary = i + 1
                }
            }
        }
        guard let b = lastBoundary, b > 0 else { return nil }
        let complete = String(chars[0..<b])
        return (complete, b)
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
        // Keep the display awake while voice chat is active — otherwise
        // the phone auto-locks mid-"thinking" and the user loses the mic
        // tap + Kokoro playback. Re-enabled in `stop()`.
        UIApplication.shared.isIdleTimerDisabled = true
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
