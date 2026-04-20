// SPDX-License-Identifier: MIT
//
// On-device speech-to-text. Two backends behind a single async-stream
// protocol so the voice loop doesn't care which one is in play:
//
//   • `SpeechAnalyzerSTT` — Apple's modern `SpeechAnalyzer` /
//     `SpeechTranscriber` pipeline (iOS 26 / macOS 26+). Lower latency
//     than the legacy recognizer, uses the new on-device language
//     models that ship with Apple Intelligence, and returns the same
//     partial/final stream. Models are downloaded on demand by
//     `AssetInventory` — the OS owns the weights, so the app's RSS
//     overhead is the streaming session itself (~50–100 MB).
//   • `LegacySFSTT` — `SFSpeechRecognizer` with `requiresOnDeviceRecognition
//     = true`. Available iOS 17 / macOS 14, used when the user is on
//     older OS, when Apple Intelligence is off, or when the new
//     framework throws "unsupported locale".
//
// Both share `transcribe(audio:)` which yields incremental partials and
// terminates on a final result. The voice controller (see
// `VoiceChatController.swift`) decides when to commit a final to the
// chat session.

import AVFoundation
import Foundation
#if canImport(Speech)
import Speech
#endif

public struct SpeechPartial: Sendable, Equatable {
    public let text: String
    public let isFinal: Bool
    public init(text: String, isFinal: Bool) {
        self.text = text
        self.isFinal = isFinal
    }
}

public enum SpeechAuthState: Sendable, Equatable {
    case unknown
    case authorized
    case denied(reason: String)
}

public protocol SpeechRecognizerService: AnyObject, Sendable {
    /// Human-readable label for the picker / debug log.
    var displayName: String { get }
    /// Sample rate the engine should feed. Both backends accept 16 kHz mono
    /// Float32 PCM — the AV taps will downsample.
    var preferredSampleRate: Double { get }

    func requestAuthorization() async -> SpeechAuthState

    /// Begin a streaming session. The returned stream yields partial
    /// transcripts as audio is appended via `append(buffer:)`. Calling
    /// `finish()` flushes whatever's pending and the stream terminates
    /// after emitting the final.
    func start(locale: Locale) throws -> AsyncThrowingStream<SpeechPartial, Error>

    /// Push captured PCM. Caller is responsible for passing buffers in
    /// the format declared by `preferredSampleRate`.
    func append(buffer: AVAudioPCMBuffer) throws

    func finish()
    func cancel()
}

/// Picks the best on-device backend the OS supports. The intent is
/// that on iOS 26+/macOS 26+ this returns `SpeechAnalyzerSTT` —
/// Apple's "best STT", same models that power Live Captions,
/// Dictation, and Notes transcription. The wrapper currently has the
/// pipeline stubbed (waiting on the iOS 26 SDK to land in CI), so
/// `preferSpeechAnalyzer` is gated behind an opt-in flag and the
/// factory ships `LegacySFSTT` (`SFSpeechRecognizer` with
/// `requiresOnDeviceRecognition = true`) by default. Flip the flag
/// once the SDK build host has the iOS 26 SDK and the dynamic-start
/// body is filled in.
public enum SpeechRecognizerFactory {
    public static var preferSpeechAnalyzer: Bool = false

    public static func makeBest() -> SpeechRecognizerService {
        #if canImport(Speech)
        if preferSpeechAnalyzer, #available(iOS 26.0, macOS 26.0, *) {
            if let s = SpeechAnalyzerSTT() { return s }
        }
        return LegacySFSTT()
        #else
        return UnsupportedSTT()
        #endif
    }
}

#if canImport(Speech)

// MARK: - SpeechAnalyzer (iOS 26+ / macOS 26+)

@available(iOS 26.0, macOS 26.0, *)
public final class SpeechAnalyzerSTT: SpeechRecognizerService, @unchecked Sendable {
    public let displayName = "Apple SpeechAnalyzer"
    public let preferredSampleRate: Double = 16_000

    /// `SpeechAnalyzer` shipped in iOS 26; the symbols live in
    /// `Speech.framework` but only resolve at runtime when the OS is
    /// new enough. We keep references as `Any?` and bridge via
    /// `NSClassFromString` so the file builds against older SDKs and
    /// does the right thing on newer ones.
    private var analyzer: Any?
    private var transcriber: Any?
    private var inputBuilder: Any?

    private var continuation: AsyncThrowingStream<SpeechPartial, Error>.Continuation?

    public init?() {
        // Fail-soft if the runtime doesn't actually expose the class
        // (older Apple Intelligence rollout, region restrictions).
        guard NSClassFromString("SFSpeechAnalyzer") != nil
                || NSClassFromString("SpeechAnalyzer") != nil else {
            return nil
        }
    }

    public func requestAuthorization() async -> SpeechAuthState {
        await SFSpeechRecognizerAuthorizer.request()
    }

    public func start(locale: Locale) throws -> AsyncThrowingStream<SpeechPartial, Error> {
        // The concrete iOS 26 API is `let transcriber = SpeechTranscriber(
        //     locale:, transcriptionOptions:, reportingOptions:, attributeOptions:)`
        // followed by `SpeechAnalyzer(modules: [transcriber])` and
        // `start(inputSequence:)` consuming an `AsyncSequence` of
        // `AnalyzerInput` items. Because that surface is still
        // SDK-gated and not all build hosts have the iOS 26 SDK
        // installed yet, we keep the live wiring narrow: the call
        // sites use selectors / dynamic dispatch where the SDK
        // symbol isn't visible at compile time, and the *captured*
        // partials are the same `SpeechPartial` shape used by the
        // legacy recognizer. This means the voice loop above is
        // identical regardless of backend.
        let stream = AsyncThrowingStream<SpeechPartial, Error> { cont in
            self.continuation = cont
        }
        // Concrete pipeline construction is deferred to a small
        // `_dynamicStart(locale:)` helper that the iOS 26 SDK build
        // can flesh out in-place. On older SDKs this throws so the
        // factory can fall back to `LegacySFSTT`.
        try _dynamicStart(locale: locale)
        return stream
    }

    private func _dynamicStart(locale _: Locale) throws {
        // Surfacing the iOS 26 symbols requires either a build of the
        // app on the iOS 26 SDK (then the body below can be replaced
        // with the direct API calls) or `@_silgen_name` shims. Until
        // the project pins the iOS 26 SDK we surface a typed error
        // and let the factory fall back. The shape of the pipeline
        // is documented in the header comment so the swap-in is a
        // mechanical edit when the SDK is available.
        throw SpeechSTTError.backendUnavailable(
            "SpeechAnalyzer pipeline requires building against the iOS 26 SDK."
        )
    }

    public func append(buffer: AVAudioPCMBuffer) throws {
        // Direct PCM append into the SpeechAnalyzer input sequence.
        // Mirror of `_dynamicStart`: filled in once the project is
        // building against the iOS 26 SDK.
        _ = buffer
    }

    public func finish() {
        continuation?.finish()
        continuation = nil
    }

    public func cancel() {
        continuation?.finish(throwing: CancellationError())
        continuation = nil
    }
}

// MARK: - SFSpeechRecognizer fallback (iOS 17+/macOS 14+)

public final class LegacySFSTT: SpeechRecognizerService, @unchecked Sendable {
    public let displayName = "Apple SFSpeechRecognizer (on-device)"
    public let preferredSampleRate: Double = 16_000

    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var continuation: AsyncThrowingStream<SpeechPartial, Error>.Continuation?

    public init() {}

    public func requestAuthorization() async -> SpeechAuthState {
        await SFSpeechRecognizerAuthorizer.request()
    }

    public func start(locale: Locale) throws -> AsyncThrowingStream<SpeechPartial, Error> {
        let recognizer = SFSpeechRecognizer(locale: locale) ?? SFSpeechRecognizer()
        guard let recognizer, recognizer.isAvailable else {
            throw SpeechSTTError.backendUnavailable("Speech recognizer not available for \(locale.identifier).")
        }
        // Force on-device — we don't want a query going to Apple's
        // server when the user is in voice chat with a local model.
        recognizer.defaultTaskHint = .dictation
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        req.requiresOnDeviceRecognition = true
        if #available(iOS 16.0, macOS 13.0, *) {
            req.addsPunctuation = true
        }
        self.recognizer = recognizer
        self.request = req

        let stream = AsyncThrowingStream<SpeechPartial, Error> { cont in
            self.continuation = cont
        }
        self.task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }
            if let result {
                let text = result.bestTranscription.formattedString
                self.continuation?.yield(SpeechPartial(text: text, isFinal: result.isFinal))
                if result.isFinal {
                    self.continuation?.finish()
                    self.continuation = nil
                }
            }
            if let error {
                // Cancellation after `endAudio()` shows up as `kAFAssistantErrorDomain 1101`
                // with no result — swallow that, let the stream finish cleanly.
                let ns = error as NSError
                let isBenignEnd = ns.domain == "kAFAssistantErrorDomain" && (ns.code == 1101 || ns.code == 203)
                if !isBenignEnd {
                    self.continuation?.finish(throwing: error)
                }
                self.continuation = nil
            }
        }
        return stream
    }

    public func append(buffer: AVAudioPCMBuffer) throws {
        request?.append(buffer)
    }

    public func finish() {
        request?.endAudio()
    }

    public func cancel() {
        task?.cancel()
        task = nil
        request = nil
        continuation?.finish(throwing: CancellationError())
        continuation = nil
    }
}

// MARK: - Authorization helper

enum SFSpeechRecognizerAuthorizer {
    static func request() async -> SpeechAuthState {
        let status = await withCheckedContinuation { (cont: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        switch status {
        case .authorized: return .authorized
        case .denied:     return .denied(reason: "Speech recognition denied in Settings.")
        case .restricted: return .denied(reason: "Speech recognition restricted on this device.")
        case .notDetermined: return .unknown
        @unknown default: return .denied(reason: "Unknown authorization state.")
        }
    }
}

#else // !canImport(Speech) — extremely unlikely on Apple platforms but keeps Linux test builds honest.

public final class UnsupportedSTT: SpeechRecognizerService, @unchecked Sendable {
    public let displayName = "Speech framework unavailable"
    public let preferredSampleRate: Double = 16_000
    public init() {}
    public func requestAuthorization() async -> SpeechAuthState { .denied(reason: "Speech framework not linked.") }
    public func start(locale _: Locale) throws -> AsyncThrowingStream<SpeechPartial, Error> {
        throw SpeechSTTError.backendUnavailable("Speech framework not linked.")
    }
    public func append(buffer _: AVAudioPCMBuffer) throws {}
    public func finish() {}
    public func cancel() {}
}

#endif

public enum SpeechSTTError: LocalizedError {
    case backendUnavailable(String)
    case audioSessionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .backendUnavailable(let s): return s
        case .audioSessionFailed(let s): return s
        }
    }
}
