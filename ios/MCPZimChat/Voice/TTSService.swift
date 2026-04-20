// SPDX-License-Identifier: MIT
//
// On-device text-to-speech with two backends behind a tiny protocol:
//
//   • `KokoroTTSService` — wraps the `KokoroSwift` SPM package
//     (https://github.com/mlalma/kokoro-ios). Kokoro v1.0 is an 82M-
//     parameter neural TTS that runs on Apple MLX. Loaded fp16 the
//     model sits at ~165 MB resident; voice embeddings are tiny
//     (~0.5 MB each, 16 voices ship in the standard pack). Synthesis
//     runs ~3.3× real-time on iPhone 13 Pro per the upstream README.
//     The user is responsible for placing the model + voices in the
//     app's `Documents/voices/` directory; until they do we transparently
//     fall back to AVSpeechSynthesizer.
//   • `SystemTTSService` — `AVSpeechSynthesizer`. Always available,
//     uses Apple's compact system voices. Quality is significantly
//     lower than Kokoro but it works on day one with no model files.
//
// Both implementations expose a single async `speak(_:)` that
// completes once the utterance has finished playing (or was
// interrupted by `stop()`), so the voice loop can serialize
// "transcribe → think → speak → listen again" without juggling
// delegates.

import AVFoundation
import Foundation

public protocol TTSService: AnyObject, Sendable {
    var displayName: String { get }
    var isSpeaking: Bool { get }
    /// Approximate steady-state RSS the wrapped engine adds, surfaced
    /// in the model picker / debug pane so the user can budget.
    var approximateMemoryMB: Int { get }

    /// Speak the given text. Returns when playback ends or was
    /// cancelled. Errors are thrown for genuine failures (model not
    /// found, audio session denied) — silence/empty input is a no-op.
    func speak(_ text: String) async throws
    func stop()
}

public enum TTSFactory {
    /// Prefer Kokoro when its assets are downloaded; fall back to
    /// `AVSpeechSynthesizer` otherwise. The model+voices live at
    /// `Application Support/models/kokoro_mlx/` — see
    /// `KokoroAssets.swift` for layout + download URLs.
    public static func makeBest(voice: String = "af_heart") -> TTSService {
        #if canImport(KokoroSwift)
        if KokoroAssets.isDownloaded,
           let kokoro = try? KokoroTTSService(voice: voice) {
            return kokoro
        }
        #endif
        return SystemTTSService()
    }
}

// MARK: - System fallback

public final class SystemTTSService: NSObject, TTSService, @unchecked Sendable {
    public let displayName = "System (AVSpeechSynthesizer)"
    public let approximateMemoryMB = 5
    public private(set) var isSpeaking: Bool = false

    private let synth = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Never>?

    public override init() {
        super.init()
        synth.delegate = self
    }

    public func speak(_ text: String) async throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        // Resolve the host language voice; AVSpeechUtterance picks the
        // user's default voice when this is nil.
        let utt = AVSpeechUtterance(string: trimmed)
        utt.voice = AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
            ?? AVSpeechSynthesisVoice(language: "en-US")
        utt.rate = AVSpeechUtteranceDefaultSpeechRate
        utt.pitchMultiplier = 1.0
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            self.continuation = cont
            self.isSpeaking = true
            self.synth.speak(utt)
        }
    }

    public func stop() {
        guard isSpeaking else { return }
        synth.stopSpeaking(at: .immediate)
    }
}

extension SystemTTSService: AVSpeechSynthesizerDelegate {
    public func speechSynthesizer(_: AVSpeechSynthesizer, didFinish _: AVSpeechUtterance) {
        finishContinuation()
    }
    public func speechSynthesizer(_: AVSpeechSynthesizer, didCancel _: AVSpeechUtterance) {
        finishContinuation()
    }
    private func finishContinuation() {
        isSpeaking = false
        continuation?.resume()
        continuation = nil
    }
}

// MARK: - Kokoro (optional)

#if canImport(KokoroSwift)
import KokoroSwift
import MLX
import MLXUtilsLibrary

/// Wraps the upstream `KokoroTTS` engine (MLX, on-device). Model
/// weights + voice-embedding pack live at the paths defined in
/// `KokoroAssets`. API shape mirrors the working CastCircle port
/// (https://github.com/jasontitus/CastCircle) — call
/// `KokoroTTS.generateAudio(voice:language:text:speed:)` with a
/// specific voice embedding (MLXArray pulled from `voices.npz`)
/// and feed the resulting Float32 PCM through `AVAudioEngine`.
public final class KokoroTTSService: NSObject, TTSService, @unchecked Sendable {
    public let displayName = "Kokoro v1.0 (MLX, on-device)"
    /// Steady-state with the bf16 82M-param model held + MLX cache.
    /// Synthesis allocates more during generation (~70 MB extra);
    /// we `Memory.clearCache()` after each utterance to keep
    /// steady-state bounded.
    public let approximateMemoryMB = 360
    public private(set) var isSpeaking: Bool = false

    /// Fifty-four voices ship in `voices.npz`. Exposed for the UI
    /// picker; `voiceName` selects the current one.
    public static let availableVoices: [String] = [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ]

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let kokoro: KokoroTTS
    private let voices: [String: MLXArray]
    public let voiceName: String
    private var stopFlag = false

    public init(voice: String = "af_heart") throws {
        let modelURL = KokoroAssets.localURL(
            for: KokoroAssets.downloads.first { $0.filename == "kokoro-v1_0.safetensors" }!
        )
        let voicesURL = KokoroAssets.localURL(
            for: KokoroAssets.downloads.first { $0.filename == "voices.npz" }!
        )
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw TTSError.modelMissing("kokoro-v1_0.safetensors missing; download the Kokoro voice first.")
        }
        guard FileManager.default.fileExists(atPath: voicesURL.path) else {
            throw TTSError.modelMissing("voices.npz missing; download the Kokoro voice first.")
        }
        self.kokoro = KokoroTTS(modelPath: modelURL)
        // `isPacked: true` matches the format of the voices.npz we
        // download from the KokoroTestApp repo — MLXUtilsLibrary's
        // newer NpyzReader signature requires it be explicit.
        guard let loadedVoices = NpyzReader.read(fileFromPath: voicesURL, isPacked: true),
              !loadedVoices.isEmpty else {
            throw TTSError.synthesisFailed("voices.npz could not be parsed.")
        }
        self.voices = loadedVoices
        self.voiceName = voice
        super.init()
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode,
                       format: AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1))
    }

    public func speak(_ text: String) async throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        stopFlag = false
        // `voices.npz` keys each voice with an `.npy` suffix, matching
        // the on-disk archive layout.
        guard let embedding = voices[voiceName + ".npy"] else {
            throw TTSError.synthesisFailed("Voice '\(voiceName)' not present in voices.npz.")
        }
        // American voices are `a*`, British are `b*`. KokoroTTS uses
        // the language hint to pick a G2P rule set.
        let language: Language = voiceName.hasPrefix("a") ? .enUS : .enGB
        let (samples, _) = try kokoro.generateAudio(
            voice: embedding, language: language, text: trimmed, speed: 1.0
        )
        // Free MLX intermediate buffers — steady-state would keep
        // creeping without this on successive utterances.
        Memory.clearCache()
        guard !samples.isEmpty else { return }

        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: 24_000, channels: 1, interleaved: false)!
        guard let buf = AVAudioPCMBuffer(pcmFormat: format,
                                         frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw TTSError.synthesisFailed("Could not allocate PCM buffer.")
        }
        buf.frameLength = AVAudioFrameCount(samples.count)
        if let dst = buf.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                dst.update(from: src.baseAddress!, count: samples.count)
            }
        }
        if !engine.isRunning {
            try engine.start()
        }
        isSpeaking = true
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(buf, at: nil, options: []) { [weak self] in
                self?.isSpeaking = false
                cont.resume()
            }
            player.play()
        }
        if stopFlag {
            player.stop()
        }
    }

    public func stop() {
        guard isSpeaking else { return }
        stopFlag = true
        player.stop()
        isSpeaking = false
    }
}

#endif // canImport(KokoroSwift)

public enum TTSError: LocalizedError {
    case modelMissing(String)
    case synthesisFailed(String)
    public var errorDescription: String? {
        switch self {
        case .modelMissing(let s): return s
        case .synthesisFailed(let s): return s
        }
    }
}
