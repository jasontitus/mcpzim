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
    /// Prefer Kokoro when it's both linked and ready (model files
    /// present in `Documents/voices/`); fall back to the system
    /// synthesizer otherwise.
    public static func makeBest(voicesDirectory: URL? = nil, voice: String = "af_heart") -> TTSService {
        #if canImport(KokoroSwift)
        if let dir = voicesDirectory ?? defaultVoicesDir(),
           KokoroTTSService.modelExists(in: dir),
           let kokoro = try? KokoroTTSService(voicesDirectory: dir, voice: voice) {
            return kokoro
        }
        #endif
        return SystemTTSService()
    }

    static func defaultVoicesDir() -> URL? {
        guard let docs = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false
        ) else { return nil }
        return docs.appendingPathComponent("voices", isDirectory: true)
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

/// Lightweight wrapper around the upstream `KokoroTTS` engine. The
/// upstream API expects a model path and a G2P implementation
/// (`MisakiSwift` is the default, English) and returns 24 kHz mono
/// PCM that we play through an `AVAudioEngine` chain.
public final class KokoroTTSService: NSObject, TTSService, @unchecked Sendable {
    public let displayName = "Kokoro v1.0 (MLX, on-device)"
    /// Steady-state with the 82M-param model held fp16. The KV/MLX
    /// caches grow during synthesis; the upstream README suggests
    /// adding ~50 MB of headroom for warm-up buffers.
    public let approximateMemoryMB = 220
    public private(set) var isSpeaking: Bool = false

    /// Filename the user needs to place under `Documents/voices/`.
    /// Matches the canonical Kokoro v1.0 release.
    public static let expectedModelFile = "kokoro-v1_0.safetensors"

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let kokoro: KokoroTTS
    private let voiceName: String
    private var stopFlag = false

    public static func modelExists(in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(expectedModelFile).path)
    }

    public init(voicesDirectory: URL, voice: String) throws {
        let modelURL = voicesDirectory.appendingPathComponent(Self.expectedModelFile)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw TTSError.modelMissing("Place \(Self.expectedModelFile) under Documents/voices/.")
        }
        // The upstream initializer signature is `KokoroTTS(modelPath:
        // g2p:)` (current 1.x). G2P defaults to MisakiSwift for enUS.
        self.kokoro = try KokoroTTS(modelPath: modelURL.path, g2p: .misakiSwift(.enUS))
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
        // Generate the waveform (Float32, 24 kHz mono). Upstream
        // returns `[Float]` along with a sample rate; we convert into
        // an `AVAudioPCMBuffer` and schedule on the player node.
        let samples = try kokoro.generateAudio(text: trimmed, voice: voiceName)
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
            // `stop()` was called mid-utterance; tear the player down so
            // the next call schedules cleanly.
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
