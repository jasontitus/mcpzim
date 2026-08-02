// SPDX-License-Identifier: MIT
//
// On-device text-to-speech backends behind a tiny protocol:
//
//   • `Supertonic3TTSService` — Core ML with a fixed-shape INT8
//     VectorEstimator running primarily on the Apple Neural Engine.
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

public enum TTSChunkBoundary: Sendable {
    case sentence
    case clause
    case softWrap
    case final
}

/// Bring neural TTS PCM up to a consistent spoken-word level before it reaches
/// AVAudioEngine. Both Kokoro and Supertonic can emit substantially quieter
/// native samples than Apple's media route expects. RMS normalization with a
/// peak ceiling raises quiet output without clipping or applying an unsafe,
/// fixed boost to already-loud voices.
enum TTSPlaybackLevel {
    /// Gain to bring the chunk to the target level (1.0 = leave as-is).
    /// One scan pass, no allocation — callers fold the multiply into
    /// their PCM-buffer write so the audio path makes ONE copy per chunk
    /// instead of a normalized intermediate array plus the buffer copy.
    static func gain(for samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 1.0 }
        var sumSquares: Float = 0
        var peak: Float = 0
        for sample in samples {
            sumSquares += sample * sample
            peak = max(peak, abs(sample))
        }
        let rms = (sumSquares / Float(samples.count)).squareRoot()
        guard rms > 0.002, peak > 0 else { return 1.0 }
        let targetRMS: Float = 0.12       // approximately -18 dBFS RMS
        let maximumGain: Float = 4.0      // at most +12 dB
        let desiredGain = min(maximumGain, max(1.0, targetRMS / rms))
        let clippingSafeGain = 0.95 / peak
        let gain = min(desiredGain, clippingSafeGain)
        guard abs(gain - 1.0) > 0.02 else { return 1.0 }
        return gain
    }

}

public protocol TTSService: AnyObject, Sendable {
    var displayName: String { get }
    var isSpeaking: Bool { get }
    /// Approximate steady-state RSS the wrapped engine adds, surfaced
    /// in the model picker / debug pane so the user can budget.
    var approximateMemoryMB: Int { get }
    /// Conservative peak headroom needed while producing one conversational
    /// chunk. The controller uses this for overlap/preparation decisions; it
    /// is deliberately separate from the steady-state UI estimate.
    var peakSynthesisMemoryMB: Int { get }

    /// Optional upper bound for text handed to one streaming synthesis call.
    /// Backends with smaller encoder windows can use this to keep their own
    /// hidden chunkers from creating audible end-of-utterance seams.
    var preferredStreamingChunkCharacters: Int? { get }

    /// Optional lower bound before the controller may split unfinished prose
    /// at a clause or word boundary. A backend that benefits from longer
    /// prosody windows can raise this while still emitting completed short
    /// sentences immediately.
    var preferredStreamingMinimumCharacters: Int? { get }

    /// Speak the given text. Returns when playback ends or was
    /// cancelled. Errors are thrown for genuine failures (model not
    /// found, audio session denied) — silence/empty input is a no-op.
    func speak(_ text: String) async throws

    /// Streaming variant: synthesise + schedule playback, return as
    /// soon as the samples are queued. Subsequent calls keep the
    /// underlying audio engine full, so sentences play back-to-back
    /// without gaps. Callers should `await awaitPlayback()` at the end
    /// of the stream to block until all queued buffers finish.
    /// Default implementation delegates to `speak`.
    func speakChunk(_ text: String) async throws

    /// Streaming entry point with the source boundary that ended this chunk.
    /// Backends may use it to distinguish an artificial word wrap from real
    /// sentence or clause prosody.
    func speakChunk(_ text: String, boundary: TTSChunkBoundary) async throws

    /// Block until all previously queued audio has finished playing.
    /// Default is a no-op (most backends are synchronous).
    func awaitPlayback() async

    /// Prepare expensive model/G2P execution before the first conversational
    /// response. Implementations must not play audio. The voice controller
    /// invokes this off the main actor only when memory headroom is healthy.
    func prepareForConversation() async throws

    func stop()
}

public extension TTSService {
    var peakSynthesisMemoryMB: Int { approximateMemoryMB }
    var preferredStreamingChunkCharacters: Int? { nil }
    var preferredStreamingMinimumCharacters: Int? { nil }
    func speakChunk(_ text: String) async throws { try await speak(text) }
    func speakChunk(_ text: String, boundary: TTSChunkBoundary) async throws {
        try await speakChunk(text)
    }
    func awaitPlayback() async {}
    func prepareForConversation() async throws {}
}

public enum TTSFactory {
    /// Build the selected on-device backend. New installs prefer Supertonic 3;
    /// Kokoro remains available for listening comparisons and the system voice
    /// is the zero-download fallback.
    public static func makeBest(voice: String = "af_heart") -> TTSService {
        if TTSBackendPreference.current == .supertonic {
            #if canImport(FluidAudio)
            return Supertonic3TTSService(voice: SupertonicVoicePreference.current)
            #endif
        }
        #if canImport(KokoroSwift)
        if TTSBackendPreference.current != .system,
           KokoroAssets.isDownloaded,
           let kokoro = try? KokoroTTSService(voice: voice) {
            return kokoro
        }
        #endif
        return SystemTTSService()
    }
}

public enum TTSBackendPreference: String, CaseIterable, Sendable {
    case supertonic
    case kokoro
    case system

    private static let key = "tts.backend"

    public var displayName: String {
        switch self {
        case .supertonic: return "Supertonic 3 (ANE INT8)"
        case .kokoro: return "Kokoro v1.0 (MLX)"
        case .system: return "System voice"
        }
    }

    public static var current: TTSBackendPreference {
        get {
            guard let raw = UserDefaults.standard.string(forKey: key),
                  let value = TTSBackendPreference(rawValue: raw)
            else { return .supertonic }
            return value
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: key) }
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
    /// Measured by MCPZimTTSBenchCLI on Apple silicon with a representative
    /// 92-character, punctuation-heavy chunk. MLX briefly reached 2.74 GB
    /// above the settled prepared footprint; round up for the iOS safety gate.
    public let peakSynthesisMemoryMB = 2_800
    #if os(macOS)
    /// Mac has enough memory and substantially faster MLX execution to give
    /// Kokoro a full prosody window. This avoids turning commas in a long
    /// sentence into separate, sentence-like utterances. Keep a margin below
    /// Kokoro's internal 400-character/510-phoneme safety cap because text
    /// normalization can expand numbers and abbreviations.
    public let preferredStreamingChunkCharacters: Int? = 360
    public let preferredStreamingMinimumCharacters: Int? = 180
    #else
    /// iPhone retains the controller's conservative 112/56-character window:
    /// Kokoro's measured synthesis transient is large enough that increasing
    /// it should be benchmarked on-device before changing this.
    public let preferredStreamingChunkCharacters: Int? = nil
    public let preferredStreamingMinimumCharacters: Int? = nil
    #endif
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
    private var isPrepared = false

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
        player.volume = 1.0
        engine.mainMixerNode.outputVolume = 1.0
    }

    /// Exercise G2P plus the length-dependent Kokoro graph with a realistic
    /// first-clause shape, then discard the samples. This moves MLX's first
    /// execution cost into the listening window rather than the first answer.
    public func prepareForConversation() async throws {
        guard !isPrepared else { return }
        guard let embedding = voices[voiceName + ".npy"] else {
            throw TTSError.synthesisFailed("Voice '\(voiceName)' not present in voices.npz.")
        }
        let language: Language = voiceName.hasPrefix("a") ? .enUS : .enGB
        let phrase = Self.prepForTTS(
            "Preparing a natural voice to answer the next question clearly."
        )
        let (samples, _) = try kokoro.generateAudio(
            voice: embedding, language: language, text: phrase, speed: 1.0
        )
        Memory.clearCache()
        guard !samples.isEmpty else {
            throw TTSError.synthesisFailed("Kokoro preparation produced no audio.")
        }
        isPrepared = true
    }

    /// Render through the same normalization, G2P, voice, and chunking path
    /// used by `speakChunk`, but return PCM instead of scheduling playback.
    /// The isolated Mac benchmark uses this to produce an audible reference
    /// file without capturing or resampling system audio.
    func renderForBenchmark(_ text: String) throws -> (samples: [Float], sampleRate: Int) {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return ([], 24_000) }
        guard let embedding = voices[voiceName + ".npy"] else {
            throw TTSError.synthesisFailed("Voice '\(voiceName)' not present in voices.npz.")
        }
        let language: Language = voiceName.hasPrefix("a") ? .enUS : .enGB
        let chunks = Self.chunkForTTS(Self.prepForTTS(raw))
        var combined: [Float] = []
        for chunk in chunks {
            let (samples, tokens) = try kokoro.generateAudio(
                voice: embedding, language: language, text: chunk, speed: 1.0
            )
            if ProcessInfo.processInfo.environment["MCPZIM_TTS_TRACE_PHONEMES"] == "1",
               let tokens {
                let trace = tokens.map { token in
                    "\(token.text)=\(token.phonemes ?? "∅")"
                }.joined(separator: " | ")
                FileHandle.standardError.write(
                    Data(("[KokoroG2P] \(trace)\n").utf8))
            }
            combined.append(contentsOf: samples)
            Memory.clearCache()
        }
        return (combined, 24_000)
    }

    public func speak(_ text: String) async throws {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }
        // Cap total TTS input. A full PA→SF turn-by-turn is 2–3 KB of
        // text, which Kokoro can handle chunk-by-chunk but whose MLX
        // transient allocations — stacked on top of Gemma's retained
        // pool — push past the 6144 MB per-process cap on iPhone 17
        // Pro Max. Trim at 1200 chars (~60 s of speech); the full text
        // stays visible in the chat log.
        let trimmed: String
        if raw.count > 1200 {
            let cutoff = raw.index(raw.startIndex, offsetBy: 1200)
            trimmed = String(raw[..<cutoff]) + " … the rest is in the chat log."
        } else {
            trimmed = raw
        }
        stopFlag = false
        // `voices.npz` keys each voice with an `.npy` suffix, matching
        // the on-disk archive layout.
        guard let embedding = voices[voiceName + ".npy"] else {
            throw TTSError.synthesisFailed("Voice '\(voiceName)' not present in voices.npz.")
        }
        // American voices are `a*`, British are `b*`. KokoroTTS uses
        // the language hint to pick a G2P rule set.
        let language: Language = voiceName.hasPrefix("a") ? .enUS : .enGB

        // Kokoro caps each utterance at 510 phoneme tokens. Normalise
        // the text first (Kokoro's G2P mangles things like `(unnamed
        // road)` and `CA 82`), then split into sentence-ish chunks, and
        // synthesise chunk N+1 while chunk N is already queued on the
        // player so `AVAudioPlayerNode`'s FIFO plays them back-to-back
        // without the audible gap you'd get by awaiting each chunk's
        // playback before synthesising the next.
        let cleaned = Self.prepForTTS(trimmed)
        let chunks = Self.chunkForTTS(cleaned)
        if !engine.isRunning { try engine.start() }
        isSpeaking = true
        defer { isSpeaking = false }

        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: 24_000, channels: 1, interleaved: false)!
        let lastIdx = chunks.count - 1
        for (i, chunk) in chunks.enumerated() {
            if stopFlag { break }
            let (nativeSamples, _) = try kokoro.generateAudio(
                voice: embedding, language: language, text: chunk, speed: 1.0
            )
            // Gain applied during the buffer copy — one pass, no
            // normalized intermediate array per chunk.
            let gain = TTSPlaybackLevel.gain(for: nativeSamples)
            Memory.clearCache()
            guard !nativeSamples.isEmpty else { continue }
            guard let buf = AVAudioPCMBuffer(pcmFormat: format,
                                             frameCapacity: AVAudioFrameCount(nativeSamples.count)) else {
                throw TTSError.synthesisFailed("Could not allocate PCM buffer.")
            }
            buf.frameLength = AVAudioFrameCount(nativeSamples.count)
            if let dst = buf.floatChannelData?[0] {
                nativeSamples.withUnsafeBufferPointer { src in
                    if gain == 1.0 {
                        dst.update(from: src.baseAddress!, count: nativeSamples.count)
                    } else {
                        for j in 0..<nativeSamples.count { dst[j] = src[j] * gain }
                    }
                }
            }
            if i == lastIdx {
                // Only wait on the tail buffer — the earlier ones are
                // queued and will play seamlessly.
                await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                    player.scheduleBuffer(buf, at: nil, options: []) { cont.resume() }
                    if !player.isPlaying { player.play() }
                }
            } else {
                player.scheduleBuffer(buf, at: nil, options: [], completionHandler: nil)
                if !player.isPlaying { player.play() }
            }
        }
        if stopFlag { player.stop() }
    }

    /// Streaming entry point: synthesise `text` and queue it on the
    /// player, but do NOT wait for playback. Next sentence can then
    /// start synthesising immediately; AVAudioPlayerNode keeps playing
    /// queued buffers in order so the listener hears seamless speech.
    public func speakChunk(_ text: String) async throws {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }
        stopFlag = false
        guard let embedding = voices[voiceName + ".npy"] else {
            throw TTSError.synthesisFailed("Voice '\(voiceName)' not present in voices.npz.")
        }
        let language: Language = voiceName.hasPrefix("a") ? .enUS : .enGB
        let cleaned = Self.prepForTTS(raw)
        let subChunks = Self.chunkForTTS(cleaned)
        if !engine.isRunning { try engine.start() }
        isSpeaking = true
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: 24_000, channels: 1, interleaved: false)!
        for sub in subChunks {
            if stopFlag { break }
            let (nativeSamples, _) = try kokoro.generateAudio(
                voice: embedding, language: language, text: sub, speed: 1.0
            )
            // Gain applied during the buffer copy — one pass, no
            // normalized intermediate array per chunk.
            let gain = TTSPlaybackLevel.gain(for: nativeSamples)
            Memory.clearCache()
            guard !nativeSamples.isEmpty else { continue }
            guard let buf = AVAudioPCMBuffer(pcmFormat: format,
                                             frameCapacity: AVAudioFrameCount(nativeSamples.count)) else {
                throw TTSError.synthesisFailed("Could not allocate PCM buffer.")
            }
            buf.frameLength = AVAudioFrameCount(nativeSamples.count)
            if let dst = buf.floatChannelData?[0] {
                nativeSamples.withUnsafeBufferPointer { src in
                    if gain == 1.0 {
                        dst.update(from: src.baseAddress!, count: nativeSamples.count)
                    } else {
                        for j in 0..<nativeSamples.count { dst[j] = src[j] * gain }
                    }
                }
            }
            player.scheduleBuffer(buf, at: nil, options: [], completionHandler: nil)
            if !player.isPlaying { player.play() }
        }
    }

    /// Block until the player's queue is drained.
    public func awaitPlayback() async {
        // Schedule an empty "marker" buffer at the tail of the queue —
        // its completion fires when all prior buffers have played.
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: 24_000, channels: 1, interleaved: false)!
        guard let marker = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1) else { return }
        marker.frameLength = 1
        marker.floatChannelData?[0][0] = 0
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(marker, at: nil, options: []) { cont.resume() }
            if !player.isPlaying { player.play() }
        }
        isSpeaking = false
    }

    /// Normalise text before it reaches Kokoro's G2P. The phonemiser is
    /// fragile around parenthesised phrases ("(unnamed road)" → "memenax"
    /// from the user's route trace), roadway abbreviations ("CA 82" →
    /// "see-ay eighty-two"), and markdown syntax. Keep this cheap +
    /// conservative: string substitution only, no lookups.
    static func prepForTTS(_ s: String) -> String {
        var t = s
        // Parenthesised notes confuse G2P. Keep content, drop brackets.
        // Route turn-by-turn comes out as "Hamilton Ave for 0.2 mi (~0.5 min)"
        // which Kokoro then reads as a phonetic gibberish — the "mi" /
        // "min" / "~" combo in particular. Expand before the phonemiser
        // ever sees it.
        t = t.replacingOccurrences(of: "(unnamed road)", with: "an unnamed road")
        t = t.replacingOccurrences(of: "(unnamed)", with: "an unnamed road")
        // Regex-based unit expansion. Word boundary on the right so we
        // don't mangle identifiers like "mission" or "minnesota".
        let unitPatterns: [(pattern: String, replacement: String)] = [
            (#"(\d+(?:\.\d+)?)\s*mi\b"#,  "$1 miles"),
            (#"(\d+(?:\.\d+)?)\s*min\b"#, "$1 minutes"),
            (#"(\d+(?:\.\d+)?)\s*km\b"#,  "$1 kilometres"),
            (#"(\d+(?:\.\d+)?)\s*hr\b"#,  "$1 hours"),
            (#"(\d+(?:\.\d+)?)\s*h\b"#,   "$1 hours"),
            (#"(\d+(?:\.\d+)?)\s*ft\b"#,  "$1 feet"),
            (#"(\d+(?:\.\d+)?)\s*m\b"#,   "$1 metres"),
        ]
        for (pat, rep) in unitPatterns {
            t = t.replacingOccurrences(of: pat, with: rep, options: .regularExpression)
        }
        // `~N` (approximately) → "about N". Same for `≈`.
        t = t.replacingOccurrences(of: "~", with: "about ")
        t = t.replacingOccurrences(of: "≈", with: "about ")
        // Decimal numbers — Kokoro's G2P ("22.4") collapses to silence
        // on some locales, which is why "22.4 miles" was being read as
        // "about miles". Expand to "22 point 4 miles" which the
        // phonemiser handles reliably.
        t = t.replacingOccurrences(
            of: #"(\d+)\.(\d+)"#, with: "$1 point $2",
            options: .regularExpression
        )
        // Drop parens after unit expansion (keeps the content flowing).
        t = t.replacingOccurrences(of: "(", with: ", ")
        t = t.replacingOccurrences(of: ")", with: "")
        // `Dr.` and `St.` are ambiguous in conversational answers. Expanding
        // them globally made "Dr. Vladimir" become "Drive Vladimir" and
        // "St. Petersburg" become "Street Petersburg." Treat them as road
        // suffixes only in the route forms we actually emit: before a comma,
        // distance/direction continuation, or the end of the utterance.
        let contextualStreetSuffixes: [(pattern: String, replacement: String)] = [
            (#"\bDr\.(?=\s*(?:,|$|\d|for\b|toward\b|then\b|and\b))"#, "Drive"),
            (#"\bSt\.(?=\s*(?:,|$|\d|for\b|toward\b|then\b|and\b))"#, "Street"),
        ]
        for (pattern, replacement) in contextualStreetSuffixes {
            t = t.replacingOccurrences(
                of: pattern, with: replacement, options: .regularExpression)
        }

        // Unambiguous highway / road abbreviations.
        let replacements: [(String, String)] = [
            ("CA ", "California State Route "),
            ("US ", "U.S. Route "),
            ("I-", "Interstate "),
            ("Ave.", "Avenue"), ("Blvd.", "Boulevard"),
            ("Rd.", "Road"), ("Ln.", "Lane"),
            ("Ct.", "Court"), ("Pl.", "Place"), ("Pkwy.", "Parkway"),
            ("Expwy.", "Expressway"), ("Expy.", "Expressway"),
            ("Hwy.", "Highway"),
        ]
        for (src, dst) in replacements {
            t = t.replacingOccurrences(of: src, with: dst)
        }
        // Markdown we don't want spoken literally.
        t = t.replacingOccurrences(of: "**", with: "")
        t = t.replacingOccurrences(of: "__", with: "")
        t = t.replacingOccurrences(of: "`", with: "")
        // Kokoro's G2P treats hyphens in compound words as phrase
        // breaks, so "deep-dish pizza" comes out with a noticeable
        // pause between each word. Collapsing to a space lets it
        // phonemise as a single phrase. Only catches intra-word
        // hyphens (letter-dash-letter); preserves dashes that mean
        // "minus" or a range ("2024-2025").
        t = t.replacingOccurrences(
            of: #"(?<=\p{L})-(?=\p{L})"#, with: " ",
            options: .regularExpression
        )
        // Collapse any doubled whitespace introduced by the above.
        while t.contains("  ") { t = t.replacingOccurrences(of: "  ", with: " ") }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Split a reply into ≤~400-character chunks at sentence boundaries.
    /// 400 chars stays comfortably below Kokoro's 510-phoneme cap even
    /// for phoneme-dense material. Falls back to soft-wrapping at
    /// commas / whitespace if a single "sentence" is still too long.
    static func chunkForTTS(_ text: String, limit: Int = 400) -> [String] {
        if text.count <= limit { return [text] }
        var out: [String] = []
        var current = ""
        // Split while keeping the terminator attached.
        var buf = ""
        for ch in text {
            buf.append(ch)
            if ".!?".contains(ch) {
                if current.count + buf.count > limit, !current.isEmpty {
                    out.append(current.trimmingCharacters(in: .whitespaces))
                    current = ""
                }
                current.append(buf)
                buf = ""
            }
        }
        if !buf.isEmpty { current.append(buf) }
        if !current.isEmpty { out.append(current.trimmingCharacters(in: .whitespaces)) }
        // Any chunk still over the cap gets soft-wrapped at commas/spaces.
        return out.flatMap { chunk -> [String] in
            guard chunk.count > limit else { return [chunk] }
            var pieces: [String] = []
            var piece = ""
            for part in chunk.split(separator: ",", omittingEmptySubsequences: false) {
                let candidate = piece.isEmpty ? String(part) : piece + "," + String(part)
                if candidate.count > limit, !piece.isEmpty {
                    pieces.append(piece.trimmingCharacters(in: .whitespaces))
                    piece = String(part)
                } else {
                    piece = candidate
                }
            }
            if !piece.isEmpty { pieces.append(piece.trimmingCharacters(in: .whitespaces)) }
            return pieces
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
