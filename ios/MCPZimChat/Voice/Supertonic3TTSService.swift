// SPDX-License-Identifier: MIT

#if canImport(FluidAudio)

import AVFoundation
import FluidAudio
import Foundation

/// Supertonic 3 playback adapter for conversational voice chat.
///
/// The VectorEstimator variant is intentionally explicit: the legacy dynamic
/// graph cannot compile for ANE and falls back to CPU/GPU. Fixed L128/L256/L512
/// INT8 buckets keep the large repeated stage approximately 94% ANE-resident.
public final class Supertonic3TTSService: NSObject, TTSService, @unchecked Sendable {
    public let displayName = "Supertonic 3 (ANE-bucketed INT8)"
    public let approximateMemoryMB = 32
    public let peakSynthesisMemoryMB = 96
    /// MCPZim's validated FluidAudio variant caps Latin input at 96 characters
    /// and otherwise synthesizes multiple independent utterances internally.
    /// Stay just below that cap so MCPZim controls continuation seams.
    public let preferredStreamingChunkCharacters: Int? = 94

    public var isSpeaking: Bool {
        stateLock.withLock { speaking }
    }

    public let voiceName: String

    private let runtime: Supertonic3Runtime
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 44_100,
        channels: 1,
        interleaved: false)!
    private let stateLock = NSLock()
    private var speaking = false
    private var stopRequested = false
    private var hasQueuedAudio = false

    public init(voice: String = "F1") {
        let selectedVoice = Supertonic3Voice(name: voice) ?? .f1
        voiceName = selectedVoice.rawValue
        runtime = Supertonic3Runtime(
            directory: Supertonic3Assets.modelDirectory,
            voice: selectedVoice)
        super.init()
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        player.volume = 1.0
        engine.mainMixerNode.outputVolume = 1.0
    }

    public func prepareForConversation() async throws {
        try await runtime.prepareAndWarmInferenceBuckets()
    }

    public func speak(_ text: String) async throws {
        try await speakChunk(text, boundary: .final)
        await awaitPlayback()
    }

    public func speakChunk(_ text: String) async throws {
        try await speakChunk(text, boundary: .final)
    }

    public func speakChunk(_ text: String, boundary: TTSChunkBoundary) async throws {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }

        stateLock.withLock { stopRequested = false }
        let result = try await runtime.synthesize(text: raw)
        let joiningQueuedAudio = stateLock.withLock { hasQueuedAudio }
        let trimmedSamples = Self.trimGeneratedSilence(
            result.samples,
            boundary: boundary,
            joiningQueuedAudio: joiningQueuedAudio)
        let samples = trimmedSamples
        guard !samples.isEmpty else {
            throw TTSError.synthesisFailed("Supertonic produced no audio.")
        }
        // Allocation-free level API (DS4 pass): fold the gain into the
        // single PCM-buffer copy below instead of a normalized copy here.
        let playbackGain = TTSPlaybackLevel.gain(for: samples)
        guard !stateLock.withLock({ stopRequested }) else { return }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count))
        else {
            throw TTSError.synthesisFailed("Could not allocate Supertonic PCM buffer.")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let destination = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { source in
                guard let baseAddress = source.baseAddress else { return }
                if playbackGain == 1.0 {
                    destination.update(from: baseAddress, count: samples.count)
                } else {
                    for i in 0..<samples.count {
                        destination[i] = baseAddress[i] * playbackGain
                    }
                }
            }
        }

        if !engine.isRunning { try engine.start() }
        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
        stateLock.withLock {
            speaking = true
            hasQueuedAudio = true
        }
        if !player.isPlaying { player.play() }
    }

    /// Supertonic predicts each short encoder window as a complete utterance,
    /// including leading/trailing quiet. When multiple windows are queued,
    /// that quiet becomes a conspicuous pause at arbitrary word wraps. Trim
    /// only the outer low-energy region and retain boundary-specific padding.
    private static func trimGeneratedSilence(
        _ samples: [Float],
        boundary: TTSChunkBoundary,
        joiningQueuedAudio: Bool
    ) -> [Float] {
        let sampleRate = 44_100
        let window = sampleRate / 100 // 10 ms
        guard samples.count > window * 4 else { return samples }

        // Roughly -54 dBFS. Windowed RMS avoids clipping quiet consonant tails
        // while still recognizing the model's near-zero padding.
        let activeRMS: Float = 0.002
        func isActive(_ lower: Int, _ upper: Int) -> Bool {
            var sum: Float = 0
            for index in lower..<upper {
                let value = samples[index]
                sum += value * value
            }
            return (sum / Float(max(1, upper - lower))).squareRoot() >= activeRMS
        }

        var firstActive: Int?
        var cursor = 0
        while cursor + window <= samples.count {
            if isActive(cursor, cursor + window) {
                firstActive = cursor
                break
            }
            cursor += window
        }

        var lastActiveEnd: Int?
        cursor = samples.count
        while cursor - window >= 0 {
            if isActive(cursor - window, cursor) {
                lastActiveEnd = cursor
                break
            }
            cursor -= window
        }
        guard let firstActive, let lastActiveEnd, firstActive < lastActiveEnd else {
            return samples
        }

        let leadingSeconds: Double = joiningQueuedAudio ? 0.005 : 0.015
        let trailingSeconds: Double
        switch boundary {
        case .softWrap: trailingSeconds = 0.025
        case .clause: trailingSeconds = 0.12
        case .sentence: trailingSeconds = 0.26
        case .final: trailingSeconds = 0.18
        }

        let start = max(0, firstActive - Int(Double(sampleRate) * leadingSeconds))
        let end = min(
            samples.count,
            lastActiveEnd + Int(Double(sampleRate) * trailingSeconds))
        guard start < end else { return samples }
        return Array(samples[start..<end])
    }

    public func awaitPlayback() async {
        let shouldWait = stateLock.withLock { hasQueuedAudio && !stopRequested }
        guard shouldWait else {
            stateLock.withLock { speaking = false }
            return
        }

        guard let marker = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1) else { return }
        marker.frameLength = 1
        marker.floatChannelData?[0][0] = 0
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(marker, at: nil, options: []) {
                continuation.resume()
            }
            if !player.isPlaying { player.play() }
        }
        stateLock.withLock {
            speaking = false
            hasQueuedAudio = false
        }
    }

    public func stop() {
        stateLock.withLock {
            stopRequested = true
            speaking = false
            hasQueuedAudio = false
        }
        player.stop()
    }
}

private actor Supertonic3Runtime {
    private let directory: URL
    private let voice: Supertonic3Voice
    private let manager: Supertonic3Manager
    private var style: Supertonic3VoiceStyle?
    private var didWarmInferenceBuckets = false

    init(directory: URL, voice: Supertonic3Voice) {
        self.directory = directory
        self.voice = voice
        manager = Supertonic3Manager(
            directory: directory,
            vectorEstimator: .aneBucketed(.int8))
    }

    func prepare() async throws {
        try await manager.initialize()
        if style == nil {
            style = try await Supertonic3ResourceDownloader.loadVoiceStyle(
                voice,
                directory: directory)
        }
    }

    /// Core ML lazily compiles each fixed VectorEstimator length on its first
    /// prediction. Exercise representative short, medium, and long utterances
    /// with one denoising step while the user is speaking. The real answer
    /// still uses FluidAudio's full eight-step quality setting.
    func prepareAndWarmInferenceBuckets() async throws {
        try await prepare()
        guard !didWarmInferenceBuckets, let style else { return }

        let probes = [
            "Ready to speak.",
            "Preparing a natural voice for a clear conversational answer.",
            "Preparing a natural voice for longer conversational answers about history and notable people.",
        ]
        for probe in probes {
            let result = try await manager.synthesize(
                text: probe,
                language: "en",
                style: style,
                totalSteps: 1,
                silenceDuration: 0)
            guard !result.samples.isEmpty else {
                throw TTSError.synthesisFailed(
                    "Supertonic inference preparation produced no audio.")
            }
        }
        didWarmInferenceBuckets = true
    }

    func synthesize(text: String) async throws -> (samples: [Float], duration: Float) {
        try await prepare()
        guard let style else {
            throw TTSError.synthesisFailed("Supertonic voice style was not loaded.")
        }
        return try await manager.synthesize(
            text: text,
            language: "en",
            style: style,
            silenceDuration: 0)
    }
}

public enum SupertonicVoicePreference {
    private static let key = "supertonic.voice"
    public static let available = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]

    public static var current: String {
        get { UserDefaults.standard.string(forKey: key) ?? "F1" }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
}

public enum Supertonic3Assets {
    /// Root passed to FluidAudio; it creates `supertonic-3-coreml/` below it.
    public static var modelDirectory: URL {
        let fileManager = FileManager.default
        let base = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Library/Application Support")
        let directory = base
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("supertonic_3", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    public static var currentBytesOnDisk: Int64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles])
        else { return 0 }

        var total: Int64 = 0
        for case let url as URL in enumerator {
            guard let values = try? url.resourceValues(forKeys: keys),
                  values.isRegularFile == true,
                  let size = values.fileSize
            else { continue }
            total += Int64(size)
        }
        return total
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

#endif
