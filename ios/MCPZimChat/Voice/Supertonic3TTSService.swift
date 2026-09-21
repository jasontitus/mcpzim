// SPDX-License-Identifier: MIT

#if canImport(FluidAudio)

import AVFoundation
import FluidAudio
import Foundation
import MCPZimKit

/// Supertonic 3 playback adapter for conversational voice chat.
///
/// The VectorEstimator variant is intentionally explicit: the legacy dynamic
/// graph cannot compile for ANE and falls back to CPU/GPU. Fixed L128/L256/L512
/// INT8 buckets keep the large repeated stage approximately 94% ANE-resident.
public final class Supertonic3TTSService: NSObject, TTSService, @unchecked Sendable {
    public let displayName = "Supertonic 3 (ANE-bucketed INT8)"
    public let approximateMemoryMB = 48
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
    private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 1, interleaved: false)!
    private let stateLock = NSLock()
    private var configurationObserver: NSObjectProtocol?
    private var playbackGate: PlaybackCompletionGate?
    private var playbackError: String?
    public var playbackFailure: String? { stateLock.withLock { playbackError } }
    private var speaking = false
    private var stopRequested = false
    private var hasQueuedAudio = false
    /// Wall-clock estimate of when the player's queued audio drains. Buffers
    /// play back-to-back in AVAudioPlayerNode's FIFO, so scheduling a chunk
    /// after this instant means the listener heard silence in between —
    /// that difference is the per-chunk dead-air (`gapSeconds`) metric.
    private var queueDrainsAt: Date?
    private var lastChunkMetrics: TTSChunkPlaybackMetrics?

    public init(voice: String = "F1") {
        let selectedVoice = Supertonic3Voice(name: voice) ?? .f1
        voiceName = selectedVoice.rawValue
        runtime = .shared
        super.init()
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        player.volume = 1.0
        engine.mainMixerNode.outputVolume = 1.0
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            // Engine operations can post notifications synchronously. Defer
            // handling so their callback cannot re-enter the state lock.
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                let gate = self.stateLock.withLock {
                    guard self.hasQueuedAudio else { return self.playbackGate }
                    self.playbackError = "The audio output changed during playback. Check the Bluetooth output, then restart voice mode."
                    self.stopRequested = true
                    return self.playbackGate
                }
                gate?.finish(.interrupted)
            }
        }
    }

    deinit {
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
    }

    public func prepareForConversation() async throws {
        // Reuse the model-only runtime warmed after setup. If that warm-up
        // was skipped, load assets here without adding multiple probe chunks.
        stateLock.withLock { if !hasQueuedAudio { playbackError = nil } }
        try await runtime.prepare(voice: voiceName)
    }

    #if DEBUG
    /// Exercise the production synthesizer without starting an audio engine
    /// or scheduling any buffer. Used by the opt-in silent device probe.
    func synthesizeWithoutPlayback(_ text: String) async throws -> (sampleCount: Int, audioSeconds: Double) {
        let result = try await runtime.synthesize(text: text, voice: voiceName)
        return (result.samples.count, Double(result.samples.count) / format.sampleRate)
    }
    #endif

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

        if let failure = playbackFailure { throw TTSError.synthesisFailed(failure) }
        stateLock.withLock { stopRequested = false }
        try Task.checkCancellation()
        let result = try await runtime.synthesize(text: raw, voice: voiceName)
        try Task.checkCancellation()
        let joiningQueuedAudio = stateLock.withLock { hasQueuedAudio }
        let trimmedSamples = Self.trimGeneratedSilence(
            result.samples,
            boundary: boundary,
            joiningQueuedAudio: joiningQueuedAudio,
            sampleRate: Int(format.sampleRate))
        let samples = trimmedSamples
        guard !samples.isEmpty else {
            throw TTSError.synthesisFailed("\(displayName) produced no audio.")
        }
        // Allocation-free level API (DS4 pass): fold the gain into the
        // single PCM-buffer copy below instead of a normalized copy here.
        let playbackGain = TTSPlaybackLevel.gain(for: samples)
        if let failure = playbackFailure { throw TTSError.synthesisFailed(failure) }
        guard !stateLock.withLock({ stopRequested }) else { return }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count))
        else {
            throw TTSError.synthesisFailed("Could not allocate speech PCM buffer.")
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
            // Silence trimming may cut between zero crossings. Taper only
            // the copied buffer edges, with no extra PCM allocation or gap.
            SpeechPCMEdges.taper(UnsafeMutableBufferPointer(start: destination, count: samples.count),
                                 sampleRate: Int(format.sampleRate))
        }

        if !engine.isRunning { try engine.start() }
        let audioSeconds = Double(samples.count) / format.sampleRate
        let trimmedSeconds =
            Double(result.samples.count - samples.count) / format.sampleRate
        let now = Date()
        stateLock.withLock {
            // Gap/queue-ahead relative to the previous chunks of this stream.
            // A drained (or brand-new) queue starts playing immediately.
            let drainAt = hasQueuedAudio ? queueDrainsAt : nil
            let gap = drainAt.map { max(0, now.timeIntervalSince($0)) } ?? 0
            let ahead = drainAt.map { max(0, $0.timeIntervalSince(now)) } ?? 0
            let startsAt = drainAt.map { max($0, now) } ?? now
            queueDrainsAt = startsAt.addingTimeInterval(audioSeconds)
            lastChunkMetrics = TTSChunkPlaybackMetrics(
                audioSeconds: audioSeconds,
                gapSeconds: gap,
                queueAheadSeconds: ahead,
                trimmedSeconds: trimmedSeconds)
            speaking = true
            hasQueuedAudio = true
        }
        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
        if !player.isPlaying { player.play() }
    }

    public func takeStreamingChunkMetrics() -> TTSChunkPlaybackMetrics? {
        stateLock.withLock {
            let metrics = lastChunkMetrics
            lastChunkMetrics = nil
            return metrics
        }
    }

    /// Supertonic predicts each short encoder window as a complete utterance,
    /// including leading/trailing quiet. When multiple windows are queued,
    /// that quiet becomes a conspicuous pause at arbitrary word wraps. Trim
    /// only the outer low-energy region and retain boundary-specific padding.
    private static func trimGeneratedSilence(
        _ samples: [Float],
        boundary: TTSChunkBoundary,
        joiningQueuedAudio: Bool,
        sampleRate: Int
    ) -> [Float] {
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
        let gate = PlaybackCompletionGate()
        let timeout: TimeInterval? = stateLock.withLock {
            guard hasQueuedAudio, !stopRequested else { return nil }
            playbackGate = gate
            // Queue duration is an estimate, not evidence that sound played.
            // Allow Bluetooth output latency after the expected final sample.
            return max(0, queueDrainsAt?.timeIntervalSinceNow ?? 0) + 8
        }
        guard let timeout else { return }
        guard let marker = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1) else {
            gate.finish(.timedOut)
            _ = await gate.wait(timeout: 0)
            finishPlayback(gate: gate, outcome: .timedOut)
            return
        }
        marker.frameLength = 1
        marker.floatChannelData?[0][0] = 0
        stateLock.withLock {
            guard !stopRequested, playbackGate === gate else {
                gate.finish(.cancelled)
                return
            }
            player.scheduleBuffer(marker, at: nil, options: [],
                                  completionCallbackType: .dataPlayedBack) { _ in
                gate.finish(.played)
            }
            if !player.isPlaying { player.play() }
        }
        let outcome = await gate.wait(timeout: timeout)
        finishPlayback(gate: gate, outcome: outcome)
    }

    private func finishPlayback(gate: PlaybackCompletionGate, outcome: PlaybackCompletionGate.Outcome) {
        stateLock.withLock {
            // A cancelled older turn cannot tear down a newer turn's graph.
            guard playbackGate === gate else { return }
            playbackGate = nil
            speaking = false
            hasQueuedAudio = false
            queueDrainsAt = nil
            if outcome == .timedOut {
                playbackError = "Audio playback did not finish. Check the Bluetooth output, then restart voice mode."
            }
            player.stop()
            engine.stop()
        }
    }

    public func stop() {
        let gate = stateLock.withLock {
            stopRequested = true
            speaking = false
            hasQueuedAudio = false
            queueDrainsAt = nil
            lastChunkMetrics = nil
            let gate = playbackGate
            playbackGate = nil
            player.stop()
            engine.stop()
            return gate
        }
        gate?.finish(.cancelled)
    }

}

private actor Supertonic3Runtime {
    static let shared = Supertonic3Runtime()
    private let directory = Supertonic3Assets.modelDirectory
    private let manager = Supertonic3Manager(
        directory: Supertonic3Assets.modelDirectory, vectorEstimator: .aneBucketed(.int8))
    private var styles: [String: Supertonic3VoiceStyle] = [:]
    private var tail: Task<Void, Never>?
    private var warmup: Task<Void, Error>?
    private var warmed = false

    func prepare(voice: String) async throws { _ = try await enqueue(text: nil, voice: voice) }
    func synthesize(text: String, voice: String) async throws -> (samples: [Float], duration: Float) {
        try await enqueue(text: text, voice: voice)
    }

    func warm(voice: String) async throws {
        if warmed { return }
        if let warmup { return try await warmup.value }
        let task = Task {
            _ = try await self.enqueue(
                text: "Welcome. You can explore nearby places, read an article, or ask a follow-up question.",
                voice: voice)
        }
        warmup = task
        do { try await task.value; warmed = true; warmup = nil }
        catch { warmup = nil; throw error }
    }

    // Actors are reentrant across await: explicitly serialize preparation,
    // warm-up and real synthesis so shared Core ML state cannot overlap.
    private func enqueue(text: String?, voice: String) async throws -> (samples: [Float], duration: Float) {
        let previous = tail
        let task = Task {
            await previous?.value
            try Task.checkCancellation()
            return try await self.perform(text: text, voice: voice)
        }
        tail = Task { _ = try? await task.value }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: { task.cancel() }
    }

    private func perform(text: String?, voice: String) async throws -> (samples: [Float], duration: Float) {
        try await manager.initialize()
        if styles[voice] == nil {
            styles[voice] = try await Supertonic3ResourceDownloader.loadVoiceStyle(
                Supertonic3Voice(name: voice) ?? .f1, directory: directory)
        }
        try Task.checkCancellation()
        guard let text else { return ([], 0) }
        guard let style = styles[voice] else { throw TTSError.synthesisFailed("Supertonic voice style was not loaded.") }
        return try await manager.synthesize(text: text, language: "en", style: style, silenceDuration: 0)
    }
}

extension Supertonic3TTSService {
    /// Model-only warm-up: no playback graph, microphone or audio session.
    static func prewarmRuntime() async throws {
        try await Supertonic3Runtime.shared.warm(voice: SupertonicVoicePreference.current)
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
    /// Root passed to FluidAudio; its downloader creates `supertonic-3/` below it.
    public static var modelDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["MCPZIM_SUPERTONIC_MODEL_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
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

    static var readyForSilentWarmup: Bool {
        let root = modelDirectory.appendingPathComponent(Repo.supertonic3.folderName)
        let voice = Supertonic3Voice(name: SupertonicVoicePreference.current) ?? .f1
        return ModelNames.Supertonic3.requiredFiles(veVariant: "ane-int8")
            .union([voice.fileName]).allSatisfy {
                FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path)
            }
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
