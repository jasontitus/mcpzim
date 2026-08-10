// SPDX-License-Identifier: MIT
//
// Repeatable macOS cold-process benchmark for the same TTS implementation
// shipped in the app. Run once without --prewarm and once with it; keeping
// those in separate processes prevents MLX's global caches from contaminating
// the comparison.

import Darwin
import Foundation

private actor PeakMemoryProbe {
    private var peakMB: Double

    init(startMB: Double) { peakMB = startMB }
    func sample(_ value: Double) { peakMB = max(peakMB, value) }
    func peak() -> Double { peakMB }
}

private struct RenderMeasurement {
    let samples: [Float]
    let sampleRate: Int
    let seconds: Double
    let startMemoryMB: Double
    let endMemoryMB: Double
    let peakMemoryMB: Double
}

@main
struct TTSBenchMain {
    private static let defaultText =
        "Dr. Vladimir V. Putin attended Leningrad State University; later, he joined the KGB in 1975."

    @MainActor
    static func main() async {
        exit(await run(args: Array(CommandLine.arguments.dropFirst())))
    }

    @MainActor
    private static func run(args: [String]) async -> Int32 {
        var prewarm = false
        var voice = "af_heart"
        var text = defaultText
        var outputURL: URL?
        var remaining = ArraySlice(args)
        while let argument = remaining.first {
            remaining = remaining.dropFirst()
            switch argument {
            case "--prewarm":
                prewarm = true
            case "--voice":
                guard let value = remaining.first else { return usage("--voice needs a value") }
                voice = value
                remaining = remaining.dropFirst()
            case "--text":
                guard let value = remaining.first else { return usage("--text needs a value") }
                text = value
                remaining = remaining.dropFirst()
            case "--output":
                guard let value = remaining.first else { return usage("--output needs a value") }
                outputURL = URL(fileURLWithPath: value)
                remaining = remaining.dropFirst()
            case "-h", "--help":
                return usage(nil)
            default:
                return usage("unknown TTS option: \(argument)")
            }
        }

        guard KokoroAssets.isDownloaded else {
            error("[TTSBench] Kokoro assets missing from \(KokoroAssets.modelDirectory.path)")
            return 2
        }

        let processStarted = ProcessInfo.processInfo.systemUptime
        let memoryAtStart = physFootprintMB()
        let service: TTSService
        do {
            service = try KokoroTTSService(voice: voice)
        } catch {
            Self.error("[TTSBench] initialization failed: \(error.localizedDescription)")
            return 3
        }
        let initializedAt = ProcessInfo.processInfo.systemUptime
        let memoryAfterInit = physFootprintMB()
        log(String(format:
            "[TTSBench] backend=%@ voice=%@ mode=%@ init=%.3fs mem=%.1fMB delta=%+.1fMB",
            service.displayName, voice, prewarm ? "prewarm" : "cold",
            initializedAt - processStarted, memoryAfterInit,
            memoryAfterInit - memoryAtStart))

        // Render-only mode produces a directly comparable WAV for listening
        // and measures both the first and immediately repeated synthesis in
        // the same process. It deliberately skips AVAudioPlayerNode so audio
        // hardware scheduling does not contaminate model latency or memory.
        if let outputURL {
            guard let kokoro = service as? KokoroTTSService else {
                Self.error("[TTSBench] render mode requires KokoroTTSService")
                return 4
            }
            let renderText = text
            do {
                let first = try await measureRender {
                    try kokoro.renderForBenchmark(renderText)
                }
                reportRender("first-synthesis", first, outputURL: nil)
                let warm = try await measureRender {
                    try kokoro.renderForBenchmark(renderText)
                }
                reportRender("warm-synthesis", warm, outputURL: outputURL)
            } catch {
                Self.error("[TTSBench] render failed: \(error.localizedDescription)")
                return 5
            }
            try? await Task.sleep(for: .seconds(1))
            log(String(format: "[TTSBench] render-settled-after-1s mem=%.1fMB", physFootprintMB()))
            return 0
        }

        if prewarm {
            let preparationStarted = ProcessInfo.processInfo.systemUptime
            do {
                try await Task.detached(priority: .utility) {
                    try await service.prepareForConversation()
                }.value
            } catch {
                Self.error("[TTSBench] preparation failed: \(error.localizedDescription)")
                return 4
            }
            let preparedAt = ProcessInfo.processInfo.systemUptime
            let memoryAfterPreparation = physFootprintMB()
            log(String(format:
                "[TTSBench] prepared=%.3fs mem=%.1fMB delta-from-init=%+.1fMB",
                preparedAt - preparationStarted, memoryAfterPreparation,
                memoryAfterPreparation - memoryAfterInit))
            // A real conversation normally has several seconds of recognition
            // and LLM work between preparation and first speech. Sample one
            // second later so we can distinguish retained state from deferred
            // Metal/task cleanup.
            try? await Task.sleep(for: .seconds(1))
            log(String(format: "[TTSBench] settled-after-1s mem=%.1fMB",
                       physFootprintMB()))
        }

        let synthesisStarted = ProcessInfo.processInfo.systemUptime
        do {
            try await service.speakChunk(text)
        } catch {
            Self.error("[TTSBench] synthesis failed: \(error.localizedDescription)")
            return 5
        }
        let audioReadyAt = ProcessInfo.processInfo.systemUptime
        let memoryAtReady = physFootprintMB()
        log(String(format:
            "[TTSBench] audio-ready=%.3fs chars=%d mem=%.1fMB delta-from-init=%+.1fMB",
            audioReadyAt - synthesisStarted, text.count, memoryAtReady,
            memoryAtReady - memoryAfterInit))

        await service.awaitPlayback()
        let playbackDoneAt = ProcessInfo.processInfo.systemUptime
        log(String(format:
            "[TTSBench] playback-drained=%.3fs total=%.3fs final-mem=%.1fMB text=%@",
            playbackDoneAt - audioReadyAt, playbackDoneAt - processStarted,
            physFootprintMB(), text))
        return 0
    }

    private static func usage(_ message: String?) -> Int32 {
        if let message { error("[TTSBench] \(message)") }
        log("Usage: MCPZimTTSBenchCLI [--prewarm] [--voice NAME] [--text TEXT] [--output WAV]")
        return message == nil ? 0 : 2
    }

    private static func measureRender(
        _ operation: @escaping @Sendable () throws -> (samples: [Float], sampleRate: Int)
    ) async throws -> RenderMeasurement {
        let started = ProcessInfo.processInfo.systemUptime
        let startMemory = physFootprintMB()
        let probe = PeakMemoryProbe(startMB: startMemory)
        let sampler = Task.detached {
            while !Task.isCancelled {
                await probe.sample(physFootprintMB())
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
        }
        do {
            let rendered = try await Task.detached(priority: .userInitiated) {
                try operation()
            }.value
            sampler.cancel()
            _ = await sampler.result
            let endMemory = physFootprintMB()
            await probe.sample(endMemory)
            return RenderMeasurement(
                samples: rendered.samples,
                sampleRate: rendered.sampleRate,
                seconds: ProcessInfo.processInfo.systemUptime - started,
                startMemoryMB: startMemory,
                endMemoryMB: endMemory,
                peakMemoryMB: await probe.peak())
        } catch {
            sampler.cancel()
            _ = await sampler.result
            throw error
        }
    }

    private static func reportRender(
        _ phase: String, _ measurement: RenderMeasurement, outputURL: URL?
    ) {
        let duration = Double(measurement.samples.count) / Double(measurement.sampleRate)
        if let outputURL {
            do {
                try FileManager.default.createDirectory(
                    at: outputURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try wavData(
                    samples: measurement.samples,
                    sampleRate: measurement.sampleRate).write(to: outputURL, options: .atomic)
            } catch {
                Self.error("[TTSBench] could not write \(outputURL.path): \(error)")
            }
        }
        log(String(format:
            "[TTSBench] phase=%@ elapsed=%.3fs audio=%.3fs rtfx=%.2fx start-mem=%.1fMB end-mem=%.1fMB peak-mem=%.1fMB peak-delta=%+.1fMB output=%@",
            phase, measurement.seconds, duration,
            duration / max(measurement.seconds, 0.001),
            measurement.startMemoryMB, measurement.endMemoryMB,
            measurement.peakMemoryMB,
            measurement.peakMemoryMB - measurement.startMemoryMB,
            outputURL?.path ?? "(not-written)"))
    }

    private static func wavData(samples: [Float], sampleRate: Int) -> Data {
        var pcm = Data(capacity: samples.count * 2)
        for sample in samples {
            var value = Int16(max(-1, min(1, sample)) * 32_767).littleEndian
            withUnsafeBytes(of: &value) { pcm.append(contentsOf: $0) }
        }
        var wav = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { wav.append(contentsOf: $0) }
        }
        wav.append(Data("RIFF".utf8))
        append(UInt32(36 + pcm.count))
        wav.append(Data("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))
        append(UInt16(2))
        append(UInt16(16))
        wav.append(Data("data".utf8))
        append(UInt32(pcm.count))
        wav.append(pcm)
        return wav
    }

    private static func log(_ message: String) {
        FileHandle.standardOutput.write(Data((message + "\n").utf8))
    }

    private static func error(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    private static func physFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}
