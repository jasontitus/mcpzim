// SPDX-License-Identifier: MIT
//
// Isolated macOS benchmark for FluidAudio's Core ML TTS engines. Run each
// backend in a fresh process so Core ML model residency and caches do not
// leak across comparisons.

import Darwin
import FluidAudio
import Foundation

private actor PeakMemoryProbe {
    private var peakMB: Double

    init(startMB: Double) {
        peakMB = startMB
    }

    func sample(_ value: Double) {
        peakMB = max(peakMB, value)
    }

    func peak() -> Double { peakMB }
}

private struct Measurement<Value> {
    let value: Value
    let seconds: Double
    let startMemoryMB: Double
    let endMemoryMB: Double
    let peakMemoryMB: Double
}

@main
struct CoreMLTTSBenchMain {
    private enum Backend: String {
        case kokoroANE = "kokoro-ane"
        case supertonic3INT8 = "supertonic3-int8"
        case supertonic3INT4 = "supertonic3-int4"
    }

    private struct Options {
        var backend: Backend?
        var cacheDirectory = URL(
            fileURLWithPath: "/tmp/mcpzim-fluidaudio", isDirectory: true)
        var outputDirectory = URL(
            fileURLWithPath: "/tmp/mcpzim-tts-output", isDirectory: true)
        var text = defaultText
        var prepareAssets = false
    }

    private static let defaultText =
        "Dr. Vladimir V. Putin attended Leningrad State University; later, he joined the KGB in 1975."

    static func main() async {
        exit(await run(arguments: Array(CommandLine.arguments.dropFirst())))
    }

    private static func run(arguments: [String]) async -> Int32 {
        let options: Options
        do {
            options = try parse(arguments)
        } catch {
            return usage(error.localizedDescription)
        }
        guard let backend = options.backend else {
            return usage("--backend is required")
        }

        do {
            try FileManager.default.createDirectory(
                at: options.cacheDirectory, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: options.outputDirectory, withIntermediateDirectories: true)

            log(String(format:
                "[CoreMLTTSBench] backend=%@ mode=%@ process-start-mem=%.1fMB cache=%@",
                backend.rawValue, options.prepareAssets ? "prepare-assets" : "benchmark",
                physicalFootprintMB(), options.cacheDirectory.path))

            switch backend {
            case .kokoroANE:
                try await runKokoroANE(options)
            case .supertonic3INT8:
                try await runSupertonic3(options, quantization: .int8)
            case .supertonic3INT4:
                try await runSupertonic3(options, quantization: .int4)
            }
            return 0
        } catch let caught {
            error("[CoreMLTTSBench] failed: \(caught)")
            return 3
        }
    }

    private static func runKokoroANE(_ options: Options) async throws {
        let manager = KokoroAneManager(
            variant: .english,
            defaultVoice: "af_heart",
            directory: options.cacheDirectory)

        let initialization = try await measure {
            try await manager.initialize(preloadVoices: ["af_heart"])
        }
        report("initialize", initialization)
        guard !options.prepareAssets else {
            await manager.cleanup()
            log("[CoreMLTTSBench] assets-ready backend=kokoro-ane")
            return
        }

        let first = try await measure {
            try await manager.synthesizeDetailed(
                text: options.text, voice: "af_heart", speed: 1.0)
        }
        reportSynthesis(
            "first-synthesis", measurement: first,
            sampleCount: first.value.samples.count,
            sampleRate: first.value.sampleRate,
            outputURL: options.outputDirectory.appendingPathComponent("kokoro-ane-first.wav"),
            normalize: false)

        let warm = try await measure {
            try await manager.synthesizeDetailed(
                text: options.text, voice: "af_heart", speed: 1.0)
        }
        reportSynthesis(
            "warm-synthesis", measurement: warm,
            sampleCount: warm.value.samples.count,
            sampleRate: warm.value.sampleRate,
            outputURL: options.outputDirectory.appendingPathComponent("kokoro-ane-warm.wav"),
            normalize: false)

        try? await Task.sleep(nanoseconds: 1_000_000_000)
        log(String(format: "[CoreMLTTSBench] resident-settled-after-1s mem=%.1fMB",
                   physicalFootprintMB()))
        await manager.cleanup()
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        log(String(format: "[CoreMLTTSBench] cleanup-settled-mem=%.1fMB", physicalFootprintMB()))
    }

    private static func runSupertonic3(
        _ options: Options, quantization: Supertonic3Quantization
    ) async throws {
        let backendName = "supertonic3-\(quantization.rawValue)"
        let manager = Supertonic3Manager(
            directory: options.cacheDirectory,
            vectorEstimator: .aneBucketed(quantization))

        let initialization = try await measure {
            try await manager.initialize()
        }
        report("initialize", initialization)

        let styleMeasurement = try await measure {
            try await Supertonic3ResourceDownloader.loadVoiceStyle(
                .f1, directory: options.cacheDirectory)
        }
        report("voice-style", styleMeasurement)
        guard !options.prepareAssets else {
            await manager.cleanup()
            log("[CoreMLTTSBench] assets-ready backend=\(backendName)")
            return
        }

        let first = try await measure {
            try await manager.synthesize(
                text: options.text, language: "en", style: styleMeasurement.value)
        }
        reportSynthesis(
            "first-synthesis", measurement: first,
            sampleCount: first.value.samples.count,
            sampleRate: 44_100,
            outputURL: options.outputDirectory.appendingPathComponent("\(backendName)-first.wav"),
            normalize: true)

        let warm = try await measure {
            try await manager.synthesize(
                text: options.text, language: "en", style: styleMeasurement.value)
        }
        reportSynthesis(
            "warm-synthesis", measurement: warm,
            sampleCount: warm.value.samples.count,
            sampleRate: 44_100,
            outputURL: options.outputDirectory.appendingPathComponent("\(backendName)-warm.wav"),
            normalize: true)

        try? await Task.sleep(nanoseconds: 1_000_000_000)
        log(String(format: "[CoreMLTTSBench] resident-settled-after-1s mem=%.1fMB",
                   physicalFootprintMB()))
        await manager.cleanup()
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        log(String(format: "[CoreMLTTSBench] cleanup-settled-mem=%.1fMB", physicalFootprintMB()))
    }

    private static func measure<Value>(
        _ operation: () async throws -> Value
    ) async rethrows -> Measurement<Value> {
        let started = ProcessInfo.processInfo.systemUptime
        let startMemory = physicalFootprintMB()
        let probe = PeakMemoryProbe(startMB: startMemory)
        let sampler = Task {
            while !Task.isCancelled {
                await probe.sample(physicalFootprintMB())
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
        }

        do {
            let value = try await operation()
            sampler.cancel()
            _ = await sampler.result
            let endMemory = physicalFootprintMB()
            await probe.sample(endMemory)
            return Measurement(
                value: value,
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

    private static func report<Value>(
        _ phase: String, _ measurement: Measurement<Value>
    ) {
        log(String(format:
            "[CoreMLTTSBench] phase=%@ elapsed=%.3fs start-mem=%.1fMB end-mem=%.1fMB peak-mem=%.1fMB peak-delta=%+.1fMB",
            phase, measurement.seconds, measurement.startMemoryMB,
            measurement.endMemoryMB, measurement.peakMemoryMB,
            measurement.peakMemoryMB - measurement.startMemoryMB))
    }

    private static func reportSynthesis(
        _ phase: String,
        measurement: Measurement<KokoroAneSynthesisResult>,
        sampleCount: Int,
        sampleRate: Int,
        outputURL: URL,
        normalize: Bool
    ) {
        writeAndReportSynthesis(
            phase, measurement: measurement,
            samples: measurement.value.samples,
            sampleCount: sampleCount, sampleRate: sampleRate,
            outputURL: outputURL, normalize: normalize)
    }

    private static func reportSynthesis(
        _ phase: String,
        measurement: Measurement<(samples: [Float], duration: Float)>,
        sampleCount: Int,
        sampleRate: Int,
        outputURL: URL,
        normalize: Bool
    ) {
        writeAndReportSynthesis(
            phase, measurement: measurement,
            samples: measurement.value.samples,
            sampleCount: sampleCount, sampleRate: sampleRate,
            outputURL: outputURL, normalize: normalize)
    }

    private static func writeAndReportSynthesis<Value>(
        _ phase: String,
        measurement: Measurement<Value>,
        samples: [Float],
        sampleCount: Int,
        sampleRate: Int,
        outputURL: URL,
        normalize: Bool
    ) {
        let duration = Double(sampleCount) / Double(sampleRate)
        let realTimeFactor = duration / max(measurement.seconds, 0.001)
        do {
            let wav = try AudioWAV.data(
                from: samples, sampleRate: Double(sampleRate), normalize: normalize)
            try wav.write(to: outputURL, options: .atomic)
        } catch {
            Self.error("[CoreMLTTSBench] could not write \(outputURL.path): \(error)")
        }
        log(String(format:
            "[CoreMLTTSBench] phase=%@ elapsed=%.3fs audio=%.3fs rtfx=%.2fx start-mem=%.1fMB end-mem=%.1fMB peak-mem=%.1fMB peak-delta=%+.1fMB output=%@",
            phase, measurement.seconds, duration, realTimeFactor,
            measurement.startMemoryMB, measurement.endMemoryMB,
            measurement.peakMemoryMB,
            measurement.peakMemoryMB - measurement.startMemoryMB,
            outputURL.path))
    }

    private static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var remaining = ArraySlice(arguments)
        while let argument = remaining.first {
            remaining = remaining.dropFirst()
            switch argument {
            case "--backend":
                guard let raw = remaining.first, let backend = Backend(rawValue: raw) else {
                    throw ArgumentError("--backend must be kokoro-ane, supertonic3-int8, or supertonic3-int4")
                }
                options.backend = backend
                remaining = remaining.dropFirst()
            case "--cache-dir":
                guard let value = remaining.first else {
                    throw ArgumentError("--cache-dir needs a value")
                }
                options.cacheDirectory = URL(fileURLWithPath: value, isDirectory: true)
                remaining = remaining.dropFirst()
            case "--output-dir":
                guard let value = remaining.first else {
                    throw ArgumentError("--output-dir needs a value")
                }
                options.outputDirectory = URL(fileURLWithPath: value, isDirectory: true)
                remaining = remaining.dropFirst()
            case "--text":
                guard let value = remaining.first else {
                    throw ArgumentError("--text needs a value")
                }
                options.text = value
                remaining = remaining.dropFirst()
            case "--prepare-assets":
                options.prepareAssets = true
            case "-h", "--help":
                throw ArgumentError("")
            default:
                throw ArgumentError("unknown option: \(argument)")
            }
        }
        return options
    }

    private static func usage(_ message: String) -> Int32 {
        if !message.isEmpty { error("[CoreMLTTSBench] \(message)") }
        log("Usage: MCPZimCoreMLTTSBenchCLI --backend kokoro-ane|supertonic3-int8|supertonic3-int4 [--prepare-assets] [--cache-dir DIR] [--output-dir DIR] [--text TEXT]")
        return message.isEmpty ? 0 : 2
    }

    private struct ArgumentError: LocalizedError {
        let description: String
        init(_ description: String) { self.description = description }
        var errorDescription: String? { description }
    }

    private static func log(_ message: String) {
        FileHandle.standardOutput.write(Data((message + "\n").utf8))
    }

    private static func error(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    private static func physicalFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}
