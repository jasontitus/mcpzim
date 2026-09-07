import CoreML
import Darwin
import FluidAudio
import Foundation
import MCPZimKit
#if HAS_KOKORO_PIPELINE
import KokoroPipeline
#endif

// No playback or microphone APIs. Each invocation measures exactly one backend.
private func now() -> Double { ProcessInfo.processInfo.systemUptime }
private func footprint() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / 4)
    let status = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    precondition(status == KERN_SUCCESS, "Cannot measure physical footprint")
    return Double(info.phys_footprint) / 1_048_576
}

private final class MemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: Double = 0
    private let timer: DispatchSourceTimer
    init() {
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "tts.memory"))
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }
    func reset() { lock.lock(); peak = footprint(); lock.unlock() }
    func sample() { lock.lock(); peak = max(peak, footprint()); lock.unlock() }
    func read() -> Double { sample(); lock.lock(); defer { lock.unlock() }; return peak }
    deinit { timer.cancel() }
}

private struct Render {
    var samples: [Float] = []
    var sampleRate = 24_000
    var firstPCM: Double?
    var chunks = 0
    var maxGap = 0.0
    var requiredStartupBuffer = 0.0
    private var lastArrival = 0.0
    init(sampleRate: Int = 24_000) { self.sampleRate = sampleRate }
    mutating func append(_ pcm: [Float], elapsed: Double) {
        guard !pcm.isEmpty else { return }
        if firstPCM == nil { firstPCM = elapsed }
        else { maxGap = max(maxGap, elapsed - lastArrival) }
        // The extra initial buffer needed to avoid underrun once frame 1 arrives.
        requiredStartupBuffer = max(requiredStartupBuffer,
            elapsed - (firstPCM ?? elapsed) - Double(samples.count) / Double(sampleRate))
        samples.append(contentsOf: pcm)
        lastArrival = elapsed
        chunks += 1
    }
}

private struct Engine {
    let initialize: () async throws -> Void
    let render: (String) async throws -> Render
    let cleanup: () async -> Void
}

@main
struct Bench {
    static func emit(_ value: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data + Data([10]))
    }

    static func main() async {
        do { try await run() }
        catch {
            emit(["event": "error", "message": String(describing: error)])
            exit(1)
        }
    }

    static func run() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let backend = args.first else { throw Failure("backend required") }
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), args.indices.contains(i + 1) else { return nil }
            return args[i + 1]
        }
        let cache = URL(fileURLWithPath: value("--cache") ?? "/private/tmp/zimfo-tts-models")
        let output = URL(fileURLWithPath: value("--output") ?? "/private/tmp/zimfo-tts-samples")
        let steps = Int(value("--steps") ?? "8") ?? 8
        let voice = value("--voice")
        let fixturePath = value("--fixture")
        guard fixturePath == nil || backend == "kokoro-staged" else {
            throw Failure("--fixture applies only to kokoro-staged; use --text for raw-text backends")
        }
        let turns = Int(value("--turns") ?? "10") ?? 10
        guard (1...30).contains(turns), (1...16).contains(steps) else { throw Failure("invalid limits") }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let corpus = [
            "Dr. Vladimir V. Putin attended Leningrad State University; later, he joined the KGB in 1975.",
            "Napoleon Bonaparte was a French general and statesman who rose to prominence during the French Revolution. He led military campaigns across Europe from 1796 to 1815.",
            "The museum is 800 meters away. The coffee shop is nearby, on the other side of the street.",
            "Would you like to hear about Napoleon's early life, his military campaigns, or his legacy?",
            "Albert Einstein was born on 14 March 1879. His work influenced the development of modern physics.",
        ]
        let fixture = try fixturePath.map { try JSONDecoder().decode(
            PreparedInput.self, from: Data(contentsOf: URL(fileURLWithPath: $0))) }
        let engine = try makeEngine(backend, cache: cache, steps: steps, voice: voice, fixture: fixture)
        let sampler = MemorySampler()
        let baseline = footprint()
        emit(["event": "start", "backend": backend, "steps": steps,
              "voice": voice ?? "default", "os": ProcessInfo.processInfo.operatingSystemVersionString,
              "baseline_mib": baseline, "cache": cache.path, "playback": false,
              "input_boundary": fixture == nil ? "raw_text" : "prepared_phoneme_tokens"])
        sampler.reset()
        let initStart = now()
        try await engine.initialize()
        let initSeconds = now() - initStart
        emit(["event": "initialize", "seconds": initSeconds,
              "peak_mib": sampler.read(), "resident_mib": footprint(),
              "includes_downloads": args.contains("--prepare")])
        if !args.contains("--prepare") {
            for turn in 0..<turns {
                // Repeat the first phrase before introducing different lengths/buckets.
                let text = fixture?.text ?? value("--text") ?? corpus[turn < 2 ? 0 : (turn - 1) % corpus.count]
                sampler.reset()
                let start = now()
                let audio = try await engine.render(text)
                let seconds = now() - start
                let peak = sampler.read()
                guard !audio.samples.isEmpty, audio.samples.allSatisfy({ $0.isFinite }),
                      audio.firstPCM != nil else { throw Failure("empty or invalid PCM") }
                let duration = Double(audio.samples.count) / Double(audio.sampleRate)
                let rms = sqrt(audio.samples.reduce(0.0) { $0 + Double($1) * Double($1) }
                    / Double(audio.samples.count))
                guard rms > 0.00001 else { throw Failure("silent PCM") }
                emit(["event": "turn", "turn": turn + 1, "characters": text.count,
                      "text": text, "seconds": seconds, "first_pcm_seconds": audio.firstPCM!,
                      "audio_seconds": duration, "rtfx": duration / seconds,
                      "chunks": audio.chunks, "max_chunk_gap_seconds": audio.maxGap,
                      "startup_buffer_seconds": audio.requiredStartupBuffer,
                      "peak_mib": peak, "resident_mib": footprint(), "rms": rms,
                      "peak_amplitude": audio.samples.map { abs($0) }.max()!,
                      "clipped_fraction": Double(audio.samples.filter { abs($0) > 1 }.count) / Double(audio.samples.count)])
                if args.contains("--save"), turn < 3 {
                    let wav = try AudioWAV.data(from: audio.samples,
                        sampleRate: Double(audio.sampleRate), normalize: false)
                    try wav.write(to: output.appendingPathComponent("\(backend)-\(voice ?? "default")-s\(steps)-turn\(turn + 1).wav"))
                }
            }
        }
        try await Task.sleep(nanoseconds: 1_000_000_000)
        emit(["event": "settled", "resident_mib": footprint()])
        await engine.cleanup()
        try await Task.sleep(nanoseconds: 1_000_000_000)
        emit(["event": "cleanup", "resident_mib": footprint(), "baseline_mib": baseline])
    }

    private static func makeEngine(_ backend: String, cache: URL, steps: Int, voice: String?, fixture: PreparedInput?) throws -> Engine {
        #if HAS_KOKORO_PIPELINE
        if backend == "kokoro-staged" {
            guard let fixture else { throw Failure("kokoro-staged requires --fixture") }
            let weights = try JSONDecoder().decode(HarmonicWeights.self,
                from: Data(contentsOf: cache.appendingPathComponent("hnsf_weights.json")))
            let box = PipelineBox()
            return Engine(initialize: {
                box.value = try KokoroPipeline(modelsDirectory: cache.appendingPathComponent("coreml"),
                    buckets: [7], linearWeights: weights.linear_weights, linearBias: weights.linear_bias)
            }, render: { text in
                guard text == fixture.text else { throw Failure("fixture text mismatch") }
                let start = now()
                let audio = try box.value!.synthesize(inputIds: fixture.input_ids,
                    attentionMask: fixture.attention_mask, refS: fixture.ref_s, speed: 1)
                var result = Render()
                result.append(audio.audio, elapsed: now() - start)
                return result
            }, cleanup: { box.value = nil })
        }
        #endif
        if backend.hasPrefix("supertonic-") {
            guard ["supertonic-int8", "supertonic-int4"].contains(backend),
                  let selected = Supertonic3Voice(rawValue: voice ?? "F1") else { throw Failure("invalid Supertonic configuration") }
            let manager = Supertonic3Manager(directory: cache,
                vectorEstimator: .aneBucketed(backend == "supertonic-int4" ? .int4 : .int8))
            let style = StyleBox()
            return Engine(initialize: {
                try await manager.initialize()
                style.value = try await Supertonic3ResourceDownloader.loadVoiceStyle(selected, directory: cache)
            }, render: { text in
                let start = now()
                var result = Render(sampleRate: 44_100)
                var remaining = text
                while let prefix = StreamingSpeechPolicy.takeSpeakablePrefix(remaining,
                    generating: false, allowEarlyClause: true, minimumClause: 56, maximumClause: 94) {
                    var spoken = prefix.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if prefix.boundary == .softWrap { spoken += "," }
                    let audio = try await manager.synthesize(text: spoken, language: "en",
                        style: style.value!, totalSteps: steps)
                    result.append(audio.samples, elapsed: now() - start)
                    remaining = String(remaining.dropFirst(prefix.consumedCharacters))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }
                return result
            }, cleanup: { await manager.cleanup(); style.value = nil })
        }
        if ["kokoro-ane", "kokoro-ane-cpu-tail"].contains(backend) {
            let manager = KokoroAneManager(variant: .english,
                defaultVoice: voice ?? "af_heart", directory: cache,
                computeUnits: backend == "kokoro-ane-cpu-tail"
                    ? KokoroAneComputeUnits(noise: .cpuOnly, tail: .cpuOnly) : .default)
            return Engine(initialize: { try await manager.initialize(preloadVoices: [voice ?? "af_heart"]) },
                render: { text in
                    let start = now()
                    let audio = try await manager.synthesizeDetailed(text: text, voice: voice ?? "af_heart", speed: 1)
                    var result = Render(sampleRate: audio.sampleRate)
                    result.append(audio.samples, elapsed: now() - start)
                    return result
                }, cleanup: { await manager.cleanup() })
        }
        if ["pocket-fp16", "pocket-int8", "pocket-ane", "pocket-ane-state"].contains(backend) {
            let manager = PocketTtsManager(defaultVoice: voice ?? "alba", directory: cache,
                precision: backend == "pocket-int8" ? .int8 : .fp16,
                placement: backend == "pocket-ane-state" ? .aneState : (backend == "pocket-ane" ? .ane : .gpu))
            return Engine(initialize: { try await manager.initialize() }, render: { text in
                let start = now()
                var result = Render()
                let stream = try await manager.synthesizeStreaming(text: text, voice: voice ?? "alba", temperature: 0.3)
                for try await frame in stream { result.append(frame.samples, elapsed: now() - start) }
                return result
            }, cleanup: { await manager.cleanup() })
        }
        #if HAS_INFLECT
        if ["inflect-micro", "inflect-nano", "inflect-micro-cpu"].contains(backend) {
            let manager = InflectManager(variant: backend == "inflect-nano" ? .nano : .micro,
                directory: cache, computeUnits: backend.hasSuffix("-cpu") ? .cpuOnly : .cpuAndGPU)
            return Engine(initialize: { try await manager.initialize() }, render: { text in
                let start = now()
                let audio = try await manager.synthesize(text: text)
                var result = Render()
                result.append(audio, elapsed: now() - start)
                return result
            }, cleanup: { await manager.cleanup() })
        }
        #endif
        throw Failure("unsupported backend: \(backend)")
    }
    private struct Failure: Error { let message: String; init(_ message: String) { self.message = message } }
    private final class StyleBox { var value: Supertonic3VoiceStyle? }
    private struct PreparedInput: Decodable {
        let text: String
        let input_ids: [Int32]
        let attention_mask: [Int32]
        let ref_s: [Float]
    }
    #if HAS_KOKORO_PIPELINE
    private struct HarmonicWeights: Decodable { let linear_weights: [Float]; let linear_bias: Float }
    private final class PipelineBox { var value: KokoroPipeline? }
    #endif
}
