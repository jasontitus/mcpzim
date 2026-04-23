// SPDX-License-Identifier: MIT
//
// Labelled memory-sampling harness used by the multi-model eval runner
// (see `EVAL_HARNESS.md` Phase 1). Wraps `MemoryStats.physFootprintMB()`
// with tagged snapshots + an optional continuous-sampling background
// task, and spits out a summary per variant (baseline, post-load,
// peaks, post-turn, lifetime peak).
//
// Sample tags follow a loose convention — `baseline`, `post_load`,
// `prefill.<scenario>.<turn>`, `decode.<scenario>.<turn>`,
// `post_turn.<scenario>.<turn>` — but any string is fine; the summary
// just picks out the ones it knows about.

import Foundation

public struct MemorySample: Sendable {
    public let tag: String
    public let rssMB: Double
    public let timestamp: Date
    public init(tag: String, rssMB: Double, timestamp: Date = Date()) {
        self.tag = tag
        self.rssMB = rssMB
        self.timestamp = timestamp
    }
}

public struct MemorySummary: Sendable {
    public let variant: String
    public let baselineMB: Double?
    public let postLoadMB: Double?
    public let lifetimePeakMB: Double
    public let postTurnHighMB: Double?      // worst post-turn snapshot
    public let sampleCount: Int
    /// Counts of continuous-sample observations at or above each
    /// threshold — answers "how often did we hit N GB?" without
    /// requiring raw sample dumps. Thresholds match the iPhone
    /// jetsam-risk bands we care about (5 / 6 / 7 GB).
    public let samplesAtOrAbove5GB: Int
    public let samplesAtOrAbove6GB: Int
    public let samplesAtOrAbove7GB: Int

    public func scorecardRow(namePad: Int = 34) -> String {
        func col(_ v: Double?) -> String {
            if let v { return String(format: "%6.0f MB", v) }
            return "    —   "
        }
        let name = variant.padding(toLength: namePad, withPad: " ", startingAt: 0)
        let basic = "\(name) | baseline=\(col(baselineMB)) post_load=\(col(postLoadMB)) peak=\(col(lifetimePeakMB)) post_turn=\(col(postTurnHighMB)) (n=\(sampleCount))"
        // A phone's jetsam cap is ~6 GB on current iPhones; the 5 / 6 / 7
        // breakdown turns the aggregate peak into a "how often would we
        // risk dying" number. 0/0/0 is safe; anything >0 on the 6/7 GB
        // columns needs attention before shipping the scenario on phone.
        let bands = String(
            format: "  ≥5GB: %d  ≥6GB: %d  ≥7GB: %d",
            samplesAtOrAbove5GB,
            samplesAtOrAbove6GB,
            samplesAtOrAbove7GB
        )
        return basic + bands
    }
}

public actor MemoryProbe {
    public let variant: String
    private var samples: [MemorySample] = []
    private var continuous: Task<Void, Never>?

    public init(variant: String) {
        self.variant = variant
    }

    /// Take a one-shot sample with the given tag.
    @discardableResult
    public func sample(_ tag: String) -> MemorySample {
        let s = MemorySample(tag: tag, rssMB: MemoryStats.physFootprintMB())
        samples.append(s)
        return s
    }

    /// Start a background sampling loop at `intervalMs`. Tags each
    /// sample with `"\(prefix).#\(i)"`. The caller should `await stop()`
    /// before summarising.
    public func startContinuous(tagPrefix: String, intervalMs: Int = 100) {
        stop()
        let interval = UInt64(max(1, intervalMs)) * 1_000_000
        continuous = Task { [weak self] in
            guard let self else { return }
            var i = 0
            while !Task.isCancelled {
                await self.sample("\(tagPrefix).#\(i)")
                i += 1
                try? await Task.sleep(nanoseconds: interval)
            }
        }
    }

    public func stop() {
        continuous?.cancel()
        continuous = nil
    }

    public func all() -> [MemorySample] { samples }

    public func summary() -> MemorySummary {
        let baseline = samples.first(where: { $0.tag == "baseline" })?.rssMB
        let postLoad = samples.first(where: { $0.tag == "post_load" })?.rssMB
        let postTurnHigh = samples
            .filter { $0.tag.hasPrefix("post_turn") }
            .map(\.rssMB).max()
        let peak = samples.map(\.rssMB).max() ?? 0
        var ge5 = 0, ge6 = 0, ge7 = 0
        for s in samples {
            if s.rssMB >= 7000 { ge7 += 1 }
            if s.rssMB >= 6000 { ge6 += 1 }
            if s.rssMB >= 5000 { ge5 += 1 }
        }
        return MemorySummary(
            variant: variant,
            baselineMB: baseline,
            postLoadMB: postLoad,
            lifetimePeakMB: peak,
            postTurnHighMB: postTurnHigh,
            sampleCount: samples.count,
            samplesAtOrAbove5GB: ge5,
            samplesAtOrAbove6GB: ge6,
            samplesAtOrAbove7GB: ge7
        )
    }
}
