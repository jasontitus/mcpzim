import Foundation

/// One leg of a benchmark: the same content downloaded over a single transport,
/// with overall + min/max throughput.
public struct BenchmarkLeg: Sendable, Identifiable {
    public let transport: Transport
    public let totalBytes: Int64
    public let durationSeconds: Double
    public let averageBytesPerSecond: Double
    public let minBytesPerSecond: Double
    public let maxBytesPerSecond: Double
    public let succeeded: Bool

    public var id: String { transport.rawValue }

    public var averageMBps: Double { averageBytesPerSecond / 1_000_000 }
    public var minMBps: Double { minBytesPerSecond / 1_000_000 }
    public var maxMBps: Double { maxBytesPerSecond / 1_000_000 }
}

/// The result of benchmarking the same file across transports back-to-back.
public struct BenchmarkReport: Sendable {
    public let name: String
    public let legs: [BenchmarkLeg]

    /// The faster transport by overall average, if at least two legs succeeded.
    public var winner: Transport? {
        let ok = legs.filter(\.succeeded)
        guard ok.count >= 2 else { return nil }
        return ok.max(by: { $0.averageBytesPerSecond < $1.averageBytesPerSecond })?.transport
    }

    /// How much faster the winner is than the slowest successful leg (e.g. 1.8×).
    public var speedup: Double? {
        let ok = legs.filter { $0.succeeded && $0.averageBytesPerSecond > 0 }
        guard ok.count >= 2,
              let best = ok.map(\.averageBytesPerSecond).max(),
              let worst = ok.map(\.averageBytesPerSecond).min(),
              worst > 0 else { return nil }
        return best / worst
    }
}

/// Live progress while a benchmark runs.
public struct BenchmarkProgress: Sendable {
    public var transport: Transport
    public var fractionComplete: Double
    public var bytesPerSecond: Double
}
