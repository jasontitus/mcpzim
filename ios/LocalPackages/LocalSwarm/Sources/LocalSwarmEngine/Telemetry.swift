import Foundation

/// A thread-safe sliding-window throughput meter. Records byte deliveries and
/// reports the rate over the most recent `window` seconds, which produces a
/// smooth, responsive speed indicator without long-run averaging lag.
public final class RateMeter {
    private struct Sample { let time: TimeInterval; let bytes: Int }

    private let window: TimeInterval
    private var samples: [Sample] = []
    private let lock = NSLock()

    public init(window: TimeInterval = 2.0) {
        self.window = window
    }

    public func record(_ bytes: Int) {
        let now = Date().timeIntervalSinceReferenceDate
        lock.lock()
        samples.append(Sample(time: now, bytes: bytes))
        prune(now)
        lock.unlock()
    }

    public func reset() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
    }

    public var bytesPerSecond: Double {
        lock.lock(); defer { lock.unlock() }
        let now = Date().timeIntervalSinceReferenceDate
        prune(now)
        guard let first = samples.first, samples.count > 0 else { return 0 }
        let span = max(now - first.time, 0.001)
        let total = samples.reduce(0) { $0 + $1.bytes }
        return Double(total) / span
    }

    /// Throughput in megabits per second (decimal megabits, matching link-rate convention).
    public var megabitsPerSecond: Double {
        bytesPerSecond * 8.0 / 1_000_000.0
    }

    private func prune(_ now: TimeInterval) {
        let cutoff = now - window
        while let first = samples.first, first.time < cutoff {
            samples.removeFirst()
        }
    }
}

public extension Double {
    /// Human-readable transfer rate from a **bytes-per-second** value, in
    /// megabytes/gigabytes per second, e.g. "4.8 MB/s" or "1.2 GB/s".
    var formattedByteRate: String {
        let megabytesPerSecond = self / 1_000_000 // decimal MB, matching ByteCountFormatter(.file)
        if megabytesPerSecond >= 1000 {
            return String(format: "%.2f GB/s", megabytesPerSecond / 1000)
        } else if megabytesPerSecond >= 1 {
            return String(format: "%.1f MB/s", megabytesPerSecond)
        } else {
            return String(format: "%.0f KB/s", max(0, self / 1000))
        }
    }
}
