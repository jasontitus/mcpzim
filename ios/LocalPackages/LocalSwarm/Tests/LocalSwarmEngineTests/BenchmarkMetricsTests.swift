import XCTest
@testable import LocalSwarmEngine

final class BenchmarkMetricsTests: XCTestCase {
    /// Steady 10 MB/s for 5s: every sustained window should be ~10 MB/s, and
    /// min ≤ avg ≤ max must hold (the old metric reported min > avg).
    func testSteadyTransferWindows() {
        let rate: Int64 = 10_000_000
        let progress = (0...5).map { (t: Double($0), bytes: rate * Int64($0)) }
        let (lo, hi) = SwarmManager.windowedRates(progress: progress, duration: 5)
        XCTAssertEqual(lo, Double(rate), accuracy: 1)
        XCTAssertEqual(hi, Double(rate), accuracy: 1)
        let avg = Double(rate * 5) / 5.0
        XCTAssertLessThanOrEqual(lo, avg)
        XCTAssertLessThanOrEqual(avg, hi)
    }

    /// A mid-transfer stall (bad conditions) must surface as a low minimum window.
    func testStallShowsAsLowMinimum() {
        // 0–1s: +10MB, 1–2s warmup skip; 2–3s: stall (+0), 3–4s: +10MB, 4–5s: +10MB
        let progress: [(t: Double, bytes: Int64)] = [
            (0, 0), (1, 10_000_000), (2, 20_000_000),
            (3, 20_000_000), // stalled window [2,3)
            (4, 30_000_000), (5, 40_000_000),
        ]
        let (lo, hi) = SwarmManager.windowedRates(progress: progress, duration: 5)
        XCTAssertEqual(lo, 0, accuracy: 1, "the stalled second should be the minimum")
        XCTAssertEqual(hi, 10_000_000, accuracy: 1)
    }

    /// Too-short transfer (no full windows) returns zeros so the caller falls back.
    func testShortTransferHasNoWindows() {
        let (lo, hi) = SwarmManager.windowedRates(progress: [(0, 0), (0.5, 5_000_000)], duration: 0.9)
        XCTAssertEqual(lo, 0)
        XCTAssertEqual(hi, 0)
    }
}
