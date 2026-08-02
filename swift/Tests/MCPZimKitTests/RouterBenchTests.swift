// SPDX-License-Identifier: MIT
//
// Micro-benchmark for the deterministic turn-classification path — the
// pre-generation work every user turn pays before anything else happens.
// Run with: swift test --filter RouterBenchTests
// Compares regex-compilation caching across revisions (DS4 pass E6).

import XCTest
@testable import MCPZimKit

final class RouterBenchTests: XCTestCase {

    func testClassifyThroughput() {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.remember(FocusEntity(name: "Apple Tv", kind: .topic, zimPath: "A/Apple_Tv"))
        let turns = [
            "Tell me about gravitational waves",
            "When were they first detected?",
            "What is the most recent version?",
            "No. What is the most recent version of Apple TV?",
            "how far is the nearest coffee shop",
            "directions to Stanford University",
            "what's around here?",
            "And the OS?",
            "yes",
            "read me the whole article",
        ]
        let loc: (lat: Double, lon: Double)? = (37.441, -122.155)
        // Warmup (populates any lazy caches so both revisions measure steady state).
        for t in turns { _ = IntentRouter.classify(t, currentLocation: loc, focus: focus) }
        let iters = 300
        let start = DispatchTime.now()
        for _ in 0..<iters {
            for t in turns {
                _ = IntentRouter.classify(t, currentLocation: loc, focus: focus)
            }
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        let perCall = elapsed / Double(iters * turns.count) * 1e6
        print(String(format: "ROUTER-BENCH: %d classifies in %.3fs → %.1f µs/classify",
                     iters * turns.count, elapsed, perCall))
    }
}
