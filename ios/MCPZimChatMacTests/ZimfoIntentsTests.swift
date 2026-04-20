// SPDX-License-Identifier: MIT
//
// End-to-end-ish tests for Zimfo's App Intents. Runs the intents'
// `perform()` directly, bypassing Siri / Shortcuts so we don't have
// to drive any UI — handy for iteration on tool-chaining logic.
//
// The tests fall into two buckets:
//   - **Pure** (always run): `ZimfoContext` round-trip, `RouteProgress`
//     math. No ZIMs, no CoreLocation.
//   - **End-to-end** (opt-in): actually spins up `ZimfoRunner`, hits
//     real libzim readers, runs real routing / geocoding. Requires at
//     least one streetzim ZIM in the app's sandbox Documents or in the
//     persisted external bookmarks. Skipped when no data is available.

import XCTest
import CoreLocation
@testable import MCPZimChatMac

final class ZimfoContextTests: XCTestCase {
    func testActiveRoutePersistsAcrossInstances() async throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zimfo-ctx-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let route = ActiveRoute(
            startedAt: Date(),
            origin: .init(lat: 38.9, lon: -77.05),
            destination: .init(lat: 38.91, lon: -77.04),
            originName: "here",
            destinationName: "White House",
            zim: "osm-dc.zim",
            totalDurationSeconds: 600,
            totalDistanceMeters: 1000,
            polyline: [.init(lat: 38.9, lon: -77.05), .init(lat: 38.91, lon: -77.04)],
            cumulativeDistanceMeters: [0, 1000],
            turnByTurn: ["step 1", "step 2"]
        )

        let a = ZimfoContext(storeURL: tmp)
        await a.setActiveRoute(route)

        let b = ZimfoContext(storeURL: tmp)
        let reloaded = await b.activeRoute
        XCTAssertEqual(reloaded?.destinationName, "White House")
        XCTAssertEqual(reloaded?.totalDistanceMeters, 1000)
        XCTAssertEqual(reloaded?.turnByTurn.count, 2)
    }

    func testClearActiveRouteIsPersistent() async throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zimfo-ctx-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let ctx = ZimfoContext(storeURL: tmp)
        await ctx.setActiveRoute(.init(
            startedAt: Date(),
            origin: .init(lat: 0, lon: 0),
            destination: .init(lat: 0, lon: 0),
            originName: "", destinationName: "", zim: "",
            totalDurationSeconds: 0, totalDistanceMeters: 0,
            polyline: [], cumulativeDistanceMeters: [], turnByTurn: []
        ))
        await ctx.clearActiveRoute()

        let reloaded = ZimfoContext(storeURL: tmp)
        let r = await reloaded.activeRoute
        XCTAssertNil(r)
    }
}

final class RouteProgressTests: XCTestCase {
    /// 10 km straight east along the equator, evenly sampled.
    private func syntheticRoute(totalMeters: Double, totalSeconds: Double, points: Int = 11) -> ActiveRoute {
        let stepLon = (totalMeters / 111_111) / Double(points - 1) // ~1m = 1/111111° lon at equator
        let coords = (0..<points).map { i in
            ActiveRoute.Coordinate(lat: 0, lon: Double(i) * stepLon)
        }
        var cum: [Double] = [0]
        for i in 1..<points {
            cum.append(Double(i) * totalMeters / Double(points - 1))
        }
        return ActiveRoute(
            startedAt: Date(timeIntervalSinceNow: -60),
            origin: coords.first!,
            destination: coords.last!,
            originName: "start",
            destinationName: "end",
            zim: "test.zim",
            totalDurationSeconds: totalSeconds,
            totalDistanceMeters: totalMeters,
            polyline: coords,
            cumulativeDistanceMeters: cum,
            turnByTurn: ["go east"]
        )
    }

    func testRemainingAtStart() {
        let r = syntheticRoute(totalMeters: 10_000, totalSeconds: 600)
        let (m, s) = RouteProgress.remaining(for: r, current: r.origin)
        XCTAssertEqual(m, 10_000, accuracy: 500)
        XCTAssertEqual(s, 600, accuracy: 30)
    }

    func testRemainingAtHalfway() {
        let r = syntheticRoute(totalMeters: 10_000, totalSeconds: 600)
        let mid = r.polyline[5] // sample #5 of 11 → ~middle
        let (m, s) = RouteProgress.remaining(for: r, current: mid)
        XCTAssertEqual(m, 5_000, accuracy: 600)
        XCTAssertEqual(s, 300, accuracy: 40)
    }

    func testRemainingAtEnd() {
        let r = syntheticRoute(totalMeters: 10_000, totalSeconds: 600)
        let (m, s) = RouteProgress.remaining(for: r, current: r.destination)
        XCTAssertLessThan(m, 500)
        XCTAssertLessThan(s, 30)
    }
}

// MARK: - End-to-end intent tests (require a real streetzim)

final class ZimfoIntentEndToEndTests: XCTestCase {
    /// Paths the test looks for. Uses the first one that exists, so a
    /// developer can drop any streetzim into the app sandbox and run the
    /// tests; if none found, the e2e tests skip rather than fail.
    private static func candidateStreetzimURLs() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let sandbox = home
            .appendingPathComponent("Library")
            .appendingPathComponent("Containers")
            .appendingPathComponent("org.mcpzim.MCPZimChatMac")
            .appendingPathComponent("Data")
            .appendingPathComponent("Documents")
        let explicit = ProcessInfo.processInfo.environment["ZIMBLE_TEST_STREETZIM"]
            .map { URL(fileURLWithPath: $0) }
        return [
            explicit,
            try? FileManager.default.contentsOfDirectory(at: sandbox, includingPropertiesForKeys: nil)
                .first(where: { $0.pathExtension == "zim" && $0.lastPathComponent.contains("osm") })
        ].compactMap { $0 }
    }

    override func setUp() async throws {
        // Canned "current location" for intents that ask for it. Lat/lon
        // of the White House — picked because our main test ZIMs include
        // DC. If a different streetzim is provided via the env var,
        // override this.
        LocationFetcher.overrideForTesting = {
            CLLocationCoordinate2D(latitude: 38.8977, longitude: -77.0365)
        }
    }

    override func tearDown() async throws {
        LocationFetcher.overrideForTesting = nil
        await ZimfoContext.shared.clearActiveRoute()
    }

    @MainActor
    func testStartRouteIntentThenRemaining() async throws {
        guard !Self.candidateStreetzimURLs().isEmpty else {
            throw XCTSkip("No streetzim ZIM available; drop one into the app's Documents folder or set $ZIMBLE_TEST_STREETZIM.")
        }

        // 1. Start a route via the intent.
        let start = StartRouteIntent()
        start.destination = "Lincoln Memorial"
        _ = try await start.perform()

        let route = await ZimfoContext.shared.activeRoute
        XCTAssertNotNil(route, "StartRouteIntent should store an active route")
        XCTAssertGreaterThan(route?.totalDistanceMeters ?? 0, 100)
        XCTAssertFalse(route?.polyline.isEmpty ?? true)

        // 2. Ask remaining — should be close to total right after start.
        let rem = RemainingRouteIntent()
        _ = try await rem.perform()
        // (We don't assert a specific string; the return type is opaque.
        // Just confirm the context was read without crashing.)
    }

    @MainActor
    func testNearbyHereIntentReturnsSomething() async throws {
        guard !Self.candidateStreetzimURLs().isEmpty else {
            throw XCTSkip("No streetzim ZIM available.")
        }
        let intent = NearbyHereIntent()
        _ = try await intent.perform()
        // The intent dialog isn't easily introspectable from here, but
        // we're asserting it doesn't throw — the underlying
        // `ZimfoRunner.nearbySummary` path does the real work.
    }
}
