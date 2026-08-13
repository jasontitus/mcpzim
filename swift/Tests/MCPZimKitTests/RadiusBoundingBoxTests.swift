// SPDX-License-Identifier: MIT
//
// Safety proof for the `nearPlaces` bounding-box prefilter. The scan gate is
// only allowed to be an optimisation: it may let through points the haversine
// then rejects, but it must NEVER reject a point the haversine would have
// accepted — a false reject silently drops a real place from the results.
// These tests fuzz the gate against the same `haversineMeters` the scan uses,
// at the latitudes and radii where the cos-scaling and the ±180 wrap bite.

import Foundation
import XCTest
@testable import MCPZimKit


final class RadiusBoundingBoxTests: XCTestCase {

    /// Deterministic PRNG so a failure is reproducible from the test name
    /// alone — a random seed would make a one-in-a-thousand geometry a
    /// once-a-month flake with no way to re-run it.
    private struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private func wrapLon(_ lon: Double) -> Double {
        var l = lon
        while l > 180 { l -= 360 }
        while l < -180 { l += 360 }
        return l
    }

    /// Centres chosen for the terms that can go wrong: the equator (cos ≈ 1),
    /// mid-latitude, the Arctic (cos ≈ 0.2, where longitude degrees are cheap
    /// and the window must widen), and either side of the antimeridian.
    private static let centers: [(lat: Double, lon: Double)] = [
        (0.0, 0.0), (37.44, -122.16), (51.5, -0.13), (-33.87, 151.2),
        (71.0, 25.8), (78.22, 15.65), (0.0, 179.98), (64.9, -179.95),
        (-45.0, 179.99), (89.0, 10.0),
    ]

    private static let radii: [Double] = [
        50, 250, 1_000, 5_000, 25_000, 120_000, 900_000,
    ]

    func testPrefilterNeverRejectsWhatHaversineAccepts() {
        var rng = SplitMix64(state: 0xC0FF_EE00_1234_5678)
        var accepted = 0
        var rejected = 0
        var inRadius = 0
        for center in Self.centers {
            for radius in Self.radii {
                let bbox = RadiusBoundingBox(centerLat: center.lat,
                                             centerLon: center.lon,
                                             radiusMeters: radius)
                // Sample tight around the boundary (where a too-narrow window
                // shows up) plus a scattering of far points (which the gate is
                // supposed to reject).
                let spanLat = radius / 111_000.0 * 3
                for i in 0..<400 {
                    let far = i % 4 == 3
                    let lat: Double
                    let lon: Double
                    if far {
                        lat = Double.random(in: -90...90, using: &rng)
                        lon = Double.random(in: -180...180, using: &rng)
                    } else {
                        let cosLat = max(0.01, cos(center.lat * .pi / 180))
                        lat = min(90, max(-90,
                            center.lat + Double.random(in: -spanLat...spanLat, using: &rng)))
                        let spanLon = spanLat / cosLat
                        lon = wrapLon(center.lon
                            + Double.random(in: -spanLon...spanLon, using: &rng))
                    }
                    let d = haversineMeters(center.lat, center.lon, lat, lon)
                    let mayBe = bbox.mayBeWithin(lat: lat, lon: lon)
                    if d <= radius {
                        inRadius += 1
                        XCTAssertTrue(
                            mayBe,
                            "prefilter dropped an in-radius point: centre "
                            + "(\(center.lat), \(center.lon)) r=\(radius) m, "
                            + "point (\(lat), \(lon)) at \(d) m")
                    }
                    if mayBe { accepted += 1 } else { rejected += 1 }
                }
            }
        }
        // Guard against a vacuous pass: the gate has to actually reject, and
        // the fuzz has to actually produce in-radius points.
        XCTAssertGreaterThan(rejected, accepted / 10,
                             "prefilter that never rejects buys nothing")
        XCTAssertGreaterThan(inRadius, 1_000,
                             "fuzz produced too few in-radius points to prove anything")
    }

    /// The scan's real shape: a few-km radius over a country-scale spread of
    /// records. This is the claim the perf finding rests on — the gate must
    /// throw away the overwhelming majority before the haversine runs.
    func testPrefilterRejectsNearlyEverythingOutsideASmallRadius() {
        var rng = SplitMix64(state: 0x5EED_0000_0BAD_F00D)
        let bbox = RadiusBoundingBox(centerLat: 37.44, centerLon: -122.16,
                                     radiusMeters: 2_000)
        var survivors = 0
        let total = 50_000
        for _ in 0..<total {
            let lat = Double.random(in: 32...42, using: &rng)
            let lon = Double.random(in: -124 ... -114, using: &rng)
            if bbox.mayBeWithin(lat: lat, lon: lon) { survivors += 1 }
        }
        XCTAssertLessThan(Double(survivors) / Double(total), 0.01,
                          "\(survivors)/\(total) records survived a 2 km gate")
    }

    /// Two points a few hundred metres apart across ±180 differ by ~360 in raw
    /// longitude. A naive `abs(lon - centerLon)` gate would drop them.
    func testPrefilterKeepsNeighboursAcrossTheAntimeridian() {
        let bbox = RadiusBoundingBox(centerLat: 0, centerLon: 179.999,
                                     radiusMeters: 5_000)
        let east = -179.995
        XCTAssertLessThan(haversineMeters(0, 179.999, 0, east), 5_000)
        XCTAssertTrue(bbox.mayBeWithin(lat: 0, lon: east))
    }

    /// A pole inside the window leaves longitude unbounded (every meridian is
    /// metres away), so the longitude gate has to switch itself off rather
    /// than compute a narrow window from cos(90°) ≈ 0.
    func testPrefilterDisablesLongitudeGateAtThePole() {
        let bbox = RadiusBoundingBox(centerLat: 89.999, centerLon: 0,
                                     radiusMeters: 50_000)
        for lon in stride(from: -180.0, through: 180.0, by: 15.0) {
            let d = haversineMeters(89.999, 0, 89.999, lon)
            if d <= 50_000 {
                XCTAssertTrue(bbox.mayBeWithin(lat: 89.999, lon: lon),
                              "dropped a point \(d) m away at lon \(lon)")
            }
        }
    }

    /// Degenerate radii must fail open — the haversine still decides, exactly
    /// as it did before the gate existed.
    func testPrefilterFailsOpenOnDegenerateInput() {
        for radius in [Double.nan, .infinity, -1, 0] {
            let bbox = RadiusBoundingBox(centerLat: 10, centerLon: 20,
                                         radiusMeters: radius)
            if radius.isNaN || radius.isInfinite || radius < 0 {
                XCTAssertTrue(bbox.mayBeWithin(lat: 70, lon: -140),
                              "radius \(radius) must not reject anything")
            }
            // A zero radius may reject, but never the centre itself.
            XCTAssertTrue(bbox.mayBeWithin(lat: 10, lon: 20))
        }
        let bbox = RadiusBoundingBox(centerLat: 10, centerLon: 20, radiusMeters: 1_000)
        XCTAssertTrue(bbox.mayBeWithin(lat: .nan, lon: .nan),
                      "unparseable coordinates fall through to the haversine")
    }
}
