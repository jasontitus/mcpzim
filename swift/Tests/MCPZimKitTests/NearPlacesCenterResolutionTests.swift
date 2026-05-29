// SPDX-License-Identifier: MIT
//
// Regression for the 2026-05-29 "could not resolve 37.44124,-122.15530"
// bug: the LFM2.5 FT stuffed the user's coordinates into near_places'
// `place` field (and lat/lon), and the adapter tried to GEOCODE the
// coordinate string as a place name → zero results. near_places must
// prefer real coords and parse a "lat,lon" `place` rather than geocode it.

import Foundation
import XCTest
@testable import MCPZimKit

final class NearPlacesCenterResolutionTests: XCTestCase {

    func testParseLatLon() {
        XCTAssertNotNil(MCPToolAdapter.parseLatLon("37.44124,-122.15530"))
        XCTAssertNotNil(MCPToolAdapter.parseLatLon(" 37.44 , -122.16 "))
        XCTAssertNil(MCPToolAdapter.parseLatLon("Palo Alto"))
        XCTAssertNil(MCPToolAdapter.parseLatLon("37.44"))        // single number
        XCTAssertNil(MCPToolAdapter.parseLatLon("200,300"))      // out of range
        XCTAssertNil(MCPToolAdapter.parseLatLon(""))
    }

    private func cafeFixture() -> StubZimService.Fixture {
        var fix = StubZimService.Fixture()
        let cafe = Place(name: "Allegro Coffee Company", kind: "poi",
                         lat: 37.441, lon: -122.155, subtype: "cafe")
        let result = NearPlacesResult(
            totalInRadius: 1, breakdown: ["cafe": 1],
            results: [(cafe, 380.0)])
        fix.nearPlaces[StubZimService.keyNearPlaces(
            lat: 37.441, lon: -122.155, kinds: ["cafe"])] = .init(result: result)
        return fix
    }

    func testCoordsInBothPlaceAndLatLonUseCoordPath() async throws {
        // The exact failing shape: coords in `place` AND lat/lon.
        let svc = StubZimService(fixture: cafeFixture())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        let r = try await adapter.dispatch(tool: "near_places", args: [
            "place": "37.441,-122.155",
            "lat": 37.441, "lon": -122.155,
            "kinds": ["cafe"], "radius_km": 1.0,
        ])
        XCTAssertNil(r["error"], "must not geocode the coord string: \(r)")
        let rows = try XCTUnwrap(r["results"] as? [[String: Any]])
        XCTAssertEqual(rows.first?["name"] as? String, "Allegro Coffee Company")
    }

    func testCoordStringInPlaceOnlyIsParsedNotGeocoded() async throws {
        // Coords only in `place` (no explicit lat/lon) → parse, don't geocode.
        let svc = StubZimService(fixture: cafeFixture())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        let r = try await adapter.dispatch(tool: "near_places", args: [
            "place": "37.441, -122.155", "kinds": ["cafe"], "radius_km": 1.0,
        ])
        XCTAssertNil(r["error"], "coord string in place must parse: \(r)")
        let rows = try XCTUnwrap(r["results"] as? [[String: Any]])
        XCTAssertEqual(rows.first?["name"] as? String, "Allegro Coffee Company")
    }
}
