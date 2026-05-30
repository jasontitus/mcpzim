// SPDX-License-Identifier: MIT
//
// Exercises the `locate` tool added for debug report AFA0ECA1, where
// "Where is Stanford Hospital?" had no clean path: the model stripped the
// name to a `near_places` category scan, then bailed to
// `get_article_section` (hard 404). `locate` resolves ONE named place to a
// single labeled pin via the geocode/search-data index — which ships even
// on `--no-llm-bundle` ZIMs — and returns it in the near_places render
// shape so the host's PlacesWebView draws a centered pin.

import Foundation
import XCTest
@testable import MCPZimKit

final class LocateToolTests: XCTestCase {

    private func stanfordFixture() -> StubZimService.Fixture {
        var fix = StubZimService.Fixture()
        let hit = Place(name: "Stanford Hospital", kind: "poi",
                        lat: 37.4350, lon: -122.1750, subtype: "hospital")
        // geocode key is the lowercased query (kinds ignored by the stub),
        // so both the place+poi pass and the all-kinds fallback resolve it.
        fix.geocode[StubZimService.keyGeocode(query: "Stanford Hospital")] = [hit]
        return fix
    }

    func testLocateResolvesNamedPlaceToSinglePin() async throws {
        let svc = StubZimService(fixture: stanfordFixture())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        let r = try await adapter.dispatch(tool: "locate", args: [
            "place": "Stanford Hospital",
        ])
        XCTAssertNil(r["error"], "locate should resolve, not error: \(r)")
        // Single pin in the near_places render shape (what PlacesWebView reads).
        let rows = try XCTUnwrap(r["results"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 1, "locate returns exactly the resolved place")
        XCTAssertEqual(rows.first?["name"] as? String, "Stanford Hospital")
        XCTAssertEqual(rows.first?["lat"] as? Double, 37.4350)
        XCTAssertEqual(r["total_in_radius"] as? Int, 1)
        // `resolved` is the map center the host pans to.
        let resolved = try XCTUnwrap(r["resolved"] as? [String: Any])
        XCTAssertEqual(resolved["name"] as? String, "Stanford Hospital")
    }

    func testLocateResultRendersAsPlaces() async throws {
        // The render gate `traceHasPlaces` requires a non-empty parsed
        // places list — confirm the kit parser sees our payload.
        let svc = StubZimService(fixture: stanfordFixture())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        let r = try await adapter.dispatch(tool: "locate", args: ["place": "Stanford Hospital"])
        let raw = String(data: try JSONSerialization.data(withJSONObject: r), encoding: .utf8)!
        let payload = parsePlacesJSON(rawResult: raw)
        XCTAssertEqual(payload.places.count, 1, "PlacesWebView would render one pin")
        XCTAssertEqual(payload.places.first?.label, "Stanford Hospital")
        XCTAssertTrue(placesToolNames.contains("locate"),
                      "locate must be in the canonical render-gate set")
    }

    func testLocateCaptionNamesResolvedPlace() {
        // Caption names what RESOLVED (which may differ from what was asked),
        // and avoids the near_places "Found N <kind>" count phrasing.
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "locate",
            args: ["place": "Stanford Hospital"],
            fullResult: [
                "resolved": ["name": "Stanford Health Care"],
                "results": [["name": "Stanford Health Care"]],
            ]
        )
        XCTAssertTrue(s.contains("Stanford Health Care"), "got: \(s)")
        XCTAssertFalse(s.contains("Found 1"), "locate must not use near-count phrasing: \(s)")
    }

    func testLocateMissThrowsGeocodeError() async throws {
        var fix = StubZimService.Fixture()
        // Both the place+poi pass and the all-kinds fallback come back empty.
        fix.geocode[StubZimService.keyGeocode(query: "Nowhere Place")] = []
        let svc = StubZimService(fixture: fix)
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        do {
            _ = try await adapter.dispatch(tool: "locate", args: ["place": "Nowhere Place"])
            XCTFail("expected a geocode miss to throw")
        } catch {
            XCTAssertTrue(String(describing: error).contains("could not resolve"),
                          "got: \(error)")
        }
    }
}
