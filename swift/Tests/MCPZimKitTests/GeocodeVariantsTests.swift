// SPDX-License-Identifier: MIT
//
// Pins the `geocodeVariants` fallback ladder. Field evidence (device log
// 2026-08-03, report 5393c74a): locate("k1 kart") threw noMatch even though
// the streetzim name index holds "K1 Speed" — the query has no comma and no
// " in ", so the only attempt was the full phrase and .contains("k1 kart")
// matched nothing. The ladder now appends progressive trailing-token-drop
// variants ("k1 kart" → "k1") AFTER the existing suffix-stripped variants,
// so exact/full-phrase matches keep winning and the token drop only runs
// when everything more specific missed.

import Foundation
import XCTest
@testable import MCPZimKit

final class GeocodeVariantsTests: XCTestCase {

    // MARK: - Variant-ladder ordering (unit)

    func testTokenDropAppendsAfterFullQuery() {
        XCTAssertEqual(DefaultZimService.geocodeVariants(of: "k1 kart"),
                       ["k1 kart", "k1"],
                       "full phrase first — exact matches must still win")
    }

    func testCommaVariantStillPrecedesTokenDrops() {
        // The pre-existing comma strip must keep its slot: "Union Square"
        // resolves cleanly and the ladder stops there, so appending
        // "Union" after it can never change a query that used to work.
        XCTAssertEqual(
            DefaultZimService.geocodeVariants(of: "Union Square, San Francisco"),
            ["Union Square, San Francisco", "Union Square", "Union"])
    }

    func testInVariantStillPrecedesTokenDrops() {
        XCTAssertEqual(
            DefaultZimService.geocodeVariants(of: "Union Square in San Francisco"),
            ["Union Square in San Francisco", "Union Square", "Union"])
    }

    func testSingleTokenQueryUnchanged() {
        XCTAssertEqual(DefaultZimService.geocodeVariants(of: "Stanford"),
                       ["Stanford"],
                       "nothing to drop — single-token behavior must not change")
    }

    func testTwoCharFloorStopsDegenerateVariants() {
        // Dropping "b" from "a b" would leave the 1-char "a", which as a
        // substring filter matches half the index. The floor rejects it.
        XCTAssertEqual(DefaultZimService.geocodeVariants(of: "a b"), ["a b"])
    }

    func testVariantListCappedAtFive() {
        let variants = DefaultZimService.geocodeVariants(
            of: "one two three four five six seven")
        XCTAssertEqual(variants.count, 5, "cap keeps a long phrase from fanning out")
        XCTAssertEqual(variants.first, "one two three four five six seven")
        XCTAssertEqual(variants, [
            "one two three four five six seven",
            "one two three four five six",
            "one two three four five",
            "one two three four",
            "one two three",
        ], "each variant drops exactly one trailing token, in order")
    }

    // MARK: - End-to-end through the real geocoder

    private final class MapReader: ZimReader, @unchecked Sendable {
        let store: [String: Data]
        init(_ json: [String: String]) {
            store = json.reduce(into: [:]) { $0[$1.key] = Data($1.value.utf8) }
        }
        var metadata: ZimMetadata { ZimMetadata(name: "osm-test") }
        var kind: ZimKind { .streetzim }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { false }
        var hasRoutingData: Bool { true }
        func read(path: String) throws -> ZimEntry? {
            guard let data = store[path] else { return nil }
            return ZimEntry(path: path, title: path, mimetype: "application/json", content: data)
        }
        func readMainPage() throws -> ZimEntry? { nil }
    }

    private func k1Service() -> DefaultZimService {
        let reader = MapReader([
            "search-data/manifest.json": #"{"chunks":{"k1":2,"un":2}}"#,
            "search-data/k1.json": """
            [{"n":"K1 Speed","t":"poi","s":"karting","a":37.3710,"o":-121.9250,"l":"Santa Clara"},
             {"n":"K12 Online School","t":"poi","s":"school","a":37.4000,"o":-121.9000,"l":"San Jose"}]
            """,
            "search-data/un.json": """
            [{"n":"Union Square Cafe","t":"poi","s":"cafe","a":37.7880,"o":-122.4074,"l":"San Francisco"},
             {"n":"Union City","t":"place","s":"city","a":37.5934,"o":-122.0439,"l":"California"}]
            """,
        ])
        return DefaultZimService(readers: [(name: "osm-test", reader: reader)])
    }

    func testK1KartResolvesViaTokenDrop() async throws {
        // The 2026-08-03 field case: no record contains "k1 kart", but the
        // dropped-token variant "k1" ranks "K1 Speed" by prefix (offset 0,
        // shortest name beats "K12 Online School").
        let hits = try await k1Service().geocode(
            query: "k1 kart", limit: 3, zim: nil, kinds: nil)
        XCTAssertEqual(hits.first?.name, "K1 Speed")
    }

    func testExactNameStillWinsWithoutFallback() async throws {
        let hits = try await k1Service().geocode(
            query: "K1 Speed", limit: 1, zim: nil, kinds: nil)
        XCTAssertEqual(hits.map(\.name), ["K1 Speed"])
    }

    func testFullPhraseMatchShortCircuitsTokenDrop() async throws {
        // "union square cafe" matches a record outright, so the ladder must
        // stop there — the later "union" variant would also match "Union
        // City", which must NOT appear.
        let hits = try await k1Service().geocode(
            query: "union square cafe", limit: 3, zim: nil, kinds: nil)
        XCTAssertEqual(hits.map(\.name), ["Union Square Cafe"])
    }
}
