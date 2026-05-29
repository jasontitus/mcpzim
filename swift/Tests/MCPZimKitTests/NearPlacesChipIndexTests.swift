// SPDX-License-Identifier: MIT
//
// Exercises the real `ZimService.nearPlaces` category-index fast path
// against the THREE index layers a streetzim can ship, using an in-memory
// `ZimReader` that serves hand-built JSON in the exact on-disk shapes:
//
//   1. New `--no-llm-bundle` ZIMs — `category-index/chip-{id}.json` (the
//      web Find-page data) + light `categories` (place/park/…). NO poi blob.
//   2. Legacy ZIMs — `category-index/{poi,place,…}.json` bundle, no chips.
//   3. search-data prefix chunks — the web search box's own data, used as
//      the fallback for generic queries and kinds with no chip.
//
// This is the regression net for the streetzim-alignment change: mcpzim
// must read the same data the web viewer reads and must NOT depend on the
// `poi`/`addr`/`street` LLM bundle that streetzim now omits by default.

import Foundation
import XCTest

@testable import MCPZimKit

final class NearPlacesChipIndexTests: XCTestCase {

    // MARK: - In-memory streetzim reader

    /// Minimal `ZimReader` backed by a path→JSON map. Reports as a
    /// streetzim with routing data so `ZimService.streetzimReaders`
    /// includes it; only `read(path:)` is meaningful here.
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

    private func service(_ json: [String: String]) -> DefaultZimService {
        DefaultZimService(readers: [(name: "osm-test", reader: MapReader(json))])
    }

    // Palo Alto-ish center; "near" fixtures sit ~200 m away, "far"
    // fixtures sit >100 km away (excluded by a 5 km radius).
    private let lat = 37.441
    private let lon = -122.155

    // MARK: - Layer 1: new --no-llm-bundle ZIM (chips)

    /// A new ZIM: manifest advertises `chips` + light `categories`, and
    /// ships chip files. Crucially there is NO `category-index/poi.json`.
    private func newZimWithChips() -> [String: String] {
        [
            "category-index/manifest.json": """
            {"total":5,
             "categories":{"place":{},"park":{},"water":{}},
             "chips":{"restaurants":{"label":"Restaurants","count":3},
                      "cafes":{"label":"Cafés","count":1},
                      "parks":{"label":"Parks","count":1}}}
            """,
            "category-index/chip-restaurants.json": """
            [{"n":"Joe's Diner","t":"poi","s":"restaurant","a":37.4423,"o":-122.1553,"l":"Palo Alto"},
             {"n":"Pizza Bella","t":"poi","s":"restaurant","a":37.4404,"o":-122.1561,"l":"Palo Alto"},
             {"n":"Distant Grill","t":"poi","s":"restaurant","a":38.58,"o":-121.49,"l":"Sacramento"}]
            """,
            "category-index/chip-cafes.json": """
            [{"n":"Blue Bottle Coffee","t":"poi","s":"cafe","a":37.4419,"o":-122.1550,"l":"Palo Alto"}]
            """,
            "category-index/chip-parks.json": """
            [{"n":"Mitchell Park","t":"park","s":"park","a":37.4250,"o":-122.1090,"l":"Palo Alto"}]
            """,
            "category-index/place.json": """
            [{"n":"Palo Alto","t":"place","s":"city","a":37.4419,"o":-122.1430,"l":"California"}]
            """,
        ]
    }

    func testNewZimBroadRestaurantUsesChipNoFilter() async throws {
        // kinds=["restaurant"] → restaurants chip, returned whole (web
        // chip-tap parity). The two PA restaurants are in radius; the
        // Sacramento one is excluded by distance, not by kind.
        let svc = service(newZimWithChips())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["restaurant"], zim: nil, hasWiki: false)
        let names = Set(r.results.map { $0.place.name })
        XCTAssertEqual(names, ["Joe's Diner", "Pizza Bella"],
                       "broad restaurant query returns the whole in-radius chip slice")
        XCTAssertEqual(r.totalInRadius, 2)
    }

    func testNewZimNicheKindNarrowsWithinChip() async throws {
        // kinds=["pizza"] → restaurants chip, but narrowed: only the
        // pizza-named record survives scanRecords' name-keyword filter.
        let svc = service(newZimWithChips())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["pizza"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Pizza Bella"],
                       "niche 'pizza' narrows within the restaurants chip")
    }

    func testNewZimCoffeeUsesCafesChip() async throws {
        let svc = service(newZimWithChips())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["coffee"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Blue Bottle Coffee"])
    }

    func testMultiWordKindResolvesToChip() async throws {
        // The model emits kinds=["coffee shop"] (two words). It must resolve
        // to the cafes chip via chipsFor's word-split — NOT miss every chip
        // and fall through to the search-data scan (which jetsammed the app
        // on a statewide ZIM). Regression for the 2026-05-29 crash.
        let svc = service(newZimWithChips())
        for term in ["coffee shop", "Coffee Shop", "gas station"] {
            let r = try await svc.nearPlaces(
                lat: lat, lon: lon, radiusKm: 5, limit: 20,
                kinds: [term], zim: nil, hasWiki: false)
            if term.contains("coffee") {
                XCTAssertEqual(r.results.map { $0.place.name }, ["Blue Bottle Coffee"],
                               "\(term) → cafes chip")
            }
            // (no fuel fixture here; the point is it doesn't scan/crash)
        }
    }

    func testHugeSearchDataSkipsFullScanNoCrash() async throws {
        // A statewide ZIM declares millions of search-data records. An
        // unmapped kind must NOT trigger the full scan (it OOM'd at 5.4 GB).
        // The record-count cap returns empty instead of loading every chunk.
        var json = newZimWithChips()
        json["search-data/manifest.json"] = #"{"chunks":{"00":600000,"01":500000}}"#
        // (No chunk files — if the guard failed and it tried to scan, it'd
        //  still not crash here, but on-device those chunks exist and OOM.)
        let svc = service(json)
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["atm"], zim: nil, hasWiki: false)   // "atm" has no chip
        XCTAssertEqual(r.results.count, 0,
                       "1.1M-record search-data must be skipped, not scanned")
    }

    func testNewZimPlaceUsesLightCategory() async throws {
        // kinds=["place"] → the manifest's light `place` category (cities),
        // not any chip. Confirms direct-category hits still work alongside
        // chips.
        let svc = service(newZimWithChips())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["place"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Palo Alto"])
    }

    func testNewZimParkPrefersChipOverCategory() async throws {
        // "park" exists as BOTH a chip and a light category. We must load
        // it once (via the chip), not twice — assert a single result.
        let svc = service(newZimWithChips())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["park"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Mitchell Park"],
                       "park resolves once via the chip, not duplicated from the category")
    }

    // MARK: - Layer 2: legacy ZIM (poi bundle, no chips)

    private func legacyZimWithPoiBundle() -> [String: String] {
        [
            "category-index/manifest.json": """
            {"total":3,"categories":{"poi":{},"place":{},"park":{}}}
            """,
            "category-index/poi.json": """
            [{"n":"Joe's Diner","t":"poi","s":"restaurant","a":37.4423,"o":-122.1553,"l":"Palo Alto"},
             {"n":"Town Hardware","t":"poi","s":"shop","a":37.4410,"o":-122.1540,"l":"Palo Alto"}]
            """,
            "category-index/place.json": """
            [{"n":"Palo Alto","t":"place","s":"city","a":37.4419,"o":-122.1430,"l":"California"}]
            """,
        ]
    }

    func testLegacyZimStillUsesPoiBundle() async throws {
        // Old installed ZIMs that predate --no-llm-bundle still have
        // poi.json; the legacy subtype-filter path must keep working.
        let svc = service(legacyZimWithPoiBundle())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["restaurant"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Joe's Diner"],
                       "legacy poi bundle still filters by subtype")
    }

    // MARK: - Layer 3: search-data fallback

    private func newZimPlusSearchData() -> [String: String] {
        var json = newZimWithChips()
        json["search-data/manifest.json"] = #"{"chunks":{"00":2}}"#
        json["search-data/00.json"] = """
        [{"n":"Chase ATM","t":"poi","s":"bank","a":37.4420,"o":-122.1551,"l":"Palo Alto"},
         {"n":"Joe's Diner","t":"poi","s":"restaurant","a":37.4423,"o":-122.1553,"l":"Palo Alto"}]
        """
        return json
    }

    func testKindWithNoChipFallsToSearchData() async throws {
        // "atm" maps to no chip and no category; the call must fall through
        // to the search-data scan (the web search box's data) and find the
        // bank-tagged record via the synonym table.
        let svc = service(newZimPlusSearchData())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["atm"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Chase ATM"],
                       "no-chip kind falls back to the search-data scan")
    }

    func testGenericQueryScansSearchData() async throws {
        // No kinds pinned → skip the kind-partitioned index entirely and
        // scan search-data, which sees every record regardless of kind.
        let svc = service(newZimPlusSearchData())
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: nil, zim: nil, hasWiki: false)
        XCTAssertEqual(Set(r.results.map { $0.place.name }),
                       ["Chase ATM", "Joe's Diner"],
                       "generic query scans the full search-data chunk")
    }
}
