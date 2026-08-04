// SPDX-License-Identifier: MIT
//
// Covers the two recovery layers for `near_places` kinds the chip table
// doesn't know. Field evidence (device log 2026-08-03, report 5393c74a):
// near_places(kinds: ["go cart place"]) on the statewide California ZIM
// mapped to no chip, the OOM guard (correctly) refused the full scan, and
// the call returned 0 hits in 2 ms with no further attempt.
//
//   1. kindSynonyms — predictable phrasings ("drugstore", "petrol
//      station") canonicalize onto the existing 11 chips before lookup.
//   2. Name-search fallback — kinds that STILL map to no chip get a
//      bounded name search over only the prefix chunks their own words
//      select; a miss stays a clean fast empty, never a full scan.

import Foundation
import XCTest
@testable import MCPZimKit

final class NearPlacesKindFallbackTests: XCTestCase {

    // MARK: - Fixtures

    /// Records every `read(path:)` so tests can assert which search-data
    /// chunks the fallback did — and crucially did NOT — load.
    private final class TrackingReader: ZimReader, @unchecked Sendable {
        private let store: [String: Data]
        private let lock = NSLock()
        private var paths: [String] = []
        init(_ json: [String: String]) {
            store = json.reduce(into: [:]) { $0[$1.key] = Data($1.value.utf8) }
        }
        var metadata: ZimMetadata { ZimMetadata(name: "osm-test") }
        var kind: ZimKind { .streetzim }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { false }
        var hasRoutingData: Bool { true }
        func read(path: String) throws -> ZimEntry? {
            lock.lock(); paths.append(path); lock.unlock()
            guard let data = store[path] else { return nil }
            return ZimEntry(path: path, title: path, mimetype: "application/json", content: data)
        }
        func readMainPage() throws -> ZimEntry? { nil }
        func searchChunkReads() -> [String] {
            lock.lock(); defer { lock.unlock() }
            return paths.filter {
                $0.hasPrefix("search-data/") && $0 != "search-data/manifest.json"
            }
        }
    }

    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func append(_ s: String) { lock.lock(); lines.append(s); lock.unlock() }
        func all() -> [String] { lock.lock(); defer { lock.unlock() }; return lines }
    }

    // Palo Alto-ish center, matching the other near-places suites.
    private let lat = 37.441
    private let lon = -122.155

    // MARK: - kindSynonyms table + chipsFor

    func testKindSynonymsLandOnExistingChips() {
        // Each phrasing must resolve to exactly the chip its canonical
        // term resolves to — the whole point of the table.
        let expected: [(String, [String])] = [
            ("petrol", ["fuel"]), ("petrol station", ["fuel"]),
            ("gasoline", ["fuel"]),
            ("drugstore", ["health"]), ("chemist", ["health"]),
            ("er", ["health"]), ("emergency room", ["health"]),
            ("urgent care", ["health"]),
            ("coffee house", ["cafes"]), ("coffeehouse", ["cafes"]),
            ("tavern", ["bars"]), ("boozer", ["bars"]),
            ("inn", ["hotels"]), ("bnb", ["hotels"]),
            ("bed and breakfast", ["hotels"]),
            ("art gallery", ["museums"]),
            ("playground", ["parks"]), ("garden", ["parks"]),
            ("bookstore", ["shops"]), ("bookshop", ["shops"]),
            ("market", ["shops"]),
        ]
        for (term, chips) in expected {
            XCTAssertEqual(DefaultZimService.chipsFor(term), chips, "term: \(term)")
        }
    }

    func testSynonymBeatsWordSplitMisroute() {
        // Without the synonym pass sitting before the word-split,
        // "bed and breakfast" split to "breakfast" → restaurants chip.
        XCTAssertEqual(DefaultZimService.chipsFor("bed and breakfast"), ["hotels"])
    }

    func testExistingChipTermsUnchanged() {
        XCTAssertEqual(DefaultZimService.chipsFor("restaurant"), ["restaurants"])
        XCTAssertEqual(DefaultZimService.chipsFor("coffee shop"), ["cafes"])
        XCTAssertEqual(DefaultZimService.chipsFor("gas station"), ["fuel"])
        XCTAssertEqual(DefaultZimService.chipsFor("ice cream"), ["restaurants", "cafes"])
        XCTAssertEqual(DefaultZimService.chipsFor("go cart place"), [],
                       "unpredictable phrasings stay unmapped — the name-search "
                           + "fallback owns them, not a guessy synonym")
    }

    func testSynonymTableInvariants() {
        // Both call sites apply the map exactly once, so values must be
        // real chipsForKind keys and never other synonym keys (else the
        // single application would land somewhere unmapped).
        for (key, value) in DefaultZimService.kindSynonyms {
            XCTAssertNotNil(DefaultZimService.chipsForKind[value],
                            "'\(key)' → '\(value)' must target a chipsForKind key")
            XCTAssertNil(DefaultZimService.kindSynonyms[value],
                         "'\(key)' → '\(value)' must not chain to another synonym")
        }
    }

    // MARK: - Synonyms end-to-end (niche/broad inheritance)

    /// Same shape as the AFA0ECA1 health-chip fixture: one chip bundling
    /// distinct subtypes.
    private func healthChipZim() -> [String: String] {
        [
            "category-index/manifest.json": """
            {"total":3,"categories":{"place":{}},
             "chips":{"health":{"label":"Health","count":3}}}
            """,
            "category-index/chip-health.json": """
            [{"n":"Stanford Hospital","t":"poi","s":"hospital","a":37.4350,"o":-122.1750,"l":"Stanford"},
             {"n":"Bright Smiles Dental","t":"poi","s":"dentist","a":37.4419,"o":-122.1550,"l":"Palo Alto"},
             {"n":"CVS Pharmacy","t":"poi","s":"pharmacy","a":37.4410,"o":-122.1540,"l":"Palo Alto"}]
            """,
        ]
    }

    func testDrugstoreNarrowsLikePharmacy() async throws {
        // "drugstore" must inherit pharmacy's NICHE behavior — chip choice,
        // niche decision, and subtype filter all see the canonical term.
        // A chip-lookup-only mapping would have returned the whole health
        // chip: the AFA0ECA1 "211 hospitals" shape all over again.
        let svc = DefaultZimService(readers: [
            (name: "osm-test", reader: TrackingReader(healthChipZim())),
        ])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["drugstore"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["CVS Pharmacy"])
        XCTAssertEqual(r.breakdown, ["pharmacy": 1],
                       "no hospital/dentist leakage from the shared chip")
    }

    func testEmergencyRoomNarrowsToHospitals() async throws {
        let svc = DefaultZimService(readers: [
            (name: "osm-test", reader: TrackingReader(healthChipZim())),
        ])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["emergency room"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Stanford Hospital"])
        XCTAssertEqual(r.breakdown, ["hospital": 1])
    }

    func testTavernReturnsWholeBarsChip() async throws {
        // "bar" is a broad term, so its synonyms are too — a tavern query
        // means "bars near me", the full chip slice like the web chip tap.
        let svc = DefaultZimService(readers: [
            (name: "osm-test", reader: TrackingReader([
                "category-index/manifest.json": """
                {"total":2,"categories":{},
                 "chips":{"bars":{"label":"Bars","count":2}}}
                """,
                "category-index/chip-bars.json": """
                [{"n":"The Rose and Crown","t":"poi","s":"pub","a":37.4423,"o":-122.1553,"l":"Palo Alto"},
                 {"n":"Antonio's Nut House","t":"poi","s":"bar","a":37.4410,"o":-122.1540,"l":"Palo Alto"}]
                """,
            ])),
        ])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["tavern"], zim: nil, hasWiki: false)
        XCTAssertEqual(Set(r.results.map { $0.place.name }),
                       ["The Rose and Crown", "Antonio's Nut House"])
    }

    // MARK: - Name-search fallback, OOM-guard path (statewide ZIM)

    /// A statewide-shaped ZIM: chips exist (so the guard applies), the
    /// declared search-data record counts exceed the full-scan cap, and
    /// the chunks the fallback's words select actually exist. The "zz"
    /// chunk is the tripwire: nothing routes to it, so any read of it
    /// means the fallback degenerated into a scan.
    private func statewideZim() -> [String: String] {
        [
            "category-index/manifest.json": """
            {"total":1,"categories":{"place":{}},
             "chips":{"restaurants":{"label":"Restaurants","count":1}}}
            """,
            "category-index/chip-restaurants.json": """
            [{"n":"Joe's Diner","t":"poi","s":"restaurant","a":37.4423,"o":-122.1553,"l":"Palo Alto"}]
            """,
            "search-data/manifest.json":
                #"{"chunks":{"go":300000,"ca":300000,"zz":100000}}"#,
            "search-data/go.json": """
            [{"n":"Go Cart Raceway","t":"poi","s":"leisure","a":37.4423,"o":-122.1553,"l":"Palo Alto"},
             {"n":"Go Cart Raceway Fresno","t":"poi","s":"leisure","a":36.7500,"o":-119.7700,"l":"Fresno"},
             {"n":"Go Cart Alley","t":"street","a":37.4419,"o":-122.1550,"l":"Palo Alto"}]
            """,
            "search-data/ca.json": """
            [{"n":"Castle Bakery","t":"poi","s":"bakery","a":37.4420,"o":-122.1551,"l":"Palo Alto"}]
            """,
            "search-data/zz.json": """
            [{"n":"Zz Kart Palace","t":"poi","s":"leisure","a":37.4410,"o":-122.1540,"l":"Palo Alto"}]
            """,
        ]
    }

    func testUnmappedKindNameFallbackOnGuardedZim() async throws {
        // The 2026-08-03 field case, with the venue present under its
        // literal name. The guard still refuses the scan, but the fallback
        // loads only the "go" and "ca" chunks (phrase/word prefixes) and
        // finds the in-radius name hit.
        let reader = TrackingReader(statewideZim())
        let svc = DefaultZimService(readers: [(name: "osm-test", reader: reader)])
        let logs = LogBox()
        await svc.setLogger { logs.append($0) }
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["go cart place"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Go Cart Raceway"],
                       "in-radius poi name hit; Fresno excluded by radius, "
                           + "'Go Cart Alley' excluded as a street record")
        XCTAssertEqual(r.totalInRadius, 1)
        let reads = Set(reader.searchChunkReads())
        XCTAssertEqual(reads, ["search-data/go.json", "search-data/ca.json"],
                       "only the prefixes the kind's words select — never zz")
        XCTAssertTrue(logs.all().contains {
            $0.contains("kind 'go cart place' unmapped — name-search fallback")
        }, "the fallback must announce itself: \(logs.all())")
    }

    func testFallbackMissStaysCleanAndFast() async throws {
        // No prefix the kind's words select exists → not a single chunk
        // read, and a clean empty result. The guard's promise ("never a
        // full scan") must survive the fallback's addition.
        let reader = TrackingReader(statewideZim())
        let svc = DefaultZimService(readers: [(name: "osm-test", reader: reader)])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["quokka sanctuary"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.count, 0)
        XCTAssertEqual(r.totalInRadius, 0)
        XCTAssertEqual(reader.searchChunkReads(), [],
                       "prefixes 'qu'/'sa' aren't in the manifest — zero chunk reads")
    }

    func testMappedKindNeverTriggersNameFallback() async throws {
        // "hotel" maps to the hotels chip, which this ZIM doesn't ship —
        // the plan comes up empty and control reaches the guard. Even so
        // it must NOT proceed to a name search: mapped kinds already had
        // their shot at the partitioned index, and a name pass would only
        // add noise ("Hotel Street", stale references) to a definitive
        // empty.
        let reader = TrackingReader(statewideZim())
        let svc = DefaultZimService(readers: [(name: "osm-test", reader: reader)])
        let logs = LogBox()
        await svc.setLogger { logs.append($0) }
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["hotel"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.count, 0)
        XCTAssertEqual(reader.searchChunkReads(), [],
                       "guard path, and no name-search reads for a chip-mapped kind")
        XCTAssertFalse(logs.all().contains { $0.contains("name-search fallback") })
    }

    // MARK: - Name-search fallback, hot-split leaves

    func testFallbackWalksHotSplitLeaves() async throws {
        // Continent-scale builds fan a hot prefix into sub-chunk leaves;
        // the fallback must reach records through the same sub_chunks
        // machinery the geocoder uses, not just plain chunks.
        let reader = TrackingReader([
            "category-index/manifest.json": """
            {"total":1,"categories":{},
             "chips":{"restaurants":{"label":"Restaurants","count":1}}}
            """,
            "search-data/manifest.json": """
            {"chunks":{"go-0":300000,"go-1":300000},
             "sub_chunks":{"go":["go-0","go-1"]}}
            """,
            "search-data/go-0.json": """
            [{"n":"Golden Fields Farm","t":"poi","s":"farm","a":36.7500,"o":-119.7700,"l":"Fresno"}]
            """,
            "search-data/go-1.json": """
            [{"n":"Go Cart World","t":"poi","s":"leisure","a":37.4404,"o":-122.1561,"l":"Palo Alto"}]
            """,
        ])
        let svc = DefaultZimService(readers: [(name: "osm-test", reader: reader)])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["go cart place"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Go Cart World"],
                       "found in a fan-out leaf via the 'go cart' term")
    }

    // MARK: - Name-search fallback, post-full-scan path (small ZIM)

    func testSmallZimFullScanMissThenNameFallback() async throws {
        // Under the record cap the full scan runs, but its kind filter
        // knows nothing about "go cart place" → zero hits. The name
        // fallback must then recover the venue the scan just walked past.
        let reader = TrackingReader([
            "search-data/manifest.json": #"{"chunks":{"go":2,"ca":1}}"#,
            "search-data/go.json": """
            [{"n":"Go Cart Raceway","t":"poi","s":"leisure","a":37.4423,"o":-122.1553,"l":"Palo Alto"},
             {"n":"Golden Bagels","t":"poi","s":"bakery","a":37.4410,"o":-122.1540,"l":"Palo Alto"}]
            """,
            "search-data/ca.json": """
            [{"n":"Castle Bakery","t":"poi","s":"bakery","a":37.4420,"o":-122.1551,"l":"Palo Alto"}]
            """,
        ])
        let svc = DefaultZimService(readers: [(name: "osm-test", reader: reader)])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: ["go cart place"], zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Go Cart Raceway"])
    }

    func testGenericQueryBehaviorUnchanged() async throws {
        // No kinds pinned → the pre-existing generic scan, untouched by
        // the fallback (it is gated on !isGeneric).
        let reader = TrackingReader([
            "search-data/manifest.json": #"{"chunks":{"go":1}}"#,
            "search-data/go.json": """
            [{"n":"Go Cart Raceway","t":"poi","s":"leisure","a":37.4423,"o":-122.1553,"l":"Palo Alto"}]
            """,
        ])
        let svc = DefaultZimService(readers: [(name: "osm-test", reader: reader)])
        let r = try await svc.nearPlaces(
            lat: lat, lon: lon, radiusKm: 5, limit: 20,
            kinds: nil, zim: nil, hasWiki: false)
        XCTAssertEqual(r.results.map { $0.place.name }, ["Go Cart Raceway"])
    }
}
