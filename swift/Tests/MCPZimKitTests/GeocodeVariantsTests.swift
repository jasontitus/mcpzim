import XCTest
@testable import MCPZimKit

final class GeocodeVariantsTests: XCTestCase {
    func testNamesAreNeverShortenedIntoDifferentEntities() {
        for name in ["k1 kart", "a good coffee shop in Salinas", "one two three four five"] {
            let variants = DefaultZimService.geocodeVariants(of: name)
            XCTAssertFalse(variants.contains("k1"))
            XCTAssertFalse(variants.contains("a good"))
            XCTAssertFalse(variants.contains("one two"))
        }
        XCTAssertEqual(DefaultZimService.geocodeVariants(of: "Union Square, San Francisco"),
                       ["Union Square, San Francisco", "Union Square"])
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

    func testK1KartCannotSilentlyBecomeK1Speed() async throws {
        let hits = try await k1Service().geocode(query: "k1 kart", limit: 3, zim: nil, kinds: nil)
        XCTAssertTrue(hits.isEmpty)
    }

    func testGeographicQualifierSurvivesFallback() async throws {
        let correct = try await k1Service().geocode(query: "Union Square, San Francisco", limit: 1, zim: nil, kinds: nil)
        XCTAssertEqual(correct.first?.name, "Union Square Cafe")
        let wrong = try await k1Service().geocode(query: "Union Square, Salinas", limit: 1, zim: nil, kinds: nil)
        XCTAssertTrue(wrong.isEmpty)
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
