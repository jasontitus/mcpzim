// SPDX-License-Identifier: MIT
//
// Parser + A* + geocoder tests that mirror tests/test_routing.py and
// tests/test_geocode.py in the Python suite.

import XCTest
@testable import MCPZimKit

final class SZRGGraphTests: XCTestCase {
    // Build a small graph in-memory so we don't need a real ZIM file.
    // 4 nodes in a 1-degree/1000-grid square:
    //     B(0, 0.01) --- C(0.01, 0.01)
    //       |                 |
    //     A(0, 0) --- D(0.01, 0)
    // Edge weights pick "North Rd" (A->B->C at 50 km/h) over "West Rd"
    // (A->D->C at 30 km/h) so we can verify A* chose correctly.
    private static func buildGridBlob() -> Data {
        let nodes: [(Double, Double)] = [(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)]
        let names = ["", "North Rd", "West Rd"]
        let d = haversineMeters(0, 0, 0, 0.01)
        let edges: [(Int, Int, Double, Int, Int)] = [
            (0, 1, d, 50, 1),
            (1, 2, d, 50, 1),
            (0, 3, d, 30, 2),
            (3, 2, d, 30, 2),
        ]
        return encodeGraphV2(nodes: nodes, edges: edges, names: names)
    }

    func testParseHeader() throws {
        let blob = Self.buildGridBlob()
        let graph = try SZRGGraph.parse(blob)
        XCTAssertEqual(graph.numNodes, 4)
        XCTAssertEqual(graph.numEdges, 4)
        XCTAssertEqual(graph.names, ["", "North Rd", "West Rd"])
    }

    func testAStarPicksFasterNamedRoute() throws {
        let blob = Self.buildGridBlob()
        let graph = try SZRGGraph.parse(blob)
        guard let route = aStar(graph: graph, origin: 0, goal: 2) else {
            return XCTFail("no route")
        }
        // A->B->C should coalesce into a single "North Rd" segment.
        XCTAssertEqual(route.roads.map { $0.name }, ["North Rd"])
        let d = haversineMeters(0, 0, 0, 0.01)
        XCTAssertEqual(route.distanceMeters, 2 * d, accuracy: 1.0)
        XCTAssertEqual(route.durationSeconds, 2 * d * 3.6 / 50, accuracy: 1.0)
    }

    func testAStarReturnsNilForDisjointComponents() throws {
        let names = [""]
        let d = haversineMeters(0, 0, 0, 0.001)
        let nodes: [(Double, Double)] = [(0, 0), (0, 0.001), (1, 1), (1, 1.001)]
        let edges: [(Int, Int, Double, Int, Int)] = [
            (0, 1, d, 50, 0), (1, 0, d, 50, 0),
            (2, 3, d, 50, 0), (3, 2, d, 50, 0),
        ]
        let graph = try SZRGGraph.parse(encodeGraphV2(nodes: nodes, edges: edges, names: names))
        XCTAssertNil(aStar(graph: graph, origin: 0, goal: 2))
    }

    func testNearestNode() throws {
        let blob = Self.buildGridBlob()
        let graph = try SZRGGraph.parse(blob)
        XCTAssertEqual(graph.nearestNode(lat: 0.0099, lon: 0.0099), 2)
        XCTAssertEqual(graph.nearestNode(lat: 0, lon: 0), 0)
    }

    func testGeocoderPrefixMatchesPython() {
        XCTAssertEqual(Geocoder.normalizePrefix("New York"), "ne")
        XCTAssertEqual(Geocoder.normalizePrefix("a"), "a_")
        XCTAssertEqual(Geocoder.normalizePrefix(""), "__")
        XCTAssertEqual(Geocoder.normalizePrefix("42nd Street"), "42")
        XCTAssertEqual(Geocoder.normalizePrefix("_private"), "_p")
    }

    func testGeocoderPrefixNonAscii() {
        // Non-ASCII first char → codepoint-hex bucket. Prevents all CJK /
        // Cyrillic / Arabic records from collapsing into a single __.json.
        //
        // Note: Swift's ``lowercased`` doesn't do NFKD decomposition, so
        // diacritics survive here. The Python writer DOES apply NFKD,
        // which strips combining marks but leaves CJK/Cyrillic/Arabic
        // base characters intact — so the codepoint in the bucket key
        // matches for non-decomposable scripts, which is the case we
        // care about.
        XCTAssertEqual(Geocoder.normalizePrefix("東京"), "u6771")
        XCTAssertEqual(Geocoder.normalizePrefix("Подвесной мост"), "u43f")
        XCTAssertEqual(Geocoder.normalizePrefix("ابوظبي"), "u627")
    }
}
