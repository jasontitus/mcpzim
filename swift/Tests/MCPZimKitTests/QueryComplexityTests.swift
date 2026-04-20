// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class QueryComplexityTests: XCTestCase {
    private func assertClassifies(_ text: String, as expected: QueryComplexity,
                                   file: StaticString = #filePath, line: UInt = #line) {
        let got = QueryComplexity.classify(text)
        XCTAssertEqual(got, expected, "«\(text)» → \(got.rawValue), expected \(expected.rawValue)",
                       file: file, line: line)
    }

    func testNavigational() {
        assertClassifies("what is in adams morgan", as: .navigational)
        assertClassifies("what's around dupont circle", as: .navigational)
        assertClassifies("bars near the white house", as: .navigational)
        assertClassifies("route from adams morgan to the capitol", as: .navigational)
        assertClassifies("How do I get to Georgetown?", as: .navigational)
        assertClassifies("directions from here to the monument", as: .navigational)
        assertClassifies("what's the nearest post office", as: .navigational)
    }

    func testExplanatory() {
        assertClassifies("how does a jet engine work?", as: .explanatory)
        assertClassifies("explain the french revolution", as: .explanatory)
        assertClassifies("why did the roman empire fall", as: .explanatory)
        assertClassifies("compare chopin and liszt", as: .explanatory)
        assertClassifies("what caused the great depression", as: .explanatory)
        assertClassifies("explain how photosynthesis works", as: .explanatory)
    }

    func testFactoid() {
        assertClassifies("when was the war of 1812?", as: .factoid)
        assertClassifies("who was Marie Curie?", as: .factoid)
        assertClassifies("how many kings of england were named henry", as: .factoid)
        assertClassifies("where is mount everest", as: .factoid)
    }

    func testTopical() {
        assertClassifies("tell me about the war of 1812", as: .topical)
        assertClassifies("what is the grand duchy of lithuania", as: .topical)
        assertClassifies("give me a history of venice", as: .topical)
        assertClassifies("tell me about aspirin", as: .topical)
    }

    // A "where is X" query is a factoid lookup in the general case,
    // but when X is a street-level entity it should feel
    // navigational. We accept that the heuristic mislabels some
    // "where is" queries for now — it's a conscious trade for
    // keeping the classifier model-free.
    func testKnownAmbiguityDocumented() {
        // Genuinely encyclopedic: factoid is right.
        assertClassifies("where was ronald reagan born", as: .factoid)
        // Street-scale: the heuristic still calls it factoid. The
        // downstream system handles by letting the model choose the
        // right tool, so this isn't a correctness bug — but it's
        // worth revisiting if we see frustration in logs.
        assertClassifies("where is dupont circle", as: .factoid)
    }
}
