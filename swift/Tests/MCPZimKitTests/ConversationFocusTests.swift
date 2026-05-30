// SPDX-License-Identifier: MIT
//
// State-transition tests for `ConversationFocus`: recency ordering, dedupe,
// the enumerated "last list", and the movement trail's jitter filter.

import Foundation
import XCTest

@testable import MCPZimKit

final class ConversationFocusTests: XCTestCase {

    func testRememberMovesToFrontAndDedupes() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Pizza", kind: .topic))
        f.remember(FocusEntity(name: "Calzone", kind: .topic))
        XCTAssertEqual(f.primaryEntity?.name, "Calzone")

        // Re-mentioning an existing entity folds the old copy and promotes it.
        f.remember(FocusEntity(name: "pizza", kind: .topic))
        XCTAssertEqual(f.primaryEntity?.name, "pizza")
        XCTAssertEqual(f.entities.filter { $0.matchKey == "pizza" }.count, 1)
        XCTAssertEqual(f.entities.count, 2)
    }

    func testMostRecentByKind() {
        var f = ConversationFocus()
        f.remember(FocusEntity(name: "Fenway Park", kind: .place, lat: 42.3, lon: -71.0))
        f.remember(FocusEntity(name: "Baseball", kind: .topic))
        // Primary is the topic, but the most-recent place is still reachable.
        XCTAssertEqual(f.primaryEntity?.name, "Baseball")
        XCTAssertEqual(f.mostRecent(kind: .place)?.name, "Fenway Park")
    }

    func testSetLastListPromotesHead() {
        var f = ConversationFocus()
        f.setLastList([
            FocusEntity(name: "First Church", kind: .place),
            FocusEntity(name: "Second Church", kind: .place),
        ])
        XCTAssertEqual(f.lastList.count, 2)
        XCTAssertEqual(f.primaryEntity?.name, "First Church")
    }

    func testEntityStackIsBounded() {
        var f = ConversationFocus()
        for i in 0..<(ConversationFocus.maxEntities + 5) {
            f.remember(FocusEntity(name: "T\(i)", kind: .topic))
        }
        XCTAssertEqual(f.entities.count, ConversationFocus.maxEntities)
        XCTAssertEqual(f.primaryEntity?.name, "T\(ConversationFocus.maxEntities + 4)")
    }

    func testTrailJitterFilter() {
        var f = ConversationFocus()
        f.updateLocation(lat: 42.3601, lon: -71.0589)
        // A sub-epsilon nudge (~a few metres) must NOT add a trail point.
        f.updateLocation(lat: 42.36011, lon: -71.05891)
        XCTAssertEqual(f.trail.count, 1)
        // A real move (~150 m north) adds one.
        f.updateLocation(lat: 42.3615, lon: -71.0589)
        XCTAssertEqual(f.trail.count, 2)
        XCTAssertGreaterThan(
            f.movedMeters(since: ConversationFocus.Coord(lat: 42.3601, lon: -71.0589)),
            100
        )
    }
}
