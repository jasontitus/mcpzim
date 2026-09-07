import XCTest
@testable import MCPZimKit

final class PlaceSearchContinuationTests: XCTestCase {
    private let here = (lat: 37.44, lon: -122.15)
    private let remote = (lat: 37.77, lon: -122.42)

    private func localFocus() -> ConversationFocus {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.recordPlaceSearch(toolName: "near_places", args: ["zim": "street.zim"], result: [
            "origin": ["lat": here.lat, "lon": here.lon], "radius_km": 5.0,
            "results": [["name": "First cafe", "lat": 38.0, "lon": -121.0]],
        ])
        focus.setLastList([.init(name: "First cafe", kind: .place, lat: 38.0, lon: -121.0)])
        focus.beginUserTurn()
        return focus
    }

    func testMuseumKeepsCoffeeSearchAreaInsteadOfGPSOrFirstResult() throws {
        let focus = localFocus()
        for question in ["Museum?", "museums", "And museums?", "What about museums?",
                         "Any good museums?", "Where is a good museum?"] {
            let intent = try XCTUnwrap(IntentRouter.classify(question, currentLocation: remote, focus: focus))
            XCTAssertEqual(intent.toolName, "near_places", question)
            XCTAssertEqual(intent.args["lat"], .double(here.lat), question)
            XCTAssertEqual(intent.args["lon"], .double(here.lon), question)
            XCTAssertEqual(intent.args["radius_km"], .double(5), question)
            XCTAssertEqual(intent.args["kinds"], .array([.string("museum")]), question)
            XCTAssertEqual(intent.args["zim"], .string("street.zim"), question)
            XCTAssertNil(intent.args["place"], question)
        }
    }

    func testNamedCityAndExplicitNearMeOverrideEarlierArea() {
        let focus = localFocus()
        let named = IntentRouter.classify("Museums in San Francisco", currentLocation: here, focus: focus)
        XCTAssertEqual(named?.toolName, "near_named_place")
        XCTAssertEqual(named?.args["place"], .string("san francisco"))
        let local = IntentRouter.classify("Museums near me", currentLocation: remote, focus: focus)
        XCTAssertEqual(local?.args["lat"], .double(remote.lat))
        XCTAssertEqual(local?.args["lon"], .double(remote.lon))
    }

    func testRemoteSearchKeepsResolvedCenterNameRadiusAndEmptyResults() throws {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.recordPlaceSearch(toolName: "near_named_place", args: ["place": "SF"], result: [
            "resolved": ["name": "San Francisco", "lat": remote.lat, "lon": remote.lon],
            "radius_km": 2.0, "results": [],
        ])
        focus.beginUserTurn()
        let intent = try XCTUnwrap(IntentRouter.classify("Library?", currentLocation: here, focus: focus))
        XCTAssertEqual(intent.args["lat"], .double(remote.lat))
        XCTAssertEqual(intent.args["radius_km"], .double(2))
        XCTAssertEqual(intent.args["center_name"], .string("San Francisco"))
        let caption = IntentRouter.synthesizePlacesReply(toolName: intent.toolName,
            args: intent.anyArgs, fullResult: ["results": [], "radius_km": 2.0])
        XCTAssertTrue(caption.contains("San Francisco"))
        XCTAssertFalse(caption.contains("near you"))
    }

    func testResetAndInterveningTurnsDoNotReuseStaleSearchArea() {
        var focus = localFocus()
        focus.beginUserTurn()
        XCTAssertNil(focus.placeSearchForFollowup)
        XCTAssertNil(IntentRouter.classify("Museum?", currentLocation: here, focus: focus))
        focus = localFocus()
        focus.reset()
        XCTAssertNil(focus.lastPlaceSearch)
        XCTAssertNil(IntentRouter.classify("Museum?", currentLocation: here, focus: focus))
    }

    func testErrorsAndInvalidCentersCannotSeedAContinuation() {
        let invalid: [[String: Any]] = [
            ["error": "no match"],
            ["origin": ["lat": 190.0, "lon": 0.0], "radius_km": 5.0],
            ["origin": ["lat": Double.nan, "lon": 0.0], "radius_km": 5.0],
            ["origin": ["lat": 37.0, "lon": -122.0], "radius_km": Double.infinity],
            ["origin": ["lat": 37.0, "lon": -122.0], "radius_km": 0.0],
            ["results": [["name": "First result", "lat": 37.0, "lon": -122.0]], "radius_km": 5.0],
        ]
        for result in invalid {
            var focus = localFocus()
            focus.recordPlaceSearch(toolName: "near_places", args: [:], result: result)
            XCTAssertNil(focus.lastPlaceSearch)
        }
    }

    func testCategoryGrammarDoesNotStealNamedVenuesDefinitionsOrSelections() {
        for text in ["The museum", "What about the museum?", "What is a museum?", "Museum of Modern Art",
                     "Second museum", "Museums in San Francisco", "Museum history", "Museum near me"] {
            XCTAssertNil(IntentRouter.placeCategoryFollowup(text), text)
        }
        let focus = localFocus()
        XCTAssertEqual(IntentRouter.classify("What is a museum?", focus: focus)?.toolName, "article_overview")
        XCTAssertEqual(IntentRouter.classify("What is a museum?", focus: focus)?.args["title"], .string("museum"))
        XCTAssertEqual(IntentRouter.classify("Tell me about A Museum", focus: focus)?.args["title"], .string("a museum"))
    }

    func testCoffeeQuestionWrapperIsNotACategoryOrQualityClaim() {
        let intent = IntentRouter.classify("Where is a good coffee shop near me?", currentLocation: here)
        XCTAssertEqual(intent?.toolName, "near_places")
        XCTAssertEqual(intent?.args["kinds"], .array([.string("coffee shop")]))
        XCTAssertNil(IntentRouter.classify("Where is a good coffee shop near me?", currentLocation: nil))
    }
}
