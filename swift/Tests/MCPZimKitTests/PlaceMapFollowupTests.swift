import XCTest
@testable import MCPZimKit

final class PlaceMapFollowupTests: XCTestCase {
    func testCoordinateSelectionProducesOnePinWithoutGeocoding() async throws {
        var focus = ConversationFocus()
        focus.setLastList([FocusEntity(name: "The Pit Bar", kind: .place,
                                      zim: "california.zim", lat: 36.85, lon: -121.64)])
        let intent = try XCTUnwrap(IntentRouter.classify("Show me The Pit Bar on the map", focus: focus))
        let result = try XCTUnwrap(MCPToolAdapter.savedMapResult(intent: intent, focus: focus))
        XCTAssertNil(MCPToolAdapter.savedMapResult(intent: intent, focus: ConversationFocus()))
        let forged = DirectIntent(toolName: "locate", args: ["place": .string("The Pit Bar"),
            "lat": .double(0), "lon": .double(0)])
        XCTAssertNil(MCPToolAdapter.savedMapResult(intent: forged, focus: focus))
        // The model-facing tool cannot use supplied coordinates to invent a pin.
        let adapter = MCPToolAdapter(service: DefaultZimService(readers: []), hasStreetzim: true)
        do {
            _ = try await adapter.dispatch(tool: "locate", args: forged.anyArgs)
            XCTFail("Ungrounded coordinates must not produce a map result")
        } catch { }
        let rows = try XCTUnwrap(result["results"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?["name"] as? String, "The Pit Bar")
        XCTAssertEqual(rows.first?["lat"] as? Double, 36.85)
        XCTAssertEqual(result["zim"] as? String, "california.zim")
        let data = try JSONSerialization.data(withJSONObject: result)
        let payload = parsePlacesJSON(rawResult: String(decoding: data, as: UTF8.self))
        XCTAssertEqual(payload.places.count, 1)
        XCTAssertEqual(payload.places.first?.lat, 36.85)
    }

    func testAmbiguousPlaceRequiresSelection() {
        var focus = ConversationFocus()
        let first = FocusEntity(name: "The Pit Bar", kind: .place, lat: 36.85, lon: -121.64)
        let second = FocusEntity(name: "The Hideaway Bar", kind: .place, lat: 36.86, lon: -121.65)
        focus.setLastList([first, second])
        XCTAssertNil(IntentRouter.mapSelectionIntent("Show me the bar on the map", focus: focus))
        // Same collapse performed by the app after a clarification pick.
        focus.setLastList([second])
        let intent = IntentRouter.mapSelectionIntent("Show me the bar on the map", focus: focus)
        XCTAssertEqual(intent?.args["lat"], .double(36.86))
        XCTAssertNil(IntentRouter.mapSelectionIntent("What did George Bernard Shaw write?", focus: focus))
        focus.setLastList([first, FocusEntity(name: first.name, kind: .place, lat: 40, lon: -120)])
        XCTAssertNil(IntentRouter.mapSelectionIntent("Show me The Pit Bar on the map", focus: focus))
    }

    func testListedPlaceMapSelectionRetainsCoordinates() {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        let initial = IntentRouter.classify("Where is a bar near here?",
            currentLocation: (36.840586643248692, -121.63241233439634), focus: focus)!
        XCTAssertEqual(initial.toolName, "near_places")
        focus.recordPlaceSearch(toolName: initial.toolName, args: initial.anyArgs, result: [
            "origin": ["lat": 36.840586643248692, "lon": -121.63241233439634],
            "radius_km": 5.0, "total_in_radius": 0
        ])
        focus.beginUserTurn()
        let wider = IntentRouter.classify("Search wider",
            currentLocation: (36.84411, -121.63411), focus: focus)
        XCTAssertEqual(wider?.args["radius_km"], .double(10))
        XCTAssertEqual(wider?.args["lat"], initial.args["lat"])
        let pit = FocusEntity(name: "The Pit Bar", kind: .place, zim: "california.zim", lat: 36.85, lon: -121.64)
        focus.setLastList([
            FocusEntity(name: "101 Wine Press", kind: .place, lat: 36.86, lon: -121.65),
            pit,
            FocusEntity(name: "The Hideaway", kind: .place, lat: 36.87, lon: -121.66),
        ])
        for query in ["Show me The Pit Bar on the map", "Show me the second one on the map"] {
            let intent = IntentRouter.classify(query, focus: focus)
            XCTAssertEqual(intent?.toolName, "locate", query)
            XCTAssertEqual(intent?.args["place"], .string(pit.name), query)
            XCTAssertEqual(intent?.args["lat"], .double(36.85), query)
            XCTAssertEqual(intent?.args["lon"], .double(-121.64), query)
            XCTAssertEqual(intent?.args["zim"], .string("california.zim"), query)
            XCTAssertNil(intent?.articleFallbackTitle)
        }
        // The app collapses a self-resolved ambiguity to this same entity.
        focus.setLastList([pit])
        XCTAssertEqual(IntentRouter.classify("Show me The Pit Bar on the map", focus: focus)?.args["lat"], .double(36.85))
        XCTAssertEqual(IntentRouter.classify("Show it on the map", focus: focus)?.args["lat"], .double(36.85))
        XCTAssertEqual(IntentRouter.classify("Who was George Bernard Shaw?", focus: focus)?.toolName, "article_overview")
        XCTAssertEqual(IntentRouter.classify("Show me Salinas on the map", focus: focus)?.args["place"], .string("Salinas"))
        XCTAssertNil(IntentRouter.classify("Show me Salinas on the map", focus: focus)?.args["lat"])
        XCTAssertNil(IntentRouter.mapSelectionIntent("Show me The Pit Bar in Salinas on the map", focus: focus)?.args["lat"])
    }
}
