import Foundation
import XCTest
@testable import MCPZimKit

final class CarmelValleyRegressionTests: XCTestCase {
    func testWiderSearchPreservesCenterCategoryAndArchive() {
        var focus = ConversationFocus()
        focus.recordPlaceSearch(toolName: "near_places",
            args: ["kinds": ["coffee shop"], "zim": "california.zim"],
            result: ["origin": ["lat": 36.4326, "lon": -121.6625],
                     "radius_km": 5, "total_in_radius": 0])
        let intent = IntentRouter.classify("search wider", focus: focus)
        XCTAssertEqual(intent?.toolName, "near_places")
        XCTAssertEqual(intent?.args["radius_km"], .double(10))
        XCTAssertEqual(intent?.args["lat"], .double(36.4326))
        XCTAssertEqual(intent?.args["lon"], .double(-121.6625))
        XCTAssertEqual(intent?.args["kinds"], .array([.string("coffee shop")]))
        XCTAssertEqual(intent?.args["zim"], .string("california.zim"))
        XCTAssertEqual(focus.lastPlaceSearch?.radiusKm, 5)
    }

    func testWiderSearchHasHardRadiusCeiling() {
        var focus = ConversationFocus()
        focus.recordPlaceSearch(toolName: "near_places", args: ["kinds": ["museum"]],
            result: ["origin": ["lat": 36.4, "lon": -121.6], "radius_km": 75])
        XCTAssertEqual(IntentRouter.classify("search wider", focus: focus)?.args["radius_km"], .double(100))
        XCTAssertNotEqual(IntentRouter.classify("search wider")?.toolName, "near_places")
    }

    func testWiderSearchCannotReuseStaleOrFailedSearch() {
        var focus = ConversationFocus()
        focus.recordPlaceSearch(toolName: "near_places", args: ["kinds": ["cafe"]],
            result: ["origin": ["lat": 36.4, "lon": -121.6], "radius_km": 5])
        focus.beginUserTurn()
        focus.beginUserTurn()
        XCTAssertNotEqual(IntentRouter.classify("search wider", focus: focus)?.toolName, "near_places")
        focus.recordPlaceSearch(toolName: "near_places", args: [:], result: ["error": "unavailable"])
        XCTAssertNotEqual(IntentRouter.classify("search wider", focus: focus)?.toolName, "near_places")
    }

    func testEmptyCaptionOffersExplicitWideningWithoutPromisingMap() {
        let text = IntentRouter.synthesizePlacesReply(toolName: "near_places",
            args: ["kinds": ["coffee shop"]],
            fullResult: ["total_in_radius": 0, "radius_km": 5])
        XCTAssertTrue(text.contains("search wider"))
        XCTAssertTrue(text.contains("10 km"))
        XCTAssertFalse(text.contains("map below"))
    }

    func testDisambiguationDoesNotChooseFirstLongWineArticle() async throws {
        var fixture = StubZimService.Fixture()
        let lead = ArticleSection(title: "", level: 0, text: "Carmel Valley may refer to several places.")
        fixture.articleByTitle[StubZimService.keyArticleByTitle(title: "Carmel Valley", section: "lead")] =
            .init(zim: "wikipedia.zim", path: "A/Carmel_Valley", title: "Carmel Valley", section: lead)
        fixture.articleSections[StubZimService.keyArticleSections(path: "A/Carmel_Valley")] =
            .init(zim: "wikipedia.zim", title: "Carmel Valley", sections: [lead])
        fixture.articleHTML[StubZimService.keyArticleSections(path: "A/Carmel_Valley")] = "<ul><li><a href=\"Carmel_Valley_AVA\">Carmel Valley AVA</a></li><li><a href=\"Carmel_Valley_Village\">Carmel Valley Village</a></li><li><a href=\"California\">California</a></li></ul>"
        for title in ["Carmel Valley AVA", "Carmel Valley Village", "California"] {
            let path = title.replacingOccurrences(of: " ", with: "_")
            fixture.articleSections[StubZimService.keyArticleSections(path: path)] =
                .init(zim: "wikipedia.zim", title: title, sections: [
                    ArticleSection(title: "", level: 0, text: String(repeating: "Real archive text. ", count: 150))])
        }
        let adapter = MCPToolAdapter(service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_overview", args: ["title": "Carmel Valley"])
        XCTAssertEqual(result["ambiguous"] as? Bool, true)
        XCTAssertNil(result["sections"])
        XCTAssertEqual(result["suggestions"] as? [String], ["Carmel Valley AVA", "Carmel Valley Village"])
        let caption = IntentRouter.synthesizeArticleMissReply(args: [:], fullResult: result)
        XCTAssertTrue(caption.contains("Which did you mean"))
        XCTAssertFalse(caption.contains("couldn't find"))
    }
}
