import XCTest
@testable import MCPZimKit

final class TopicDiscoveryTests: XCTestCase {
    private func adapter() async -> MCPToolAdapter {
        var fixture = StubZimService.Fixture()
        fixture.inventory = InventoryResult(zims: [InventoryEntry(
            name: "encyclopedia.zim", kind: .wikipedia,
            metadata: ZimMetadata(name: "encyclopedia"), hasRoutingData: false)],
            capabilities: [])
        let names = ["Missing", "Astronomy", "Botany", "MissingToo", "Chemistry"]
        let html = "<p>" + names.map { "<a href=\"A/\($0)\">\($0)</a>" }
            .joined(separator: " ") + "</p>"
        fixture.mainPages = [ArticleResult(zim: "encyclopedia.zim", path: "A/Main_Page",
            title: "Main Page", mimetype: "text/html", text: html, bytes: html.utf8.count)]
        for name in ["Astronomy", "Botany", "Chemistry"] {
            fixture.articleHTML[StubZimService.keyArticleSections(path: "A/\(name)")] =
                "<p>\(name) is a branch of science with a long history of observation and study. "
                + "Researchers use evidence and experiments to explain the natural world.</p>"
        }
        return await MCPToolAdapter(service: StubZimService(fixture: fixture), hasStreetzim: false)
    }

    func testSparsePagesAdvanceByCandidateAndWrapOnlyAtEnd() async throws {
        let adapter = await adapter()
        let first = try await adapter.dispatch(tool: "discover_topics", args: ["limit": 2])
        let firstRows = try XCTUnwrap(first["topics"] as? [[String: Any]])
        XCTAssertEqual(firstRows.compactMap { $0["title"] as? String }, ["Astronomy", "Botany"])
        XCTAssertEqual(first["next_offset"] as? Int, 3)
        XCTAssertEqual(first["cycle_complete"] as? Bool, false)
        let second = try await adapter.dispatch(tool: "discover_topics", args: [
            "limit": 2, "offset": try XCTUnwrap(first["next_offset"] as? Int),
        ])
        let secondRows = try XCTUnwrap(second["topics"] as? [[String: Any]])
        XCTAssertEqual(secondRows.compactMap { $0["title"] as? String }, ["Chemistry"])
        XCTAssertEqual(second["next_offset"] as? Int, 0)
        XCTAssertEqual(second["cycle_complete"] as? Bool, true)
        let restarted = try await adapter.dispatch(tool: "discover_topics", args: [
            "limit": 2, "offset": Int.max,
        ])
        XCTAssertEqual((restarted["topics"] as? [[String: Any]])?
            .compactMap { $0["title"] as? String }, ["Astronomy", "Botany"])
    }

    func testCancelledDiscoveryDoesNotReturnASuccessfulPage() async throws {
        let adapter = await adapter()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await adapter.dispatch(tool: "discover_topics", args: [:])
        }
        do {
            _ = try await task.value
            XCTFail("Cancelled discovery should stop before opening candidates")
        } catch is CancellationError {
        } catch { XCTFail("Unexpected error: \(error)") }
    }

    func testPinnedMedicalArticleCannotBeReplacedByALargerWikipediaStubMatch() async throws {
        var fixture = StubZimService.Fixture()
        fixture.inventory = InventoryResult(zims: [
            InventoryEntry(name: "medicine.zim", kind: .mdwiki,
                metadata: ZimMetadata(name: "medicine"), hasRoutingData: false),
            InventoryEntry(name: "wikipedia.zim", kind: .wikipedia,
                metadata: ZimMetadata(name: "wikipedia"), hasRoutingData: false),
        ], capabilities: [])
        let short = ArticleSection(title: "", level: 0,
            text: "Diabetes is a group of metabolic disorders involving high blood sugar.")
        fixture.articleByTitle[StubZimService.keyArticleByTitle(title: "Diabetes", section: "lead")] =
            .init(zim: "medicine.zim", path: "A/Diabetes", title: "Diabetes", section: short)
        fixture.articleSections[StubZimService.keyArticleSections(path: "A/Diabetes")] =
            .init(zim: "medicine.zim", title: "Diabetes", sections: [short])
        fixture.articleSections[StubZimService.keyArticleSections(path: "A/Diabetes_other")] =
            .init(zim: "wikipedia.zim", title: "Diabetes", sections: [
                ArticleSection(title: "", level: 0, text: String(repeating: "A longer encyclopedia account. ", count: 100)),
            ])
        fixture.search[StubZimService.keySearch(query: "Diabetes")] = [
            SearchHitResult(zim: "wikipedia.zim", kind: .wikipedia,
                path: "A/Diabetes_other", title: "Diabetes", snippet: "A larger article."),
        ]
        let adapter = await MCPToolAdapter(service: StubZimService(fixture: fixture), hasStreetzim: false)
        let pinned = try await adapter.dispatch(tool: "article_overview", args: [
            "title": "Diabetes", "zim": "medicine.zim",
        ])
        XCTAssertEqual(pinned["zim"] as? String, "medicine.zim")
        let unpinned = try await adapter.dispatch(tool: "article_overview", args: ["title": "Diabetes"])
        XCTAssertEqual(unpinned["zim"] as? String, "wikipedia.zim",
            "An unqualified request should retain the existing stub-rescue behavior")
    }
}
