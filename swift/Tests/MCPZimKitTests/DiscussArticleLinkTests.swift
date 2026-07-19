// SPDX-License-Identifier: MIT

import XCTest

@testable import MCPZimKit

final class DiscussArticleLinkTests: XCTestCase {

    func testDiscussArticleReturnsRealOutboundWikipediaLinks() async throws {
        var fixture = StubZimService.Fixture()
        let path = "A/Russia"
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Russia", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: path, title: "Russia",
            section: ArticleSection(
                title: "", level: 0,
                text: "Russia is a country spanning Eastern Europe and North Asia."))
        fixture.articleSections[
            StubZimService.keyArticleSections(path: path)
        ] = .init(
            zim: "wikipedia.zim", title: "Russia",
            sections: [
                ArticleSection(
                    title: "", level: 0,
                    text: "Russia is a country spanning Eastern Europe and North Asia."),
                ArticleSection(
                    title: "History", level: 2,
                    text: "The Russian Civil War followed the revolution."),
            ])
        fixture.articleHTML[
            StubZimService.keyArticleSections(path: path)
        ] = """
        <p>After the revolution, the <a href="A/Russian_Civil_War">civil war</a>
        involved the Red and White movements.</p>
        """

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(
            tool: "discuss_article", args: ["title": "Russia"])
        let links = try XCTUnwrap(result["linked_articles"] as? [[String: Any]])
        let allowed = ArticleHeuristics.linkedArticleTitleKeys(links)

        XCTAssertTrue(allowed.contains("civil war"))
        XCTAssertTrue(allowed.contains("russian civil war"))
        XCTAssertTrue(ArticleHeuristics.isDirectlyLinkedArticle(
            "Russian Civil War", allowedTitleKeys: allowed))
    }

    func testSimilarButUnlinkedArticleCannotEscapePreparedTopic() {
        let links: [[String: Any]] = [
            ["title": "civil war", "path": "A/Russian_Civil_War"],
        ]
        let allowed = ArticleHeuristics.linkedArticleTitleKeys(links)

        XCTAssertFalse(ArticleHeuristics.isDirectlyLinkedArticle(
            "Civil Wars in Russia", allowedTitleKeys: allowed))
        XCTAssertFalse(ArticleHeuristics.isDirectlyLinkedArticle(
            "Russia (company)", allowedTitleKeys: allowed))
    }
}
