// SPDX-License-Identifier: MIT
//
// Tests for the deterministic drift engine: real wikilink extraction, turning
// tool results into vetted threads, ranking/deduping against discussed
// entities, and the no-LLM offer caption.

import Foundation
import XCTest

@testable import MCPZimKit

final class ConversationThreadsTests: XCTestCase {

    // MARK: - WikiLinks

    func testWikiLinksKeepsArticlesDropsCruft() {
        let html = """
        <p>The <a href="Stanford_White">Stanford White</a> firm also built the
        <a href="../A/Cantor_Arts_Center">Cantor Arts Center</a>. See
        <a href="File:Church.jpg">a photo</a>, the
        <a href="Category:Churches">category</a>, an
        <a href="https://example.com">external site</a>, and a
        <a href="#refs">footnote</a>.<a href="Stanford_White">again</a></p>
        """
        let links = WikiLinks.parse(html: html)
        let titles = links.map(\.title)
        XCTAssertEqual(titles, ["Stanford White", "Cantor Arts Center"])
        // Relative prefixes stripped from the path.
        XCTAssertEqual(links[1].path, "A/Cantor_Arts_Center")
    }

    func testWikiLinksSkipsInfoboxAndBoilerplate() {
        // Infobox / reference links (in <table>, plus identifier fields) must
        // NOT be offered as drift topics — only prose <p> links should.
        // Mirrors the medicine-article failure (2026-05-30): offers were
        // "MedlinePlus, Drugs.com, Trade names" instead of real topics.
        let html = """
        <table class="infobox"><tr><td><a href="Trade_names">Trade names</a></td></tr>
        <tr><td><a href="MedlinePlus">MedlinePlus</a> <a href="Drugs.com">Drugs.com</a></td></tr>
        <tr><td><a href="British_Approved_Name">BAN</a></td></tr></table>
        <p>Aspirin is used to reduce <a href="Fever">fever</a> and treat
        <a href="Kawasaki_disease">Kawasaki disease</a> and cut the risk of
        <a href="Heart_attack">heart attack</a>.</p>
        """
        let titles = WikiLinks.parse(html: html).map(\.title)
        XCTAssertEqual(titles, ["fever", "Kawasaki disease", "heart attack"])
        XCTAssertFalse(titles.contains("Trade names"))
        XCTAssertFalse(titles.contains("MedlinePlus"))
        XCTAssertFalse(titles.contains("Drugs.com"))
    }

    func testWikiLinksDecodesEntitiesAndStripsInnerTags() {
        let html = #"<a href="AT&amp;T_Building"><i>AT&amp;T</i> Building</a>"#
        let links = WikiLinks.parse(html: html)
        XCTAssertEqual(links.first?.title, "AT&T Building")
    }

    func testWikiLinksRespectsCap() {
        let html = (1...20).map {
            "<a href=\"Article_\($0)\">Article \($0)</a>"
        }.joined(separator: " ")
        XCTAssertEqual(WikiLinks.parse(html: html, max: 5).count, 5)
    }

    // MARK: - extract

    func testPlacesResultBecomesNearbyThreads() {
        let result: [String: Any] = [
            "results": [
                ["wiki_title": "Fenway Park", "lat": 42.3467, "lon": -71.0972,
                 "wiki_path": "A/Fenway_Park", "distance_m": 420],
                ["name": "Symphony Hall", "lat": 42.3426, "lon": -71.0853],
            ]
        ]
        let threads = ConversationThreads.extract(
            toolName: "near_places", result: result)
        XCTAssertEqual(threads.count, 2)
        XCTAssertEqual(threads[0].label, "Fenway Park")
        XCTAssertEqual(threads[0].source, .nearbyPlace)
        XCTAssertEqual(threads[0].note, "420 m away")
        XCTAssertEqual(threads[0].zimPath, "A/Fenway_Park")
    }

    func testArticleResultYieldsLinksThenSections() {
        let result: [String: Any] = [
            "html": "<a href=\"Stanford_White\">Stanford White</a>",
            "sections": [
                ["title": "", "text": "lead"],            // lead skipped
                ["title": "History", "text": "..."],
                ["title": "References", "text": "..."],     // boilerplate skipped
            ],
        ]
        let threads = ConversationThreads.extract(
            toolName: "article_overview", result: result)
        let labels = threads.map(\.label)
        XCTAssertTrue(labels.contains("Stanford White"))
        XCTAssertTrue(labels.contains("History"))
        XCTAssertFalse(labels.contains("References"))
        XCTAssertFalse(labels.contains(""))
    }

    /// article_overview now attaches a pre-parsed `related:[{title,path}]`
    /// array (from WikiLinks.parse over the raw HTML); the extractor should
    /// prefer it over re-parsing `html`, and carry `path` into `zimPath` so a
    /// follow-up can re-fetch the exact article.
    func testArticleRelatedArrayBecomesWikilinkThreads() {
        let result: [String: Any] = [
            "related": [
                ["title": "Stanford White", "path": "A/Stanford_White"],
                ["title": "1906 San Francisco earthquake", "path": "A/1906_quake"],
            ],
            "sections": [["title": "History", "text": "..."]],
        ]
        let threads = ConversationThreads.extract(
            toolName: "article_overview", result: result)
        let white = threads.first { $0.label == "Stanford White" }
        XCTAssertNotNil(white)
        XCTAssertEqual(white?.source, .wikilink)
        XCTAssertEqual(white?.zimPath, "A/Stanford_White")
        XCTAssertTrue(threads.contains { $0.label == "1906 San Francisco earthquake" })
        // The section is still offered as a deeper thread.
        XCTAssertTrue(threads.contains { $0.label == "History" && $0.source == .section })
    }

    // MARK: - rank

    func testRankDropsDiscussedAndOrdersLateralFirst() {
        var focus = ConversationFocus()
        focus.remember(FocusEntity(name: "History", kind: .topic))  // already discussed
        let threads = [
            DiscoveryThread(label: "History", kind: .topic, source: .section),
            DiscoveryThread(label: "Architecture", kind: .topic, source: .section),
            DiscoveryThread(label: "Stanford White", kind: .topic, source: .wikilink),
        ]
        let ranked = ConversationThreads.rank(threads, focus: focus, max: 4)
        // "History" dropped (discussed); wikilink ranks before section.
        XCTAssertEqual(ranked.map(\.label), ["Stanford White", "Architecture"])
    }

    func testRankCaps() {
        let threads = (1...10).map {
            DiscoveryThread(label: "L\($0)", kind: .place, source: .nearbyPlace)
        }
        XCTAssertEqual(ConversationThreads.rank(threads, focus: ConversationFocus(), max: 3).count, 3)
    }

    // MARK: - offer

    func testOfferPhrasing() {
        XCTAssertEqual(
            ConversationThreads.offer([
                DiscoveryThread(label: "Stanford White", kind: .topic, source: .wikilink),
            ]),
            "Want to hear about Stanford White?"
        )
        XCTAssertEqual(
            ConversationThreads.offer([
                DiscoveryThread(label: "A", kind: .topic, source: .wikilink),
                DiscoveryThread(label: "B", kind: .topic, source: .wikilink),
                DiscoveryThread(label: "C", kind: .topic, source: .wikilink),
            ]),
            "Want to hear about A, B, or C?"
        )
        XCTAssertNil(ConversationThreads.offer([]))
    }

    func testOfferIncludesDistanceNote() {
        let line = ConversationThreads.offer([
            DiscoveryThread(label: "Fenway Park", kind: .place,
                            source: .nearbyPlace, note: "420 m away"),
        ])
        XCTAssertEqual(line, "Want to hear about Fenway Park (420 m away)?")
    }
}
