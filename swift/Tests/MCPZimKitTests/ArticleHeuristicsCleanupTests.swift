// SPDX-License-Identifier: MIT
//
// Wikipedia article-lead cleanup: strip title repetition and
// disambiguation pages before we render a lead as a place's
// preview. Matches the four on-device failures in the 2026-04-22
// screenshot:
//
//   * "Oak Grove" → `Oak_Grove` disambiguation article opening
//     "Oak Grove may refer to:" — rejected as unusable.
//   * "Palo Alto Junior Museum and Zoo" → article lead repeats the
//     title three times before the actual first sentence — the
//     duplicate paragraphs are stripped.
//   * Wine Cellar → topical-concept article `Wine_cellar` —
//     rejected via the tightened name-match in the fallback
//     (covered by `MCPToolAdapter.fetchWikiExcerptsByNameSearch`
//     tests, not here).

import Foundation
import XCTest

@testable import MCPZimKit

final class ArticleHeuristicsCleanupTests: XCTestCase {

    // MARK: - isDisambiguationArticle

    func testDisambigByTitleSuffix() {
        XCTAssertTrue(ArticleHeuristics.isDisambiguationArticle(
            title: "Oak Grove (disambiguation)",
            leadText: "Oak Grove may refer to: …"
        ))
    }

    func testDisambigByLeadOpener() {
        // Real capture from the streetzim + enwiki ZIM:
        XCTAssertTrue(ArticleHeuristics.isDisambiguationArticle(
            title: "Oak Grove",
            leadText: "Oak Grove\n\nOak Grove may refer to:"
        ))
    }

    func testDisambigNotRaisedForNormalArticle() {
        // "also refer to" in running prose should NOT trigger the
        // disambig guard — we want the word-boundary "may refer to"
        // match only.
        XCTAssertFalse(ArticleHeuristics.isDisambiguationArticle(
            title: "Palo Alto",
            leadText: "Palo Alto is a charter city in the northwestern "
                + "corner of Santa Clara County. Residents also refer "
                + "to the downtown as University Avenue."
        ))
    }

    func testDisambigCaseInsensitive() {
        XCTAssertTrue(ArticleHeuristics.isDisambiguationArticle(
            title: "Foo (Disambiguation)",   // mixed-case suffix
            leadText: "Foo may refer to: …"
        ))
    }

    // MARK: - stripLeadingTitleRepetition

    func testStripsThreefoldLeadingTitle() {
        // Verbatim-style capture of what PlacesListView was
        // rendering — the title repeated three times before the
        // first real sentence.
        let input = """
        Palo Alto Junior Museum and Zoo

        Palo Alto Junior Museum and Zoo

        Palo Alto Junior Museum and Zoo

        Palo Alto Junior Museum and Zoo is located in Palo Alto, California.
        """
        let out = ArticleHeuristics.stripLeadingTitleRepetition(
            input, title: "Palo Alto Junior Museum and Zoo"
        )
        XCTAssertEqual(
            out,
            "Palo Alto Junior Museum and Zoo is located in Palo Alto, California."
        )
    }

    func testStripsWhenLeadingLineHasTrailingPunct() {
        let input = """
        Aspirin.

        Aspirin is a medication used to reduce pain …
        """
        let out = ArticleHeuristics.stripLeadingTitleRepetition(
            input, title: "Aspirin"
        )
        XCTAssertTrue(out.hasPrefix("Aspirin is a medication"),
                      "expected prose start, got: \(out)")
    }

    func testLeavesProseAloneWhenNoRepetition() {
        let input = "A charter city in the northwestern corner of Santa Clara County."
        let out = ArticleHeuristics.stripLeadingTitleRepetition(
            input, title: "Palo Alto"
        )
        XCTAssertEqual(out, input)
    }

    func testEmptyTitleIsNoop() {
        let text = "Some lead text."
        XCTAssertEqual(
            ArticleHeuristics.stripLeadingTitleRepetition(text, title: ""),
            text
        )
    }

    func testEmptyTextIsNoop() {
        XCTAssertEqual(
            ArticleHeuristics.stripLeadingTitleRepetition("", title: "X"),
            ""
        )
    }

    func testDoesNotStripMidBodyTitleMentions() {
        // The title appearing inside the first real paragraph must
        // survive — we only strip leading whole-paragraph repeats.
        let input = """
        Palo Alto

        Palo Alto is a charter city. Historic Palo Alto structures still stand.
        """
        let out = ArticleHeuristics.stripLeadingTitleRepetition(
            input, title: "Palo Alto"
        )
        XCTAssertTrue(out.contains("Historic Palo Alto structures"),
                      "mid-body mention must survive: \(out)")
        XCTAssertFalse(out.hasPrefix("Palo Alto\n\nPalo Alto"),
                       "leading repeat should be stripped: \(out)")
    }
}
