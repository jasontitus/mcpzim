// SPDX-License-Identifier: MIT
//
// Section retrieval for "let's discuss X" — grounded single-article RAG.
// Uses the deterministic HashingEmbedder so these run with zero model
// assets; the on-device path can swap in NLContextualEmbedding for
// semantic matches.

import XCTest
@testable import MCPZimKit

final class DiscussionRetrievalTests: XCTestCase {

    private let sections = [
        ArticleSection(title: "", level: 0,
                       text: "Foo is a country in northern Europe with a long history."),
        ArticleSection(title: "History", level: 2,
                       text: "Foo was founded by settlers and later joined a union."),
        ArticleSection(title: "Economy", level: 2,
                       text: "The economy of Foo relies on exports, trade, and manufacturing of goods."),
        ArticleSection(title: "Geography", level: 2,
                       text: "Mountains, rivers, and forests cover much of the land of Foo."),
    ]

    func testRankPutsLeadFirstThenTopicalSection() {
        let out = ArticleHeuristics.rankSectionsForQuestion(
            "what is the economy like", sections: sections, k: 2)
        XCTAssertEqual(out.first?.title, "", "lead is always the anchor")
        XCTAssertEqual(out.dropFirst().first?.title, "Economy",
                       "an economy question retrieves the Economy section")
    }

    func testRankGeographyQuestion() {
        let out = ArticleHeuristics.rankSectionsForQuestion(
            "tell me about the mountains and rivers", sections: sections, k: 2)
        XCTAssertEqual(out.dropFirst().first?.title, "Geography")
    }

    func testRankEmptyAndBounds() {
        XCTAssertTrue(
            ArticleHeuristics.rankSectionsForQuestion("x", sections: [], k: 3).isEmpty)
        // k = 1 → just the lead anchor.
        let one = ArticleHeuristics.rankSectionsForQuestion(
            "economy", sections: sections, k: 1)
        XCTAssertEqual(one.count, 1)
        XCTAssertEqual(one.first?.title, "")
    }

    // MARK: - Multi-article: topic core, coverage, corpus query

    func testTopicCoreStripsSubArticleWrappers() {
        XCTAssertEqual(
            ArticleHeuristics.topicCore("History of Lithuania (1219-1295)"), "Lithuania")
        XCTAssertEqual(ArticleHeuristics.topicCore("Economy of Japan"), "Japan")
        XCTAssertEqual(ArticleHeuristics.topicCore("List of the United States"), "United States")
        XCTAssertEqual(ArticleHeuristics.topicCore("Solar panel"), "Solar panel")
    }

    func testQuestionKeywordsDropInterrogatives() {
        XCTAssertEqual(ArticleHeuristics.questionKeywords("What is the population?"),
                       ["population"])
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("How have they gotten along with Poland?"),
            ["poland"])
        // All-stopword follow-up → empty (host treats as "covered").
        XCTAssertTrue(ArticleHeuristics.questionKeywords("tell me more").isEmpty)
    }

    func testCoverageGatesCorpusFallback() {
        // The anchor covers a question whose keyword appears; not one whose
        // keyword is absent (which is what triggers the corpus pull).
        XCTAssertTrue(
            ArticleHeuristics.sectionsCoverQuestion(sections, "what is the economy"))
        XCTAssertFalse(
            ArticleHeuristics.sectionsCoverQuestion(sections, "what is the population"))
        // No content keywords → covered (stay on anchor).
        XCTAssertTrue(ArticleHeuristics.sectionsCoverQuestion(sections, "tell me more"))
    }

    func testMultiSourceRanksAcrossArticles() {
        let solar = [
            ArticleSection(title: "", level: 0, text: "A solar panel converts sunlight to electricity."),
            ArticleSection(title: "Efficiency", level: 2, text: "Panel efficiency depends on the cells used."),
        ]
        let cell = [
            ArticleSection(title: "", level: 0, text: "A solar cell is a photovoltaic device."),
            ArticleSection(title: "Perovskite solar cells", level: 2,
                           text: "Perovskite cells are an emerging high-efficiency thin-film technology."),
        ]
        // Lexical match (HashingEmbedder is exact-token — semantic/morph
        // matching is the on-device NLContextualEmbedding upgrade): the
        // question shares "perovskite"/"thin-film" with the right section.
        let out = ArticleHeuristics.rankSectionsMultiSource(
            "tell me about thin-film perovskite cells",
            sources: [("Solar panel", solar), ("Solar cell", cell)], k: 1)
        XCTAssertEqual(out.first?.article, "Solar cell")
        XCTAssertEqual(out.first?.section.title, "Perovskite solar cells")
    }
}
