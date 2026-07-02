// SPDX-License-Identifier: MIT
//
// Regression coverage for discuss-mode passage retrieval, built from the
// REAL failure on device (2026-07-01, LFM2.5 FT + full English Wikipedia):
// discussing "Putin", the question "What about his parents?" retrieved
// Pets | Sports | Cult of personality, and "How about his time in
// Germany?" missed the KGB/Dresden section. Root cause: ranking was pure
// HashingEmbedder n-gram cosine — hash noise on short strings — and
// `questionKeywords` let "his" through as a keyword.

import XCTest
@testable import MCPZimKit

final class DiscussRetrievalTests: XCTestCase {

    /// Section titles from the actual Putin article (as logged on
    /// device), with condensed but realistic bodies.
    private var putinSections: [ArticleSection] {
        func s(_ title: String, _ text: String) -> ArticleSection {
            ArticleSection(title: title, level: 2, text: text)
        }
        return [
            ArticleSection(title: "", level: 0, text:
                "Vladimir Vladimirovich Putin is a Russian politician and former "
                + "intelligence officer who has served as President of Russia since 2012."),
            s("Early life", """
                Putin was born on 7 October 1952 in Leningrad. His mother, Maria \
                Ivanovna Putina, was a factory worker, and his father, Vladimir \
                Spiridonovich Putin, was a conscript in the Soviet Navy. His \
                grandfather was a personal cook to Lenin and Stalin. Two brothers \
                died before his birth, one of diphtheria during the siege of \
                Leningrad. He studied law at Leningrad State University.
                """),
            s("Intelligence career", """
                Putin joined the KGB in 1975. From 1985 to 1990 he served in \
                Dresden, East Germany, using a cover identity as a translator. \
                After the fall of the Berlin Wall he returned from Germany to \
                Leningrad. His KGB career ended with the rank of lieutenant colonel.
                """),
            s("Political career", """
                In 1991 Putin joined the office of the Saint Petersburg mayor. He \
                rose through the Kremlin administration, became acting president \
                in 1999, and won the 2000 presidential election.
                """),
            s("After the 2022 invasion of Ukraine", """
                Putin launched a full-scale invasion of Ukraine in February 2022. \
                He had annexed Crimea in 2014, declaring in a speech that Crimea \
                has always been an inseparable part of Russia. Sanctions followed \
                the annexation of Crimea and the invasion.
                """),
            s("Personal life", """
                Putin married Lyudmila Shkrebneva in 1983; they have two daughters, \
                Maria and Katerina. The couple announced their divorce in 2013.
                """),
            s("Pets", """
                Putin has received several dogs from foreign leaders, including a \
                Bulgarian shepherd and an Akita. His labrador Koni attended meetings.
                """),
            s("Sports", """
                Putin practises judo and ice hockey. He holds a black belt and has \
                co-authored a book on judo.
                """),
            s("Cult of personality", """
                Russian media have cultivated an image of Putin as a strong \
                outdoorsman, publishing photographs of shirtless horseback riding.
                """),
            s("Assessments", """
                Scholars describe Putin's rule as authoritarian, marked by the \
                suppression of political opposition and independent media.
                """),
            s("Awards and honours", """
                Putin has received numerous state decorations from Russia and \
                foreign governments.
                """),
        ]
    }

    private func topTitles(_ question: String, k: Int = 3) -> [String] {
        ArticleHeuristics.rankSectionsMultiSource(
            question, sources: [(title: "Putin", sections: putinSections)], k: k
        ).map { $0.section.title.isEmpty ? "lead" : $0.section.title }
    }

    func testParentsQuestionRetrievesEarlyLife() {
        let top = topTitles("What about his parents?")
        XCTAssertEqual(top.first, "Early life", "got: \(top)")
        XCTAssertFalse(top.contains("Pets"), "Pets outranked family: \(top)")
        XCTAssertFalse(top.contains("Sports"), "got: \(top)")
    }

    func testGermanyQuestionRetrievesIntelligenceCareer() {
        let top = topTitles("How about his time in Germany?")
        XCTAssertEqual(top.first, "Intelligence career", "got: \(top)")
    }

    func testCrimeaQuestionRetrievesUkraineSection() {
        let top = topTitles("What has he said about crimea?")
        XCTAssertEqual(top.first, "After the 2022 invasion of Ukraine", "got: \(top)")
    }

    func testWifeQuestionRetrievesPersonalLife() {
        let top = topTitles("Was he ever married?")
        XCTAssertEqual(top.first, "Personal life", "got: \(top)")
    }

    func testDogQuestionStillFindsPets() {
        // Keyword scoring must not break on-topic quirky questions.
        let top = topTitles("Does he have any dogs?")
        XCTAssertEqual(top.first, "Pets", "got: \(top)")
    }

    func testKeywordlessQuestionStillReturnsSomething() {
        let top = topTitles("Tell me more")
        XCTAssertFalse(top.isEmpty)
    }

    func testQuestionKeywordsDropPronounsAndFiller() {
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("What about his parents?"),
            ["parents"])
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("How about his time in Germany?"),
            ["germany"])
        XCTAssertEqual(
            ArticleHeuristics.questionKeywords("What has he said about crimea?"),
            ["crimea"])
    }

    func testParentsCountsAsCoveredByEarlyLife() {
        // Synonym-aware coverage: "parents" is answered by mother/father
        // prose, so no corpus pull should fire.
        XCTAssertTrue(ArticleHeuristics.sectionsCoverQuestion(
            putinSections, "What about his parents?"))
    }
}
