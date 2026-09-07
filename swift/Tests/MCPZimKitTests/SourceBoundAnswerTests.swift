import XCTest
@testable import MCPZimKit

final class SourceBoundAnswerTests: XCTestCase {
    func testReportedConversationalQuestionsUseTheirSectionContext() {
        // Synthetic prose deliberately omits the question's wording, as the
        // real phone session did. Other sections have distracting word hits.
        let examples: [(String, String, String)] = [
            ("What was Mira's early life like?", "Early life and education",
             "She was born in a small town. Her parents moved to the coast when she was five."),
            ("How did Mira's career develop?", "Political career",
             "She joined the town council in 1990. In 1994 she became mayor."),
            ("What about Mira's family?", "Family",
             "She married Alex in 1983. They have two daughters."),
            ("How has Mira dealt with the West and NATO?", "United States, Western Europe, and NATO",
             "She met the visiting delegation in 2001. The meeting ended without an agreement."),
        ]
        for (question, heading, text) in examples {
            let passages: [SourceBoundAnswer.Passage] = [
                .init(article: "Mira", text: "Mira is a politician."),
                .init(article: "Mira", section: heading, text: text),
                .init(article: "Mira", section: "Wealth", text: "A newspaper reported a rumor about her family."),
            ]
            let reply = SourceBoundAnswer.answer(question: question, topic: "Mira", passages: passages,
                maxSentences: 2)
            XCTAssertEqual(reply.text, SourceBoundAnswer.sentences(text).joined(separator: "\n\n"), question)
            XCTAssertTrue(reply.excerpts.allSatisfy { $0.passageIndex == 1 }, question)
        }
    }

    func testHeadingCannotCoverAnUnsupportedExtraFactOrQualifier() {
        let passages: [SourceBoundAnswer.Passage] = [
            .init(article: "Mira", section: "Early life and education",
                  text: "Her parents moved to the coast. A secret dossier was later released."),
            .init(article: "Mira", section: "Family", text: "She married Alex in 1983."),
        ]
        for question in ["What was Mira's early life secret password?",
                         "What about Mira's family lunar expedition?",
                         "What about Mira's family in 1921?"] {
            XCTAssertFalse(SourceBoundAnswer.answer(question: question, topic: "Mira",
                passages: passages).hasEvidence, question)
        }
    }

    func testRetrievalAliasesCannotSubstituteAnOrganizationOrRelative() {
        XCTAssertFalse(SourceBoundAnswer.answer(question: "What about Mira and NATO?", topic: "Mira",
            passages: [.init(article: "Mira", section: "Foreign policy",
                text: "She discussed foreign relations with the delegation.")]).hasEvidence)
        XCTAssertFalse(SourceBoundAnswer.answer(question: "Who was Mira's mother?", topic: "Mira",
            passages: [.init(article: "Mira", section: "Family",
                text: "Her father was Alex, who ran the family business.")]).hasEvidence)
    }

    func testMotherIdentityKeepsBothParentsNamesAndRejectsIncidentalMentions() {
        let identity = "His parents, secular Ashkenazi Jews, were Hermann Einstein, a salesman and engineer, and Pauline Koch."
        let distractor = "Elsa was his first cousin on his mother's side. He spent his time in the care of his mother."
        let reply = SourceBoundAnswer.answer(question: "Who was his mother?", topic: "Einstein",
            passages: [
                .init(article: "Einstein", section: "Early life", text: identity),
                .init(article: "Einstein", section: "Relationships", text: distractor),
            ])
        XCTAssertEqual(reply.text, identity)
        XCTAssertEqual(reply.excerpts.first?.passageIndex, 0)
        XCTAssertFalse(SourceBoundAnswer.answer(question: "Who was his mother?", topic: "Einstein",
            passages: [.init(article: "Einstein", section: "Relationships", text: distractor)]).hasEvidence)
    }

    func testParentWordInArticleTitleDoesNotTurnAnOverviewIntoAParentQuestion() {
        let text = "She lived near the coast. She wrote several books."
        XCTAssertEqual(SourceBoundAnswer.answer(question: "Who was Mother Example?", topic: "Mother Example",
            passages: [.init(article: "Mother Example", text: text)]).excerpts.count, 2)
    }

    func testSuggestedQuestionsRoundTripToSourceEvenWithDifferentWording() throws {
        let sections: [ArticleSection] = [
            .init(title: "Early life", level: 2, text: "She was born near the coast."),
            .init(title: "Education", level: 2, text: "She studied chemistry in the capital."),
            .init(title: "Political career", level: 2, text: "In 1994 she became mayor."),
            .init(title: "Family", level: 2, text: "She married Alex in 1983."),
            .init(title: "Foreign policy", level: 2, text: "She met a visiting delegation."),
        ]
        let offers = ConversationThreads.contextualQuestions(topic: "Mira", sections: sections,
            after: "Tell me about Mira", max: 10)
        XCTAssertEqual(offers.count, 5)
        for offer in offers {
            let selected = try XCTUnwrap(ConversationSuggestionSelection.selection(
                for: ConversationSuggestionSelection.action(offer), suggestions: offers))
            let scoped = ConversationSuggestionSelection.sections(for: selected,
                articleTitle: "Mira", zim: nil, sections: sections)
            XCTAssertEqual(scoped.count, 1)
            let passages = scoped.map { SourceBoundAnswer.Passage(article: "Mira",
                section: $0.title, text: $0.text) }
            let reply = SourceBoundAnswer.answer(question: selected.prompt ?? "", topic: "Mira",
                passages: passages, sectionOverview: true)
            XCTAssertTrue(reply.hasEvidence, offer.label)
            XCTAssertEqual(reply.text, scoped.first?.text)
            XCTAssertTrue(reply.excerpts.allSatisfy {
                SourceBoundAnswer.sentences(passages[$0.passageIndex].text).contains($0.text)
            })
        }
    }

    func testDocumentTitlesDoNotBecomeRepeatedProse() {
        let sentence = "Albert Einstein was a theoretical physicist."
        let html = "<html><head><title>Albert Einstein</title></head><body><h1>Albert Einstein</h1><p>\(sentence)</p></body></html>"
        let passages = SourceBoundAnswer.passages(toolName: "get_article", result: [
            "title": "Albert Einstein", "mimetype": "text/html", "text": html, "zim": "wiki.zim"])
        XCTAssertEqual(passages.map(\.text), [sentence])
        XCTAssertEqual(SourceBoundAnswer.answer(question: "Who was Albert Einstein?",
            topic: "Albert Einstein", passages: passages).text, sentence)
    }
    func testOverviewCopiesCompleteSourceSentencesAndPreservesNegation() {
        let text = "A duet is a composition for two performers. The performers do not have to take turns. Both parts may be performed simultaneously."
        let passages = [SourceBoundAnswer.Passage(article: "Duet", library: "fixture.zim", text: text)]
        let reply = SourceBoundAnswer.answer(question: "Tell me about Duet", topic: "Duet", passages: passages)
        XCTAssertEqual(reply.excerpts.count, 3)
        XCTAssertTrue(reply.text.contains("do not have to take turns"))
        XCTAssertTrue(reply.excerpts.allSatisfy { SourceBoundAnswer.sentences(text).contains($0.text) })
        XCTAssertTrue(AnswerAttribution.attribute(answer: reply.text, passages: [
            .init(article: "Duet", section: "lead", text: text),
        ]).allSatisfy(\.isSupported))
    }

    func testUnknownFactCannotUseTrainingKnowledgeOrQuestionClaims() {
        let passages = [SourceBoundAnswer.Passage(article: "Einstein",
            text: "Einstein was a physicist who studied in Zurich. The FBI created a secret dossier on Einstein in 1932.")]
        for question in ["What was Einstein's secret password?", "Explain his lunar expedition in 1921"] {
            let reply = SourceBoundAnswer.answer(question: question, topic: "Einstein", passages: passages)
            XCTAssertFalse(reply.hasEvidence, question)
            XCTAssertEqual(reply.text, "")
        }
        XCTAssertFalse(SourceBoundAnswer.answer(question: "Who was Einstein?", topic: "Einstein", passages: []).hasEvidence)
    }

    func testParentsNamesBeatAnEarlierFamilyBusinessMention() {
        let text = "Einstein's father and uncle founded an electrical equipment business in Munich. His parents, secular Ashkenazi Jews, were Hermann Einstein, a salesman and engineer, and Pauline Koch."
        let reply = SourceBoundAnswer.answer(question: "Tell me about his parents", topic: "Einstein",
            passages: [.init(article: "Einstein", section: "Early life", text: text)], maxSentences: 1)
        XCTAssertEqual(reply.text, "His parents, secular Ashkenazi Jews, were Hermann Einstein, a salesman and engineer, and Pauline Koch.")
    }

    func testNoClippingOffAQualificationAtTheDisplayBudget() {
        let text = "The experiment appeared successful in the initial report, but later measurements did not confirm that result."
        let reply = SourceBoundAnswer.answer(question: "Describe the experiment", topic: "experiment",
            passages: [.init(article: "Experiment", text: text)], maxCharacters: 30)
        XCTAssertEqual(reply.text, text)
    }

    func testCitationsAndWhitespaceNormalizeWithoutChangingNamesOrNumbers() {
        let reply = SourceBoundAnswer.answer(question: "Who was Einstein?", topic: "Einstein",
            passages: [.init(article: "Einstein", text: "Einstein was born in Ulm in 1879.[12]\n His mother was Pauline Koch.")])
        XCTAssertTrue(reply.text.contains("1879"))
        XCTAssertFalse(reply.text.contains("[12]"))
        XCTAssertTrue(reply.text.contains("Pauline Koch"))
    }

    func testToolEvidenceWhitelistRejectsErrorsSummariesAndSearchSnippets() {
        let arbitrary: [String: Any] = ["title": "Invented", "text": "A model invented this statement."]
        for tool in ["search", "model_summary", "get_article_section_error", "unknown"] {
            XCTAssertTrue(SourceBoundAnswer.passages(toolName: tool, result: arbitrary).isEmpty)
        }
        XCTAssertTrue(SourceBoundAnswer.passages(toolName: "get_article_section",
            result: arbitrary.merging(["error": "missing article"]) { _, b in b }).isEmpty)
    }

    func testRawHTMLIsParsedAndArchiveAndSectionArePreserved() throws {
        let passages = SourceBoundAnswer.passages(toolName: "get_article", result: [
            "title": "Duet", "zim": "music.zim", "mimetype": "text/html",
            "text": "<p>A duet is a composition for two performers.</p><h2>History</h2><p>Performers have written duets for many centuries.</p>",
        ])
        XCTAssertEqual(passages.count, 2)
        XCTAssertEqual(passages.last?.section, "History")
        XCTAssertEqual(passages.last?.library, "music.zim")
        XCTAssertFalse(try XCTUnwrap(passages.first).text.contains("<p>"))
    }

    func testComparisonIncludesBothArticlesWithoutInventingAConclusion() {
        let reply = SourceBoundAnswer.answer(question: "Compare France and Germany", topic: "France",
            passages: [
                .init(article: "France", text: "France is a country in Europe. Paris is its capital. French is its official language."),
                .init(article: "Germany", text: "Germany is a country in Europe. Berlin is its capital."),
            ])
        XCTAssertTrue(reply.text.contains("France is a country"))
        XCTAssertTrue(reply.text.contains("Germany is a country"))
    }

    func testShortFactsAndInitialsAreNotDroppedOrCutOff() {
        XCTAssertEqual(SourceBoundAnswer.answer(question: "Describe Color", topic: "Color",
            passages: [.init(article: "Color", text: "It is red.")]).text, "It is red.")
        XCTAssertEqual(SourceBoundAnswer.sentences("His father founded J. Einstein & Cie. in Munich."),
            ["His father founded J. Einstein & Cie. in Munich."])
    }
}
