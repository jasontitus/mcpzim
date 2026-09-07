import XCTest
@testable import MCPZimKit

final class ConversationSuggestionSelectionTests: XCTestCase {
    func testSectionChoiceIsBoundToItsArticleArchiveAndHeading() {
        let choice = DiscoveryThread(label: "What about her family?", kind: .topic,
            source: .section, zim: "wiki.zim", articleTitle: "Mira", sectionTitle: "Family")
        let sections: [ArticleSection] = [
            .init(title: "", level: 0, text: "Mira is a politician."),
            .init(title: "Family", level: 2, text: "She married Alex."),
            .init(title: "Children", level: 3, text: "They have two daughters."),
            .init(title: "Career", level: 2, text: "She became mayor."),
        ]
        XCTAssertEqual(ConversationSuggestionSelection.sections(for: choice,
            articleTitle: "Mira", zim: "wiki.zim", sections: sections).map(\.title), ["Family", "Children"])
        for (title, zim) in [("Someone else", "wiki.zim"), ("Mira", "medicine.zim")] {
            XCTAssertTrue(ConversationSuggestionSelection.sections(for: choice,
                articleTitle: title, zim: zim, sections: sections).isEmpty)
        }
        XCTAssertTrue(ConversationSuggestionSelection.sections(for: choice,
            articleTitle: "Mira", zim: "wiki.zim", sections: []).isEmpty)
        XCTAssertNil(ConversationSuggestionSelection.selection(
            for: "What about her family secret password?", suggestions: [choice]))
        var missingIdentity = choice
        missingIdentity.sectionTitle = nil
        XCTAssertTrue(ConversationSuggestionSelection.sections(for: missingIdentity,
            articleTitle: "Mira", zim: "wiki.zim", sections: sections).isEmpty)
    }

    private let history = DiscoveryThread(
        label: "Early history", kind: .topic, source: .section,
        prompt: "How did this city get started?")
    private let park = DiscoveryThread(
        label: "Riverside Park", kind: .place, source: .nearbyPlace,
        lat: 37, lon: -122, prompt: "Show me Riverside Park on the map")

    func testSectionLabelPreservesQuestionInsteadOfInventingArticleTitle() {
        XCTAssertEqual(ConversationSuggestionSelection.prompt(
            for: "Tell me about the early history!", suggestions: [history, park]),
            history.prompt)
    }

    func testSpokenChoiceUsesDisplayedOrderAndMapAction() {
        XCTAssertEqual(ConversationSuggestionSelection.prompt(
            for: "The second suggestion", suggestions: [history, park]), park.prompt)
        XCTAssertEqual(ConversationSuggestionSelection.prompt(
            for: "Option 1", suggestions: [park, history]), park.prompt)
    }

    func testSingleOfferAffirmative() {
        XCTAssertEqual(ConversationSuggestionSelection.prompt(
            for: "Yes, please.", suggestions: [park]), park.prompt)
        XCTAssertNil(ConversationSuggestionSelection.prompt(
            for: "yes", suggestions: [history, park]))
    }

    func testSpokenCuesRoundTripCustomActions() throws {
        let another = DiscoveryThread(label: "Another nearby lead", kind: .place,
            source: .nearbyPlace, lat: 37, lon: -122, prompt: "Another nearby lead")
        for thread in [history, park, another] {
            let cue = try XCTUnwrap(ConversationSuggestionSelection.voiceCue([thread]))
            let spoken = try XCTUnwrap(cue.components(separatedBy: "“").dropFirst().first?
                .components(separatedBy: "”").first)
            XCTAssertEqual(ConversationSuggestionSelection.prompt(
                for: spoken, suggestions: [thread]), thread.prompt)
        }
        XCTAssertTrue(ConversationSuggestionSelection.voiceCue([park])?
            .contains("on the map") == true)
    }

    func testMedicalArticleSelectionRetainsSampledArchive() throws {
        let result: [String: Any] = ["zim": "medicine.zim", "topics": [
            ["title": "Diabetes", "path": "A/Diabetes", "preview": "A medical topic."],
        ]]
        let offers = ConversationThreads.extract(toolName: "discover_topics", result: result)
        let selected = try XCTUnwrap(ConversationSuggestionSelection.selection(
            for: "Tell me about Diabetes", suggestions: offers))
        let intent = try XCTUnwrap(ConversationSuggestionSelection.articleIntent(for: selected))
        XCTAssertEqual(intent.toolName, "article_overview")
        XCTAssertEqual(intent.args["zim"], .string("medicine.zim"))
        XCTAssertEqual(intent.args["title"], .string("Diabetes"))
        XCTAssertNil(ConversationSuggestionSelection.articleIntent(for: history))
    }

    func testNewQuestionsAndPlaceListSelectionsAreNotReplaced() {
        for text in ["Why did the second world war start?", "the second one",
                     "What happened in its early history?", "continue",
                     "no thanks", "option 8", ""] {
            XCTAssertNil(ConversationSuggestionSelection.prompt(
                for: text, suggestions: [history, park]), text)
        }
    }

    func testNoOfferAndAmbiguousLabels() {
        XCTAssertNil(ConversationSuggestionSelection.prompt(for: "yes", suggestions: []))
        let duplicate = DiscoveryThread(label: history.label, kind: .topic,
            source: .section, prompt: "A different question")
        XCTAssertNil(ConversationSuggestionSelection.prompt(
            for: history.label, suggestions: [history, duplicate]))
    }

    func testTopicFallbackAndIdempotentFullPrompt() {
        let topic = DiscoveryThread(label: "Roman Empire", kind: .topic, source: .wikilink)
        XCTAssertEqual(ConversationSuggestionSelection.prompt(
            for: "Roman Empire", suggestions: [topic]), "tell me about Roman Empire")
        XCTAssertEqual(ConversationSuggestionSelection.prompt(
            for: "tell me about Roman Empire", suggestions: [topic]),
            "tell me about Roman Empire")
    }
}
