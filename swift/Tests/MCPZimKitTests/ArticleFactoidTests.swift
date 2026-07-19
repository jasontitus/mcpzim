// SPDX-License-Identifier: MIT

import Foundation
import XCTest
@testable import MCPZimKit

final class ArticleFactoidTests: XCTestCase {

    func testFoundingDateQuestionsRouteWithoutLLM() {
        for query in [
            "When was Tesla founded?",
            "What year was Tesla established",
            "In what year was Tesla incorporated?",
        ] {
            let intent = IntentRouter.classify(query)
            XCTAssertEqual(intent?.toolName, "article_factoid", "query: \(query)")
            XCTAssertEqual(intent?.args["title"], .string("tesla"))
            XCTAssertEqual(intent?.args["predicate"], .string("foundation"))
            XCTAssertEqual(intent?.args["implicit"], .bool(false))
        }
    }

    func testShortenedWhenWasCompanyUsesTentativeFastPath() {
        let intent = IntentRouter.classify("When was Apple?")
        XCTAssertEqual(intent?.toolName, "article_factoid")
        XCTAssertEqual(intent?.args["title"], .string("apple"))
        XCTAssertEqual(intent?.args["implicit"], .bool(true))
    }

    func testUnrelatedWhenQuestionsRemainOnGeneralPath() {
        // The router may tentatively probe this shortened shape, but the
        // adapter's company-language gate must reject the Alamo article and
        // return control to the general historical route.
        XCTAssertEqual(IntentRouter.classify("When was the Alamo?")?.args["implicit"],
                       .bool(true))
        XCTAssertNotEqual(IntentRouter.classify("When did Tesla release the Model S?")?.toolName,
                          "article_factoid")
    }

    func testReferentialFoundingQuestionUsesConversationFocus() {
        var focus = ConversationFocus()
        focus.remember(FocusEntity(name: "Tesla, Inc.", kind: .topic))
        let intent = IntentRouter.classify("When was it founded?", focus: focus)
        XCTAssertEqual(intent?.toolName, "article_factoid")
        XCTAssertEqual(intent?.args["title"], .string("Tesla, Inc."))
    }

    func testAgeQuestionUsesTentativeGroundedFactoid() {
        let intent = IntentRouter.classify("How old is San Francisco?")
        XCTAssertEqual(intent?.toolName, "article_factoid")
        XCTAssertEqual(intent?.args["title"], .string("San Francisco"))
        XCTAssertEqual(intent?.args["predicate"], .string("age"))
        XCTAssertEqual(intent?.args["tentative"], .bool(true))
    }

    func testReferentialAgeQuestionUsesConversationFocus() {
        var focus = ConversationFocus()
        focus.remember(FocusEntity(name: "Stanford University", kind: .topic))
        let intent = IntentRouter.classify("How old is it?", focus: focus)
        XCTAssertEqual(intent?.toolName, "article_factoid")
        XCTAssertEqual(intent?.args["title"], .string("Stanford University"))
        XCTAssertEqual(intent?.args["predicate"], .string("age"))
    }

    func testFactoidClarificationSelectionRetainsOriginalPredicate() {
        var focus = ConversationFocus()
        focus.setLastList([
            FocusEntity(name: "George Washington", kind: .topic),
            FocusEntity(name: "Washington (state)", kind: .topic),
            FocusEntity(name: "Washington, D.C.", kind: .topic),
        ])
        let intent = IntentRouter.factoidSelectionIntent(
            "the second one", predicate: "age", focus: focus)
        XCTAssertEqual(intent?.toolName, "article_factoid")
        XCTAssertEqual(intent?.args["title"], .string("Washington (state)"))
        XCTAssertEqual(intent?.args["predicate"], .string("age"))
        XCTAssertEqual(intent?.args["tentative"], .bool(true))
    }

    func testFactoidClarificationAcceptsDescriptiveNamePick() {
        var focus = ConversationFocus()
        focus.setLastList([
            FocusEntity(name: "George Washington", kind: .topic),
            FocusEntity(name: "Washington (state)", kind: .topic),
            FocusEntity(name: "Washington, D.C.", kind: .topic),
        ])
        let intent = IntentRouter.factoidSelectionIntent(
            "the state", predicate: "age", focus: focus)
        XCTAssertEqual(intent?.args["title"], .string("Washington (state)"))
    }

    func testFoundationExtractorProtectsCompanyAbbreviation() {
        let lead = "Tesla, Inc. was incorporated in July 2003 by Martin Eberhard "
            + "and Marc Tarpenning. It later expanded into energy products."
        let fact = IntentRouter.extractFoundationFact(from: lead, title: "Tesla, Inc.")
        XCTAssertEqual(fact,
                       "Tesla, Inc. was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning.")
    }

    func testFoundationExtractorReplacesAmbiguousOpeningPronoun() {
        let lead = "It is an American automaker. The company was founded in 2003."
        let fact = IntentRouter.extractFoundationFact(from: lead, title: "Tesla, Inc.")
        XCTAssertEqual(fact, "Tesla, Inc. was founded in 2003.")
    }

    func testFoundationExtractorRequiresYearAndVerb() {
        XCTAssertNil(IntentRouter.extractFoundationFact(
            from: "The company was founded by two engineers."))
        XCTAssertNil(IntentRouter.extractFoundationFact(
            from: "The company released its first product in 2008."))
    }

    func testPlaceOriginExtractorTiesSettlerArrivalToNamedSettlement() {
        let lead = """
        George Vancouver first visited the area in 1792. Arthur A. Denny and
        the Denny Party arrived at Alki Point on November 13, 1851. The
        settlement was moved to Elliott Bay in 1852 and named Seattle.
        """
        XCTAssertEqual(
            IntentRouter.extractPlaceOriginFact(from: lead, title: "Seattle"),
            "Arthur A. Denny and the Denny Party arrived at Alki Point on November 13, 1851. The settlement was moved to Elliott Bay in 1852 and named Seattle."
        )
    }

    func testPlaceOriginExtractorPrefersStatehoodOverEarlierTreaty() {
        let lead = """
        The state was formed from territory claimed in the Oregon Treaty of
        1846. It was admitted to the Union as the 42nd state in 1889.
        """
        XCTAssertEqual(
            IntentRouter.extractPlaceOriginFact(
                from: lead, title: "Washington (state)"),
            "It was admitted to the Union as the 42nd state in 1889."
        )
    }

    func testPlaceOriginExtractorDoesNotTreatFoundingFatherAsFounded() {
        let lead = "George Washington (February 22, 1732 – December 14, 1799) "
            + "was a Founding Father and the first president of the United States. "
            + "He established a strong national government in 1788."
        XCTAssertNil(IntentRouter.extractPlaceOriginFact(
            from: lead, title: "George Washington"))
    }

    func testPlaceOriginExtractorAcceptsFoundingDateButNotFoundingFather() {
        let lead = "The Residence Act, adopted on July 16, 1790, approved the "
            + "creation of the capital district and is considered the city's "
            + "founding date. A statehood bill passed the House in 2021."
        XCTAssertEqual(
            IntentRouter.extractPlaceOriginFact(
                from: lead, title: "Washington, D.C."),
            "The Residence Act, adopted on July 16, 1790, approved the creation of the capital district and is considered the city's founding date."
        )
    }

    func testAdapterSkipsDisambiguationAndUsesCompanyArticle() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "tesla", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Tesla", title: "Tesla",
            section: ArticleSection(
                title: "", level: 0,
                text: "Tesla may refer to: Nikola Tesla; Tesla, Inc.; or the tesla unit."))
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "tesla, Inc.", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Tesla,_Inc.", title: "Tesla, Inc.",
            section: ArticleSection(
                title: "", level: 0,
                text: "Tesla, Inc. is an American automaker. The company was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning."))

        let service = StubZimService(fixture: fixture)
        let adapter = await MCPToolAdapter(service: service, hasStreetzim: false)
        let result = try await adapter.dispatch(
            tool: "article_factoid",
            args: ["title": "tesla", "predicate": "foundation"])

        XCTAssertEqual(result["title"] as? String, "Tesla, Inc.")
        XCTAssertEqual(result["resolution"] as? String, "direct")
        XCTAssertEqual(result["fact"] as? String,
                       "Tesla, Inc. was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning.")
        XCTAssertTrue(IntentRouter.articleFactoidResultIsUsable(result))
    }

    func testImplicitCompanyQuestionAcceptsAppleInc() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "apple", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Apple", title: "Apple",
            section: ArticleSection(
                title: "", level: 0,
                text: "Apple may refer to the fruit, Apple Inc., or several other topics."))
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "apple, Inc.", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Apple_Inc.", title: "Apple Inc.",
            section: ArticleSection(
                title: "", level: 0,
                text: "Apple Inc. is an American multinational technology company. It was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne."))

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "apple", "predicate": "foundation", "implicit": true,
        ])
        XCTAssertEqual(result["title"] as? String, "Apple Inc.")
        XCTAssertEqual(result["fact"] as? String,
                       "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.")
    }

    func testImplicitHistoricalQuestionRejectsNonCompanyArticle() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "the alamo", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Alamo_Mission", title: "Alamo Mission",
            section: ArticleSection(
                title: "", level: 0,
                text: "The Alamo is a historic Spanish mission and fortress compound founded in 1718. The Battle of the Alamo took place in 1836."))

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "the alamo", "predicate": "foundation", "implicit": true,
        ])
        XCTAssertNotNil(result["error"])
        XCTAssertFalse(IntentRouter.articleFactoidResultIsUsable(result))
    }

    func testAgeFactoidFallsBackToHistoryAndCalculatesApproximateAge() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "San Francisco", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/San_Francisco", title: "San Francisco",
            section: ArticleSection(
                title: "", level: 0,
                text: "San Francisco is a city in Northern California."))
        fixture.articleSections[
            StubZimService.keyArticleSections(path: "A/San_Francisco")
        ] = .init(
            zim: "wikipedia.zim", title: "San Francisco",
            sections: [
                ArticleSection(title: "Geography", level: 2,
                               text: "The city sits on a peninsula."),
                ArticleSection(
                    title: "History", level: 2,
                    text: "On June 29, 1776, Spanish colonists established the Presidio of San Francisco and Mission San Francisco de Asís."),
            ])

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "San Francisco", "predicate": "age", "tentative": true,
        ])
        let currentYear = Calendar.current.component(.year, from: Date())
        XCTAssertEqual(result["predicate"] as? String, "age")
        XCTAssertEqual(result["evidence"] as? String,
                       "On June 29, 1776, Spanish colonists established the Presidio of San Francisco and Mission San Francisco de Asís.")
        XCTAssertEqual(result["fact"] as? String,
                       "On June 29, 1776, Spanish colonists established the Presidio of San Francisco and Mission San Francisco de Asís. That was about \(currentYear - 1776) years ago.")
        XCTAssertTrue(IntentRouter.articleFactoidResultIsUsable(result))
    }

    func testAgeFactoidStripsUSStateQualifierForOfflineTitleLookup() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Seattle", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Seattle", title: "Seattle",
            section: ArticleSection(
                title: "", level: 0,
                // Mirrors the device failure: the extracted lead slice may
                // omit the state even though the canonical search hit proves
                // the qualified title refers to this article.
                text: "Seattle is a city in the Pacific Northwest. It was founded in 1851."))
        fixture.search[
            StubZimService.keySearch(query: "Seattle Washington")
        ] = [
            SearchHitResult(
                zim: "wikipedia.zim", kind: .wikipedia,
                path: "A/Seattle", title: "Seattle",
                snippet: "Seattle is the largest city in the state of Washington."),
        ]

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "Seattle, Washington", "predicate": "age", "tentative": true,
        ])
        XCTAssertEqual(result["title"] as? String, "Seattle")
        XCTAssertEqual(result["resolution"] as? String, "direct")
        XCTAssertTrue(IntentRouter.articleFactoidResultIsUsable(result))
    }

    func testAgeFactoidRejectsStrippedQualifierWhenArticleContradictsIt() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Portland", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Portland,_Oregon", title: "Portland",
            section: ArticleSection(
                title: "", level: 0,
                text: "Portland is the most populous city in Oregon. It was founded in 1845."))

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "Portland, Washington", "predicate": "age", "tentative": true,
        ])
        XCTAssertNotNil(result["error"])
        XCTAssertFalse(IntentRouter.articleFactoidResultIsUsable(result))
    }

    func testAgeFactoidNeverFallsThroughToCompanyTitleVariants() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Seattle", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Seattle", title: "Seattle",
            section: ArticleSection(
                title: "", level: 0,
                text: "Seattle is a city in the Pacific Northwest."))
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Seattle, Inc.", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Seattle_Computer_Products",
            title: "Seattle Computer Products, Inc.",
            section: ArticleSection(
                title: "", level: 0,
                text: "Seattle Computer Products, Inc. was founded in 1978."))

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "Seattle", "predicate": "age", "tentative": true,
        ])
        XCTAssertNil(result["fact"])
        XCTAssertNotEqual(result["title"] as? String,
                          "Seattle Computer Products, Inc.")
    }

    func testAgeFactoidInspectsCenturySubsectionsForCityIncorporation() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Seattle", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Seattle", title: "Seattle",
            section: ArticleSection(
                title: "", level: 0,
                text: "Seattle is the most populous city in Washington."))
        fixture.articleSections[
            StubZimService.keyArticleSections(path: "A/Seattle")
        ] = .init(
            zim: "wikipedia.zim", title: "Seattle",
            sections: [
                ArticleSection(title: "History", level: 2,
                               text: "The area has been inhabited for thousands of years."),
                ArticleSection(title: "19th century", level: 3,
                               text: "The City of Seattle was incorporated in 1869."),
            ])

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "Seattle", "predicate": "age", "tentative": true,
        ])
        XCTAssertEqual(result["evidence"] as? String,
                       "The City of Seattle was incorporated in 1869.")
        XCTAssertTrue(IntentRouter.articleFactoidResultIsUsable(result))
    }

    func testAgeFactoidReturnsGroundedDisambiguationCandidates() async throws {
        var fixture = StubZimService.Fixture()
        fixture.articleByTitle[
            StubZimService.keyArticleByTitle(title: "Washington", section: "lead")
        ] = .init(
            zim: "wikipedia.zim", path: "A/Washington", title: "Washington",
            section: ArticleSection(
                title: "", level: 0,
                text: "Washington may refer to several people, places, and other topics."))
        fixture.search[StubZimService.keySearch(query: "Washington")] = [
            SearchHitResult(
                zim: "wikipedia.zim", kind: .wikipedia,
                path: "A/Washington_(state)", title: "Washington (state)",
                snippet: "A state in the Pacific Northwest."),
            SearchHitResult(
                zim: "wikipedia.zim", kind: .wikipedia,
                path: "A/Washington,_D.C.", title: "Washington, D.C.",
                snippet: "The capital city of the United States."),
            SearchHitResult(
                zim: "wikipedia.zim", kind: .wikipedia,
                path: "A/George_Washington", title: "George Washington",
                snippet: "The first president of the United States."),
        ]

        let adapter = MCPToolAdapter(
            service: StubZimService(fixture: fixture), hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_factoid", args: [
            "title": "Washington", "predicate": "age", "tentative": true,
        ])
        XCTAssertEqual(result["ambiguous"] as? Bool, true)
        XCTAssertEqual(result["suggestions"] as? [String], [
            "Washington (state)", "Washington, D.C.", "George Washington",
        ])
        XCTAssertFalse(IntentRouter.articleFactoidResultIsUsable(result))
        XCTAssertEqual(
            IntentRouter.synthesizeArticleFactoidReply(
                args: ["title": "Washington"], fullResult: result),
            "Which “Washington” did you mean — Washington (state), Washington, D.C., or George Washington?")
    }
}
