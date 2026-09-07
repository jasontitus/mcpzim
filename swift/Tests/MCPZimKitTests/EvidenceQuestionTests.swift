import XCTest
@testable import MCPZimKit

final class EvidenceQuestionTests: XCTestCase {
    let topic = "George Bernard Shaw"
    let works = "He wrote more than sixty plays, including major works such as Man and Superman (1902), Pygmalion (1913) and Saint Joan (1923)."
    func reply(_ question: String, _ text: String, section: String = "lead") -> SourceBoundAnswer.Reply {
        SourceBoundAnswer.answer(question: question, topic: topic,
            passages: [.init(article: topic, section: section, text: text)])
    }
    func testShawFollowupsSelectSameGroundedWorkList() {
        let text = "George Bernard Shaw was an Irish playwright. His influence extended beyond his death. " + works
        for q in ["What did he write?", "What plays did he write?", "Which plays did Shaw write?"] {
            XCTAssertEqual(reply(q, text).text, works, q)
        }
    }
    func testOtherPeoplesWritingCannotBorrowArticleIdentity() {
        for text in [
            "Shaw inspired Behrman, who wrote plays including Example Play.",
            "George Bernard Shaw was a playwright. Behrman admired Shaw. He wrote Example Play.",
            "George Bernard Shaw was a playwright. He continued to write prolifically until his death.",
            "George Bernard Shaw did not write Pygmalion.",
            "George Bernard Shaw said he would write Pygmalion.",
            "George Bernard Shaw wrote about Pygmalion.",
            "George Bernard Shaw wrote that Behrman authored Example Play.",
            "George Bernard Shaw met Behrman. He wrote Example Play."
        ] { XCTAssertFalse(reply("What did he write?", text).hasEvidence, text) }
    }
    func testExtraFacetsAndWorkTypesCannotBeDropped() {
        let text = "George Bernard Shaw was a playwright. " + works
        XCTAssertFalse(reply("What novels did he write?", text).hasEvidence)
        XCTAssertFalse(reply("What plays did he write?", "Shaw wrote a book about Pygmalion.", section: "Plays").hasEvidence)
        XCTAssertFalse(reply("What plays did he write in prison?", text).hasEvidence)
    }
    func testNamedSubjectAndPossessiveWorkList() {
        XCTAssertTrue(reply("What did he write?", "Shaw authored Pygmalion.").hasEvidence)
        XCTAssertTrue(reply("What plays did he write?", "Shaw's plays include Pygmalion.").hasEvidence)
    }
    func testLocalRecommendationIsCategoryAtExplicitCenter() {
        for q in ["Where is a good coffee shop in Salinas", "Where's a nice museum near Salinas?"] {
            let intent = IntentRouter.classify(q, currentLocation: (36.4, -121.6))
            XCTAssertEqual(intent?.toolName, "near_named_place")
            XCTAssertEqual(intent?.args["place"], .string("salinas"))
        }
        XCTAssertEqual(IntentRouter.classify("Where is a good coffee shop in Salinas")?.args["kinds"], .array([.string("coffee shop")]))
        XCTAssertEqual(IntentRouter.classify("Where is HP Garage?")?.toolName, "locate")
    }
    func testFuzzyNameDoesNotAuthorizeIdentity() {
        XCTAssertFalse(EntityResolutionPolicy.sameTitle("David crocket", "David Crocket Graham"))
        XCTAssertFalse(EntityResolutionPolicy.sameTitle("David crocket", "Davy Crockett"))
        XCTAssertTrue(EntityResolutionPolicy.sameTitle("george bernard shaw", "George_Bernard_Shaw"))
    }
}

final class EntityResolutionIntegrationTests: XCTestCase {
    final class Reader: ZimReader, @unchecked Sendable {
        var metadata: ZimMetadata { .init(name: "wikipedia-test") }
        var kind: ZimKind { .wikipedia }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { true }
        var hasRoutingData: Bool { false }
        func read(path: String) throws -> ZimEntry? {
            guard ["A/David_Crocket_Graham", "A/Davy_Crockett", "A/Lenin"].contains(path) else { return nil }
            let title = path == "A/Lenin" ? "Vladimir Lenin" : path.dropFirst(2).replacingOccurrences(of: "_", with: " ")
            return .init(path: path, title: title, mimetype: "text/html",
                         content: Data("<html><body><p>\(title) was a historical person with a biography.</p></body></html>".utf8))
        }
        func readMainPage() throws -> ZimEntry? { nil }
        func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit] {
            [.init(path: "A/David_Crocket_Graham", title: "David Crocket Graham"),
             .init(path: "A/Davy_Crockett", title: "Davy Crockett")]
        }
    }
    func testFuzzyCrockettRequiresConfirmationThroughRealServiceAndAdapter() async throws {
        let service = DefaultZimService(readers: [(name: "wikipedia.zim", reader: Reader())])
        let adapter = MCPToolAdapter(service: service, hasStreetzim: false)
        let result = try await adapter.dispatch(tool: "article_overview", args: ["title": "David crocket"])
        XCTAssertEqual(result["ambiguous"] as? Bool, true)
        XCTAssertEqual(result["resolution"] as? String, "unconfirmed_title")
        XCTAssertNil(result["sections"])
        XCTAssertTrue((result["suggestions"] as? [String] ?? []).contains("Davy Crockett"))
    }
    func testRealRedirectPathRemainsUsable() async throws {
        let service = DefaultZimService(readers: [(name: "wikipedia.zim", reader: Reader())])
        let result = try await service.articleByTitle(title: "Lenin", zim: nil, section: "lead")
        XCTAssertEqual(result.title, "Vladimir Lenin")
    }
}

final class HistoricalConversationCompatibilityTests: XCTestCase {
    func testLoggedNonAuthorshipQuestionsKeepExistingEvidenceContract() {
        for question in [
            "What is Lenin's legacy?", "What was Lenin's personal life like?",
            "What was Putin's early life like?", "Where did Putin go to school?",
            "How did Putin's career develop?", "What about Putin's family?",
            "How has Putin dealt with the West and NATO?", "Who was his mother?",
            "What was his secret password?", "How did its history unfold?"
        ] { XCTAssertNil(EvidenceQuestion.parse(question), question) }
    }
    func testLoggedNearestCoffeeInNamedCityDoesNotBecomeCategoryText() {
        let intent = IntentRouter.classify("Where is the nearest coffee shop in Carmel Valley?",
                                           currentLocation: (36.43257, -121.66241))
        XCTAssertEqual(intent?.toolName, "near_named_place")
        XCTAssertEqual(intent?.args["place"], .string("carmel valley"))
        XCTAssertEqual(intent?.args["kinds"], .array([.string("coffee shop")]))
    }
}
