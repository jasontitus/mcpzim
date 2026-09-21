import XCTest
@testable import MCPZimKit

final class OfflineKnowledgeTests: XCTestCase {
    private final class Reader: ZimReader, @unchecked Sendable {
        let pages: [String: (String, String)]
        let kind: ZimKind
        let suggestions: [ZimSearchHit]
        init(_ pages: [String: (String, String)], kind: ZimKind = .wikipedia, suggestions: [ZimSearchHit] = []) {
            self.pages = pages; self.kind = kind; self.suggestions = suggestions
        }
        var metadata: ZimMetadata { .init(name: "fixture", date: "2020-01") }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { true }
        var hasRoutingData: Bool { false }
        func read(path: String) throws -> ZimEntry? {
            pages[path].map { .init(path: path, title: $0.0, mimetype: "text/html", content: Data($0.1.utf8)) }
        }
        func readMainPage() throws -> ZimEntry? { nil }
        func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit] { Array(suggestions.prefix(limit)) }
    }
    private func knowledge(_ reader: Reader) -> OfflineKnowledge {
        .init(service: DefaultZimService(readers: [(name: "fixture.zim", reader: reader)]))
    }
    private let mira = OfflineArticle(zim: "fixture.zim", path: "A/Mira", title: "Mira")

    func testCurrentDocumentExportsCompleteSourceSectionsWithAttribution() async throws {
        let k = knowledge(Reader([mira.path: (mira.title, "<p>Mira was a writer.</p><h2>School</h2><p>Mira studied in Paris.</p>")]))
        let data = try await k.document(article: mira)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("Downloaded article: Mira"))
        XCTAssertTrue(text.contains("fixture.zim"))
        XCTAssertTrue(text.contains("Mira was a writer."))
        XCTAssertTrue(text.contains("Mira studied in Paris."))
        XCTAssertFalse(text.contains("<p>"))
    }

    func testCurrentDocumentRefusesOversizedUnicodeWithoutPartialExport() async throws {
        let prose = String(repeating: "日本語の文章。", count: 8000)
        let k = knowledge(Reader([mira.path: (mira.title, "<p>\(prose)</p>")]))
        do { _ = try await k.document(article: mira); XCTFail("Oversized Unicode document exported") }
        catch OfflineKnowledge.Failure.tooLarge { }
    }

    func testCurrentDocumentRejectsWrongTitleAndDisambiguation() async throws {
        for page in [("Different person", "<p>A different person's biography.</p>"),
                     ("Mira", "<p>Mira may refer to:</p>")] {
            let k = knowledge(Reader([mira.path: page]))
            do { _ = try await k.document(article: mira); XCTFail("Invalid document exported") }
            catch OfflineKnowledge.Failure.staleArticle { }
        }
    }

    func testExactArticleUsesParsedLocalProseAndFollowupFacet() async throws {
        let subject = "Mira was a writer born in 1900."
        let family = "Mira married Alex in 1930. They had two daughters."
        let k = knowledge(Reader(["A/Mira": ("Mira", "<h1>Mira</h1><p>\(subject)</p><h2>Family</h2><p>\(family)</p>")]))
        let resolution = try await k.resolve(topic: "Mira")
        XCTAssertEqual(resolution, .article(mira))
        let result = try await k.answer(question: "Tell me about Mira", article: mira)
        XCTAssertTrue(result.contains(subject), result)
        XCTAssertFalse(result.contains("<p>"))
        let followup = try await k.answer(question: "What about her family?", article: mira)
        XCTAssertTrue(followup.contains("married Alex"), followup)
    }

    func testFuzzySingleHitRequiresSelection() async throws {
        let k = knowledge(Reader(["A/Mars": ("Mars", "<p>Mars is a planet.</p>")],
                                 suggestions: [.init(path: "A/Mars", title: "Mars")]))
        let result = try await k.resolve(topic: "Marz")
        guard case .choices(let choices) = result else { return XCTFail("A fuzzy hit must not become an answer") }
        XCTAssertEqual(choices.map(\.title), ["Mars"])
        let question = OfflineKnowledge.confirmedQuestion("Tell me about Marz", requestedTopic: "Marz", article: choices[0])
        let answer = try await k.answer(question: question, article: choices[0])
        XCTAssertTrue(answer.contains("Mars is a planet"), answer)
    }

    func testDisambiguationPageProducesBoundedDistinctChoices() async throws {
        let hits = (0..<10).map { ZimSearchHit(path: "A/Mercury_\($0)", title: "Mercury (\($0))") }
        var pages = Dictionary(uniqueKeysWithValues: hits.map { ($0.path, ($0.title, "<p>This is a specific meaning of Mercury.</p>")) })
        pages["A/Mercury"] = ("Mercury", "<p>Mercury may refer to:</p>")
        let k = knowledge(Reader(pages, suggestions: [.init(path: "A/Mercury", title: "Mercury"), hits[0], hits[0]] + Array(hits.dropFirst())))
        guard case .choices(let choices) = try await k.resolve(topic: "Mercury") else { return XCTFail("Must disambiguate") }
        XCTAssertEqual(choices.count, 5)
        XCTAssertFalse(choices.contains { $0.path == "A/Mercury" })
        XCTAssertEqual(Set(choices.map(\.path)).count, choices.count)
    }

    func testUnsupportedRelationshipDoesNotReturnLead() async throws {
        let k = knowledge(Reader(["A/Mira": ("Mira", "<p>Mira was a writer born in 1900.</p>")]))
        let result = try await k.answer(question: "What was her secret password?", article: mira)
        XCTAssertTrue(result.contains("don't see an answer"), result)
        XCTAssertFalse(result.contains("born in 1900"))
    }

    func testLiveQuestionsNeverTreatArchivedHoursAsCurrent() async throws {
        let k = knowledge(Reader([:]))
        let result = try await k.answer(question: "Is it open right now?", article: mira)
        XCTAssertTrue(result.contains("cannot verify live"))
        XCTAssertFalse(OfflineKnowledge.needsLiveData("When was Mira born?"))
    }

    func testLongSentenceIsNotCutBeforeNegation() async throws {
        let sentence = "Mira was born " + String(repeating: "a very long time ago ", count: 80) + "but not in 1900."
        let k = knowledge(Reader(["A/Mira": ("Mira", "<p>\(sentence)</p>")]))
        do {
            _ = try await k.answer(question: "Tell me about Mira", article: mira)
            XCTFail("Must refuse an overlong sentence instead of truncating it")
        } catch OfflineKnowledge.Failure.tooLarge { }
    }

    func testStreetzimBundledWikipediaWorksWithoutWikipediaDownload() async throws {
        let k = knowledge(Reader(["wiki-article/Mira": ("Mira", "<p>Mira was a writer born in 1900.</p>")], kind: .streetzim))
        guard case .article(let source) = try await k.resolve(topic: "Mira") else { return XCTFail("Missing bundled article") }
        XCTAssertEqual(source.path, "wiki-article/Mira")
        let answer = try await k.answer(question: "Tell me about Mira", article: source)
        XCTAssertTrue(answer.contains("writer"))
    }

    func testMissingAndChangedArticleFailClosed() async throws {
        let k = knowledge(Reader(["A/Mira": ("Other person", "<p>This is another person.</p>")]))
        do { _ = try await k.answer(question: "Tell me about Mira", article: mira); XCTFail("Wrong identity") }
        catch OfflineKnowledge.Failure.staleArticle { }
        let empty = OfflineKnowledge(service: DefaultZimService(readers: []))
        do { _ = try await empty.resolve(topic: "Mira"); XCTFail("Empty library") }
        catch OfflineKnowledge.Failure.noLibrary { }
    }

    func testInputAndImplicitContextBoundaries() throws {
        for invalid in [" ", String(repeating: "a", count: 501), "Mira\u{0000}"] {
            XCTAssertThrowsError(try OfflineKnowledge.validatedInput(invalid))
        }
        XCTAssertNil(OfflineKnowledge.topic(in: "Who were her parents?"))
        XCTAssertNil(OfflineKnowledge.topic(in: "How does it work?"))
        XCTAssertNil(OfflineKnowledge.topic(in: "Discuss its history"))
        XCTAssertNil(OfflineKnowledge.topic(in: "delete every file"))
        XCTAssertEqual(OfflineKnowledge.topic(in: "Tell me about Albert Einstein")?.lowercased(), "albert einstein")
    }

    func testCancellationDoesNotReturnAnswer() async throws {
        let k = knowledge(Reader(["A/Mira": ("Mira", "<p>Mira was a writer.</p>")]))
        let task = Task {
            // Cancel inside the task so the production boundary is reached;
            // cancelling a sleep would only test Swift's sleep implementation.
            withUnsafeCurrentTask { $0?.cancel() }
            return try await k.answer(question: "Tell me about Mira", article: mira)
        }
        do { _ = try await task.value; XCTFail("Cancelled request answered") } catch is CancellationError { }
    }

    func testCancellationDoesNotResolveAnArticle() async throws {
        let k = knowledge(Reader(["A/Mira": ("Mira", "<p>Mira was a writer.</p>")]))
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await k.resolve(topic: "Mira")
        }
        do { _ = try await task.value; XCTFail("Cancelled request resolved") } catch is CancellationError { }
    }

    func testUnreadableAndDisambiguationSuggestionsCannotBecomeChoices() async throws {
        let k = knowledge(Reader([
            "A/Meanings": ("Meanings", "<p>This term may refer to:</p>"),
            "A/Mars": ("Mars", "<p>Mars is a planet.</p>")
        ], suggestions: [
            .init(path: "A/Missing", title: "Missing"),
            .init(path: "A/Meanings", title: "Meanings"),
            .init(path: "A/Mars", title: "Mars")
        ]))
        let result = try await k.resolve(topic: "Marz")
        XCTAssertEqual(result, .choices([.init(zim: "fixture.zim", path: "A/Mars", title: "Mars")]))
        let noHits = knowledge(Reader([:]))
        let empty = try await noHits.resolve(topic: "Unknown subject")
        XCTAssertEqual(empty, .choices([]))
    }

    func testAnswerRejectsOversizedArticleBytes() async throws {
        let html = "<p>Mira was a writer.</p><!--" + String(repeating: "x", count: OfflineKnowledge.maxHTMLBytes) + "-->"
        let k = knowledge(Reader(["A/Mira": ("Mira", html)]))
        do { _ = try await k.answer(question: "Tell me about Mira", article: mira); XCTFail("Oversized HTML accepted") }
        catch OfflineKnowledge.Failure.tooLarge { }
    }

    func testMarkupHeavyWikipediaPageStillAnswersFromShortProse() async throws {
        let html = "<div data-mw='" + String(repeating: "x", count: 700_000)
            + "'><p>Albert Einstein was a theoretical physicist who developed the theory of relativity.</p></div>"
        let article = OfflineArticle(zim: "fixture.zim", path: "A/Albert_Einstein", title: "Albert Einstein")
        let k = knowledge(Reader([article.path: (article.title, html)]))
        let answer = try await k.answer(question: "Tell me about Albert Einstein", article: article)
        XCTAssertTrue(answer.contains("theoretical physicist"), answer)
        XCTAssertFalse(answer.contains("data-mw"))
        XCTAssertLessThan(answer.count, 900)
    }

    func testParsedProseAndSectionBudgetsStillRejectOversizedWork() async throws {
        let excessiveProse = "<p>" + String(repeating: "Mira wrote books. ", count: 30_000) + "</p>"
        let excessiveSections = (0..<251).map { "<h2>Chapter \($0)</h2><p>Mira wrote a book.</p>" }.joined()
        for html in [excessiveProse, excessiveSections] {
            let k = knowledge(Reader([mira.path: (mira.title, html)]))
            do { _ = try await k.answer(question: "Tell me about Mira", article: mira); XCTFail("Scoring budget bypassed") }
            catch OfflineKnowledge.Failure.tooLarge { }
        }
    }

    func testCancellationAfterDiagnosticCallbackStopsAnswer() async throws {
        let k = knowledge(Reader([mira.path: (mira.title, "<p>Mira was a writer.</p>")]))
        let task = Task {
            try await k.answer(question: "Tell me about Mira", article: mira) { measurement in
                if case .htmlBytes = measurement { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        do { _ = try await task.value; XCTFail("Cancelled after diagnostics but still answered") }
        catch is CancellationError { }
    }

    func testBirthplaceSubjectsAcrossDirectAndConversationalPhrasing() {
        for question in [
            "Where was Vladimir Putin born?",
            "Where is Vladimir Putin born?",
            "Tell me where Vladimir Putin was born",
            "Please tell me, where was Vladimir Putin born?",
            "Could you please tell me where Vladimir Putin was born?",
            "When was Vladimir Putin born?",
            "Tell me when Vladimir Putin was born.",
        ] {
            XCTAssertEqual(OfflineKnowledge.topic(in: question), "Vladimir Putin", question)
        }
        XCTAssertEqual(OfflineKnowledge.topic(in: "where was vladimir putin born"), "vladimir putin")
        XCTAssertEqual(OfflineKnowledge.topic(in: "Where was Anne-Marie O’Neill born?"), "Anne-Marie O’Neill")
        XCTAssertEqual(OfflineKnowledge.topic(in: "Tell me about Albert Einstein")?.lowercased(), "albert einstein")
    }

    func testBirthplaceInferenceDoesNotGuessPronounsRelativesOrMultiplePeople() {
        for question in [
            "Where was he born?", "Tell me where she was born", "Where was my father born?",
            "Where was your mother born?", "Where were we born?", "Where was someone born?",
            "Where was Putin’s father born?", "Where was Vladimir Putin and Alex born?",
            "Where was Vladimir Putin or Alex born?",
            "Where was Vladimir Putin born and where does he live?"
        ] { XCTAssertNil(OfflineKnowledge.topic(in: question), question) }
    }

    func testBirthplaceAnswerKeepsNamedPersonAndPlaceRelation() async throws {
        let k = knowledge(Reader([mira.path: (mira.title, """
        <p>Mira (born 1900) was a writer.</p>
        <h2>Family</h2><p>Her mother was born in Riga.</p>
        <h2>Early life</h2><p>Mira was born in London on 1 January 1900.</p>
        """)]))
        let question = "Tell me where Mira was born"
        let title = try XCTUnwrap(OfflineKnowledge.topic(in: question))
        guard case .article(let source) = try await k.resolve(topic: title) else { return XCTFail("Named subject did not resolve") }
        let answer = try await k.answer(question: question, article: source)
        XCTAssertTrue(answer.contains("Mira was born in London"), answer)
        XCTAssertFalse(answer.contains("Riga"), answer)
        XCTAssertFalse(answer.contains("(born 1900)"), answer)
    }

    func testBirthplaceQuestionAbstainsOnDateOnlyAndRelativeEvidence() async throws {
        for prose in [
            "Mira (born 1900) was a writer.",
            "Mira was born in October 1900.",
            "Mira was a writer. Her mother was a poet. She was born in Riga.",
            "Mira was married to Alex. He was born in Riga.",
            "Mira was born in 1900, and her mother was born in Riga."
        ] {
            let k = knowledge(Reader([mira.path: (mira.title, "<p>\(prose)</p>")]))
            let answer = try await k.answer(question: "Where was Mira born?", article: mira)
            XCTAssertTrue(answer.contains("don't see an answer"), answer)
        }
    }

    func testBirthplaceDoesNotSubstituteArticleSubjectForRelative() async throws {
        let k = knowledge(Reader([mira.path: (mira.title, "<p>Mira was born in London, where her father worked.</p>")]))
        for question in ["Where was her father born?", "Where was Mira's father born?"] {
            let answer = try await k.answer(question: question, article: mira)
            XCTAssertTrue(answer.contains("don't see an answer"), answer)
        }
    }

    func testBirthplaceSupportsConfirmedDisambiguatedBiography() async throws {
        let source = OfflineArticle(zim: mira.zim, path: "A/John_Smith_(explorer)", title: "John Smith (explorer)")
        let k = knowledge(Reader([source.path: (source.title, "<p>John Smith was born in Lincolnshire.</p>")]))
        let answer = try await k.answer(question: "Where was John Smith (explorer) born?", article: source)
        XCTAssertTrue(answer.contains("John Smith was born in Lincolnshire"), answer)
    }

    func testAnswerUsesSelectedArchiveWhenTitlesCollide() async throws {
        let first = Reader(["A/Mira": ("Mira", "<p>Mira was a writer born in 1900.</p>")])
        let selected = Reader(["A/Mira": ("Mira", "<p>Mira was a writer born in 1910.</p>")])
        let k = OfflineKnowledge(service: DefaultZimService(readers: [
            (name: "first.zim", reader: first), (name: "selected.zim", reader: selected)
        ]))
        let article = OfflineArticle(zim: "selected.zim", path: "A/Mira", title: "Mira")
        let answer = try await k.answer(question: "When was Mira born?", article: article)
        XCTAssertTrue(answer.contains("1910"), answer)
        XCTAssertFalse(answer.contains("1900"), answer)
    }

    func testConfirmedCorrectionPreservesQuestionFacetAndOtherWords() {
        let article = OfflineArticle(zim: "fixture.zim", path: "A/Mars", title: "Mars")
        XCTAssertEqual(OfflineKnowledge.confirmedQuestion("When was Marz discovered?", requestedTopic: "Marz", article: article),
                       "When was Mars discovered?")
        XCTAssertEqual(OfflineKnowledge.confirmedQuestion("Compare Marzipan with Marz", requestedTopic: "Marz", article: article),
                       "Compare Marzipan with Mars")
        XCTAssertEqual(OfflineKnowledge.confirmedQuestion("Who were her parents?", requestedTopic: nil, article: article),
                       "Who were her parents?")
    }
}
