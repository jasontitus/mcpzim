import XCTest
@testable import MCPZimKit

final class ValidatedDateEvidenceTests: XCTestCase {
    private let question = "When did Lithuania join NATO?"
    private func passage(_ text: String) -> SourceBoundAnswer.Passage {
        .init(article: "Lithuania", section: "Foreign relations", library: "wiki.zim", text: text)
    }

    func testReusePreservesExactTextAndProvenance() throws {
        let source = passage("Lithuania became a full member of NATO in March 2004.")
        let memory = try XCTUnwrap(ValidatedDateEvidence(question: question, passages: [source], directFact: true))
        XCTAssertEqual(memory.answer(to: "What year?", currentQuestion: question), [source])
        XCTAssertEqual(memory.answer(to: "Which year?", currentQuestion: question), [source])
    }

    func testCannotReuseForChangedSubjectRelationOrQualifier() throws {
        let memory = try XCTUnwrap(ValidatedDateEvidence(question: question,
            passages: [passage("Lithuania joined NATO in 2004.")], directFact: true))
        for prior in [nil, "When did Bulgaria join NATO?", "When did Lithuania join the EU?"] {
            XCTAssertNil(memory.answer(to: "What year?", currentQuestion: prior))
        }
        for request in ["What year did Bulgaria join?", "What year was that disputed?", "What date?", "What year is it now?"] {
            XCTAssertNil(memory.answer(to: request, currentQuestion: question))
        }
    }

    func testRejectsAmbiguousUncertainMissingAndBackgroundDates() {
        for text in ["It joined NATO in 2004 and the EU in 2007.",
                     "It did not join in 2004.", "It didn't join in 2004.", "It wasn’t in 2004.", "It might have joined in 2004.",
                     "The accession date is unknown.", "It joined NATO."] {
            XCTAssertNil(ValidatedDateEvidence(question: question, passages: [passage(text)], directFact: true))
        }
        XCTAssertNil(ValidatedDateEvidence(question: question,
            passages: [passage("It joined NATO in 2004.")], directFact: false))
        XCTAssertNil(ValidatedDateEvidence(question: "How does it relate to geopolitics?",
            passages: [passage("It joined NATO in 2004.")], directFact: true))
    }
}
