import XCTest
@testable import MCPZimKit

final class ExplorationPlanTests: XCTestCase {
    func testEventTransitionsRequireSubjectAwareSelection() {
        for question in ["How did it end?", "Why did the empire collapse?", "When did it dissolve?", "How did it begin?"] {
            XCTAssertTrue(ExplorationPlan.requiresEventSelection(question), question)
            XCTAssertTrue(ExplorationPlan.requiresRelationshipSelection(question), question)
        }
        for question in ["Tell me about End of Days", "What did Shaw write?", "How far is the bar?", "How old was he?", "How did Mira's career develop?"] {
            XCTAssertFalse(ExplorationPlan.requiresEventSelection(question), question)
        }
        XCTAssertEqual(ExplorationPlan.refineDateQuestion("What year?",
            previous: "How did the Grand Duchy of Lithuania end?"),
            "When did the Grand Duchy of Lithuania end?")
        let passages: [SourceBoundAnswer.Passage] = [
            .init(article: "Grand Duchy of Lithuania", section: "Territorial expansion",
                  text: "The rule of the Mongols did not end, though their influence waned."),
            .init(article: "Grand Duchy of Lithuania", section: "Partitions",
                  text: "In 1812 the plans to recreate the Grand Duchy came to an end."),
            .init(article: "Grand Duchy of Lithuania", section: "Administration",
                  text: "The separate court came to an end in 1447."),
        ]
        for overview in [false, true] {
            XCTAssertFalse(SourceBoundAnswer.answer(question: "How did it end?",
                topic: "Grand Duchy of Lithuania", passages: passages,
                sectionOverview: overview).hasEvidence)
        }
    }

    func testSubjectAndMediumTransitionsPreserveTheRequestedFacet() {
        XCTAssertEqual(ExplorationPlan.substituteSubject(in: "When did Bulgaria join NATO?", old: "Bulgaria", new: "Lithuania"), "When did Lithuania join NATO?")
        XCTAssertEqual(ExplorationPlan.substituteSubject(in: "When did it join NATO?", old: "Bulgaria", new: "Lithuania"), "When did Lithuania join NATO?")
        XCTAssertNil(ExplorationPlan.substituteSubject(in: "What year?", old: "Bulgaria", new: "Lithuania"))
        XCTAssertEqual(ExplorationPlan.mediumTitle(question: "And the movie?", anchor: "9 to 5 (song)"), "9 to 5 (film)")
        XCTAssertNil(ExplorationPlan.mediumTitle(question: "And the movie with a secret password?", anchor: "9 to 5 (song)"))
        XCTAssertNil(ExplorationPlan.mediumTitle(question: "And the movie?", anchor: "Lithuania"))
    }

    func testDateRefinementAndOverlongValidSelection() throws {
        XCTAssertEqual(ExplorationPlan.refineDateQuestion("What year?", previous: "When did Lithuania join NATO?"), "When did Lithuania join NATO?")
        XCTAssertNil(ExplorationPlan.refineDateQuestion("What year?", previous: "How is Lithuania doing now?"))
        let plan = try XCTUnwrap(ExplorationPlan.decode(#"{"question":"Date","need":"fact","time":"current","queries":["Lithuania"]}"#))
        XCTAssertFalse(plan.requiresTemporalCaution(originalQuestion: "When did Lithuania join NATO?"))
        let windows = (0..<5).map { ExplorationEvidence.Window(id: $0, passage: .init(article: "Test", text: "Exact sentence number \($0).")) }
        let selected = try XCTUnwrap(ExplorationEvidence.selected(#"{"direct":[0,1,2,3],"background":[4]}"#, windows: windows))
        XCTAssertEqual(selected.direct.count, 3)
        XCTAssertTrue(selected.background.isEmpty)
    }

    func testChronologicalContinuationCannotDropTheInheritedFacet() {
        let current = "Then what happened in Soviet times"
        let contextual = current + " — buddhism"
        XCTAssertEqual(ExplorationEvidence.namedTerms(current), ["soviet"])
        XCTAssertFalse(ExplorationEvidence.coversContextAnchor(.init(article: "Buddhism in Buryatia", text: "Buddhism was persecuted in Soviet times."), anchor: "Mongolia"))
        XCTAssertTrue(ExplorationEvidence.coversContextAnchor(.init(article: "Buddhism in Mongolia", text: "Buddhism was persecuted in Soviet times."), anchor: "Mongolia"))
        XCTAssertFalse(ExplorationEvidence.coversInheritedFacet(.init(article: "Soviet Union", text: "The Soviet Union invaded Poland in 1939."), current: current, contextual: contextual))
        XCTAssertTrue(ExplorationEvidence.coversInheritedFacet(.init(article: "Mongolia", text: "Buddhist monasteries were closed during the purges."), current: current, contextual: contextual))
    }

    func testExplicitFacetDoesNotInheritAnUnrelatedExplorationFrame() {
        XCTAssertFalse(ExplorationPlan.needsFrameResolution("How about the Mongols?"))
        XCTAssertFalse(ExplorationPlan.needsFrameResolution("What about geography and climate?"))
        XCTAssertTrue(ExplorationPlan.needsFrameResolution("How is it relevant to the war in Ukraine?"))
        XCTAssertTrue(ExplorationPlan.needsFrameResolution("Then what happened in Soviet times?"))
        XCTAssertTrue(ExplorationPlan.needsFrameResolution("What year?"))
    }

    func testMalformedPlanFallbackOnlyResolvesNavigation() {
        let plan = ExplorationPlan.fallback(question: "When did it expand into Siberia?", anchor: "Russia")
        XCTAssertEqual(plan.question, "When did Russia expand into Siberia?")
        XCTAssertEqual(plan.time, .historical)
        XCTAssertTrue(plan.queries.allSatisfy { $0.count <= 120 })
        XCTAssertTrue(ExplorationPlan.requiresRelationshipSelection("What were the sides in the civil war?"))
        XCTAssertFalse(ExplorationPlan.requiresRelationshipSelection("What did Shaw write?"))
    }

    func testPlanIsBoundedNavigationNotProse() {
        let valid = #"{"question":"How does Lithuania relate to Ukraine?","need":"connection","time":"archive","queries":["Lithuania Ukraine relations"]}"#
        XCTAssertNotNil(ExplorationPlan.decode(valid))
        XCTAssertNil(ExplorationPlan.decode("Lithuania supports Ukraine"))
        XCTAssertNil(ExplorationPlan.decode(valid.replacingOccurrences(of: "connection", with: "invent")))
        XCTAssertNil(ExplorationPlan.decode(valid.replacingOccurrences(of: "[\"Lithuania Ukraine relations\"]", with: "[]")))
    }

    func testModelCannotEraseOriginalTemporalOrOpinionRequirement() throws {
        let plan = try XCTUnwrap(ExplorationPlan.decode(#"{"question":"Old cultural ties","need":"fact","time":"historical","queries":["Lithuania"]}"#))
        for q in ["How do Ukrainians feel about Lithuania now?", "How does it relate to modern geopolitics?", "What is the latest version?"] {
            XCTAssertTrue(plan.requiresTemporalCaution(originalQuestion: q))
        }
        XCTAssertTrue(plan.requiresOpinionEvidence(originalQuestion: "How do Ukrainians feel about Lithuania now?"))
        XCTAssertFalse(plan.requiresTemporalCaution(originalQuestion: "When was the duchy founded?"))
    }

    func testNamedSubjectsCannotBeReplacedByUnrelatedPollsOrGovernmentActors() throws {
        let q = "How do Ukrainians feel about Lithuania now?"
        XCTAssertFalse(ExplorationEvidence.coversNamedTerms(.init(article: "Lithuania", text: "Ukrainians supported NATO membership in a 2016 poll."), question: q))
        XCTAssertTrue(ExplorationEvidence.coversNamedTerms(.init(article: "Relations", text: "Lithuanian support for Ukraine was discussed in 2022."), question: q))
        // This tests entity coverage only; policy still forbids presenting
        // government support as proof of public opinion.
        let plan = try XCTUnwrap(ExplorationPlan.decode(#"{"question":"Geopolitics","need":"opinion","time":"current","queries":["Geopolitics"]}"#))
        XCTAssertFalse(plan.requiresOpinionEvidence(originalQuestion: "How does it relate to modern geopolitics?"))
    }

    func testSelectionReconstructsExactArchiveTextAndRejectsInvalidIDs() throws {
        let passage = SourceBoundAnswer.Passage(article: "Test", section: "Relations", library: "archive.zim",
            text: "A survey in 2019 reported mixed views. It did not measure views in 2026.")
        let windows = [ExplorationEvidence.Window(id: 0, passage: passage)]
        let result = try XCTUnwrap(ExplorationEvidence.selected(#"{"direct":[],"background":[0]}"#, windows: windows))
        XCTAssertEqual(result.background, [passage])
        for proposal in [#"{"direct":[1],"background":[]}"#, #"{"direct":[-1],"background":[]}"#,
                         #"{"direct":[0],"background":[0]}"#, #"{"direct":["0"],"background":[]}"#,
                         #"{"answer":"Everyone agrees"}"#] {
            XCTAssertNil(ExplorationEvidence.selected(proposal, windows: windows), proposal)
        }
    }
}
