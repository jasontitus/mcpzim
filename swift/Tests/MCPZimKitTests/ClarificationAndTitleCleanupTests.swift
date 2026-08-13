// SPDX-License-Identifier: MIT
//
// Regression tests for the 2026-08-03 war-of-1812 device session
// (eval/corpus/raw/5393c74a/2026-08-03_23-14-36.log), which chained four
// conversational failures:
//   1. "Tell me about the war of 1812 what were the what were the causes?"
//      → the stuttered interrogative tail stayed in the parsed title and
//      search-rescue bound the turn to "1812 Louisiana hurricane".
//   2. Mid-discussion "…died on each side in the war?" parsed a stateless
//      title of "the war", which failed the substring check against the
//      pinned "War of 1812" and LEFT the discussion (dropping a warm KV
//      cache — the P0 "reset-after-divergence every turn" symptom).
//   3. The clarification answer "The war of 1812" re-triggered the same
//      clarification, because both offered candidates contain war + 1812.
//   4. "Not the one about capital punishment" was answered as if it were
//      the question, losing the original casualty facet (host-side stash;
//      the resolver's negation pick tested here is its foundation).

import XCTest
@testable import MCPZimKit

final class ClarificationAndTitleCleanupTests: XCTestCase {

    // MARK: - collapseStutter

    func testCollapsesDoubledTrigram() {
        XCTAssertEqual(
            IntentRouter.collapseStutter(
                "the war of 1812 what were the what were the causes"),
            "the war of 1812 what were the causes")
    }

    func testCollapsesTripleRepeatAndSingleWordEcho() {
        XCTAssertEqual(
            IntentRouter.collapseStutter("the the the war"), "the war")
        XCTAssertEqual(
            IntentRouter.collapseStutter("tell me tell me tell me about rome"),
            "tell me about rome")
    }

    func testLeavesLegitimateRepetitionAloneWhenNotAdjacent() {
        // "New York, New York" style doubles ARE collapsed (adjacent), but
        // separated repeats survive.
        XCTAssertEqual(
            IntentRouter.collapseStutter("the dog chased the dog"),
            "the dog chased the dog")
    }

    // MARK: - strippingTrailingInterrogativeClause

    func testStripsTrailingQuestionClause() {
        XCTAssertEqual(
            IntentRouter.strippingTrailingInterrogativeClause(
                "the war of 1812 what were the causes"),
            "the war of 1812")
        XCTAssertEqual(
            IntentRouter.strippingTrailingInterrogativeClause(
                "the eiffel tower when was it built"),
            "the eiffel tower")
    }

    func testKeepsTitlesContainingQuestionWords() {
        // Interrogative NOT followed by an auxiliary — never cut.
        XCTAssertEqual(
            IntentRouter.strippingTrailingInterrogativeClause("doctor who"),
            "doctor who")
        XCTAssertEqual(
            IntentRouter.strippingTrailingInterrogativeClause(
                "the man who sold the world"),
            "the man who sold the world")
    }

    func testFullPatternExtractsCleanTitle() {
        // End-to-end through the stateless router: the field turn must
        // dispatch article_overview on the war, not the stuttered tail.
        let intent = IntentRouter.classify(
            "Tell me about the war of 1812 what were the what were the causes?",
            currentLocation: nil)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "the war of 1812")
    }

    // MARK: - locational prepositions in article titles

    func testLocationalPrepositionIsNotPartOfTheTitle() {
        // "what's in dupont circle?" dispatched
        // article_overview("in dupont circle") — a title no ZIM holds,
        // surviving only on search rescue. Surfaced 2026-08-13 by the
        // conversational eval's first real run.
        for (query, expected) in [
            ("what's in dupont circle?", "dupont circle"),
            ("what's around adams morgan?", "adams morgan"),
            ("what's near georgetown?", "georgetown"),
        ] {
            let intent = IntentRouter.classify(query, currentLocation: nil)
            XCTAssertEqual(intent?.anyArgs["title"] as? String, expected, query)
        }
    }

    func testExplicitCategoryStillReachesTheMap() {
        let intent = IntentRouter.classify(
            "bars in adams morgan", currentLocation: nil)
        XCTAssertEqual(intent?.toolName, "near_named_place")
        XCTAssertEqual(intent?.anyArgs["place"] as? String, "adams morgan")
    }

    // MARK: - titleNamesPinnedSubject

    func testLeadingArticleAnaphoraStaysPinned() {
        XCTAssertTrue(IntentRouter.titleNamesPinnedSubject(
            "the war", inHand: ["war of 1812"]))
        XCTAssertTrue(IntentRouter.titleNamesPinnedSubject(
            "putin", inHand: ["vladimir putin"]))
    }

    func testUnrelatedTitleStillLeaves() {
        XCTAssertFalse(IntentRouter.titleNamesPinnedSubject(
            "the art", inHand: ["stuttgart"]))
        XCTAssertFalse(IntentRouter.titleNamesPinnedSubject(
            "the french revolution", inHand: ["war of 1812"]))
    }

    // MARK: - clarificationPick

    private let warCandidates = [
        FocusEntity(name: "War of 1812", kind: .topic),
        FocusEntity(
            name: "Capital punishment by the United States military § War of 1812/Creek War",
            kind: .topic),
    ]

    func testExactTitleBeatsTokenOverlap() {
        // Field failure 3: this reply re-asked the same clarification.
        let pick = ReferenceResolver.clarificationPick(
            "The war of 1812", candidates: warCandidates)
        XCTAssertEqual(pick?.name, "War of 1812")
    }

    func testNegationExcludesNamedCandidate() {
        // Field failure 4 (resolver half).
        let pick = ReferenceResolver.clarificationPick(
            "Not the one about capital punishment", candidates: warCandidates)
        XCTAssertEqual(pick?.name, "War of 1812")
    }

    func testUniqueTokenPickStillWorks() {
        let pick = ReferenceResolver.clarificationPick(
            "the capital punishment one", candidates: warCandidates)
        XCTAssertEqual(
            pick?.name,
            "Capital punishment by the United States military § War of 1812/Creek War")
    }

    func testPositionalPick() {
        XCTAssertEqual(
            ReferenceResolver.clarificationPick(
                "the second one", candidates: warCandidates)?.name,
            warCandidates[1].name)
        XCTAssertEqual(
            ReferenceResolver.clarificationPick(
                "number 1", candidates: warCandidates)?.name,
            "War of 1812")
    }

    func testUnresolvableReplyReturnsNil() {
        XCTAssertNil(ReferenceResolver.clarificationPick(
            "what about the weather", candidates: warCandidates))
        // Ambiguous-on-purpose: matches both candidates.
        XCTAssertNil(ReferenceResolver.clarificationPick(
            "the 1812 one", candidates: warCandidates))
    }

    // MARK: - namedCandidate (ambiguity-gate self-resolution)

    func testTurnNamingCandidateVerbatimSelfResolves() {
        // Mac replay of the field session: this turn re-triggered the
        // clarification even though it contains the full title.
        let pick = ReferenceResolver.namedCandidate(
            in: "What was the cause of the war of 1812 in North America?",
            candidates: warCandidates)
        XCTAssertEqual(pick?.name, "War of 1812")
    }

    func testTurnWithoutFullNameStaysAmbiguous() {
        // "the war" alone names neither candidate fully.
        XCTAssertNil(ReferenceResolver.namedCandidate(
            in: "How many people died on each side in the war?",
            candidates: warCandidates))
    }

    func testSingleTokenNamesNeedMoreEvidence() {
        // One-token names hit inside unrelated sentences too easily.
        XCTAssertNil(ReferenceResolver.namedCandidate(
            in: "What is the capital of France?",
            candidates: [
                FocusEntity(name: "Paris", kind: .topic),
                FocusEntity(name: "London", kind: .topic),
            ]))
    }

    // MARK: - lastListKind gates comparison follow-ups

    func testDisambiguationListNeverRoutesToCompare() {
        // Mac replay 2026-08-04: the disambiguation offer left two topics
        // in lastList and "each side" then dispatched
        // compare_articles(War of 1812, French invasion of Russia).
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.setLastList([
            FocusEntity(name: "The War of 1812", kind: .topic),
            FocusEntity(name: "French invasion of Russia", kind: .topic),
        ], kind: .disambiguation)
        let intent = IntentRouter.classify(
            "How many people died on each side in the war?", focus: focus)
        XCTAssertNotEqual(intent?.toolName, "compare_articles")
    }

    func testComparisonListStillRoutesToCompare() {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.setLastList([
            FocusEntity(name: "World War I", kind: .topic),
            FocusEntity(name: "World War II", kind: .topic),
        ], kind: .comparison)
        let intent = IntentRouter.classify(
            "How many people were killed in each?", focus: focus)
        XCTAssertEqual(intent?.toolName, "compare_articles")
    }
}
