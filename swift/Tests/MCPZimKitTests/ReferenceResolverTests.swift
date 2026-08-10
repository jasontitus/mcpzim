// SPDX-License-Identifier: MIT
//
// Coreference tests — the heart of the conversational redesign. These pin
// down that "it" / "the second one" / "who built it" resolve deterministically
// against the focus, so the 4B model never has to do coreference itself.

import Foundation
import XCTest

@testable import MCPZimKit

final class ReferenceResolverTests: XCTestCase {

    private func focusWithPrimary(
        _ name: String, kind: FocusEntity.Kind = .topic,
        lat: Double? = nil, lon: Double? = nil
    ) -> ConversationFocus {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: name, kind: kind, lat: lat, lon: lon))
        return f
    }

    // MARK: - Pronouns

    func testPronounBindsToPrimaryAndRewrites() {
        let f = focusWithPrimary("Stanford Memorial Church")
        let r = ReferenceResolver.resolve("who built it?", focus: f)
        XCTAssertTrue(r.isContinuation)
        XCTAssertEqual(r.boundEntity?.name, "Stanford Memorial Church")
        XCTAssertEqual(r.rewrittenQuery, "who built Stanford Memorial Church")
    }

    func testThatBindsToPrimary() {
        let f = focusWithPrimary("Marie Curie")
        let r = ReferenceResolver.resolve("when did that happen", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "Marie Curie")
        XCTAssertEqual(r.rewrittenQuery, "when did Marie Curie happen")
    }

    // MARK: - Elliptical subjectless follow-ups

    func testEllipticalHowOldAppendsSubject() {
        let f = focusWithPrimary("the Colosseum")
        let r = ReferenceResolver.resolve("how old", focus: f)
        XCTAssertTrue(r.isContinuation)
        XCTAssertEqual(r.boundEntity?.name, "the Colosseum")
        XCTAssertEqual(r.rewrittenQuery, "how old is the Colosseum")
    }

    func testBareMoreBecomesTellMeMore() {
        let f = focusWithPrimary("Pizza")
        let r = ReferenceResolver.resolve("tell me more", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "Pizza")
        XCTAssertEqual(r.rewrittenQuery, "tell me more about Pizza")
    }

    // MARK: - Fresh queries must NOT bind

    func testFreshProperSubjectDoesNotBind() {
        let f = focusWithPrimary("Pizza")
        let r = ReferenceResolver.resolve("tell me about quantum tunnelling", focus: f)
        XCTAssertNil(r.boundEntity, "a fresh subject must not hijack the focus")
    }

    func testEmptyFocusNeverBinds() {
        let f = ConversationFocus()
        let r = ReferenceResolver.resolve("how old", focus: f)
        XCTAssertNil(r.boundEntity)
    }

    // MARK: - List selection

    private func focusWithList(_ names: [String]) -> ConversationFocus {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.setLastList(names.map { FocusEntity(name: $0, kind: .place) })
        return f
    }

    func testOrdinalSelectsListSlot() {
        let f = focusWithList(["Old North Church", "Trinity Church", "King's Chapel"])
        let r = ReferenceResolver.resolve("tell me about the second one", focus: f)
        if case .listSelection(let idx, let e) = r.binding {
            XCTAssertEqual(idx, 1)
            XCTAssertEqual(e.name, "Trinity Church")
        } else {
            XCTFail("expected a list selection, got \(r.binding)")
        }
    }

    func testOrdinalInOrdinaryProseDoesNotSelectListSlot() {
        let f = focusWithList(["Old North Church", "Trinity Church"])
        let r = ReferenceResolver.resolve(
            "first of all, tell me about architecture", focus: f)
        if case .listSelection = r.binding {
            XCTFail("ordinary prose beginning with an ordinal must not pick a list item")
        }
    }

    func testTheOtherOneInTwoItemList() {
        let f = focusWithList(["North Korea", "South Korea"])
        // primaryEntity is the list head (North Korea); "the other" → South.
        let r = ReferenceResolver.resolve("what about the other one", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "South Korea")
    }

    func testTheOtherRequiresPrimaryToBelongToList() {
        var f = focusWithList(["North Korea", "South Korea"])
        f.remember(FocusEntity(name: "Japan", kind: .place))
        let r = ReferenceResolver.resolve("what about the other one", focus: f)
        if case .listSelection = r.binding {
            XCTFail("the other must not guess a list slot when its anchor is outside the list")
        }
    }

    func testTheLastOne() {
        let f = focusWithList(["A Cafe", "B Cafe", "C Cafe"])
        let r = ReferenceResolver.resolve("directions to the last one", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "C Cafe")
    }

    func testDescriptiveNounSelector() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.setLastList([
            FocusEntity(name: "City Museum", kind: .place),
            FocusEntity(name: "Grace Cathedral", kind: .place),
        ])
        // "the cathedral" uniquely matches one item.
        let r = ReferenceResolver.resolve("tell me about the cathedral", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "Grace Cathedral")
    }

    func testAmbiguousDescriptiveIsFlagged() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.setLastList([
            FocusEntity(name: "Old North Church", kind: .place),
            FocusEntity(name: "Trinity Church", kind: .place),
        ])
        let r = ReferenceResolver.resolve("what about the church", focus: f)
        if case .ambiguous(let xs) = r.binding {
            XCTAssertEqual(xs.count, 2)
        } else {
            XCTFail("expected ambiguity, got \(r.binding)")
        }
    }

    // MARK: - Drift-thread acceptance ("yes" / "the war")

    private func focusWithThreads(
        primary: String, _ threads: [DiscoveryThread]
    ) -> ConversationFocus {
        var f = ConversationFocus()
        f.beginUserTurn()
        if !primary.isEmpty { f.remember(FocusEntity(name: primary, kind: .topic)) }
        f.setThreads(threads)
        return f
    }

    func testBareYesAcceptsLeadThread() {
        let f = focusWithThreads(primary: "Stanford Memorial Church", [
            DiscoveryThread(label: "Stanford White", kind: .topic, source: .wikilink),
            DiscoveryThread(label: "World War II", kind: .topic, source: .wikilink),
        ])
        let r = ReferenceResolver.resolve("yes", focus: f)
        XCTAssertTrue(r.isContinuation)
        XCTAssertEqual(r.boundEntity?.name, "Stanford White",
            "a bare affirmative accepts the lead drift thread")
        if case .thread = r.binding {} else {
            XCTFail("expected a .thread binding, got \(r.binding)")
        }
    }

    func testNamedThreadBindsByLabel() {
        let f = focusWithThreads(primary: "Stanford Memorial Church", [
            DiscoveryThread(label: "Stanford White", kind: .topic, source: .wikilink),
            DiscoveryThread(label: "World War II", kind: .topic, source: .wikilink),
        ])
        let r = ReferenceResolver.resolve("tell me about the war", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "World War II",
            "'the war' uniquely names the WWII thread")
    }

    func testBareYesWithNoOfferDoesNotBindThread() {
        // No threads open: "yes" has nothing to accept and must fall through
        // (not get rewritten into a nonsense "yes <subject>" continuation).
        let f = focusWithPrimary("Pizza")
        let r = ReferenceResolver.resolve("yes", focus: f)
        if case .thread = r.binding {
            XCTFail("no offer open — 'yes' must not bind a thread")
        }
    }

    func testEllipticalStemDoesNotMatchThreadLabel() {
        // "how old?" must bind the SUBJECT, not a thread whose label merely
        // contains "old" — the stopword/length filter in matchOpenThread is
        // what prevents that false match.
        let f = focusWithThreads(primary: "the Colosseum", [
            DiscoveryThread(label: "Old North Church", kind: .topic, source: .wikilink),
        ])
        let r = ReferenceResolver.resolve("how old", focus: f)
        XCTAssertEqual(r.boundEntity?.name, "the Colosseum",
            "an elliptical stem binds the subject, not a label-overlap thread")
    }
}
