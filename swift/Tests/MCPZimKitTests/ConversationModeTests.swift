// SPDX-License-Identifier: MIT
//
// "What's in Georgetown?" is its cafés or its history, and the sentence
// alone doesn't say which. The router guesses the map, carries an
// article fallback for the host to use when the streetzim doesn't know
// the name, and honours an explicit mode when the user has stated one.
// Surfaced 2026-08-13 when the conversational eval ran for the first
// time and every neighbourhood turn had been going to Wikipedia.

import XCTest
@testable import MCPZimKit

final class ConversationModeTests: XCTestCase {

    // MARK: - Auto: guess the map, keep an article fallback

    func testLocationalTurnPrefersTheMapWithAnArticleFallback() {
        for query in ["what's in dupont circle?",
                      "what's around adams morgan?",
                      "what's near georgetown?"] {
            let intent = IntentRouter.classify(query, currentLocation: nil)
            XCTAssertEqual(intent?.toolName, "near_named_place", query)
            XCTAssertNotNil(intent?.articleFallbackTitle, query)
        }
        let dupont = IntentRouter.classify("what's in dupont circle?",
                                           currentLocation: nil)
        XCTAssertEqual(dupont?.anyArgs["place"] as? String, "dupont circle")
        XCTAssertEqual(dupont?.articleFallbackTitle, "dupont circle")
    }

    func testAbstractSubjectsNeverBecomeAPlaceGuess() {
        // Same sentence shape, not a place. These must stay articles, so
        // the user never hears "your maps don't cover a black hole".
        for query in ["what's in a black hole?",
                      "what's in the water?",
                      "what's in the news?",
                      "what's in my future?"] {
            let intent = IntentRouter.classify(query, currentLocation: nil)
            XCTAssertNotEqual(intent?.toolName, "near_named_place", query)
            XCTAssertNil(intent?.articleFallbackTitle, query)
        }
    }

    func testUnambiguousTurnsIgnoreTheMode() {
        // An explicit category is always the map…
        for mode in ConversationMode.allCases {
            let bars = IntentRouter.classify("bars in adams morgan",
                                             currentLocation: nil, mode: mode)
            XCTAssertEqual(bars?.toolName, "near_named_place", "\(mode)")
        }
        // …and a named subject is always the article.
        for mode in ConversationMode.allCases {
            let who = IntentRouter.classify("who was Napoleon",
                                            currentLocation: nil, mode: mode)
            XCTAssertEqual(who?.toolName, "article_overview", "\(mode)")
        }
    }

    // MARK: - Explicit modes settle the ambiguous middle

    func testEncyclopediaModeKeepsAmbiguousTurnsOnWikipedia() {
        let intent = IntentRouter.classify(
            "what's in georgetown?", currentLocation: nil, mode: .encyclopedia)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "georgetown")
    }

    func testLocalModeStaysOnTheMapWithNoSilentArticleFallback() {
        let intent = IntentRouter.classify(
            "what's in georgetown?", currentLocation: nil, mode: .local)
        XCTAssertEqual(intent?.toolName, "near_named_place")
        // The user said stay local: a miss should say so, not quietly
        // answer from Wikipedia.
        XCTAssertNil(intent?.articleFallbackTitle)
    }

    // MARK: - Spoken switches

    func testSpokenModeCommands() {
        XCTAssertEqual(IntentRouter.conversationModeCommand("Let's talk local"), .local)
        XCTAssertEqual(IntentRouter.conversationModeCommand("local mode"), .local)
        XCTAssertEqual(IntentRouter.conversationModeCommand("Wikipedia mode"), .encyclopedia)
        XCTAssertEqual(IntentRouter.conversationModeCommand("back to normal"), .auto)
    }

    func testOrdinaryTurnsAreNotModeCommands() {
        for query in ["what's in georgetown?",
                      "tell me about local government",
                      "what does wikipedia say about the war of 1812",
                      "let's talk about local elections in 1932"] {
            XCTAssertNil(IntentRouter.conversationModeCommand(query), query)
        }
    }

    // MARK: - place-name gate

    func testPlaceNameGate() {
        XCTAssertTrue(IntentRouter.looksLikePlaceName("dupont circle"))
        XCTAssertTrue(IntentRouter.looksLikePlaceName("san francisco"))
        XCTAssertFalse(IntentRouter.looksLikePlaceName("a black hole"))
        XCTAssertFalse(IntentRouter.looksLikePlaceName("the water"))
        XCTAssertFalse(IntentRouter.looksLikePlaceName(
            "a very long phrase that could not be a place name"))
    }
}
