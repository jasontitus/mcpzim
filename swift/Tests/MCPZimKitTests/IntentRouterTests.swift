// SPDX-License-Identifier: MIT
//
// Behavioural tests for the fast-path intent router + reply
// synthesiser. Covers the patterns the iOS chat surface will actually
// send and the guards that keep questions from getting mis-routed as
// places searches.

import Foundation
import XCTest

@testable import MCPZimKit

final class IntentRouterTests: XCTestCase {

    // MARK: - `<category> in <place>`

    func testClassifyCategoryInPlace() {
        let i = IntentRouter.classify("bars in North Beach")
        XCTAssertEqual(i?.toolName, "near_named_place")
        XCTAssertEqual(i?.args["place"], .string("north beach"))
        XCTAssertEqual(i?.args["kinds"], .array([.string("bar")]))
        XCTAssertEqual(i?.args["radius_km"], .double(5))
    }

    func testClassifyPluralSingularisation() {
        // Regular -s plural: museums → museum.
        XCTAssertEqual(
            IntentRouter.classify("museums in Mountain View")?.args["kinds"],
            .array([.string("museum")])
        )
        // -ies plural: libraries → library (not "librarie").
        XCTAssertEqual(
            IntentRouter.classify("libraries in Mountain View")?.args["kinds"],
            .array([.string("library")])
        )
        // -ches plural: churches → church.
        XCTAssertEqual(
            IntentRouter.classify("churches in Mountain View")?.args["kinds"],
            .array([.string("church")])
        )
        // Already singular: cafe stays cafe (above 3-char guard).
        XCTAssertEqual(
            IntentRouter.classify("cafe in Mountain View")?.args["kinds"],
            .array([.string("cafe")])
        )
    }

    func testClassifyHandlesPrepositionVariants() {
        for preposition in ["in", "near", "around", "at"] {
            let q = "cafes \(preposition) Palo Alto"
            let i = IntentRouter.classify(q)
            XCTAssertEqual(i?.toolName, "near_named_place", "variant: \(q)")
            XCTAssertEqual(i?.args["place"], .string("palo alto"), "variant: \(q)")
        }
    }

    func testClassifyTrimsWhitespaceAndPunctuation() {
        let i = IntentRouter.classify("  bars in San Francisco?  ")
        XCTAssertEqual(i?.args["place"], .string("san francisco"))
    }

    // MARK: - "continue" / "keep reading" paging

    func testContinueReadingPositives() {
        // LITERAL "keep reading the article" verbs → article paging.
        for q in [
            "continue", "Continue.", "continue reading", "keep reading",
            "read on", "read more", "read me more", "next section",
            "next page", "resume reading",
            "please continue", "can you keep reading", "and continue",
        ] {
            XCTAssertTrue(IntentRouter.isContinueReading(q),
                          "should be a reading-continue: \(q)")
        }
    }

    func testOpenEndedFollowupsDeferToContinuation() {
        // Open-ended follow-ups belong to the focus-aware continuationIntent
        // (re-open the subject + offer drift threads), NOT article paging —
        // so isContinueReading says false and lets them fall through to
        // classify. Split per CONVERSATIONAL_REDESIGN.md so the two features
        // stop fighting over "tell me more" / "more" / "go on".
        for q in [
            "tell me more", "more", "more please", "go on", "what else",
            "and then", "say more", "keep going", "go ahead", "keep talking",
        ] {
            XCTAssertFalse(IntentRouter.isContinueReading(q),
                           "open-ended follow-up should defer to continuationIntent: \(q)")
        }
    }

    func testContinueReadingNegatives() {
        // Real queries that must NOT be swallowed as "continue" — they
        // carry their own intent and have to reach classify / the LLM.
        for q in [
            "tell me about the grand duchy of lithuania",
            "more coffee shops near me",
            "what is the capital of France",
            "directions to Philz Coffee",
            "compare north and south korea",
            "read me the article about Paris",
            "bars in North Beach",
            "where am I",
            "",
        ] {
            XCTAssertFalse(IntentRouter.isContinueReading(q),
                           "should NOT be a reading-continue: \(q)")
        }
    }

    // MARK: - Corrections / restatements (ASR mishear recovery)

    func testCorrectionRoutesToArticle() {
        // The real 2026-05-29 capture: "grand Duchy of Lithuania" heard
        // as "Dutch Lithuania"; the user restated and the model reused
        // its wrong guess. The restatement must route the CORRECTED words
        // straight to article_overview.
        let i = IntentRouter.classify(
            "I was actually talking about the grand Duchy of Lithuania")
        XCTAssertEqual(i?.toolName, "article_overview")
        XCTAssertEqual(i?.args["title"], .string("the grand duchy of lithuania"))
    }

    func testCorrectionVariantsRouteToArticle() {
        for q in [
            "I meant Napoleon Bonaparte",
            "no, I said Napoleon Bonaparte",
            "I was referring to Napoleon Bonaparte",
            "actually I meant Napoleon Bonaparte",
            "I'm talking about Napoleon Bonaparte",
            "I meant tell me about Napoleon Bonaparte",
        ] {
            let i = IntentRouter.classify(q)
            XCTAssertEqual(i?.toolName, "article_overview", "variant: \(q)")
            XCTAssertEqual(i?.args["title"], .string("napoleon bonaparte"),
                           "variant: \(q)")
        }
    }

    func testCorrectionPreservesOtherIntents() {
        // A correction that restates a DIFFERENT intent must re-route to
        // that intent, not be forced into an article lookup.
        XCTAssertEqual(
            IntentRouter.classify("I meant directions to Philz Coffee")?.toolName,
            "route_from_places")
        XCTAssertEqual(
            IntentRouter.classify("I was actually talking about bars in North Beach")?.toolName,
            "near_named_place")
    }

    // MARK: - Article miss → did-you-mean (no confabulation)

    func testArticleMissReplyWithSuggestions() {
        let reply = IntentRouter.synthesizeArticleMissReply(
            args: ["title": "Dutch Lithuania"],
            fullResult: [
                "requested_title": "Dutch Lithuania",
                "suggestions": ["Grand Duchy of Lithuania", "Lithuania"],
            ])
        XCTAssertTrue(reply.contains("couldn't find"))
        XCTAssertTrue(reply.contains("Dutch Lithuania"))
        XCTAssertTrue(reply.contains("Grand Duchy of Lithuania"))
        // Must NOT invent / describe the missed entity.
        XCTAssertFalse(reply.lowercased().contains("republic"))
    }

    func testArticleMissReplyNoSuggestions() {
        let reply = IntentRouter.synthesizeArticleMissReply(
            args: ["title": "Zxqw Nonsense"],
            fullResult: ["requested_title": "Zxqw Nonsense", "suggestions": [String]()])
        XCTAssertTrue(reply.contains("couldn't find"))
        XCTAssertTrue(reply.contains("different way"))
    }

    func testDidYouMeanFiltersFullTextNoise() {
        // The real failure: a keyword search for the mis-heard "Dutch
        // Lithuania" surfaced an unrelated song. The overlap filter must
        // keep the real shared-token title and drop the noise.
        let candidates = [
            SearchHitResult(zim: "w", kind: .wikipedia, path: "a",
                            title: "Black Friday (Tom Odell song)", snippet: ""),
            SearchHitResult(zim: "w", kind: .wikipedia, path: "b",
                            title: "Grand Duchy of Lithuania", snippet: ""),
        ]
        let out = MCPToolAdapter.didYouMeanTitles(
            requested: "the grand dutch lithuania", candidates: candidates, limit: 3)
        XCTAssertEqual(out, ["Grand Duchy of Lithuania"])
    }

    func testDidYouMeanMatchesMorphologicalVariants() {
        // Debug report 563C79D3: "Einstein's gravity waves" missed
        // "Gravitational wave" because the filter required an exact shared
        // word ("gravity" ≠ "gravitational", "waves" ≠ "wave"). Stem/prefix
        // matching should bridge it while still dropping unrelated noise.
        let candidates = [
            SearchHitResult(zim: "w", kind: .wikipedia, path: "a",
                            title: "Gravitational wave", snippet: ""),
            SearchHitResult(zim: "w", kind: .wikipedia, path: "b",
                            title: "Black Friday (song)", snippet: ""),
        ]
        let out = MCPToolAdapter.didYouMeanTitles(
            requested: "einstein’s gravity waves", candidates: candidates, limit: 3)
        XCTAssertEqual(out, ["Gravitational wave"])
    }

    // MARK: - "nearest / closest / nearby <category>" (GPS-anchored)

    func testNearestCategoryRoutesToNearPlaces() {
        // Regression for debug report 563C79D3: "Where is the nearest coffee
        // shop?" matched no pattern, fell to the LLM, and the model asked for
        // location despite the GPS preamble. Must resolve deterministically.
        let here = (lat: 37.44, lon: -122.15)
        for q in [
            "where is the nearest coffee shop",
            "where's the nearest coffee shop",
            "nearest coffee shop",
            "the closest coffee shop",
            "where is the nearest coffee shop to me",
            "coffee shop nearby",
            "is there a coffee shop nearby",
            "find me the nearest coffee shop",
        ] {
            let i = IntentRouter.classify(q, currentLocation: here)
            XCTAssertEqual(i?.toolName, "near_places", "variant: \(q)")
            XCTAssertEqual(i?.args["kinds"], .array([.string("coffee shop")]),
                           "variant: \(q)")
            XCTAssertEqual(i?.args["lat"], .double(37.44), "variant: \(q)")
        }
    }

    func testNearestPluralSingularised() {
        let here = (lat: 1.0, lon: 2.0)
        XCTAssertEqual(
            IntentRouter.classify("nearest pharmacies", currentLocation: here)?.args["kinds"],
            .array([.string("pharmacy")]))
        XCTAssertEqual(
            IntentRouter.classify("any restaurants nearby", currentLocation: here)?.args["kinds"],
            .array([.string("restaurant")]))
    }

    func testNearestWithoutLocationBailsToLLM() {
        // No GPS fix → must not guess "me"/"here" as a place; fall through
        // (nil) so the LLM, which carries the coords in its preamble, handles it.
        XCTAssertNil(IntentRouter.classify("nearest coffee shop", currentLocation: nil))
    }

    func testNearbyQuestionWordNotBoundAsCategory() {
        // "what's nearby" must never become near_places(kinds=["what's"]).
        let here = (lat: 1.0, lon: 2.0)
        for q in ["what's nearby", "what is nearby", "anything nearby"] {
            let i = IntentRouter.classify(q, currentLocation: here)
            XCTAssertNotEqual(i?.args["kinds"], .array([.string("what's")]),
                              "variant: \(q)")
        }
    }

    // MARK: - "let's discuss X" → discuss_article

    func testDiscussRoutesToDiscussArticle() {
        for q in [
            "let's discuss the Grand Duchy of Lithuania",
            "lets discuss the Grand Duchy of Lithuania",
            "let's talk about the Grand Duchy of Lithuania",
            "discuss the Grand Duchy of Lithuania",
            "can we discuss the Grand Duchy of Lithuania",
            "I want to talk about the Grand Duchy of Lithuania",
            "dig into the Grand Duchy of Lithuania",
        ] {
            let i = IntentRouter.classify(q)
            XCTAssertEqual(i?.toolName, "discuss_article", "variant: \(q)")
            XCTAssertEqual(i?.args["title"],
                           .string("the grand duchy of lithuania"), "variant: \(q)")
        }
    }

    func testDiscussBarePronounNotRouted() {
        // "discuss it" has no explicit subject — leave it to the LLM/focus,
        // not a discuss_article("it") lookup that would miss.
        XCTAssertNotEqual(
            IntentRouter.classify("let's discuss it")?.toolName, "discuss_article")
        XCTAssertNotEqual(
            IntentRouter.classify("discuss this")?.toolName, "discuss_article")
    }

    func testHowDoesItWorkRoutesToArticle() {
        for (q, title) in [
            ("how do combustion engines work", "combustion engine"),
            ("how does a solar panel work", "solar panel"),
            ("how do vaccines work?", "vaccine"),
            ("how does the internal combustion engine work", "internal combustion engine"),
        ] {
            let i = IntentRouter.classify(q)
            XCTAssertEqual(i?.toolName, "article_overview", "variant: \(q)")
            XCTAssertEqual(i?.args["title"], .string(title), "variant: \(q)")
        }
        // "how do I get to X" stays directions (ends in a place, not "work").
        XCTAssertNotEqual(
            IntentRouter.classify("how do I get to the airport")?.toolName,
            "article_overview")
    }

    func testHowFarRoutesToRoute() {
        for q in [
            "how far is it to the airport",
            "how far to San Jose",
            "how far away is Stanford",
            "distance to the ferry building",
            "how long to drive to Oakland",
            "how long does it take to get to the airport",
        ] {
            let i = IntentRouter.classify(q)
            XCTAssertEqual(i?.toolName, "route_from_places", "variant: \(q)")
            XCTAssertEqual(i?.args["origin"], .string("my location"), "variant: \(q)")
            XCTAssertFalse((i?.args["destination"].map {
                if case .string(let s) = $0 { return s.isEmpty } else { return true }
            }) ?? true, "destination should be non-empty: \(q)")
        }
        // Bare "how far is it" (no destination) is a focus continuation,
        // not a standalone route.
        XCTAssertNotEqual(
            IntentRouter.classify("how far is it")?.toolName, "route_from_places")
    }

    // MARK: - Focus-bound follow-ups (continuationIntent — review P1s)

    private func focusWithPlace(
        _ name: String, lat: Double, lon: Double
    ) -> ConversationFocus {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: name, kind: .place, lat: lat, lon: lon))
        return f
    }

    func testContinuationTellMeMoreReopensPlaceArticle() {
        // After "where am I?" records the place (P1b), "tell me more about it"
        // re-opens its article — no travel/proximity cue, so article_overview.
        let f = focusWithPlace("San Francisco", lat: 37.7793, lon: -122.4193)
        let i = IntentRouter.continuationIntent("tell me more about it", focus: f)
        XCTAssertEqual(i?.toolName, "article_overview")
        XCTAssertEqual(i?.args["title"], .string("San Francisco"))
    }

    func testContinuationWhatsNearThereUsesRememberedCoords() {
        // "what's near there?" against a remembered place routes to near_places
        // AT its coords — proves a routed destination (P1c) or a what_is_here
        // place (P1b) carried its coordinates into focus, and that anaphoric
        // "there" binds (it isn't a pronoun, so this needs the locative rule).
        let f = focusWithPlace("Ferry Building", lat: 37.7955, lon: -122.3937)
        let i = IntentRouter.continuationIntent("what's near there?", focus: f)
        XCTAssertEqual(i?.toolName, "near_places")
        XCTAssertEqual(i?.args["lat"], .double(37.7955))
        XCTAssertEqual(i?.args["lon"], .double(-122.3937))
    }

    func testContinuationYesAcceptsOfferedThread() {
        // A bare "yes" after an offer re-opens the lead drift thread (P1a).
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Colosseum", kind: .topic))
        f.setThreads([
            DiscoveryThread(label: "Ancient Rome", kind: .topic, source: .wikilink),
            DiscoveryThread(label: "Vespasian", kind: .topic, source: .wikilink),
        ])
        let i = IntentRouter.continuationIntent("yes", focus: f)
        XCTAssertEqual(i?.toolName, "article_overview")
        XCTAssertEqual(i?.args["title"], .string("Ancient Rome"))
    }

    func testClassifyQuestionsAreNotPlaces() {
        // Don't misclassify "how does X work" or "where can I find Y"
        // as places searches. Some of these ("tell me about volcanoes
        // in Hawaii", "what is the fish in the sea") now route to
        // article_overview — that's the intended broadening; the
        // article lookup gracefully falls back to a no-match caption
        // instead of paying the 15 s LLM prefill.
        for q in [
            "how does rain form in clouds",
            "where can I find help in SF",
            "what is the fish in the sea",
            "tell me about volcanoes in Hawaii",
            "can you find bars in North Beach",
        ] {
            XCTAssertNotEqual(
                IntentRouter.classify(q)?.toolName, "near_named_place",
                "misclassified as places: \(q)")
        }
    }

    // MARK: - `<category> near me`

    func testClassifyNearMeRequiresLocation() {
        let i = IntentRouter.classify("bars near me")
        XCTAssertNil(i, "no location → fast-path should decline")
    }

    func testClassifyNearMeWithLocation() {
        let here = (lat: 37.441, lon: -122.155)
        let i = IntentRouter.classify("bars near me", currentLocation: here)
        XCTAssertEqual(i?.toolName, "near_places")
        XCTAssertEqual(i?.args["lat"], .double(37.441))
        XCTAssertEqual(i?.args["lon"], .double(-122.155))
        XCTAssertEqual(i?.args["kinds"], .array([.string("bar")]))
    }

    func testClassifyNearMeAroundHere() {
        let here = (lat: 1.0, lon: 2.0)
        let a = IntentRouter.classify("bars near me", currentLocation: here)
        let b = IntentRouter.classify("bars around here", currentLocation: here)
        XCTAssertEqual(a?.toolName, "near_places")
        XCTAssertEqual(b?.toolName, "near_places")
    }

    // MARK: - `directions to <place>`

    func testClassifyDirections() {
        let cases = [
            ("directions to San Francisco",    "san francisco"),
            ("direction to SFO",               "sfo"),
            ("route to Fenway Park",           "fenway park"),
            ("navigate to Union Square",       "union square"),
            ("how do i get to the museum",     "the museum"),
            ("how to get to City Hall",        "city hall"),
            ("how get to Central Park",        "central park"),  // no "do i"
            ("take me to Golden Gate",         "golden gate"),
        ]
        for (query, expectedDest) in cases {
            let i = IntentRouter.classify(query)
            XCTAssertEqual(i?.toolName, "route_from_places", "query: \(query)")
            XCTAssertEqual(i?.args["origin"], .string("my location"),
                           "query: \(query)")
            XCTAssertEqual(i?.args["destination"], .string(expectedDest),
                           "query: \(query)")
        }
    }

    func testClassifyDirectionsWithPolitePrefix() {
        // Voice input routinely prepends "give me", "show me",
        // "can you", "please". Before this change these fell to the
        // LLM, which sometimes emitted malformed JSON (double commas)
        // and silently dropped the turn — see dropped-request.log.
        let cases: [(String, String)] = [
            ("give me directions to San Francisco",      "san francisco"),
            ("show me directions to Palo Alto",          "palo alto"),
            ("get me directions to the museum",          "the museum"),
            ("find me directions to Union Square",       "union square"),
            ("please give me directions to SFO",         "sfo"),
            ("can you give me directions to SF",         "sf"),
            ("can you show me the route to Fenway",      "fenway"),
            ("could you show me directions for SFO",     "sfo"),
            ("I need directions to Central Park",        "central park"),
            ("I want directions to the museum",          "the museum"),
            ("I'd like directions to San Jose",          "san jose"),
            ("please take me to Golden Gate",            "golden gate"),
        ]
        for (query, expectedDest) in cases {
            let i = IntentRouter.classify(query)
            XCTAssertEqual(i?.toolName, "route_from_places", "query: \(query)")
            XCTAssertEqual(i?.args["destination"], .string(expectedDest),
                           "query: \(query)")
        }
    }

    func testDirectionsTakesPrecedenceOverInOrNear() {
        // "Library in Mountain View" — destinations can contain "in" or
        // "near"; the directions pattern needs to win so we don't
        // accidentally hit near_named_place(place="mountain view",
        // kinds=["library"]) for a routing intent.
        let i = IntentRouter.classify("directions to Library in Mountain View")
        XCTAssertEqual(i?.toolName, "route_from_places")
        XCTAssertEqual(i?.args["destination"],
                       .string("library in mountain view"))
    }

    // MARK: - `compare <A> and <B>`

    func testClassifyCompareTwoEntities() {
        let cases: [(String, String, String)] = [
            ("compare North Korea and South Korea", "north korea", "south korea"),
            ("compare Apple and Google",            "apple",       "google"),
            ("compare Tesla vs Ford",               "tesla",       "ford"),
            ("compare Tesla vs. Ford",              "tesla",       "ford"),
            ("compare Python versus Rust",          "python",      "rust"),
            ("compare Swift with Kotlin",           "swift",       "kotlin"),
            ("compare mercury to venus",            "mercury",     "venus"),
        ]
        for (query, a, b) in cases {
            let i = IntentRouter.classify(query)
            XCTAssertEqual(i?.toolName, "compare_articles", "query: \(query)")
            XCTAssertEqual(i?.args["titles"],
                           .array([.string(a), .string(b)]),
                           "query: \(query)")
        }
    }

    func testComparisonCasualtyFollowUpAlignsBothArticles() {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.setLastList([
            FocusEntity(name: "World War I", kind: .topic),
            FocusEntity(name: "World War II", kind: .topic),
        ])

        let intent = IntentRouter.classify(
            "How many people were killed in each?", focus: focus)
        XCTAssertEqual(intent?.toolName, "compare_articles")
        XCTAssertEqual(
            intent?.args["titles"],
            .array([.string("World War I"), .string("World War II")]))
        XCTAssertEqual(intent?.args["section"], .string("Casualties"))
    }

    func testComparisonInterpretiveFollowUpStaysSynthesisOnly() {
        var focus = ConversationFocus()
        focus.beginUserTurn()
        focus.setLastList([
            FocusEntity(name: "World War I", kind: .topic),
            FocusEntity(name: "World War II", kind: .topic),
        ])

        XCTAssertNil(IntentRouter.classify(
            "What changed between the two that made WWII so much more deadly?",
            focus: focus))
    }

    func testCompareExpandsSharedSuffix() {
        // "Compare north and south korea" literally parses as
        // ["north", "south korea"], but the intended English reading
        // is ["north korea", "south korea"] — "Korea" is a shared
        // suffix the speaker dropped from the first half. Exactly
        // this failure appeared on device (fast-path fired, but
        // dispatched with the wrong titles; tool returned empty; UI
        // showed "Comparing north and south korea. Results below."
        // with nothing below). Heuristic re-attaches the last word
        // of the second title only when the first is a single
        // directional/ordinal word.
        let cases: [(String, String, String)] = [
            ("compare north and south korea",    "north korea",    "south korea"),
            ("compare east and west germany",    "east germany",   "west germany"),
            ("compare northern and southern ireland",
                                                  "northern ireland", "southern ireland"),
            ("compare old and new testament",    "old testament",  "new testament"),
            ("compare big and little league",    "big league",     "little league"),
        ]
        for (query, a, b) in cases {
            let i = IntentRouter.classify(query)
            XCTAssertEqual(i?.toolName, "compare_articles", "query: \(query)")
            XCTAssertEqual(i?.args["titles"],
                           .array([.string(a), .string(b)]),
                           "query: \(query)")
        }
    }

    func testCompareDoesNotExpandNonDirectionalSingleWord() {
        // "Apple", "cats", "python" aren't directional/ordinal —
        // don't append the last word of the second title, it'd
        // be a lossy transformation.
        let cases: [(String, String, String)] = [
            ("compare apple and google maps",    "apple",   "google maps"),
            ("compare cats and dogs",            "cats",    "dogs"),
            ("compare python vs rust lang",      "python",  "rust lang"),
        ]
        for (query, a, b) in cases {
            let i = IntentRouter.classify(query)
            XCTAssertEqual(i?.args["titles"],
                           .array([.string(a), .string(b)]),
                           "query: \(query)")
        }
    }

    func testCompareDoesNotDoubleAppendWhenSpeakerWasExplicit() {
        // "compare north korea and south korea" — first half is
        // already "north korea" (2 words), so the heuristic's
        // aWords.count == 1 guard keeps us from appending anything.
        let i = IntentRouter.classify("compare north korea and south korea")
        XCTAssertEqual(i?.args["titles"],
                       .array([.string("north korea"), .string("south korea")]))
    }

    func testCompareIgnoresSingleEntity() {
        // `compare foo` with no connector shouldn't fake a second entity.
        XCTAssertNil(IntentRouter.classify("compare physics"))
        XCTAssertNil(IntentRouter.classify("compare things"))
    }

    // MARK: - `tell me about X` / `what is X` / `who is X`

    func testClassifyArticleOverview() {
        let cases: [(String, String)] = [
            ("tell me about Palo Alto",           "palo alto"),
            ("tell me more about Newton's laws",  "newton's laws"),
            ("what is aspirin",                   "aspirin"),
            ("what's gravity",                    "gravity"),
            ("what is the Eiffel Tower",          "the eiffel tower"),
            ("who is Marie Curie",                "marie curie"),
            ("who was Alan Turing",               "alan turing"),
            ("give me an overview of quantum mechanics",
                                                  "quantum mechanics"),
            ("overview of jazz",                  "jazz"),
        ]
        for (query, subject) in cases {
            let i = IntentRouter.classify(query)
            XCTAssertEqual(i?.toolName, "article_overview", "query: \(query)")
            XCTAssertEqual(i?.args["title"], .string(subject),
                           "query: \(query)")
        }
    }

    func testArticleOverviewSkipsNavigationalPronouns() {
        // "what is my next turn" is a route-status question, not an
        // article lookup. Falling through to the LLM is correct.
        let queries = [
            "what is my next turn",
            "what is this",
            "who was here",
            "what is it",
            "what's my ETA",
        ]
        for q in queries {
            let i = IntentRouter.classify(q)
            XCTAssertNotEqual(i?.toolName, "article_overview",
                              "should not route to article_overview: \(q)")
        }
    }

    // MARK: - `what is here` / `where am I`

    func testClassifyWhatIsHere() {
        let queries = [
            "what is here",
            "what's here",
            "where am I",
            "where am i",
            "what is around me",
            "what's around me",
            "what is near me",
            "what's near here",
            "what do you see",
        ]
        for q in queries {
            let i = IntentRouter.classify(q)
            XCTAssertEqual(i?.toolName, "what_is_here", "query: \(q)")
            XCTAssertTrue(i?.args.isEmpty ?? false,
                          "what_is_here takes no args from router: \(q)")
        }
    }

    // MARK: - Reply synthesis (article_overview / compare / what_is_here)

    func testSynthesizeArticleOverviewUsesLeadSentence() {
        let fullResult: [String: Any] = [
            "title": "Palo Alto",
            "sections": [
                [
                    "title": "lead",
                    "text": "Palo Alto is a charter city in Santa Clara "
                        + "County. It hosts Stanford University. The city "
                        + "was founded in 1894."
                ]
            ]
        ]
        let s = IntentRouter.synthesizeArticleOverviewReply(
            args: ["title": "Palo Alto"], fullResult: fullResult
        )
        XCTAssertTrue(s.contains("Palo Alto is a charter city"),
                      "synth should open with the lead first sentence")
    }

    func testSynthesizeArticleOverviewHandlesMissingArticle() {
        let fullResult: [String: Any] = [
            "error": "No article titled 'Xyz' found in the loaded ZIMs."
        ]
        let s = IntentRouter.synthesizeArticleOverviewReply(
            args: ["title": "Xyz"], fullResult: fullResult
        )
        XCTAssertTrue(s.contains("Xyz"),
                      "error path should name the requested title")
    }

    func testSynthesizeCompareUsesEachArticlesLead() {
        let fullResult: [String: Any] = [
            "articles": [
                [
                    "title": "North Korea",
                    "sections": [[
                        "text": "North Korea is a country in East Asia. "
                            + "Its capital is Pyongyang."
                    ]]
                ],
                [
                    "title": "South Korea",
                    "sections": [[
                        "text": "South Korea is a country in East Asia. "
                            + "Its capital is Seoul."
                    ]]
                ],
            ]
        ]
        let s = IntentRouter.synthesizeCompareReply(
            args: ["titles": ["North Korea", "South Korea"]],
            fullResult: fullResult
        )
        XCTAssertTrue(s.contains("North Korea"))
        XCTAssertTrue(s.contains("South Korea"))
        XCTAssertTrue(s.contains("Pyongyang") || s.contains("East Asia"))
    }

    // MARK: - Fast-path usability gating

    func testCompareResultIsUsableForRelationsArticle() {
        let ok: [String: Any] = [
            "strategy": "dedicated_relations_article",
            "sections": [["text": "some real content."]]
        ]
        XCTAssertTrue(IntentRouter.compareResultIsUsable(ok))
    }

    func testCompareResultIsUsableForTwoGoodArticles() {
        let ok: [String: Any] = [
            "articles": [
                ["title": "A", "sections": [["text": "lead A"]]],
                ["title": "B", "sections": [["text": "lead B"]]],
            ]
        ]
        XCTAssertTrue(IntentRouter.compareResultIsUsable(ok))
    }

    func testCompareResultNotUsableWhenBothErrored() {
        // This is the dropped-request scenario: tool dispatched OK,
        // returned 200 bytes, but both articles came back with the
        // "Could not fetch" error. The LLM should get a shot.
        let bad: [String: Any] = [
            "articles": [
                ["title": "north", "error": "Could not fetch"],
                ["title": "south korea", "error": "Could not fetch"],
            ]
        ]
        XCTAssertFalse(IntentRouter.compareResultIsUsable(bad))
    }

    func testCompareResultNotUsableWhenOneErroredOneGood() {
        // One-sided "comparison" isn't a comparison — LLM retry is
        // more useful than rendering a lone article as if both
        // subjects matched.
        let partial: [String: Any] = [
            "articles": [
                ["title": "Xyz", "error": "Could not fetch"],
                ["title": "Real", "sections": [["text": "lead"]]],
            ]
        ]
        XCTAssertFalse(IntentRouter.compareResultIsUsable(partial))
    }

    func testCompareResultNotUsableOnTopLevelError() {
        let e: [String: Any] = [
            "error": "compare_articles needs at least two non-empty titles."
        ]
        XCTAssertFalse(IntentRouter.compareResultIsUsable(e))
    }

    func testArticleOverviewUsabilityChecks() {
        let ok: [String: Any] = [
            "title": "Palo Alto",
            "sections": [["text": "Palo Alto is a charter city…"]]
        ]
        let missing: [String: Any] = [
            "error": "title 'Xyz' not found in any Wikipedia ZIM"
        ]
        let emptySections: [String: Any] = [
            "title": "Foo",
            "sections": []
        ]
        XCTAssertTrue(IntentRouter.articleOverviewResultIsUsable(ok))
        XCTAssertFalse(IntentRouter.articleOverviewResultIsUsable(missing))
        XCTAssertFalse(IntentRouter.articleOverviewResultIsUsable(emptySections))
    }

    func testWhatIsHereUsabilityChecks() {
        let ok: [String: Any] = ["nearest_named_place": "Civic Center"]
        let noPlace: [String: Any] = ["nearest_named_place": ""]
        let missing: [String: Any] = [
            "error": "No named place within 1.5 km."
        ]
        XCTAssertTrue(IntentRouter.whatIsHereResultIsUsable(ok))
        XCTAssertFalse(IntentRouter.whatIsHereResultIsUsable(noPlace))
        XCTAssertFalse(IntentRouter.whatIsHereResultIsUsable(missing))
    }

    func testSynthesizeCompareRendersDedicatedRelationsArticle() {
        // Verified on-device via `MCPZimEvalCLI --probe-compare`:
        // compare_articles for ["north korea", "south korea"] against
        // a real Wikipedia ZIM returns a `strategy:
        // "dedicated_relations_article"` payload with top-level
        // `sections`, NOT an `articles` array. Before the fix the
        // synth fell through to "I couldn't find articles matching
        // those titles" — a lie, since the tool succeeded and the
        // relations article was right there.
        let fullResult: [String: Any] = [
            "strategy": "dedicated_relations_article",
            "resolved_title": "North Korea – South Korea relations",
            "path": "A/North_Korea_-_South_Korea_relations",
            "zim": "wikipedia_en_all_maxi_2025-10.zim",
            "requested": ["north korea", "south korea"],
            "sections": [[
                "title": "lead",
                "text": "North Korea–South Korea relations are the "
                    + "diplomatic, political, economic, and cultural "
                    + "relations between the two Korean states on "
                    + "either side of the Korean Demilitarized Zone."
            ]]
        ]
        let s = IntentRouter.synthesizeCompareReply(
            args: ["titles": ["north korea", "south korea"]],
            fullResult: fullResult
        )
        XCTAssertTrue(s.contains("North Korea – South Korea relations"),
                      "should name the resolved relations article: \(s)")
        XCTAssertTrue(s.contains("diplomatic"),
                      "should include the lead text snippet: \(s)")
        XCTAssertFalse(s.contains("couldn't find"),
                       "must not fall through to the not-found branch "
                       + "when the relations article WAS found")
    }

    func testSynthesizeCompareRequiresBothSubjectsToHaveContent() {
        // On-device repro: fast-path dispatched with
        // titles=["north","south korea"], tool returned one valid
        // article + one error. Old synth emitted just "north" +
        // the South Korea lead, which reads as an article lookup,
        // not a comparison. New synth insists on two subjects with
        // real text; otherwise falls back to a clear "couldn't
        // find these titles — try the full names" message.
        let fullResult: [String: Any] = [
            "articles": [
                ["title": "north", "error": "Could not fetch"],
                [
                    "title": "South Korea",
                    "sections": [[
                        "text": "South Korea is a country in East Asia."
                    ]]
                ],
            ]
        ]
        let s = IntentRouter.synthesizeCompareReply(
            args: ["titles": ["north", "south korea"]],
            fullResult: fullResult
        )
        XCTAssertFalse(s.contains("East Asia"),
                       "must not surface one-sided content as a comparison")
        XCTAssertTrue(s.contains("couldn't") || s.contains("try"),
                      "should prompt the user to rephrase")
        XCTAssertTrue(s.contains("north"),
                      "should name the title that failed")
    }

    func testSynthesizeCompareBothErrored() {
        let fullResult: [String: Any] = [
            "articles": [
                ["title": "north",       "error": "Could not fetch"],
                ["title": "south korea", "error": "Could not fetch"],
            ]
        ]
        let s = IntentRouter.synthesizeCompareReply(
            args: ["titles": ["north", "south korea"]],
            fullResult: fullResult
        )
        XCTAssertTrue(s.contains("north") && s.contains("south korea"),
                      "should name both failing titles")
    }

    func testSynthesizeWhatIsHereNearby() {
        let fullResult: [String: Any] = [
            "nearest_named_place": "Civic Center",
            "admin_area": "San Francisco",
            "distance_m": 50,
        ]
        let s = IntentRouter.synthesizeWhatIsHereReply(fullResult: fullResult)
        XCTAssertTrue(s.contains("in Civic Center"),
                      "<=100 m should say 'in X': got \(s)")
        XCTAssertTrue(s.contains("San Francisco"))
    }

    func testSynthesizeWhatIsHereMetersAway() {
        let fullResult: [String: Any] = [
            "nearest_named_place": "Palo Alto",
            "distance_m": 420,
        ]
        let s = IntentRouter.synthesizeWhatIsHereReply(fullResult: fullResult)
        XCTAssertTrue(s.contains("420 m from Palo Alto"),
                      "sub-km should render metres: got \(s)")
    }

    func testSynthesizeWhatIsHereKilometresAway() {
        let fullResult: [String: Any] = [
            "nearest_named_place": "Palo Alto",
            "distance_m": 2400,
        ]
        let s = IntentRouter.synthesizeWhatIsHereReply(fullResult: fullResult)
        XCTAssertTrue(s.contains("2.4 km"),
                      ">=1000 m should render km: got \(s)")
    }

    func testFirstSentencesCutsOnBoundary() {
        let text = "First sentence. Second sentence. Third sentence that "
            + "keeps going for a while to push us well past the budget."
        let s = IntentRouter.firstSentences(text, maxChars: 30)
        // 30 chars budget puts us past "First sentence." (15) but not
        // all the way through "Second sentence." — cuts cleanly at the
        // first terminator.
        XCTAssertTrue(s.hasSuffix("."), "should keep terminator: \(s)")
        XCTAssertFalse(s.hasSuffix("…"),
                       "boundary cut should not ellipsis: \(s)")
    }

    func testFirstSentencesFallsBackToEllipsis() {
        let text = "A long phrase with no terminator in the budget window"
        let s = IntentRouter.firstSentences(text, maxChars: 20)
        XCTAssertTrue(s.hasSuffix("…"),
                      "should ellipsis when no terminator: \(s)")
    }

    // MARK: - Misses → LLM fall-through

    func testClassifyReturnsNilForFreeformQueries() {
        // Queries that don't match any fast path — fall through to
        // the LLM. "what's the weather" and "who was Ada Lovelace"
        // used to be here too, but they're now covered by the
        // `article_overview` fast path (those articles may or may
        // not exist in the loaded ZIMs; the tool returns cleanly
        // either way, which is still faster than a 15 s prefill).
        let queries = [
            "explain gravity",
            "summarise the article about Jeff Dean",
            "",
            "?!?",
        ]
        for q in queries {
            XCTAssertNil(IntentRouter.classify(q), "unexpected match for: \(q)")
        }
    }

    // MARK: - Pure singularise

    func testSingulariseRules() {
        XCTAssertEqual(IntentRouter.singularize("bars"),      "bar")
        XCTAssertEqual(IntentRouter.singularize("museums"),   "museum")
        XCTAssertEqual(IntentRouter.singularize("libraries"), "library")
        XCTAssertEqual(IntentRouter.singularize("churches"),  "church")
        XCTAssertEqual(IntentRouter.singularize("buses"),     "bus")
        XCTAssertEqual(IntentRouter.singularize("boxes"),     "box")
        // Edge cases — don't butcher these.
        XCTAssertEqual(IntentRouter.singularize("class"),     "class")  // -ss
        XCTAssertEqual(IntentRouter.singularize("glass"),     "glass")
        XCTAssertEqual(IntentRouter.singularize("is"),        "is")     // too short
        XCTAssertEqual(IntentRouter.singularize("cafe"),      "cafe")   // already singular
    }

    // MARK: - `synthesizePlacesReply`

    func testSynthesiseFoundN() {
        let args: [String: Any] = [
            "place": "North Beach",
            "kinds": ["bar"]
        ]
        let result: [String: Any] = [
            "total_in_radius": 15, "radius_km": 1.5
        ]
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "near_named_place", args: args, fullResult: result
        )
        XCTAssertTrue(s.contains("Found 15 bars near North Beach"),
                      "got: \(s)")
        XCTAssertTrue(s.contains("within 1.5 km"), "got: \(s)")
        // 2026-07-02: caption reworded for voice — "Tap a pin / tap List"
        // read like UI chrome when spoken; the map reference stays.
        XCTAssertTrue(s.contains("on the map below"), "got: \(s)")
    }

    func testSynthesiseNamesNearestHit() {
        // "Where's the nearest coffee shop?" must NAME the closest hit
        // with distance + compass direction, not just count them
        // (real capture 2026-07-02: "Found 186 coffee shops… tap a pin").
        let args: [String: Any] = [
            "lat": 37.44, "lon": -122.15, "kinds": ["coffee shop"],
        ]
        let result: [String: Any] = [
            "total_in_radius": 186, "radius_km": 5,
            "results": [
                ["name": "Blue Bottle Coffee", "distance_m": 250,
                 "direction": "north-east", "lat": 37.44, "lon": -122.15],
                ["name": "Peet's", "distance_m": 400, "lat": 37.44, "lon": -122.15],
            ],
        ]
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "near_places", args: args, fullResult: result
        )
        XCTAssertTrue(
            s.contains("The nearest coffee shop is Blue Bottle Coffee, 250 m north-east"),
            "got: \(s)")
        XCTAssertTrue(s.contains("185 more coffee shops"), "got: \(s)")
    }

    func testSynthesiseZeroResults() {
        let args: [String: Any] = ["place": "Mountain View", "kinds": ["museum"]]
        let result: [String: Any] = ["total_in_radius": 0, "radius_km": 1]
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "near_named_place", args: args, fullResult: result
        )
        XCTAssertTrue(s.contains("No museums found"), "got: \(s)")
        // Zero-results caption drops the "tap List" hint since there's
        // nothing to list.
        XCTAssertFalse(s.contains("tap List"), "got: \(s)")
    }

    func testSynthesiseAlreadyPluralKind() {
        // If the user asked with a plural, we don't double-pluralise
        // back to "barsss".
        let args: [String: Any] = ["place": "SF", "kinds": ["bars"]]
        let result: [String: Any] = ["total_in_radius": 3]
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "near_named_place", args: args, fullResult: result
        )
        XCTAssertTrue(s.contains("3 bars near SF"), "got: \(s)")
        XCTAssertFalse(s.contains("barss"), "got: \(s)")
    }

    func testSynthesiseUserCentric() {
        // near_places with lat/lon → "near you"
        let args: [String: Any] = [
            "lat": 37.44, "lon": -122.15,
            "kinds": ["restaurant"]
        ]
        let result: [String: Any] = ["total_in_radius": 7]
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "near_places", args: args, fullResult: result
        )
        XCTAssertTrue(s.contains("7 restaurants near you"), "got: \(s)")
    }

    func testSynthesiseRadiusFormatting() {
        for (km, expected) in [
            (0.5, "500 m"),     // <1 km → metres
            (1.0, "1 km"),      // integer
            (5.0, "5 km"),
            (2.5, "2.5 km")     // fractional
        ] {
            let args: [String: Any] = ["place": "X", "kinds": ["bar"]]
            let result: [String: Any] = ["total_in_radius": 1, "radius_km": km]
            let s = IntentRouter.synthesizePlacesReply(
                toolName: "near_named_place", args: args, fullResult: result
            )
            XCTAssertTrue(
                s.contains("within \(expected)"),
                "km=\(km): got \(s)"
            )
        }
    }

    func testSynthesiseStoriesShape() {
        let args: [String: Any] = ["place": "Palo Alto"]
        let result: [String: Any] = [
            "stories": [
                ["place_name": "HP Garage", "lat": 1.0, "lon": 2.0, "excerpt": "x"]
            ],
            "count": 1,
            "radius_km": 3
        ]
        let s = IntentRouter.synthesizePlacesReply(
            toolName: "nearby_stories_at_place", args: args, fullResult: result
        )
        // `kinds` was nil → fallback bucket is "places".
        XCTAssertTrue(s.contains("Found 1 places near Palo Alto"), "got: \(s)")
        XCTAssertTrue(s.contains("3 km"), "got: \(s)")
    }

    func testDiscussionFacetTitlesDoNotMeanTopicChanges() {
        XCTAssertTrue(IntentRouter.isDiscussionFacetTitle("the combatants"))
        XCTAssertTrue(IntentRouter.isDiscussionFacetTitle("his parents"))
        XCTAssertTrue(IntentRouter.isDiscussionFacetTitle("Casualties"))
        XCTAssertFalse(IntentRouter.isDiscussionFacetTitle("Donald Trump"))
        XCTAssertFalse(IntentRouter.isDiscussionFacetTitle("The French Revolution"))
    }
}
