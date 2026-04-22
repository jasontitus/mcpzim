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

    func testClassifyQuestionsAreNotPlaces() {
        // Don't misclassify "how does X work" or "where can I find Y"
        // as places searches — those go to the LLM.
        XCTAssertNil(IntentRouter.classify("how does rain form in clouds"))
        XCTAssertNil(IntentRouter.classify("where can I find help in SF"))
        XCTAssertNil(IntentRouter.classify("what is the fish in the sea"))
        XCTAssertNil(IntentRouter.classify("tell me about volcanoes in Hawaii"))
        XCTAssertNil(IntentRouter.classify("can you find bars in North Beach"))
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

    // MARK: - Misses → LLM fall-through

    func testClassifyReturnsNilForFreeformQueries() {
        let queries = [
            "what's the weather",
            "explain gravity",
            "summarise the article about Jeff Dean",
            "who was Ada Lovelace",
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
        XCTAssertTrue(s.contains("Tap a pin"), "got: \(s)")
        XCTAssertTrue(s.contains("tap List"), "got: \(s)")
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
}
