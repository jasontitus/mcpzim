// SPDX-License-Identifier: MIT
//
// Coverage for the 2026-07-01 conversational fixes:
//   * locational follow-ups route to `distance_to` (distance+direction
//     answer) instead of a `near_places` POI dump;
//   * kind-aware rebinding — "how far is it?" after a topic turn binds the
//     most recent PLACE, not the topic;
//   * GeoMath bearings/compass points;
//   * drift threads from `what_is_here` and drill-in article fetches.

import XCTest
@testable import MCPZimKit

final class ConversationContinuationTests: XCTestCase {

    // MARK: - Fixtures

    private func focusWithPlace() -> ConversationFocus {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(
            name: "Ferry Building", kind: .place,
            zimPath: "A/Ferry_Building", lat: 37.7955, lon: -122.3937))
        return f
    }

    // MARK: - Distance follow-ups → distance_to

    func testHowFarFollowUpRoutesToDistanceTo() {
        let intent = IntentRouter.continuationIntent(
            "how far is it?", focus: focusWithPlace())
        XCTAssertEqual(intent?.toolName, "distance_to")
        XCTAssertEqual(intent?.anyArgs["place"] as? String, "Ferry Building")
        XCTAssertEqual(intent?.anyArgs["lat"] as? Double, 37.7955)
    }

    func testWhichWayFollowUpRoutesToDistanceTo() {
        let intent = IntentRouter.continuationIntent(
            "which way is that?", focus: focusWithPlace())
        XCTAssertEqual(intent?.toolName, "distance_to")
    }

    func testCanIWalkFollowUpRoutesToDistanceTo() {
        let intent = IntentRouter.continuationIntent(
            "can I walk there? it seems close", focus: focusWithPlace())
        XCTAssertEqual(intent?.toolName, "distance_to")
    }

    // MARK: - Kind-aware rebinding

    func testLocationalFollowUpRebindsToMostRecentPlace() {
        // Place first, then a TOPIC becomes primary — "how far is it?"
        // must bind the place, not dispatch an overview of the topic.
        var f = focusWithPlace()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Ohlone", kind: .topic, zimPath: "A/Ohlone"))
        let intent = IntentRouter.continuationIntent("how far is it?", focus: f)
        XCTAssertEqual(intent?.toolName, "distance_to")
        XCTAssertEqual(intent?.anyArgs["place"] as? String, "Ferry Building")
    }

    func testDirectionsFollowUpStillRoutesToRoute() {
        let intent = IntentRouter.continuationIntent(
            "directions to it please", focus: focusWithPlace())
        XCTAssertEqual(intent?.toolName, "route_from_places")
        XCTAssertEqual(intent?.anyArgs["destination"] as? String, "Ferry Building")
    }

    func testProximityFollowUpStillRoutesToNearPlaces() {
        let intent = IntentRouter.continuationIntent(
            "what's around it?", focus: focusWithPlace())
        XCTAssertEqual(intent?.toolName, "near_places")
        XCTAssertEqual(intent?.anyArgs["lat"] as? Double, 37.7955)
    }

    func testLocationalWithNoPlaceFallsBackToOverview() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "General relativity", kind: .topic))
        let intent = IntentRouter.continuationIntent("how far is it?", focus: f)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "General relativity")
    }

    // MARK: - GeoMath

    func testBearingCardinalPoints() {
        // Due north / east / south / west from the origin.
        XCTAssertEqual(GeoMath.compassPoint(
            fromLat: 0, fromLon: 0, toLat: 1, toLon: 0), "north")
        XCTAssertEqual(GeoMath.compassPoint(
            fromLat: 0, fromLon: 0, toLat: 0, toLon: 1), "east")
        XCTAssertEqual(GeoMath.compassPoint(
            fromLat: 0, fromLon: 0, toLat: -1, toLon: 0), "south")
        XCTAssertEqual(GeoMath.compassPoint(
            fromLat: 0, fromLon: 0, toLat: 0, toLon: -1), "west")
    }

    func testBearingIntercardinal() {
        XCTAssertEqual(GeoMath.compassPoint(
            fromLat: 0, fromLon: 0, toLat: 1, toLon: 1), "north-east")
        XCTAssertEqual(GeoMath.compassPoint(
            fromLat: 0, fromLon: 0, toLat: -1, toLon: -1), "south-west")
    }

    func testCompassPointDegreeBuckets() {
        XCTAssertEqual(GeoMath.compassPoint(degrees: 0), "north")
        XCTAssertEqual(GeoMath.compassPoint(degrees: 359), "north")
        XCTAssertEqual(GeoMath.compassPoint(degrees: 45), "north-east")
        XCTAssertEqual(GeoMath.compassPoint(degrees: 100), "east")
        XCTAssertEqual(GeoMath.compassPoint(degrees: 200), "south")      // [157.5, 202.5)
        XCTAssertEqual(GeoMath.compassPoint(degrees: 220), "south-west") // [202.5, 247.5)
        XCTAssertEqual(GeoMath.compassPoint(degrees: 292), "west")
    }

    func testHaversineSanity() {
        // SF Ferry Building → Coit Tower is ~1.3 km.
        let d = GeoMath.haversineMeters(37.7955, -122.3937, 37.8024, -122.4058)
        XCTAssertGreaterThan(d, 1000)
        XCTAssertLessThan(d, 1800)
    }

    // MARK: - Drift threads: what_is_here + drill-ins

    func testWhatIsHereOffersNearbySiblingsAsThreads() {
        // The extractor reads the reverse-geocode's wiki-backed
        // runners-up (`nearby`) — drift goes to the neighbours, not
        // back to the place the user is standing in.
        let result: [String: Any] = [
            "nearest_named_place": "North Beach",
            "wiki_title": "North Beach, San Francisco",
            "nearby": [
                ["name": "Coit Tower", "wikipedia": "en:Coit Tower",
                 "lat": 37.8024, "lon": -122.4058, "distance_m": 240],
                ["name": "Washington Square",
                 "wikipedia": "en:Washington Square, San Francisco",
                 "lat": 37.8004, "lon": -122.4103, "distance_m": 400],
            ],
        ]
        let threads = ConversationThreads.extract(
            toolName: "what_is_here", result: result)
        XCTAssertEqual(threads.map(\.label), ["Coit Tower", "Washington Square"])
        XCTAssertEqual(threads.first?.kind, .topic)
        XCTAssertEqual(threads.first?.note, "240 m away")
    }

    // MARK: - Possessive facets (real capture 2026-07-01, "Putin's early life")

    func testPossessiveFacetStripsToEntity() {
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacet(from: "putin's early life"),
            "putin")
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacet(from: "vladimir putin and his early life"),
            "vladimir putin")
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacet(from: "marie curie and her career"),
            "marie curie")
        // Real titles with possessives survive untouched.
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacet(from: "hitchhiker's guide to the galaxy"),
            "hitchhiker's guide to the galaxy")
    }

    func testTellMeAboutPossessiveFacetRoutesToEntityOverview() {
        let intent = IntentRouter.classify("Tell me about Putin’s early life")
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "putin")
    }

    func testExplicitSubjectWithPossessivePronounIsNotHijacked() {
        // Turn 1 subject in focus; turn 2 NAMES a new subject and uses
        // "his" — the resolver must NOT bind "his" back to the old
        // entity (that re-dispatched turn 1's failed title on device).
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "putin's early life", kind: .topic))
        f.beginUserTurn()
        let resolved = ReferenceResolver.resolve(
            "Tell me about Vladimir Putin and his early life", focus: f)
        XCTAssertNil(resolved.boundEntity,
                     "possessive after an explicit subject must not bind")
        // And the router should extract the clean entity title.
        let intent = IntentRouter.classify(
            "Tell me about Vladimir Putin and his early life", focus: f)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "vladimir putin")
    }

    func testApostropheLessPossessiveRetryStrip() {
        // Voice dictation drops the apostrophe ("putins childhood").
        // Conservative strip leaves it (too risky first-pass)…
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacet(from: "putins childhood"),
            "putins childhood")
        // …the aggressive variant (miss-retry only) recovers the entity.
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacetAggressive(from: "putins childhood"),
            "putin")
        // No change when the trailing phrase isn't a whitelisted facet.
        XCTAssertEqual(
            IntentRouter.stripPossessiveFacetAggressive(from: "boston red sox"),
            "boston red sox")
    }

    func testHowAboutWithNewSubjectClassifiesAsOverview() {
        // Mid-discussion "How about Donald Trump's childhood?" must
        // classify as a fresh overview (the resolver refuses to bind a
        // possessive after an explicit subject), so the discussion-exit
        // check can see the topic change. Real capture 2026-07-01.
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Putin", kind: .topic))
        let intent = IntentRouter.classify(
            "How about Donald Trump’s childhood?", focus: f)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "donald trump")
    }

    func testWhatAboutPronounSubjectDoesNotMatchOverviewPattern() {
        // With no focus to bind against, "what about his parents" must
        // NOT become article_overview(title: "his parents").
        XCTAssertNil(IntentRouter.classify("What about his parents?"))
    }

    // MARK: - Attribute questions (device capture 2026-07-19: "What is the
    // most recent version?" → Versioning file system; "What is the current
    // version?" → Nicene Creed)

    func testAttributeQuestionBindsToFocusEntity() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Apple Tv", kind: .topic, zimPath: "A/Apple_Tv"))
        for q in ["What is the most recent version?",
                  "What is the current version?",
                  "what's the price?"] {
            let resolved = ReferenceResolver.resolve(q, focus: f)
            XCTAssertEqual(resolved.boundEntity?.name, "Apple Tv", "q: \(q)")
            let intent = IntentRouter.classify(q, focus: f)
            XCTAssertEqual(intent?.toolName, "article_overview", "q: \(q)")
            XCTAssertEqual(intent?.anyArgs["title"] as? String, "Apple Tv", "q: \(q)")
        }
    }

    func testProperNounAttributeShapeStaysFresh() {
        // "What is the Nicene Creed?" is a REAL topic despite leading "the" —
        // the capitalised proper noun keeps it out of the attribute binding.
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Apple Tv", kind: .topic))
        let resolved = ReferenceResolver.resolve("What is the Nicene Creed?", focus: f)
        XCTAssertNil(resolved.boundEntity)
        // Lowercase non-attribute topics also stay fresh.
        let resolved2 = ReferenceResolver.resolve("what is quantum entanglement", focus: f)
        XCTAssertNil(resolved2.boundEntity)
    }

    func testNoCorrectionStripAndAttributeOfReduction() {
        // "No. What is the most recent version of Apple TV?" must reach the
        // overview pattern (leading "No." stripped) and reduce the title to
        // the entity after "of".
        let intent = IntentRouter.classify("No. What is the most recent version of Apple TV?")
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "apple tv")
        // Genuine "of" titles are untouched ("the history of France" leads
        // with an attribute-less noun … actually "history" is not in the
        // attribute vocabulary, so the phrase passes through intact).
        let intent2 = IntentRouter.classify("Tell me about the history of France")
        XCTAssertEqual(intent2?.anyArgs["title"] as? String, "the history of france")
    }

    func testLeadingConnectiveDoesNotDefeatTopicSwitch() {
        // "And tell me about Donald Trump" mid-discussion — the leading
        // "And" broke every ^-anchored stateless pattern, so no intent
        // classified, discussion mode never exited, and the model
        // confabulated Trump facts from Putin passages (device capture
        // 2026-07-02).
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Putin", kind: .topic))
        let intent = IntentRouter.classify("And tell me about Donald Trump", focus: f)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "donald trump")
        // A bare connective is still a continuation, not a stripped-empty
        // misfire.
        XCTAssertNil(IntentRouter.classify("and?"))
        // "Andrew Jackson" is not "and rew Jackson".
        XCTAssertEqual(
            IntentRouter.classify("Tell me about Andrew Jackson")?.anyArgs["title"] as? String,
            "andrew jackson")
    }

    func testExplicitNameDoesNotBindPartialThreadLabel() {
        // Real capture 2026-07-02: with "The Trump Organization" offered
        // as a drift thread, "Tell me about Donald Trump" bound to it on
        // the shared token "trump" and answered about the company.
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Putin", kind: .topic))
        f.setThreads([DiscoveryThread(
            label: "The Trump Organization", kind: .topic, source: .wikilink)])
        let resolved = ReferenceResolver.resolve(
            "Tell me about Donald Trump", focus: f)
        XCTAssertNil(resolved.boundEntity,
                     "partial label overlap must not hijack an explicit name")
        // A genuine thread pick (all content words in the label) still binds.
        let picked = ReferenceResolver.resolve(
            "the trump organization", focus: f)
        XCTAssertEqual(picked.boundEntity?.name, "The Trump Organization")
    }

    func testAffirmativePlusNamePicksFromLastList() {
        // Disambiguation offer registered [resolved, alternate]; the user
        // answers "Yes - gravitational waves" (real capture 2026-07-02 —
        // it fell through and re-answered from the fluid article).
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Gravity waves", kind: .topic))
        f.setLastList([
            FocusEntity(name: "Gravity waves", kind: .topic),
            FocusEntity(name: "Gravitational wave", kind: .topic),
        ])
        let resolved = ReferenceResolver.resolve("Yes - gravitational waves", focus: f)
        XCTAssertEqual(resolved.boundEntity?.name, "Gravitational wave")
        // And the router turns the pick into the right overview.
        let intent = IntentRouter.classify("Yes - gravitational waves", focus: f)
        XCTAssertEqual(intent?.toolName, "article_overview")
        XCTAssertEqual(intent?.anyArgs["title"] as? String, "Gravitational wave")
    }

    func testQuestionMentioningListItemDoesNotPick() {
        // Extra content words mean it's a QUESTION, not a pick.
        var f = ConversationFocus()
        f.beginUserTurn()
        f.setLastList([
            FocusEntity(name: "Gravity waves", kind: .topic),
            FocusEntity(name: "Gravitational wave", kind: .topic),
        ])
        let resolved = ReferenceResolver.resolve(
            "What did Einstein say about gravitational waves?", focus: f)
        if case .listSelection = resolved.binding {
            XCTFail("question must not be treated as a list pick")
        }
    }

    func testSubjectlessPossessiveStillBinds() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Vladimir Putin", kind: .topic))
        let resolved = ReferenceResolver.resolve("what's his name?", focus: f)
        XCTAssertEqual(resolved.boundEntity?.name, "Vladimir Putin")
    }

    func testWhoBuiltItStillBinds() {
        var f = ConversationFocus()
        f.beginUserTurn()
        f.remember(FocusEntity(name: "Stanford Memorial Church", kind: .topic))
        let resolved = ReferenceResolver.resolve("who built it", focus: f)
        XCTAssertEqual(resolved.boundEntity?.name, "Stanford Memorial Church")
    }

    func testWhatIsHereWithoutNearbyYieldsNoThreads() {
        let result: [String: Any] = [
            "nearest_named_place": "Somewhereville",
            "distance_m": 900,
        ]
        let threads = ConversationThreads.extract(
            toolName: "what_is_here", result: result)
        XCTAssertTrue(threads.isEmpty)
    }

    func testGetArticleSectionRelatedBecomesWikilinkThreads() {
        // Drill-in fetches now carry `related[]` — the drift offer must
        // survive a `get_article_section` turn.
        let result: [String: Any] = [
            "title": "Golden Gate Bridge",
            "section": "History",
            "text": "…",
            "related": [
                ["title": "Joseph Strauss", "path": "A/Joseph_Strauss"],
                ["title": "Art Deco", "path": "A/Art_Deco"],
            ],
        ]
        let threads = ConversationThreads.extract(
            toolName: "get_article_section", result: result)
        XCTAssertEqual(
            threads.filter { $0.source == .wikilink }.map(\.label),
            ["Joseph Strauss", "Art Deco"])
    }
}
