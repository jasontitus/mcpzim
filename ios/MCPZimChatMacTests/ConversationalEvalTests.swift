// SPDX-License-Identifier: MIT
//
// End-to-end conversational evaluation. Drives a real `ChatSession` with
// real Gemma-4 and the loaded streetzim, then scores the model's
// responses against a scenario library. Assertions are intentionally
// LOOSE — small models vary turn-to-turn. The harness reports which
// scenarios pass so we can tell at a glance whether a tool-surface /
// prompt change helped or hurt.
//
// Enable with:
//   ZIMBLE_TEST_STREETZIM=/abs/path/to/osm-washington-dc-test.zim \
//   ZIMBLE_TEST_WIKIPEDIA=/abs/path/to/wikipedia_en_*.zim \
//   xcodebuild test -scheme MCPZimChatMacTests …
//
// Skipped (green) when the env var isn't set. The scenarios target DC
// because that's the region the maintainer's test ZIM covers; adapt if
// you point it at a different streetzim.

import XCTest
import MCPZimKit
@testable import MCPZimChatMac

@MainActor
final class ConversationalEvalTests: XCTestCase {

    // MARK: - Scenario model

    struct TurnExpect {
        /// Tools that MUST have been called this turn.
        var toolsCalledAny: [String] = []
        /// Tools that MUST NOT be called this turn (e.g. raw-coord
        /// tools on the conversational surface).
        var toolsNotCalled: [String] = []
        /// Per-tool minimum call counts — asserts the tool appears at
        /// least N times in this turn. Used to catch "model called
        /// get_article_section once and stopped" on explanatory turns
        /// where we want multi-section grounding.
        var minimumToolCallCounts: [String: Int] = [:]
        /// Substrings all required (case-insensitive).
        var allOf: [String] = []
        /// Any one of these required (case-insensitive).
        var anyOf: [String] = []
        /// None of these may appear (e.g. leaked coordinates).
        var noneOf: [String] = []
        /// If true, this turn MUST have invoked at least one tool.
        /// Used to catch "model answered from memory without querying".
        var mustCallTool: Bool = false
    }

    struct Scenario {
        let name: String
        let turns: [(user: String, expect: TurnExpect)]
    }

    /// Scenarios modelled on realistic DC-area conversations. All
    /// queries target places the OSM test ZIM is known to cover.
    static let scenarios: [Scenario] = [
        // --- Neighborhood exploration ---------------------------------
        .init(
            name: "adams_morgan_breakdown",
            turns: [(
                user: "what's around adams morgan?",
                expect: TurnExpect(
                    toolsCalledAny: ["near_named_place"],
                    toolsNotCalled: ["near_places", "plan_driving_route", "geocode"],
                    anyOf: ["bar", "cafe", "shop", "amenity"],
                    noneOf: ["latitude ", "longitude "]
                )
            )]
        ),
        .init(
            name: "adams_morgan_drill_colleges",
            turns: [
                (user: "what's around adams morgan?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"])),
                (user: "tell me about the colleges",
                 expect: TurnExpect(
                     toolsCalledAny: ["near_named_place"],
                     anyOf: ["college", "university"],
                     mustCallTool: true
                 )),
            ]
        ),
        .init(
            name: "adams_morgan_drill_neighborhoods",
            turns: [
                (user: "what's around adams morgan?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"])),
                (user: "what about the neighborhoods?",
                 expect: TurnExpect(
                     toolsCalledAny: ["near_named_place"],
                     anyOf: ["neighbourhood", "neighborhood", "found"],
                     mustCallTool: true
                 )),
            ]
        ),
        .init(
            name: "adams_morgan_drill_bars",
            turns: [
                (user: "what's around adams morgan?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"])),
                (user: "list the bars",
                 expect: TurnExpect(
                     toolsCalledAny: ["near_named_place"],
                     mustCallTool: true
                 )),
            ]
        ),
        .init(
            name: "dupont_circle_breakdown",
            turns: [(
                user: "what's in dupont circle?",
                expect: TurnExpect(
                    toolsCalledAny: ["near_named_place"],
                    toolsNotCalled: ["near_places"],
                    anyOf: ["bar", "cafe", "shop", "amenity", "park", "restaurant"]
                )
            )]
        ),
        .init(
            name: "georgetown_breakdown",
            turns: [(
                user: "what's around georgetown?",
                expect: TurnExpect(
                    toolsCalledAny: ["near_named_place"],
                    anyOf: ["shop", "cafe", "restaurant", "amenity", "school"]
                )
            )]
        ),

        // --- Routing --------------------------------------------------
        .init(
            name: "route_adams_to_dupont",
            turns: [(
                user: "route from adams morgan to dupont circle",
                expect: TurnExpect(
                    toolsCalledAny: ["route_from_places"],
                    toolsNotCalled: ["plan_driving_route", "geocode"],
                    anyOf: ["mile", "km", "minute", "hour"]
                )
            )]
        ),
        .init(
            name: "route_followup_duration",
            turns: [
                // Use the OSM-standard name so the streetzim geocode
                // actually finds it — "the capitol" was too vague and
                // resolved to garbage in the test ZIM.
                (user: "route from union station to the united states capitol",
                 expect: TurnExpect(toolsCalledAny: ["route_from_places"])),
                (user: "how long does that take?",
                 expect: TurnExpect(
                     anyOf: ["minute", "hour", "m.", ":"]
                 )),
            ]
        ),
        .init(
            name: "route_short_walk",
            turns: [(
                user: "how do i get from the white house to the washington monument?",
                expect: TurnExpect(
                    toolsCalledAny: ["route_from_places"],
                    toolsNotCalled: ["plan_driving_route"],
                    anyOf: ["mile", "km", "minute", "street", "road"]
                )
            )]
        ),

        // --- Landmark lookup (cross-ZIM wiki if available) -----------
        .init(
            name: "landmark_white_house",
            turns: [(
                user: "where is the white house?",
                expect: TurnExpect(
                    // Either the streetzim (POI with coords) OR the
                    // wikipedia search/article is a valid way to
                    // answer. Any of them counts.
                    anyOf: ["pennsylvania", "washington", "avenue", "address",
                            "government", "building", "white house",
                            "d.c.", "district of columbia"]
                )
            )]
        ),

        // --- Multi-turn with pronoun references ----------------------
        // These exercise whether the model carries context across turns
        // ("where is X" → "how do I get there") and whether it resolves
        // deictic references ("near here", "is there a X nearby")
        // correctly against the last-mentioned place.
        .init(
            name: "explore_then_ask_post_office",
            turns: [
                (user: "what's around dupont circle?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"])),
                (user: "is there a post office near there?",
                 expect: TurnExpect(
                     toolsCalledAny: ["near_named_place"],
                     anyOf: ["post", "office", "no ", "none", "not"],
                     mustCallTool: true
                 )),
            ]
        ),
        .init(
            name: "explore_then_coffee_comparison",
            turns: [
                (user: "any cafés near dupont circle?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"],
                                    anyOf: ["cafe", "café", "coffee"])),
                (user: "how about near adams morgan?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"],
                                    mustCallTool: true)),
            ]
        ),
        .init(
            name: "landmark_then_route_there",
            turns: [
                (user: "where is the national zoo?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"])),
                (user: "how do i get there from dupont circle?",
                 expect: TurnExpect(
                     toolsCalledAny: ["route_from_places"],
                     toolsNotCalled: ["plan_driving_route"],
                     anyOf: ["mile", "km", "minute", "hour", "error", "unable"]
                 )),
            ]
        ),
        .init(
            name: "drill_in_then_route_to_item",
            turns: [
                (user: "what museums are near the capitol?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"],
                                    anyOf: ["museum", "smithsonian", "gallery",
                                            "attraction", "none", "no "])),
                (user: "how do i get to the closest one from union station?",
                 expect: TurnExpect(
                     toolsCalledAny: ["route_from_places"],
                     anyOf: ["mile", "km", "minute", "error", "unable"]
                 )),
            ]
        ),
        .init(
            name: "breakdown_then_pronoun_drill_in",
            turns: [
                (user: "what's in georgetown?",
                 expect: TurnExpect(toolsCalledAny: ["near_named_place"])),
                // "list them" is genuinely ambiguous (them = what?).
                // Either a clarifying question OR a re-query is a
                // reasonable answer; only a confabulated list would
                // be wrong. We can't easily assert that negative in a
                // one-line regex, so accept either path here.
                (user: "list them",
                 expect: TurnExpect(
                     anyOf: ["which", "specify", "category", "type",
                             "cafe", "bar", "shop", "amenity", "park",
                             "please", "tell me"]
                 )),
            ]
        ),
        .init(
            name: "ambiguous_here_needs_clarification",
            turns: [
                // With no established reference, the model should
                // either ask for clarification or default to the last
                // place mentioned (here: none).
                (user: "is there a library near here?",
                 expect: TurnExpect(
                     anyOf: ["which", "where", "location", "specify",
                             "provide", "library"]
                 )),
            ]
        ),

        // --- Cross-ZIM: streetzim POI → wikipedia article ------------
        // Requires BOTH env vars. The model should:
        //   1. Call near_named_place(place="Adams Morgan") to find
        //      nearby POIs with `wikipedia` fields populated.
        //   2. For at least one of those, call get_article(path=…) and
        //      use the body to answer.
        // If ZIMBLE_TEST_WIKIPEDIA is unset the scenario falls through
        // to the assertion-only "it tried to chain" check.
        .init(
            name: "neighborhood_stories_via_wikipedia",
            turns: [(
                user: "tell me stories about adams morgan",
                expect: TurnExpect(
                    toolsCalledAny: ["near_named_place"],
                    anyOf: ["history", "historic", "named", "founded",
                            "neighbourhood", "neighborhood", "origin",
                            "washington", "district"]
                )
            )]
        ),

        // --- Pure wiki (doesn't need streetzim) ----------------------
        // Phrased as a direct topic lookup instead of a "what is X
        // used for" medical question — the latter triggers small
        // models' medical-advice refusals without them ever calling
        // the search tool.
        .init(
            name: "wiki_what_is_aspirin",
            turns: [(
                user: "look up the wikipedia article on aspirin",
                expect: TurnExpect(
                    toolsCalledAny: ["search"],
                    anyOf: ["pain", "inflammation", "fever", "blood",
                            "medicine", "medication", "salicylic",
                            "drug", "article"]
                )
            )]
        ),

        // --- Regression: explanatory wiki question MUST ground in
        //     sections, not just search snippets. First pass on this
        //     scenario caught the model answering from prior
        //     knowledge + snippets, never calling
        //     `get_article_section`. The fix was tightening the
        //     explanatory system-prompt hint to a fixed 5-step
        //     chain with an explicit minimum of 2
        //     `get_article_section` calls.
        .init(
            name: "quantum_encryption_must_ground_in_sections",
            turns: [(
                user: "how does quantum computing affect encryption?",
                expect: TurnExpect(
                    toolsCalledAny: ["search", "get_article_section"],
                    minimumToolCallCounts: ["get_article_section": 2],
                    anyOf: ["quantum", "shor", "post-quantum",
                            "cryptography", "rsa", "lattice",
                            "mceliece"],
                    noneOf: ["i do not have", "i don't have specific",
                             "training data"],
                    mustCallTool: true
                )
            )]
        ),

        // --- Regression: explanatory turn + factoid follow-up. The
        //     follow-up uses a pronoun ("that") and must either
        //     (a) call a tool to ground its answer, or (b) cite
        //     an article from the prior turn. Before the grounding-
        //     policy change, the model answered "1957" from memory
        //     with no citation — which is right, but the app can't
        //     distinguish that from hallucination. This test
        //     enforces EITHER a fresh tool call OR a citation
        //     string, AND the correct year.
        .init(
            name: "sputnik_explanatory_then_year_followup",
            turns: [
                (user: "how did the russian sputnik program affect american politics about space?",
                 expect: TurnExpect(
                     toolsCalledAny: ["search", "get_article_section"],
                     minimumToolCallCounts: ["get_article_section": 2],
                     anyOf: ["cold war", "space race", "nasa",
                             "satellite", "soviet", "1957",
                             "eisenhower"],
                     mustCallTool: true
                 )),
                (user: "what year was that?",
                 expect: TurnExpect(
                     allOf: ["1957"],
                     // If no fresh tool call, the answer must at
                     // least cite a prior article — rules out
                     // pure-memory fallbacks.
                     anyOf: ["per ", "according to", "article",
                             "sputnik 1", "list_article_sections",
                             "get_article_section"],
                     noneOf: ["i do not have", "i don't have specific"]
                 )),
            ]
        ),
    ]

    // MARK: - Harness

    private static var streetzimPath: String? {
        ProcessInfo.processInfo.environment["ZIMBLE_TEST_STREETZIM"]
    }
    private static var wikipediaPath: String? {
        ProcessInfo.processInfo.environment["ZIMBLE_TEST_WIKIPEDIA"]
    }
    /// Optional provider override. When set to a ModelProvider `id`
    /// ("apple-foundation-models", "gemma4-4b-it-4bit", "mock"), the
    /// harness switches to that model before loading. Unset → default
    /// (Gemma-4). Lets you run the same scenarios against different
    /// on-device backends and diff the scorecard.
    private static var modelId: String? {
        ProcessInfo.processInfo.environment["ZIMBLE_TEST_MODEL_ID"]
    }

    /// Shared session — model load is ~5 s + ~2.5 GB. Reused across tests.
    private static var sharedSession: ChatSession?

    private func makeOrReuseSession() async throws -> ChatSession {
        if let s = Self.sharedSession { return s }
        guard let streetzim = Self.streetzimPath else {
            throw XCTSkip("Set ZIMBLE_TEST_STREETZIM to run conversational eval.")
        }
        let session = ChatSession()
        var urls: [URL] = [URL(fileURLWithPath: streetzim)]
        if let wiki = Self.wikipediaPath {
            urls.append(URL(fileURLWithPath: wiki))
        }
        await session.addReaders(urls: urls)
        if let id = Self.modelId {
            await session.select(modelId: id)
            print("=== harness using model: \(session.selectedModel.displayName) ===")
        }
        await session.loadSelectedModel()
        // `loadSelectedModel()` returns before the observed state flips
        // to .ready — wait until the session publishes ready (or fails).
        let deadline = Date().addingTimeInterval(180)
        while Date() < deadline {
            switch session.modelState {
            case .ready: break
            case .failed(let msg):
                throw XCTSkip("Model load failed: \(msg)")
            default:
                try? await Task.sleep(for: .milliseconds(500))
                continue
            }
            break
        }
        guard case .ready = session.modelState else {
            throw XCTSkip("Model didn't reach ready in 180s: \(session.modelState)")
        }
        // Scenarios can generate a lot of debug lines (streetzim chunk
        // loads, streaming progress). Default cap of 500 was truncating
        // early tool dispatches before we scanned them.
        session.maxDebugEntries = 20_000
        Self.sharedSession = session
        return session
    }

    private func runTurn(_ session: ChatSession, text: String) async {
        session.send(text)
        let deadline = Date().addingTimeInterval(120)
        while session.isGenerating && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(200))
        }
        XCTAssertFalse(session.isGenerating,
                       "turn did not complete within 120s: \(text)")
    }

    private func toolsCalled(in session: ChatSession, since index: Int) -> [String] {
        var called: [String] = []
        for entry in session.debugEntries.dropFirst(index) where entry.category == "Tool" {
            let msg = entry.message
            guard msg.hasPrefix("dispatching ") else { continue }
            let rest = msg.dropFirst("dispatching ".count)
            let name = rest.prefix { $0.isLetter || $0.isNumber || $0 == "_" }
            if !name.isEmpty { called.append(String(name)) }
        }
        return called
    }

    private func assertTurn(
        _ expect: TurnExpect,
        assistantText: String,
        toolsCalled: [String],
        scenario: String, turn: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let lower = assistantText.lowercased()
        let uniqueTools = Set(toolsCalled)
        for needed in expect.toolsCalledAny {
            XCTAssertTrue(
                uniqueTools.contains(needed),
                "[\(scenario)#\(turn)] expected tool '\(needed)'. Called: \(toolsCalled). Reply: \(assistantText.prefix(400))",
                file: file, line: line
            )
        }
        for banned in expect.toolsNotCalled {
            XCTAssertFalse(
                uniqueTools.contains(banned),
                "[\(scenario)#\(turn)] tool '\(banned)' should NOT have been called. Called: \(toolsCalled)",
                file: file, line: line
            )
        }
        for (tool, minCount) in expect.minimumToolCallCounts {
            let actual = toolsCalled.filter { $0 == tool }.count
            XCTAssertGreaterThanOrEqual(
                actual, minCount,
                "[\(scenario)#\(turn)] expected '\(tool)' called ≥\(minCount) time(s); got \(actual). Called: \(toolsCalled)",
                file: file, line: line
            )
        }
        if expect.mustCallTool {
            XCTAssertFalse(
                toolsCalled.isEmpty,
                "[\(scenario)#\(turn)] model answered without calling any tool. Reply: \(assistantText.prefix(400))",
                file: file, line: line
            )
        }
        for needed in expect.allOf {
            XCTAssertTrue(
                lower.contains(needed.lowercased()),
                "[\(scenario)#\(turn)] expected substring '\(needed)'. Reply: \(assistantText.prefix(400))",
                file: file, line: line
            )
        }
        if !expect.anyOf.isEmpty {
            XCTAssertTrue(
                expect.anyOf.contains { lower.contains($0.lowercased()) },
                "[\(scenario)#\(turn)] expected any of \(expect.anyOf). Reply: \(assistantText.prefix(400))",
                file: file, line: line
            )
        }
        for banned in expect.noneOf {
            XCTAssertFalse(
                lower.contains(banned.lowercased()),
                "[\(scenario)#\(turn)] banned substring '\(banned)' appeared. Reply: \(assistantText.prefix(400))",
                file: file, line: line
            )
        }
    }

    private func runScenario(_ s: Scenario) async throws {
        let session = try await makeOrReuseSession()
        session.resetConversation()
        print("\n=== scenario: \(s.name) ===")
        for (idx, turn) in s.turns.enumerated() {
            let mark = session.debugEntries.count
            await runTurn(session, text: turn.user)
            let tools = toolsCalled(in: session, since: mark)
            let assistantText = (session.messages.last { $0.role == .assistant }?.text) ?? ""
            print("  turn \(idx) «\(turn.user)»")
            print("    tools=\(tools)")   // ordered list, with repeats — repeats matter
            print("    reply=«\(assistantText.prefix(220))…»")
            assertTurn(turn.expect,
                       assistantText: assistantText,
                       toolsCalled: tools,
                       scenario: s.name, turn: idx)
        }
    }

    // MARK: - One XCTest per scenario (stable names, easy to re-run one)

    func test_00_adams_morgan_breakdown() async throws { try await runScenario(Self.scenarios[0]) }
    func test_01_adams_morgan_drill_colleges() async throws { try await runScenario(Self.scenarios[1]) }
    func test_02_adams_morgan_drill_neighborhoods() async throws { try await runScenario(Self.scenarios[2]) }
    func test_03_adams_morgan_drill_bars() async throws { try await runScenario(Self.scenarios[3]) }
    func test_04_dupont_circle_breakdown() async throws { try await runScenario(Self.scenarios[4]) }
    func test_05_georgetown_breakdown() async throws { try await runScenario(Self.scenarios[5]) }
    func test_06_route_adams_to_dupont() async throws { try await runScenario(Self.scenarios[6]) }
    func test_07_route_followup_duration() async throws { try await runScenario(Self.scenarios[7]) }
    func test_08_route_short_walk() async throws { try await runScenario(Self.scenarios[8]) }
    func test_09_landmark_white_house() async throws { try await runScenario(Self.scenarios[9]) }
    func test_10_explore_then_ask_post_office() async throws { try await runScenario(Self.scenarios[10]) }
    func test_11_explore_then_coffee_comparison() async throws { try await runScenario(Self.scenarios[11]) }
    func test_12_landmark_then_route_there() async throws { try await runScenario(Self.scenarios[12]) }
    func test_13_drill_in_then_route_to_item() async throws { try await runScenario(Self.scenarios[13]) }
    func test_14_breakdown_then_pronoun_drill_in() async throws { try await runScenario(Self.scenarios[14]) }
    func test_15_ambiguous_here_needs_clarification() async throws { try await runScenario(Self.scenarios[15]) }
    func test_16_neighborhood_stories_via_wikipedia() async throws { try await runScenario(Self.scenarios[16]) }
    func test_17_wiki_what_is_aspirin() async throws { try await runScenario(Self.scenarios[17]) }
    func test_18_quantum_encryption_must_ground_in_sections() async throws { try await runScenario(Self.scenarios[18]) }
    func test_19_sputnik_explanatory_then_year_followup() async throws { try await runScenario(Self.scenarios[19]) }
}
