// SPDX-License-Identifier: MIT
//
// Headless multi-model eval harness. For each Gemma/Qwen variant
// listed below, runs the full scenario library end-to-end through
// `ChatSession.send(...)` with a `StubZimService` as the tool backend
// and records a scorecard row per `(variant, scenario)` plus a
// per-variant memory profile.
//
// Previously lived as `MultiModelEvalTests.swift` inside the
// `MCPZimChatMacTests` XCTest bundle — which injected into
// `MCPZimChatMac.app`, forced a second `ChatSession` to stand up
// inside the same process (the app's own `@main` state), and caused
// MLX slice-update crashes when the two sessions competed for the
// GPU. Now it's a plain class driven from `ios/MCPZimEval/main.swift`
// so `MCPZimEvalCLI` runs it in its own clean headless process.

import Foundation
import MCPZimKit

@MainActor
final class EvalHarness {

    struct Variant {
        let id: String
        let displayName: String
        let repo: String
        let templateKind: TemplateKind
        /// Path to an already-downloaded MLX weight directory. When set,
        /// the provider loads straight from disk and skips HubClient —
        /// used for locally-quantized variants that aren't published on
        /// HuggingFace (e.g. `Qwen3.5-4B-text-4bit` we convert with
        /// `mlx_lm.convert`). `repo` stays populated for display +
        /// `isCached` path-matching.
        let localDirectory: URL?
        enum TemplateKind { case gemma4, qwenChatML, gemma3 }

        init(
            id: String, displayName: String, repo: String,
            templateKind: TemplateKind, localDirectory: URL? = nil
        ) {
            self.id = id
            self.displayName = displayName
            self.repo = repo
            self.templateKind = templateKind
            self.localDirectory = localDirectory
        }
    }
    static let variants: [Variant] = [
        .init(id: "gemma4-e2b-it-4bit",
              displayName: "Gemma 4 E2B (4-bit · multimodal)",
              repo: "mlx-community/gemma-4-e2b-it-4bit",
              templateKind: .gemma4),
        // Gemma 4 E2B Text-int4 retired from the default matrix on
        // 2026-04-21: it emitted zero tool calls across all 9
        // scenarios in two consecutive runs. The int4 quantization
        // degrades the instruction-tune fidelity enough that the
        // model's `<|tool_call>` emission probability collapses to
        // ~0 when the system turn carries 21 tools. Put it back with
        // `--variant gemma4-e2b-it-4bit-text` if you want to confirm
        // the failure mode or test a re-quantised build.
        // Qwen 3 1.7B retired from the default matrix on 2026-04-21:
        // it never produced a tool call in 3 runs, and when its
        // fallback to `search` hit a missing stub fixture MLX crashed
        // in `broadcast_shapes (1,8,128,128) vs (1,8,129,128)` during
        // the prompt-cache diverge rollback. That fatal aborts the
        // whole process, tanking Qwen 3 4B's run too. Put it back
        // with `--variant qwen3-1-7b-4bit` once we either (a) have
        // subprocess isolation so one crash doesn't kill the run or
        // (b) fix the MLX KV-cache diverge bug upstream.
        .init(id: "qwen3-4b-4bit",
              displayName: "Qwen 3 4B (4-bit)",
              repo: "mlx-community/Qwen3-4B-4bit",
              templateKind: .qwenChatML),
        // Gemma 3 4B IT text-only 4-bit — added 2026-04-23. The
        // multimodal `gemma-3-4b-it-4bit` weights mismatch our vendored
        // mlx-swift-lm 3.31.3 `Gemma3TextModel` `o_proj` shape
        // (expectedShape [2560,128] vs actualShape [2560,256]); the
        // `-text-4b-it-` variant is the text-only checkpoint for the
        // `Gemma3Text` load path. 7/9 on the mac-only Python eval —
        // this harness re-runs it through the real Swift app path.
        .init(id: "gemma3-4b-it-text-4bit",
              displayName: "Gemma 3 4B IT (4-bit · text)",
              repo: "mlx-community/gemma-3-text-4b-it-4bit",
              templateKind: .gemma3),
        // Qwen 3.5 4B text-only at full bf16 precision. On Mac (36 GB
        // RAM) this fits easily and gives us a ceiling-of-quality
        // reference to compare against the 4-bit quant below. On
        // iPhone the ~9 GB weights wouldn't fit — intentionally
        // bypassable by `--variant qwen3-` for phone-bound runs.
        // Works because our vendored mlx-swift-lm 3.31.3 already
        // registers `qwen3_5_text` via `Qwen35TextModel` (PRs
        // #97/#120/#135 upstream).
        .init(id: "qwen35-4b-text-bf16",
              displayName: "Qwen 3.5 4B Text (bf16, unquantized)",
              repo: "principled-intelligence/Qwen3.5-4B-text-only",
              templateKind: .qwenChatML),
        // 2B tier retired from the default matrix on 2026-04-21:
        // every 2B variant (Base, Instruct, custom-quants at 4-bit and
        // 6-bit) capped between 2/9 and 7/9 — the architecture does
        // not hold 21-tool schemas well at this size regardless of
        // quantisation scheme. The 4B quants below are the production
        // line. 2B entries kept in the file as commented-out blocks
        // so they're easy to resurrect if we want to retest with a
        // reduced tool surface or a different model family.
        // Official mlx-community 4-bit of Qwen 3.5 4B (not text-only,
        // bundles same hybrid attention). 5.347 bpw Q4 / group size 64.
        .init(id: "qwen35-4b-4bit",
              displayName: "Qwen 3.5 4B (4-bit)",
              repo: "mlx-community/Qwen3.5-4B-MLX-4bit",
              templateKind: .qwenChatML),
        // MLX-native bf16 variants from mlx-community — properly
        // keyed for the Qwen35Model vision-capable class (our
        // `Qwen35TextModel.sanitize` handles the
        // `model.language_model.*` prefix). Useful upper-bound on
        // what the 4-bit / 2B quantizations cost in accuracy vs
        // the full-precision weights.
        .init(id: "qwen35-4b-mlx-bf16",
              displayName: "Qwen 3.5 4B (bf16, mlx-community)",
              repo: "mlx-community/Qwen3.5-4B-MLX-bf16",
              templateKind: .qwenChatML),
        // ─── 4B quant-scheme exploration (2026-04-21) ────────────
        // Goal: find the smallest and/or highest-quality quant that
        // still clears the 9/9 bar set by the baseline `MLX-4bit`.
        // All sourced from `Qwen/Qwen3.5-4B` upstream; different
        // quantisation recipes probe different points on the
        // quality/size curve.
        //
        // Existing mlx-community ready-to-download:
        .init(id: "qwen35-4b-optiq-4bit",
              displayName: "Qwen 3.5 4B (OptiQ-4bit, mixed-precision)",
              repo: "mlx-community/Qwen3.5-4B-OptiQ-4bit",
              templateKind: .qwenChatML),
        .init(id: "qwen35-4b-3bit",
              displayName: "Qwen 3.5 4B (3-bit)",
              repo: "mlx-community/Qwen3.5-4B-3bit",
              templateKind: .qwenChatML),
        .init(id: "qwen35-4b-mixed-2-6",
              displayName: "Qwen 3.5 4B (mixed 2-bit/6-bit)",
              repo: "mlx-community/Qwen3.5-4B-mixed_2_6",
              templateKind: .qwenChatML),
        .init(id: "qwen35-4b-8bit",
              displayName: "Qwen 3.5 4B (8-bit)",
              repo: "mlx-community/Qwen3.5-4B-MLX-8bit",
              templateKind: .qwenChatML),
        .init(id: "qwen35-4b-6bit",
              displayName: "Qwen 3.5 4B (6-bit)",
              repo: "mlx-community/Qwen3.5-4B-6bit",
              templateKind: .qwenChatML),
    ]

    struct TurnExpect {
        var toolsCalledAny: [String] = []
        var toolsNotCalled: [String] = []
        var responseIncludesAny: [String] = []
        var responseExcludes: [String] = []
    }

    struct Scenario {
        let name: String
        let turns: [(user: String, expect: TurnExpect)]
        /// Optional synthetic host state for tools that read active-route /
        /// current-GPS from `HostStateProvider` (route_status, what_is_here).
        /// Set once per scenario; scenarios that don't touch those tools
        /// leave this nil and the provider returns an empty snapshot.
        let hostState: HostStateSnapshot?

        init(
            name: String,
            turns: [(user: String, expect: TurnExpect)],
            hostState: HostStateSnapshot? = nil
        ) {
            self.name = name
            self.turns = turns
            self.hostState = hostState
        }
    }

    /// Shared mutable cell holding the current scenario's synthetic host
    /// state. The adapter's `HostStateProvider` reads from here; the
    /// harness writes a fresh snapshot before each scenario so
    /// route_status / what_is_here calls see the right world without the
    /// adapter being reconstructed between scenarios.
    final class TestHostStateHolder: @unchecked Sendable {
        private let lock = NSLock()
        private var snapshot: HostStateSnapshot = .init(
            activeRoute: nil, currentLocation: nil
        )
        func set(_ s: HostStateSnapshot) {
            lock.lock(); defer { lock.unlock() }
            snapshot = s
        }
        func current() -> HostStateSnapshot {
            lock.lock(); defer { lock.unlock() }
            return snapshot
        }
    }

    static let scenarios: [Scenario] = [
        // — Existing Phase 1 scenarios (streetzim coverage) —
        .init(name: "restaurants_in_sf", turns: [(
            user: "Are there any good restaurants in San Francisco?",
            expect: TurnExpect(
                toolsCalledAny: ["near_named_place"],
                toolsNotCalled: ["near_places", "route_from_places"],
                responseIncludesAny: ["souvla", "zuni", "nopa", "restaurant"],
                responseExcludes: ["i do not have", "i don't have specific"]
            )
        )]),

        // — Phase 4: new composite tools from TOOL_DESIGN.md —

        // "Something interesting around me" — user is at SF City Hall
        // coords baked into the model preamble. Exercises `nearby_stories`
        // with has_wiki=true → excerpt fetches.
        .init(
            name: "nearby_stories_here",
            turns: [(
                user: "Tell me something interesting about where I am.",
                expect: TurnExpect(
                    toolsCalledAny: ["nearby_stories"],
                    toolsNotCalled: ["search", "near_places"],
                    // The canned excerpts carry these words so the
                    // model's reply (summarising the story list) should
                    // surface at least one.
                    responseIncludesAny: ["transamerica", "coit", "city hall",
                                           "financial district", "telegraph hill"],
                    responseExcludes: ["i don't have", "i cannot"]
                )
            )],
            hostState: HostStateSnapshot(
                activeRoute: nil,
                currentLocation: .init(lat: 37.7793, lon: -122.4193)
            )
        ),

        // "Interesting stories from Palo Alto" — exercises the named-place
        // variant, which geocodes first.
        .init(name: "nearby_stories_palo_alto", turns: [(
            user: "Tell me some interesting stories from Palo Alto.",
            expect: TurnExpect(
                toolsCalledAny: ["nearby_stories_at_place"],
                responseIncludesAny: ["hewlett", "packard", "garage",
                                       "stanford", "palo alto"],
                responseExcludes: []
            )
        )]),

        // "Tell me about Palo Alto" — article_overview with section picker.
        .init(name: "tell_me_about_palo_alto", turns: [(
            user: "Tell me about Palo Alto.",
            expect: TurnExpect(
                toolsCalledAny: ["article_overview"],
                toolsNotCalled: ["search", "get_article",
                                  "list_article_sections"],
                responseIncludesAny: ["palo alto", "silicon valley",
                                       "stanford", "santa clara"],
                responseExcludes: []
            )
        )]),

        // "How is X different from Y" — compare_articles batches both.
        .init(name: "compare_musk_bezos", turns: [(
            user: "How is Elon Musk different from Jeff Bezos?",
            expect: TurnExpect(
                toolsCalledAny: ["compare_articles"],
                responseIncludesAny: ["musk", "bezos", "tesla", "amazon",
                                       "spacex", "blue origin"],
                responseExcludes: []
            )
        )]),

        // "How have the US and Iran gotten along" — compare_articles with
        // the relations-article probe. On 2026-04-21 we folded
        // `article_relationship` into `compare_articles` so there's one
        // "two entities" tool. The legacy name is still accepted via a
        // dispatch shim but the model should pick `compare_articles`.
        .init(name: "relations_us_iran", turns: [(
            user: "How have the United States and Iran gotten along historically?",
            expect: TurnExpect(
                toolsCalledAny: ["compare_articles", "article_relationship"],
                responseIncludesAny: ["iran", "united states", "relations",
                                       "1979", "revolution", "sanctions"],
                responseExcludes: []
            )
        )]),

        // "Read me the article on the HP Garage" — narrate_article +
        // pass-through. The assistant reply is the (stub) article body
        // verbatim, not a model-generated summary.
        .init(name: "narrate_hp_garage", turns: [(
            user: "Please read me the full article about the HP Garage.",
            expect: TurnExpect(
                toolsCalledAny: ["narrate_article"],
                toolsNotCalled: ["get_article_section", "article_overview"],
                // Stub article body contains all of these — pass-through
                // means the model's output IS the body, so these are a
                // strong regression signal if pass-through ever breaks
                // and the model starts paraphrasing instead.
                responseIncludesAny: ["hewlett", "packard", "addison avenue",
                                       "birthplace of silicon valley"],
                responseExcludes: ["i don't know", "i cannot"]
            )
        )]),

        // "Where am I?" — what_is_here needs injected GPS (host state).
        .init(
            name: "what_is_here_in_sf",
            turns: [(
                user: "Where am I?",
                expect: TurnExpect(
                    toolsCalledAny: ["what_is_here"],
                    responseIncludesAny: ["san francisco", "civic center",
                                           "california"],
                    responseExcludes: []
                )
            )],
            hostState: HostStateSnapshot(
                activeRoute: nil,
                currentLocation: .init(lat: 37.7793, lon: -122.4193)
            )
        ),

        // "How much longer?" — route_status reads the injected active
        // route. Single turn because ChatSession doesn't auto-populate
        // ZimfoContext from route_from_places today; pre-seeding the
        // host state tests the tool in isolation.
        .init(
            name: "how_much_longer",
            turns: [(
                user: "How much longer until I get there?",
                expect: TurnExpect(
                    toolsCalledAny: ["route_status"],
                    // Synthetic route is halfway done → answer should
                    // mention a partial duration or a next turn.
                    responseIncludesAny: ["minute", "min", "kilometre", "km",
                                           "mile", "us-101", "el camino"],
                    responseExcludes: ["no active route",
                                        "plan a route first"]
                )
            )],
            hostState: HostStateSnapshot(
                activeRoute: EvalHarness.demoActiveRoute,
                currentLocation: .init(lat: 37.5, lon: -122.2)
            )
        ),
    ]

    /// Synthetic "Palo Alto → San Francisco" active route used by the
    /// `how_much_longer` scenario. Halving the polyline to three points
    /// keeps the snap-to-vertex math simple: the test's GPS sits on the
    /// middle vertex so `route_status` returns ≈50% done.
    ///
    /// Computed (not stored) because Swift forbids one static stored
    /// property's initializer from referencing another via `Self.` —
    /// and `scenarios` needs this value.
    static var demoActiveRoute: RouteSnapshot {
        RouteSnapshot(
            origin: .init(lat: 37.4419, lon: -122.1430),
            destination: .init(lat: 37.7749, lon: -122.4194),
            originName: "Palo Alto",
            destinationName: "San Francisco",
            totalDistanceMeters: 51_000,
            totalDurationSeconds: 3_000,
            polyline: [
                .init(lat: 37.4419, lon: -122.1430),
                .init(lat: 37.5, lon: -122.2),
                .init(lat: 37.7749, lon: -122.4194),
            ],
            cumulativeDistanceMeters: [0, 25_500, 51_000],
            turnByTurn: [
                "Head north on El Camino Real for 5.0 km (~8 min)",
                "Take US-101 N for 40.0 km (~35 min)",
                "Exit to 4th St for 1.0 km (~5 min)",
            ]
        )
    }

    struct Scorecard {
        struct Row {
            let variant: String
            let scenario: String
            let toolsCalled: [String]
            let toolsOk: Bool
            let responseOk: Bool
            let responseSnippet: String
            let errors: [String]
        }
        var rows: [Row] = []
        var memory: [MemorySummary] = []

        func markdown() -> String {
            var out = "\n## Eval scorecard\n\n"
            out += "| variant | scenario | tools ✓ | response ✓ | called | snippet |\n"
            out += "|---|---|---|---|---|---|\n"
            for r in rows {
                let called = r.toolsCalled.joined(separator: ",")
                let snip = r.responseSnippet
                    .replacingOccurrences(of: "\n", with: " ")
                    .prefix(250)
                out += "| \(r.variant) | \(r.scenario) | \(r.toolsOk ? "✓" : "✗") | \(r.responseOk ? "✓" : "✗") | \(called) | \(snip) |\n"
            }
            out += "\n### Memory\n\n```\n"
            for m in memory { out += m.scorecardRow() + "\n" }
            out += "```\n"
            return out
        }
    }

    struct RunOptions {
        /// Restrict to variants whose id contains one of these substrings.
        /// Empty = no filter (all cached variants run).
        var variantFilter: [String] = []
        /// Restrict to scenarios whose name contains one of these substrings.
        /// Empty = no filter.
        var scenarioFilter: [String] = []
    }

    struct RunResult {
        let scorecard: Scorecard
        let scenariosWithNoWinner: [String]
    }

    func run(_ options: RunOptions = .init()) async throws -> RunResult {
        var scorecard = Scorecard()
        let matchedVariants = Self.variants.filter { v in
            options.variantFilter.isEmpty
                || options.variantFilter.contains(where: { v.id.contains($0) })
        }
        for v in matchedVariants {
            if let dir = v.localDirectory {
                // Locally-quantized variant — just check the weights
                // file exists on disk under the override path.
                let weights = dir.appendingPathComponent("model.safetensors")
                guard FileManager.default.fileExists(atPath: weights.path) else {
                    print("[eval] skipping \(v.displayName): no weights at \(weights.path)")
                    continue
                }
            } else {
                guard isCached(repo: v.repo) else {
                    print("[eval] skipping \(v.displayName): weights not cached")
                    continue
                }
            }
            let (rows, memSummary) = try await runVariant(v, options: options)
            scorecard.rows.append(contentsOf: rows)
            scorecard.memory.append(memSummary)
        }

        let scenariosRun = Self.scenarios.filter { s in
            options.scenarioFilter.isEmpty
                || options.scenarioFilter.contains(where: { s.name.contains($0) })
        }
        var losers: [String] = []
        for scen in scenariosRun {
            let wins = scorecard.rows.filter {
                $0.scenario == scen.name && $0.toolsOk && $0.responseOk
            }
            if wins.isEmpty { losers.append(scen.name) }
        }
        return RunResult(scorecard: scorecard, scenariosWithNoWinner: losers)
    }

    // MARK: -

    private func runVariant(
        _ v: Variant, options: RunOptions
    ) async throws -> (rows: [Scorecard.Row], memory: MemorySummary) {
        let probe = MemoryProbe(variant: v.displayName)
        await probe.sample("baseline")

        let template: any ModelTemplate = {
            switch v.templateKind {
            case .gemma4:      return Gemma4Template()
            case .qwenChatML:  return QwenChatMLTemplate()
            case .gemma3:      return Gemma3Template()
            }
        }()
        let provider = Gemma4Provider(
            id: v.id, displayName: v.displayName, huggingFaceRepo: v.repo,
            template: template,
            localWeightsDirectory: v.localDirectory
        )
        let fixture = Self.fixtureForEval()
        let stub = StubZimService(fixture: fixture)
        // Shared mutable cell the installed HostStateProvider reads from.
        // Rewritten per-scenario so route_status / what_is_here see
        // scenario-specific route + GPS state without reconstructing
        // the adapter between scenarios (which would reload weights).
        let hostHolder = TestHostStateHolder()
        let adapter = MCPToolAdapter(
            service: stub, hasStreetzim: true,
            surface: .conversational,
            categoryVocabulary: Self.categoryVocabulary
        )
        await adapter.installHostStateProvider { [hostHolder] in
            hostHolder.current()
        }

        let session = ChatSession.forTesting(
            providers: [provider], adapter: adapter, initialModelId: v.id
        )
        session.maxDebugEntries = 20_000
        try await provider.load()
        await probe.sample("post_load")

        var rows: [Scorecard.Row] = []
        let scenariosToRun = Self.scenarios.filter { s in
            options.scenarioFilter.isEmpty
                || options.scenarioFilter.contains(where: { s.name.contains($0) })
        }
        for scen in scenariosToRun {
            // Swap in this scenario's synthetic host state (or clear it
            // back to the empty default for scenarios that don't care).
            hostHolder.set(scen.hostState ?? .init(
                activeRoute: nil, currentLocation: nil
            ))
            // Mirror the same GPS fix into `session.currentLocation` so
            // `composeSystemMessage` bakes concrete lat/lon into the
            // preamble. Without this, the location-recipes block says
            // "permission hasn't resolved" and the model has no coords
            // to pass to `nearby_stories(lat, lon)` / `near_places`.
            if let loc = scen.hostState?.currentLocation {
                session.currentLocation = (lat: loc.lat, lon: loc.lon)
            } else {
                session.currentLocation = nil
            }
            let row = try await runScenario(
                scen, session: session, variant: v, probe: probe
            )
            rows.append(row)
            session.resetConversation()
        }

        let summary = await probe.summary()
        return (rows, summary)
    }

    private func runScenario(
        _ scen: Scenario,
        session: ChatSession,
        variant: Variant,
        probe: MemoryProbe
    ) async throws -> Scorecard.Row {
        var allToolsCalled: [String] = []
        var lastAssistant = ""
        var errors: [String] = []
        var toolsOk = true
        var responseOk = true

        for (ti, turn) in scen.turns.enumerated() {
            await probe.startContinuous(
                tagPrefix: "decode.\(scen.name).t\(ti)", intervalMs: 100
            )
            session.send(turn.user)
            let deadline = Date().addingTimeInterval(120)
            while session.isGenerating && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(200))
            }
            await probe.stop()
            await probe.sample("post_turn.\(scen.name).t\(ti)")

            guard !session.isGenerating else {
                errors.append("turn \(ti) did not complete within 120s")
                toolsOk = false; responseOk = false
                break
            }

            // Collect what just happened on the last assistant message.
            let lastIdx = session.messages.lastIndex { $0.role == .assistant } ?? -1
            let assistant = lastIdx >= 0 ? session.messages[lastIdx] : nil
            lastAssistant = assistant?.text ?? ""
            let turnTools = assistant?.toolCalls.map(\.name) ?? []
            allToolsCalled.append(contentsOf: turnTools)

            // Dump the debug log tail + assistant text on the first
            // scenario so we can see why a zero-output variant failed.
            print("[\(variant.displayName) / \(scen.name) t\(ti)] assistant.text.count=\(lastAssistant.count), tools=\(turnTools)")
            print("[\(variant.displayName) / debug tail]")
            for e in session.debugEntries.suffix(40) {
                print("  [\(e.category)] \(e.message.prefix(240))")
            }
            if !lastAssistant.isEmpty {
                print("[\(variant.displayName) assistant.text]\n\(lastAssistant.prefix(600))\n[/assistant]")
            }

            let unique = Set(turnTools)
            // `toolsCalledAny` semantics = "the model must have called at
            // LEAST ONE of these tools". Previously the loop required
            // EVERY listed tool to appear, which contradicts the name and
            // made it impossible to list equivalent alternatives (e.g.
            // accepting either the new `compare_articles` or the legacy
            // `article_relationship` for the relations scenario — only
            // one gets called at runtime). Fixed 2026-04-21.
            if !turn.expect.toolsCalledAny.isEmpty,
               !turn.expect.toolsCalledAny.contains(where: { unique.contains($0) })
            {
                toolsOk = false
                errors.append(
                    "t\(ti): response called none of "
                    + "\(turn.expect.toolsCalledAny) (called=\(turnTools))"
                )
            }
            for banned in turn.expect.toolsNotCalled {
                if unique.contains(banned) {
                    toolsOk = false
                    errors.append("t\(ti): banned tool '\(banned)' was called")
                }
            }
            let lower = lastAssistant.lowercased()
            if !turn.expect.responseIncludesAny.isEmpty,
               !turn.expect.responseIncludesAny.contains(where: { lower.contains($0.lowercased()) })
            {
                responseOk = false
                errors.append("t\(ti): response didn't include any of \(turn.expect.responseIncludesAny)")
            }
            for banned in turn.expect.responseExcludes {
                if lower.contains(banned.lowercased()) {
                    responseOk = false
                    errors.append("t\(ti): response contained banned '\(banned)'")
                }
            }
        }

        return Scorecard.Row(
            variant: variant.displayName,
            scenario: scen.name,
            toolsCalled: allToolsCalled,
            toolsOk: toolsOk,
            responseOk: responseOk,
            responseSnippet: String(lastAssistant.prefix(500)),
            errors: errors
        )
    }

    // MARK: - Fixture

    /// The composite fixture the harness runs against. Organised by
    /// scenario so each block is self-contained — easy to grep to find
    /// which scenario owns a given key and easy to extend when a new
    /// scenario lands.
    static func fixtureForEval() -> StubZimService.Fixture {
        var f = StubZimService.Fixture()
        Self.addRestaurantsInSFFixture(&f)
        Self.addNearbyStoriesHereFixture(&f)
        Self.addNearbyStoriesAtPlaceFixture(&f)
        Self.addArticleOverviewPaloAltoFixture(&f)
        Self.addCompareMuskBezosFixture(&f)
        Self.addRelationsUSIranFixture(&f)
        Self.addNarrateHPGarageFixture(&f)
        Self.addWhatIsHereInSFFixture(&f)
        return f
    }

    // Kept under the old name for any out-of-tree callers that might
    // reference it; `fixtureForEval()` is the canonical entry point now.
    @available(*, deprecated, renamed: "fixtureForEval")
    static func fixtureForPhase1() -> StubZimService.Fixture {
        fixtureForEval()
    }

    // MARK: Fixture builders — one per scenario

    /// "Are there any good restaurants in San Francisco?" → near_named_place.
    private static func addRestaurantsInSFFixture(_ f: inout StubZimService.Fixture) {
        let sfCenter = Place(
            name: "San Francisco", kind: "place",
            lat: 37.7749, lon: -122.4194,
            subtype: "city", location: "California, USA"
        )
        let restaurants: [(place: Place, distanceMeters: Double)] = [
            (Place(name: "Souvla", kind: "poi", lat: 37.7767, lon: -122.4334,
                   subtype: "restaurant", location: "Hayes Valley"),
             distanceMeters: 1350),
            (Place(name: "Zuni Cafe", kind: "poi", lat: 37.7726, lon: -122.4229,
                   subtype: "restaurant", location: "Market Street"),
             distanceMeters: 440),
            (Place(name: "Nopa", kind: "poi", lat: 37.7747, lon: -122.4376,
                   subtype: "restaurant", location: "Divisadero"),
             distanceMeters: 1600),
        ]
        let nearResult = NearPlacesResult(
            totalInRadius: restaurants.count,
            breakdown: ["restaurant": restaurants.count],
            results: restaurants
        )
        for kinds in [["restaurant"], nil, []] as [[String]?] {
            f.nearNamedPlace[
                StubZimService.keyNearNamedPlace(place: "San Francisco", kinds: kinds)
            ] = .init(resolved: sfCenter, result: nearResult)
        }
    }

    /// "Tell me something interesting about where I am." User is at the
    /// SF City Hall coords baked into the scenario's hostState. The tool
    /// fans out to articleByTitle for each wiki-linked POI.
    private static func addNearbyStoriesHereFixture(_ f: inout StubZimService.Fixture) {
        let sfPOIs: [(place: Place, distanceMeters: Double)] = [
            (Place(
                name: "San Francisco City Hall", kind: "poi",
                lat: 37.7793, lon: -122.4193,
                subtype: "historic", location: "Civic Center",
                wiki: "en:San Francisco City Hall", wikidata: "Q7415906"
            ), distanceMeters: 20),
            (Place(
                name: "Transamerica Pyramid", kind: "poi",
                lat: 37.7952, lon: -122.4028,
                subtype: "tourism", location: "Financial District",
                wiki: "en:Transamerica Pyramid", wikidata: "Q212134"
            ), distanceMeters: 2400),
            (Place(
                name: "Coit Tower", kind: "poi",
                lat: 37.8024, lon: -122.4058,
                subtype: "tourism", location: "Telegraph Hill",
                wiki: "en:Coit Tower", wikidata: "Q860776"
            ), distanceMeters: 3000),
        ]
        let res = NearPlacesResult(
            totalInRadius: sfPOIs.count,
            breakdown: ["historic": 1, "tourism": 2],
            results: sfPOIs
        )
        // `nearby_stories(lat=37.7793, lon=-122.4193)` with no kinds →
        // keyed coord rounded to 3 decimals. Cover the common kinds-
        // filter variants models reach for so the stub isn't the
        // thing failing the scenario.
        for k in [nil, ["tourism"], ["historic"], ["tourism", "historic"]] as [[String]?] {
            f.nearPlaces[
                StubZimService.keyNearPlaces(lat: 37.7793, lon: -122.4193, kinds: k)
            ] = .init(result: res)
        }

        // Lead articles for each wiki-linked POI. `articleByTitle` key
        // is `(title, section)` lowercased; `nearby_stories` always
        // asks for section="lead".
        addLeadArticle(
            &f, wikiTag: "en:San Francisco City Hall",
            zim: "wikipedia_en.zim", path: "A/San_Francisco_City_Hall",
            title: "San Francisco City Hall",
            body: "San Francisco City Hall is the seat of government for the " +
                "City and County of San Francisco, California. Re-opened in " +
                "1915 in its open space area in the city's Civic Center, it is " +
                "a Beaux-Arts monument to the City Beautiful movement that " +
                "epitomized the high-minded American Renaissance of the 1880s " +
                "to 1917. The present building replaced an earlier City Hall " +
                "that was destroyed during the 1906 earthquake."
        )
        addLeadArticle(
            &f, wikiTag: "en:Transamerica Pyramid",
            zim: "wikipedia_en.zim", path: "A/Transamerica_Pyramid",
            title: "Transamerica Pyramid",
            body: "The Transamerica Pyramid is a 48-story futurist skyscraper " +
                "in San Francisco, California, at 600 Montgomery Street in the " +
                "Financial District. Designed by William Pereira and completed " +
                "in 1972, it was the tallest building in San Francisco from " +
                "completion until 2018 when the Salesforce Tower opened."
        )
        addLeadArticle(
            &f, wikiTag: "en:Coit Tower",
            zim: "wikipedia_en.zim", path: "A/Coit_Tower",
            title: "Coit Tower",
            body: "Coit Tower is a 210-foot (64 m) slender white concrete tower " +
                "in the Telegraph Hill neighborhood of San Francisco, California, " +
                "offering panoramic views over the city and the bay. Built between " +
                "1932 and 1933 using Lillie Hitchcock Coit's bequest to beautify " +
                "the city, it is decorated with a series of fresco murals."
        )
    }

    /// "Tell me some interesting stories from Palo Alto."
    /// First a geocode("Palo Alto") → then nearPlaces(lat, lon, hasWiki=true).
    private static func addNearbyStoriesAtPlaceFixture(_ f: inout StubZimService.Fixture) {
        let paloAltoCenter = Place(
            name: "Palo Alto", kind: "place",
            lat: 37.4419, lon: -122.1430,
            subtype: "city", location: "California, USA"
        )
        f.geocode[StubZimService.keyGeocode(query: "Palo Alto")] = [paloAltoCenter]

        let paPOIs: [(place: Place, distanceMeters: Double)] = [
            (Place(
                name: "HP Garage", kind: "poi",
                lat: 37.4445, lon: -122.1498,
                subtype: "historic", location: "Addison Avenue",
                wiki: "en:HP Garage", wikidata: "Q2720242"
            ), distanceMeters: 750),
            (Place(
                name: "Stanford University", kind: "poi",
                lat: 37.4275, lon: -122.1697,
                subtype: "tourism", location: "Palo Alto",
                wiki: "en:Stanford University", wikidata: "Q41506"
            ), distanceMeters: 2800),
        ]
        let paResult = NearPlacesResult(
            totalInRadius: paPOIs.count,
            breakdown: ["historic": 1, "tourism": 1],
            results: paPOIs
        )
        // Multiple key variants so the fixture matches whether the
        // model calls `nearby_stories_at_place` without a kinds filter
        // (original flow) OR with `kinds=["tourism"]` / `kinds=["historic"]`
        // (Gemma-4 multimodal's observed behavior — it reaches for a
        // category filter despite the tool schema not requiring one).
        for k in [nil, ["tourism"], ["historic"], ["tourism", "historic"]] as [[String]?] {
            f.nearPlaces[
                StubZimService.keyNearPlaces(
                    lat: paloAltoCenter.lat, lon: paloAltoCenter.lon, kinds: k
                )
            ] = .init(result: paResult)
        }

        addLeadArticle(
            &f, wikiTag: "en:HP Garage",
            zim: "wikipedia_en.zim", path: "A/HP_Garage",
            title: "HP Garage",
            body: "The HP Garage is a private museum where the company Hewlett-Packard " +
                "was founded. It is located at 367 Addison Avenue in Palo Alto, " +
                "California. Considered the birthplace of Silicon Valley, the garage " +
                "was designated a California Historical Landmark in 1989."
        )
        addLeadArticle(
            &f, wikiTag: "en:Stanford University",
            zim: "wikipedia_en.zim", path: "A/Stanford_University",
            title: "Stanford University",
            body: "Stanford University is a private research university in Stanford, " +
                "California. Stanford was founded in 1885 by Leland and Jane Stanford " +
                "in memory of their only child, Leland Stanford Jr. The university " +
                "has produced many successful companies, entrepreneurs, and Nobel " +
                "laureates across its history in Silicon Valley."
        )
    }

    /// "Tell me about Palo Alto." → article_overview. Needs a title lookup
    /// AND a full section list (the overview picker iterates sections).
    private static func addArticleOverviewPaloAltoFixture(_ f: inout StubZimService.Fixture) {
        let lead = ArticleSection(
            title: "", level: 0,
            text: "Palo Alto is a charter city in the northwestern corner of Santa Clara " +
                "County, California, named after a coastal redwood tree known as El Palo " +
                "Alto. The city was established by Leland Stanford when he founded " +
                "Stanford University in 1885, and has grown into one of the main " +
                "headquarters cities for companies in Silicon Valley. As of 2020, the " +
                "population was 68,572."
        )
        let history = ArticleSection(
            title: "History", level: 2,
            text: "The history of Palo Alto is tightly bound to the history of Stanford " +
                "University, founded in 1891. Before European settlement the area was " +
                "home to the Muwekma Ohlone people. The HP Garage at 367 Addison Avenue " +
                "is widely considered the birthplace of Silicon Valley."
        )
        let geography = ArticleSection(
            title: "Geography", level: 2,
            text: "Palo Alto lies at the northern end of Silicon Valley, adjacent to " +
                "Stanford University and bordered by the San Francisco Bay to the east. " +
                "The city spans 25.8 square miles (67 km²), much of it upland terrain " +
                "in the Santa Cruz Mountains foothills."
        )
        let economy = ArticleSection(
            title: "Economy", level: 2,
            text: "Palo Alto's economy is dominated by technology. Tesla, HP, and Palantir " +
                "all maintain significant presences in the city, and the University Avenue " +
                "corridor hosts dozens of venture-capital firms."
        )

        f.articleByTitle[StubZimService.keyArticleByTitle(title: "Palo Alto", section: "lead")] =
            .init(zim: "wikipedia_en.zim", path: "A/Palo_Alto",
                  title: "Palo Alto", section: lead)
        f.articleSections[StubZimService.keyArticleSections(path: "A/Palo_Alto")] =
            .init(zim: "wikipedia_en.zim", title: "Palo Alto",
                  sections: [lead, history, geography, economy])
    }

    /// "How is Elon Musk different from Jeff Bezos?" → compare_articles.
    /// Each title resolves via articleByTitle + articleSections.
    private static func addCompareMuskBezosFixture(_ f: inout StubZimService.Fixture) {
        // Elon Musk
        let muskLead = ArticleSection(
            title: "", level: 0,
            text: "Elon Reeve Musk is a businessman known for his key roles in Tesla, " +
                "SpaceX, and the X Corp (formerly Twitter). Since 2025 he has been the " +
                "wealthiest person in the world, with a net worth valued at approximately " +
                "US$400 billion."
        )
        let muskCareer = ArticleSection(
            title: "Career", level: 2,
            text: "Musk co-founded the online bank that became PayPal in 1999, sold to " +
                "eBay in 2002. He founded SpaceX in 2002 to reduce space-transportation " +
                "costs, and joined Tesla in 2004, leading its electric-vehicle pivot."
        )
        let muskVentures = ArticleSection(
            title: "Ventures", level: 2,
            text: "Beyond Tesla and SpaceX, Musk has founded or co-founded Neuralink (brain " +
                "computer interfaces), The Boring Company (tunneling), xAI (artificial " +
                "intelligence), and has publicly led the platform formerly known as Twitter."
        )
        f.articleByTitle[StubZimService.keyArticleByTitle(title: "Elon Musk", section: "lead")] =
            .init(zim: "wikipedia_en.zim", path: "A/Elon_Musk",
                  title: "Elon Musk", section: muskLead)
        f.articleSections[StubZimService.keyArticleSections(path: "A/Elon_Musk")] =
            .init(zim: "wikipedia_en.zim", title: "Elon Musk",
                  sections: [muskLead, muskCareer, muskVentures])

        // Jeff Bezos
        let bezosLead = ArticleSection(
            title: "", level: 0,
            text: "Jeffrey Preston Bezos is an American businessman and investor, best " +
                "known as the founder, executive chairman, and former president and CEO " +
                "of Amazon. He also founded the aerospace manufacturer Blue Origin and " +
                "acquired the newspaper The Washington Post in 2013."
        )
        let bezosCareer = ArticleSection(
            title: "Career", level: 2,
            text: "Bezos founded Amazon in 1994 as an online bookstore, expanding it " +
                "into one of the world's largest marketplaces. Under his leadership " +
                "Amazon launched AWS (2006), Prime (2005), and Kindle (2007)."
        )
        let bezosVentures = ArticleSection(
            title: "Ventures", level: 2,
            text: "Bezos founded Blue Origin in 2000 with a focus on reusable launch " +
                "vehicles and eventual human spaceflight. Bezos Expeditions, his " +
                "personal investment vehicle, has backed early-stage companies including " +
                "Google, Twitter, and Airbnb."
        )
        f.articleByTitle[StubZimService.keyArticleByTitle(title: "Jeff Bezos", section: "lead")] =
            .init(zim: "wikipedia_en.zim", path: "A/Jeff_Bezos",
                  title: "Jeff Bezos", section: bezosLead)
        f.articleSections[StubZimService.keyArticleSections(path: "A/Jeff_Bezos")] =
            .init(zim: "wikipedia_en.zim", title: "Jeff Bezos",
                  sections: [bezosLead, bezosCareer, bezosVentures])
    }

    /// "How have the US and Iran gotten along?" → article_relationship.
    /// Probes "Iran–United States relations" first (alphabetical + en-dash).
    private static func addRelationsUSIranFixture(_ f: inout StubZimService.Fixture) {
        let lead = ArticleSection(
            title: "", level: 0,
            text: "Relations between Iran and the United States have been hostile since " +
                "the Iranian Revolution of 1979. The United States has not had formal " +
                "diplomatic relations with Iran since April 1980, and the two countries " +
                "have been in a sustained state of confrontation over nuclear policy, " +
                "sanctions, and regional influence."
        )
        let hist = ArticleSection(
            title: "History", level: 2,
            text: "Before 1979, Iran under the Shah was one of the closest US allies in " +
                "the Middle East. The 1979 Islamic Revolution ended that partnership; the " +
                "subsequent hostage crisis (November 1979 to January 1981) fractured " +
                "relations. Since then, episodes including the Iran-Iraq War, the 2003 " +
                "invasion of Iraq, and the 2015 Joint Comprehensive Plan of Action have " +
                "shaped the relationship."
        )
        let sanctions = ArticleSection(
            title: "Sanctions", level: 2,
            text: "The United States has imposed sanctions on Iran since the 1979 hostage " +
                "crisis, broadened significantly under the Clinton administration and " +
                "tightened again after the US withdrew from the JCPOA in 2018. Sanctions " +
                "target Iran's banking sector, oil exports, and individuals linked to " +
                "the Islamic Revolutionary Guard Corps."
        )
        // Canonical en-dash form is what my helper tries first. Also
        // cover the hyphen variant in case the stub gets queried with it.
        for title in ["Iran–United States relations", "Iran-United States relations"] {
            f.articleByTitle[StubZimService.keyArticleByTitle(title: title, section: "lead")] =
                .init(zim: "wikipedia_en.zim", path: "A/Iran–United_States_relations",
                      title: "Iran–United States relations", section: lead)
        }
        f.articleSections[
            StubZimService.keyArticleSections(path: "A/Iran–United_States_relations")
        ] = .init(zim: "wikipedia_en.zim", title: "Iran–United States relations",
                  sections: [lead, hist, sanctions])
    }

    /// "Read me the full article on the HP Garage." → narrate_article.
    /// Same article used by nearby_stories_at_place, but narrate_article
    /// needs the full section list to concatenate into TTS text — so the
    /// articleSections fixture is added here even though articleByTitle
    /// was set up earlier.
    private static func addNarrateHPGarageFixture(_ f: inout StubZimService.Fixture) {
        // nearby_stories fixture already defined the lead. narrate_article
        // asks for "lead" too, then fetches the full outline separately.
        let lead = ArticleSection(
            title: "", level: 0,
            text: "The HP Garage is a private museum where the company Hewlett-Packard " +
                "was founded. It is located at 367 Addison Avenue in Palo Alto, " +
                "California. Considered the birthplace of Silicon Valley, the garage " +
                "was designated a California Historical Landmark in 1989."
        )
        let history = ArticleSection(
            title: "History", level: 2,
            text: "Bill Hewlett and Dave Packard began their partnership in the garage " +
                "at 367 Addison Avenue in 1939. Their first successful product, an audio " +
                "oscillator called the HP200A, was built there and sold to Walt Disney " +
                "Studios for the production of the film Fantasia. The founding of the " +
                "partnership in the garage is widely considered the moment Silicon Valley " +
                "began."
        )
        let preservation = ArticleSection(
            title: "Preservation", level: 2,
            text: "In 2000, Hewlett-Packard bought the property and restored it as a " +
                "museum. The garage, the adjacent one-story house and the shed at the " +
                "back of the property were all restored to their 1939 condition and " +
                "collectively designated California Historical Landmark No. 976."
        )
        // narrate_article's `articleByTitle` fixture for "HP Garage"
        // was already added by nearby_stories_at_place; override here
        // just to be explicit about ownership (same data).
        f.articleByTitle[StubZimService.keyArticleByTitle(title: "HP Garage", section: "lead")] =
            .init(zim: "wikipedia_en.zim", path: "A/HP_Garage",
                  title: "HP Garage", section: lead)
        f.articleSections[StubZimService.keyArticleSections(path: "A/HP_Garage")] =
            .init(zim: "wikipedia_en.zim", title: "HP Garage",
                  sections: [lead, history, preservation])
    }

    /// "Where am I?" → what_is_here. Uses the scenario's injected GPS
    /// (via hostState.currentLocation) and reverse-geocodes by scanning
    /// nearPlaces(kinds=["place"]).
    private static func addWhatIsHereInSFFixture(_ f: inout StubZimService.Fixture) {
        let sfAdmin = Place(
            name: "San Francisco", kind: "place",
            lat: 37.7793, lon: -122.4193,
            subtype: "city", location: "California, USA",
            wiki: "en:San Francisco", wikidata: "Q62"
        )
        let civicCenter = Place(
            name: "Civic Center", kind: "place",
            lat: 37.7793, lon: -122.4193,
            subtype: "neighbourhood", location: "San Francisco, California"
        )
        let res = NearPlacesResult(
            totalInRadius: 2,
            breakdown: ["city": 1, "neighbourhood": 1],
            // Civic Center sits right on the GPS fix → becomes nearest.
            results: [
                (civicCenter, distanceMeters: 5),
                (sfAdmin, distanceMeters: 10),
            ]
        )
        f.nearPlaces[
            StubZimService.keyNearPlaces(
                lat: 37.7793, lon: -122.4193, kinds: ["place"]
            )
        ] = .init(result: res)

        // Some older streetzims carry neighbourhoods without wiki links;
        // the tool falls back to the next hit with a wiki, which is the
        // "San Francisco" city record. We don't exercise that path here
        // — Civic Center is non-wiki, so `what_is_here` just returns
        // name + admin_area without a wiki summary. That's a valid
        // result shape for the expectation set (response mentions
        // "San Francisco" + "Civic Center").

        // For the optional wiki-summary branch — when the nearest hit
        // DOES carry a wiki tag — cover en:San Francisco too so a model
        // that re-orders the response still gets text.
        addLeadArticle(
            &f, wikiTag: "en:San Francisco",
            zim: "wikipedia_en.zim", path: "A/San_Francisco",
            title: "San Francisco",
            body: "San Francisco, officially the City and County of San Francisco, is " +
                "a commercial, financial, and cultural center in Northern California. " +
                "The city proper is the fourth most populous city in California, " +
                "with a population of 808,437 residents as of 2022."
        )
    }

    /// Helper: register a "lead" articleByTitle row. The wiki-tag form
    /// ("en:Foo") is what near_places hands back in the `wiki` field;
    /// the bare title ("Foo") is what the model often passes when
    /// echoing. Cover both so scenarios don't care which the model used.
    private static func addLeadArticle(
        _ f: inout StubZimService.Fixture,
        wikiTag: String, zim: String, path: String,
        title: String, body: String
    ) {
        let section = ArticleSection(title: "", level: 0, text: body)
        let response = StubZimService.ArticleByTitleResponse(
            zim: zim, path: path, title: title, section: section
        )
        // Wiki tag form (what `Place.wiki` carries verbatim).
        f.articleByTitle[StubZimService.keyArticleByTitle(title: wikiTag, section: "lead")] = response
        // Bare title form (what the model tends to use when echoing).
        f.articleByTitle[StubZimService.keyArticleByTitle(title: title, section: "lead")] = response
    }

    static let categoryVocabulary = [
        "restaurant", "cafe", "museum", "park", "library",
        "post_office", "atm", "fuel", "hotel", "hospital",
        "pharmacy", "school", "university", "college", "bank",
        "place_of_worship", "tourism",
    ]

    // MARK: -

    private func hfCacheRoot() -> String {
        if let override = ProcessInfo.processInfo.environment["MCPZIM_HF_CACHE"],
           !override.isEmpty
        { return override }
        if let home = ProcessInfo.processInfo.environment["HOME"],
           !home.contains("/Library/Containers/"), !home.isEmpty
        { return "\(home)/.cache/huggingface/hub" }
        return "/Users/\(NSUserName())/.cache/huggingface/hub"
    }

    private func isCached(repo: String) -> Bool {
        let dir = "\(hfCacheRoot())/models--\(repo.replacingOccurrences(of: "/", with: "--"))"
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue
        else { return false }
        let snapshots = "\(dir)/snapshots"
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: snapshots),
              let first = entries.first
        else { return false }
        let snapshotDir = "\(snapshots)/\(first)"
        // Accept either single-shard (`model.safetensors`) or multi-shard
        // (`model.safetensors.index.json` + `model-0000N-of-NNNNN.safetensors`)
        // layouts. Larger models like `mlx-community/Qwen3.5-4B-MLX-bf16`
        // are multi-shard and the old single-file check missed them.
        let fm = FileManager.default
        if fm.fileExists(atPath: "\(snapshotDir)/model.safetensors") { return true }
        if fm.fileExists(atPath: "\(snapshotDir)/model.safetensors.index.json") {
            return true
        }
        if let files = try? fm.contentsOfDirectory(atPath: snapshotDir),
           files.contains(where: {
               $0.hasPrefix("model-") && $0.hasSuffix(".safetensors")
           })
        { return true }
        return false
    }
}
