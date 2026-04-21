// SPDX-License-Identifier: MIT
//
// Phase 1 of the multi-model eval harness described in
// `EVAL_HARNESS.md`. For each Gemma 4 variant the app ships, runs a
// scenario library end-to-end through `ChatSession.send(...)` with a
// `StubZimService` as the tool backend, and records a scorecard row
// per `(variant, scenario)` plus a per-variant memory profile.
//
// v1 scope: 1 scenario × 2 variants. Phase 2 abstracts model templates
// so Qwen can plug in; Phase 3 adds more scenarios. Skip gracefully
// if the model's HF weights aren't in the local cache.

import XCTest
import MCPZimKit
@testable import MCPZimChatMac

@MainActor
final class MultiModelEvalTests: XCTestCase {

    struct Variant {
        let id: String
        let displayName: String
        let repo: String
        let templateKind: TemplateKind
        enum TemplateKind { case gemma4, qwenChatML }
    }
    static let variants: [Variant] = [
        .init(id: "gemma4-e2b-it-4bit",
              displayName: "Gemma 4 E2B (4-bit · multimodal)",
              repo: "mlx-community/gemma-4-e2b-it-4bit",
              templateKind: .gemma4),
        .init(id: "gemma4-e2b-it-4bit-text",
              displayName: "Gemma 4 E2B Text (4-bit · text-only)",
              repo: "mlx-community/Gemma4-E2B-IT-Text-int4",
              templateKind: .gemma4),
        .init(id: "qwen3-1-7b-4bit",
              displayName: "Qwen 3 1.7B (4-bit)",
              repo: "mlx-community/Qwen3-1.7B-4bit",
              templateKind: .qwenChatML),
        .init(id: "qwen3-4b-4bit",
              displayName: "Qwen 3 4B (4-bit)",
              repo: "mlx-community/Qwen3-4B-4bit",
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
    }

    static let scenarios: [Scenario] = [
        .init(name: "restaurants_in_sf", turns: [(
            user: "Are there any good restaurants in San Francisco?",
            expect: TurnExpect(
                toolsCalledAny: ["near_named_place"],
                toolsNotCalled: ["near_places", "route_from_places"],
                responseIncludesAny: ["souvla", "zuni", "nopa", "restaurant"],
                responseExcludes: ["i do not have", "i don't have specific"]
            )
        )]),
    ]

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
                    .prefix(80)
                out += "| \(r.variant) | \(r.scenario) | \(r.toolsOk ? "✓" : "✗") | \(r.responseOk ? "✓" : "✗") | \(called) | \(snip) |\n"
            }
            out += "\n### Memory\n\n```\n"
            for m in memory { out += m.scorecardRow() + "\n" }
            out += "```\n"
            return out
        }
    }

    func testAllVariantsAllScenarios() async throws {
        var scorecard = Scorecard()
        for v in Self.variants {
            guard isCached(repo: v.repo) else {
                print("[eval] skipping \(v.displayName): weights not cached")
                continue
            }
            let (rows, memSummary) = try await runVariant(v)
            scorecard.rows.append(contentsOf: rows)
            scorecard.memory.append(memSummary)
        }
        print(scorecard.markdown())

        if scorecard.rows.isEmpty {
            throw XCTSkip("no cached variants to evaluate")
        }

        // Phase 1 gate: at least ONE variant must pass every scenario.
        // We do NOT gate on every variant yet — the Text-int4 variant
        // is expected to fail and we want to see it in the scorecard.
        for scen in Self.scenarios {
            let wins = scorecard.rows.filter {
                $0.scenario == scen.name && $0.toolsOk && $0.responseOk
            }
            XCTAssertFalse(wins.isEmpty,
                "scenario '\(scen.name)' failed on every cached variant — " +
                "no model is carrying this workload correctly.")
        }
    }

    // MARK: -

    private func runVariant(
        _ v: Variant
    ) async throws -> (rows: [Scorecard.Row], memory: MemorySummary) {
        let probe = MemoryProbe(variant: v.displayName)
        await probe.sample("baseline")

        let template: any ModelTemplate = {
            switch v.templateKind {
            case .gemma4:      return Gemma4Template()
            case .qwenChatML:  return QwenChatMLTemplate()
            }
        }()
        let provider = Gemma4Provider(
            id: v.id, displayName: v.displayName, huggingFaceRepo: v.repo,
            template: template
        )
        let fixture = Self.fixtureForPhase1()
        let stub = StubZimService(fixture: fixture)
        let adapter = MCPToolAdapter(
            service: stub, hasStreetzim: true,
            surface: .conversational,
            categoryVocabulary: Self.categoryVocabulary
        )

        let session = ChatSession.forTesting(
            providers: [provider], adapter: adapter, initialModelId: v.id
        )
        session.maxDebugEntries = 20_000
        try await provider.load()
        await probe.sample("post_load")

        var rows: [Scorecard.Row] = []
        for scen in Self.scenarios {
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
            for required in turn.expect.toolsCalledAny {
                if !unique.contains(required) {
                    toolsOk = false
                    errors.append("t\(ti): missing expected tool '\(required)' (called=\(turnTools))")
                }
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
            responseSnippet: String(lastAssistant.prefix(200)),
            errors: errors
        )
    }

    // MARK: - Fixture

    static func fixtureForPhase1() -> StubZimService.Fixture {
        var f = StubZimService.Fixture()

        // Scenario: "restaurants in San Francisco"
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
        f.nearNamedPlace[
            StubZimService.keyNearNamedPlace(
                place: "San Francisco", kinds: ["restaurant"]
            )
        ] = .init(resolved: sfCenter, result: nearResult)
        // Some models drop the kinds filter for a category-less first
        // probe; cover both.
        f.nearNamedPlace[
            StubZimService.keyNearNamedPlace(
                place: "San Francisco", kinds: nil
            )
        ] = .init(resolved: sfCenter, result: nearResult)
        f.nearNamedPlace[
            StubZimService.keyNearNamedPlace(
                place: "San Francisco", kinds: []
            )
        ] = .init(resolved: sfCenter, result: nearResult)

        return f
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
        return FileManager.default.fileExists(
            atPath: "\(snapshots)/\(first)/model.safetensors")
    }
}
