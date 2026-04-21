// SPDX-License-Identifier: MIT
//
// Prompt-layout experiments for the KV cache. Simulates the full
// multi-turn tokenization that ChatSession does on iOS and reports
// LCP between:
//   • primeCache baseline → iter 0 of turn 1
//   • iter 0 of turn 1   → iter 1 of turn 1 (post tool response)
//   • iter 1 of turn 1   → iter 0 of turn 2 (new user query)
//   • iter 0 of turn 2   → iter 1 of turn 2
//
// Running:
//   GEMMA_SMOKE_MODE=prompt-experiment ./GemmaSmoke
//
// The experiment exercises different preamble layouts (A, B, C…) so
// we can iterate on the iOS preamble without rebuilding + installing
// every time.

import Foundation
import Gemma4SwiftCore
import MCPZimKit
import MLX
import MLXLLM
import MLXLMCommon

struct PromptExperiment {
    let container: ModelContainer

    // MARK: - Layout variants

    /// Layout A: current iOS behaviour. categoryHint sits INSIDE the
    /// static system message, so a navigational turn diverges from a
    /// topical-primed cache right where the classification block begins.
    static func layoutA(categoryHint: String, locationLine: String) -> String {
        return """
        You are a helpful assistant with access to tools over locally-loaded \
        ZIM archives. Call tools immediately whenever they can answer the \
        user's question — do NOT ask the user to confirm, and do NOT ask \
        which ZIM to use. Pick sensible defaults for optional arguments. \
        Only respond in prose after you have the tool result.\(locationLine)

        Follow-up interpretation: when the user's current message is SHORT \
        or begins with "and", "what about", "how about", "ok", "then", "also", \
        "more on", "more about", treat it as a follow-up to the immediately \
        previous turn in THIS conversation. Carry the prior subject forward.

        Medical questions are in-scope. For clearly clinical queries search \
        mdwiki. IMPORTANT: do NOT set `kind: "mdwiki"` (or any `kind` filter) \
        unless the question is unambiguously medical.

        For routing questions, keep the reply SHORT. Your reply MUST include: \
        total distance and duration, a single-sentence summary of the major \
        roads, and at most the FIRST 3–4 turns from `turn_by_turn`.

        For "what's nearby" lead with the `by_category` breakdown. Only names \
        from the current `results` array are trustworthy.

        For "tell me about X" questions: `search` → `list_article_sections` → \
        `get_article_section`. Answer from sections read.

        === This turn's classification ===
        \(categoryHint)

        === Grounding policy ===
        Every factual claim in your reply should trace to a tool result. \
        If the user asks a follow-up that refers back to a prior topic, \
        reuse the article(s) from the earlier turn. Cite section / article \
        names inline. If the loaded ZIMs don't cover the question, say so.
        """
    }

    /// Layout B: static preamble has NO classification block. categoryHint
    /// is prepended to the user turn text. Everything above the user turn
    /// is invariant across query complexities, so primeCache + any
    /// user-message turn share the same preamble+tools token prefix.
    static func layoutB_preamble(locationLine: String) -> String {
        return """
        You are a helpful assistant with access to tools over locally-loaded \
        ZIM archives. Call tools immediately whenever they can answer the \
        user's question — do NOT ask the user to confirm, and do NOT ask \
        which ZIM to use. Pick sensible defaults for optional arguments. \
        Only respond in prose after you have the tool result.\(locationLine)

        Follow-up interpretation: when the user's current message is SHORT \
        or begins with "and", "what about", "how about", "ok", "then", "also", \
        "more on", "more about", treat it as a follow-up to the immediately \
        previous turn in THIS conversation. Carry the prior subject forward.

        Medical questions are in-scope. For clearly clinical queries search \
        mdwiki. IMPORTANT: do NOT set `kind: "mdwiki"` (or any `kind` filter) \
        unless the question is unambiguously medical.

        For routing questions, keep the reply SHORT. Your reply MUST include: \
        total distance and duration, a single-sentence summary of the major \
        roads, and at most the FIRST 3–4 turns from `turn_by_turn`.

        For "what's nearby" lead with the `by_category` breakdown. Only names \
        from the current `results` array are trustworthy.

        For "tell me about X" questions: `search` → `list_article_sections` → \
        `get_article_section`. Answer from sections read.

        === Grounding policy ===
        Every factual claim in your reply should trace to a tool result. \
        If the user asks a follow-up that refers back to a prior topic, \
        reuse the article(s) from the earlier turn. Cite section / article \
        names inline. If the loaded ZIMs don't cover the question, say so.
        """
    }

    /// Compose the full user-turn text for layout B: category hint prepended.
    static func layoutB_userTurn(_ userText: String, categoryHint: String) -> String {
        return "[Classification: \(categoryHint.replacingOccurrences(of: "\n", with: " "))]\n\n\(userText)"
    }

    // MARK: - Categories

    static let navigationalHint = """
    The user's current turn looks NAVIGATIONAL (routing / "what's around" / \
    nearest-X). Use streetzim tools. Do NOT call `search` or read Wikipedia \
    articles for this turn.
    """

    static let locationLine = """

    === Current location ===
    The user is physically at lat=37.44121, lon=-122.15530 right now. \
    Treat this as load-bearing context for every directions / nearest / \
    nearby question. For "directions to <place>" always call \
    `route_from_places(origin="my location", destination="<place>")`.
    """

    // MARK: - Minimal tool set used for behavioural tests

    /// Just the tool subset that a "directions to X" / "nearest Y"
    /// query would plausibly call. Narrower than the full iOS
    /// registry but enough to test whether the model picks the right
    /// tool with vs without the classification prefix.
    static func behaviorTools() -> [Gemma4ToolFormat.ToolDeclaration] {
        typealias TD = Gemma4ToolFormat.ToolDeclaration
        typealias P = Gemma4ToolFormat.ToolDeclaration.Parameter
        return [
            TD(name: "route_from_places",
               description: "Plan a driving route. origin and destination are free-text place names. Use origin=\"my location\" to route from the user's current GPS fix.",
               parameters: [
                   P(name: "origin", type: .string, description: "Origin place name", required: true),
                   P(name: "destination", type: .string, description: "Destination place name", required: true),
                   P(name: "zim", type: .string, description: "Streetzim filename", required: false),
               ]),
            TD(name: "near_places",
               description: "Find places of interest near a lat/lon.",
               parameters: [
                   P(name: "lat", type: .number, description: "Latitude", required: true),
                   P(name: "lon", type: .number, description: "Longitude", required: true),
                   P(name: "radius_km", type: .number, description: "Radius km", required: false),
                   P(name: "kinds", type: .array, description: "Filter kinds", required: false),
                   P(name: "limit", type: .integer, required: false),
               ]),
            TD(name: "search",
               description: "Full-text search across loaded Wikipedia ZIMs.",
               parameters: [
                   P(name: "query", type: .string, description: "Search query", required: true),
                   P(name: "limit", type: .integer, required: false),
               ]),
            TD(name: "get_article_section",
               description: "Fetch a section of an article.",
               parameters: [
                   P(name: "zim", type: .string, required: true),
                   P(name: "path", type: .string, required: true),
                   P(name: "section", type: .string, required: true),
               ]),
        ]
    }

    // MARK: - Running

    func run() async throws {
        print("\n== PROMPT LAYOUT EXPERIMENT ==")
        // Minimal synthetic tool decl list — enough to bulk the preamble
        // without needing the full iOS MCPZim adapter. We just want to
        // measure LCP, not quality.
        let tools: [Gemma4ToolFormat.ToolDeclaration] = []

        for (label, runner) in [
            ("A-current", runLayoutA),
            ("B-category-in-user-turn", runLayoutB),
        ] {
            print("\n-- Layout \(label) --")
            try await runner(tools)
        }

        let mode = ProcessInfo.processInfo.environment["EXPERIMENT_WALL_TIME"] ?? "1"
        if mode != "0" {
            print("\n== WALL-TIME COMPARISON (actual generate) ==")
            print("tip: set EXPERIMENT_WALL_TIME=0 to skip this (~2 min)")
            for (label, runner) in [
                ("A-current", wallTimeLayoutA),
                ("B-category-in-user-turn", wallTimeLayoutB),
            ] {
                print("\n-- Wall time: Layout \(label) --")
                try await runner(tools)
            }
        }

        let behaveMode = ProcessInfo.processInfo.environment["EXPERIMENT_BEHAVIOR"] ?? "1"
        if behaveMode != "0" {
            print("\n== BEHAVIOR TEST (does the model still pick the right tool?) ==")
            print("Each query is fed to iter 0 with different preamble shapes.")
            print("We print what the model emits so you can eyeball whether the")
            print("classification hint is earning its keep.")
            try await runBehaviorTest()
        }
    }

    /// Run three queries through three layouts and print the decoded
    /// model output for iter 0. Shows whether removing the
    /// classification prefix changes tool-selection behavior.
    private func runBehaviorTest() async throws {
        let tools = Self.behaviorTools()
        let staticPreamble = Self.layoutB_preamble(locationLine: Self.locationLine)
        struct Scenario { let label: String; let text: String; let hint: String }
        let scenarios: [Scenario] = [
            .init(label: "directions-to-sf", text: "Directions to San Francisco",
                  hint: Self.navigationalHint),
            .init(label: "directions-to-palo-alto", text: "Directions to Palo Alto",
                  hint: Self.navigationalHint),
            .init(label: "whats-nearby", text: "What's nearby?",
                  hint: Self.navigationalHint),
            .init(label: "topical-plasma", text: "tell me about plasma physics",
                  hint: "The user's current turn looks TOPICAL. Chain: search → list_article_sections → get_article_section."),
        ]
        struct Layout { let label: String; let userBuilder: (String, String) -> String }
        let layouts: [Layout] = [
            .init(label: "with-classification",
                  userBuilder: { text, hint in Self.layoutB_userTurn(text, categoryHint: hint) }),
            .init(label: "no-classification",
                  userBuilder: { text, _ in text }),
        ]
        for scn in scenarios {
            print("\n--- scenario: \(scn.label) — \(scn.text.debugDescription) ---")
            for layout in layouts {
                let userText = layout.userBuilder(scn.text, scn.hint)
                let prompt = Gemma4PromptTemplate.render(
                    systemMessage: staticPreamble, tools: tools,
                    turns: [ChatTurn(role: .user, text: userText)]
                )
                let tokens = (await container.encode(prompt)).map { Int32($0) }
                let params = GenerateParameters(
                    maxTokens: 60, temperature: 0.3, topP: 0.9, prefillStepSize: 128)
                let output = try await container.perform { context in
                    let kv = context.model.newCache(parameters: params)
                    let input = LMInput(tokens: MLXArray(tokens))
                    let stream = try MLXLMCommon.generateTokens(
                        input: input, cache: kv, parameters: params, context: context)
                    var ids: [Int32] = []
                    var decoded = ""
                    for await event in stream {
                        guard case .token(let id) = event else { continue }
                        ids.append(Int32(id))
                        decoded = context.tokenizer.decode(tokens: ids.map { Int($0) })
                        if decoded.contains("<tool_call|>") || decoded.contains("<turn|>") {
                            break
                        }
                    }
                    return decoded
                }
                print("  [\(layout.label)] emitted:")
                // Indent and limit to ~300 chars so the output stays readable.
                let clipped = output.count > 300
                    ? String(output.prefix(300)) + "…"
                    : output
                for line in clipped.split(separator: "\n", omittingEmptySubsequences: false) {
                    print("    | \(line)")
                }
            }
        }
    }

    private func runLayoutA(tools: [Gemma4ToolFormat.ToolDeclaration]) async throws {
        let primePreamble = Self.layoutA(
            categoryHint: Self.navigationalHint, // ← intentional: prime with nav hint
            locationLine: Self.locationLine
        )
        // But iOS primes with .topical. Simulate that drift explicitly.
        let primedTopicalPreamble = Self.layoutA(
            categoryHint: """
            The user's current turn looks TOPICAL ("tell me about X" / \
            "what is X"). Fixed chain: search → list_article_sections → \
            get_article_section.
            """,
            locationLine: Self.locationLine
        )
        _ = primePreamble
        let navPreamble = Self.layoutA(
            categoryHint: Self.navigationalHint,
            locationLine: Self.locationLine
        )

        let primed = Gemma4PromptTemplate.bos
            + Gemma4ToolFormat.formatSystemTurn(systemMessage: primedTopicalPreamble, tools: tools)
        let turn1Iter0 = Gemma4PromptTemplate.render(
            systemMessage: navPreamble, tools: tools,
            turns: [ChatTurn(role: .user, text: "Directions to San Francisco")]
        )
        let turn1Iter1 = Gemma4PromptTemplate.render(
            systemMessage: navPreamble, tools: tools,
            turns: [
                ChatTurn(role: .user, text: "Directions to San Francisco"),
                ChatTurn(role: .assistant, text: Self.assistantEmission1),
                ChatTurn(role: .tool, text: Self.toolResponse1),
            ]
        )
        let turn2Iter0 = Gemma4PromptTemplate.render(
            systemMessage: navPreamble, tools: tools,
            turns: [
                ChatTurn(role: .user, text: "Directions to San Francisco"),
                ChatTurn(role: .assistant, text: Self.assistantEmission1),
                ChatTurn(role: .tool, text: Self.toolResponse1),
                ChatTurn(role: .assistant, text: Self.finalReply1),
                ChatTurn(role: .user, text: "How about to San Jose"),
            ]
        )
        try await report(primed: primed, turn1Iter0: turn1Iter0,
                         turn1Iter1: turn1Iter1, turn2Iter0: turn2Iter0)
    }

    private func runLayoutB(tools: [Gemma4ToolFormat.ToolDeclaration]) async throws {
        // Layout B's preamble is invariant across complexities.
        let staticPreamble = Self.layoutB_preamble(locationLine: Self.locationLine)
        let primed = Gemma4PromptTemplate.bos
            + Gemma4ToolFormat.formatSystemTurn(systemMessage: staticPreamble, tools: tools)
        let userTurn1 = Self.layoutB_userTurn(
            "Directions to San Francisco", categoryHint: Self.navigationalHint
        )
        let turn1Iter0 = Gemma4PromptTemplate.render(
            systemMessage: staticPreamble, tools: tools,
            turns: [ChatTurn(role: .user, text: userTurn1)]
        )
        let turn1Iter1 = Gemma4PromptTemplate.render(
            systemMessage: staticPreamble, tools: tools,
            turns: [
                ChatTurn(role: .user, text: userTurn1),
                ChatTurn(role: .assistant, text: Self.assistantEmission1),
                ChatTurn(role: .tool, text: Self.toolResponse1),
            ]
        )
        let userTurn2 = Self.layoutB_userTurn(
            "How about to San Jose", categoryHint: Self.navigationalHint
        )
        let turn2Iter0 = Gemma4PromptTemplate.render(
            systemMessage: staticPreamble, tools: tools,
            turns: [
                ChatTurn(role: .user, text: userTurn1),
                ChatTurn(role: .assistant, text: Self.assistantEmission1),
                ChatTurn(role: .tool, text: Self.toolResponse1),
                ChatTurn(role: .assistant, text: Self.finalReply1),
                ChatTurn(role: .user, text: userTurn2),
            ]
        )
        try await report(primed: primed, turn1Iter0: turn1Iter0,
                         turn1Iter1: turn1Iter1, turn2Iter0: turn2Iter0)
    }

    private func report(primed: String, turn1Iter0: String,
                        turn1Iter1: String, turn2Iter0: String) async throws {
        let t0 = (await container.encode(primed)).map { Int32($0) }
        let t1 = (await container.encode(turn1Iter0)).map { Int32($0) }
        let t2 = (await container.encode(turn1Iter1)).map { Int32($0) }
        let t3 = (await container.encode(turn2Iter0)).map { Int32($0) }
        print(String(format: "  primed tokens   : %5d", t0.count))
        print(String(format: "  turn1 iter 0    : %5d (LCP vs primed = %5d, hit=%@)", t1.count,
                     lcp(t0, t1),
                     lcp(t0, t1) == t0.count ? "YES" : "no"))
        // simulated cache state after iter 0's generate (prompt + assistant emission)
        let simAsstTokens = (await container.encode(Self.assistantEmission1)).map { Int32($0) }
        let cachedAfterIter0 = t1 + simAsstTokens
        print(String(format: "  cached after I0 : %5d", cachedAfterIter0.count))
        print(String(format: "  turn1 iter 1    : %5d (LCP vs cached = %5d, hit=%@)", t2.count,
                     lcp(cachedAfterIter0, t2),
                     lcp(cachedAfterIter0, t2) == cachedAfterIter0.count ? "YES" : "no"))
        // after iter 1: cache has + final reply
        let simFinalTokens = (await container.encode(Self.finalReply1)).map { Int32($0) }
        let cachedAfterIter1 = t2 + simFinalTokens
        print(String(format: "  cached after I1 : %5d", cachedAfterIter1.count))
        print(String(format: "  turn2 iter 0    : %5d (LCP vs cached = %5d, hit=%@)", t3.count,
                     lcp(cachedAfterIter1, t3),
                     lcp(cachedAfterIter1, t3) == cachedAfterIter1.count ? "YES" : "no"))
    }

    // MARK: - Static sample strings

    static let assistantEmission1 = "<|tool_call>call:route_from_places{origin:\"my location\",destination:\"San Francisco\"}<tool_call|>"
    static let toolResponse1 = "{\"distance_km\":53,\"duration_min\":65,\"turn_by_turn\":[\"Head northeast on Addison Ave\",\"Turn left onto Bryant St\",\"Turn right onto Embarcadero Rd\",\"Merge onto US-101 N via the ramp toward San Francisco\",\"Take exit 433B for 7th St\",\"Turn right onto 7th St\",\"Turn right onto Mission St\"],\"turn_by_turn_total\":27}"
    static let finalReply1 = "The driving route to San Francisco is approximately 33.1 miles and should take about 37 minutes. The route primarily involves Addison Avenue, Bryant Street, Channing Avenue, Middlefield Road, and Willow Road to get you there."

    private func lcp(_ a: [Int32], _ b: [Int32]) -> Int {
        var i = 0; let n = min(a.count, b.count)
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    // MARK: - Wall-time comparison (real generate with cache reuse)

    /// Runs the full "directions to SF → tool response → final reply →
    /// directions to SJ → tool response → final reply" flow and reports
    /// wall-time for each step, with the cache-reuse logic mirrored
    /// from `Gemma4Provider.generate`.
    private func wallTimeLayoutA(tools: [Gemma4ToolFormat.ToolDeclaration]) async throws {
        // Layout A primes with .topical but uses .navigational on the
        // real turn — the miss at iter 0 is the whole point.
        let primedPreamble = Self.layoutA(
            categoryHint: """
            The user's current turn looks TOPICAL. Chain: search → \
            list_article_sections → get_article_section.
            """,
            locationLine: Self.locationLine
        )
        let runPreamble = Self.layoutA(
            categoryHint: Self.navigationalHint, locationLine: Self.locationLine
        )
        let primePrompt = Gemma4PromptTemplate.bos
            + Gemma4ToolFormat.formatSystemTurn(systemMessage: primedPreamble, tools: tools)
        let user1 = "Directions to San Francisco"
        let user2 = "How about to San Jose"
        try await simulateRun(
            primePrompt: primePrompt,
            renderTurn1Iter0: { Gemma4PromptTemplate.render(
                systemMessage: runPreamble, tools: tools,
                turns: [ChatTurn(role: .user, text: user1)]) },
            renderTurn1Iter1: { Gemma4PromptTemplate.render(
                systemMessage: runPreamble, tools: tools,
                turns: [
                    ChatTurn(role: .user, text: user1),
                    ChatTurn(role: .assistant, text: Self.assistantEmission1),
                    ChatTurn(role: .tool, text: Self.toolResponse1),
                ]) },
            renderTurn2Iter0: { Gemma4PromptTemplate.render(
                systemMessage: runPreamble, tools: tools,
                turns: [
                    ChatTurn(role: .user, text: user1),
                    ChatTurn(role: .assistant, text: Self.assistantEmission1),
                    ChatTurn(role: .tool, text: Self.toolResponse1),
                    ChatTurn(role: .assistant, text: Self.finalReply1),
                    ChatTurn(role: .user, text: user2),
                ]) }
        )
    }

    private func wallTimeLayoutB(tools: [Gemma4ToolFormat.ToolDeclaration]) async throws {
        // Layout B: invariant preamble, classification prepended in
        // user turn. Prime matches every real turn.
        let staticPreamble = Self.layoutB_preamble(locationLine: Self.locationLine)
        let primePrompt = Gemma4PromptTemplate.bos
            + Gemma4ToolFormat.formatSystemTurn(systemMessage: staticPreamble, tools: tools)
        let user1 = Self.layoutB_userTurn(
            "Directions to San Francisco", categoryHint: Self.navigationalHint)
        let user2 = Self.layoutB_userTurn(
            "How about to San Jose", categoryHint: Self.navigationalHint)
        try await simulateRun(
            primePrompt: primePrompt,
            renderTurn1Iter0: { Gemma4PromptTemplate.render(
                systemMessage: staticPreamble, tools: tools,
                turns: [ChatTurn(role: .user, text: user1)]) },
            renderTurn1Iter1: { Gemma4PromptTemplate.render(
                systemMessage: staticPreamble, tools: tools,
                turns: [
                    ChatTurn(role: .user, text: user1),
                    ChatTurn(role: .assistant, text: Self.assistantEmission1),
                    ChatTurn(role: .tool, text: Self.toolResponse1),
                ]) },
            renderTurn2Iter0: { Gemma4PromptTemplate.render(
                systemMessage: staticPreamble, tools: tools,
                turns: [
                    ChatTurn(role: .user, text: user1),
                    ChatTurn(role: .assistant, text: Self.assistantEmission1),
                    ChatTurn(role: .tool, text: Self.toolResponse1),
                    ChatTurn(role: .assistant, text: Self.finalReply1),
                    ChatTurn(role: .user, text: user2),
                ]) }
        )
    }

    /// Single-perform variant: runs prime + all 3 generates inside one
    /// `container.perform` so the `[KVCache]` doesn't have to cross
    /// actor boundaries. MLX on macOS appears to release the KV state
    /// between perform calls (SIGSEGV when we try to reuse it), which
    /// is why the earlier multi-perform version crashed.
    private func simulateRunInSinglePerform(
        primeTokens: [Int32],
        t1i0: [Int32],
        t1i1: [Int32],
        t2i0: [Int32]
    ) async throws -> (dPrime: Double, r1: (wall: Double, hit: Bool, common: Int, generated: Int),
                       r2: (wall: Double, hit: Bool, common: Int, generated: Int),
                       r3: (wall: Double, hit: Bool, common: Int, generated: Int)) {
        let stopMarkers = ["<turn|>", "<|turn>"]
        let toolClose = "<tool_call|>"
        return try await container.perform { context in
            @Sendable
            func runOne(
                prompt: [Int32], cached: [Int32], existing: [KVCache]?,
                maxTokens: Int, detectToolCall: Bool
            ) async throws -> (wall: Double, hit: Bool, common: Int,
                         newCache: [KVCache], newCached: [Int32],
                         generatedIDs: [Int32]) {
                var lcpN = 0
                let n = min(cached.count, prompt.count)
                while lcpN < n && cached[lcpN] == prompt[lcpN] { lcpN += 1 }
                let params = GenerateParameters(
                    maxTokens: maxTokens, temperature: 0.3, topP: 0.9,
                    prefillStepSize: 128)
                let kv: [KVCache]
                let inputs: [Int32]
                let hit: Bool
                if lcpN == cached.count, lcpN > 0, let existing, !existing.isEmpty {
                    kv = existing
                    inputs = Array(prompt[lcpN...])
                    hit = true
                } else {
                    kv = context.model.newCache(parameters: params)
                    inputs = prompt
                    hit = false
                }
                let t0 = Date()
                let input = LMInput(tokens: MLXArray(inputs))
                let stream = try MLXLMCommon.generateTokens(
                    input: input, cache: kv, parameters: params, context: context)
                var tokenIDs: [Int32] = []
                var decodedSoFar = ""
                var cutoff: Int?
                streamLoop: for await event in stream {
                    guard case .token(let id) = event else { continue }
                    tokenIDs.append(Int32(id))
                    decodedSoFar = context.tokenizer.decode(tokens: tokenIDs.map { Int($0) })
                    if detectToolCall, cutoff == nil, decodedSoFar.contains(toolClose) {
                        cutoff = tokenIDs.count
                        break streamLoop
                    }
                    for m in stopMarkers where decodedSoFar.contains(m) {
                        cutoff = tokenIDs.count
                        break streamLoop
                    }
                }
                let keep = cutoff ?? tokenIDs.count
                let committed = Array(tokenIDs.prefix(keep))
                let dt = Date().timeIntervalSince(t0)
                return (dt, hit, lcpN, kv, prompt + committed, committed)
            }

            // 1) prime cache (no sampling)
            print("  … priming cache with \(primeTokens.count) tokens")
            fflush(stdout)
            let tPrime = Date()
            let primeParams = GenerateParameters(maxTokens: 0, prefillStepSize: 128)
            let primedCache = context.model.newCache(parameters: primeParams)
            let primeInput = LMInput(tokens: MLXArray(primeTokens))
            _ = try TokenIterator(input: primeInput, model: context.model,
                                   cache: primedCache, parameters: primeParams)
            let dPrime = Date().timeIntervalSince(tPrime)
            print(String(format: "  … prime done in %.2fs", dPrime))
            fflush(stdout)

            // 2) turn 1 iter 0 — generate until tool-call close
            print("  … running turn 1 iter 0 (prompt=\(t1i0.count))")
            fflush(stdout)
            let r1 = try await runOne(
                prompt: t1i0, cached: primeTokens, existing: primedCache,
                maxTokens: 60, detectToolCall: true)
            print(String(format: "  … turn 1 iter 0 done in %.2fs hit=%@ generated=%d",
                         r1.wall, r1.hit ? "YES" : "no", r1.generatedIDs.count))
            fflush(stdout)
            // 3) turn 1 iter 1 — continue to final reply
            print("  … running turn 1 iter 1 (prompt=\(t1i1.count))"); fflush(stdout)
            let r2 = try await runOne(
                prompt: t1i1, cached: r1.newCached, existing: r1.newCache,
                maxTokens: 120, detectToolCall: false)
            print(String(format: "  … turn 1 iter 1 done in %.2fs hit=%@ generated=%d",
                         r2.wall, r2.hit ? "YES" : "no", r2.generatedIDs.count))
            fflush(stdout)
            // 4) turn 2 iter 0 — new user query
            print("  … running turn 2 iter 0 (prompt=\(t2i0.count))"); fflush(stdout)
            let r3 = try await runOne(
                prompt: t2i0, cached: r2.newCached, existing: r2.newCache,
                maxTokens: 60, detectToolCall: true)
            print(String(format: "  … turn 2 iter 0 done in %.2fs hit=%@ generated=%d",
                         r3.wall, r3.hit ? "YES" : "no", r3.generatedIDs.count))
            fflush(stdout)

            return (dPrime,
                    (r1.wall, r1.hit, r1.common, r1.generatedIDs.count),
                    (r2.wall, r2.hit, r2.common, r2.generatedIDs.count),
                    (r3.wall, r3.hit, r3.common, r3.generatedIDs.count))
        }
    }

    /// (UNUSED) Multi-perform variant — segfaults on macOS when we try
    /// to reuse a `[KVCache]` across separate `container.perform`
    /// calls. Kept here in case the MLX behaviour changes.
    private final class CacheSim: @unchecked Sendable {
        var cached: [Int32] = []
        var cache: [KVCache]?
        func reset() { cached = []; cache = nil }
        /// Run a prefill-only pass (no sampling). Leaves cache with
        /// exactly `tokens` committed.
        func primeCache(tokens: [Int32], container: ModelContainer) async throws {
            let kv: [KVCache] = try await container.perform { context in
                let params = GenerateParameters(maxTokens: 0, prefillStepSize: 128)
                let kv = context.model.newCache(parameters: params)
                let input = LMInput(tokens: MLXArray(tokens))
                _ = try TokenIterator(input: input, model: context.model,
                                       cache: kv, parameters: params)
                return kv
            }
            self.cache = kv
            self.cached = tokens
        }
        /// Mirror Gemma4Provider.generate's cache-hit decision + run
        /// generateTokens until a stop marker appears or maxTokens
        /// exhausted. Returns (wallSeconds, tokensGenerated, hit, common, newPromptCount).
        func runGenerate(
            prompt: [Int32], maxTokens: Int,
            stopMarkers: [String], toolCallClose: String?,
            container: ModelContainer
        ) async throws -> (wall: Double, generated: Int, hit: Bool, common: Int, prompt: Int) {
            let t0 = Date()
            // LCP decision outside of container.perform so we don't
            // touch actor-isolated state inside the sendable closure.
            var lcpN = 0
            let n = min(self.cached.count, prompt.count)
            while lcpN < n && self.cached[lcpN] == prompt[lcpN] { lcpN += 1 }
            let existingForRun = self.cache
            let cachedCount = self.cached.count
            let isHit = (lcpN == cachedCount && lcpN > 0 && existingForRun != nil && !(existingForRun!.isEmpty))
            let promptCopy = prompt
            let maxT = maxTokens
            let (stream, tokenizer, returnedCache): (AsyncStream<TokenGeneration>, any Tokenizer, [KVCache]) =
                try await container.perform { context in
                    let params = GenerateParameters(
                        maxTokens: maxT, temperature: 0.3, topP: 0.9,
                        prefillStepSize: 128)
                    let kvCache: [KVCache]
                    let inputs: [Int32]
                    if isHit, let existing = existingForRun {
                        kvCache = existing
                        inputs = Array(promptCopy[lcpN...])
                    } else {
                        kvCache = context.model.newCache(parameters: params)
                        inputs = promptCopy
                    }
                    let input = LMInput(tokens: MLXArray(inputs))
                    let s = try MLXLMCommon.generateTokens(
                        input: input, cache: kvCache, parameters: params, context: context)
                    return (s, context.tokenizer, kvCache)
                }
            self.cache = returnedCache
            self.cached = prompt
            var tokenIDs: [Int32] = []
            var decodedSoFar = ""
            var tokensAtCutoff: Int?
            streamLoop: for await event in stream {
                guard case .token(let id) = event else { continue }
                tokenIDs.append(Int32(id))
                let full = tokenizer.decode(tokens: tokenIDs.map { Int($0) })
                decodedSoFar = full
                if let close = toolCallClose, tokensAtCutoff == nil,
                   decodedSoFar.contains(close) {
                    tokensAtCutoff = tokenIDs.count
                    break streamLoop
                }
                for m in stopMarkers where decodedSoFar.contains(m) {
                    tokensAtCutoff = tokenIDs.count
                    break streamLoop
                }
            }
            let keep = tokensAtCutoff ?? tokenIDs.count
            let committed = Array(tokenIDs.prefix(keep))
            self.cached.append(contentsOf: committed)
            let dt = Date().timeIntervalSince(t0)
            return (dt, committed.count, isHit, lcpN, prompt.count)
        }
    }

    private func simulateRun(
        primePrompt: String,
        renderTurn1Iter0: () -> String,
        renderTurn1Iter1: () -> String,
        renderTurn2Iter0: () -> String
    ) async throws {
        let primeTokens = (await container.encode(primePrompt)).map { Int32($0) }
        let t1i0 = (await container.encode(renderTurn1Iter0())).map { Int32($0) }
        let t1i1 = (await container.encode(renderTurn1Iter1())).map { Int32($0) }
        let t2i0 = (await container.encode(renderTurn2Iter0())).map { Int32($0) }
        let result = try await simulateRunInSinglePerform(
            primeTokens: primeTokens, t1i0: t1i0, t1i1: t1i1, t2i0: t2i0
        )
        print(String(format: "  primeCache (%d tokens)       : %.2fs",
                     primeTokens.count, result.dPrime))
        print(String(format: "  turn1 iter 0 (prompt=%d): %.2fs  hit=%@ common=%d  generated=%d",
                     t1i0.count, result.r1.wall, result.r1.hit ? "YES" : "no ",
                     result.r1.common, result.r1.generated))
        print(String(format: "  turn1 iter 1 (prompt=%d): %.2fs  hit=%@ common=%d  generated=%d",
                     t1i1.count, result.r2.wall, result.r2.hit ? "YES" : "no ",
                     result.r2.common, result.r2.generated))
        print(String(format: "  turn2 iter 0 (prompt=%d): %.2fs  hit=%@ common=%d  generated=%d",
                     t2i0.count, result.r3.wall, result.r3.hit ? "YES" : "no ",
                     result.r3.common, result.r3.generated))
        let total = result.r1.wall + result.r2.wall + result.r3.wall
        print(String(format: "  ── model wall total (3 generates): %.2fs  (primeCache %.2fs extra)",
                     total, result.dPrime))
    }
}

import Tokenizers
