// SPDX-License-Identifier: MIT
//
// Regression tests for the "model loaded, but doesn't emit tool calls"
// failure mode. We hit this once silently — a Gemma 4 E2B variant
// (`mlx-community/Gemma4-E2B-IT-Text-int4`) loaded cleanly on-device but
// answered narratively instead of emitting `<|tool_call>…<tool_call|>`
// blocks. None of our existing tests exercised the model's generation
// output, so the regression only surfaced via human reports from the
// phone. This file is the gap.
//
// For each Gemma provider variant the app ships:
//   1. Skip gracefully if the HF weights aren't cached locally
//      (so CI on a fresh workstation doesn't wait on a 2.5 GB download).
//   2. Load the model.
//   3. Render a canonical "directions to" prompt with the seven
//      place/route/article tools declared via `Gemma4ToolFormat`.
//   4. Run `generate(prompt:)` with a tight token budget.
//   5. Assert the decoded output contains a `<|tool_call>` marker
//      before any `<turn|>` stop. If a variant fails this, the
//      quant didn't preserve Gemma's tool-calling fine-tune and we
//      need to pick a different checkpoint — or at least gate that
//      variant off the model picker.

import XCTest
import MCPZimKit
@testable import MCPZimChatMac

@MainActor
final class GemmaToolEmissionTests: XCTestCase {

    /// (provider id, display name, HF repo id) — kept in sync with
    /// ChatSession's provider list.
    struct Variant {
        let id: String
        let displayName: String
        let repo: String
    }

    static let variants: [Variant] = [
        .init(id: "gemma4-e2b-it-4bit",
              displayName: "Gemma 4 E2B (4-bit · multimodal)",
              repo: "mlx-community/gemma-4-e2b-it-4bit"),
        .init(id: "gemma4-e2b-it-4bit-text",
              displayName: "Gemma 4 E2B Text (4-bit · text-only)",
              repo: "mlx-community/Gemma4-E2B-IT-Text-int4"),
    ]

    func testEachVariantEmitsToolCall() async throws {
        for v in Self.variants {
            try await runVariant(v)
        }
    }

    /// Tighter scenario mirroring the real app prompt: full preamble
    /// (with live location), all 13 production tools declared, and a
    /// user turn asking about restaurants in a *different* named city.
    /// The model should call `near_named_place`, NOT `near_places` with
    /// the user's local coordinates. Catches the "app says 'restaurants
    /// in SF' but searches near Menlo Park" bug — the regression the
    /// minimal 3-tool test misses because the small-prompt context
    /// gives the model no reason to conflate the two tools.
    func testEachVariantPicksNearNamedPlaceForNamedCity() async throws {
        for v in Self.variants {
            try await runNamedPlaceScenario(v)
        }
    }

    // MARK: -

    private func runVariant(_ v: Variant) async throws {
        guard isCached(repo: v.repo) else {
            throw XCTSkip("""
                \(v.displayName) weights not in HF cache \
                (expected ~/.cache/huggingface/hub/models--\
                \(v.repo.replacingOccurrences(of: "/", with: "--"))). \
                Run `huggingface-cli download \(v.repo)` to enable.
                """)
        }
        let provider = Gemma4Provider(
            id: v.id, displayName: v.displayName, huggingFaceRepo: v.repo
        )
        try await provider.load()
        let prompt = canonicalPrompt()
        let stream = provider.generate(
            prompt: prompt,
            parameters: .init(maxTokens: 64, temperature: 0.0, topP: 1.0)
        )
        var output = ""
        for try await chunk in stream { output += chunk }
        // The model should produce a `<|tool_call>` marker before any
        // turn-close marker. Narrative output (reciting a route from
        // memory, refusing, etc.) fails the assertion.
        let toolCallOpen = "<|tool_call>"
        let turnClose = "<turn|>"
        let openIdx = output.range(of: toolCallOpen)?.lowerBound
        let closeIdx = output.range(of: turnClose)?.lowerBound
        XCTAssertNotNil(
            openIdx,
            "\(v.displayName) did NOT emit `<|tool_call>`. " +
            "Raw output (truncated to 400 chars):\n\(output.prefix(400))"
        )
        if let openIdx, let closeIdx {
            XCTAssertLessThan(
                openIdx, closeIdx,
                "\(v.displayName) emitted `<turn|>` before `<|tool_call>` — " +
                "model ended the turn without a tool. Output:\n\(output.prefix(400))"
            )
        }
    }

    private func runNamedPlaceScenario(_ v: Variant) async throws {
        guard isCached(repo: v.repo) else {
            throw XCTSkip("\(v.displayName) weights not cached; skipping named-place scenario")
        }
        let provider = Gemma4Provider(
            id: v.id, displayName: v.displayName, huggingFaceRepo: v.repo
        )
        try await provider.load()
        let prompt = try await fullAppPrompt(
            userMessage: "Are there any good restaurants in San Francisco?"
        )
        let stream = provider.generate(
            prompt: prompt,
            parameters: .init(maxTokens: 96, temperature: 0.0, topP: 1.0)
        )
        var output = ""
        var rawChunks: [String] = []
        for try await chunk in stream {
            output += chunk
            rawChunks.append(chunk)
        }
        // Print prompt + output to the test log so a failure surfaces
        // *what* the model actually produced. An empty `output` means
        // the provider halted on the first stop marker (e.g. `<turn|>`)
        // without yielding any text — the Gemma4Provider stream loop
        // hides pure stop-marker runs.
        print("[ToolEmission] prompt suffix (last 200 chars): \(String(prompt.suffix(200)))")
        print("[ToolEmission] \(v.displayName) → output len=\(output.count) chunks=\(rawChunks.count)")
        print("[ToolEmission] chunks=\(rawChunks.prefix(20))")
        print("[ToolEmission] output='\(output)'")
        // Must emit a tool call and it must target `near_named_place`.
        XCTAssertTrue(
            output.contains("<|tool_call>"),
            "\(v.displayName) emitted no tool call. output.count=\(output.count), " +
            "first chunks=\(rawChunks.prefix(8))"
        )
        XCTAssertTrue(
            output.contains("<|tool_call>call:near_named_place"),
            "\(v.displayName) called the wrong tool — should have used " +
            "`near_named_place` for a named-city query. Output:\n\(output.prefix(400))"
        )
        XCTAssertFalse(
            output.contains("call:near_places"),
            "\(v.displayName) used `near_places` (raw-coord tool) for a " +
            "named-city query — it'd search the user's neighborhood, not " +
            "San Francisco. Output:\n\(output.prefix(400))"
        )
        // Anti-check: user's local coords must not have leaked into the
        // tool arguments for this turn.
        XCTAssertFalse(
            output.contains("37.44") || output.contains("-122.15"),
            "\(v.displayName) leaked the user's lat/lon into the tool " +
            "call for a named-city query. Output:\n\(output.prefix(400))"
        )
    }

    /// Mirror the real app prompt: the same static preamble ChatSession
    /// builds (location line included), the full 13-tool registry, and
    /// a single user turn. We construct it out of the same primitives
    /// the app uses (`ChatSession.composeSystemMessage`,
    /// `MCPToolAdapter.from(service:)`), then convert the registry to
    /// Gemma-4 declarations with the same mapping `ChatSession.
    /// toolDeclarations(registry:)` uses.
    private func fullAppPrompt(userMessage: String) async throws -> String {
        // Location block — matches the "=== Current location ===" block
        // `composeSystemMessage` appends.
        let locationLine = "\n=== Current location ===\n" +
            "The user is physically at lat=37.44118, lon=-122.15537 right now.\n"
        let systemMessage = ChatSession.composeSystemMessage(
            categoryHint: "", locationLine: locationLine
        )

        // Adapter with empty readers: still lists streetzim-gated tools
        // because we pass `hasStreetzim: true` directly. No live queries
        // fire in this test — we only need tool *declarations*.
        let categoryVocabulary = [
            "restaurant", "cafe", "museum", "park", "library",
            "post_office", "atm", "fuel", "hotel", "hospital",
            "pharmacy", "school", "university", "college", "bank",
            "place_of_worship",
        ]
        let service = DefaultZimService(readers: [])
        let adapter = MCPToolAdapter(
            service: service, hasStreetzim: true,
            surface: .conversational,
            categoryVocabulary: categoryVocabulary
        )
        let registry = await adapter.registry
        let tools = toolDeclarations(registry: registry)

        let systemTurn = Gemma4ToolFormat.formatSystemTurn(
            systemMessage: systemMessage, tools: tools
        )
        let userTurn = "<|turn>user\n\(userMessage)<turn|>\n<|turn>model\n"
        return Gemma4PromptTemplate.bos + systemTurn + userTurn
    }

    /// Same mapping as `ChatSession.toolDeclarations(registry:)` — copied
    /// here because that method is private and reading `self.library`.
    /// Test has no live ZIMs so we skip the `zim`-enum injection.
    private func toolDeclarations(
        registry: MCPToolRegistry
    ) -> [Gemma4ToolFormat.ToolDeclaration] {
        registry.tools.map { tool -> Gemma4ToolFormat.ToolDeclaration in
            let schema = (try? JSONSerialization.jsonObject(with: tool.inputSchemaJSON)) as? [String: Any] ?? [:]
            let properties = schema["properties"] as? [String: Any] ?? [:]
            let required = Set((schema["required"] as? [String]) ?? [])
            let params: [Gemma4ToolFormat.ToolDeclaration.Parameter] = properties.keys.sorted().map { key in
                let raw = (properties[key] as? [String: Any]) ?? [:]
                let typeStr = ((raw["type"] as? String) ?? "string").lowercased()
                let type: Gemma4ToolFormat.ToolDeclaration.Parameter.ParamType = {
                    switch typeStr {
                    case "integer": return .integer
                    case "number":  return .number
                    case "boolean": return .boolean
                    case "array":   return .array
                    case "object":  return .object
                    default:        return .string
                    }
                }()
                let description = raw["description"] as? String
                let enumValues = (raw["enum"] as? [Any])?.compactMap { $0 as? String }
                return .init(
                    name: key, type: type,
                    description: description,
                    required: required.contains(key),
                    enumValues: (enumValues?.isEmpty ?? true) ? nil : enumValues
                )
            }
            return .init(name: tool.name, description: tool.description, parameters: params)
        }
    }

    private func hfCacheRoot() -> String {
        // macOS test host containers rewrite NSHomeDirectory to something
        // like `…/Library/Containers/.../Data`, where the HF cache does
        // NOT live. Prefer the explicit override, then the real user home
        // via `HOME`, then `NSUserName()` as the ultimate fallback.
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
        // Require at least one snapshot with a model.safetensors symlink.
        let snapshots = "\(dir)/snapshots"
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: snapshots),
              let first = entries.first
        else { return false }
        return FileManager.default.fileExists(
            atPath: "\(snapshots)/\(first)/model.safetensors")
    }

    private func canonicalPrompt() -> String {
        let tools: [Gemma4ToolFormat.ToolDeclaration] = [
            .init(name: "route_from_places",
                  description: "Plan a driving route between two named places. " +
                    "Origin accepts \"my location\" (host substitutes) or a place name.",
                  parameters: [
                    .init(name: "origin", type: .string, description: "Origin place name or \"my location\"", required: true),
                    .init(name: "destination", type: .string, description: "Destination place name", required: true),
                  ]),
            .init(name: "near_named_place",
                  description: "Find points of interest near a named place. Geocodes internally.",
                  parameters: [
                    .init(name: "place", type: .string, description: "Named place", required: true),
                    .init(name: "radius_km", type: .number, description: "Radius in km"),
                    .init(name: "kinds", type: .array, description: "Category filter like [\"restaurant\"]"),
                  ]),
            .init(name: "get_article",
                  description: "Fetch a Wikipedia article by title.",
                  parameters: [
                    .init(name: "title", type: .string, description: "Article title", required: true),
                  ]),
        ]
        let systemMessage = """
        You are Zimfo, an offline assistant.
        Current location: lat=37.44118, lon=-122.15537.
        When the user asks for directions between named places, call
        route_from_places with origin and destination.
        """
        let systemTurn = Gemma4ToolFormat.formatSystemTurn(
            systemMessage: systemMessage, tools: tools
        )
        let userTurn = "<|turn>user\nGive me directions to San Francisco.<turn|>\n<|turn>model\n"
        return Gemma4PromptTemplate.bos + systemTurn + userTurn
    }
}
