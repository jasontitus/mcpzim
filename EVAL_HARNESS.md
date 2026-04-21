# Model evaluation harness design

## Goal

For any candidate model variant (Gemma 4 E2B / E4B at 2-bit / 4-bit / 8-bit, Qwen 3.5 2B / 4B, future MoE / UD quants…), answer two questions in a single test run:

1. **Quality**: does it drive the real `ChatSession.send()` tool-dispatch pipeline correctly across a canonical scenario library — right tool, right args, right follow-up, right content in the final reply?
2. **Memory**: what's the baseline RSS, load-time peak, prefill peak, decode peak, and steady-state after each turn? Does it fit on 6 GB / 8 GB / 12 GB iPhones under a live Kokoro TTS session?

Output is a **scorecard**: one row per `(variant, scenario)` with tool-call sequence, response snippet, pass/fail per assertion bucket, plus a per-variant memory profile. Runs in Mac test target, no device round-trip.

## Non-goals

- Exercising every tool. The scenario library picks representative calls (a couple of routing, a couple of POI, a couple of article lookups) — not exhaustive.
- Stable numeric comparisons. Token-level output varies run-to-run; we use loose substring / tool-sequence assertions, same as `ConversationalEvalTests`.
- Network / GPU benchmarks. Memory is what matters on iPhone; throughput we measure informally via wall-time.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ MultiModelEvalTests.swift                                      │
│                                                                │
│  for variant in ModelVariantRegistry.all {                     │
│    let memProbe = MemoryProbe(tag: variant.displayName)        │
│    let session  = ChatSession.forTesting(                      │
│                     adapter: StubZimService.adapter,           │
│                     providerFactory: variant.provider)         │
│    for scenario in ScenarioLibrary.all {                       │
│      for turn in scenario.turns {                              │
│        session.send(turn.user)                                 │
│        await session.waitForGenerationIdle()                   │
│        let msg = session.lastAssistantMessage()                │
│        memProbe.sample("turn.\(i)")                            │
│        record(variant, scenario, turn, msg.toolCalls, msg.text)│
│      }                                                         │
│      session.resetConversation()                               │
│    }                                                           │
│    report.add(variant, memProbe.summary())                     │
│  }                                                             │
│  print(report.markdownScorecard())                             │
│  XCTAssertNoRegression(report)                                 │
└────────────────────────────────────────────────────────────────┘
```

## Pieces to build

### 1. `StubZimService`

Location: `swift/Sources/MCPZimKit/StubZimService.swift`.

Deterministic `ZimService` that returns canned structured payloads from an in-memory table keyed by `(method, canonicalised-args)`. Every scenario's tool calls have a matching fixture; calls that miss the table throw `.noFixture(method, args)` so we know when a model drifts off the scenario.

```swift
public actor StubZimService: ZimService {
    public struct Fixture {
        var inventory: InventoryResult?
        var nearNamedPlace: [NamedPlaceKey: (resolved: Place, result: NearPlacesResult)]
        var routeFromPlaces: [RouteKey: (resolved: (Place, Place), route: Route)]
        var articleByTitle: [TitleKey: (zim: String, path: String, title: String, section: ArticleSection)]
        var search: [SearchKey: [SearchHitResult]]
        // ...
    }
    public init(fixture: Fixture) { ... }
    // Protocol impls dispatch to fixture, throw .noFixture on miss
}
```

v1 fixtures cover: `near_named_place(San Francisco | Adams Morgan | Palo Alto, kinds=[restaurant|cafe|museum])`, `route_from_places(my location → San Francisco | Adams Morgan)`, `search(aspirin | Lithuania)`, `get_article(Aspirin | History of Lithuania)`. Roughly 15 fixture entries total — enough for the scenario library.

### 2. `ChatSession.forTesting(adapter:providerFactory:)`

New internal convenience initializer that bypasses the usual `addReaders(urls:)` / `adapter = await MCPToolAdapter.from(service:)` boot path. Takes a pre-built adapter + a provider factory so the test controls both the service surface and the model. Default behaviour unchanged.

### 3. Model family abstraction — `ModelTemplate`

This is the Qwen-enabling piece. Currently `Gemma4Provider` owns the format end-to-end. Split out:

```swift
public protocol ModelTemplate: Sendable {
    /// Turn open / close markers ("`<|turn>`" vs "`<|im_start|>`").
    var userTurnOpen: String { get }   var userTurnClose: String { get }
    var assistantTurnOpen: String { get }  var assistantTurnClose: String { get }
    var bos: String { get }
    var stopMarkers: [String] { get }

    /// Render the whole transcript (system + history + new user turn).
    func renderTranscript(systemPreamble: String, turns: [ChatTurn]) -> String

    /// Tool declarations in the model's native format.
    func formatToolDeclarations(_ tools: [MCPTool]) -> String

    /// Extract any tool calls from a streamed chunk of output.
    /// Called after every chunk; returns a completed call when
    /// the tool-call close marker is seen.
    func extractToolCalls(from stream: inout String) -> [ParsedToolCall]

    /// Format a tool response turn to feed back to the model.
    func formatToolResponse(name: String, payload: [String: Any]) -> String
}
```

Implementations:

| Template | Used by | Key differences |
|---|---|---|
| `Gemma4Template` | Gemma 4 family | Turn markers `<\|turn\|>`, tool-call `<\|tool_call>call:X{…}<tool_call\|>` custom format, EOS token 50 |
| `QwenChatMLTemplate` | Qwen 2 / 2.5 / 3.5 family | Turn markers `<\|im_start\|>` / `<\|im_end\|>`, tool-call `<tool_call>{"name":…,"arguments":…}</tool_call>` JSON format, EOS `<\|im_end\|>` / `<\|endoftext\|>` |
| `Llama3Template` | Llama 3 / 3.1 | `<\|start_header_id\|>user<\|end_header_id\|>\n\n` style, tool-call via `<\|python_tag\|>{…}<\|eom_id\|>` |

`ModelProvider` gains a `var template: any ModelTemplate { get }` requirement. `Gemma4Provider` returns `Gemma4Template()`; a new `QwenProvider` returns `QwenChatMLTemplate()`. `ChatSession`'s tool loop calls `selectedModel.template.extractToolCalls(from:)` instead of hard-coding Gemma's `<|tool_call>` pattern.

Migration cost: moderate. Everywhere we currently reference `Gemma4PromptTemplate.render`, `Gemma4ToolFormat.formatSystemTurn`, or `Gemma4ToolCallParser` becomes a `selectedModel.template.*` call. Test before / after to make sure Gemma behaviour is identical.

### 4. `QwenProvider`

Swift `ModelProvider` backed by `MLXLLM.QwenXModel` (upstream mlx-swift-lm ships Qwen 2 / 2.5 / 3 variants). Same load path as `Gemma4Provider` (via `LLMModelFactory.shared.loadContainer(from:#hubDownloader():…)`), just different `modelConfiguration.id` and different `template`.

v1 target repos (text-only, MLX-hosted):
- `mlx-community/Qwen2.5-3B-Instruct-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit` (memory-budget pressure test)
- `mlx-community/Qwen3-4B-Instruct-4bit` (if available)

### 5. `MemoryProbe`

Thin wrapper around `task_vm_info` (same Mach call `MemoryStats.formatted()` uses today), exposing:

```swift
public struct MemorySample { let tag: String; let rssMB: Double; let peakMB: Double; let timestamp: Date }

public actor MemoryProbe {
    public func sample(_ tag: String) -> MemorySample
    public func startContinuous(intervalMs: Int = 100)
    public func stop() -> [MemorySample]  // all samples captured since last start
    public func summary() -> MemorySummary  // baseline / peak / post-load / per-turn
}
```

Harness takes a baseline before `selectedModel.load()`, another right after, samples continuously during generate(), another after each turn. Report picks out:
- `baseline_mb`: pre-load RSS
- `post_load_mb`: RSS after weights resident
- `prefill_peak_mb`: max RSS during prefill window
- `decode_peak_mb`: max RSS during decode window
- `post_turn_mb`: RSS 500 ms after generate finishes
- `lifetime_peak_mb`: max RSS seen across all turns

### 6. Scenario library

`swift/Sources/MCPZimKit/EvalScenarios.swift` (or inside the test target — let's keep it in MCPZimKit so `gemma-smoke` can reuse).

```swift
public struct EvalScenario { let name: String; let turns: [EvalTurn] }
public struct EvalTurn {
    let user: String
    let expected: Expected
}
public struct Expected {
    var toolsCalledInOrder: [String]       // exact sequence
    var toolsCalledAny: [String]           // any-of fallback
    var toolsNotCalled: [String]
    var responseIncludesAny: [String]
    var responseExcludes: [String]
    var minimumToolCallCounts: [String: Int]
}
```

v1 scenarios (scoped to tools the stub fixture covers):

1. `restaurants_in_sf` — "are there any good restaurants in San Francisco?" → calls `near_named_place(place:"San Francisco", kinds:["restaurant"])`, must NOT call `near_places`, response mentions ≥2 restaurant names.
2. `directions_to_sf` — "directions to San Francisco" → calls `route_from_places`, response mentions time + distance.
3. `nearest_post_office` — "where's the nearest post office?" → calls `near_places(lat, lon, kinds:["post_office"])`, response mentions an address or name.
4. `tell_me_about_aspirin` — "tell me about aspirin" → `search(aspirin)` + `get_article` (or `article_overview` once it lands), response mentions "pain" or "inflammation".
5. `overview_of_lithuania` — "give me an overview of the history of Lithuania" → `search` + `get_article_section` ≥2 calls, response mentions "grand duchy" / "soviet" / "independence".
6. `compare_iran_us` — "how have the US and Iran gotten along?" → one of `article_relationship`, `compare_articles`, or sequential `get_article`s; response mentions specific events.
7. `how_much_longer` (multi-turn) — turn 1 "directions to Palo Alto", turn 2 "how much longer?" → turn 2 calls `route_status` (new tool) and returns remaining duration.

### 7. Scorecard output

Markdown table printed via `XCTAssert`-wrapping `print()`:

```
| variant                   | scenario                  | tools(✓/total) | response | baseline | peak    | post-turn |
|---------------------------|---------------------------|----------------|----------|----------|---------|-----------|
| gemma-4-e2b-it-4bit       | restaurants_in_sf         | 1/1 ✓          | ✓        | 3.3 GB   | 4.7 GB  | 4.1 GB    |
| gemma-4-e2b-it-4bit-text  | restaurants_in_sf         | 0/1 ✗          | ✗        | 3.1 GB   | 4.2 GB  | 3.8 GB    |
| qwen2.5-3b-instruct-4bit  | restaurants_in_sf         | 1/1 ✓          | ✓        | 2.0 GB   | 3.1 GB  | 2.6 GB    |
| …                                                                                                          |
```

One row per `(variant, scenario)`. `tools(✓/total)` = scenario tool-sequence assertions passed / attempted. `response` = response-substring assertions passed. Memory columns per variant (not per scenario — captured once per variant's run, max across all scenarios).

Plus a variant-summary section at the bottom ranking by aggregate score + listing memory envelopes. This is the artefact we actually use to pick a model.

## Phases of work

Enumerated so we can commit as each phase lands.

**Phase 1 — StubZimService + harness scaffolding.**
- Write `StubZimService` with fixtures for scenarios 1–3.
- Add `ChatSession.forTesting(adapter:providerFactory:)`.
- Write `MemoryProbe` (wraps existing `MemoryStats`).
- Write `MultiModelEvalTests.swift` with 2 Gemma variants × 3 scenarios.
- Verify end-to-end: Gemma multimodal passes, Gemma Text-int4 fails the known scenarios, memory numbers land.

**Phase 2 — model-family abstraction.**
- Extract `ModelTemplate` protocol; port `Gemma4Provider` to use it without behavior change.
- Confirm existing tests still pass.
- Commit.

**Phase 3 — Qwen.**
- Implement `QwenChatMLTemplate` + `QwenProvider` + `QwenToolCallParser`.
- Add Qwen variants to the eval registry.
- Run harness; add Qwen rows to scorecard.

**Phase 4 — scenario coverage + new composite tools.**
- Flesh out scenarios 4–7 (Wikipedia / multi-turn).
- Wire `route_status`, `nearby_stories`, `article_overview`, `compare_articles` from `TOOL_DESIGN.md`.
- Each new tool gets a fixture in `StubZimService` + a scenario that exercises it.

**Phase 5 — memory-scorecard hardening.**
- Add XCTAssert gates: fail if peak > device-tier budget × 1.05 (regression guard).
- Add run-to-run delta tracking (persist scorecard to disk; new run compares to baseline).

## What this doc changes about `TOOL_DESIGN.md`

Nothing gets deleted. The scenario library here is the concrete test surface the tool catalogue there proposes. When `TOOL_DESIGN.md`'s proposed tools (`route_status`, `nearby_stories`, …) land, they each come with at least one scenario in this harness.

## Where I'm starting

Phase 1, piece 1: `swift/Sources/MCPZimKit/StubZimService.swift`. Smallest increment that produces a visible output (scenario 1 for 2 variants with memory numbers). Everything else is additive on that base.
