# Zimfo — architecture notes

Zimfo is an on-device chat app (macOS + iOS) that answers questions
from locally-loaded ZIM archives (Wikipedia, streetzim street data,
mdwiki, generic). A local LLM (Gemma 4 4B via MLX) drives a tool loop
over those archives. Nothing leaves the device.

This doc captures the architecture as of the Phase 1 section-chunking
work. It focuses on the pieces that took the most iteration to get
right — provider abstraction, tool surface, prompt path selection,
memory budgeting — and deliberately skips UI details that are already
self-documenting in the views.

## Ground truth: `MCPZimKit` (SwiftPM)

Transport-agnostic Swift package shared by the iOS/macOS app. No
UIKit, SwiftUI, or libzim references here — the app injects
`ZimReader`s from its own libzim bridge. Major types:

- `ZimService` (protocol) + `DefaultZimService` (actor) — the facade
  the tool adapter dispatches against. Backed by a list of
  `(name, ZimReader)` pairs. Holds per-zim caches for the routing
  graph, streetzim search-data chunks, category manifests, and
  streetzim-meta bboxes.
- `MCPToolAdapter` (actor) — turns `ZimService` into an MCP-style
  tool registry + a single `dispatch(tool:args:)` entry point. Has
  a `ToolSurface` (`.conversational` / `.full`) so conversational
  LLMs don't see raw-coord tools that they misuse, while
  programmatic callers (Siri AppIntents, tests) keep the full
  surface.
- `SZRGGraph` — zero-copy routing-graph parser. Supports format
  versions 2 and 3 (v3 widens the edge record's `geom_idx` to a
  full u32 to address >16.78 M geoms; Japan's extract tripped v2).
- `Geocoder`, `Routing`, `ArticleSections`, `ChatToolCallParser`,
  `Gemma4PromptTemplate`, `Gemma4ToolCallParser`, `Gemma4ToolFormat`
  — everything else.

Unit tests live under `swift/Tests/MCPZimKitTests/`. `swift test`
runs them without the app target.

## Tool surface

### Conversational (used by the chat UI)

| Tool                    | Purpose                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `list_libraries`        | Inventory loaded ZIMs + capabilities.                                                                                                            |
| `search`                | Keyword FTS across all ZIMs. Short keywords, not sentences. Model description steers location questions to `near_named_place`.                   |
| `get_article`           | Fetch one ZIM entry as text. Body trimmed at `articleCapKB` (device-tier default).                                                               |
| `list_article_sections` | Outline of `<h2>`/`<h3>` sections with byte sizes. Navigation / references / see-also stripped.                                                  |
| `get_article_section`   | Fetch just one named section. `"lead"` returns the intro. Case-insensitive, prefix-tolerant match.                                               |
| `get_main_page`         | Home-page entry of one or every ZIM.                                                                                                             |
| `zim_info`              | `streetzim-meta.json` descriptor (bbox, has-routing, feature counts). Models use this to decide coverage before routing.                         |
| `near_named_place`      | "What's around `<place>`?" Carries `by_category` breakdown + top-N results across all categories. Documented drill-in protocol via `kinds=[…]`.  |
| `route_from_places`     | Plan a driving route between two free-text names. Returns distance, duration, turn-by-turn (capped), locale-formatted by ChatSession.            |

### Full surface (programmatic callers)

Adds back `geocode`, `near_places`, `plan_driving_route` — raw-coord
tools. The conversational surface hides them because small LLMs
routinely pass text where lat/lon goes, or default `(0,0)` on
missing args.

## Providers (`ModelProvider`)

The chat loop only talks to a `ModelProvider`. Three ship today:

1. **`Gemma4Provider`** — default. Gemma 4 4B E2B 4-bit via MLX +
   Swift-Gemma4-Core. Runs in-process with MLX/Metal dispatches.
   Tool calls travel via Gemma's native `<|tool_call>…<tool_call|>`
   DSL; prompt built by `Gemma4PromptTemplate`; responses parsed
   by `Gemma4ToolCallParser`.
2. **`FoundationModelsProvider` (text-loop mode)** — Apple on-device
   LLM (Apple Intelligence). Text-loop flavor: same `<tool_call>
   {json}</tool_call>` path as everything else. Fresh
   `LanguageModelSession` per `generate()` call (required to sidestep
   `concurrentRequests` when ChatSession breaks mid-stream on a tool
   call).
3. **`FoundationModelsProvider` (native-tools mode)** — Same class,
   `useNativeTools=true`. Uses Apple's structured `Tool` +
   `@Generable` protocol. ChatSession short-circuits the text tool
   loop and calls a dedicated `runNativeToolsTurn` that streams one
   response from a persistent session — tool calls round-trip through
   our `Tool.call()` methods inside the framework. System message
   goes in as `instructions:` (cached once per session).

Apple FM is slower than Gemma in practice: first-token latency of
20–40 s, a 4,096-token context ceiling, and the system daemon
unloads the model on idle (our `prewarm(promptPrefix:)` on install
+ composer-focus prewarm helps only partially). Kept selectable in
the picker but not the default.

See `Providers/FoundationModelsProvider.swift` for the warm session
+ error-recovery logic and `FoundationModelsNativeTools.swift` for
the seven `Tool` conformances that mirror the conversational
surface.

## ChatSession

Single `@Observable`, main-actor type that owns:

- The library (`LibraryEntry`s for each opened ZIM), the active
  `DefaultZimService`, and the `MCPToolAdapter` built from it.
- The selected provider and its load state. Persists the choice via
  `UserDefaults[chat.selectedModelId]`.
- The transcript (`messages`), tool-call traces, and debug log.
- Generation tunables (`articleCapKB` persisted, max reply tokens
  from `DeviceProfile.current`).

### Dispatch loop (`runGenerationLoop`)

For Gemma and Apple FM text-loop:

1. Render the full transcript (`formatTranscript`) with the tool
   preamble.
2. Call `selectedModel.generate(prompt:)`.
3. Stream chunks into the assistant message; on each chunk scan for
   a `<tool_call>…</tool_call>` block.
4. When found: dispatch via `adapter`, inject the tool response as
   a new turn, re-render, loop.
5. Cap at 6 iterations. If we hit the cap with no user-facing text,
   force a final no-tool-call summary turn so the user sees
   something.

For Apple FM native-tools:

1. `setNativeInstructions(systemMessage)` — rebuilds the session
   only if the instructions text changed.
2. `runNativeToolsTurn(provider:)` — pulls only the newest user
   message, hands it to `generateNativeTurn(userMessage:)`, streams
   the reply. The framework handles tool calls internally via our
   `Tool.call()` bodies; no text-level tool loop.

### Prompt shape

Tool preamble + behavioural rules both live in the system message
(`toolsPreamble(registry:)` + inline instructions built each turn).
Gemma gets its native DSL form; every other provider gets the
generic `<|role|>…` template. System-prompt rules currently cover:

- Call tools immediately (don't ask for confirmation).
- Routing: distance + duration + numbered turn-by-turn, verbatim.
- "What's nearby": lead with `by_category` breakdown; don't invent
  names from counts.
- "Tell me about X" / "how does X work" / "explain X": `search` →
  `list_article_sections` → pick 1–3 sections → `get_article_section`
  per pick → answer. Factoid = 1 lead section; topical = 3 sections;
  explanatory = sections across 2–3 articles.

## Device tiering (`DeviceProfile`)

Memory budgets scale by tier so 6 GB iPhones don't jetsam:

| Tier       | Trigger        | Article cap | Max reply | MLX cache |
|------------|----------------|-------------|-----------|-----------|
| `tight`    | ~4 GB iPhone   | 6 KB        | 256 tok   | 256 MB    |
| `snug`     | ~6 GB iPhone   | 12 KB       | 384 tok   | 384 MB    |
| `balanced` | ~8 GB iPhone   | 16 KB       | 512 tok   | 512 MB    |
| `generous` | 12+ GB iPhone  | 24 KB       | 512 tok   | 640 MB    |
| `mac`      | macOS          | 24 KB       | 512 tok   | 512 MB    |

All three knobs drive real memory costs:

- Article cap → prompt size → KV-cache spike on stream open.
- Max reply tokens → MLX pre-reserves KV-cache headroom
  proportional to this.
- MLX cache limit → post-stream buffer pool ceiling.

User can override `articleCapKB` via Library → Generation. Tier
label is shown in that section for clarity.

## App intents (Siri)

`ZimfoIntents.swift` hosts the intents ("Ask Zimfo how to get to X",
"Ask Zimfo what's around here", "Ask Zimfo how much longer", etc.).
`ZimfoRunner` builds a service/adapter from the app's persisted
library (sandbox Documents + external bookmarks). `ZimfoContext` is
an actor that persists the active route (polyline + cumulative
distance) across launches so follow-up queries like "how much
longer?" can answer from snap-to-polyline progress.

## Eval harness

`MCPZimChatMacTests/ConversationalEvalTests` is an XCTest-based
scenario runner that drives a real `ChatSession` with a real model
and scores replies against lenient assertions (expected tool calls,
required substrings, banned substrings). Gated on
`ZIMBLE_TEST_STREETZIM` / `ZIMBLE_TEST_WIKIPEDIA` env vars (passed
via `TEST_RUNNER_` prefix). `ZIMBLE_TEST_MODEL_ID` selects which
provider to evaluate — rerun the same scenarios against any model.

`tools/eval.sh` wraps it, optionally sampling Apple Neural Engine
utilization via `sudo powermetrics -s ane …` and summarizing at the
end.

## Phase 1 (section chunking) — what it changed

Before: `get_article` returned up to 24 KB of the raw article body.
The model saw a truncated monolith; on a 442 KB Wikipedia article
the prompt hit 33 K chars / 11 K tokens and spiked MLX's KV cache
to ~14 GB at stream open (enough to jetsam on any iPhone).

After: `list_article_sections` + `get_article_section` let the model
read just the relevant sections. A typical section is ~1–3 KB, so
three sections in a prompt is ~10 KB instead of 24 KB. Nav/refs are
stripped server-side so the outline the model picks from is
signal-dense.

`ArticleSections.parse(html:)` is a regex-based heading splitter —
no SwiftSoup dependency, fast enough to run on every call. It
drops `<script>` / `<style>` / `<table>` / `<figure>` / `<nav>`
blocks before text extraction and decodes the handful of HTML
entities Kiwix emits.

## Phase 2 — complexity router + category-aware retrieval + map-reduce

`QueryComplexity.classify(_:)` is a keyword-heuristic classifier
in MCPZimKit that buckets each user turn into one of four
categories:

- `navigational` — routing / "what's around" / "near me" queries.
  Handled by streetzim tools.
- `factoid` — short, single-fact lookup. Lead section is usually
  enough.
- `topical` — "tell me about X". Fixed chain: `search` →
  `list_article_sections` → lead + at least one content section.
- `explanatory` — "explain X" / "why did X happen" / "compare X
  and Y". Fixed chain: minimum 2 `get_article_section` calls
  across one or two articles, then synthesise. Map-reduce
  post-pass kicks in when ≥2 sources were pulled.

Every turn logs its classification to the `[Router]` debug pane
category and the system prompt gets a per-turn "This turn's
classification: …" block with the category-specific chain. Unit
tests in `QueryComplexityTests.swift` cover the signals.

### 2c — map-reduce synthesis for explanatory turns

After `runGenerationLoop()` finishes, if the classification was
`.explanatory` and the model fetched ≥2 `get_article_section`
sources, `runMapReduce(userQuery:sectionTraces:)` kicks in:

- **Map** — each section body goes through its own stateless
  `generate()` call with a tight prompt ("list 3–6 bullet points
  from this text that answer the user's question, no outside
  knowledge"). `maxTokens: 256`. Peak KV-cache reservation is
  bounded by the largest single section, not the sum of all
  sections.
- **Reduce** — the digests are concatenated into one prompt
  asking for a grounded, cite-inline synthesis. Streamed to the
  UI, replacing the first-pass text.

The first-pass synthesis is "wasted" (we discard and redo) —
chosen over a bigger refactor that would intercept the normal
loop before its final prose step. Worth the quality lift on
multi-source questions. `[MapReduce]` debug category traces the
phases.

## Phase 3 — semantic reranker on BM25 top-20

Reason: libzim's Xapian BM25 alone puts "Crypto-shredding" at #1
for "quantum computing's effect on encryption" (dense keyword
overlap), while "Post-quantum cryptography" is actually the
correct article. A sentence-embedding cosine rerank moves the
right article up.

- `SemanticReranker` (actor singleton, iOS/macOS only) wraps
  `NLContextualEmbedding(language: .english)`. ANE-accelerated,
  ~100 MB model that ships with the OS.
- `MCPToolAdapter` grows an optional `HitReranker` callback. The
  cross-platform MCPZimKit stays free of `NaturalLanguage`
  imports.
- When installed, `search` over-fetches the top 20 BM25
  candidates, reranks by cosine on `(title + snippet)` vs query
  embedding, returns the requested top-K.
- Per-article embedding cache keyed by `zim:path` in memory
  (cheap, survives for the process lifetime).
- Graceful passthrough if the framework can't load (stays BM25).

Related quick win: `ZimService.search` now populates each hit's
`snippet` field with the first ~220 chars of the article's lead
paragraph (via `ArticleSections.parse(html:)`). The model can
read snippets to judge topical relevance even without the
reranker — which is what caught the Crypto-shredding misfire
interactively before Phase 3 landed.

## Sources-used audit trail

Every assistant reply that made tool calls now renders a
**"Sources used (N)"** DisclosureGroup at the top of its bubble.
Expanded by default. Shows each tool call's args + result;
article-returning tools (`get_article`, `get_article_section`)
render the `text` field as readable prose with a `Title §
Section` header instead of raw JSON. Directly answers "did the
model use Wikipedia or its training priors?"

Grounding policy in the system prompt pairs with this: every
factual claim must trace to a tool result (this turn or an
earlier turn), follow-ups reuse prior articles, cite inline, and
say "I don't have that" when the ZIMs don't cover the question.

## Voice chat (both platforms)

`Views/VoiceChatView.swift` + `Voice/*` host a hands-free loop
that's one sheet away from the composer's 🎤 button:

- `SpeechRecognizerService` — Apple's on-device Speech framework
  (best-on-device mode).
- `VoiceChatController` — mic tap → silence VAD → fires
  `ChatSession.send(_:)` → watches the assistant stream → hands
  the final reply to `TTSService` → back to listening.
- `TTSService` — `AVSpeechSynthesizer` by default, optional
  Kokoro backend (`#if canImport(KokoroSwift)`).
- Works on macOS unchanged; Info-Mac.plist has the mic +
  speech-recognition strings and the app entitlement carries
  `com.apple.security.device.audio-input`.

Because voice goes through `ChatSession.send`, voice queries get
the full Phase 2/3 tooling — router, section chunking, rerank,
map-reduce.

## Eval harness (Phase 2/3 additions)

`ConversationalEvalTests.swift` gained:

- `TurnExpect.minimumToolCallCounts: [String: Int]` — asserts a
  tool was called at least N times in a turn. Catches "model
  called `get_article_section` once and stopped" regressions.
- `test_18_quantum_encryption_must_ground_in_sections` — pins
  the "≥2 sections before synthesis" rule.
- `test_19_sputnik_explanatory_then_year_followup` — covers the
  explanatory → factoid-pronoun arc and the "1957 from memory"
  anti-pattern (reply must include "1957" AND a citation / tool
  call — rules out pure training-knowledge fallback).

Score at the time of writing: 14/20 pass. The failing two new
regressions (18, 19) are the motivation for Phase 2d below.

## Next (Phase 2d)

Programmatic enforcement for the minimum-section rule. The
tightened prompt helps but Gemma-4 2B still cheats on multi-
section explanatory turns. After `runGenerationLoop()`, if
`lastQueryComplexity` is `.explanatory` / `.topical` and fewer
than the required `get_article_section` calls happened, re-enter
the tool loop with a synthetic user turn "you've only fetched N
sections — call `list_article_sections` then
`get_article_section` on one more before answering." Structural,
not aspirational.

Also open for Phase 3+:

- Persist embedding cache to disk (Application Support) so
  second launches hit it warm.
- Rerank section-level (not article-level) for drill-in queries
  to pick the best section without forcing `list_article_sections`.
- Apple-FM native-tool surface: add a `Tool` conformance for
  `list_article_sections` / `get_article_section` so the native-
  tools path doesn't regress to single-section synthesis.

See conversation history for rationale on any of the above.
