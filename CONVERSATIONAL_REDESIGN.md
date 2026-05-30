# Conversational redesign — from "topic box" to "walking companion"

## The problem with the current design

Today the app is a **single-shot question box**. A turn is processed in
isolation: `IntentRouter.classify` regex-matches the raw text against a
handful of canonical shapes (`"<X> near me"`, `"directions to Y"`, `"tell
me about Z"`, `"compare A and B"`), dispatches **one** tool, and renders a
templated caption. Anything that doesn't match falls to the LLM loop, which
re-encodes the whole transcript and leans on a ~650-line preamble whose
"follow-up rule" *asks the 4B model to do coreference in its head*.

That produces exactly the complaint:

> Right now I can only explicitly ask for a topic and for more on that
> topic, which is not interesting.

Concretely, the gaps are:

1. **No conversational memory the code can act on.** `IntentRouter` is
   stateless and runs *first*, with zero context. "Who built it?", "the
   older one", "what about over there", "tell me more" either fail to match
   any pattern (→ confused LLM) or mis-match. There is no place that knows
   what "it" *is*.
2. **Answers are dead ends.** A result is a templated caption + a map/article.
   Nothing invites the conversation onward. There is no notion of *threads*
   you could pull next.
3. **No topic drift.** The conversation can't *move*. You can go deeper on
   one subject (re-fetch the same article) but you can't follow an
   association ("this church was designed by the same architect as the
   museum you passed").
4. **Location is a coordinate, not a journey.** GPS is injected as a single
   `lat,lon` line. The app doesn't know you've *moved*, so it can't tie what
   you're discussing to where you now are.

`TOOL_DESIGN.md` already commits to the right philosophy — **"if a question
shape requires more than one tool call to answer reliably, package it as a
composite tool"** and **"the model never invents coordinates or synonym
candidates."** We extend that philosophy to *conversation itself*:
coreference, list-selection, and topic-adjacency are deterministic Swift,
not things we beg the 4B model to get right.

## Product decisions (locked)

- **Reactive, not ambient.** The companion speaks only when spoken to — but
  **every answer ends by offering 1–3 concrete threads** the user can pull
  ("Want the architect, Stanford White, or the 1906 quake that damaged it?").
  No background location loop, no battery/privacy cost, no interrupt-the-user
  UX. (Ambient interjections are a possible later phase; see *Future*.)
- **Hybrid drift engine.** Swift deterministically *extracts and vets* the
  candidate threads (real wikilinks from the article, real nearby places from
  the POI result — never hallucinated). The model only *phrases the offer*
  from that vetted list. This keeps suggestions grounded in the loaded ZIMs
  while letting the language stay natural.

## The shape of the new design

A thin **discourse state layer** sits between the raw user turn and the
tool/LLM machinery. Three new pure-Swift components in `MCPZimKit` carry it
(all `Sendable`/`Equatable`, no UI/ZIM deps, exercised by `swift test`):

```
            ┌─────────────────────────────────────────────┐
 user turn  │  ReferenceResolver.resolve(text, focus)     │
 ─────────► │   • "it"/"that"      → primary entity        │  ResolvedReference
            │   • "the second one" → focus.lastList[1]     │ ───────────────┐
            │   • "who built it?"  → subject := primary    │                │
            └─────────────────────────────────────────────┘                │
                                                                           ▼
         ┌──────────────────────────────────────────────────────────────────┐
         │  IntentRouter.classify(text, location, focus:)                    │
         │   • continuation + bound entity → fetch the RIGHT article/place   │
         │   • otherwise fall through to today's stateless patterns          │
         └──────────────────────────────────────────────────────────────────┘
                                                                           │
                              tool dispatch  ◄───────────────────────────────┘
                                     │
                                     ▼
         ┌──────────────────────────────────────────────────────────────────┐
         │  ConversationThreads.extract(toolName, result)                    │
         │   • article  → outbound WIKILINKS  + section headings             │
         │   • places   → the POIs we just showed (nearbyPlace threads)      │
         │   • rank/dedupe against what's already been discussed             │
         └──────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
         focus.remember(entity) · focus.setLastList(...) · focus.setThreads(...)
                                     │
                                     ▼   (vetted threads handed to the model)
         model phrases the answer + "you could go to: A, B, C"
```

`ConversationFocus` is the state object the host (`ChatSession`) owns and
mutates each turn. It tracks:

- **`entities`** — most-recent-first stack of subjects in play (topic /
  place / route), each with its ZIM path and/or coordinates so a follow-up
  can re-fetch the *exact* thing without re-searching.
- **`lastList`** — the enumerated list we last showed the user, in display
  order, so "the second one" / "the other church" resolve deterministically.
- **`openThreads`** — the vetted drift candidates surfaced from the last
  result.
- **`here` + `trail`** — current GPS *and a short movement trail*, so the
  host can tell you've walked into a new area.

## New components (this PR — pure, tested, no toolchain needed on the server)

| File | What it is |
|---|---|
| `swift/Sources/MCPZimKit/ConversationFocus.swift` | The discourse state: `FocusEntity`, `DiscoveryThread`, `ConversationFocus` with `remember` / `setLastList` / `setThreads` / `updateLocation`. |
| `swift/Sources/MCPZimKit/ReferenceResolver.swift` | Deterministic coreference: pronouns, ordinals/selectors ("the second one", "the other one", "the older one"), and elliptical subjects ("who built it?" → attach the active entity). Emits a `ResolvedReference` with the bound entity + a rewritten query. |
| `swift/Sources/MCPZimKit/ConversationThreads.swift` | The deterministic half of the hybrid drift engine: `WikiLinks.parse(html:)` pulls real outbound article links; `extract` turns any tool result into vetted `DiscoveryThread`s; `rank` dedupes against discussed entities; `offer` is a fallback caption for the no-LLM fast path. |
| `swift/Tests/MCPZimKitTests/ReferenceResolverTests.swift` | Coreference cases. |
| `swift/Tests/MCPZimKitTests/ConversationThreadsTests.swift` | Wikilink extraction + thread ranking + offer-line cases. |
| `swift/Tests/MCPZimKitTests/ConversationFocusTests.swift` | State transitions (recency, dedupe, list, trail/movement). |

`IntentRouter.classify` gains an **additive, defaulted** `focus:` parameter
(existing call sites and tests are untouched). When a `focus` is supplied and
the turn is a continuation that binds to a known entity, the router fetches
the *right* article/place instead of falling through to a context-blind LLM.

## Integration plan (next PR — `ChatSession` wiring, done on a Mac where `swift build`/`swift test` run)

These edits are deliberately **not** in this PR: `ChatSession.swift` is 3180
lines and can't be compiled in the server's Linux container, so editing it
blind risks breaking the build. Each step below is anchored to a real
file:line so it can be applied and tested on macOS.

1. **Own the state.** Add `private var focus = ConversationFocus()` next to
   `currentLocation` (`ChatSession.swift:196`). Call `focus.beginUserTurn()`
   in `send()` (around `:1680`) and `focus.updateLocation(...)` from the
   `LocationFetcher` subscription (`:1137`).

2. **Resolve before routing.** In `send()` where it calls
   `IntentRouter.classify(...)` (`:1696`), pass `focus: focus`. The router
   now resolves "it"/"the second one"/elliptical follow-ups to the right
   tool call. Keep the existing fallback to the LLM loop unchanged.

3. **Record what we fetched.** After every successful tool dispatch (the
   `adapter.dispatch` site, `:2035`, and the fast-path dispatch in `send`),
   call `focus.remember(FocusEntity(...))` for the subject, and
   `focus.setLastList(...)` when the tool returned an enumerable list
   (`placesToolNames`, `search`, `compare_articles`). Use
   `ConversationThreads.extract(toolName:result:)` →
   `ConversationThreads.rank(_, focus:)` → `focus.setThreads(...)`.

4. **Offer threads in the reply.** Two surfaces:
   - *Fast path (no LLM):* append `ConversationThreads.offer(focus.openThreads)`
     to the synthesized caption (e.g. in `IntentRouter.synthesize*Reply`
     callers).
   - *LLM path:* the preamble's `=== Threads ===` block (below) lists the
     vetted threads for *this* turn and instructs the model to end its reply
     by naturally offering 1–3 of them. Inject it the same way `locationLine`
     is injected at the tail of `composeSystemMessage` (`:644`) so the
     prompt-cache prefix stays stable.

5. **Preamble: continuation rule becomes deterministic.** The prose
   "follow-up rule" (`:614–633`) can shrink: the resolver has already bound
   the referent and rewritten the query before the model runs, so the model
   no longer has to guess what "it" means. Keep the *cache-reuse* guidance
   ("if the answer's already in the last 2 turns, don't re-fetch").

### New preamble block (tail-injected, like `locationLine`)

```
=== Threads you can offer ===
After you answer, you MAY end with a short, natural offer of where to go
next — pick 1–3 of these vetted threads (do NOT invent others; these are the
only ones grounded in the loaded archives):
  • Stanford White (the architect)        [topic]
  • 1906 San Francisco earthquake          [topic]
  • Cantor Arts Center — 350 m away         [place]
Phrase it conversationally ("If you like, I can tell you about the architect,
Stanford White, or the 1906 quake that damaged it."). If none fit the flow,
skip the offer.
```

## New companion tools (later phase, once threads are flowing)

These ride on the same machinery and each subsumes a multi-call chain the
4B model currently bungles:

- **`continue_thread(label)`** — the user picked a thread by name or by
  reference. Resolves `label` against `focus.openThreads` (exact → fuzzy)
  and dispatches `article_overview` / `near_places` for it. The verbal
  equivalent of tapping a suggestion chip.
- **`explore_here(lat, lon)`** — supersedes `nearby_stories`: returns one
  vivid lead-paragraph hook for where you are **plus** 3–4 pre-vetted
  threads (notable nearby places + their wikilinks), so the *first* reply is
  already a branch point.
- **`whats_around_now()`** — uses `focus.trail`: if you've crossed into a new
  named area since the last such call, leads with "you've reached <area>"
  and a fresh hook; otherwise nudges along the current thread.

## Future (explicitly out of scope now)

- **Ambient interjections.** A debounced background location-trigger loop
  that offers a tidbit when you enter a notable area, optionally spoken via
  the existing Kokoro TTS path. Deferred per the product decision above —
  the reactive thread-offer flow ships first and is the foundation it would
  build on.
- **Conversation branching / "go back to X".** `focus.entities` already keeps
  the stack; a `return_to(entity)` affordance is a small addition once the
  state layer is live.

## Related work — who else is in this space

Worth knowing what exists, because the *combination* here looks novel even
though each ingredient has prior art.

**Location-discovery / walking companions (the product analog).**
- **Field Trip** (Niantic / Google, 2012–2019) was the canonical "ambient
  local guide": GPS-triggered cards with historical tidbits and reviews as you
  walked. It validated the *ambient* mode we deliberately deferred — and note
  it was **shut down in 2019**, so the niche is currently open. It was
  proactive-only, single-source-card, **not conversational** and had no
  follow-up/drift. ([Wikipedia](https://en.wikipedia.org/wiki/Field_Trip_(application)),
  [TechCrunch](https://techcrunch.com/2012/09/27/google-launches-fieldtrip-a-location-aware-app-that-helps-you-find-cool-stuff-around-you/))
- **VoiceMap**, **izi.TRAVEL**, **Detour** (defunct) are GPS-triggered audio
  walking tours — location-aware narration, but **authored and linear**: a
  human writes a fixed route; you can't ask anything. ([VoiceMap](https://voicemap.me/),
  [izi.TRAVEL](https://apps.apple.com/us/app/izi-travel-audio-tour-guide/id554726752))
- **Wikipedia's own "Nearby"/Places tab** lists articles around you — pure
  discovery, no conversation, online. ([Dexerto](https://www.dexerto.com/entertainment/wikipedia-app-users-discover-wild-location-feature-that-shows-articles-near-you-3348880/))

**Offline-Wikipedia + LLM (mcpzim's own neighbourhood).** The MCP-over-ZIM
pattern this repo uses is *not* unique — there are several siblings, which is
useful for borrowing chunking/citation ideas:
- **llm-tools-kiwix** — exposes ZIM chunks + metadata + citations to LLMs via a
  CLI/Python server. Closest cousin to mcpzim. ([GitHub](https://github.com/mozanunal/llm-tools-kiwix))
- **OpenZIM MCP Server** — structured ZIM access over MCP. ([guide](https://skywork.ai/skypage/en/openzim-mcp-server-offline-ai-knowledge/1981670209707151360))
- **zim-llm** — ZIM → vector DB → offline RAG. The embeddings approach below,
  already built for the desktop case. ([GitHub](https://github.com/rouralberto/zim-llm))
- **Project N.O.M.A.D** (Ollama + Qdrant offline RAG) and **Volo** (Kiwix RAG to
  cut hallucinations) round out the offline-RAG cluster. ([N.O.M.A.D](https://topaiproduct.com/2026/03/21/project-n-o-m-a-d-went-from-zero-to-5000-github-stars-by-combining-survivalism-with-local-ai/))

**Conversational-search research** backs the specific moves in this design:
mixed-initiative search (system *asks* clarifying questions when a reference is
**ambiguous** — exactly our `.ambiguous` binding) and query
suggestion/refinement (our **thread offers**) are the two studied mechanisms
for keeping an exploratory conversation moving. ([arXiv 2112.07308](https://arxiv.org/abs/2112.07308),
[arXiv 2109.05955](https://arxiv.org/pdf/2109.05955))

**The gap we fill.** Nobody combines *offline* Wikipedia depth **+** maps/POIs
**+** a small **on-device** model **+** a deterministic discourse layer
(coreference, list-selection, location-trail, grounded topic-drift) into a
**reactive walking companion**. Field Trip had the place sense but no
conversation; the audio tours are scripted; the ZIM-RAG tools are
desktop Q&A boxes with no place or drift. That intersection is the product.

## On-device embeddings for candidate recall (your idea — and a strong fit)

The drift engine above is keyword/heuristic and link-graph based. A small
on-device embedding index makes it dramatically better at *recall*, and your
instinct to **build it incrementally from the articles we actually touch** is
exactly right — far better than embedding all of Wikipedia.

**Why incremental beats "embed everything".** Full-Wikipedia embeddings are
~120 GB in fp32 for ~36 M passages; even binary-quantized that's still several
GB and a huge one-time build. ([HF: building a 2 GB Wikipedia vector DB](https://gpt3experiments.substack.com/p/building-a-vector-database-in-2gb),
[HF: embedding quantization](https://huggingface.co/blog/embedding-quantization))
You don't need it. Two cheap, bounded indices cover the walking-companion case:

1. **Touch-index (grows with the conversation).** Every time we open an article
   or a nearby-place lead, embed its lead paragraph(s) and store the vector
   keyed by ZIM path. This is tiny (kilobytes per article), privacy-preserving,
   and grows precisely along the user's interests. It powers:
   - *Fuzzy follow-ups that don't keyword-match* — "the romantic one", "the one
     with the politics angle", "the sad story you mentioned" → nearest-neighbour
     over what we've already shown, feeding `ReferenceResolver`'s list/
     descriptive selectors when token overlap fails.
   - *Drift ranked by the conversation, not just the last article* — rank
     `openThreads` by similarity to the **centroid of the focus** (everything
     discussed this session), so offers feel like they follow the thread of the
     whole stroll, not just the last sentence.
   - *"What does this remind you of"* lateral jumps across topics already seen.

2. **Local-area seed-index (built once per area, bounded by radius).** "All the
   Wikipedia about the places around me" is already a resource here. A
   background pass can embed the lead of every place-article within N km (a few
   hundred to low-thousands of vectors — megabytes), giving **instant grounded
   semantic recall for where you physically are** without touching the other
   ~6 M articles. This is the on-device analog of WMF's "Related Pages"
   (More-Like-This) and reading-session/`Wikipedia2Vec` navigation vectors.
   ([WMF Navigation Vectors](https://meta.wikimedia.org/wiki/Research:Wikipedia_Navigation_Vectors),
   [Wikipedia2Vec](https://arxiv.org/abs/1812.06280))

**Free tier first (no model needed).** The Wikipedia **hyperlink graph** —
which `WikiLinks.parse` already extracts — *is* a relatedness signal: articles
that link each other are related, established in the relatedness literature.
([Studying the Wikipedia Hyperlink Graph](https://arxiv.org/pdf/1503.01655))
So we ship link-graph drift now, and layer embeddings in as tier 2 for the
recall cases links miss.

**Concrete stack (all proven on mobile).**
- *Embedder:* `all-MiniLM-L6-v2` (384-d) or `bge-small-en`, ~20–90 MB, runs on
  device; on iOS via Core ML or Apple's `NLEmbedding`. ([sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  [On-device RAG guide](https://medium.com/google-developer-experts/on-device-rag-for-app-developers-embeddings-vector-search-and-beyond-47127e954c24))
- *Store:* `sqlite-vec` (vectors as BLOBs + cosine) or ObjectBox's built-in
  HNSW — both designed for on-device. ([sqlite as vector DB](https://www.sqliteforum.com/p/sqlite-as-a-vector-database))
- *Quantization:* int8 (4–8× smaller) keeps the touch-index trivially small;
  not even needed until the index is large.

**Status — the index is built (`swift/Sources/MCPZimKit/Embeddings.swift`).**
Shipped and unit-tested:
- `VectorMath` (cosine / normalise / centroid), `TextEmbedder` protocol,
  `HashingEmbedder` (dependency-free FNV-1a feature-hashing baseline so the
  pipeline works with **zero model assets**), and the `EmbeddingIndex` actor
  (`add` / `nearest` / `centroid` / `scores`, LRU-capped, vectors stored
  normalised).
- `ConversationThreads.orderBySimilarity(_:scores:)` re-ranks the existing
  vetted threads by a precomputed key→score map — a stable sort kept sync/pure
  so the async embedding work stays in the host.

**Remaining host wiring (macOS/iOS, next):**
1. Conform a Core ML / `NLEmbedding` model to `TextEmbedder` for real semantics
   (the `HashingEmbedder` is the fallback until then).
2. In `updateFocusAfterTool`, after recording an article/place, `await
   index.add(key: zimPath, title:, vector: embedder.embed(leadParagraph))`.
3. Before offering threads, compute `centroid(of: focus.entities' keys)` →
   `scores(for: threadKeys, against: centroid)` → `orderBySimilarity` so the
   offer follows the whole conversation, not just the last sentence. Falls back
   cleanly to the current source-priority order when the index is cold.

Nothing in the shipped discourse core had to change to adopt this — it's
strictly additive, exactly as intended.

## Why this is the right increment

- It attacks the actual complaint — follow-ups and drift — at the layer that
  was missing (deterministic discourse state), not by adding more prose to a
  preamble the small model can't reliably follow.
- It stays faithful to the codebase's stated philosophy: tools/Swift carry
  the thinking; the model orchestrates and phrases.
- The core lands fully unit-tested with **zero** ZIM/UI/toolchain coupling,
  and the risky orchestrator wiring is a separate, file:line-anchored step
  the user applies where the compiler can check it.
- The embeddings layer you suggested has a clean, additive home in the same
  types — link-graph drift ships now, semantic recall layers on top later.

