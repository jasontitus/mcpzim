# Conversational rework — implementation summary

Branch: `claude/conversational-local-discovery-sLU9y`

This is the what-was-built companion to the design in
[`CONVERSATIONAL_REDESIGN.md`](./CONVERSATIONAL_REDESIGN.md). It records the
commits, the new components, how each was verified, and what's left.

## Goal

Turn the app from a single-shot "ask a topic / ask for more on that topic"
box into a **reactive walking companion**: follow-ups resolve naturally
("who built it", "the second one", "tell me more"), the conversation can
**drift across topics**, and every answer ends by offering grounded threads
to pull next. Faithful to the codebase's stated philosophy — *Swift carries
the thinking; the small on-device model just orchestrates and phrases.*

Two product decisions (asked + locked):
- **Reactive only** — no ambient/background interjections; answers end with
  concrete thread offers.
- **Hybrid drift** — Swift extracts/vets the candidate threads; the model
  phrases the offer.

## What shipped, by commit

| Commit | Scope |
|---|---|
| `87fd304` | Conversational core (discourse state + coreference + drift) — pure MCPZimKit, fully tested |
| `c09dbed` | ChatSession wiring — follow-ups + thread offers go live in the app |
| `4fae321` | On-device incremental embeddings index — pure MCPZimKit, tested |
| `2b008d6` | Embedder wiring — index fed by the app's existing `NLContextualEmbedding` |

## New components (MCPZimKit — `Sendable`/`Equatable`, no UI/ZIM/toolchain coupling)

### `ConversationFocus.swift`
The discourse state the host owns and mutates each turn:
- `entities` — recency-ordered, deduped, bounded stack of subjects in play
  (topic / place / route), each with its ZIM path and/or coordinates.
- `lastList` — the enumerated list last shown (for "the second one").
- `openThreads` — vetted drift candidates from the last result.
- `here` + `trail` — current GPS and a jitter-filtered movement trail.
- Types: `FocusEntity`, `DiscoveryThread`.

### `ReferenceResolver.swift`
Deterministic coreference, so the 4B model never has to do it:
- pronouns ("who built **it**" → "who built Stanford Memorial Church"),
- ordinals/selectors ("the second one", "the other one", "the last one"),
- descriptive picks ("the cathedral"),
- subjectless ellipsis ("how old" → "how old is X"; "tell me more" → "tell me
  more about X"),
- flags genuine ambiguity (`.ambiguous`) instead of guessing.

### `ConversationThreads.swift`
The deterministic half of the hybrid drift engine:
- `WikiLinks.parse(html:)` — extracts real outbound article links from Kiwix
  HTML (drops File:/Category:/external/anchor links; dedupes by destination).
- `extract(toolName:result:)` — turns a tool result into vetted threads
  (nearby POIs from places results; wikilinks + section headings from
  articles).
- `rank(_:focus:)` — dedupes against what's already discussed; orders lateral
  moves (places/wikilinks) before deeper ones (sections); caps the list.
- `orderBySimilarity(_:scores:)` — stable, sync re-rank by a precomputed
  key→score map (the async embedding work stays in the host).
- `offer(_:)` — the deterministic "Want to hear about X or Y?" caption.

### `Embeddings.swift`
On-device incremental semantic recall — the "little embeddings DB from the
articles we touch" idea, built runtime-only (nothing bundled):
- `VectorMath` — cosine / normalise / centroid.
- `TextEmbedder` protocol + `HashingEmbedder` — a dependency-free FNV-1a
  feature-hashing baseline so the pipeline runs with zero model assets.
- `EmbeddingIndex` actor — `add` / `nearest` / `centroid` / `scores`,
  LRU-capped, vectors stored normalised so kNN is a dot product.

### `IntentRouter.swift` (modified, additive)
`classify` gains an optional `focus:` parameter. A binding follow-up now
fetches the right article/place via `continuationIntent` instead of falling
through to a context-blind LLM. Existing call sites and tests untouched.

## App wiring (iOS target)

### `ChatSession.swift`
- `focus: ConversationFocus` + `embeddingIndex: EmbeddingIndex` on the session.
- `beginUserTurn()` per send; `updateLocation()` fed from the
  `LocationFetcher` subscription (movement trail).
- `IntentRouter.classify(..., focus:)` so follow-ups resolve.
- `updateFocusAfterTool(...)` — after every dispatch (fast-path + in-loop):
  records the subject, captures the enumerated list, extracts+ranks threads,
  and fire-and-forget-indexes the article lead for future semantic recall.
- `appendThreadOfferIfUseful()` (async) — ends terminal fast-path replies and
  the LLM prose reply with a thread offer, re-ranked by similarity to the
  conversation centroid when the index is warm; only offers wiki-backed
  places; skipped if the model already offered.

### `SemanticReranker.swift`
- `embedText(_:)` — exposes a public sentence embedding over the
  already-loaded `NLContextualEmbedding` (reused, not a new model), returning
  `nil` for a clean degrade when assets are cold.

## Verification

A **Swift 6.2 toolchain was installed** on the Linux container to compile- and
run-check the work:
- The whole MCPZimKit core (focus, resolver, threads, router continuation)
  passes a **42-check runtime harness**.
- The embeddings layer (vector math, hashing embedder determinism +
  lexical-overlap ordering, index dedupe/LRU/nearest/centroid/scores, thread
  re-rank) passes its harness.
- Every new public type is collision-free across the module.
- The exact MCPZimKit API calls the ChatSession/embedder wiring makes
  type-check and run.
- XCTest suites added: `ConversationFocusTests`, `ReferenceResolverTests`,
  `ConversationThreadsTests`, `EmbeddingsTests`.

**Not buildable on the Linux container:** `ChatSession.swift` and
`SemanticReranker.swift` live in the iOS target (MLX / CoreLocation /
NaturalLanguage / `os` / CryptoKit deps), so they build on macOS. The edits
there are small, guarded, and reuse existing paths, but they are the one
unverified surface — run `swift test` + an app build on macOS to confirm.

## Graceful-degradation chain (by design)

1. Embedding model warm → threads ranked by conversation centroid.
2. Model cold / first turn on a topic → deterministic source-priority order.
3. No threads → no offer. The embedding path never blocks or errors a reply.

## Data sources confirmed

- **streetzim** files carry (per this repo's own code, not the minimal public
  README): routing graph + geocoder, `wiki`/`wikidata` + `wiki_title`/
  `wiki_path` (the place→Wikipedia bridge that makes drift work), Overture
  business fields (`website`/`phone`/`brand`), and `kind`/`subtype`/`location`.
- **No bundled LLM/embeddings artifact** — focus, threads, and the embeddings
  index are all derived on-device at runtime from what the ZIM tools return.

## Remaining

### 1. Build + test on macOS (required — the one unverified surface)
- `swift test` in `swift/` to green the four new XCTest suites.
- Build the iOS app to confirm `ChatSession.swift` + `SemanticReranker.swift`
  compile (they couldn't be built on the Linux container). Likely-fine but
  unchecked: the async `appendThreadOfferIfUseful` call sites, the
  fire-and-forget `Task` in `indexText`, and the `embedText` addition.

### 2. Close the wikilink-drift gap (highest-value follow-up)
The drift engine extracts wikilinks via `ConversationThreads.articleThreads`,
but that only fires when the tool result carries `html` / `links` / `related`.
The article tools (`article_overview`, `get_article_section`) currently return
**sections only**, so today's live drift is *section headings + nearby places*
— the lateral "go to a related subject" wikilinks don't fire yet.
- Extend `ZimService` / `MCPToolAdapter` so article results include a small,
  ranked `related: [{title, path}]` array (reuse `WikiLinks.parse` over the raw
  article HTML before it's stripped to sections).
- Then `extract` surfaces real lateral threads and the conversation can
  actually *move across topics*, not just drill down.

### 3. Decide LLM-path offer: deterministic append vs. model-phrased
Today the LLM-path offer is a deterministic line appended after the model's
prose (chosen to avoid a prompt-cache chicken-and-egg, since the system message
is built once before the tool runs). The design's "hybrid" ideal is the model
phrasing the offer from a vetted `=== Threads ===` preamble block. To get true
model phrasing, inject that block per-turn (the design doc has the exact
injection point) — or keep the append, which is simpler and already grounded.

### 4. Smaller gaps / polish
- `what_is_here` produces no threads (not in `extract`'s switch) — add a
  neighbourhood/place thread so "where am I" also offers a next step.
- A `continue_thread(label)` tool so the user can pick an offered thread by
  voice ("tell me about the architect") and have it resolve against
  `focus.openThreads` and dispatch — the verbal equivalent of a suggestion chip.
- Tune the follow-up opener/pronoun word lists and the offer phrasing against
  real walking transcripts (`ios/MCPZimChatMacTests/ConversationalEvalTests`
  is the place to add follow-up/drift eval cases).

### 5. Embeddings — optional next tiers
- Conform a Core ML model to `TextEmbedder` if you want an alternative to
  `NLContextualEmbedding`; `HashingEmbedder` remains the zero-asset fallback.
- Local-area **seed-index**: background-embed the leads of place-articles
  within N km for instant grounded recall of where you physically are.

### 6. Explicitly deferred
- **Ambient interjections** (proactive "you've reached X" while walking, via a
  debounced background location loop + TTS) — the reactive flow ships first and
  is the foundation this would build on.
- Conversation branching / "go back to X" (`focus.entities` already keeps the
  stack; a `return_to(entity)` affordance is a small later addition).


## Prior art (full notes in the design doc)

Field Trip (Niantic, the ambient analog — shut down 2019), VoiceMap /
izi.TRAVEL (scripted GPS audio tours), llm-tools-kiwix / OpenZIM-MCP / zim-llm
(offline-Wikipedia RAG siblings). None combine offline Wikipedia depth + maps/
POIs + a small on-device model + a deterministic discourse layer into a
reactive walking companion — that intersection is the gap this fills.
