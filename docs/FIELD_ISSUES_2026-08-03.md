# Field issues from device logs — 2026-08-03

Source sessions: `eval/corpus/raw/5393c74a/2026-08-03_21-34-40.log` (k1 kart /
go-cart) and the war-of-1812 session (`2026-08-03_23-14-36.log`, preserved in
the corpus). Replay transcripts with
`MCPZimEvalCLI --probe-discuss --turn … --turn …` as the acceptance test.

**STATUS 2026-08-04: all items below are FIXED and replay-verified except the
P2 verify-only pair.** Mac replay of the war transcript now shows turn 2
answering casualties from the pinned article (attribution 1.00) and turn 3
appending warm (`reused=996 · prefill=1.9s`) where the field log had
`reused=0 · 6s` every turn.

## P0 — grounded KV cache diverging every turn — FIXED (was routing, not KV)

The doc's original suspect (`rawAssistantText`/A2 byte divergence) was
WRONG: the field log's own turn 3 appended warm (`reused=937 ·
prefill=0.004s`), proving the cache machinery fine. Every cold turn was a
cache DISCARD caused by conversational misrouting (below) — the discussion
pin kept being cleared, and rebinding pays a full re-prefill by design.
Fixing the routing fixed the "regression". Three levels:

1. `IntentRouter.titleNamesPinnedSubject` — "the war" (stateless title
   parse of "…died on each side in the war?") now token-matches the pinned
   "War of 1812" after article-stripping, so the turn stays in the
   discussion. Substring matching had missed it and the turn LEFT.
2. `ConversationFocus.lastListKind` — a disambiguation offer leaves two
   topics in `lastList`, and `comparisonContinuationRoute` treated ANY
   two-topic list as a compared pair: "each side" routed to
   `compare_articles(War of 1812, French invasion of Russia)` and exited
   the discussion. Comparison follow-ups now require `.comparison`.
3. Discussion-leave paths now LOG (`discussion leave: tool(title) vs
   pinned topic`) — the field log had no trace of why the pin dropped,
   which is why the wrong suspect was written down.

## P1 — silent dead turn on fast-path locate miss — FIXED

"Where is k1 kart" → `ZimService.geocodeVariants` now appends progressive
trailing-token-drop variants ("k1 kart" → "k1"), so "K1 Speed" resolves by
prefix (the index never contained "kart"). The geocode-miss reply also logs
under `[Assistant]` now — the field log could not show whether the miss
message rendered at all, so the honest-miss handler looked dead when it
may not have been.

## P1 — near_places unmapped kind returns instant zero — FIXED

`kindSynonyms` (21 phrasings → existing chips: petrol→fuel, drugstore→
pharmacy, er/urgent care→health members, bookstore→shops, …) consulted in
`chipsFor` and at `effectiveKinds`, so synonyms inherit niche/broad chip
behavior. Kinds that still map to no chip get a bounded name-search
fallback over the prefix-chunked search-data (≤48 chunk loads, ≤200
matches, poi/place rows only, radius-filtered) before returning empty —
never a full scan, so the statewide-ZIM OOM guard stands.

## P1 — wrong-article binding + lost facet in war-of-1812 session — FIXED

1. Title cleanup: `IntentRouter.collapseStutter` (dictation doubles) +
   `strippingTrailingInterrogativeClause` (interrogative + auxiliary only,
   so "Doctor Who" / "The Man Who Sold the World" survive) run before the
   tell-me-about title dispatch. The field turn now dispatches
   `article_overview("the war of 1812")` instead of search-rescuing the
   stuttered tail to "1812 Louisiana hurricane".
2. Anaphora: covered by the P0 fixes above — "the war" stays on the
   pinned subject.
3. Corrections/picks: the ambiguity gate now stashes `pendingDisambiguation`
   (question + candidates, one turn). The next turn resolves via
   `ReferenceResolver.clarificationPick` (negation "not the one about X",
   exact name, positional, unique-token) and re-runs the ORIGINAL question
   against the pick via `groundedQuestionOverride` — the pick text is no
   longer answered as if it were the question. A turn that literally NAMES
   one candidate (`ReferenceResolver.namedCandidate`, ≥2-token contiguous
   run) self-resolves the gate instead of re-asking — the field session's
   exact-title reply "The war of 1812" had re-triggered the same
   clarification.

Regression tests: `ClarificationAndTitleCleanupTests` (23),
`GeocodeVariantsTests` (9), `NearPlacesKindFallbackTests` (13), plus
existing comparison tests retagged with `.comparison`.

## P2 — verify-only (still open)

- Disneyland map tile AbortErrors (code=20, z7–z14): user reports the map
  looked fine — consistent with MapLibre canceling superseded tiles during
  the initial camera jump. Confirm benign, then drop these lines to a
  counter to keep logs readable.
- mcp-deploy-verify.sh still reads backgrounded/suspended as "app died";
  teach it to check the session log tail for a clean "backgrounded" line
  before declaring failure (two false alarms so far).

## Standing context

Attribution chips (496c7a14) are live and already catching bad retrieval —
keep the `[Attrib]` line in any refactor. Firebase key rotation is pending
at next TestFlight cycle (see memory + gitignore note). The Mac
`GemmaToolEmissionTests` failure (E2B int4 weight-shape mismatch) predates
all of this — likely fallout of the mlx-swift-lm 0.31.3 pin-back — and
only affects the Mac eval harness, not the shipping llama.cpp path.
