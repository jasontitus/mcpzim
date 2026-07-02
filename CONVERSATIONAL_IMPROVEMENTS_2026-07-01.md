# Conversational improvements — 2026-07-01

Follow-up round to [`CONVERSATIONAL_REDESIGN.md`](./CONVERSATIONAL_REDESIGN.md) /
[`CONVERSATIONAL_REWORK_SUMMARY.md`](./CONVERSATIONAL_REWORK_SUMMARY.md), driven
by a four-way deep review of the shipped pipeline (discourse layer, tool
surface, eval evidence, maps UX). Each item names the defect the review found
and what changed. Originally developed against the pre-LFM branch and rebased
onto main 2026-07-01; the per-provider preamble work from that branch was
DROPPED in the rebase — main's single trained-format prompt path (every
provider renders via its own template; the FTs are trained on exactly that
shape) supersedes it.

## 1. Composite tools' depth reached the model as 160 words of lead

`article_overview` / `compare_articles` pre-chew lead + priority narrative
sections (`pickOverview`) and relations-article sections — but
`trimForModel` cut every payload to lead-only/160 words, so "history of X"
and "how have A and B gotten along" produced shallow first answers.

**Change** (`ChatSession.trimForModel`): keep the lead (160-word cap) PLUS up
to two picked sections (120-word cap each), with a `sections_dropped` marker.
Worst case ≈ 520 tokens/article — an order of magnitude under the raw-article
payload that caused the 2026-04 jetsam.

## 2. Drift died on drill-in

Drill-in tools (`get_article_section`, `narrate_article`) produced ZERO
threads and every dispatch overwrote `focus.openThreads` — so the "want to
hear about X?" offer vanished on deep-dive turns.

**Change** (`MCPToolAdapter.swift`, `ConversationThreads.swift`,
`ChatSession.swift`):

- `get_article_section` / `get_article_by_title` now emit the same
  best-effort `related[]` wikilinks `article_overview` carries.
- `updateFocusAfterTool` keeps the previous threads when a tool contributes
  none (stale-but-grounded beats empty), and returns early on `error`
  results so missed fetches stop polluting focus with phantom entities.
- (main had independently added `what_is_here` → `nearby` sibling threads;
  that version stands.)

## 3. Follow-up correctness

- **"How far is it?" answered with a POI dump.** Distance-shaped follow-ups
  ("how far", "which way", "can I walk") now route to the new `distance_to`
  tool; travel-shaped ones ("directions", "how long", "get to") still route
  to `route_from_places` (real driving duration); proximity ones ("what's
  around it") to `near_places`.
- **Pronoun binding ignored kind.** "How far is it?" after a topic turn bound
  "it" to the topic and silently fell through to `article_overview`. A
  locational follow-up now rebinds to `focus.mostRecent(kind: .place)`.
- **Ambiguity was detected then discarded.** `.ambiguous` (descriptive
  selector matching several list items) never reached the user; the stateless
  patterns guessed. `send()` now answers with a deterministic clarifying
  question ("Which one do you mean — A, B, or C?") and leaves `lastList`
  intact so the user's pick resolves next turn. Runs after the
  continue-reading and discussion-mode short-circuits.
- `resetConversation` also clears the offer history (see §6).

## 4. Maps: distance, direction, and route progress

- **New `distance_to` tool** (streetzim-gated, declared): geocode →
  straight-line distance + compass direction + honest walking-time estimate
  (×1.3 detour at 5 km/h). Packages the geocode+trig chain the model can't
  run itself.
- **New `GeoMath`** (MCPZimKit): haversine, initial bearing, 8-wind compass
  words ("north-east") — prose/TTS-friendly.
- **Bearings everywhere**: `near_places` results carry `direction` from the
  query origin; `what_is_here` carries `direction` to the nearest place. The
  app can finally say "200 m north-east of you".
- **`route_status` un-broken for chat**: `setActiveRoute` was only called
  from the Siri App Intent path, so routes planned by typing/voice left
  `activeRoute == nil` and "how much longer?" errored.
  `updateFocusAfterTool` now persists chat-planned routes into
  `ZimfoContext` (`persistActiveRoute`, mirroring the intent path's
  construction).

## 5. Long sessions no longer grow until OOM

The full transcript was re-fed every turn; long walks hit the ~700 MB
headroom guard and aborted with "reset the conversation". `runGenerationLoop`
now windows history to the last 10 user exchanges, trimming in CHUNKS down
to 6 (not a per-turn slide) so the prompt prefix — and the KV-cache LCP
match — stays stable for several turns between trims. Older subjects survive
in the discourse state (`focus`), which lives outside the transcript.

## 6. Offers stop repeating

`appendThreadOfferIfUseful` now remembers the last ~12 offered thread keys
and won't re-offer them; a fresh offer or silence both beat nagging.

## Verification

- `swift test`: 303 green post-rebase, including the new
  `ConversationContinuationTests` (distance/direction/walk follow-up
  routing, kind-aware rebinding, no-place fallback, GeoMath
  cardinal/intercardinal/bucket edges, haversine sanity, drill-in
  `related[]` + `what_is_here` sibling thread extraction).
- Pre-rebase (on the old branch): macOS eval CLI + iOS device builds green;
  MCPZimEvalCLI matrix showed no regression (failures = the six
  pre-existing chain/wrong-tool scenarios). Re-verify post-rebase via the
  eval CLI before shipping.

## Deferred (next round)

- Real turn maneuvers (left/right/onto from polyline bearings) — turn-by-turn
  is still "road for X km" segments.
- Walking/cycling profiles — the v4/v5 graph carries foot/bicycle/oneway
  access bits but `SZRGGraph` skip-parses them; Walk/Bike pills still re-route
  only the JS viewer, not the tool result.
- `along_route` corridor search ("coffee on the way?").
- Model-phrased thread offers (per-turn `=== Threads ===` preamble block) —
  the deterministic append stands.
- Python MCP server parity (no composites / proximity tools server-side).
