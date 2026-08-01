# Performance fixes applied — 2026-07-31

The quick-win tier of `PERFORMANCE_REVIEW.md` (repo root), applied on
`bonsai-mlx-compare` in commit `352a4599`. 16 findings, 15 files.
Verification: `swift test` 437 green · `pytest` 41 green · Mac app,
MCPZimEvalCLI, and iOS device builds all succeed · deployed to the
development phone 2026-07-31 and confirmed alive in a live voice session.

## What changed and why it matters

### KV-prefix stability (Theme A — the latency headline)
The shipping hybrid models (LFM2.5, Bonsai) cannot partially truncate
recurrent state: ANY byte divergence in the rebuilt prompt forces a full
4–15 s re-prefill instead of the designed ~23-token append.

| Fix | Finding | Mechanism |
|---|---|---|
| Preamble GPS frozen per conversation, `%.3f` (~110 m), refresh only past 300 m | A1 | `ChatSession.preambleLocationSnapshot`; tool args keep precise live coords via `substituteCurrentLocation` |
| CoreLocation `distanceFilter = 25` | H2 | delegate fires on movement, not GPS jitter |
| `ChatMessage.rawAssistantText` — prompts rebuild from the model's exact emission | A2 | captured at the reasoning scrub (`recordRawEmissionIfScrubbed`), thread-offer append, and disambiguation appendix |
| Token-budget trim cuts to a 75 % watermark in one step | A3 | window start stays stable for several turns between trims |

### Latency / configuration
- **D7**: pre-turn GPS wait is skipped when authorization is denied or a
  previous wait already timed out this session (was +4 s TTFT per
  navigational turn with no fix obtainable).
- **H1**: the Bonsai-27B force-select migration is now `#if DEBUG`.
  Release fresh installs default to **LFM2.5** — previously every
  TestFlight/App Store first launch force-selected a ~5.5 GB-peak model
  at the 6 GB jetsam line.
- **E7**: `SemanticReranker.embedText` memo — drift-thread labels no
  longer pay a transformer forward pass per turn.

### UI hot loops
- **E3**: `isLatestAssistant` passed into `MessageRow` as a stored `let`;
  route/places trace classification memoized by trace id. Streaming no
  longer re-renders every row (and re-parses map JSON) at 10 Hz.
- **E4**: `displayText` regexes compiled once (`static let`); one
  pipeline run per body pass.
- **E5**: five scroll watchers collapsed to one signal; non-animated
  scroll while generating.
- **E8**: `HeroMediaView` resolves media in `.task(id:)` off-main,
  scanning only the first 64 KB.
- **D3**: `zim://` scheme handler reads on a utility queue (map tiles /
  article sheets no longer block the main thread mid-generation);
  stopped-task race handled.
- **D6**: log appends are `queue.async`; per-request GET logging demoted
  to a 1-in-25 counter.

### Engine (routing/geospatial)
- **F2**: A* stale-pop check via stored `g` — no haversine per pop.
- **F3**: heuristic ceiling from the graph's true max edge speed
  (monolithic only — SZRC cells carry no speed data without defeating
  lazy loading; spatial keeps its 80 km/h constant).
- **F6**: category-manifest cache actually caches now (was dead code).
- **F7**: `Int64` deltas remove the antimeridian `Int32` trap.
- **B8**: search fan-out early-exits across readers once `limit` fills.

### Python MCP server
- **PY2**: `nearest_node` equirectangular comparison — no trig in loop.
- **PY6**: true speed ceiling (admissible on >100 km/h builds) +
  stale-pop guard.
- **PY7**: polylines opt-in (`include_polyline`, default off) — route
  responses shed 100 KB–1 MB of LLM context per call.
- **PY8**: 32-entry article LRU + per-archive Xapian `Searcher` reuse.
- **PY10**: geocoder chunk cache LRU-capped at 64.

## How to verify the headline fix in the field
Walk around with the app and hold a multi-turn conversation. In the
debug log, the per-turn `[Perf]` rows should keep `reused=` high
(hundreds of tokens) on every follow-up. Before this pass, any movement
between turns collapsed `reused=0` with a multi-second `prefill=`.

## Deferred (structural tier — see PERFORMANCE_REVIEW.md §Suggested order)
Article/section LRU cache + lead-only fast path (B1/B2/B5/D4),
single-pass HTML stripper (B3), compact place records (C1/C2/F5),
`nonisolated` read+parse and per-cell spatial expansion (D1/D2),
incremental tool-call scan + MLX streaming detokenizer (E1/E2), llama
context free on backgrounding (C4), map-reduce pre-routing (A4), and the
remaining Python mediums (PY3/PY4/PY5). Each is a self-contained
medium refactor; the review's cost estimates say B1/B2/B3 (content
pipeline) are the next-largest wins.
