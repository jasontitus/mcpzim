# Performance review — 2026-08-01

A deep static performance review of the first-party code: the `mcpzim`
Python MCP server, the `MCPZimKit` Swift engine, and the iOS/macOS app
(`ios/MCPZimChat`). Five areas were reviewed in parallel by independent
passes (Python server; ZIM/content pipeline; routing + geospatial;
conversation/intent layer; chat loop + inference providers; UI/WebView/voice),
plus a cross-cutting pass over build config and test coverage. Every finding
below was verified against the source at the cited `file:line`; cost numbers
are back-of-envelope estimates from code inspection, not measurements — see
[Measurement gaps](#measurement-gaps).

Vendored packages (`ios/LocalPackages/*`) are out of scope except where
first-party code calls into them. `tools/` is offline eval tooling, not
runtime code, and is not reviewed.

**The short version.** The macro-architecture is in good shape: the fast-path
intent router skips whole generations, tool outputs are aggressively trimmed
for the model, streaming UI updates are coalesced to 10 Hz, heavy loads are
memoized and prewarmed, and the KV-prefix-reuse design is genuinely careful.
The problems are concentrated in four places:

1. **Silent KV-prefix invalidation** — several code paths change prompt bytes
   *before or inside* the conversation history, which on the shipping hybrid
   LFM2.5 (no partial KV truncation possible) converts the advertised
   ~23-token follow-up prefill into a **full 4–15 s re-prefill**. The worst
   offender: live GPS interpolated into the system preamble at ~1 m
   resolution, so a *walking user* — the headline use case — likely pays a
   full re-prefill on every turn.
2. **Full-article parsing to produce tiny outputs** — both the Python and
   Swift content pipelines decompress and strip entire 100 KB–1 MB articles
   (with ~30 string passes and per-call regex compilation) to emit a 220-char
   snippet or an 800-char lead, N+1 style per search hit, with no
   article-level cache. Search turns plausibly burn 1–6 s of CPU that could
   be tens of ms.
3. **Boxed data representations with unbounded caches** — place records held
   as `[[String: Any]]` JSON dictionaries (~300–700 B/record, cached forever;
   ~150–350 MB after one city-wide scan) and, in Python, street graphs decoded
   into boxed lists (~1.5–2 GB at state scale).
4. **Main-thread and actor bottlenecks** — synchronous ZIM decompression on
   the main thread (WebView scheme handler, search-hit enrichment), a single
   service actor that serializes the "parallel" excerpt fan-outs, and an
   actor hop + array allocation per node expansion in spatial A*.

One release-config flag that isn't strictly perf but swamps every perf number
in this doc: the Bonsai-27B selection migration force-selects a ~5.5 GB-peak
model for **every** iOS install ([H1](#h1)).

---

## Findings index

Severity is impact-weighted for the real workload (on-device tool-loop chat on
iPhone-class hardware; multi-GB ZIMs; city-scale street graphs). "Conf" is
confidence that the finding is real and the fix is a win.

| ID | Sev | Area | Finding | Conf |
|----|-----|------|---------|------|
| [A1](#a1) | **High** | KV reuse | GPS at `%.5f` in system preamble invalidates the whole prefix while moving | High |
| [A2](#a2) | **High** | KV reuse | Post-hoc mutation of final assistant reply breaks byte-exact rebuild | High |
| [B1](#b1) | **High** | Content | `search` does N+1 full-article parses for 220-char snippets (Swift + Python) | High |
| [B2](#b2) | **High** | Content | No article/section cache; composite tools re-read + re-parse 2–4× per turn | High |
| [B3](#b3) | **High** | Content | `stripHTML`: ~30 full-text passes, ~15 regex compiles per call, per section | High |
| [B4](#b4) | **High** | Content | `ArticleSections.parse` O(M·n) grapheme index arithmetic | High |
| [C1](#c1) | **High** | Memory | Place records as `[[String: Any]]`, cached unbounded (~150–350 MB) + slow scans | High |
| [D2](#d2) | **High** | Concurrency | Spatial A*: actor hop + fresh array per node expansion | High |
| [D3](#d3) | **High** | Concurrency | `zim://` WebView handler: sync ZIM decompress on the main thread | High |
| [E3](#e3) | **High** | UI hot loop | Every assistant row invalidated at 10 Hz; tool-trace JSON re-parsed each time | High |
| [H1](#h1) | **High** | Config | Bonsai 27B (~5.5 GB peak) force-selected for every iOS install | High |
| [PY1](#py1) | **High** | Python | Search snippets BeautifulSoup-parse the full article per hit | High |
| [PY2](#py2) | **High** | Python | `nearest_node`: O(N) pure-Python haversine scan, ×2 per route | High |
| [PY3](#py3) | **High** | Python | `Graph.parse` eagerly decodes all polylines into boxed objects | High |
| [D1](#d1) | Med-High | Concurrency | `DefaultZimService` actor serializes the "parallel" excerpt fan-outs | High |
| [D4](#d4) | Med-High | Concurrency | `enrichSearchHits`: sync read + full strip of 3 articles on the MainActor | High |
| [D7](#d7) | Med-High | Latency | Up to +4 s TTFT waiting for a GPS fix that may never come | High |
| [E1](#e1) | Med-High | Token loop | Tool-call detection re-scans the whole buffer per chunk; Gemma3 bare-fence escalation | High |
| [F1](#f1) | Med-High | Algorithm | Swift nearest-node snapping: linear haversine scan, ×2 per route | High |
| [A3](#a3) | Medium | KV reuse | Token-budget trim slides one exchange per turn → permanent prefix churn | High |
| [A4](#a4) | Medium | KV reuse | Map-reduce discards first-pass answer and clobbers the seq-0 KV mirror | High |
| [C2](#c2) | Medium | Memory | Hot-prefix geocode leaves re-decompressed + re-parsed on every query | High |
| [C3](#c3) | Medium | Memory | Graph load: blob + parsed arrays co-resident; full SHA pass; no eviction | High |
| [C4](#c4) | Medium | Memory | Backgrounding frees MLX cache but not the llama.cpp context | High |
| [D5](#d5) | Medium | Concurrency | Kokoro TTS stack rebuilt on the main actor on every voice-sheet open | High |
| [D8](#d8) | Medium | Concurrency | MLX generation not cancelled on stream termination; races the KV mirror | High |
| [E2](#e2) | Medium | Token loop | Gemma4Provider re-detokenizes the entire output on every token | High |
| [E4](#e4) | Medium | UI hot loop | `displayText`: ~7 regex compiles per call, 2–3 calls per row per push | High |
| [E5](#e5) | Medium | UI hot loop | Five stacked `onChange` watchers → animated scroll storm at 10 Hz | High |
| [E6](#e6) | Medium | Per-turn | IntentRouter compiles ~15–25 regexes per `classify`; runs 2–3× per turn | High |
| [E7](#e7) | Medium | Per-turn | Drift-thread labels re-embedded (transformer forward pass) every turn | High |
| [E8](#e8) | Medium | UI | `HeroMediaView` reads + regex-scans a full article inside SwiftUI `body` | High |
| [F2](#f2) | Medium | Algorithm | A* recomputes the haversine heuristic on every pop for stale detection | High |
| [F3](#f3) | Medium | Algorithm | 100 km/h heuristic ceiling ≈ half-strength h on 50 km/h city graphs | High |
| [F4](#f4) | Medium | Algorithm | SZRG/SZCI parsers decode fixed-stride binary per-element / per-byte | High |
| [B5](#b5) | Medium | Content | `articleByTitle(section:"lead")` parses the whole article for ~800 chars | High |
| [B7](#b7) | Medium | Content | `compare_articles` relations probe: up to 10 candidates × 16 probes + suggests | High |
| [H2](#h2) | Medium | Battery | Continuous GPS for the app lifetime once authorized | High |
| [A5](#a5) | Low-Med | KV reuse | Split multi-byte UTF-8 token pieces become U+FFFD and break next-turn LCP | High |
| [B6](#b6) | Low-Med | Content | `WikiLinks.parse`: two full-document regex passes + eager anchor materialization | High |
| [B8](#b8) | Low-Med | Content | Search fan-out doesn't early-break across readers once `limit` is filled | High |
| [C7](#c7) | Low-Med | Memory | No `autoreleasepool` around per-article Foundation-heavy loops | Med |
| [D6](#d6) | Low-Med | Concurrency | `LogArchive` `queue.sync` file write on the caller's thread; 40-line map bursts | High |
| [PY4](#py4) | Medium | Python | Sync `def` tools block the asyncio event loop | Med-High |
| [PY5](#py5) | Medium | Python | `html.parser` backend + 30 sequential selector passes, no size cap | High |
| [PY6](#py6) | Medium | Python | A*: unvalidated heuristic ceiling; stale heap pops re-expanded | High |
| [PY7](#py7) | Medium | Python | Full unsimplified polyline (10k+ points) in every route tool response | High |
| — | Low | various | [Smaller findings](#low-severity-findings) (E9–E11, F5–F7, C5–C6, D9–D10, H3–H6, PY8–PY10) | — |

---

## Theme A — KV-prefix invalidation (the highest-leverage theme)

Context that makes these findings bite: the shipping LFM2.5 is a hybrid
(6 attention + 18 shortconv layers). llama.cpp cannot partially truncate the
recurrent state, so **any** prompt divergence at **any** position degrades to
a full seq-0 wipe + full re-prefill — self-documented at
`ios/MCPZimChat/Providers/LlamaCppProvider.swift:599-649` as "~4 s at a full
8k, ~15 s at 32k". The prefix-stability engineering elsewhere in the
codebase is excellent (see [Done well](#done-well)); these are the leaks.

<a id="a1"></a>
### A1 (High) — Live GPS at ~1 m resolution in the system preamble

- **Where:** `ios/MCPZimChat/Chat/ChatSession.swift:864-869`
  (`locationLineText`, `String(format: "%.5f", …)` ≈ 1.1 m resolution,
  interpolated ~8× into the preamble), fed per-turn at `:2140`;
  `currentLocation` updated on every CoreLocation callback (`:1386-1398`);
  `AppIntents/LocationFetcher.swift:46` sets `kCLLocationAccuracyHundredMeters`
  but **no `distanceFilter`**; LCP reuse in `LlamaCppProvider.swift:616-649`
  and `Gemma4Provider.swift:564`.
- **What happens:** the preamble is the first turn of the prompt (system +
  tools fold into the first user turn), so a coordinate change between turns
  diverges the token stream near position ~0. While the user is moving —
  walking or driving, the app's headline use case — every CL fix (~1 Hz)
  produces new coordinates at far finer resolution than the requested
  accuracy, so essentially **every turn while moving pays a full re-prefill
  (~2–15 s added TTFT + prefill energy)** instead of the advertised ~23-token
  append. The in-code defense (`ChatSession.swift:778-785`, "location last in
  the preamble") only covers the prewarm→turn-1 transition, not turn N→N+1.
  Two independent review passes converged on this finding.
- **Fix (cheap → structural):**
  1. Quantize the preamble coordinates to `%.3f` (~110 m — consistent with
     the accuracy actually requested).
  2. Snapshot the location line per conversation and refresh only past a
     movement threshold (the 600 m `seedReseedMeters` hysteresis at
     `ChatSession.swift:327` is the existing pattern).
  3. Best: move the volatile coordinate out of the prefix entirely — inject
     it into the *latest* user turn (append position). Tool dispatch already
     substitutes precise coordinates separately (`substituteCurrentLocation`),
     so the preamble never needed meter precision.

<a id="a2"></a>
### A2 (High) — Post-hoc mutation of the final assistant reply

- **Where:** `ChatSession.swift:4281-4306` (`appendThreadOfferIfUseful` trims
  the reply and appends an offer line the model never generated),
  `:3925-3927` (disambiguation appendix `messages[idx].text +=`),
  `:2918-2936` (stores display-scrubbed text as the transcript source),
  `:2837-2839` (map-reduce replaces the reply), `:2668-2680` (forced-summary
  instruction turn fed to the model but never recorded). Next-turn rebuild
  reads `msg.text` at `:2188-2190`.
- **What happens:** the KV mirror holds the tokens the model actually
  generated, but the next prompt is rebuilt from the mutated `msg.text`. A
  whitespace trim, an appended offer line, or a replaced reply changes bytes
  at the last-assistant-reply position → failed partial truncation on the
  hybrid → **full re-prefill next turn**. The grounded-discussion path
  already does this correctly by storing the raw emission (`:3538-3541`), so
  the invariant is known — the final reply is the one leg that leaks.
- **Fix:** store the raw emission on `ChatMessage` (e.g. `rawAssistantText`)
  and rebuild prompts from it, keeping `text` for display; deliver the thread
  offer via the existing `suggestions` chips instead of mutating `text`;
  record the forced-summary instruction as a round-trip.

<a id="a3"></a>
### A3 (Medium) — Token-budget trimming slides one exchange per turn

- **Where:** `ChatSession.swift:2246-2253` (drop one oldest exchange per
  iteration), `:2312-2340` (each iteration re-renders the ~100 KB prompt and
  re-tokenizes ~20–30k tokens on the MainActor via
  `llama.promptTokenCount`, `LlamaCppProvider.swift:495-500`).
- **What happens:** the exchange-count window correctly trims in chunks
  (10→6) to preserve the prefix between trims, but the token-budget path
  trims minimally. A long article-heavy session that reaches the budget then
  shifts the window start on **every** turn → LCP collapses to the system
  turn → full ~30k-token re-prefill **every turn, forever**. The repeated
  render+tokenize per trim iteration also duplicates the tokenize the
  provider does again at generate time.
- **Fix:** trim to a lower watermark (70–80 % of budget) in one step,
  mirroring the count-based path; move `promptTokenCount` off the MainActor
  or return the tokens for reuse by `generate`.

<a id="a4"></a>
### A4 (Medium) — Map-reduce discards the first-pass answer and clobbers seq-0

- **Where:** `ChatSession.swift:2088-2095`, `:2745-2864`; mirror interaction
  at `LlamaCppProvider.swift:662, 705`; single sequence (`n_seq_max=1`) at
  `:456-471, 674`.
- **What happens:** an explanatory turn runs the full tool loop + a first-pass
  reply (discarded — self-acknowledged "yes, that's a wasted generation"),
  then one map generation per section (≥2 × 1–2k-token prefills) + a reduce —
  roughly 2–3× the turn's latency and energy. Every side-family prompt also
  repoints the KV mirror, so returning to the main transcript is a guaranteed
  cold prefill.
- **Fix:** route straight to map-reduce when `complexity == .explanatory` is
  known pre-generation (it is — the router computes it), or skip map-reduce
  when the first pass already cited ≥2 sections. Longer-term: `n_seq_max=2`
  and run side generations on seq 1 (needs a probe for hybrid-model state
  interaction).

<a id="a5"></a>
### A5 (Low-Med) — Split multi-byte UTF-8 pieces become U+FFFD

- **Where:** `LlamaCppProvider.swift:752-772`. The comment claims
  `String(cString:)` "will round-trip them on the next chunk"; it does not —
  each piece is decoded independently, so a code point split across two BPE
  tokens (diacritics; "Žemaičių" appears in this repo's own test data)
  becomes replacement characters in the recorded emission.
- **What happens:** garbled glyphs in UI/TTS, and re-tokenizing "�" next turn
  diverges from `cachedTokens` → full re-prefill on the hybrid. English-only
  replies are unaffected.
- **Fix:** buffer raw bytes and emit only complete UTF-8 sequences (standard
  llama.swiftui pattern); decode via `withUnsafeBufferPointer` instead of
  `.prefix().map{}` per token.

---

## Theme B — Content pipeline: full-article work for tiny outputs

The same architectural gap appears in both implementations: nothing between
libzim and the tool layer caches a parsed article, and the HTML stripper is
run far more often, over far more bytes, than the output requires.

<a id="b1"></a>
### B1 (High) — `search` N+1: full read + full parse per hit for a snippet

- **Where:** `swift/Sources/MCPZimKit/ZimService.swift:360-371`
  (`leadSnippet`: read entire article → `Data`→`String` → parse **all**
  sections → keep `.first`), called from the hit loops at `:253, 269`;
  amplified by `MCPToolAdapter.swift:492-516` (reranker over-fetch
  `fetchLimit = max(limit*2, 20)`, plus a kind-fallback that can re-run the
  whole search; the dedupe set is per-call, so overlapping hits re-parse).
- **Cost:** up to 20 full read+parse cycles per `search` call at 50–300 ms
  each (see B3/B4 for the multiplier) ≈ **1–6 s CPU and 10–30 MB transient
  allocations per search turn**, serialized on the service actor (D1) —
  including snippets for the 10 over-fetch hits the reranker will discard.
- **Fix:** lead-only extraction (scan raw UTF-8 for the first `<h2`/`<h3`,
  strip only that prefix, cap at a few KB); strip lazily *after*
  rerank/limit; small per-`(zim, path)` snippet LRU so the title pass, FTS
  pass, and kind-fallback never parse the same lead twice.

<a id="b2"></a>
### B2 (High) — No article/section cache; composite tools re-parse 2–4×

- **Where:**
  - `ArticleHeuristics.swift:45-53` — `sectionsByTitle` fetches + parses the
    article **twice** (the comment even predicts it: "the obvious thing to
    collapse").
  - `MCPToolAdapter.swift:964-1025` — `article_overview` adds a **third**
    read (`relatedLinks`, 2 regex passes over full HTML) and a **fourth**
    (`disambiguationAlternates`, hatnote regex over full HTML).
  - `MCPToolAdapter.swift:1295-1357` — `narrate_article` paging re-runs
    `sectionsByTitle` (2 reads + 2 parses) per "continue" to emit one
    ~700-char chunk; a 10-page narration ≈ 20 full parses of the same
    article.
  - `ZimService.swift:394-409` + `MCPToolAdapter.swift:554-563` —
    `get_article_section` parses the whole article for one section, then
    re-reads it again for related links.
- **Cost:** ~200–700 ms per composite-tool turn where one read + one parse
  would do; battery proportional.
- **Fix:** a tiny LRU actor cache keyed `(zim, path)` holding the last 2–4
  parsed `[ArticleSection]` + raw HTML collapses every case above; make
  `articleByTitle` return the sections it already parsed
  (`ZimService.swift:504`); cache sections across narration turns.

<a id="b3"></a>
### B3 (High) — `stripHTML`: ~30 passes, ~15 regex compiles, per section

- **Where:** `ArticleSections.swift:150-202` (+ helpers at 206-320);
  same pattern in `ArticleHeuristics.swift:184-204` (`stripCitations`).
- **What happens:** per call: 9 literal block-break `replacingOccurrences`
  passes + 1 tag-strip regex + 10 literal entity passes + ~13 helper regex
  passes — each allocating a fresh full-size String — and **~15 regex
  compiles**, because `replacingOccurrences(options: .regularExpression)`
  compiles on every call and there is not a single cached
  `NSRegularExpression` in the package (verified). `parse` calls `stripHTML`
  once per section body *and* once per heading title: a 40-heading article ≈
  **1,200 regex compilations and ~30 article-sized scans (~15 MB
  reallocated) per parse**. This is the inner multiplier for B1/B2/B5.
- **Fix:** hoist all patterns into `static let NSRegularExpression`s; replace
  the block-break + tag-strip + entity chain with one single-pass UTF-8
  state-machine scanner (~25 passes → 1); gate `stripSpeechArtifacts` on a
  cheap trigger-byte scan.

<a id="b4"></a>
### B4 (High) — O(M·n) grapheme index arithmetic in `ArticleSections.parse`

- **Where:** `ArticleSections.swift:95-96, 103, 110-113` —
  `String.distance(from: startIndex, …)` / `index(offsetBy:)` per marker,
  each O(position) grapheme-breaking from the start of the article.
- **Cost:** ~40 headings × 500 K chars ≈ tens of millions of grapheme steps ≈
  **tens to ~200 ms per article**, pure overhead: the code converts
  `Range<String.Index>` → `Int` → back to `String.Index`.
- **Fix:** store the `String.Index` range directly in `Marker` and slice —
  zero conversions (or run the parse over UTF-16/NSRange, which the regex
  matches already are).

<a id="b5"></a>
### B5 (Medium) — Lead-only requests parse the entire article

- **Where:** `ZimService.swift:502-513`; all excerpt-enrichment call sites
  pass `section: "lead"` (`MCPToolAdapter.swift:1476-1478, 1669-1671,
  1844-1846`).
- **Cost:** `fetchWikiExcerpts` (cap 10) and `buildNearbyStories` each
  trigger 8–20 full parses to keep ~800 chars each — 10–50× the necessary
  string work; a `near_places` turn with 10 wiki-tagged pins ≈ **0.5–3 s of
  serialized CPU** (compounds with D1).
- **Fix:** lead fast path — find the first `<h2`/`<h3` in the UTF-8 bytes and
  strip only that prefix when the requested section is lead/nil.

<a id="b6"></a>
### B6 (Low-Med) — `WikiLinks.parse` full-document double regex

- **Where:** `ConversationThreads.swift:544-615, 644-654`; runs on article
  fetches via `relatedLinks` (`MCPToolAdapter.swift:1022`).
- **Cost:** a `.dotMatchesLineSeparators` regex over the whole document plus
  a full prose copy, then eager `NSTextCheckingResult` materialization for
  *every* anchor (500–2,000 on large pages) although the loop stops at 8–20
  accepted links; `decodeAndStrip` (regex + 6 literal replaces) runs before
  the cheap `isArticleLink` filter. Several–tens of ms + MB-scale transient
  allocations per article fetch.
- **Fix:** `enumerateMatches` with early `stop`; filter on `href` before
  stripping; static regexes; lazy `<p>`-block scan.

<a id="b7"></a>
### B7 (Medium) — `compare_articles` relations probe fan-out

- **Where:** `MCPToolAdapter.swift:1116-1119, 1239-1287`; candidates from
  `ArticleHeuristics.swift:764-787`.
- **Cost:** for the common no-relations-article case, up to 10 candidate
  titles × (8 direct-path reads + a Xapian suggest per Wikipedia reader)
  before the real fetch — tens to hundreds of ms of pure miss cost, and a
  suggester false-positive pays a full double parse (B2).
- **Fix:** existence-check with direct-path reads only (no suggester);
  heuristic-gate the "Foreign relations of X" candidates; short-circuit
  after the first suggest miss.

<a id="b8"></a>
### B8 (Low-Med) — Search fan-out lacks early exit across readers

- **Where:** `ZimService.swift:244-276`.
- **Cost:** with 3 ZIMs loaded and `limit` already filled by reader 1,
  readers 2–3 still pay a Xapian suggest each (limit 20), and natural-language
  queries add up to 3–4 FTS variants per reader — worst case ~12 index
  queries per `search()`, ×2 with the kind-fallback, at ~5–50 ms each
  on-device.
- **Fix:** `if results.count >= limit { break }` at the top of the reader
  loop and before each `leadSnippet`; skip FTS variants whose keyword core
  duplicates one already searched.

Also in this theme: discussion-mode section ranking re-embeds and
re-lowercases every pinned section on each follow-up turn
(`ArticleHeuristics.swift:456-464, 584-596, 691-701` — ~50–200 ms/turn;
cache per-section embeddings + lowercased bodies alongside the pinned
sections).

---

## Theme C — Memory: representations and unbounded caches

<a id="c1"></a>
### C1 (High) — Place records as `[[String: Any]]`, cached unbounded

- **Where:** `ZimService.swift:127` (cache decl, no eviction), `:864-872`
  (full scan caches every prefix), `:1593-1612` (`loadChunk` caches by
  default), `:927-1009` (`scanRecords`: 6–10 `as?` casts + NSNumber bridging
  per record, full haversine before any cheap reject);
  `Geocoder.swift:119-153` consumes the same dicts. Two independent passes
  converged on this finding.
- **Cost:** ~300–700 B per NS-bridged record vs ~100 B compact; the generic
  scan is allowed `maxFullScanRecords = 500_000` (`:1229`) and permanently
  caches every chunk it loads → **~150–350 MB retained after one city-wide
  "what's near me"**, in a process that also holds a 3.6 GB model. Rescans
  cost **~150–500 ms of dynamic-cast CPU** before any filtering. The in-code
  war story ("453 chunks loaded, 5.4 GB RSS, jetsam", `:846-853`) shows the
  cap fixed the crash but not the representation.
- **Fix:** decode each chunk once into a compact struct array (or SoA:
  `[Float]` lat/lon, `[UInt8]` kind, interned names) — 5–7× smaller,
  cast-free scans; bounding-box pre-filter before haversine (rejects >95 %
  with two compares); byte-budgeted LRU on the chunk cache. The geocoder's
  `rank` gets faster for free.

<a id="c2"></a>
### C2 (Medium) — Hot-prefix geocode leaves re-parsed on every query

- **Where:** `ZimService.swift:672-687` (`cache: leaves.count == 1` — a
  deliberate RAM guard, `:1603-1605`); per-turn call chains issue 2–4
  overlapping geocodes (`:1341-1354`; `MCPToolAdapter.swift:621-628`,
  `:1411-1416`).
- **Cost:** fan-out prefixes are by construction the hottest names; every
  geocode re-runs cluster decompress + JSON parse over dozens of multi-MB
  leaves — plausibly hundreds of ms per query, repeated within a single turn.
- **Fix:** falls out of C1 (compact leaves are cheap enough to LRU-cache);
  or memoize `(query, zim) → [Place]` for the last few geocodes.

<a id="c3"></a>
### C3 (Medium; High at country scale) — Graph-load co-residency + SHA pass

- **Where:** `ZimService.swift:1438-1459`; `SZRGChunked.swift:105-131`
  (reassembles up to 4 GB contiguous `Data`, then a full `SHA256.hash`
  second pass); the "+2 GB on graph load" spike is self-documented at
  `SZRGGraph.swift:83-93`. Parsed graphs live forever in `graphs[pair.name]`.
- **Fix:** incremental `SHA256.update` per chunk; spill reassembly to a temp
  file and parse from `Data(…, options: .alwaysMapped)` so the raw bytes are
  evictable page cache instead of dirty RSS.

<a id="c4"></a>
### C4 (Medium) — Backgrounding shrinks MLX but not llama.cpp

- **Where:** `ChatSession.swift:1415-1467` — the background/memory-warning
  handler calls `resetPromptCache()` only on `Gemma4Provider`; with the
  default llama.cpp models selected, backgrounding frees nothing: the
  pre-allocated n_ctx KV (~226 MB LFM @32k) + compute buffers stay resident
  on top of weights, raising suspended footprint → more background kills →
  every relaunch pays full model load + full first-turn prefill.
- **Fix:** on `didEnterBackground`, free the llama *context* (keep the
  model): `llama_free(ctx)` + clear `cachedTokens`, lazily re-init on next
  turn — one re-prefill you were probably paying after a long suspend anyway.

<a id="c7"></a>
### C7 (Low-Med) — No `autoreleasepool` around per-article loops

- **Where:** `ZimService.swift:244-276, 373-389`;
  `ArticleSections.swift:229-245, 264-293` (NSRegularExpression `matches`
  arrays, NSString substrings).
- **Cost:** a 20-article search job accumulates autoreleased Foundation
  temporaries until the pool drains — transient RSS spikes of tens of MB
  against a jetsam budget already carrying the model.
- **Fix:** `autoreleasepool { }` per article in the hit/excerpt loops; the
  B3 single-pass scanner removes most of the temporaries anyway.

Smaller: C5 — full `rawResult` JSON (50–500 KB per route) retained on every
trace for the conversation lifetime + unbounded `SemanticReranker.cache`
(`Message.swift:107-135`, `ChatSession.swift:2539-2540`,
`SemanticReranker.swift:32`; keep raw payloads only on the latest map-bearing
message, LRU-cap the reranker cache). C6 — `Array(repeating: [], count:
numGeoms)` pins ~8 MB of pointers for the no-geometry case
(`SZRGGraph.swift:254-256`; consumers already bounds-check).

---

## Theme D — Concurrency structure

<a id="d1"></a>
### D1 (Med-High) — One actor serializes all CPU-heavy content work

- **Where:** `ZimService.swift:123` (`DefaultZimService` is an actor; parse
  work at `:364, 396, 504` is actor-isolated) vs. the task groups at
  `MCPToolAdapter.swift:1123-1144, 1463-1506, 1837-1899` that fan out up to
  10 concurrent `articleByTitle` calls.
- **What happens:** every task-group child hops onto the same actor, so the
  "parallel" excerpt fetches run strictly serially — the groups buy only
  overhead — and a 100–300 ms parse (or a 1.2 s graph load, or a multi-second
  chunk scan) queues every unrelated tool call behind it. `ZimReader`
  implementations are documented thread-safe, so the serialization is
  self-inflicted.
- **Fix:** make read+parse `nonisolated` (readers are `Sendable`,
  `ArticleSections.parse` is pure); touch the actor only for cache
  lookups/stores. The existing task groups then get real 2–4× parallelism on
  P-cores.

<a id="d2"></a>
### D2 (High) — Spatial A*: actor hop + array allocation per pop

- **Where:** `Router.swift:261-262` (`try await graph.edgesOfNode(current)`
  inside the pop loop); `SZRGSpatial.swift:242-261` (fresh `[SpatialEdge]`
  built per call).
- **Cost:** every pop (budgets: 200k optimal, 400k greedy —
  `Router.swift:340, 345`) crosses the actor executor twice and heap-allocates
  a transient edge array even on warm cache hits ≈ **0.2–1.0 s of pure
  concurrency/allocator overhead** on a budget-sized search.
- **Fix:** return the immutable `Sendable` `SZRCCell` once per cell and
  expand synchronously against `cell.cellAdj`/`cell.edges` while the
  frontier stays in-cell — one hop per cell touch, not per node.

<a id="d3"></a>
### D3 (High) — `zim://` scheme handler: sync ZIM reads on the main thread

- **Where:** `ios/MCPZimChat/Views/ZimURLSchemeHandler.swift:32-107`;
  `LibzimReader.swift:69-71` is a straight blocking libzim call.
- **Cost:** `WKURLSchemeHandler.start` arrives on the main thread and nothing
  hops off: one map open issues the viewer HTML, MapLibre bundle, style,
  fonts, and ~dozens of glyph/tile requests (the code's own log comment
  counts 38), each a blocking cluster decompress — landing mid-generation
  when the main thread is busiest; pan/zoom adds per-tile stalls. Article
  sheets and hero images share the handler.
- **Fix:** dispatch lookup+read to a utility queue and marshal
  `didReceive/didFinish` back to the invoking thread; the handler is already
  `@unchecked Sendable` with no shared mutable state.

<a id="d4"></a>
### D4 (Med-High) — `enrichSearchHits` on the MainActor

- **Where:** `ChatSession.swift:2945-2971` (called at `:2531-2533`): up to 3
  full articles read (sync zstd decompress), UTF-8 decoded, and full
  `stripHTML`-stripped **on the main thread**, mid-turn, to keep 400 chars
  each.
- **Cost:** ~50–300 ms of UI freeze and serial turn latency per `search`
  call — the one hot-path exception to the otherwise correct off-main tool
  execution.
- **Fix:** move into the adapter/service actor or `Task.detached`; strip
  only the first ~4 KB; cache per `zim:path` (shares B2's cache).

<a id="d5"></a>
### D5 (Medium) — Kokoro TTS stack rebuilt on the main actor per sheet open

- **Where:** `VoiceChatView.swift:13, 37-47` (controller in sheet `@State`);
  `TTSService.swift:64-72, 168-195` (parses the full ~45 MB `voices.npz` —
  all 26 voices for the 1 used — plus KokoroTTS construction against ~312 MB
  of weights, synchronously, on the main actor).
- **Cost:** main-thread stall on every voice-sheet open; dismissing the sheet
  destroys the stack, so every session re-pays load latency + energy.
- **Fix:** build off-main with a "preparing voice…" state; cache the service
  at process level (invalidate on voice change); extract only the selected
  voice if the reader allows.

<a id="d6"></a>
### D6 (Low-Med) — Synchronous log writes on the caller's thread

- **Where:** `Common/LogArchive.swift:57-62` (`queue.sync { handle.write }`);
  `ChatSession.swift:256-277` (every `debug()`: interpolation +
  `MemoryStats` `task_info` syscall + `print` + os_log + the sync write,
  regardless of `showDebugPane`); `ZimURLSchemeHandler.swift:50` logs every
  resource request.
- **Cost:** ~30–60 lines per turn is bounded, but a map load emits ~40 lines
  → ~40 main-thread `write(2)` calls exactly while tiles decode (D3). A slow
  flash write stalls UI mid-turn.
- **Fix:** `queue.async` for the append (durability comes from the kernel
  holding the data, not from blocking the caller); demote the per-request
  `GET` log to a counter or gate it behind `showDebugPane`; keep `print()`
  DEBUG-only.

<a id="d7"></a>
### D7 (Med-High) — Up to +4 s TTFT waiting for a GPS fix

- **Where:** `ChatSession.swift:2124-2127` (`awaitLocationIfAny(maxWait: 4)`
  for `.navigational` **and** `.topical` turns), `:1054-1060` (100 ms poll
  loop).
- **Cost:** with location denied/restricted or no fix obtainable,
  `currentLocation` stays nil forever, so every "tell me about X"-class LLM
  turn stalls the full 4 s before prefill starts. Fast-path turns skip it,
  which masks it in simple tests.
- **Fix:** wait only for `.navigational`; consult authorization state and
  skip when denied; remember "no fix obtainable" after the first timeout.

<a id="d8"></a>
### D8 (Medium) — MLX generation not cancelled on stream termination

- **Where:** `Gemma4Provider.swift:497-503` (no `continuation.onTermination`),
  `:700-701, 724` (stop/tool markers hardcoded to Gemma-4's, so a Qwen-style
  `</tool_call>` never matches); `ModelProvider.swift:84-85`
  (`cancelGeneration()` default no-op, not overridden).
- **Cost:** on user Stop or a template-mismatched tool call, the producer
  task keeps decoding to `maxTokens` (~1024) — wasted GPU/battery — and its
  tail `cachedTokens.append` races the next `generate()`'s mirror assignment
  (guaranteed cache miss next turn). The llama provider handles all of this
  correctly.
- **Fix:** `continuation.onTermination { task.cancel() }` + a cancellation
  check in the token loop; source markers from `template`.

Smaller: D9 — the llama decode loop runs as an unprioritized `Task.detached`
(`LlamaCppProvider.swift:513`; use `.userInitiated` or a dedicated thread so
token throughput doesn't degrade first under thermal pressure). D10 — a
250 ms pre-prefill sleep fires on *every* turn after the first tool use in a
session, not just when a map WebView is actually mounting
(`ChatSession.swift:2354-2364`; gate on the previous message carrying a
route/places trace).

Python equivalent: [PY4](#py4).

---

## Theme E — Hot loops: per-token and per-UI-push waste

<a id="e1"></a>
### E1 (Med-High) — Tool-call detection re-scans the whole buffer per chunk

- **Where:** call site `ChatSession.swift:2398-2411, 4712-4720` (every
  chunk, unthrottled, on the MainActor); scans in
  `Gemma3Template.swift:209-221` (5 openers incl. the bare ``` ``` `` fence),
  `QwenChatMLTemplate.swift:158-161`, `ChatToolCallParser.swift:45-56`
  (3 fallback openers); `repairJSON` compiles 3 fresh regexes per attempt
  (`Gemma3Template.swift:351-369`).
- **Cost:** 4–8 `String.range(of:)` scans over the growing buffer per chunk
  → O(replyLen²); ~50–250 µs/chunk at the 2k-token tail. Worse: Gemma3's
  bare-fence opener matches any markdown fence in prose, after which **every
  subsequent token** pays body copy + JSON decode + 3 regex compiles + a
  second decode + 3 fallback scans (~0.3–1 ms/token for the rest of the
  reply).
- **Fix:** incremental scanning (track `searchStart = lastScanEnd −
  maxMarkerLen`); cheap per-chunk trigger (`<` or backtick present); require
  `{` after a fence before attempting JSON; remember failed fence positions;
  static regexes in `repairJSON`.

<a id="e2"></a>
### E2 (Medium) — Gemma4Provider re-detokenizes everything per token

- **Where:** `Gemma4Provider.swift:725-740` — `tokenizer.decode` over the
  full `tokenIDs` (with a fresh `.map { Int($0) }` array) on every token,
  plus O(n) `hasPrefix`/`contains` scans.
- **Cost:** ~131k cumulative token decodes for a 512-token reply ≈ 0.5–1.5 s
  extra CPU per long reply, quadratic in length. The vendored `MLXLMCommon`
  already ships a streaming detokenizer.
- **Fix:** incremental detokenizer; bounded-tail marker scan.

<a id="e3"></a>
### E3 (High) — Every assistant row invalidated at 10 Hz, re-parsing trace JSON

- **Where:** `ChatView.swift:343-344` (`MessageRow.body` reads
  `session.messages` to compute `isLatestAssistant`) and `:349-377`
  (`traceHasRoute`/`traceHasPlaces` JSON-parse the full `rawResult` inline in
  the ViewBuilder); `RouteWebView.swift:16-29`, `PlacesWebView.swift:23-36`.
- **Cost:** `messages` is one `@Observable` property rewritten at 10 Hz
  during streaming, so **every instantiated assistant row** — not just the
  streaming one — re-runs `body` 10×/s for the whole generation; each run
  re-parses route polylines (10–100 KB JSON) per map-bearing trace. Three
  map messages ≈ 30–60 full JSON parses/sec on the main thread, concurrent
  with inference, worsening with transcript length.
- **Fix:** compute `isLatestAssistant` once in the `ForEach` and pass it as a
  stored `let`; classify traces once at creation (`kind` enum on
  `ToolCallTrace`) or memoize by `trace.id`. Both changes are local.

<a id="e4"></a>
### E4 (Medium) — `displayText` regex pipeline recompiled per call

- **Where:** `ChatView.swift:527-565`; call sites at `:196, 387, 514`.
- **Cost:** ~10 full passes and ~7 ICU regex compiles per call, ×2–3 calls
  per row per push, × every instantiated row (via E3) ≈ thousands of regex
  compilations/sec on the main thread during streaming.
- **Fix:** `static let` compiled regexes (or one alternation); cache the
  stripped result keyed `(message.id, text.count)`; reuse the computed value
  between `displayed` and `bubble`.

<a id="e5"></a>
### E5 (Medium) — Animated scroll-to-bottom storm

- **Where:** `ChatView.swift:23-38, 99-103` — five stacked `onChange`
  watchers each calling `scrollToBottom`; the text watcher fires at 10 Hz
  with `withAnimation(.easeOut(0.15))`.
- **Cost:** ten interrupted animations/sec, each forcing layout down to the
  bottom anchor; multi-value events fire 2–3 redundant `scrollTo`s in one
  tick.
- **Fix:** single pin-to-bottom driver — non-animated `scrollTo` while
  `isGenerating`, or iOS 17 `.defaultScrollAnchor(.bottom)`.

<a id="e6"></a>
### E6 (Medium) — IntentRouter/ReferenceResolver per-call regex compilation

- **Where:** `IntentRouter.swift:827-853` (helpers compile fresh
  `NSRegularExpression`s; ~14 inline patterns across `classify`), regex-mode
  `replacingOccurrences` at `:809-818, 560, 323-325`;
  `ReferenceResolver.swift:527-533` (per-call regex) and `:336-341, 410-416,
  493-517` (40–70-element `Set` unions rebuilt per call). Classify/resolve
  run 2–3× per turn (`ChatSession.swift:2011, 2034, 2061`;
  `IntentRouter.swift:400, 610`), on the main actor.
- **Cost:** ~15–25 ICU compiles ≈ 1–2 ms per classify, doubled in discussion
  mode, before anything else happens on the turn — the voice-UI jank window.
- **Fix:** static compiled patterns (a lazy `[String: NSRegularExpression]`
  cache inside `match`/`matches` is a two-line change); hoist the merged
  sets; pass the already-computed `ResolvedReference` into `classify` instead
  of re-resolving.

<a id="e7"></a>
### E7 (Medium) — Drift-thread labels re-embedded every turn

- **Where:** `ChatSession.swift:4334-4339` calls
  `SemanticReranker.shared.embedText(t.label)` per open thread per turn;
  `SemanticReranker.swift:126-132` has no cache on `embedText` (unlike
  `rerank`'s hit cache); labels recur across turns by design
  (`ChatSession.swift:4122-4124`).
- **Cost:** `NLContextualEmbedding` is a real transformer forward pass
  (~5–50 ms per string) × up to 4 labels × every turn ≈ up to ~200 ms +
  battery, spent before suggestion chips appear.
- **Fix:** bounded memo `[String: [Float]]` in `embedText`, or store label
  vectors in the existing LRU-capped `EmbeddingIndex`.

<a id="e8"></a>
### E8 (Medium) — `HeroMediaView` reads a full article inside `body`

- **Where:** `HeroMediaView.swift:23, 93-96, 129-149` — sync ZIM read +
  UTF-8 decode + several regex passes (matching **all** `<img>` tags) over
  up to ~2 MB of HTML, in `body`, on the main actor, when the trace lands
  mid-generation; a second full read of an article the tool already fetched.
- **Fix:** resolve in `.task(id: trace.id)` into `@State`; cache by trace
  id; scan only the first ~64 KB (hero media is at the top).

Smaller: E9 — map views re-parse `rawResult` 3–5× per body pass via computed
properties (`PlacesWebView.swift:78…`, `RouteWebView.swift:70-109, 245-319`;
parse once per trace id). E10 — the voice poller re-sanitizes the full
accumulated reply every 150 ms (`VoiceChatController.swift:401-441`) and
`Gemma4ToolCallParser.impliedBodyEnd` re-walks the open tool-call body per
chunk (`Gemma4ToolCallParser.swift:63-96`) — both bounded, both fixable by
tracking an offset. E11 — tool schemas re-serialized per render and
`toolDeclarations` re-parsed per turn (`QwenChatMLTemplate.swift:383-417`,
`ChatSession.swift:4673-4706`; memoize on the registry identity — the bytes
are deterministic, so this is CPU waste only, not a prefix hazard).

---

## Theme F — Algorithms (routing + geospatial)

<a id="f1"></a>
### F1 (Med-High) — Nearest-node snapping: linear haversine scan, ×2 per route

- **Where:** `SZRGGraph.swift:364-372` (the comment already says "swap in a
  k-d tree"); called at `ZimService.swift:569-570`; spatial variant
  `Router.swift:193-209` scans 2N `Int32`s per endpoint.
- **Cost:** O(N) trig ≈ 30–60 ms per call at 500k nodes, twice per
  `planDrivingRoute`; the spatial variant's N can be millions. Two side
  notes: the spatial delta math is checked `Int32` arithmetic that **traps**
  on antimeridian-spanning data (lon diff > 214.7°), and the metric lacks
  `cos(lat)` scaling (accuracy bias at high latitude).
- **Fix:** equirectangular *squared* distance with hoisted `cos(lat)` (no
  trig in the loop, same argmin, ~5–10×), or reuse the already-present cell
  grid for ring-by-ring probing; deltas in `Int64`.

<a id="f2"></a>
### F2 (Medium) — A* pays a haversine per pop for stale-entry detection

- **Where:** `Router.swift:89-91` — stale check computed as
  `curG < current.f - heuristic(current.node) - 1e-9`, i.e. one full
  haversine (with `cos(goalLat)` re-derived inside) per pop; edge cost
  `dist * 3.6 / max(1, speed)` re-derived per relaxation (`:98-99`) and
  again in reconstruction (`:138-139`).
- **Cost:** ~30–60 ms of pure trig on a cross-city search that a stored
  scalar makes free.
- **Fix:** carry `g` in `QueueItem` and test `item.g > gScore[item.node]`
  (exact, no epsilon, no trig); optionally precompute per-edge
  `costSeconds: [Float]` (+4 MB at 1M edges).

<a id="f3"></a>
### F3 (Medium) — 100 km/h heuristic ceiling on 50 km/h graphs

- **Where:** `Router.swift:60-64` (monolithic, 100 km/h; the comment says it
  mirrors the JS viewer byte-for-byte), `:233` (spatial, 80 km/h).
- **Cost:** on a city graph whose true max speed is ~50 km/h the heuristic
  is half-strength, pushing A* toward Dijkstra — typically **2–4× more
  pops**, multiplying every per-pop cost above (D2, F2).
- **Fix:** one O(E) `max(edgeSpeedKmh)` at parse; use `min(100, maxSpeed)` —
  still admissible, identical routes, fewer expansions. Keep the constant
  behind a flag if JS byte-parity matters.

<a id="f4"></a>
### F4 (Medium) — Binary parsers decode per-element / per-byte

- **Where:** `SZRGSpatial.swift:366-369, 410-413` (names blob copied **one
  byte per iteration** — the index reaches ~150 MB on Japan-scale data per
  `ZimService.swift:1468`), `:459-490` (`SZRCCell.parse`, re-run on every
  cell-cache miss mid-route, 32-cell cache); `SZRGGraph.swift:151-196,
  398-427` (~8M appends + ~7M throwing 4-byte reads ≈ 100–250 ms of the
  observed ~1.2 s parse).
- **Fix:** these are fixed-stride little-endian sections —
  `[UInt8]/[UInt32](unsafeUninitializedCapacity:)` + `copyBytes`, then one
  vectorizable transform pass. `loadNodeShards`
  (`ZimService.swift:1508-1527`) already demonstrates the right pattern.

Smaller: F5 — top-K by full sort with per-hit interpolated string keys
(`ZimService.swift:885-899`, `Geocoder.swift:143-152`; bounded heap +
hashable struct keys). F6 — `loadCategoryManifest`'s cache is dead code
(`ZimService.swift:1014-1025` — the cached value is read and explicitly
discarded, and nothing ever stores under the key; every kind-filtered
`near_places` re-reads + re-parses the manifest). F7 — the antimeridian
`Int32` trap noted under F1 is a robustness bug worth a one-line `Int64` fix
independent of perf.

---

## Theme G — Python MCP server (`mcpzim/`)

The Python server shares the Swift engine's two big content-pipeline gaps
(B1/B3), plus graph-representation costs the Swift port already solved.

<a id="py1"></a>
### PY1 (High) — Search snippets parse the full article per hit

- **Where:** `mcpzim/content.py:257-274` (`_hit_for`: `bytes(item.content)`
  → full `html_to_text`), from `content.py:241, 251`; `server.py:90-94`
  searches all ZIMs sequentially.
- **Cost:** default 10 hits/ZIM × full decompress + BS4 parse + 30 selector
  passes to keep ~220 chars ≈ **1–3+ s per `search` call** — the most-used
  tool — all while holding `zim.lock` (`content.py:235-253`; contrast
  `fetch_article`, which correctly releases the lock before parsing).
- **Fix:** parse a bounded slice (first 64 KB) or a lightweight
  tag-stripper for snippets; build snippets after releasing the lock; reuse
  one `Searcher` per archive (`content.py:238, 248` re-opens the Xapian
  index per call).

<a id="py2"></a>
### PY2 (High) — `nearest_node`: O(N) pure-Python haversine, ×2 per route

- **Where:** `routing.py:171-181`, called at `:383-384` on every routing
  request (graph cached; this is not).
- **Cost:** ~1–2 µs/node in CPython → 0.2–0.4 s per endpoint at city scale,
  **2–8 s at state scale**, every request.
- **Fix:** grid-bucket index built once in `Graph.parse` (expected-O(1)
  lookups); or at minimum an equirectangular squared-distance comparison
  (no trig in the loop, ~5–10×).

<a id="py3"></a>
### PY3 (High) — `Graph.parse` decodes everything into boxed objects

- **Where:** `routing.py:74-160`; five per-index list comprehensions re-walk
  the unpacked edge tuple (`:99-108`); `_decode_geom` decodes **every**
  polyline eagerly (`:120`).
- **Cost:** 16 B/edge on disk → ~140–170 B boxed; geometry points → ~112 B
  tuples. State-scale ≈ **1.5–2 GB resident forever** (`RouterCache` never
  evicts) + >200 MB transient during parse + tens of seconds of first-request
  stall — almost all spent on polylines a route never touches.
- **Fix:** decode geometry lazily in `_reconstruct_route` (only route
  edges); store nodes/edges as `array.array`/`memoryview.cast` (stdlib,
  ~10× memory cut) — the Swift port's `decodeGeoms: false` + flat arrays is
  the proof of concept.

<a id="py4"></a>
### PY4 (Medium) — Sync `def` tools block the asyncio event loop

- **Where:** every `@mcp.tool()` in `server.py:61-227` is a sync function.
- **Cost:** FastMCP runs non-coroutine tools inline on the event loop:
  pipelined tool calls serialize fully, and during a 10–30 s first routing
  call the server can't answer MCP pings — client-timeout risk. The per-ZIM
  locks (`library.py:66`) are never contended in this model.
- **Fix:** `async def` + `anyio.to_thread.run_sync` for
  search/get_article/routing; keep trivial tools sync. (Verify against the
  pinned `mcp` SDK version.)

<a id="py5"></a>
### PY5 (Medium) — `html.parser` + 30 sequential selector passes

- **Where:** `content.py:100-137`; strip list `:47-75`.
- **Cost:** pure-Python parser is 5–10× slower than lxml; each of 30
  `soup.select` calls walks the whole tree, then `find_all(True)` walks it
  again — 0.5–3 s per large `get_article`, no input-size cap, full text
  returned wholesale (LLM context cost too).
- **Fix:** prefer `lxml` when importable (`beautifulsoup4` is already a
  dependency; `lxml` is absent from `pyproject.toml`); merge the selectors
  into one comma-joined `select` (single traversal); optional `max_chars`
  truncation with a flag.

<a id="py6"></a>
### PY6 (Medium) — A*: unvalidated ceiling; stale pops re-expanded

- **Where:** `routing.py:24, 101, 231-279` — speeds decode from a full byte
  (0–255) but the heuristic assumes ≤100 km/h (inadmissible if the builder
  emits 110/120/130 → suboptimal routes + re-expansion blowups; loose if the
  graph is slower → toward-Dijkstra); no stale-pop guard; per-relaxation
  cost arithmetic.
- **Fix:** `max(edge_speed_kmh)` at parse as the ceiling; carry `g` in the
  heap tuple and skip stale pops; precompute `edge_cost_s`.

<a id="py7"></a>
### PY7 (Medium) — Full polyline in every route tool response

- **Where:** `routing.py:222, 297-320`; returned at `server.py:160-163,
  209-216`.
- **Cost:** 100 KB–1 MB JSON per response — for an LLM consumer that is tens
  of thousands of wasted context tokens per routing call, the dominant
  end-to-end cost of the tool. (`turn_by_turn`/`roads` already carry the
  useful signal. The iOS app solves this exact problem with
  `trimForModel` — polyline → first/last/count.)
- **Fix:** omit by default behind `include_polyline`, or Douglas-Peucker to
  ~100 points, or encoded-polyline string.

Smaller: PY8 — no article/Searcher caching (`content.py:155-186, 238, 248`;
a 32-entry LRU is trivial). PY9 — `search_zim` holds the per-ZIM lock across
all snippet parsing and `RouterCache.graph_for` can double-parse
concurrently (`content.py:235-253`, `routing.py:352-363`) — latent until
PY4 is fixed. PY10 — `Geocoder._chunks` unbounded (`geocode.py:78-103`;
small LRU).

---

## Theme H — Product/config flags with perf impact

<a id="h1"></a>
### H1 (High — flag for confirmation) — Bonsai 27B force-selected on every iOS install

- **Where:** `ChatSession.swift:1318-1332` — the one-shot migration key
  `chat.didSelectBonsai27B1BitV2` sets Bonsai 27B for any install that
  hasn't run this build, inside `#if os(iOS)` (verified: **not**
  `#if DEBUG`), and the fallback is `?? bonsai27b_1bit`.
- **Why it matters:** the comment says "on the development phone", but the
  gate is platform-wide: every iOS install's first launch selects the
  ~5.5 GB-peak Bonsai over the 3.64 GB LFM2.5 — at or over the jetsam line
  on 6 GB phones before TTS/WebKit overhead, and a larger download. If this
  build ships as-is, it invalidates the README's headline memory numbers.
- **Fix:** gate on `#if DEBUG` or a device allowlist; fall back to
  `lfm25_ft`; pick per `DeviceProfile` tier.

<a id="h2"></a>
### H2 (Medium, battery) — Continuous GPS for the app lifetime

- **Where:** `LocationFetcher.swift:44-49, 93-96, 146-154, 212-222` —
  `startUpdatingLocation()` from launch, never stopped while authorized,
  including sessions that never touch a map. Mitigations are real
  (hundred-meter accuracy, `pausesLocationUpdatesAutomatically`, stops in
  background) but a moving user pays continuously, and each fix fans out to
  `currentLocation` → map-view body re-runs (E9) → and, via A1, potentially
  a full re-prefill.
- **Fix:** window updates to when a route/places/voice-nav context is
  active; otherwise stop after a fresh fix and re-arm on demand (`once()`
  already supports the wait path). Add a `distanceFilter` (e.g. 25–50 m) so
  delegate callbacks track movement, not jitter — this also directly
  mitigates A1.

Smaller: H3 — `TimelineView(.animation)` holds the display link at full
cadence (ProMotion) precisely during prefill (`ChatView.swift:858`;
`minimumInterval: 1/20`), and LibraryView's 1 Hz timeline ticks when idle
(`LibraryView.swift:24`). H4 — every Siri intent re-opens all ZIMs
(`ZimfoRunner.swift:19-66`; cache the runner per process). H5 — MLX pool
drain + double GPU sync per `generate` = up to 12 drains per tool-heavy turn
(`Gemma4Provider.swift:512-513, 823-824`; clear once per user turn — but
this is a documented, deliberate memory trade). H6 — AVSpeech fallback
serializes synthesis/playback per sentence (`TTSService.swift:54-57,
90-105`) and the Kokoro downloader delivers progress on the main queue
(`KokoroDownloader.swift:92, 115-131`).

---

<a id="done-well"></a>
## What's already done well

A fair review has to say this loudly: much of this codebase shows deliberate,
measured performance engineering, and several "usual suspects" this review
went looking for turned out to be already handled.

**KV/prompt economics.** `toolRoundTrips` persists exact emission bytes so
tool turns re-prefill only the new suffix; `.sortedKeys` JSON everywhere in
the render path (deterministic re-serialization); Qwen's hidden
`<think>` prefix re-inserted for byte-append-only re-renders; the
grounded-discussion prompt cache is the same LCP idea done right at a second
layer (append-only, raw emission stored, bounded appends, logged compaction);
`/no_think` injection and the compact Gemma3 tool block save hundreds of KV
tokens per turn; tool outputs are aggressively trimmed for the model
(polyline → first/last/count, capped lists, word-capped sections,
device-tiered `articleCapKB`). The LlamaCppProvider mirror discipline
(pessimistic wipe on throw, EOG exclusion, two-tier reuse with hybrid-safe
fallback, per-stage perf telemetry) is exactly right.

**Fast paths that skip whole generations.** The deterministic IntentRouter
dispatch (places/routing/continue-reading, did-you-mean, geocode-miss
short-circuit) removes entire 5–20 s generations from the most common turns,
and mid-conversation fast-path turns are structured so the next LLM turn is
still a KV append.

**Memory as a first-class concern.** Device tiering drives article caps,
reply caps, and MLX cache limits; `MemoryProbe` tracks jetsam-risk bands;
graph parse skips polyline decode (documented ~600 MB saving); the
place-scan paths have bbox short-circuits, a 500 K-record jetsam cap, and
chip/category index fast paths; `LibzimReader` caps libzim's cluster cache
at 64 MB specifically to evict the streetzim mega-cluster; live map
WebViews are restricted to the newest message with `dismantleUIView`
teardown; exactly one model is resident at a time.

**Hot-loop hygiene where it was known to matter.** Streaming UI pushes are
coalesced to 10 Hz in every consumer loop; transcript bubbles are plain
`Text` (no per-push markdown parse); GPS-tick map updates push a dot-only
JS update instead of reloading; route payloads are downsampled to ≤400
points before MapLibre; the voice loop tears down the mic during
thinking/speaking and runs single-pass RMS VAD; Kokoro TTS streams
sentence-by-sentence with buffer queueing.

**Engine data structures.** Real binary min-heap with lazy deletion;
CSR-style flat adjacency with node-indexed state (documented ~80 MB saving
over dictionaries); zero-copy `withUnsafeBytes` binary parsing (with a
comment recording the 100× memory regression it replaced); SZCI names kept
as bytes+offsets, cell keys packed into `Int64`; per-ZIM caches with
negative-result sentinels; prewarming moves the ~1.2 s graph parse off the
first query. On the Python side: correct double-checked `RouterCache`,
lazy imports, CSR adjacency, module-level regexes, `join`-based string
pipelines, and `fetch_article` releasing the ZIM lock before parsing.

---

<a id="measurement-gaps"></a>
## Measurement gaps

Every cost figure above is an estimate from code inspection. The repo has
extensive *model* benchmarking (`tools/llama-smoke`, `tools/llm-smoke`,
eval harnesses) but **zero engine benchmarks**: no timing for
`ArticleSections.parse`/`stripHTML` on a real article, `SZRGGraph.parse` or
A* on a real graph, end-to-end `search` latency, or the Python equivalents
(`tests/` are 4-node micro-graphs; no `measure`/XCTMetric anywhere in the
Swift tests). Two findings would have been caught by a single log line:
A1 (log the LCP length per turn — the provider already computes it) and
PY3 (log parse duration + RSS in `graph_for`).

Recommended before optimizing beyond the trivially safe fixes:

1. **Per-turn KV telemetry:** log `reused/prefilled` token counts per turn
   (both providers already know them). A walking-session field test of A1
   is one afternoon and would confirm the biggest finding in this review.
2. **Engine microbenchmarks:** a small `swift test` target (or
   `package-benchmark`) over one bundled article and one real `graph.bin`:
   `stripHTML`, `ArticleSections.parse`, `search`, `nearestNode`, `aStar`,
   `SZRCCell.parse`. Python: `pytest-benchmark` over the same shapes.
3. **Instruments passes** during a streamed reply with two map bubbles in
   the transcript (validates E3/E4/D3) and during a `near_places` full scan
   (validates C1).

---

## Suggested order of work

Quick wins first (small, local, high confidence), then structural.

| # | Change | Findings | Effort |
|---|--------|----------|--------|
| 1 | Quantize/freeze preamble GPS (+ `distanceFilter`) | A1, H2 | Small |
| 2 | Store raw assistant emission; offer via chips | A2 | Small |
| 3 | Watermark-based token trim | A3 | Small |
| 4 | Static regexes: stripHTML, IntentRouter, displayText, repairJSON | B3, E4, E6, E1 | Small |
| 5 | `String.Index` ranges in ArticleSections.parse | B4 | Small |
| 6 | Precompute trace `kind`; pass `isLatestAssistant` down | E3 | Small |
| 7 | Async scheme-handler reads; async log writes | D3, D6 | Small |
| 8 | A* stale-pop via stored `g`; true speed ceiling (Swift + Python) | F2, F3, PY6 | Small |
| 9 | Fix dead manifest cache; early-break search fan-out | F6, B8 | Small |
| 10 | Gate location wait on authorization; `.navigational` only | D7 | Small |
| 11 | DEBUG-gate the Bonsai migration | H1 | Small |
| 12 | Article/section LRU cache (service actor) + lead-only fast path | B1, B2, B5, D4 | Medium |
| 13 | Single-pass HTML stripper | B3 | Medium |
| 14 | Compact place-record decode + bbox pre-filter + LRU | C1, C2, F5 | Medium |
| 15 | `nonisolated` read+parse; per-cell spatial expansion | D1, D2 | Medium |
| 16 | Incremental tool-call scanning; streaming detokenizer (MLX) | E1, E2, E10 | Medium |
| 17 | llama context free on backgrounding | C4 | Medium |
| 18 | Python: snippet slice + lxml + thread offload; lazy geoms + `array` graph; polyline opt-in | PY1–PY7 | Medium |
| 19 | Grid index for nearest-node (Swift + Python) | F1, PY2 | Medium |
| 20 | Map-reduce routing decision pre-generation; seq-1 side prompts | A4 | Larger |

Items 1–3 alone should restore the "~23-token follow-up prefill" behavior
the architecture was built to deliver — likely the single largest
user-visible latency win available, and entirely in code the project
already understands deeply (the same invariant is implemented correctly in
`toolRoundTrips` and the grounded-discussion cache).
