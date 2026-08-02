# DS4 review — per-finding disposition ledger

Companion to `docs/PERFORMANCE_FIXES_2026-08-02.md` (narrative) and
`DS4_REVIEW_PERF.md` (the review itself, repo root). One row per distinct
finding; batch-25's test-anchored duplicates are folded into the primary
row. Applied in commits `a365c24` (code) + `f9207d1`/`c741670` (docs) on
`claude/review-bugs-performance-s61ch3`.

Legend: **Applied** — fixed as suggested (or better). **Partial** — the
safe subset applied, remainder deferred with reason. **Deferred** — real
finding, fix withheld deliberately (reason given). **Dismissed** — the
claim doesn't hold on inspection (evidence given).

Summary: **74 applied rows** (3 partial; a few rows cover multiple
flagged sites, e.g. the streaming-scan gate closes four findings at one
call site) · **16 deferred rows** (row 5 alone folds ~20 vendored-VLM
vision findings) · **9 dismissed rows**.

---

## Applied — Python server (`mcpzim/`)

| Finding | Sev | What was done |
|---|---|---|
| `routing.py:186` nearest_node O(n) scan | med | Lazily-built uniform grid (~16 nodes/cell), ring search with exact lower-bound cutoff; same argmin. Fuzz-verified vs brute force (500 queries, 0 mismatches; 0.076 ms vs 54 ms warm on 500k nodes) |
| `content.py:304` full BS4 parse per search hit | med | Snippets only for first 8 hits per `search_zim` call, parsed from a 96 KB HTML prefix (memoryview-sliced before copy) |
| `content.py:118` ~30 `select()` tree walks | low | One combined selector-group pass (pilcrow/edit anchors folded in); `decomposed` guard for nested matches |

## Applied — Swift engine (`swift/Sources/MCPZimKit/`)

| Finding | Sev | What was done |
|---|---|---|
| `ZimService:896/681` chunk-cache pinning (observed 5.4 GB jetsam) | high | LRU on `chunks` capped at one full-scan's records (`maxCachedChunkRecords`), covering `loadChunk` AND `loadCategoryChunk`. Chose the review's LRU alternative over `cache:false` to keep repeat-query speed |
| `IntentRouter:1231/1221` regex compile per call (~15–20/turn + per-sentence loops) | high | New `RegexCache` (locked, keyed by pattern+options); wired into `match`/`matches` |
| `SZRGSpatial:248` per-pop actor hop + `[SpatialEdge]` alloc in A* | med | New `SpatialGraph.cell(containingNode:)`; `Router.aStarSpatial` keeps a one-cell memo and walks the flat stride-5 edge array. `edgesOfNode` kept for tests/other callers |
| `SZRGSpatial:292` cell fetches serialised on the actor | med | Fetch+parse `Task.detached`; `inFlight` still dedups same-cell loads |
| `SZRGGraph:368` per-node haversine in nearestNode | med | Equirectangular squared-distance argmin (review's own smallest fix; k-d tree noted as follow-up) |
| `ZimService:915` summarize sorts + string-keys every hit | med | Fused dedup/breakdown/top-K single pass, struct key. Sort retained — see Deferred #24 note on why accumulation can't be capped |
| `ZimService:363` leadSnippet parses whole article per hit | med | `ArticleSections.parse` over a 64 K-char prefix (lead is at the top; `parse` strip-HTML'd every section) |
| `ArticleSections:110` Int-offset `index(offsetBy:)` walks | med | Markers store `String.Index` directly; heading regex also moved to RegexCache |
| `ConversationThreads:610/656/701` regex compiles (incl. per-link) | med | RegexCache in `parseLinks`/`proseParagraphs`/`decodeAndStrip` |
| `ConversationThreads:664` `out +=` paragraph concat | low | `[Substring]` + `joined` (done while in the function; see Dismissed for the general `+=` class) |
| `MCPToolAdapter:113` route_status per-vertex haversine w/ recomputed origin trig | med | Equirect argmin, origin cos hoisted. Cached last-index memo unnecessary once trig is gone (~µs scan) |
| `MCPToolAdapter:246` registry re-serialises schemas every turn | low | Built once, cached (`hasStreetzim`/`surface`/vocabulary are immutable post-init) |
| `MCPToolAdapter:1047` article_overview reads body twice more | low | One shared fetch feeds `relatedLinks` + `disambiguationAlternates` (defaulted `html:` param keeps other call sites source-compatible) |
| `MCPToolAdapter:2253` re-filter+alloc just to count | low | Counted in the candidates pass |
| `Embeddings` full sort for k-NN (batch-25) | med | Bounded top-k insertion buffer |
| `Embeddings:222` `[String].contains` in loop | low | `Set` membership |
| `Geocoder` rank builds a Place per match | low | Score first, build `Place`s only for the `limit` winners (kind short-circuit was already ordered correctly) |
| `Geocoder:133` `branch + "-"` per leaf | low | Hoisted |
| `Gemma3Template:353` repairJSON compiles 3 regexes/call + `,,` rescan loop | low | RegexCache + single `,{2,}` pass |
| `ReferenceResolver:589` regex compile per call | low | RegexCache |
| `MemoryProbe:83` unbounded continuous samples | low | Cap at 50k with pairwise-max decimation — `peak` (the jetsam metric) stays exact; band counts approximate past the cap, documented |
| `QwenChatMLTemplate:149` + `Gemma3Template:210` + `Gemma4ToolCallParser:34` whole-buffer re-scans per streamed chunk | med/low | Fixed at the single call site instead of per-template cursors — see ChatSession gate below. Templates stay stateless |
| `SZRGSpatial:410/487` byte-append blob copies | low | Single bulk (memcpy-path) copies. The "doubles memory per cached cell" claim is transient-only — the source `Data` is released after parse (noted, not a retained double) |

## Applied — iOS app (`ios/MCPZimChat/`, `ios/MCPZimEval/`)

| Finding | Sev | What was done |
|---|---|---|
| `Gemma4Provider:780` O(n²) full re-decode per token | high | **Partial.** Exact subset applied: per-token `map { Int($0) }` allocation removed (parallel `[Int]`), `<tool_call|>` detection on a bounded tail (`tailContains`). The full-decode-per-token itself is Deferred #1 — the windowed rewrite needs tokenizer segment-additivity guarantees we can only verify on-device; it was already on the July follow-ups list |
| `Gemma4Provider:444` 750 ms full-tree `dirSizeBytes` poll | med | 2.5 s cadence (Hub progressHandler still delivers fine-grained updates between ticks) |
| `Gemma4Provider:142` + `FoundationModelsProvider:75` continuation arrays grow forever | med ×2 | UUID-keyed dicts, removed in `onTermination` (async removal — termination can fire re-entrantly during registration) |
| `SemanticReranker:32` unbounded per-hit vector cache | med | Capped at 2048 vectors, wholesale flush (same policy the file already used for `embedTextCache`) |
| `ZimfoContext+Adapter:24` polyline copied per tool dispatch | med | Memoized on new `ZimfoContext.routeVersion`; lock-guarded holder because extensions can't add stored properties and ZimfoContext deliberately doesn't import MCPZimKit |
| `ZimfoRunner:19` full ZIM stack rebuild per Siri intent | med | Memoized on a cheap library fingerprint (Documents `.zim` names + bookmark blob hashes); `@MainActor` statics, no races |
| `ZimfoContext:119` whole route JSON rewritten per location update | low | Location goes to a tiny sidecar file; combined snapshot only on route set/clear; `load()` merges with old-install fallback |
| `LocationFetcher:72` subscribers append-only | low | Token-keyed subscribe/unsubscribe; `ChatSession.deinit` releases (eval harnesses build many sessions per process) |
| `ChatSession:394` sync `print` on the MainActor hot path | med | `#if DEBUG` (os_log + LogArchive keep Release parity) |
| `ChatSession:392` `removeFirst(1)` ring shift per log line | low | Slack-sized bulk drops (1/64th the shifts; small-cap-safe formula) |
| `ChatSession:3074` re-render+re-tokenize per dropped exchange | low | Sheds a char-density-estimated batch per render+tokenize pass; outer loop still re-checks the exact count, so correctness is unchanged |
| `ChatSession:2997` `turnsChars()` full reduce per drop | low | Running total |
| `ChatSession:3824` enrichSearchHits strips whole articles | low | Bounded 64 KB prefix strip (lossy UTF-8 decode absorbs the split codepoint). The full entry read remains — the `ZimReader` protocol has no range API (same constraint as HeroMediaView, Dismissed) |
| `ChatSession:4159` ~2 embeds × every section before discussion | low | Capped at 32 sections/call; retrieval degrades to deterministic order beyond it (documented behavior of missing vectors) |
| `ChatView:229` displayText pipeline ×2 per 10 Hz push | med | 16-entry (hash,count)-keyed memo shared by indicator + bubble |
| `ChatView:473` live WKWebView per article trace in history | med | `isLatestAssistant` guard — the exact pattern the route/places branches above it already had |
| `DebugPane:21` DateFormatter per row per body pass | med | `static let` (+ POSIX locale) |
| `PastLogsView:70` formatters + re-stat per row | med | Static formatters + new `LogArchive.allFileInfos()` returning the enumeration's prefetched metadata |
| `PlacesWebView:78` re-parse rawResult ~5×/body eval | med | Parsed once at view init (trace is immutable); `resolveSpec` reads the stored `zim` field. Library-membership checks stay in body (`@Environment` unavailable at init) |
| `RouteWebView:70-109` 4 parses + downsample + geoJSON per GPS tick | med | Endpoints/turns/zim fields/geoJSON all parsed+built once at init |
| `VoiceChatController:775` sanitize whole reply every 75 ms | med | Recompute only when raw text length changed (streaming appends; display scrubs change length too) |
| `LlamaCppProvider:1127` whole-buffer stop-sequence scan per token | med | Bounded rolling tail (≥2× max stop length kept, so cross-piece markers still land); dead `buffered` accumulator deleted — it existed only for this scan. The `+=` half of the finding is the dismissed class |
| `LlamaCppProvider:1117` 64-byte buffer + map alloc per token | low | Hoisted reusable `[UInt8]` scratch, `withMemoryRebound`, no `map` |
| `DebugReport:103` encode+b64+chunk-log on main | low | **Partial.** b64 + chunked os_log loop detached at utility QoS; the JSON encode stays sync because the returned hash (the API contract) derives from it |
| `LogArchive:84` whole-file read for a 3-line tail | low | 16 KB seek+tail read |
| `KokoroDownloader:116` Observation write per progress callback | low | Coalesced to ~8 Hz |
| `TTSService:42` normalized copy + buffer copy per chunk | low | `gain(for:)` scan (no alloc) + gain applied during the PCM-buffer write; unused `normalized` removed |
| `EvalHarness:886` 20k-entry debug ring inflating measured RSS | low | 2k (only `suffix(40)` and since-mark windows are read). `ConversationalEvalTests`'s own 20k left alone — the review itself dismissed that one |

## Applied — vendored Kokoro (`ios/LocalPackages/kokoro-ios/`)

| Finding | Sev | What was done |
|---|---|---|
| `KokoroTTS:364` `.item()` per output frame | **critical** | One `asArray` bulk sync; alignment matrix filled natively in a single fused loop |
| `KokoroTTS:353` `.item()` + tensor build per phoneme | high | Same rewrite (the two loops became one) |
| `TimestampPredictor:57` `.item()`/slice-sum per token | high | One bulk `asType(.float32).asArray` (explicit cast preserves the old `.item()` conversion semantics); native array math after |
| `ConvWeighted:100` weightNorm + bias reshape per conv call | med | Precomputed at init — safe because KokoroSwift passes weights through inits and has **zero** `update(parameters:)` call sites (grep-verified); norm helpers made static for pre-`super.init` use |
| `LSTM:151` `insert(at: 0)` per backward step | low | Append + one `reverse()` |
| `DurationEncoder:123` zeros+copy pad when shapes already match | med | Conditional on actual length mismatch |
| `TextEncoder:143` zeros + `_updateInternal` "pad" | low | Deleted — `_updateInternal` replaces contents wholesale, so the step never padded anything; it was a same-shape alloc + forced eval for a no-op in **all** cases |
| `Tokenizer:20` map→filter→map, 3 passes | low | Single fused pass with `reserveCapacity` |

## Applied — vendored mlx-swift-lm (safe subset)

| Finding | Sev | What was done |
|---|---|---|
| `LoraTrain:46` re-tokenize dataset every epoch | low | Lazily-filled per-index token cache on the iterator |
| `WiredMemoryUtils:46` re-encode growing seed string | med | Encode seed once, tile the ids (filler only needs to be valid tokens — the function's own doc) |
| `Evaluate:1889/1922` `(batch ?? []) + [element]` reducers | low | **Partial.** In-place `append` (removes the temp array + concat alloc). CoW still copies once per call because the public reducer signature borrows `batch`; a true O(1) needs an `inout` API change |
| `MediaProcessing:176` `clearCaches()` per video frame | low | `clearCache:` parameter; video paths clear once per sequence; single-image default unchanged |
| `Idefics3:534` + `Paligemma:552` hidden-states collected then discarded | low ×2 | `outputHiddenStates: false` (review verified the consumed value doesn't come from the collection) |
| `FalconH1:658` per-token `[KVCache]` alloc for an ignored arg | low | `cache: nil` (the function unconditionally returns nil) |
| `GLM4MOELite:411` fp32 scores promote the MoE combine tensor | low | Scores cast to activation dtype before the broadcast multiply |
| `NanoChat:17` hand-composed RMSNorm | med | Fused weight-less `MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, ...)` — the exact idiom Qwen3Next already uses in this repo |

## Applied — tools

| Finding | Sev | What was done |
|---|---|---|
| `bench_memory.py:100` + `bench_memory_gemma4.py:40` O(target²) preamble build | med ×2 | Incremental per-paragraph encode ("\n\n" joins keep BPE edge effects inside the existing truncation tolerance) |
| `eval_gemma4_native.py:63` full-buffer marker scan per token | low | 256-char rolling tail (`decoded` keeps full text) |
| `llm-smoke/eval.py:417` schema re-dump per case | low | Module-level `TOOLS_JSON` |
| `llama-smoke/eval.py:1396` schema re-dump per tool-loop render | low | `_TOOLS_SCHEMA_JSON` used when the module constant is passed |

## Applied — not tied to a single finding

- `ChatSession.runGenerationLoop` **marker gate**: the per-chunk
  `extractToolCall` scan now waits for the first `<` or backtick in the
  stream (every opener across all templates/parsers starts with one; the
  induction is per-chunk, so nothing can be missed). This is the chosen
  fix for the four separate "template re-scans the buffer per token"
  findings — one call-site gate instead of per-template scan cursors,
  keeping templates stateless.
- `RouteProgress.remaining` (ZimfoContext): equirect argmin. Not flagged
  by DS4, but it is the documented duplicate of the flagged
  `RouteSnapshot.remaining` — fixed for consistency.
- New `RegexCache.swift`; `LogArchive.allFileInfos()`;
  `ZimfoContext.routeVersion`; `ChatSession.deinit`.

---

## Deferred (real findings, deliberately not fixed) — and why

Unifying constraint: **this pass ran without Xcode/Metal**, so anything
whose correctness can't be argued line-by-line (MLX graph math, tokenizer
semantics, model numerics) was left alone rather than shipped unverified.
Ordered by expected value:

| # | Finding | Sev | Why deferred |
|---|---|---|---|
| 1 | `Gemma4Provider:780` full re-decode per token (+ MLXLMCommon `Tokenizer:96` NaiveStreamingDetokenizer) | high/med | An exact incremental detokenizer needs verified SentencePiece segment-additivity (leading-space handling at segment starts); a wrong window silently corrupts visible chat text AND `tokensAtCutoff`, which would break the KV-prefix reuse the July pass built. Was already on July's follow-up list. The surrounding overhead (per-token alloc, whole-buffer contains) IS fixed |
| 2 | `Jamba:281`, `GatedDelta:241`, VLM `Qwen35:100` sequential SSM/delta scans | high/med ×3 | Associative-scan or chunked-eval rewrites of recurrence math; log-space cumsum variants are numerically unstable. Unverifiable without a Metal device |
| 3 | `Gemma2:72` hand-composed attention → fused SDPA | med | Depends on the pinned MLX version exposing soft-capping in SDPA; silently wrong attention is worse than slow attention |
| 4 | `ToolCallProcessor:143` parse-per-chunk | med | The review's suggested gate ("only parse when `jsonBracesBalanced`") is **unsafe**: the balance counter is not string-aware, so a tool call whose string argument contains `{`/`}` would never be gated in — dispatch would stall forever. Prerequisite: a quote/escape-aware walker (behavior change in a tested vendored parser) |
| 5 | VLM per-patch/host-scatter set: `Gemma3:857`, `Gemma4:60/118/1569/1856`, `GlmOcr:105/509/941`, `Idefics3:715`, `LFM2VL:215/726`, `Mistral3:150/706`, `Pixtral:224/298/854`, `Qwen25VL:550`, `Qwen2VL:245`, `Qwen3VL:542/924/1328`, VLM `Qwen35:366` | med/low ×~20 | Vectorized rewrites of vendored vision paths — index arithmetic that must be verified against real image inputs. The app's current usage is text-only, so value is speculative |
| 6 | `Gemma3nText:435/593/601`, `Internlm2:47` per-layer host syncs / rebuilt masks | med ×3, low | Require threading host-side offsets through vendored layer APIs (signature changes) or caching in modules; models unused by the app |
| 7 | `Gemma.swift:25` (`1+weight`), `NemotronH:82` (ones), `Qwen3Next:272` (scalars), `Gemma4Text:687` (PLE slices) | low ×4 | All need a cached `MLXArray` stored on a `Module` — mlx-swift reflects stored MLXArray properties as **parameters**, which `update(parameters:, verify: .all)` weight loading would then reject. Needs a non-parameter box pattern this codebase doesn't have yet |
| 8 | `Gemma.swift:36` clipResidual fp32 round-trip, `Phi:66` fp32 queries | med ×2 | The upcasts are deliberate numerics (overflow clamping / phi stability); "keep the stream fp32" changes model output tolerances — needs eval runs |
| 9 | `RoPEUtils:144/293`, `SuScaledRoPE:43`, `KVCache:671` | med, low ×3 | Layout/caching changes inside per-token attention math; wrong = silently wrong positions |
| 10 | `DoRA+Layers:22` cache adapted-weight norm | med | Caching a value that must stay live for training gradients; correct only with an inference/training mode split |
| 11 | `Evaluate:444` FrequencyPenalty per-token histogram | low | The persistent-accumulator alternative must exactly reproduce scatter-add semantics over a sliding context; fiddly for a configured-only path |
| 12 | `Router:196` spatial nearest-node index, `Router:76` A* scratch reuse | med/low | A k-d/grid index over the eager node table is real work with real payoff — good standalone follow-up now that the kit tests run on Linux; scratch reuse needs actor-held buffers |
| 13 | `Embeddings:101` HashingEmbedder per-token allocs | low | UTF-8-byte rewrite of a hash embedder; output must stay bit-identical or every stored key invalidates |
| 14 | `StreamingSpeechPolicy:44` `Array(text)` per call | med* | *Contested: batch-18 dismissed this same site (called only on new fragments; consumption bounds the tail), batch-25 re-raised it. Sided with the dismissal — an integer→String.Index rewrite of boundary logic is high-risk for a bounded cost |
| 15 | `HotSplitGeocoder` leaf caching | med | Deliberately uncached (`cache:false`) to protect RAM per in-code comments; the new LRU bounds the *cached* sites. Re-caching leaves would churn the LRU during fan-outs |
| 16 | `PromptExperiment:533` per-token re-decode | low | Dev smoke harness bounded at 60–120 tokens by its own config |

## Dismissed (claims that don't hold) — and the evidence

| Finding | Claimed | Why dismissed |
|---|---|---|
| `Gemma4PromptTemplate:78`, MLXLMCommon `ChatSession:264`, `SmolVLM2:114`, `ModelProvider:195` — `out += piece` "O(n²)" | med/low ×4 | Native Swift `String.append` on a uniquely-referenced buffer is amortized O(1) (geometric growth). Batch-22 of the review itself dismissed this exact pattern in LFM25/QwenChatML `renderTranscript`; batches 7/16/12/21 re-flagged it inconsistently. (Python `+=` is different — and the Python cases WERE fixed) |
| Kokoro `DurationEncoder:109` / `TextEncoder:131` per-layer re-masks "redundant" | low ×2 | The masks are load-bearing: AdaLayerNorm rewrites zeroed padding rows to `beta` (mean 0/var 0 → normalized 0 → `+beta`), and convs with padding smear nonzero values into padded positions. Removing the re-masks changes model output |
| Kokoro `LSTM:88` per-step `xProj` slice → "pre-split once" | med | A pre-split still needs a per-step squeeze: one view op per step either way. The O(seqLen) sequential graph is inherent to the LSTM recurrence, not the slicing |
| `llama-smoke/eval.py:738` "psutil.Process() per sample" | med | Already hoisted: `proc = psutil.Process()` sits outside the `while` loop, one per `start()`. The per-sample `memory_info()` syscall is the measurement itself |
| `generate_chains3.py:941` flush per row | low | Deliberate crash-resume checkpointing — batch-26 dismissed the identical pattern in the three sibling generators for exactly that reason. Batching would trade lost seconds-per-row teacher-LLM output for noise-level syscall savings |
| `HotSplitGeocoder` "cap only applies post-exact-match" | med | The `matching.count >= max(200, limit*8)` break runs at the end of **every** leaf iteration, for substring queries too. The residual cost (rare-name queries reading many leaves) is inherent to hash-by-full-name bucketing |
| `HeroMediaView:129` full read for a 64 KB scan | low | The `ZimReader` protocol has no range-read API; the review's own text says "otherwise accept the current cost". Accepted |
| `SZRGSpatial:410` blob copy "doubles memory for every cached cell" | (part of low) | Transient only: the source `Data` is released after parse; only the `[UInt8]` copy is retained. The real cost — the byte-by-byte append loop — was fixed |
| `ConversationalEvalTests:429` 20k debug ring | (batch-35) | The review's own batch-35 dismissed it (test harness, per-turn mark scans); left untouched deliberately while the EvalHarness twin WAS lowered |

---

## Test-only changes — NOT committed to the repo

The Linux `swift test` run (437/437 green, Swift 5.10.1/Ubuntu 24.04) used
a scratch copy of `swift/` with five compile shims. They are deliberately
absent from the repo because the package only declares iOS/macOS support;
listed here so the test claim is reproducible:

1. `LinuxTestShims.swift` — no-op `os.Logger` (with privacy-annotated
   interpolation), pure-Swift SHA-256 (verified byte-for-byte against
   Python hashlib on 3 vectors incl. multi-block), pass-through
   `autoreleasepool`.
2. `MemoryStats.swift` — `#if canImport(Darwin)` guard, Linux returns 0
   (the API already documents 0 as its failure value).
3. `SZRGChunked.swift` + its test — CryptoKit import behind
   `#if canImport(CryptoKit)`.
4. `MCPToolAdapter.swift` — `import os` behind `#if canImport(os)`.
5. `Gemma4ToolFormat.swift` — the `CFGetTypeID`/`CFBooleanGetTypeID`
   NSNumber-boolean check behind `#if canImport(Darwin)` (on Linux,
   `JSONSerialization` yields native Swift `Bool`, which the line above
   it already handles).

If Linux CI is ever wanted, these ~140 lines plus a workflow file are the
whole cost. None of them alter behavior on Apple platforms (all are
`#if`-gated to non-Darwin) and none touch logic changed by this pass.

## Verification status by area

| Area | Status |
|---|---|
| `mcpzim/` Python | `pytest` 41/41 + nearest_node fuzz (0/500 mismatches) + strip differential check |
| `swift/` MCPZimKit | **437/437 tests green** (Linux toolchain, shims above) |
| `ios/MCPZimChat` app | Compiles-by-inspection only — **needs an Xcode build** (UIKit/SwiftUI) |
| `ios/LocalPackages` (Kokoro, mlx-swift-lm) | Compiles-by-inspection only — **needs Xcode + Metal/MLX**; Kokoro fixes are exact-math rewrites, best smoke-checked with a long voice-mode reply |
| `tools/` Python | `py_compile` clean; behavior-preserving hoists |
