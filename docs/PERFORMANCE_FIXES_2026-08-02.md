# Performance fixes applied — 2026-08-02

The safe tier of the DS4 sweep review (`DS4_REVIEW_PERF.md`, repo root —
376 files, ~80 findings across 37 batches), applied on
`claude/review-bugs-performance-s61ch3`. This pass covers every
critical/high finding, all first-party mediums with mechanical fixes, the
Kokoro TTS hot path, and the low-risk subset of the vendored mlx-swift-lm
findings. Deep MLX graph rewrites are deferred (§ Deferred) because they
need on-device verification.

Verification: `pytest` 41 green plus a fuzz harness for the new
`nearest_node` grid (500 random queries vs brute force, 0 mismatches;
warm query 0.076 ms vs 54 ms linear on 500k nodes). MCPZimKit's full
test suite — **437 tests, 0 failures** — was run in-session on a Linux
Swift 5.10.1 toolchain against a scratch copy with four Darwin-only
corners shimmed (no-op `os.Logger`, pure-Swift SHA-256 verified against
hashlib, pass-through `autoreleasepool`, `MemoryStats` → 0; plus a
`canImport(Darwin)` guard around `Gemma4ToolFormat`'s CFBoolean check).
The shims touch no logic this pass changed. Still to run on a Mac
before merging: the iOS/macOS app builds and the Kokoro / mlx-swift-lm
package builds (they need Xcode + Metal/MLX, which no Linux environment
can provide).

## Bugs (unbounded growth / leak class)

| Fix | Severity | Mechanism |
|---|---|---|
| `DefaultZimService.chunks` LRU (`maxCachedChunkRecords` = one full-scan) | high | the unbounded chunk cache was the observed 5.4 GB-RSS jetsam; eviction keeps repeat-query speed. Covers `loadChunk` AND `loadCategoryChunk` |
| `Gemma4Provider`/`FoundationModelsProvider` state-stream continuations keyed by UUID, removed `onTermination` | medium ×2 | every picker/status subscriber was retained and yielded-to forever |
| `SemanticReranker` per-hit cache capped at 2048 vectors (wholesale flush, same policy as `embedTextCache`) | medium | unbounded `zim:path → [Double]` map could reach tens of MB |
| `LocationFetcher.subscribe` returns a token; `ChatSession.deinit` unsubscribes | low | eval harnesses build many sessions per process; dead closures accumulated in the singleton |
| `MemoryProbe` continuous sampler bounded (pairwise-max decimation past 50k samples — `peak` stays exact) | low | multi-hour soaks grew the sample array without bound |
| `EvalHarness` debug ring 20k → 2k | low | the ring survives `resetConversation()` and inflated the RSS the probe was measuring |

## Python server

- `routing.py nearest_node`: lazily-built uniform grid over `(lat, lon)`
  (~16 nodes/cell, ring search with an exact lower-bound cutoff). Same
  argmin as the scan it replaces.
- `content.py search_zim`: full snippet parse only for the first 8 hits,
  and only over a 96 KB HTML prefix (was: full BS4 parse of up to 50
  whole articles per search).
- `content.py html_to_text`: the ~30 per-selector `soup.select` passes
  are now ONE combined selector group pass (nested matches guarded via
  `decomposed`).

## Swift engine (MCPZimKit)

- `RegexCache` (new): process-wide compiled-`NSRegularExpression` cache.
  Wired into `IntentRouter.match/matches` (~15–20 compiles per user turn
  plus per-sentence loops), `ReferenceResolver.firstMatch`,
  `WikiLinks.parseLinks/proseParagraphs/decodeAndStrip`,
  `Gemma3Template.repairJSON` (also: comma-run collapse is one `,{2,}`
  pass instead of a rescan loop), and `ArticleSections.parse`.
- `ArticleSections.parse`: markers hold `String.Index` directly — the
  Int-offset round-trip walked the string from the top per section.
- `ZimService.leadSnippet`: parses a 64 K-char prefix instead of
  strip-HTML-ing every section of every candidate hit (runs per hit ×
  overfetch on the search path).
- `summarize` (near_places): fused dedup/breakdown/top-K single pass,
  struct dedup key instead of interpolated strings, no full dedup array.
  `totalInRadius`/`breakdown` semantics unchanged.
- `SZRGGraph.nearestNode`: equirectangular squared-distance scan (no
  per-node trig), mirroring the July Python fix (PY2).
- Spatial A* (`Router.aStarSpatial`): one-cell memo walks the immutable
  cell's flat stride-5 edge array via the new
  `SpatialGraph.cell(containingNode:)` — no per-pop actor hop, no
  per-expansion `[SpatialEdge]` materialisation (up to `popLimit` = 200k
  per route). `edgesOfNode` kept for existing callers/tests.
- `SpatialGraph.ensureCell`: fetch+parse now `Task.detached` — cell loads
  for *different* cells no longer serialise behind the actor.
- SZCI/SZRC parsers: blob copies are single bulk `memcpy`-path copies
  instead of byte-append loops.
- `MCPToolAdapter`: `remaining(at:)` equirect argmin (was per-vertex
  haversine per `route_status`); `registry` built once and cached;
  `article_overview` fetches the body once for `relatedLinks` +
  `disambiguationAlternates` (was 2 extra whole-article reads);
  `fetchWikiExcerpts` counts wiki-tagged rows in the candidates pass.
- `EmbeddingIndex.nearest`: bounded top-k selection; `scores`: `Set`
  membership. `Geocoder.rank`: `Place`s built only for the `limit`
  winners; loop-invariant `branchPrefix` hoisted.
- `RouteProgress.remaining` (ZimfoContext): same equirect argmin as its
  MCPZimKit twin.

## iOS app (MCPZimChat)

- Streaming tool-call scans gated in `ChatSession.runGenerationLoop`:
  the per-chunk whole-buffer parses only start once a `<` or backtick has
  streamed in (every opener across templates starts with one) — kills
  the O(n²) prose-streaming scans flagged against
  QwenChatMLTemplate/Gemma3Template/Gemma4ToolCallParser/ChatToolCallParser
  at the single call site, with templates staying stateless.
- `Gemma4Provider`: per-token `tokenIDs.map { Int($0) }` allocation
  removed (parallel `[Int]` maintained); `<tool_call|>` detection scans a
  bounded tail instead of the whole reply per token; download-progress
  poll 750 ms → 2.5 s (each tick walks the whole HF cache tree);
  continuation leak fixed (above).
- `LlamaCppProvider`: stop-sequence detection on a bounded rolling tail
  (whole-buffer `contains` per sampled token was O(n²)); detokenise
  scratch buffer hoisted + `map` removed. `buffered` deleted (it existed
  only for the stop scan).
- `ChatSession`: `debug()` `print` is now `#if DEBUG` (os_log +
  LogArchive keep Release parity) and the ring trims in slack-sized
  bulk drops; history-trim loops keep a running char total and shed
  batches of exchanges per render+tokenize pass; `enrichSearchHits`
  strips a 64 KB prefix; discussion embeddings capped at 32 sections per
  call.
- `ZimfoRunner.load()` memoized on a library fingerprint (was: full ZIM
  re-open per Siri intent). `ZimfoContext`: location updates persist to a
  tiny sidecar (was: full route+polyline JSON rewrite per intent);
  `mcpSnapshot()` route conversion memoized on a new `routeVersion`.
- Views: `HeroMediaView` now latest-assistant-only (each one is a live
  WKWebView — same jetsam guard the route/places branches already had);
  `MessageRow.displayText` memoized (indicator + bubble ran the full
  regex pipeline twice per 10 Hz push); `DebugPane`/`PastLogsView` static
  formatters + `LogArchive.allFileInfos()` batched metadata;
  `PlacesWebView`/`RouteWebView` parse `trace.rawResult` once at init
  (was: per body evaluation × several accesses, plus a fresh
  downsample + geoJSON build per GPS tick).
- Voice: `VoiceChatController` re-sanitizes only when the reply text
  changed (was: full regex pass every 75 ms poll); TTS gain applied
  during the PCM-buffer copy (one pass, no intermediate array);
  `KokoroDownloader` progress coalesced to ~8 Hz.
- `LogArchive.previousSessionUncleanTail`: bounded 16 KB tail read;
  `DebugReport`: base64 + chunked os_log loop moved off the main actor.

## Kokoro TTS (vendored, hot path)

- **[critical]** `createAlignmentTarget`: ONE `asArray` bulk sync — was
  one `.item()` full-pipeline stall per phoneme PLUS one per output
  frame (thousands per synthesis).
- **[high]** `TimestampPredictor`: same bulk-extract; all per-token
  `.item()`/slice-sum syncs removed.
- `ConvWeighted`: normalized weight + reshaped bias computed once at
  init (weights are init-time constants in this port; no
  `update(parameters:)` exists in KokoroSwift) — was a full L2-norm
  reduction per conv call per audio frame.
- `LSTM.backwardDirection`: append+reverse (was `insert(at: 0)` —
  O(seqLen²) shifts). `DurationEncoder`: pad-copy skipped when seq_len
  already matches. `TextEncoder`: the zeros+`_updateInternal` "pad" was
  a no-op by construction — removed. `Tokenizer`: single fused pass.

## mlx-swift-lm (vendored, safe subset)

- `LoRABatchIterator`: per-index token cache — tokenization (the
  dominant LoRA-training CPU cost) ran O(dataset × epochs).
- `WiredMemoryUtils.makeTokenIds`: encode the seed once, tile ids.
- `Generation/TokenGeneration.collect`: in-place append (was
  `(batch ?? []) + [element]`).
- `MediaProcessing`: `clearCaches()` once per video sequence instead of
  per frame (`clearCache:` parameter, default preserves single-image
  behavior).
- `Idefics3`/`Paligemma`: `outputHiddenStates: false` where the collected
  states were discarded. `FalconH1`: dead per-token `[KVCache]` alloc for
  an ignored argument removed. `GLM4MOELite`: MoE scores cast to the
  activation dtype BEFORE the broadcast multiply (was silently promoting
  the combine tensor to fp32). `NanoChat`: fused weight-less
  `MLXFast.rmsNorm` (same idiom Qwen3Next already uses).

## Tools

- `bench_memory.py` / `bench_memory_gemma4.py`: incremental preamble
  tokenization (was O(target²) re-encodes; dominated bench wall time at
  the 40k size). `llm-smoke/eval.py`: `TOOLS_JSON` hoisted.
  `eval_gemma4_native.py`: rolling-tail stop scan.
  `llama-smoke/eval.py`: `_TOOLS_SCHEMA_JSON` serialized once for
  `_lfm2_render`.

## Deferred (need on-device verification or a design decision)

In rough order of expected value:

1. **Incremental detokenizer** for `Gemma4Provider` (and MLXLMCommon's
   `NaiveStreamingDetokenizer`): the full re-decode per token is still
   O(n²). Exact windowed decode needs verified segment-additivity of the
   tokenizer (SentencePiece leading-space semantics) — carried over from
   the July follow-ups list; the allocation/scan overhead around it is
   fixed above.
2. **MLX linear-attention prefill scans** — Jamba `ssmStep`,
   GatedDelta/Qwen3.5 `gatedDeltaOps` per-token fallbacks: need
   associative-scan or chunked-eval rewrites; correctness is
   unverifiable without a Metal device.
3. **Gemma2 hand-composed attention → fused SDPA with soft-capping**:
   depends on the installed MLX version exposing the capping mode.
4. **VLM per-patch loops + host `maskedScatter`s** (Gemma3/Gemma4 VLM,
   Pixtral, Mistral3, LFM2VL, GlmOcr, Qwen2VL/Qwen3VL mrope):
   vectorized rewrites of vendored vision paths; the app's current
   usage is text-only.
5. **Gemma3nText / Internlm2 per-layer host syncs**, RoPE variant
   copies (Proportional/Yarn/SuScaled), `RotatingKVCache` mask reuse,
   DoRA denominator cache (would break training-time gradients if
   cached naively), Gemma `clipResidual`/`1+weight` (needs a
   non-parameter cache slot to survive weight loading).
6. **`ToolCallProcessor` gated parse**: the review's suggested
   brace-balance gate is UNSAFE as written — `jsonBracesBalanced` is not
   string-aware, so a string argument containing `{`/`}` would stall
   dispatch forever. Prerequisite: a quote/escape-aware walker.
7. `StreamingSpeechPolicy.takeSpeakablePrefix` `Array(text)`: batch-18
   itself dismissed this (called only on new fragments; consumption
   bounds the tail); an index-based rewrite is high-risk/low-reward.
8. Spatial `nearestNodeSpatial` k-d/grid index and `aStar` scratch-buffer
   reuse; geocode leaf-scan caching (deliberately uncached to protect
   RAM — the new chunk LRU covers the other sites).

## Dismissed (review claims that don't hold)

- Swift `String +=` / `out +=` accumulation findings
  (Gemma4PromptTemplate, MLXLMCommon ChatSession, SmolVLM2,
  ModelProvider fallback): native Swift `String.append` is amortized
  O(1) — batch-22's own dismissal note; batch-16/21 re-flagged the same
  pattern.
- DurationEncoder/TextEncoder **per-layer re-masks**: NOT redundant —
  AdaLayerNorm rewrites zeroed padding rows to `beta`, and convs smear
  into padding; removing the re-masks changes model output.
- LSTM per-step `xProj` slice pre-split: a pre-split + per-step squeeze
  costs the same one view-op per step as the slice it replaces.
- `llama-smoke MemoryProbe`: `psutil.Process()` is already constructed
  once per `start()`, outside the sampling loop.
- `generate_chains3` per-row flush: deliberate crash-resume
  checkpointing (batch-26 dismissed the identical pattern in the sibling
  generators); batching flushes would trade lost teacher-LLM rows for
  noise-level syscall savings.
- `HotSplitGeocoder` substring-scan cap: the `matching.count >=
  max(200, limit*8)` early-exit already applies to every leaf iteration.
