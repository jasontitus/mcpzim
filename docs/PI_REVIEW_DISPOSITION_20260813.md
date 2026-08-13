# 2026-08-13 review — disposition ledger

Companion to `PI_REVIEW_VERDICTS_20260813.md` (the verified triage) and the
two raw sweep reports. Same format as `docs/DS4_REVIEW_DISPOSITION.md`: one
row per finding, with the mechanism applied or the reason it was withheld.

Legend: **Applied** — fixed. **Partial** — safe subset applied, remainder
deferred with reason. **Deferred** — real finding, withheld deliberately.
**Refuted** — the verdicts pass disproved the claim; no change.

---

## Applied — correctness / security ("Fix first")

| # | Finding | Sev | What was done |
|---|---|---|---|
| 1 | `ConversationalEvalTests.swift:408` — 20-scenario eval entirely dead (`setupState` never `.ready`, every `send()` early-returns) | high | Harness now sets `setupState = .ready` after its own model load, mirroring `ChatSession.forTesting`. Added a structural guard in `runTurn`: `send()` appends user+assistant synchronously, so the test asserts `messages.count` grew by 2 — a swallowed `send()` can never again read as a pass |
| 1b | `ios/tools/eval.sh:89` — `\|\| true` hid the failing eval from automation | high | Capture `PIPESTATUS[0]`, tear down the ANE sampler, then `exit` with the real status. Display-only grep guarded so an empty log can't abort before the summary under `pipefail` |

## Applied — additional confirmed findings (main context)

| Finding | Sev | What was done |
|---|---|---|
| `ChatSession.swift` `triggerArticleRead` / `triggerDirectionsToCoord` missing `!isGenerating` | low | Both public entry points now carry `send()`'s guard. A popup button tapped mid-stream was appending a second placeholder while the in-flight generation wrote to `messages[count-1]`, so the reply landed in the wrong bubble and two Tasks raced `isGenerating`/`finishedAt` |
| `ChatSession.swift` `formatDuration` — "1h 60m" | nit | Round to whole minutes first, then split, so the carry reaches the hour. Regression test `DurationFormattingTests` pins 7199 s → "2h", 3599 s → "1h" |
| `ChatSession.swift` lifecycle observers never removed | nit | Tokens stored in `lifecycleObservers`, removed in `deinit` beside the existing LocationFetcher unsubscribe |
| `ChatSession.swift:623` setup busy-wait with no stall deadline | low | Deadline the **stall**, not the total (a 60 s total deadline is what this loop replaced, because healthy multi-GB downloads exceed it): the loop tracks a state+progress signature and fails with an actionable message after 300 s of no advance, so `dismissSetupFailure()` gives the user an escape instead of a permanently dead composer |
| `AppTelemetry.swift:71` — iOS `FirebaseApp.configure()` unguarded | low | Mirrors the macOS plist guard. The plists are gitignored since the 2026-08-03 secret incident, so a fresh clone crashed at launch instead of running without telemetry |
| `DebugReport.swift:54` — gist PAT in UserDefaults | low | Keychain-backed (`ThisDeviceOnly`, so it stays out of backups), with one-time migration + erase of any token an older build stored in UserDefaults |
| `SemanticReranker.swift:49` — one flaky asset download permanently disabled reranking | low | Only terminal outcomes latch `loadAttempted`: a nil embedder (a capability answer) latches; a failed download or load retries on the next call |
| `ZimfoIntents.swift:25` — Siri speaks "1 hours 1 minutes" | nit | Independent pluralization per unit, plus the same round-then-split carry fix |
| `ZimfoContext.swift:103` — sidecar JSON encode + atomic write per GPS fix | low | Location-only updates coalesce to one write per 30 s; route changes still persist immediately (that's the state an intent can't reconstruct). In-memory value stays exact |

## Applied — security / correctness, by lane

| Area | Finding | What was done |
|---|---|---|
| WebViews | `PlacesWebView` bridge + navigation policy (Fix-first #3) | Navigation gate inverted to deny-by-default: only `zim:`/`about:` commit, a user tap on an allowlisted scheme is handed to the system, everything else cancelled and logged. `mcpzim` handler now requires the posting frame to still be on `zim://` (checked via `frameInfo.request.url` with `securityOrigin` as cross-check, since the capture script is injected `forMainFrameOnly: false`). `UIApplication.open` allowlisted per action (`tel` for Call; `http`/`https`/`mailto` for Website). All three `isInspectable` sites `#if DEBUG`-gated |
| WebViews | Same class of hole in `RouteWebView` | Had **no** navigation policy at all (WebKit default `.allow`) while rendering the same untrusted shared-ZIM HTML and receiving live GPS. Given the mirror gate |
| WebViews | Article sheet had no `navigationDelegate` (found by the lane, closed in main context) | New `ArticleWebCoordinator` with the same policy, retained via the representables' `makeCoordinator()` |
| WebViews | View-`init` JSON parse at streaming cadence (Perf #4) | Both views memoize on `ToolCallTrace.id` — a `let UUID` on an immutable struct, so it is stable across exactly the re-inits being skipped |
| MCPZimKit | SZCI **v2** header drives a pre-read allocation (Fix-first #2) | Two layers: parse-time ceil-consistency on `(numNodes, numNodeShards, nodesPerShard)` with `multipliedReportingOverflow`, then a byte-grounded probe (shard count ≤ 1000, first/last shard read, 8-byte multiples, exact node accounting) **before** `[Int32](count:)` is allocated. 9 tests drive the real load path with a read-counting reader, so "no allocation" is proven by zero reads |
| MCPZimKit | `nearPlaces` haversine per record (Perf #5) | Bounding-box prefilter derived from the exact haversine in use. The report's suggested formula was **unsafe** — the lane's fuzz test caught it dropping in-radius points at 78°N/89°N, because poleward points have a smaller cosine than the centre; the shipped version uses the worst-case cosine over the latitude window. Measured **~21%** of scan wall time, not the 40–60% estimated (per-record dictionary lookups dominate) |
| MCPZimKit | `ConversationThreads` match materialization (Perf #8) | `enumerateMatches` + `stop.pointee` at the cap, in both `parseLinks` and `proseParagraphs` |
| MCPZimKit | Uncached `NSRegularExpression` (Perf duplicate group) | All 19 `.regularExpression` call sites plus 2 explicit constructions in `ArticleHeuristics` routed through `RegexCache` |
| Providers | Gemma4 generation not cancellable (Fix-first #4) | `onTermination` + cancellation checks + a real `cancelGeneration()` with the generation-ID idiom. The in-loop check `break`s rather than throws, deliberately: throwing would skip the KV-mirror commit and leave `cachedTokens` describing fewer tokens than the GPU cache holds, so the next turn would claim a hit against a cache offset N positions ahead |
| Providers | Gemma4 KV-cache data race (Fix-first #4) | Single `NSLock` + atomic snapshot triple, no `await` under the lock, plus a `cacheEpoch` so a background `resetPromptCache()` that lands mid-turn wins instead of being resurrected by the in-flight turn's write-back |
| Providers | `LlamaCppProvider` dangling `vocab` (Fix-first #7) | `vocab` nil'd on the failure path alongside `model`. Audit found a second issue: `load()` was not idempotent, so a model re-switch stranded the old context (~3 GB leak) — teardown preamble added under `modelLock` |
| Providers | Native tools bypass adapter clamps | `limit` → 1…50 and `radiusKm` → 0.05…100, mirroring `MCPToolAdapter` exactly |
| Streaming UI | Strip + markdown re-parse per push (Perf #2) | The old memo key was `(hashValue, count)` of the growing text — a fingerprint of the current text, so it missed by construction on every push **and** cost two full traversals to compute the miss. Re-keyed to `(utf8.count, raw)`; strip bounded to the window after the buffer's last `<` (every pattern starts with `<`); inline markdown parsing memoized per block, so a push re-parses only the block that changed. Measured 291 ms → 8 ms over a 300-push stream, and equivalence to the old pipeline pinned by 3,621 differential cases |
| Python | Boxed lists for million-node graphs (Perf #6) | Typed `array` throughout, fed from generators so the boxed intermediate is never materialized. `d` for coordinates (float32's ULP near ±180° is ~1.7 m — it would coarsen 1 cm source data 150×), `I`/`i` for indices, `B` for the one-byte speed field. **48.1 MB → 10.8 MB** retained; 720-route differential check showed byte-identical output |
| Python | `routing.py:471` raw `KeyError` | Clean `ValueError("no route data in <zim>: …")`, matching the module's existing idiom |
| Shell | `ls \| head` under `errexit`, wrong `RESULT` parse, hardcoded paths | Globs/`awk` instead of SIGPIPE-prone pipelines (deliberately not `\|\| true`, which would reintroduce the masking this same review flags in `eval.sh`); `eval_ft_pcgaming.sh` now merges k=v across **both** `RESULT` lines like its Python sibling — verified 0/13 → 3/13 against a stub with known ground truth; paths derived from the repo root |

## Found while fixing — not in either report

| Finding | What was done |
|---|---|
| `MarkdownMessageParserTests`, `OfflineContentCatalogTests`, `TTSNormalizationTests` exist on disk but have **zero references in `project.pbxproj`** — they had never executed, the same failure mode as the dead conversational eval the review did catch | Regenerated the project from `project.yml` (XcodeGen) so every test file is wired in. `MarkdownMessageParserTests` and `TTSNormalizationTests` now run and pass |
| `OfflineContentCatalogTests` had rotted while dead — it targets a `WikipediaArchiveCatalog` type that no longer exists, and asserts `maxi` (with-images) editions are ignored, which the product deliberately changed | Deleted as superseded by `CatalogParsingTests` (same parser, broader coverage). Its one still-valid assertion — non-English editions are filtered out — was carried over rather than lost |
| `ZimfoRunner.load()` can build two runners concurrently (pre-existing; the off-main-actor move widens the window slightly) | Flagged, not fixed — outside the review's findings. Worth in-flight dedupe |

## Deferred — real findings, withheld deliberately

| Finding | Sev | Why deferred |
|---|---|---|
| `ChatSession.swift:3351` per-chunk full-buffer tool-call opener scan | low | The suggested `lastScanned` window must overlap by (longest marker − 1) or a marker straddling a chunk boundary is missed — a missed tool call is a silent functional regression on the hottest conversational path. The report itself bounds the waste at a few KB (reply-token cap). Not worth the risk in a batch this size; revisit standalone with tests that fuzz chunk boundaries |
| `ChatSession.swift:4013` `enrichSearchHits` reads whole articles to keep 64 KB | low | Needs a bounded/prefix read on the `ZimReader` protocol (`read(path:maxBytes:)`); a protocol change touching every reader implementation, off-main already. Same conclusion the DS4 pass reached for `HeroMediaView` |
| `MCPZimChatApp.swift:36` launch-time main-thread work | low | Deferring Firebase configure + provider construction past first frame reorders app startup — the exact area where a mistake costs a launch crash rather than jank. Wants its own change with a launch-time measurement before/after |

## Refuted by the verdicts pass — no change made

Recorded so a future sweep doesn't re-raise them:

- `testflight-assign-internal.rb:25` [high] — `OpenSSL::BN#to_s(2)` returns raw
  big-endian bytes, not a digit string; verified against a real P-256
  signature. JWT is well-formed.
- `SpeechRecognizerService.swift:258` [medium] — the SDK declares
  `recognitionTaskWithRequest:` non-null, so the proposed `guard let`
  wouldn't compile.
- `Gemma4Provider.swift:799` [low] — Swift `String +=` on a uniquely
  referenced local is amortized O(appended), not O(n²).
- `PastLogsView.swift:79` [low] — `LogArchive` prunes to 20 files each
  session start, so the "unbounded directory" premise is false.
