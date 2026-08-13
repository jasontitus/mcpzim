# Pi sweep review (perf focus) — mcpzim-0aad28ab

Exhaustive per-file pass: 202 code files across 27 batches.

## Findings

# Batch 1 — performance findings

- [high] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:69 — cold-cache `buildFresh()` (also the bookmark loop at :91) opens every ZIM archive synchronously on the main actor. `ZimfoRunner` is `@MainActor`, so its static `buildFresh` is MainActor-isolated, and `LibzimReader(url:)` → `ZimArchive(fileURL:)` + metadata + fulltext/title index reads is blocking disk I/O (the file's own comment pegs it at "seconds"). The memoized `cached` is per-process, so every Siri/Shortcuts invocation that launches a fresh process pays this on the first intent. Consequence: main thread blocked for seconds on the first intent in a fresh process, risking Siri's intent time budget / an ANR in-app. Fix: make the archive-opening work `nonisolated` and run it on a background actor (or `Task.detached`) — `LibzimReader` is `@unchecked Sendable` — then hand the built `ZimfoRunner` back to the main actor.
- [low] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:48 — `libraryFingerprint()` performs a synchronous `contentsOfDirectory` of Documents plus a `UserDefaults` array read on the main actor on every `load()` call, i.e. every App Intent, even on a cache hit. Consequence: per-invocation main-thread directory scan (cheap at a handful of files, but needless syscall + string sort on the main actor for every intent). Fix: compute the fingerprint once and reuse the cached runner without re-listing when nothing can have changed, or move the listing off the main actor.
- [low] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:154 — `firstLead(of:)` eagerly materializes the entire article text: `trimmed.split(separator: "\n")` builds an array of every line's `Substring`, then `.filter` builds a second array, only to return the first line longer than 40 chars. `lookupTopicLead` already loaded the full article body just to extract a 400-char lead. Consequence: on large Wikipedia articles (tens-to-hundreds of KB, thousands of lines) this allocates thousands of Substrings and an intermediate array for a single first-paragraph result. Fix: iterate lines lazily and early-exit (`text.enumerateSubstrings(in:options:.byLines)` or a manual `firstIndex(of: "\n")` scan) so only the first qualifying line is touched.
- [low] ios/MCPZimChat/App/MCPZimChatApp.swift:36 — `App.init` runs `AppTelemetry.configure()` (which calls `FirebaseApp.configure()`, synchronous SDK init that reads the GoogleService plist) on the main thread, and the property initializers at :28–29 synchronously construct `ChatSession()` (builds ~15 model providers and reads the previous-session log tail via `LogArchive.previousSessionUncleanTail()`) and `ZimSwarmController()` (Swarm engine + FileManager setup) before first frame. Consequence: all of this adds to the launch/first-frame budget on the main thread before any UI appears. Fix: defer Firebase configure and heavyweight provider/controller construction until after first frame (or move the synchronous file/log reads off the main thread).
- [low] ios/MCPZimChat/AppIntents/ZimfoContext.swift:103 — `updateLastLocation` → `persistLocation()` synchronously `JSONEncoder().encode` + atomic `data.write` of a sidecar file on every location update. ChatSession's `LocationFetcher.subscribe` callback invokes `updateLastLocation` on every CoreLocation fix (25 m distance filter), so a driving session writes the file roughly once per 25 m of movement on the actor executor. Consequence: thousands of small encode + write + rename disk operations over a long drive (each tiny, but a per-fix synchronous disk write blocking the actor). Fix: throttle/coalesce the sidecar write (e.g. only persist when an active route exists and the fix moved materially, or debounce to once per N seconds).

## Coverage
eval/run_prepared_discussion_ab.sh — clean
ios/MCPZimChat/App/MCPZimChatApp.swift — findings: 1
ios/MCPZimChat/AppIntents/LocationFetcher.swift — clean
ios/MCPZimChat/AppIntents/ZimfoContext.swift — findings: 1
ios/MCPZimChat/AppIntents/ZimfoIntents.swift — clean
ios/MCPZimChat/AppIntents/ZimfoRunner.swift — findings: 3
ios/MCPZimChat/Chat/AppTelemetry.swift — clean
- [low] ios/MCPZimChat/Chat/ChatSession.swift:3351 — per-token-chunk full-buffer tool-call detection & parsing in the generation streaming loop — `containsToolCallOpener(buffer)` runs up to 5 linear `contains`/`range(of:)` scans over the entire accumulated buffer on every chunk until an opener appears (and line 4035 scans the same `"<|tool_call>"` literal twice), then `extractToolCall(in:)` re-runs the template parser + `ChatToolCallParser.firstCall` (each doing a `range(of:)` scan, balanced-brace walk, and `JSONSerialization` parse over the whole buffer) on every subsequent chunk. Cost is O(chunks × buffer length), i.e. quadratic in the reply/prelude length; wasted CPU on every generated turn, growing with reply length and stream chunk granularity. Bounded by the reply-token cap (≈256–1024 tokens → a few KB), so it is low rather than a hang, but it is pure waste on the hottest path. Fix: keep a `lastScanned = buffer.startIndex` and only scan `buffer[lastScanned...]` for the opener; run the full JSON parse only when the suffix contains a plausible closing marker.
- [low] ios/MCPZimChat/Chat/ChatSession.swift:4013 — `enrichSearchHits` materializes the full decompressed article for each of the top-3 search hits, then keeps only a 64 KB prefix and a 400-char preview — `ZimReader.read(path:)` returns a `ZimEntry` whose `content: Data` is the entire entry (confirmed in ZimReader.swift:61), so `data.prefix(64 * 1024)` discards the bulk of a commonly 100 KB–1 MB+ HTML article. Consequence: every `search` tool dispatch (LLM tool loop and fast-path rescue paths) decompresses up to 3 whole articles to extract a 400-char lead, adding ZIM-decompress latency and a transient memory spike on the search path (off-main via `Task.detached`, so it does not hang the UI, but it is wasted CPU/memory). Fix: add a bounded/prefix read to `ZimReader` (e.g. `read(path:maxBytes:)` that stops the entry decode at 64 KB) instead of reading the full entry and truncating afterwards.

## Coverage
ios/MCPZimChat/Chat/ChatSession.swift — findings: 2
- [medium] ios/MCPZimChat/Providers/FoundationModelsProvider.swift:352 — `generate()` (the default text-loop path) constructs a brand-new `LanguageModelSession()` on every call and passes the full re-rendered transcript to `streamResponse(to:)`; a tool-heavy turn re-enters `generate()` once per tool iteration, and each entry re-prefills the entire conversation prefix from scratch (no KV-cache reuse). Consequence: TTFT and per-turn latency scale with transcript length × number of tool round-trips (the same 10–15 s warmup the file's own header says the native-tools path was built to avoid paying "once per tool iteration"). Fix: reuse the warmed session per turn and pass only the new user/tool message, using `instructions:` for the static preamble (already the design of `generateNativeTurn`/`ensureWarmSession`); the code comment confirms this is the known fix.
- [low] ios/MCPZimChat/Providers/Gemma4Provider.swift:799 — `decodedSoFar += newText` accumulates the full decoded reply string on every generated token inside the streaming loop, copying the whole accumulated buffer each iteration (O(n²) in reply length). The accumulated text is only consumed by `tailContains`, which reads a fixed trailing window, so the full accumulation is redundant. Consequence: quadratic string copying/allocation churn in the per-token hot loop; bounded by `maxTokens` (~256–512 tokens) so sub-millisecond at current caps, but it grows with any future reply-length increase and is pure waste. Fix: keep only a bounded rolling window (e.g. last `marker.count` characters, or a byte-counted ring) instead of the full `decodedSoFar`.
- [low] ios/MCPZimChat/Providers/FoundationModelsProvider.swift:206 — `current.hasPrefix(lastText)` (same pattern at line 383) rescans the entire accumulated reply text on every chunk: `lastText` is the full prior accumulated string, so `hasPrefix` compares O(reply length) characters per chunk, i.e. O(n²) over the stream. Consequence: quadratic per-token comparison work in the token-diff loop; negligible at the current `maximumResponseTokens` cap but scales if the budget grows. Fix: diff on the framework's per-chunk delta (or track a String.Index cursor) instead of re-prefix-scanning the whole accumulated string each token.
## Coverage
ios/MCPZimChat/Chat/DebugReport.swift — clean
ios/MCPZimChat/Chat/Message.swift — clean
ios/MCPZimChat/Common/DeviceProfile.swift — clean
ios/MCPZimChat/Common/DiagnosticsUploader.swift — clean
ios/MCPZimChat/Common/LogArchive.swift — clean
ios/MCPZimChat/Common/SemanticReranker.swift — clean
ios/MCPZimChat/Common/ZimfoContext+Adapter.swift — clean
ios/MCPZimChat/Libzim/LibzimBridge.h — clean
ios/MCPZimChat/Libzim/LibzimBridge.mm — clean
ios/MCPZimChat/Libzim/LibzimReader.swift — clean
ios/MCPZimChat/Providers/FoundationModelsNativeTools.swift — clean
ios/MCPZimChat/Providers/FoundationModelsProvider.swift — findings: 2
ios/MCPZimChat/Providers/Gemma4Provider.swift — findings: 1
- [medium] ios/MCPZimChat/Sharing/ZimSwarmController.swift:250 — `importCompletedSwarm` (a `@MainActor` method) runs the whole `for source in fileURLs` loop (lines 233–265) on the main thread, doing synchronous per-file filesystem work: `attributesOfItem` stat calls (line 250, and again at 254 when the destination exists), `moveReplacing` → `createDirectory` + `fileExists` + `moveItem` (361–363), and `uniqueDestination`'s `fileExists` probe loop. — Importing a multi-file share (a voice-model directory tree or a folder swarm with many `.zim` files — dozens to hundreds of files) issues that many stat/create/move syscalls synchronously on the main thread, freezing the UI and blowing the 250 ms hang budget; the cost scales linearly with transferred file count. — Smallest safe fix: hop the entire loop to a detached/utility background task (e.g. `Task.detached`) and only mutate the `@Published` summary fields (`lastImportSummary`, `lastSkippedCount`) back on the main actor.

- [low] ios/MCPZimChat/Sharing/ZimCatalog.swift:266 — `StreetZimCatalog.firstMatch` compiles a fresh `NSRegularExpression` on every call, and it is invoked 4× per map card (lines 242–244, 247) inside the `for (tier, body) in sections { for card in … }` loop. — On a ~50-region catalog page this is ~200 regex compilations per `load()`, adding tens of milliseconds of one-time latency to the download-catalog screen; `WikipediaZimCatalog.parse` already compiles its pattern once (line 85) while this one does not. — Smallest safe fix: hoist the four constant patterns to `static let` pre-compiled `NSRegularExpression`s and pass them into `firstMatch` (or a `firstMatch(in:)` taking the compiled regex).

- [low] ios/MCPZimChat/Sharing/ZimDownloadManager.swift:431 — `SessionDelegateShim.urlSession(_:downloadTask:didWriteData:)` spawns a `Task { @MainActor }` on every URLSession progress tick; the resulting `progress` (line 250) calls `updateSleepBlocker()` (line 270) each time, which recomputes `hasActiveDownloads` via an `items.contains { $0.state == .downloading }` scan (line 318 → 105) even though active-download state only changes on download/pause/resume/cancel/finish, not on progress. — During a fast multi-GB download, `didWriteData` fires per network buffer, producing a stream of main-actor task allocations plus redundant O(items) scans and a redundant `isIdleTimerDisabled` write on every tick (main-thread churn and needless wakeups). — Smallest safe fix: drop the `updateSleepBlocker()` call from `progress` and keep it only on the state-transition paths (`download`/`pause`/`resume`/`cancel`/`finished`/`failed`), where download activity actually changes.

## Coverage
ios/MCPZimChat/Providers/LlamaCppProvider.swift — clean
ios/MCPZimChat/Providers/MockProvider.swift — clean
ios/MCPZimChat/Providers/ModelProvider.swift — clean
ios/MCPZimChat/Sharing/ChatSession+ModelSharing.swift — clean
ios/MCPZimChat/Sharing/ZimCatalog.swift — findings: 1
ios/MCPZimChat/Sharing/ZimDownloadManager.swift — findings: 1
ios/MCPZimChat/Sharing/ZimSwarmController.swift — findings: 1
- [high] ios/MCPZimChat/Views/NearbyShareView.swift:175 — `shareableVoiceSizeLabel` (plus `shareableModelSizeLabel`:166 and `hasShareableFiles` read at :96/:158) call into `ZimSwarmController.shareableVoiceBytes` / `shareableVoiceDirectories`, which synchronously walk the Kokoro + Supertonic model directories on disk (`KokoroAssets.isDownloaded`/`currentBytesOnDisk` do per-file `attributesOfItem`; `Supertonic3Assets.currentBytesOnDisk` runs a recursive `FileManager.enumerator` with `resourceValues` per file). These computed properties are re-evaluated on every body pass of `NearbyShareContent`, which is `@ObservedObject` on `SwarmManager` and republishes on every discovery change and transfer-progress tick — main-thread recursive directory enumeration + per-file stat repeated (potentially several times/sec) during transfers, jank/hangs scaling with the voice-model directory size (compiled `.mlmodelc` + MLX assets can be hundreds of files). — smallest safe fix: compute the byte totals once (cache in `@State` on appear, or memoize in `ZimSwarmController` keyed by a directory fingerprint and invalidate after a download completes) instead of reading them in a body-evaluated computed property.
- [medium] ios/MCPZimChat/Views/DownloadCatalogView.swift:199 — `status(of:)` calls `ZimDownloadManager.alreadyInLibrary(filename:)`, which performs a synchronous `FileManager.fileExists(atPath:)` per catalog row, evaluated on the main thread for every visible row on each body/scroll pass and re-run for all visible rows whenever `@ObservedObject downloads` publishes during active downloads — per-row disk stat on the main thread causes scroll jank on a catalog of dozens–hundreds of items (Wikipedia editions + StreetZIM tiers). — smallest safe fix: enumerate the Documents directory once (`contentsOfDirectory`) into a `Set` of filenames and test membership, or cache the in-library check.
- [low] ios/MCPZimChat/Views/DownloadCatalogView.swift:176 — `availableSpaceLabel` and `exceedsFreeSpace` (:181) each call `ZimDownloadManager.availableLibraryBytes()`, a `resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])` filesystem query; both are read in `goBar` (and `exceedsFreeSpace` again in the button action and the alert message) so the volume-capacity stat runs up to ~3× per body evaluation on the main thread — repeated main-thread stat on every re-render of the catalog screen. — smallest safe fix: compute the free-space value once per selection change and cache it, then reuse for the label, warning flag, and alert.
- [medium] ios/MCPZimChat/Views/MarkdownMessageText.swift:12 — `blocks` recomputes `MarkdownMessageParser.parse(source)` and every block's `InlineMarkdownText.attributed` re-invokes `AttributedString(markdown:options:)` (Foundation's Markdown parser) on every body evaluation. During streaming the assistant row re-renders at ~10 Hz with the growing text, so the whole message is re-tokenized and each block re-markdown-parsed per push on the main thread (the sibling `displayText` strip pipeline in ChatView.swift was memoized precisely for this 10 Hz path, but the parse pipeline is not) — dropped frames/scroll jank during long streaming replies, scaling with message length and block count; a large `MarkdownMessageTable` multiplies this by rows×columns. — smallest safe fix: memoize the parsed block array keyed by a (hash, count) of `source`, and/or render only the changed tail of the streamed message.
- [low] ios/MCPZimChat/Views/MarkdownMessageParser.swift:188 — `orderedItem(in:)` materializes `Array(indented.text)` (an `Array<Character>`) for every non-empty line that reaches it — i.e. every paragraph line on every parse, and the parser is re-run at ~10 Hz over the full growing streaming message — thousands of per-line `Array<Character>` allocations per pass in a hot path. — smallest safe fix: use `indented.text.prefix(while: { $0.isNumber })` directly on the String without materializing an Array.
- [low] ios/MCPZimChat/Views/NearbyShareView.swift:476 — `SwarmFormat.bytes` (and `rate`:480) call `ByteCountFormatter.string(fromByteCount:countStyle:)`, constructing a fresh formatter per call; they are invoked per row in `SwarmTransferRow`, `SwarmFileSelectionSheet`, and the share header — per-row formatter allocation during list rendering (minor scroll churn). — smallest safe fix: hoist a `static let` `ByteCountFormatter` (as `LibraryView` and `PastLogsView` already do) and reuse it.
- [low] ios/MCPZimChat/Views/PastLogsView.swift:79 — `reload()` runs `LogArchive.shared.allFileInfos()` (a `contentsOfDirectory(includingPropertiesForKeys:)` scan of the persisted logs directory) synchronously on the main thread from `.onAppear`; the debug-log directory grows by one file per launch with no pruning — a main-thread stall when opening Past Logs once many launches have accumulated. — smallest safe fix: load in `Task.detached(priority: .utility)` and assign the result to `@State` (the per-file detail read in `LogDetailView` is already off-main).

## Coverage
ios/MCPZimChat/Views/ChatView.swift — clean
ios/MCPZimChat/Views/DebugPane.swift — clean
ios/MCPZimChat/Views/DownloadCatalogView.swift — findings: 2
ios/MCPZimChat/Views/HeroMediaView.swift — clean
ios/MCPZimChat/Views/LibraryView.swift — clean
ios/MCPZimChat/Views/MarkdownMessageParser.swift — findings: 1
ios/MCPZimChat/Views/MarkdownMessageText.swift — findings: 1
ios/MCPZimChat/Views/ModelPickerView.swift — clean
ios/MCPZimChat/Views/NearbyShareView.swift — findings: 2
ios/MCPZimChat/Views/OfflineContentSetupView.swift — clean
ios/MCPZimChat/Views/PastLogsView.swift — findings: 1
- [medium] ios/MCPZimChat/Views/RouteWebView.swift:54 — `RouteWebView.init` re-parses `trace.rawResult` JSON (10–100 KB, ~1500-point polyline), runs `downsample`, and rebuilds the geoJSON string via ~400 `String(format:)` calls on every SwiftUI render pass, not once per view — SwiftUI re-initializes a child View's stored `let` properties each time the parent `MessageRow.body` re-evaluates, and `MessageRow.body` re-runs at ~10 Hz during streaming (per the `TraceKindCache` comment in ChatView.swift:805-809, which memoizes the *classification* but leaves the view `init` un-memoized) plus every GPS tick — the "parse once at init" comment does not hold. Concrete consequence: ~5–15 ms of main-thread JSON parse + NSString-format/join work per tick, i.e. tens of ms/sec of main-thread CPU and dropped frames/battery drain during every streamed route answer; scales with polyline size. Smallest safe fix: memoize the parsed fields by `trace.id` (mirroring `TraceKindCache`) or compute them lazily once in `@State`/`.onAppear`/`.task` instead of inside `init`.
- [medium] ios/MCPZimChat/Views/PlacesWebView.swift:66 — `PlacesWebView.init` performs two full JSON parses of the same immutable `trace.rawResult` (`parsePlaces(from:)` at line 68 plus a second `JSONSerialization.jsonObject` at lines 69–70 to recover the top-level `zim` field that `parsePlacesJSON` already decodes but discards), and the whole `init` re-runs on every parent `MessageRow.body` re-evaluation (~10 Hz during streaming + GPS ticks) for the same reason as RouteWebView. Concrete consequence: doubled JSON materialization (two `data(using: .utf8)` conversions + two `JSONSerialization` trees) per render tick on the main thread, scaling with the up-to-100-place result payload. Smallest safe fix: extend `PlacesPayload`/`parsePlacesJSON` to also surface the `zim` string so `init` does a single parse, and memoize the parse by `trace.id` (or move it to `@State`/`.onAppear`) so it doesn't re-run at streaming cadence.

## Coverage
ios/MCPZimChat/Views/PlacesWebView.swift — findings: 1
ios/MCPZimChat/Views/RootView.swift — clean
ios/MCPZimChat/Views/RouteWebView.swift — findings: 1
ios/MCPZimChat/Views/VoiceChatView.swift — clean
ios/MCPZimChat/Views/ZimURLSchemeHandler.swift — clean
ios/MCPZimChat/Voice/KokoroAssets.swift — clean
ios/MCPZimChat/Voice/KokoroDownloader.swift — clean
ios/MCPZimChat/Voice/ObjCExceptionWrapper.h — clean
ios/MCPZimChat/Voice/ObjCExceptionWrapper.m — clean
ios/MCPZimChat/Voice/SpeechRecognizerService.swift — clean
ios/MCPZimChat/Voice/Supertonic3TTSService.swift — clean
- [medium] ios/MCPZimChat/Voice/VoiceChatController.swift:849 — `sanitizeForSpeech(raw)` re-runs ~8 regex passes over the ENTIRE growing assistant reply every time `raw.count` changes, and `raw.count` changes on essentially every token during streaming decode — the loop is O(n²) in reply length (each pass is O(n), triggered n/token times) and runs on the main actor, competing with UI and TTS scheduling — concrete consequence: CPU churn and dropped UI frames on long voice replies (a 2–3 KB streamed reply re-sanitizes the whole string hundreds of times) — smallest safe fix: hoist the compiled `NSRegularExpression`/`Regex` objects to static constants and re-sanitize only the appended delta (track `lastSanitizedCount` and scrub `full[lastSanitizedCount...]`, prepending the prior scrubbed text), or skip re-sanitization unless the delta introduces a `<` that could start a tool/think marker.
- [low] ios/MCPZimChat/Voice/VoiceChatController.swift:883 — `String(full.suffix(full.count - spokenUpTo))` allocates a fresh copy of the entire remaining unspeakable text on every 75 ms poll while speech lags behind text — concrete consequence: repeated O(remaining) allocation churn scaling with reply length during generation-bound/eager-deferred turns — smallest safe fix: use the `Substring` directly (`let newFragment = full.suffix(...)`) without the `String(...)` copy; `takeSpeakablePrefix` only reads it.
- [low] ios/MCPZimChat/Voice/TTSService.swift:628 — `prepForTTS` compiles ~11 `NSRegularExpression`s on every invocation (`replacingOccurrences(of:with:options:.regularExpression)` at lines 628, 639, 655, and 684), and it is called per TTS chunk and per `speak`/`speakChunk` turn — concrete consequence: avoidable regex-compilation cost paid repeatedly on the synthesis path each turn — smallest safe fix: precompile the patterns into `static let` `NSRegularExpression` instances (or Swift `Regex` literals) and reuse them.
- [low] ios/MCPZimChat/Voice/TTSService.swift:536 — the gain-apply path uses a scalar per-sample loop `for j in 0..<nativeSamples.count { dst[j] = src[j] * gain }` (also at line 581) instead of a SIMD multiply — concrete consequence: each non-unity-gain chunk pays a scalar multiply over tens of thousands to ~1M+ Float samples (24 kHz), measurably slower chunk synthesis than necessary — smallest safe fix: replace with `vDSP_vsmul` (Accelerate) on the two float buffers, or `vDSP_vclip` if clamping is ever needed.
## Coverage
ios/MCPZimChat/Voice/TTSService.swift — findings: 2
ios/MCPZimChat/Voice/VoiceChatController.swift — findings: 2
ios/MCPZimChatMacTests/CatalogParsingTests.swift — clean
ios/MCPZimChatMacTests/ConversationalEvalTests.swift — clean
ios/MCPZimChatMacTests/GemmaToolEmissionTests.swift — clean
ios/MCPZimChatMacTests/MarkdownMessageParserTests.swift — clean
ios/MCPZimChatMacTests/ModelSharingTests.swift — clean
ios/MCPZimChatMacTests/OfflineContentCatalogTests.swift — clean
ios/MCPZimChatMacTests/TTSNormalizationTests.swift — clean
ios/MCPZimChatMacTests/ZimfoIntentsTests.swift — clean
ios/MCPZimCoreMLTTSBench/CoreMLTTSBenchMain.swift — clean
ios/MCPZimEval/EvalCLI.swift — clean
# Performance findings — batch 8

All five files in this batch are one-shot macOS CLI tools (eval harness, probe
commands, TTS benchmark) rather than app hot paths. Findings below are limited
to defensible redundant-recomputation / quadratic-scan patterns in the harness
control flow; every loop over data sized by user input (LLM/tool responses,
samples, article lists) was checked and is bounded or dominated by model
latency, so no N+1 / per-item-I/O / unbounded-cache findings exist here.

- [low] ios/MCPZimEval/EvalHarness.swift:838 — the "scenarios with no winner" pass re-filters the entire `scorecard.rows` array (`variants × scenarios` rows) once per scenario, making it O(V·S²) — each scenario run re-scans every row it already computed, so the end-of-run summary does a full matrix scan per scenario; negligible today (~12 variants × ~17 scenarios ≈ 3.4k string comparisons) but the redundant rescan grows quadratically as the hardcoded variant/scenario matrix expands — single pass over `rows` once, accumulating winning scenario names into a `Set<String>`, then derive `losers` by set difference.
- [low] ios/MCPZimEval/EvalHarness.swift:867 — `Self.fixtureForEval()` (which constructs all 14 scenario fixtures with dozens of large `ArticleSection` string literals and dictionary inserts) is rebuilt inside `runVariant`, i.e. once per matched variant, even though the fixture is identical and variant-independent — the same pure in-memory fixture is reconstructed ~12× per run; a few ms each, dwarfed by model load/decode seconds, so impact is minor but it is redundant recomputation on the per-variant loop — hoist the fixture to a `static let` (or build once before the variant loop) and pass the shared value into each `runVariant`.

## Coverage
ios/MCPZimEval/EvalHarness.swift — findings: 2
ios/MCPZimEval/LlamaCppProbeCLI.swift — clean
ios/MCPZimEval/ProbeCompareCLI.swift — clean
ios/MCPZimEval/ProbeE2ECLI.swift — clean
ios/MCPZimTTSBench/TTSBenchMain.swift — clean
- [medium] mcpzim/geocode.py:163 — Geocoder.search sorts the entire `scored` list of every matching record then slices `[:limit]` — for a 2-char streetzim prefix chunk (tens of thousands of address/POI records) and a common query like "main", the full O(n log n) sort dominates the request when only `limit` (default 5) results are returned; geocode latency grows superlinearly with chunk density — replace `scored.sort(...)` + `scored[:limit]` with `heapq.nsmallest(limit, scored, key=lambda s: (s[0], s[1]))` (O(n log limit)).
- [low] mcpzim/library.py:221 — find_by_path calls `z.path.resolve()` for every ZIM on every invocation, and the same realpath is recomputed as the cache key in geocode.py:85/96 and routing.py:453 on each geocode/route request — `Path.resolve()` issues realpath/lstat syscalls per component, wasted on the get_article/geocode/plan_driving_route hot paths every time a `zim` name is supplied — resolve once at scan time and store the resolved path on `OpenZim`/`ZimInfo` (or compare against a pre-resolved `Library` map).

## Coverage
ios/project.yml — clean
ios/scripts/mcp-crashes.sh — clean
ios/scripts/mcp-deploy-verify.sh — clean
ios/scripts/mcp-logs.sh — clean
ios/scripts/mcp-report.sh — clean
ios/scripts/testflight-assign-internal.rb — clean
ios/scripts/testflight-upload.sh — clean
ios/scripts/verify-app-signature.sh — clean
ios/tools/eval.sh — clean
mcpzim/__init__.py — clean
mcpzim/__main__.py — clean
mcpzim/cli.py — clean
mcpzim/content.py — clean
mcpzim/geocode.py — findings: 1
mcpzim/library.py — findings: 1
mcpzim/routing.py — clean
mcpzim/server.py — clean
pyproject.toml — clean
swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift — clean
swift/Package.swift — clean
- [medium] swift/Sources/MCPZimKit/Embeddings.swift:101 — `HashingEmbedder.embed` allocates per token (`tokenize` does `.map(String.init)`), then per token ≥4 chars materializes an `Array<Character>` (line 112) and 2–3 fresh `String`s per 3-gram (`String(padded[j..<j+3])`, `"n:"+…`, and `"#"+feature` in `bump` line 128). `HashingEmbedder` is the default `TextEmbedder` for `ArticleHeuristics.rankSectionsForQuestion` / `rankSectionsMultiSource`, which embed the question plus every section's title AND full body text, so a 40-section article triggers ~80 embeds and hundreds of thousands of transient heap allocations per discussion turn, scaling with total section prose — ARC retain/release churn and measurable latency on the article-discussion hot path on-device — hash over the `Substring`/UTF8 view directly (avoid `String` materialization), compute a rolling 3-gram hash instead of `Array<Character>` + per-gram substrings, and fold `"n:"`/`"#"` into the hash call without string concatenation.
- [low] swift/Sources/MCPZimKit/ArticleSections.swift:167 — `stripHTML` performs ~30 full-string passes over each section body (≈15 `regexReplace` passes including `removeSpansByClass`/two `removeNestedBlock`/`removeBlock` calls, plus 9 `blockBreaks` and 10 `entities` literal `replacingOccurrences` passes), and `parse` invokes it once per section body (and once per heading) on the article-open path — a ~30× constant-factor multiplier on article parsing: a 400 KB Wikipedia body becomes ~12 MB of regex scans and string copies per open, adding tens of ms of latency every time an uncached/evicted article is opened for overview/narration — consolidate the repeated passes into a single-pass scanner or fewer combined regex/literal passes.
- [low] swift/Sources/MCPZimKit/ArticleHeuristics.swift:424 — `groundedExtractiveAnswer` runs four bare `replacingOccurrences(of:with:options:.regularExpression)` calls (lines 424, 428, 431, 433) inside the `passages.enumerated().flatMap` loop; each call recompiles its `NSRegularExpression` from scratch (the same anti-pattern the file's own `stripCitations` comment at line 187 warns about), so a grounded follow-up turn recompiles 4 patterns per passage — grows linearly with the number of grounded passages and adds ~tens of µs of pure compilation per call on a turn hot path — route these through the existing `RegexCache.shared.compiled(...)` (already used elsewhere in the same module).
## Coverage
swift/Sources/MCPZimKit/AnswerAttribution.swift — clean
swift/Sources/MCPZimKit/ArticleCache.swift — clean
swift/Sources/MCPZimKit/ArticleHeuristics.swift — findings: 1
swift/Sources/MCPZimKit/ArticleSections.swift — findings: 1
swift/Sources/MCPZimKit/ChatToolCallParser.swift — clean
swift/Sources/MCPZimKit/ChatTurn.swift — clean
swift/Sources/MCPZimKit/ConversationFocus.swift — clean
swift/Sources/MCPZimKit/ConversationThreads.swift — clean
swift/Sources/MCPZimKit/Embeddings.swift — findings: 1
swift/Sources/MCPZimKit/Gemma3Template.swift — clean
swift/Sources/MCPZimKit/Gemma4PromptTemplate.swift — clean
swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift — clean
# Performance findings — batch 11

- [low] swift/Sources/MCPZimKit/Geocoder.swift:173 — `let lower = name.lowercased()` inside the per-record loop of `rank(_:query:limit:kinds:)` allocates a full lowercase String copy for every candidate record — `rank` is the geocode hot path and runs over chunks that can hold thousands of records (ZimService loads a whole `{prefix}.json` chunk when `leaves.count == 1`, and the file itself notes a broad substring query "can match thousands of records"), so each lookup pays one heap allocation + character copy per record name, plus the caller (ZimService) already lowercased the same name in its pre-filter — measurable allocation/ARC churn at thousands of records per query — replace with a copy-free case-insensitive search `name.range(of: q, options: [.caseInsensitive])` (compute the score offset from the returned range via `name.distance(from:to:)`), or use `name.localizedCaseInsensitiveContains(q)` when the offset is not needed.

## Coverage
swift/Sources/MCPZimKit/Gemma4ToolFormat.swift — clean
swift/Sources/MCPZimKit/Geocoder.swift — findings: 1
swift/Sources/MCPZimKit/GeoMath.swift — clean
swift/Sources/MCPZimKit/IntentRouter.swift — clean
swift/Sources/MCPZimKit/LFM25Template.swift — clean
- [low] swift/Sources/MCPZimKit/QwenChatMLTemplate.swift:73 — `formatSystemTurn` re-serializes every tool declaration via `toolJSONLine` (`JSONSerialization.data(withJSONObject:options:[.sortedKeys])`) on every `renderTranscript`/`formatSystemTurn` call — the tool list is immutable after init and `ChatSession` renders the transcript multiple times per turn (prefill, rewind/retry, generation), so each LLM turn re-encodes ~10 tool JSON blobs that never change, paying dict construction + key-sorting + Data allocation per tool per call — smallest safe fix: cache the rendered `<tools>` block (or the concatenated `toolJSONLine` string) keyed by the immutable tools array, and splice it into the system turn instead of re-serializing.
## Coverage
swift/Sources/MCPZimKit/MCPToolAdapter.swift — clean
swift/Sources/MCPZimKit/MemoryProbe.swift — clean
swift/Sources/MCPZimKit/MemoryStats.swift — clean
swift/Sources/MCPZimKit/ModelTemplate.swift — clean
swift/Sources/MCPZimKit/PlacesPayload.swift — clean
swift/Sources/MCPZimKit/QueryComplexity.swift — clean
swift/Sources/MCPZimKit/QwenChatMLTemplate.swift — findings: 1
- [medium] swift/Sources/MCPZimKit/SZRGGraph.swift:432 — `nearestNode(lat:lon:)` is a linear O(numNodes) scan over the full `lat`/`lon` arrays, and `ZimService.planDrivingRoute` calls it twice per request (origin + goal) on the monolithic-graph path — concrete consequence: on a country/continent-scale graph with millions of nodes, every routing request pays ~2 full Float64 scans (tens of ms, growing linearly with graph size) before A* even starts — smallest safe fix: build a k-d tree / spatial grid over node coordinates once at parse time and query it, or at minimum cache the nearest-node result per coordinate (the code comment itself acknowledges the k-d tree swap).
- [medium] swift/Sources/MCPZimKit/Router.swift:201 — `nearestNodeSpatial(index:lat:lon:)` is a linear scan over `index.nodesScaled` (2×numNodes entries), called twice per `planDrivingRoute` on the spatial (SZCI/SZRC) path — concrete consequence: for continent-scale v2 spatial graphs whose sharded node table is tens of millions of entries, each route request burns two O(numNodes) scans of Int64→Double math before A* begins, adding latency that scales with total node count — smallest safe fix: same as above — index the eager node table (grid/hash/k-d) once at parse so nearest-node lookup is O(1)/O(log n) instead of O(n).
- [low] swift/Sources/MCPZimKit/Router.swift:338 — the `aStarSpatial` reconstruction loop awaits `graph.decodeGeomForEdge(sourceNode:geomLocal:)` once per edge of the final path; that method is actor-isolated on `SpatialGraph`, so each iteration is a separate actor hop plus a repeated `cellForNode` dictionary lookup + `ensureCell` cache check — concrete consequence: for a long route (thousands of edges) reconstruction pays one suspension/actor round-trip per edge, latency proportional to path edge count (the search loop already memoizes the current cell, but reconstruction does not) — smallest safe fix: memoize the last cell across reconstruction iterations (as the pop loop already does) or add a batch decode that resolves all edges' cells in one pass.
- [low] swift/Sources/MCPZimKit/SZRGGraph.swift:290 — when geometry decoding is skipped (`decodeGeoms=false`, the mcpzim default for v5, or no SZGM companion), the parser still materializes one empty `[(lat,lon)]` placeholder per geometry via `Array(repeating: [], count: numGeoms)` — concrete consequence: at the v2 geometry cap (2^24 ≈ 16.78M geoms on continental graphs) this is ~134 MB of empty-array bookkeeping on a path whose entire purpose is to save memory, partially undercutting the ~600 MB the comment claims it saves — smallest safe fix: store geoms lazily (e.g. a `[Int: [(Double,Double)]]` of only decoded indices, or an optional backing buffer) so the skip path allocates nothing per geom.
## Coverage
swift/Sources/MCPZimKit/ReferenceResolver.swift — clean
swift/Sources/MCPZimKit/RegexCache.swift — clean
swift/Sources/MCPZimKit/Router.swift — findings: 2
swift/Sources/MCPZimKit/StreamingSpeechPolicy.swift — clean
swift/Sources/MCPZimKit/StubZimService.swift — clean
swift/Sources/MCPZimKit/SZRGChunked.swift — clean
swift/Sources/MCPZimKit/SZRGEncoder.swift — clean
swift/Sources/MCPZimKit/SZRGGraph.swift — findings: 2
swift/Sources/MCPZimKit/SZRGSpatial.swift — clean
swift/Sources/MCPZimKit/ToolLoopGuard.swift — clean
swift/Sources/MCPZimKit/ZimReader.swift — clean
- [medium] swift/Sources/MCPZimKit/ZimService.swift:1258 — `scanRecords` computes a full haversine (2×sin, 2×cos, asin, sqrt per record via `haversineMeters`) before the radius check, with no cheap lat/lon bounding-box pre-filter — on the `nearPlaces` hot path a chip/category load or a full search-data scan iterates up to `maxFullScanRecords` (500k) records and pays ~6 transcendental calls per record even when the radius is a few km and >99% of records fall outside; this adds seconds of CPU and drains battery per query on a country-scale ZIM (the very cost the code's OOM-guard comments acknowledge at ~4 s scans) — smallest safe fix: before calling `haversineMeters`, reject with `abs(rlat - centerLat) > dLat || abs(rlon - centerLon) > dLon` using `dLat = radiusMeters/111_320` and `dLon = radiusMeters/(111_320*cos(centerLat*Double.pi/180))`, then only compute the accurate distance for survivors.
- [low] swift/Sources/MCPZimKit/ZimService.swift:372 — `keywordCore` rebuilds a 60-entry `Set<String>` stopword list on every call (the `let stop: Set<String> = [...]` literal is a runtime allocation, not a compile-time constant), and it runs once per `search` plus once inside `queryVariants` — every search query allocates and populates a 60-element hash set twice for a constant that never changes — smallest safe fix: hoist to `private static let stopwords: Set<String>` and reference it.

## Coverage
swift/Sources/MCPZimKit/ZimService.swift — findings: 2
swift/Tests/MCPZimKitTests/AnswerAttributionTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleFactoidTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleHeuristicsCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/ArticleSpeechCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/BundledArticleTests.swift — clean
swift/Tests/MCPZimKitTests/ChatToolCallParserTests.swift — clean
swift/Tests/MCPZimKitTests/ClarificationAndTitleCleanupTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationContinuationTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationFocusTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationThreadsTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussArticleLinkTests.swift — clean
# Performance review — batch 15

No performance findings. All files in this batch are XCTest test files exercising MCPZimKit's heuristics, geocoding, prompt templating, and payload parsing. Loops iterate small fixed test-case arrays (8–20 string literals) or bounded in-memory fixtures (largest: 16×16 = 256 JSON leaves in HotSplitGeocoderTests, built once per test). No N+1 queries, unbounded caches, per-item I/O over unbounded data, or allocation churn on hot paths. Test fixtures are bounded and out of scope per the performance-review checklist.

## Coverage
swift/Tests/MCPZimKitTests/DiscussionRetrievalTests.swift — clean
swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean
swift/Tests/MCPZimKitTests/EmbeddingsTests.swift — clean
swift/Tests/MCPZimKitTests/Gemma4PromptTemplateTests.swift — clean
swift/Tests/MCPZimKitTests/Gemma4ToolFormatTests.swift — clean
swift/Tests/MCPZimKitTests/GeocodeVariantsTests.swift — clean
swift/Tests/MCPZimKitTests/HotSplitGeocoderTests.swift — clean
swift/Tests/MCPZimKitTests/IntentRouterTests.swift — clean
swift/Tests/MCPZimKitTests/LFM25TemplateTests.swift — clean
swift/Tests/MCPZimKitTests/LocateToolTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesCenterResolutionTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesChipIndexTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesKindFallbackTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesWikiEnrichmentTests.swift — clean
swift/Tests/MCPZimKitTests/PlacesPayloadTests.swift — clean
swift/Tests/MCPZimKitTests/QueryComplexityTests.swift — clean
- [low] tools/fine-tune/eval_ft_pcgaming.sh:72 — the nested `for m`/`for s` loop invokes `eval.py` as a fresh subprocess once per (model × scenario) cell, and each invocation re-loads a multi-GB Q4_K_M GGUF from disk into RAM before inferring — running the full 4-model × 13-scenario grid costs 52 model loads, so wall-clock is dominated by reloads (each ~seconds-to-minutes of disk read + buffer setup for a 1B–4B quant) rather than inference — group scenarios under a single model load per process (pass multiple `--scenario`s or add a loop inside eval.py) so each model is loaded 1× instead of 13×.
- [low] tools/fine-tune/generate_chains.py:236 — `out_fh.flush()` runs after every written chain inside the concurrent worker, forcing a disk write syscall per record despite the file being opened in default buffered mode — at larger `--n` (thousands of chains) the per-record flush adds latency and I/O churn to the generation pipeline (and each flush also publishes a partial line for the resume path to re-count) — flush every N records (e.g. every 5, matching the existing progress cadence) or once at the end of `run()`, keeping the buffered writes between checkpoints.

## Coverage
swift/Tests/MCPZimKitTests/QwenClippedToolCallTests.swift — clean
swift/Tests/MCPZimKitTests/ReferenceResolverTests.swift — clean
swift/Tests/MCPZimKitTests/RouterBenchTests.swift — clean
swift/Tests/MCPZimKitTests/SanitizedWikiTagTests.swift — clean
swift/Tests/MCPZimKitTests/SanitizeZimArgTests.swift — clean
swift/Tests/MCPZimKitTests/StreamingSpeechPolicyTests.swift — clean
swift/Tests/MCPZimKitTests/SZRGGraphTests.swift — clean
swift/Tests/MCPZimKitTests/SZRGSpatialTests.swift — clean
swift/Tests/MCPZimKitTests/SZRGv5AndChunkedTests.swift — clean
swift/Tests/MCPZimKitTests/ToolLoopGuardTests.swift — clean
tools/bonsai-ab/compare.sh — clean
tools/fine-tune/convert_to_lfm2_native.py — clean
tools/fine-tune/eval_ft_pcgaming.sh — findings: 1
tools/fine-tune/finetune_cuda.py — clean
tools/fine-tune/finetune_cuda.sh — clean
tools/fine-tune/finetune_lfm2.sh — clean
tools/fine-tune/finetune_unsloth.py — clean
tools/fine-tune/finetune_unsloth.sh — clean
tools/fine-tune/finetune.sh — clean
tools/fine-tune/generate_chains.py — findings: 1
No defensible performance findings in this batch.

This batch is offline training-data generation and smoke-test tooling. Runtime is dominated by GPU model inference (~1–2s per row) and model load (tens of seconds); the Python/Shell/Swift hot paths are all bounded by user-specified `--n` (defaults 100–800) or fixed `maxTokens` (60–200). No DB access, no caches/registries, no request handlers, no unbounded collections exist in any listed file.

Dismissed candidates (each fails the "name the growing input / concrete consequence" bar):
- `out_fh.flush()` per row in generate.py / generate_places_diverse.py / generate_chains3.py — deliberate crash-resume durability; a flush syscall is ~6 orders of magnitude below the per-row inference cost and the skill's per-row-commit false-positive (resume logic present) applies.
- `tokenizer.decode(tokens: ids.map { Int($0) })` re-decoding the whole accumulated token list on every generated token in PromptExperiment.swift (runBehaviorTest, CacheSim.runGenerate, simulateRunInSinglePerform.runOne) — genuine O(n²), but n is capped at maxTokens=60–120 in every call site, so no measurable consequence.
- Whole-seed-list + whole-coroutine-list materialization (`asyncio.gather(*[...])`, `create_task` for all targets) — bounded by user `--n`; semaphore already bounds concurrent work.
- `_eval_preamble()`/`_build_tool_block()` rebuilt once per row — a ~KB constant string rebuilt against seconds of inference; negligible.
- Resume line-count (`sum(1 for _ in fh)`) reading the whole output file at startup — one-time O(file) cold-path cost, necessary for the resumable design.
- `rows = [l for l in open(src) ...]` in retry_lfm2_train.sh — full-file load is inherent to `random.shuffle` of the 95/5 split.

## Coverage
tools/fine-tune/generate_chains3.py — clean
tools/fine-tune/generate_places_diverse.py — clean
tools/fine-tune/generate.py — clean
tools/fine-tune/retry_lfm2_train.sh — clean
tools/fine-tune/split_chain_rows.py — clean
tools/fine-tune/train_all_cuda.sh — clean
tools/fine-tune/train_all.sh — clean
tools/fine-tune/v7_eval_and_memsweep.sh — clean
tools/gemma-smoke/Package.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/main.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — clean
tools/llama-smoke/bench.py — clean
- [medium] tools/llm-smoke/bench_memory.py:142 — `run()` calls `load(model_id)` on every (model, variant, size) combo instead of once per model — the same multi-GB weights are re-loaded repeatedly: `mlx-community/gemma-3-4b-it-4bit` appears twice in CONFIGS["gemma3"] (default + bounded_512) × 2 sizes = 4 full loads, `gemma-4-e2b-it-4bit` 2 variants × 2 sizes = 4 loads, Qwen3-4B 2 variants × 2 sizes = 4 loads — each `load()` is seconds-to-tens-of-seconds of weight I/O and Metal allocation, so bench wall time is inflated 2–4× per model — smallest safe fix: hoist `load(model_id)` out of `run()` into a per-model_id cache (as bench_kv.py:163 already does), iterating variants/sizes under a single load.
- [medium] tools/llm-smoke/bench_memory_gemma4.py:84 — same pattern: `run()` calls `load(model_id)` for every (model × variant × size) triple in main (2 models × 2 variants × 3 sizes = 12 loads where 2 unique models exist) — 6× redundant full model loads per model multiply bench wall time and GPU/CPU allocation churn — smallest safe fix: load each unique model_id once in main and pass `model, processor` into `run()`.
- [low] tools/llama-smoke/grid.py:491 — the grid opens the scorecard with `open(out_path, "w")` and never reads it back, so there is no resume/checkpoint: an interrupted or re-run grid re-executes every already-scored (model, quant, KV, scenario) combo from scratch, each up to 600 s, on a matrix of 30+ models × quants × 3 KV × 16 scenarios — smallest safe fix: parse the existing out file at startup and skip combos whose row is already present (or append per-combo lines to a sidecar state file).
- [low] tools/logpipe/ingest.sh:68 — a fresh `python3 parse_log.py` subprocess is spawned per log file (per-item subprocess; fork+exec plus Python startup ~50–100 ms each), and the corpus is fully scanned twice per run (line 60 builds `seen`, line 83 recounts `total`), both O(corpus) as conversations.jsonl grows monotonically with every ingest — for a bulk `--from` export of hundreds of logs this adds seconds-to-minutes of process churn and redundant JSON parsing — smallest safe fix: import `parse_log.parse` in-process (the embedded `python3 -` already isolates the ingest), and track `len(new_convs)`/append count instead of re-reading the corpus for the total.
- [nit] tools/llama-smoke/eval.py:181 — `name_pool = BAR_NAMES + [f"Bar {i}", ...]` is rebuilt on every iteration of the fixture loop, re-copying the 25 constant BAR_NAMES strings each of n iterations (n bounded at 500–1000 by the fixture, so impact is negligible today) — smallest safe fix: hoist BAR_NAMES to a module-level tuple and append only the six i-dependent names per iteration.

## Coverage
tools/llama-smoke/eval.py — findings: 1
tools/llama-smoke/grid.py — findings: 1
tools/llama-smoke/sweep.sh — clean
tools/llm-smoke/bench_kv.py — clean
tools/llm-smoke/bench_memory_gemma4.py — findings: 1
tools/llm-smoke/bench_memory.py — findings: 1
tools/llm-smoke/bench.py — clean
tools/llm-smoke/eval_gemma4_native.py — clean
tools/llm-smoke/eval_gemma4.py — clean
tools/llm-smoke/eval.py — clean
tools/llm-smoke/gemma4_format.py — clean
tools/logpipe/ingest.sh — findings: 1
tools/logpipe/parse_log.py — clean
tools/logpipe/prep_judge.py — clean
tools/logpipe/report.py — clean
# Batch 19 — performance review

Scope: five test files. Each was read in full and cross-checked against the modules they import (`mcpzim.content`, `mcpzim.geocode`, `mcpzim.library`, `mcpzim.routing`) and against the performance-review and python-performance-review checklists.

All files are test code that runs once in CI on small, bounded inputs: fixture strings of a few hundred characters, dicts of a handful of keys, 4-node graphs, and 2–3 point polylines. There are no loops over unbounded data, no per-item I/O or network round-trips, no queries, no caches/registries that grow at runtime, no async handlers, and no pandas/numpy usage in any of the listed files. Module imports are lazy (bs4/libzim are imported inside functions in the modules under test), so test collection does not trigger heavy imports. The only loops in the batch are over fixed enumerations (parametrize lists of ≤13 ints, a 2-iteration geometry loop in `test_geom_decoder_roundtrip`, a 4-edge adjacency check), which are bounded by construction. No defensible performance finding is exhibited by the actual code in this batch.

## Coverage
tests/__init__.py — clean
tests/test_content.py — clean
tests/test_geocode.py — clean
tests/test_library.py — clean
tests/test_routing.py — clean
- [medium] swift/Sources/MCPZimKit/MCPToolAdapter.swift:984 — `dispatchArticleOverview` probes candidate articles with serial whole-article reads: the disambiguation loop (lines 976–996, up to 6 `WikiLinks.parseAll` links × 2 path forms = ≤12 `service.articleSections` calls) and the sibling thin-stub loop (lines 1000–1023, ≤12 search hits) each `try? await service.articleSections(...)` one at a time, where every call is a full ZIM decompress + HTML section parse of a distinct article — when an ambiguous query ("Apple TV", "Tesla", "gravity waves") resolves to a disambiguation page or a thin stub, `article_overview` pays the *sum* of N full-article parses (hundreds of ms on-device) instead of the max before it finds the first substantive candidate — mirror the file's own `withTaskGroup` fan-out (compare_articles:1532, buildNearbyStories:1811) and fetch the ≤6/≤12 candidate sections concurrently, then pick the first substantial one in display order to preserve the "first link wins" semantics.

## Coverage
swift/Sources/MCPZimKit/MCPToolAdapter.swift — findings: 1
- [low] swift/Sources/MCPZimKit/IntentRouter.swift:623 — Direct `String.replacingOccurrences(of:options:.regularExpression)` / `String.range(of:options:.regularExpression)` calls recompile their `NSRegularExpression` on every invocation, bypassing the shared `RegexCache` that the `match()`/`matches()` helpers (lines 1408/1419) already use — several of these sit on the unconditional per-turn `classify()` path (`wikipediaSourceDirective` whitespace-collapse line 623 and its two `lower.range(...)` probes, `readArticleIntent` line 763, plus the conditional ones at 1185/1189/1204/1215/1228/1250/1396), so every user message pays ~4–15 fresh regex compilations (~tens of µs each, ≈0.2–1.5 ms/turn) before the LLM fast-path runs — route these literal patterns through `RegexCache.shared.compiled` (or hoist to `static let` `NSRegularExpression` constants), mirroring the existing `match`/`matches` helpers.

## Coverage
swift/Sources/MCPZimKit/IntentRouter.swift — findings: 1
ios/MCPZimChat/Providers/LlamaCppProvider.swift — clean
- [medium] ios/MCPZimChat/Views/ChatView.swift:723 — `MessageRow.displayText` re-runs the full multi-NSRegularExpression strip pipeline (`closedBlockRegexes`/`strayOpenerRegexes` + range scans in `computeDisplayText`) over the *entire* accumulated message text on the main actor at every streaming push (~10 Hz). The memo at :727 is keyed on `(hash, count)`, so while the reply streams every push appends text → new count → guaranteed cache miss → `computeDisplayText` rescans the whole growing string. `showThinkingIndicator` (:232) and the assistant row body (:499) both call it per push. — Main-thread O(n) regex work per push, O(n²) bytes scanned over the lifetime of a long reply; for the `narrate_article`/`article_overview` paths that stream whole Wikipedia articles (tens of KB), this is sustained regex matching on the UI thread at 10 Hz and can cause dropped frames / jank mid-generation. — Strip only the newly appended tail incrementally (track last processed length/UTF16 offset and run the stray-opener/`<think>` logic on the suffix), or throttle the strip to a lower cadence instead of every push.

## Coverage
swift/Tests/MCPZimKitTests/IntentRouterTests.swift — clean
ios/MCPZimChat/Views/ChatView.swift — findings: 1
ios/MCPZimEval/ProbeE2ECLI.swift — clean
- [low] tools/fine-tune/generate.py:683 — `_build_tool_block()` re-renders the constant tool-schema block on every successful row (`trajectory_to_jsonl` → `_eval_preamble()` runs per row in the worker loop), iterating the full `TOOLS_SCHEMA` and rebuilding the same ~1-2KB string each time — redundant recomputation scaled by `--n` (800 rows → 800 identical rebuilds of a block that never changes), negligible wall time but trivially avoidable — hoist `SYSTEM_PREAMBLE + "\n" + _build_tool_block()` to a module-level constant (eval.py already memoizes the equivalent `_TOOLS_SCHEMA_JSON` in its LFM2 path).
- [low] tools/fine-tune/generate_chains3.py:571 — same redundant recomputation: `_eval_preamble()` calls `_build_tool_block()` on every row render (`chain_to_messages`/`narrate_to_messages` are invoked per row in `emit_one`) — the constant tool block is rebuilt once per generated trajectory instead of once per run — hoist the `SYSTEM_PREAMBLE + _build_tool_block()` prefix to a module-level constant and only vary the optional `currentLocation` suffix per row.
- [medium] tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift:533 — inside the autoregressive decode loop (`streamLoop` in `simulateRunInSinglePerform`'s `runOne`), every token step calls `context.tokenizer.decode(tokens: tokenIDs.map { Int($0) })`, re-decoding the entire accumulated sequence and allocating a fresh `[Int]` per step — O(n²) tokenize/decode CPU per generation (maxTokens 60/120 ⇒ ~1800/~7260 redundant token decodes per generate), adding host-side CPU between GPU steps and lengthening the wall-time comparison — decode only the newly produced token (or the last few) and append, as the mlx-swift examples do.
- [medium] tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift:259 — identical per-token full-sequence re-decode (`context.tokenizer.decode(tokens: ids.map { Int($0) })`) in `runBehaviorTest`'s decode loop, plus a fresh `ids.map { Int($0) }` allocation each step — O(n²) decode CPU per 60-token behavior-test generation — decode incrementally (feed only the new token) instead of re-decoding the whole accumulated `ids` array every iteration.
- [low] tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift:662 — same O(n²) per-token full-sequence re-decode in `CacheSim.runGenerate` (`tokenizer.decode(tokens: tokenIDs.map { Int($0) })`); the class is currently unused (kept for a possible MLX behavior change) but still compiled — if it is ever revived, apply the same incremental-decode fix or drop the dead code.

## Coverage
tools/fine-tune/generate_chains3.py — findings: 1
tools/fine-tune/generate.py — findings: 1
swift/Sources/MCPZimKit/ReferenceResolver.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — findings: 3
- [medium] swift/Sources/MCPZimKit/ConversationThreads.swift:619 — `WikiLinks.parseLinks` calls `re.matches(in: source, range: range)`, which materializes an `[NSTextCheckingResult]` for EVERY `<a>` anchor in the full article HTML before the `for` loop runs, even though the loop `break`s after `max` (8) links — for a Kiwix Wikipedia article (100s of KB, thousands of anchors) the whole document is scanned and one result object + NSRange is allocated per anchor, all but 8 of which are discarded, so drift extraction latency and allocation scale with article HTML size on every conversation turn that falls back to HTML parsing (no pre-attached `links`/`related` array) — smallest safe fix: use `re.enumerateMatches(in: options: range: using:)` and set `stop.pointee = true` once `out.count >= max`, which stops the scan at the 8th link and avoids the full result array (same for the `proseParagraphs` accumulation at line 662 if the full-match array is not needed).

- [low] ios/MCPZimChat/Views/LibraryView.swift:25 — `TimelineView(.periodic(from: .now, by: 1.0))` re-evaluates its body once per second for as long as the Library screen is on screen, regardless of `session.modelState` (the per-second "Ns elapsed" display is only needed during `.downloading`) — a perpetual 1Hz SwiftUI body re-evaluation/`modelStateDescription` recomputation on an otherwise static settings screen, wasting CPU/energy on the idle path — smallest safe fix: only wrap the status row in `TimelineView` when `modelState` is `.downloading`/`.loading` (or gate the elapsed text so the periodic schedule stops when not downloading).

## Coverage
swift/Sources/MCPZimKit/ConversationThreads.swift — findings: 1
ios/MCPZimChat/Views/LibraryView.swift — findings: 1
swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean
swift/Sources/MCPZimKit/SZRGSpatial.swift — clean
- [medium] mcpzim/routing.py:103 — Graph.parse stores the routing arrays (lat/lon at :103-104, edge_targets/edge_dist_m/edge_speed_kmh/edge_geom_idx/edge_name_idx at :111-120, geoms at :132) as `list[float]`/`list[int]`/`list[tuple[float,float]]` of boxed Python objects. The docstring states city/country graphs "run to millions of nodes" and the graph is "kept entirely in memory"; the A* inner loop (routing.py:371-384) reads `speeds[e]`, `dists[e]`, `targets[e]` on every edge relaxation of every `plan_route` request (server.py:163/218 → plan_route → astar). Boxed scalars cost 24-32 B each vs 8 B in `array('d')`/`array('I')`, so a multi-million-node graph costs several GB RSS (lat+lon alone ≈ 1 GB per 20M nodes before edge/geom arrays) and the boxed lookups destroy cache locality on the hottest path. Consequence: multi-GB memory per cached streetzim graph (OOM risk in the server process) plus slower per-request A* as graphs grow. Smallest safe fix: use `array('d')`/`array('I')` (or numpy) for the parallel node/edge arrays instead of Python lists, keeping the same CSR indexing.
## Coverage
ios/MCPZimChatMacTests/ConversationalEvalTests.swift — clean
tools/llm-smoke/eval.py — clean
mcpzim/routing.py — findings: 1
swift/Tests/MCPZimKitTests/SZRGSpatialTests.swift — clean
# Batch 26 — performance findings

## Summary
Reviewed 4 files (2 Swift test files, the Gemma3 ModelTemplate, and the XcodeGen project spec) under the general performance checklist plus ios-performance and swift review lenses. No defensible performance findings. The only hot path in the batch — `Gemma3Template.firstToolCall(in:)` called per streaming chunk — operates on a generated buffer bounded by `maxTokens` (~512 tokens / few KB), so the per-chunk `range(of:)` prefix rescan is constant-factor and negligible; `repairJSON` is invoked only on the post-decode-failure path (never mid-stream, since `findCall` early-returns when no close marker exists in strict mode) and its regexes are cached. Swift `String +=` accumulation is amortized O(1) (growable COW storage), so the transcript/tool-block building is not an O(n²) string-concat bug.

## Findings

(none)

## Coverage
swift/Tests/MCPZimKitTests/ArticleFactoidTests.swift — clean
swift/Tests/MCPZimKitTests/ConversationContinuationTests.swift — clean
swift/Sources/MCPZimKit/Gemma3Template.swift — clean
ios/project.yml — clean
# Performance review — batch 27

## Findings

No defensible performance findings. Both files are non-hot paths (an XCTest suite and an offline LLM data-generation batch job); all candidate patterns are bounded, deliberate, or dominated by network latency.

## Coverage
swift/Tests/MCPZimKitTests/ConversationThreadsTests.swift — clean
tools/fine-tune/generate_places_diverse.py — clean

## Run stats

input 1525470 tok (+12520448 cached), output 315731 tok, cost $0.98 — 226 files in 22m (593.9 files/h, 0.8 min/batch)
