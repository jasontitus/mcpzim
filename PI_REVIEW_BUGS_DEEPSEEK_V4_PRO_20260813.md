# Pi sweep review — mcpzim-0aad28ab

Exhaustive per-file pass: 202 code files across 25 batches.

## Findings

- [low] ios/MCPZimChat/AppIntents/LocationFetcher.swift:218 — race between the cached-fix snapshot and waiter registration: `latestOrWait` checks `latest` inside `await MainActor.run { ... }` (line 218) and only later appends the waiter via a separate `Task { @MainActor in self?.waiters.append(...) }` (line 238); a fix (or a `.denied` auth transition) that lands in that window drains an empty `waiters`, sets `latest`, and the just-registered waiter misses it — `once()` then stalls to the full 15s timeout even though a fresh fix is available (e.g. a stationary user whose single first fix is missed), causing spurious `LocationError.timeout` in intents/turn pipelines — smallest safe fix: perform the cache check and waiter enqueue in one main-actor hop (make `latestOrWait` `@MainActor` and append the waiter synchronously) instead of a deferred `Task`.
- [low] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:59 — `ZimfoRunner` is `@MainActor`, so `buildFresh()` runs on the main actor and synchronously opens every ZIM archive via `LibzimReader(url:)` (lines 69 and 91) plus `MCPToolAdapter.from(...)`; the first `ZimfoRunner.load()` per process blocks the main actor for seconds (multi-GB ZIM metadata/title/fulltext index reads) — in an App Intent this can exceed Siri's execution-time budget and in the main app it freezes the UI — smallest safe fix: move reader construction off the main actor (a `nonisolated`/`Task.detached` builder) and only hand the finished runner back to the main-actor cache.
- [low] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:90 — `startAccessingSecurityScopedResource()` is never balanced with `stopAccessingSecurityScopedResource()`; if `LibzimReader(url:)` returns nil the security-scoped access is leaked, and on success the scope is held for the whole process lifetime — consequence: leaked sandbox file-access scope and a stale handle on replaced bookmarks — smallest safe fix: `defer { url.stopAccessingSecurityScopedResource() }` after a successful start (or at least stop on the failure path).
- [low] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:43 — `libraryFingerprint()` keys the memoized runner only on ZIM filenames and `Data.hashValue` of bookmark blobs, not file content/size/mtime; replacing a `.zim` file with a new version under the same filename is not detected, so `load()` keeps returning the cached runner with stale readers for the lifetime of the process — consequence: Siri intents serve stale offline search/route data until the process exits — smallest safe fix: include file size + modification date (or an inode/creation-date tuple) in the fingerprint.
- [low] ios/MCPZimChat/Chat/AppTelemetry.swift:71 — the iOS branch calls `FirebaseApp.configure()` with no check that `GoogleService-Info.plist` exists, while the macOS branch (lines 62–69) explicitly guards and degrades gracefully — consequence: an iOS build with a missing plist crashes at launch via an uncaught `FIRApp` exception — smallest safe fix: mirror the macOS guard on iOS (`Bundle.main.path(forResource: "GoogleService-Info", ofType: "plist")` check before `configure()`).
- [nit] ios/MCPZimChat/AppIntents/ZimfoIntents.swift:25 — `formatDuration` pluralizes singular units when both hours and minutes are present (`"1 hours 1 minutes"` for 90 minutes) — consequence: Siri speaks grammatically wrong durations — smallest safe fix: pluralize each unit independently (`1 hour 1 minute`).

## Coverage
eval/run_prepared_discussion_ab.sh — clean
ios/MCPZimChat/App/MCPZimChatApp.swift — clean
ios/MCPZimChat/AppIntents/LocationFetcher.swift — findings: 1
ios/MCPZimChat/AppIntents/ZimfoContext.swift — clean
ios/MCPZimChat/AppIntents/ZimfoIntents.swift — findings: 1
ios/MCPZimChat/AppIntents/ZimfoRunner.swift — findings: 3
ios/MCPZimChat/Chat/AppTelemetry.swift — findings: 1
- [low] ios/MCPZimChat/Chat/ChatSession.swift:623 — `runSetupIfNeeded()` busy-waits in `modelWait: while true` with no stall timeout, only exiting on `.ready` or `.failed` — a hung download (bytes stop flowing but the provider never surfaces a failure) leaves `setupState == .running` forever, and `send()` guards on `setupState != .ready` so the composer is permanently blocked; `dismissSetupFailure()` only handles `.failed`, so the user has no escape — smallest safe fix: track a wall-clock stall deadline (e.g. reset whenever `fraction` advances) and transition to `.failed` (or `.ready` with a warning) when no progress is observed for a bounded interval, so the user can reach Settings/retry.

- [low] ios/MCPZimChat/Chat/ChatSession.swift:4094 — `triggerArticleRead` (and `triggerDirectionsToCoord` at line 4136) guard only `setupState == .ready` and never `!isGenerating` — if a pin-popup "Read article"/"Directions" button on an earlier message is tapped while a reply is still streaming, these append a new `user` + empty `assistant` message and start a second Task while the in-flight `generationTask` keeps calling `appendToAssistant`, which writes to `messages[messages.count - 1]` — now the trigger's placeholder, not the streaming reply — corrupting the transcript and racing `isGenerating`/`finishedAt` between two Tasks — smallest safe fix: `guard !isGenerating else { return }` at the top of both public entry points (mirroring `send()`).

- [nit] ios/MCPZimChat/Chat/ChatSession.swift:6370 — `formatDuration` computes `let m = (total % 3600 + 30) / 60` without carrying the rounding overflow into hours, so any duration in the last minute before an hour boundary renders wrong: e.g. 7199s → `h=1, m=60` → "1h 60m" (and 3599s → "60m") instead of "2h"/"1h" — concrete consequence: routing replies near an hour boundary echo a nonsensical "1h 60m" duration string to the user — smallest safe fix: add minutes first and re-derive hours, e.g. `let m = (total + 30) / 60; let h = m / 60; let mm = m % 60`.

- [nit] ios/MCPZimChat/Chat/ChatSession.swift:1862 — `NotificationCenter.default.addObserver(forName:object:queue:using:)` return tokens for the memory-warning (1862) and didEnterBackground (1902) observers are discarded and never removed in `deinit` (which already cleans up `locationSubscription`) — in the eval harness that constructs many sessions per process, every session leaves two permanently-registered observers that keep firing on each notification for the rest of the process lifetime — smallest safe fix: store the two observer tokens and call `NotificationCenter.default.removeObserver(_:)` on them in `deinit`.

## Coverage
ios/MCPZimChat/Chat/ChatSession.swift — findings: 4
- [medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:530 — `generate()` never wires `continuation.onTermination` and the chunk loop has no `Task.checkCancellation()`, so when the consumer cancels/abandons the stream (user stop tap, conversation reset, model switch) the MLX `generateTokens` task keeps running to maxTokens or a stop marker — background GPU/memory churn racing the next `generate()` and yields buffered into a dead stream. FoundationModelsProvider.generate() wires `onTermination = { task.cancel() }`; this provider omits it. — smallest safe fix: add `continuation.onTermination = { @Sendable _ in task.cancel() }` and `try Task.checkCancellation()` inside the `chunkLoop` for-await, mirroring FoundationModelsProvider.

- [medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:165 — `promptKVCache`/`cachedTokens`/`generatedTokensThisTurn` are plain mutable vars on an `@unchecked Sendable` class, written from `generate()`/`primeCache()` (inside `container.perform` and on the Task executor) while `resetPromptCache()` (185) and `hasPromptKVCache` (173) read/write them from arbitrary threads (memory-warning/backgrounded notifications arrive on main) with no lock or actor — concurrent Array reassignment/append is a data race (UB / torn reads, worst case crash or a corrupted cache-reuse decision). — smallest safe fix: guard all four accessors with a single lock (or move the cache state into an actor) and route `resetPromptCache`/`hasPromptKVCache` through it.

- [low] ios/MCPZimChat/Common/SemanticReranker.swift:49 — `loadIfNeeded()` sets `loadAttempted = true` before the asset download/load, then returns early on any failure, so a transient `requestAssets()` network error (or a one-off `load()` failure) permanently disables semantic reranking for the process lifetime even after assets become available. — concrete consequence: after one flaky download the reranker silently degrades to BM25 until the app is restarted. — smallest safe fix: only set `loadAttempted = true` on success, or reset it to `false` in the failure paths so the next call retries.

- [medium] ios/MCPZimChat/Providers/FoundationModelsNativeTools.swift:63 — the native-tool wrappers pass model-supplied `limit`/`radiusKm` to `ZimService` with only a `> 0` floor and no upper bound (`nearNamedPlace` at 63–64, `search` at 175–176), whereas the text-loop `MCPToolAdapter` clamps these (`limit` to 1…50, radius to 0.05…100 at MCPToolAdapter.swift:699-700/511). A malformed or prompt-injected tool argument therefore bypasses the clamp: `ZimService.search` computes `limit * 2` for overfetch (integer-overflow trap / huge allocation) and `nearPlaces` computes `radiusKm * 1000` (full-ZIM spatial scan → OOM/jetsam/hang), and `LibzimReader.searchFullText` does `Int32(limit)` (trap past Int32.max). — smallest safe fix: clamp in each native tool call exactly as MCPToolAdapter does, e.g. `min(50, max(1, arguments.limit))` and `min(100, max(0.05, arguments.radiusKm))`.

- [low] ios/MCPZimChat/Chat/DebugReport.swift:54 — the GitHub gist PAT is stored/read via `UserDefaults.standard` (plaintext), which is included in unencrypted device/iCloud backups, instead of the Keychain. — concrete consequence: a gist-scoped PAT exfiltrates with any device backup of a DEBUG build where the developer pasted it. — smallest safe fix: store the token in the Keychain (e.g. `SecItemAdd`/`SecItemCopyMatching`) rather than UserDefaults. (DEBUG-gated and personal-dev, hence low, not critical.)

## Coverage
ios/MCPZimChat/Chat/DebugReport.swift — findings: 1
ios/MCPZimChat/Chat/Message.swift — clean
ios/MCPZimChat/Common/DeviceProfile.swift — clean
ios/MCPZimChat/Common/DiagnosticsUploader.swift — clean
ios/MCPZimChat/Common/LogArchive.swift — clean
ios/MCPZimChat/Common/SemanticReranker.swift — findings: 1
ios/MCPZimChat/Common/ZimfoContext+Adapter.swift — clean
ios/MCPZimChat/Libzim/LibzimBridge.h — clean
ios/MCPZimChat/Libzim/LibzimBridge.mm — clean
ios/MCPZimChat/Libzim/LibzimReader.swift — clean
ios/MCPZimChat/Providers/FoundationModelsNativeTools.swift — findings: 1
ios/MCPZimChat/Providers/FoundationModelsProvider.swift — clean
ios/MCPZimChat/Providers/Gemma4Provider.swift — findings: 2
# Pi sweep — batch 4

## Findings
- [medium] ios/MCPZimChat/Providers/LlamaCppProvider.swift:598 — on the `llama_init_from_model` failure path, `llama_model_free(m)` frees the model that owns the vocab captured at line 577, but `self.vocab` is left dangling (only `self.model` is reset) — a subsequent `promptTokenCount()` passes the freed vocab pointer to `llama_tokenize` (use-after-free crash), which is plausible on iOS where context init OOMs on a ~3 GB model and the session later probes/compacts — set `self.vocab = nil` (and `self.ctx` already nil) before `llama_model_free(m)` on the failure branch.
- [low] ios/MCPZimChat/Providers/LlamaCppProvider.swift:1201 — `llama_token_to_piece` writes into a fixed 64-byte buffer and its negative "buffer too small" return is silently ignored by the `if n > 0` guard — any token whose piece exceeds 63 bytes is dropped from the emitted text (while still being fed back into KV), producing missing characters in the reply — detect `n < 0` and retry with the reported required size (or grow `pieceBuf` to the required length before the call).
- [medium] ios/MCPZimChat/Sharing/ZimDownloadManager.swift:256 — `progress()` unconditionally flips any non-downloading item back to `.downloading`, so a `didWriteData` callback already in flight when `pause()` sets `.paused` (line 163) runs afterward and resets the state; the task's `NSURLErrorCancelled` completion then returns at line 473 without restoring `.paused` — the row is stuck "downloading" with no running task, so `resume()` early-returns at line 185, `isInFlight`/`hasActiveDownloads` stay true, and `SleepBlocker` keeps the idle timer disabled indefinitely — make `progress()` leave `.paused` items untouched (or gate the flip on a per-item paused flag/attempt counter) so the pause state survives trailing callbacks.

## Coverage
ios/MCPZimChat/Providers/LlamaCppProvider.swift — findings: 2
ios/MCPZimChat/Providers/MockProvider.swift — clean
ios/MCPZimChat/Providers/ModelProvider.swift — clean
ios/MCPZimChat/Sharing/ChatSession+ModelSharing.swift — clean
ios/MCPZimChat/Sharing/ZimCatalog.swift — clean
ios/MCPZimChat/Sharing/ZimDownloadManager.swift — findings: 1
ios/MCPZimChat/Sharing/ZimSwarmController.swift — clean
# Batch 5 review — ios/MCPZimChat/Views/*

- [low] ios/MCPZimChat/Views/HeroMediaView.swift:225 — `isLikelySpacer` regex `\bwidth\s*=\s*["']?1["']?` has no end-anchor, so it matches any `width`/`height` attribute whose value merely *begins* with `1` (e.g. `width="100"`, `height="165"`, `width=100`) — legitimate hero images are misclassified as 1-px spacers and silently skipped, so the hero-media feature renders nothing for most Wikipedia thumbnails (heights 100–199 are the common case); the same flaw is on line 226 for `height` — anchor the digit so only the literal value `1` matches, e.g. `["']?1(?!\d)["']?`.
- [low] ios/MCPZimChat/Views/ChatView.swift:781 — the assistant "Copy reply" button copies the raw `message.text` instead of the `displayText(...)`-stripped text shown in the bubble; during streaming and tool round-trips the buffer still contains `<tool_call>…</tool_call>` / `<think>…</think>` markup that the UI deliberately hides, so pasting emits hidden markup and unscrubbed reasoning — copy `Self.displayText(message.text, role: .assistant)` instead of `message.text`.
- [low] ios/MCPZimChat/Views/LibraryView.swift:219 — the GitHub PAT (gist scope) entered via `SecureField` is persisted in plaintext `UserDefaults` (`DebugReportConfig.githubToken`, key `debug.report.githubToken`) rather than the Keychain; the token is extractable from unencrypted device backups or by any process able to read the app container — store it via the Keychain instead (DEBUG-gated, so low severity).

## Coverage
ios/MCPZimChat/Views/ChatView.swift — findings: 1
ios/MCPZimChat/Views/DebugPane.swift — clean
ios/MCPZimChat/Views/DownloadCatalogView.swift — clean
ios/MCPZimChat/Views/HeroMediaView.swift — findings: 1
ios/MCPZimChat/Views/LibraryView.swift — findings: 1
ios/MCPZimChat/Views/MarkdownMessageParser.swift — clean
ios/MCPZimChat/Views/MarkdownMessageText.swift — clean
ios/MCPZimChat/Views/ModelPickerView.swift — clean
ios/MCPZimChat/Views/NearbyShareView.swift — clean
ios/MCPZimChat/Views/OfflineContentSetupView.swift — clean
ios/MCPZimChat/Views/PastLogsView.swift — clean
- [medium] ios/MCPZimChat/Voice/SpeechRecognizerService.swift:272 — `LegacySFSTT.start`'s recognition callback swallows benign end-of-audio errors (`kAFAssistantErrorDomain` 1101/203) but never finishes the stream — the code comment says "let the stream finish cleanly" yet `taskContinuation.finish()` is only called in the non-benign branch, so on the no-speech/cancelled path the `AsyncThrowingStream` never terminates and any consumer awaiting it hangs (the voice controller only survives via a separate 0.8s force-submit workaround in VoiceChatController) — concrete consequence: a silent/no-speech turn leaves the STT stream open forever, relying on a caller-side timeout instead of the protocol's "finish() terminates the stream" contract — smallest safe fix: in the benign branch also call `taskContinuation.finish()` (not `finish(throwing:)`).
- [medium] ios/MCPZimChat/Voice/SpeechRecognizerService.swift:258 — the optional return of `recognizer.recognitionTask(with:)` is assigned to `self.task` without a nil check — `SFSpeechRecognizer` can return nil (e.g. recognizer not available for the locale, resources unavailable) and then no result/error callback ever fires, so the returned stream never yields nor terminates — concrete consequence: the voice loop blocks forever in `for try await partial in stream` with no way to recover — smallest safe fix: `guard let task = recognizer.recognitionTask(...) else { taskContinuation.finish(throwing: ...); return stream }`.
- [medium] ios/MCPZimChat/Views/PlacesWebView.swift:680 — the `mcpzim` `WKScriptMessageHandler` dispatches native actions (`directions`/`article`/`share`/`website`/`call`) from `message.body` without validating `message.frameInfo.securityOrigin`, and the webview is not restricted to `zim://` (the `decidePolicyFor` guard at line 666 only cancels `.linkActivated` http/https/tel/mailto/facetime; programmatic/iframe/meta-refresh navigations pass as `.other` and load remote content over the network) — concrete consequence: a malicious or compromised ZIM page (or any remote page the webview is navigated to) can post `{action:"website"/"call", value:...}` and drive `UIApplication.shared.open` (line 768) to open arbitrary URL schemes natively — smallest safe fix: validate `message.frameInfo.securityOrigin.host` matches the expected zim host before handling, and/or block non-`zim://` navigations in `decidePolicyFor` instead of only `.linkActivated`.
- [low] ios/MCPZimChat/Voice/KokoroAssets.swift:75 — `isDownloaded` only requires each file to exist with `size > 0` and never checks the known `expectedBytes` (327_115_152 / 47_204_864) — concrete consequence: a truncated or corrupt model file left on disk is treated as complete, `KokoroDownloader.downloadIfNeeded` skips it, and TTS initialization then fails to parse the weights with no re-download — smallest safe fix: verify `size == expectedBytes` (or within a small tolerance) before declaring an asset present.
- [low] ios/MCPZimChat/Views/PlacesWebView.swift:581 — `applyFocusIfChanged` hand-escapes place labels/descriptions into a JS string literal but only escapes `\`, `"`, and `<`, omitting newline/CR and the JS line terminators U+2028/U+2029 — concrete consequence: a place label or description containing a newline produces a JavaScript SyntaxError in the injected `evaluateJavaScript` call, so tapping a list row silently fails to open the focus popup — smallest safe fix: escape `\n`, `\r`, U+2028, and U+2029 (or build the payload via `JSONSerialization` like `loadPlacesSpec` does instead of manual string interpolation).

## Coverage
ios/MCPZimChat/Views/PlacesWebView.swift — findings: 2
ios/MCPZimChat/Views/RootView.swift — clean
ios/MCPZimChat/Views/RouteWebView.swift — clean
ios/MCPZimChat/Views/VoiceChatView.swift — clean
ios/MCPZimChat/Views/ZimURLSchemeHandler.swift — clean
ios/MCPZimChat/Voice/KokoroAssets.swift — findings: 1
ios/MCPZimChat/Voice/KokoroDownloader.swift — clean
ios/MCPZimChat/Voice/ObjCExceptionWrapper.h — clean
ios/MCPZimChat/Voice/ObjCExceptionWrapper.m — clean
ios/MCPZimChat/Voice/SpeechRecognizerService.swift — findings: 2
ios/MCPZimChat/Voice/Supertonic3TTSService.swift — clean
# Batch 7 findings

- [medium] ios/MCPZimChat/Voice/VoiceChatController.swift:668 — `submitFinal` dedupes against `lastSubmittedTranscript`, which is never reset between listening cycles — a user who repeats an identical query across turns has the second utterance silently dropped, and `resumeListeningAfterCycle()` re-arms the mic without ever sending it to the session (the guard is only meant to absorb the force-submit-grace + recognizer-`isFinal` double-submit within one turn) — reset `lastSubmittedTranscript = ""` in `beginListening()` (or when a turn completes) so the dedupe scope is one turn, not the controller's lifetime.

- [low] ios/MCPZimChat/Voice/TTSService.swift:719 — `chunkForTTS`'s "soft-wrap" fallback only splits on commas, so a chunk over the 400-char limit with no commas is returned whole (the doc comment claims it wraps at "commas / whitespace" but no whitespace split exists) — a long comma-free sentence (e.g. a long prose run or list of space-separated words) reaches `kokoro.generateAudio` as one oversized utterance and can exceed Kokoro's 510-phoneme cap, causing synthesis failure or silent truncation — add a whitespace-boundary fallback (split at the last space before `limit`) for chunks that remain oversized after the comma pass.

## Coverage
ios/MCPZimChat/Voice/TTSService.swift — findings: 1
ios/MCPZimChat/Voice/VoiceChatController.swift — findings: 1
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
# Pi sweep — batch 8

## Findings

- [low] ios/MCPZimEval/EvalHarness.swift:893 — `runVariant` calls `provider.load()` for every variant but never calls `provider.unload()` (and the per-variant loop in `run()` at :813–828 has no autoreleasepool/teardown), so the harness relies on ARC/deinit to release each MLX model's Metal buffers before the next variant loads. — If MLX defers GPU-buffer release (the file's own header documents MLX GPU state not being cleanly torn down between sessions), later variants' `MemoryProbe` "baseline"/"post_load" samples and the absolute `maxPeakMB` scenario caps are contaminated by the previous model's resident weights, and running the full 8-variant matrix can exhaust RAM on a 36 GB Mac. — Call `await provider.unload()` at the end of `runVariant` (or wrap each variant iteration in `autoreleasepool` + explicit unload) so per-variant memory measurements are isolated.

- [low] ios/MCPZimEval/EvalHarness.swift:956 — The per-scenario peak-memory cap (`scen.maxPeakMB`, e.g. 6500 MB) is checked against a single post-turn sample (`postTurnSample.rssMB`) taken after `probe.stop()` and after the polling loop has observed `session.isGenerating == false`, rather than against the peak recorded during the decode window by `probe.startContinuous(...)`/`stop()`. — The jetsam the phone-target scenarios are designed to reproduce happens at the instantaneous mid-prefill/decode peak; a transient spike above the cap that subsides before the post-turn sample is not caught, so a variant can pass the memory ceiling even though it peaked over the jetsam headroom. — Compare the cap against the continuous-window peak (e.g. from the `startContinuous`/`stop` samples or `probe.summary()`) for that turn, not just the post-turn instantaneous RSS.

## Coverage
ios/MCPZimEval/EvalHarness.swift — findings: 2
ios/MCPZimEval/LlamaCppProbeCLI.swift — clean
ios/MCPZimEval/ProbeCompareCLI.swift — clean
ios/MCPZimEval/ProbeE2ECLI.swift — clean
ios/MCPZimTTSBench/TTSBenchMain.swift — clean
# Pi sweep — batch 9

## Findings

- [high] ios/scripts/testflight-assign-internal.rb:25 — `fixed_width` builds the ES256 signature bytes with `integer.to_s(2)`, which returns a *string of ASCII '0'/'1' characters* whose `.bytesize` is the bit-length (≈256 for a real P-256 component), not the raw big-endian bytes. The guard `bytes.bytesize > width` (width=32) therefore fires on essentially every real signature and the script aborts with "invalid ES256 signature component" before making any App Store Connect call. Even if the guard were removed, the code would `base64url`-encode ASCII bits instead of the 32-byte r||s, producing an invalid JWT. Net effect: the internal-TestFlight assignment step in testflight-upload.sh (run by default unless `MCPZIM_SKIP_INTERNAL_ASSIGNMENT=1`) always exits 1 and never assigns the build. Smallest safe fix: pack the integer as fixed-width big-endian bytes, e.g. `hex = integer.to_s(16); abort_with(...) if hex.length > width * 2; bytes = [hex.rjust(width * 2, "0")].pack("H*")`.

- [low] ios/scripts/mcp-crashes.sh:87 — `fname = os.path.basename('$file')` interpolates the caller-supplied path into Python source inside `python3 -c "..."`. A filename containing a single quote terminates the string literal and lets the rest of the path execute as Python (or, at minimum, crashes the summary parse). `summary <file>` and the `full_scan` summary pass a path straight through. Smallest safe fix: pass the path as a real argument, e.g. `python3 -c '...' "$file"` and read `sys.argv[1]`, or use `os.environ`.

- [low] ios/tools/eval.sh:89 — `xcodebuild test ... | tee "$TEST_LOG" > /dev/null || true` discards xcodebuild's exit status, so `eval.sh` always exits 0 even when every test failed. Any automation keying off the script's exit code silently reports success despite failing suites. Smallest safe fix: drop the `|| true` and propagate the pipeline status, e.g. run under `set -o pipefail` (already set) and `exit ${PIPESTATUS[0]}` after the tee.

- [medium] mcpzim/routing.py:471 — `_read_graph_bin` only reads `routing-data/graph.bin`, but `open_zim` (mcpzim/library.py:160-161) sets `has_routing=True` when *either* `routing-data/graph.bin` or `routing-data/graph.json` exists. A streetzim that ships only `graph.json` is classified as routable, gets `plan_driving_route`/`route_from_places` tools registered, and then `zim.archive.get_entry_by_path("routing-data/graph.bin")` raises an unhandled `KeyError` instead of a clean "no route data" error. Smallest safe fix: restrict `has_routing` to `graph.bin` (the only format `Graph.parse` supports), or have `_read_graph_bin` fall back to / reject `graph.json` gracefully.

- [low] swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift:83 — `runToolCall` interpolates the model-supplied `name` (and the `error` text on line 85) unescaped into the `<tool_response name="...">` wrapper. A tool name containing `"`, `>`, or `</tool_response>` corrupts the response marker that the next iteration's `extractToolCall` relies on, breaking the loop or injecting content into the transcript. Smallest safe fix: validate/whitelist `name` (e.g. `[A-Za-z0-9_.-]+`) before embedding, and XML-escape the error text.

## Coverage
ios/project.yml — clean
ios/scripts/mcp-crashes.sh — findings: 1
ios/scripts/mcp-deploy-verify.sh — clean
ios/scripts/mcp-logs.sh — clean
ios/scripts/mcp-report.sh — clean
ios/scripts/testflight-assign-internal.rb — findings: 1
ios/scripts/testflight-upload.sh — clean
ios/scripts/verify-app-signature.sh — clean
ios/tools/eval.sh — findings: 1
mcpzim/__init__.py — clean
mcpzim/__main__.py — clean
mcpzim/cli.py — clean
mcpzim/content.py — clean
mcpzim/geocode.py — clean
mcpzim/library.py — clean
mcpzim/routing.py — findings: 1
mcpzim/server.py — clean
pyproject.toml — clean
swift/Examples/Gemma4Integration/Gemma4ToolLoop.swift — findings: 1
swift/Package.swift — clean
# Findings — batch 10

- [medium] swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift:50 — `firstCall(in:)` applies the `impliedBodyEnd` fallback (which needs only balanced `{…}` braces) even on the streaming parse path, with no distinction between streaming and end-of-stream (unlike `ChatToolCallParser`/`Gemma3Template`, which gate leniency behind an `allowImplicitClose`/`firstToolCallAfterClip` flag). During streaming, once the model has emitted `<|tool_call>call:NAME{…balanced…}` but not yet the `<tool_call|>` sentinel, the call is dispatched and the returned `range` ends at the closing `}`, so the host splicing out `range` leaves the later-arriving `<tool_call|>` bytes dangling in the buffer — a stray sentinel can surface in the user-visible transcript and misalign the tool-response round-trip. `Gemma4Template.firstToolCallAfterClip` (default impl) also delegates to the same lenient parser, so the documented "streaming is strict / post-clip is lenient" contract is not honored. — Add a strict/lenient split: make `firstCall(in:)` require the `<tool_call|>` sentinel (return nil while it's absent) and add a `firstCallAfterClip(in:)` that enables the `impliedBodyEnd` fallback, then wire `Gemma4Template.firstToolCallAfterClip` to the lenient variant.

- [low] swift/Sources/MCPZimKit/AnswerAttribution.swift:68 — The connective-prose short-circuit returns `SentenceAttribution(sentence:passageIndex: nil, support: 1)`, but `isSupported` is defined as `passageIndex != nil` (line 27). This yields a self-contradictory value: full coverage (`support == 1.0`) yet `isSupported == false`. Any consumer that flags hallucinations via `!isSupported` (or the same file's `logLine`, which prints `sN→UNSUPPORTED 1.00`) will flag innocuous connectives ("Sure!", "Happy to help.") that the code comment explicitly says to "never flag". — Represent connective prose as a distinct state rather than overloading `support`, or have `logLine`/callers treat `support >= supportThreshold` as "not flagged" and skip the "UNSUPPORTED" label when `support == 1 && passageIndex == nil`.

- [nit] swift/Sources/MCPZimKit/ArticleHeuristics.swift:837 — `contextualizedDiscussionQuestion` tests keyword-poor follow-ups with `kws.allSatisfy { Self.unitAttributeKeywords.contains($0) }`, but `questionKeywords` returns unstemmed tokens and `unitAttributeKeywords` lists only singular forms ("year", "age", "name"). A plural unit follow-up such as "What years?" yields `kws == ["years"]`, which is not in the set, so it is not recognized as keyword-poor and does not inherit the previous question's context — the exact retrieval defect the function was added to fix (unit-only elliptical follow-ups ranking e.g. "Demographics"). — Stem each keyword (or check both singular/plural) before the `unitAttributeKeywords.contains` test, e.g. `kws.allSatisfy { Self.unitAttributeKeywords.contains(stem($0)) }`.

## Coverage
swift/Sources/MCPZimKit/AnswerAttribution.swift — findings: 1
swift/Sources/MCPZimKit/ArticleCache.swift — clean
swift/Sources/MCPZimKit/ArticleHeuristics.swift — findings: 1
swift/Sources/MCPZimKit/ArticleSections.swift — clean
swift/Sources/MCPZimKit/ChatToolCallParser.swift — clean
swift/Sources/MCPZimKit/ChatTurn.swift — clean
swift/Sources/MCPZimKit/ConversationFocus.swift — clean
swift/Sources/MCPZimKit/ConversationThreads.swift — clean
swift/Sources/MCPZimKit/Embeddings.swift — clean
swift/Sources/MCPZimKit/Gemma3Template.swift — clean
swift/Sources/MCPZimKit/Gemma4PromptTemplate.swift — clean
swift/Sources/MCPZimKit/Gemma4ToolCallParser.swift — findings: 1
- [low] swift/Sources/MCPZimKit/GeoMath.swift:22 — haversineMeters returns `r * 2 * atan2(sqrt(a), sqrt(1 - a))` without clamping `a` to ≤1; floating-point rounding on near-antipodal coordinates can push `a` a hair above 1, making `sqrt(1 - a)` NaN. The NaN then propagates into user-facing distance text (MCPToolAdapter.swift:2063), POI distance sorting (ZimService.swift:1258), and the Router A* heuristic (Router.swift:66/248/311). — clamp `a = min(max(a, 0), 1)` before the `sqrt`/`atan2` calls.
- [low] swift/Sources/MCPZimKit/IntentRouter.swift:1311 — `expandSharedSuffix` appends only the last word of the second title as the shared suffix, so multi-word suffixes mis-parse: "compare first and second world war" yields `["first war", "second world war"]`, dispatching a bogus "first war" article lookup that misses and degrades the comparison reply. — validate the expanded first title against the index before dispatch, or extend the heuristic to try a 2-word tail when the second title has ≥3 words.
## Coverage
swift/Sources/MCPZimKit/Gemma4ToolFormat.swift — clean
swift/Sources/MCPZimKit/Geocoder.swift — clean
swift/Sources/MCPZimKit/GeoMath.swift — findings: 1
swift/Sources/MCPZimKit/IntentRouter.swift — findings: 1
swift/Sources/MCPZimKit/LFM25Template.swift — clean
- [medium] swift/Sources/MCPZimKit/MCPToolAdapter.swift:1849 — `nearby_stories` advertises "omit `place` to anchor on the user's current GPS" (tool description line 443, schema line 2881 "the user's current GPS is the implicit center"), but `dispatchNearbyStories` requires explicit numeric `lat`/`lon` and never consults `hostStateProvider` (unlike `what_is_here`/`distance_to`/`route_status` which do) — the headline "tell me something interesting about where I am" flow fails with "nearby_stories requires numeric `lat` and `lon`" whenever the model omits coords — fall back to `hostStateProvider()?.currentLocation` before the guard, mirroring `dispatchDistanceTo`.
- [low] swift/Sources/MCPZimKit/MCPToolAdapter.swift:746 — `near_places`' schema documents "the user's current GPS is the implicit center" (line 2752), but the dispatch only accepts explicit `lat`/`lon` or a geocodable `place` and never reads `hostStateProvider` — "what's around me" fails with "near_places needs a center" instead of using the host GPS fix — add the same `hostStateProvider` fallback used by `what_is_here`/`distance_to`.
- [nit] swift/Sources/MCPZimKit/MemoryProbe.swift:147 — the "≥5GB/≥6GB/≥7GB" threshold bands compare `rssMB` against 5000/6000/7000, but `rssMB` is `phys_footprint / 1_048_576` (MiB), so the bands actually fire at ~4.88/5.86/6.84 GiB — the eval scorecard's jetsam-risk bands are shifted by ~2–5% and can misread a near-cap sample — compare against 5/6/7 × 1024 or relabel the bands GiB.
- [nit] swift/Sources/MCPZimKit/MemoryStats.swift:29 — `physFootprintMB()` divides by `1_048_576` (2^20, MiB) but its doc comment claims "base-10, matching most memory UIs" — every log/scorecard that prints "MB" is actually MiB (~5% larger) — divide by `1_000_000` for true base-10, or fix the comment to say binary (MiB).

## Coverage
swift/Sources/MCPZimKit/MCPToolAdapter.swift — findings: 2
swift/Sources/MCPZimKit/MemoryProbe.swift — findings: 1
swift/Sources/MCPZimKit/MemoryStats.swift — findings: 1
swift/Sources/MCPZimKit/ModelTemplate.swift — clean
swift/Sources/MCPZimKit/PlacesPayload.swift — clean
swift/Sources/MCPZimKit/QueryComplexity.swift — clean
swift/Sources/MCPZimKit/QwenChatMLTemplate.swift — clean
- [medium] swift/Sources/MCPZimKit/SZRGSpatial.swift:388 — `SZCIIndex.parse` reads `numNodes` (and `numNodeShards`/`nodesPerShard` at :418-419) from the untrusted SZCI v2 header without any cap or cross-check, while every other declared size in the same function is validated with `requireBytes`. `numNodes` is then consumed by `ZimService.loadNodeShards` (`ZimService.swift:1869`: `var combined = [Int32](repeating: 0, count: idx.numNodes * 2)`), which allocates `8 × numNodes` bytes before any shard byte is read — a crafted ZIM with `numNodes ≈ 0xFFFFFFFF` drives a ~34 GB allocation and an OOM/jetsam crash on the route path (`planDrivingRoute → cachedSpatialGraph → loadSpatialGraph → loadNodeShards`). The sibling monolithic parser (`SZRGGraph.parse`) explicitly guards this class (`numGeoms <= numEdges`, plus `requireTable` for every table), so the spatial path is the gap. — Fix: in `loadNodeShards` (or `SZCIIndex.parse` v2), reject/validate `numNodes` against an overflow-safe `numNodeShards × nodesPerShard` bound and a hard total-bytes cap before allocating, and overflow-check `shard * nodesPerShard * 2 * 4` (`ZimService.swift:1876`).

- [low] swift/Sources/MCPZimKit/SZRGSpatial.swift:581 — `SZRGInt.readVarintLE` silently returns a partial `result` when a varint exceeds 64 bits (`shift >= 64`) or runs off the end of the buffer, instead of signalling truncation the way the monolithic `RawCursor.readVarint` in `SZRGGraph.swift` does (`throw SZRGError.truncated("varint too long")`). `SZRCCell.decodeGeom` only rejects deltas that fall outside `Int32` range, so an over-long/unterminated varint whose garbage value happens to fit `Int32` is zigzag-decoded into wrong or duplicated polyline coordinates and accepted as a valid geometry rather than failing the cell load. — Consequence: a corrupt SZRC geom blob produces silently wrong route polylines instead of a loud load error, diverging from the file's own "reject malformed varints as truncated" intent. — Fix: make `readVarintLE` throwing (or return a sentinel) on `shift >= 64`/exhausted buffer and propagate it from `decodeGeom`.

## Coverage
swift/Sources/MCPZimKit/ReferenceResolver.swift — clean
swift/Sources/MCPZimKit/RegexCache.swift — clean
swift/Sources/MCPZimKit/Router.swift — clean
swift/Sources/MCPZimKit/StreamingSpeechPolicy.swift — clean
swift/Sources/MCPZimKit/StubZimService.swift — clean
swift/Sources/MCPZimKit/SZRGChunked.swift — clean
swift/Sources/MCPZimKit/SZRGEncoder.swift — clean
swift/Sources/MCPZimKit/SZRGGraph.swift — clean
swift/Sources/MCPZimKit/SZRGSpatial.swift — findings: 2
swift/Sources/MCPZimKit/ToolLoopGuard.swift — clean
swift/Sources/MCPZimKit/ZimReader.swift — clean
- [low] swift/Sources/MCPZimKit/ZimService.swift:285 — search() renders a lead snippet for every accepted hit via leadSnippet→renderLeadSnippet, which calls reader.read(path:) and therefore decompresses the FULL article body per candidate (ZimEntry.content is the whole Data; the 64 KB cap in leadPrefixHTML only truncates parsing, not the read). With the adapter's fetchLimit up to 100, a single search can force up to limit full-article decompressions on the hot path. Consequence: multi-second search latency and transient memory spikes on device for large ZIMs; the 64-entry snippet cache only helps within-turn overlap, not the first query. Smallest safe fix: add a bounded partial-read/snippet API to ZimReader (or reuse a stored snippet index), or cap snippet rendering to a small top-K of candidates instead of every accepted hit.
- [low] swift/Sources/MCPZimKit/ZimService.swift:1653 — nearNamedPlace resolves its `place` string via geocodeResolved (line 762), which does not apply the parseLatLon "lat,lon" short-circuit that the public geocode() (line 734) applies; a coordinate string ("37.44,-122.15") therefore works for geocode/route_from_places but throws ZimServiceError.noMatch when passed to near_named_place. Consequence: inconsistent coordinate-string handling between sibling place tools — a user/model passing a coordinate to "what's near X" gets a resolution failure instead of the synthetic Place. Smallest safe fix: have nearNamedPlace first try Self.parseLatLon(place) (or route through geocode) before falling back to geocodeResolved.
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
- [medium] swift/Tests/MCPZimKitTests/NearPlacesChipIndexTests.swift:157 — `testHugeSearchDataSkipsFullScanNoCrash` cannot fail if the record-count OOM guard is removed — the manifest declares chunks `{"00":600000,"01":500000}` but ships no `search-data/00.json`/`01.json` files, so an unguarded full scan would read `nil` for every chunk and still return `0` results; the assertion `XCTAssertEqual(r.results.count, 0)` passes on both the guarded and unguarded paths, giving false confidence on a critical OOM-prevention path (the comment even concedes "if the guard failed and it tried to scan, it'd still not crash here") — add the chunk files, or use a tracking reader and assert zero `search-data/*` reads (the `zz`-chunk tripwire pattern already used in NearPlacesKindFallbackTests) so the test discriminates guarded vs unguarded.
- [nit] swift/Tests/MCPZimKitTests/PlacesPayloadTests.swift:198 — `testResultsExcerptWinsOverKindDistance` declares a dead `json` multi-line literal whose content is invalid JSON (it embeds a literal `" + "` mid-string) and only uses it via `_ = json` to silence the unused warning; the actual assertion runs against a separate hand-cleaned literal — confusing dead code that invites a reader to "fix" the wrong string — delete the dead `json` literal and its `_ = json` suppression (or make the dead literal the actual input once corrected).
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
swift/Tests/MCPZimKitTests/NearPlacesChipIndexTests.swift — findings: 1
swift/Tests/MCPZimKitTests/NearPlacesKindFallbackTests.swift — clean
swift/Tests/MCPZimKitTests/NearPlacesWikiEnrichmentTests.swift — clean
swift/Tests/MCPZimKitTests/PlacesPayloadTests.swift — findings: 1
swift/Tests/MCPZimKitTests/QueryComplexityTests.swift — clean
# Batch 16 findings

- [medium] tools/fine-tune/generate_chains.py:25 — `sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")` hardcodes an absolute path into a specific user's home directory (a *different* checkout than this repo's `tools/llama-smoke`) — `from eval import ...` / `from generate import ...` raise `ModuleNotFoundError` on any machine where that path doesn't exist, so the chain-data generator cannot run at all off-machine (and `generate.py` isn't even present in this repo's `tools/llama-smoke/`, so the repo-local path wouldn't work either without vendoring) — derive the path relative to `__file__` (e.g. `Path(__file__).resolve().parents[2] / "llama-smoke"`) or take it as a `--llama-smoke-dir` argument.
- [medium] tools/fine-tune/eval_ft_pcgaming.sh:80-82 — under `set -euo pipefail`, `pass=$(echo "$out" | grep -oE "passed=[A-Za-z]+" | cut -d= -f2)` (and the `wall=`/`peak=` lines below) abort the whole script if a `RESULT` line omits a field (e.g. `peak_mb` missing/empty on a CPU run), because `grep` exits 1, `pipefail` propagates it, and `set -e` kills the shell mid-loop — one malformed RESULT line silently discards every remaining model×scenario result — append `|| true` (or `|| echo ""`) to each parse pipeline, or guard each field with `if [[ "$out" == *peak_mb=* ]]`.
- [low] tools/fine-tune/finetune.sh:135 — `BASE_SNAPSHOT=$(ls -d ~/.cache/huggingface/hub/models--${BASE_MODEL//\//--}/snapshots/*/ 2>/dev/null | head -1)` runs under `set -euo pipefail`; when the glob matches nothing (fresh machine, `HF_HOME` elsewhere, cleaned cache), `ls` exits 2, `pipefail` propagates it and `set -e` aborts the pipeline immediately after the fuse step — the `[[ -n "$BASE_SNAPSHOT" ]]` guard on the next line is dead code because the assignment never returns — append `|| true` to the pipeline (or use a glob-expansion test first) so the tokenizer restore degrades gracefully as intended.
- [low] tools/fine-tune/finetune_lfm2.sh:172 — same `BASE_SNAPSHOT=$(ls -d ~/.cache/huggingface/hub/.../snapshots/*/ 2>/dev/null | head -1)` pattern under `set -euo pipefail`; this one is *unguarded* (Step 3.6 runs unconditionally), so when the snapshot glob has no match the script aborts every run before the HF→GGUF convert step, and the `[[ -n "$BASE_SNAPSHOT" ]]` guard never gets a chance to skip — append `|| true` to the pipeline.
- [low] tools/fine-tune/finetune_cuda.py:149 — `enc["labels"] = list(enc["input_ids"])` sets the LM labels to the entire tokenized transcript with no assistant-only masking; the loss therefore trains the model to reproduce the system preamble and the user turns (not just its own replies), wasting capacity and making the fine-tuned model prone to emitting user-turn/preamble text at inference — mask non-assistant tokens to `-100` (e.g. from the chat template's role boundaries) so the loss is computed only over assistant responses.

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
tools/fine-tune/finetune_cuda.py — findings: 1
tools/fine-tune/finetune_cuda.sh — clean
tools/fine-tune/finetune_lfm2.sh — findings: 1
tools/fine-tune/finetune_unsloth.py — clean
tools/fine-tune/finetune_unsloth.sh — clean
tools/fine-tune/finetune.sh — findings: 1
tools/fine-tune/generate_chains.py — findings: 1
# Pi sweep — batch 17

## Findings

- [medium] tools/fine-tune/generate_chains3.py:908 — resume logic is broken: `todo` is computed as `args.n - existing` and only used for the early-return check/print, but the per-template loop still spawns `range(target)` tasks where `target = total_targets[template]` is derived from the full `args.n` (not scaled down by `existing`) — re-running a partially-complete generation appends ~`args.n` new rows instead of topping up to `args.n`, so the documented "resumable" behavior silently overshoots (e.g. 400 existing + `--n 800` → ~1200 rows) and doubles GPU/API cost — smallest safe fix: scale each template's target by the remaining fraction (e.g. `int(round(target * todo / args.n))` or track per-template existing counts) before building the task lists.

- [low] tools/fine-tune/split_chain_rows.py:117 — sub-rows A (`msgs[0:4]`) and B (`msgs[0:8]`) are emitted unconditionally without checking `_est_tokens(...) <= max_seq`, even though the tool's stated contract is that every emitted row "fits cleanly under a target max_seq_length"; only sub-row C is size-gated — a chain whose turn-1 preamble + `tool_response_1` is unusually large still produces an oversized A/B row that the FT loop then truncates (losing the tail the tool was built to preserve) — smallest safe fix: wrap the A and B appends in the same `_est_tokens(...) <= max_seq` check (or apply `_abbreviate_tool_response_1`) before writing them.

- [medium] tools/fine-tune/train_all_cuda.sh:35 — the `CANDIDATES` batch sizes contradict the file's own documented memory analysis: the comment (lines 27–32) concludes "Gemma 1B drops to bsz=2 and Gemma 4B to bsz=1" because Gemma 3 "thrashes at bsz=4 even at the 1B size … pegs all 24GB → cudaMalloc thrashing → 7× slower", yet the array still sets `google/gemma-3-1b-it` to bsz=4 and `google/gemma-3-4b-it` to bsz=2 — if the analysis is correct the Gemma candidates will thrash or OOM on a 24GB card (7× slowdown or build failure) — smallest safe fix: set Gemma 1B to `2` and Gemma 4B to `1` per the comment (or correct the comment if the values were deliberately re-verified).

- [low] tools/fine-tune/train_all.sh:43 — the mlx-community model tags `Qwen3.5-4B-Instruct-bf16` and `Qwen3.5-1.7B-Instruct-bf16` contain the `.5` typo that the sibling script `train_all_cuda.sh` explicitly documents ("the `.5` in the mlx tag was a typo; Qwen3-4B is the real model"); these two candidate IDs therefore likely 404 on Hugging Face and silently fail that candidate's build (only surfaced at the end as "BUILD FAILED") — smallest safe fix: correct the tags to the real mlx-community names (e.g. `Qwen3-4B-Instruct-bf16` / `Qwen3-1.7B-Instruct-bf16`).

- [low] tools/fine-tune/v7_eval_and_memsweep.sh:16 — the GGUF wait loop (`for i in $(seq 1 120)`) has no post-loop guard: if the Q4_K_M GGUF never stabilizes within ~30 minutes the script falls through and runs `grid.py` / `llama-quantize` against a missing or still-growing GGUF, producing garbage results or spurious failures instead of aborting — smallest safe fix: after the loop, test `[ -f "$GGUF" ]` (and size) and `exit 1` with a clear message if the file was never confirmed stable.

## Coverage
tools/fine-tune/generate_chains3.py — findings: 1
tools/fine-tune/generate_places_diverse.py — clean
tools/fine-tune/generate.py — clean
tools/fine-tune/retry_lfm2_train.sh — clean
tools/fine-tune/split_chain_rows.py — findings: 1
tools/fine-tune/train_all_cuda.sh — findings: 1
tools/fine-tune/train_all.sh — findings: 1
tools/fine-tune/v7_eval_and_memsweep.sh — findings: 1
tools/gemma-smoke/Package.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/main.swift — clean
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — clean
tools/llama-smoke/bench.py — clean
- [low] tools/llama-smoke/sweep.sh:21 — `head -10` sits at the end of a pipeline under `set -euo pipefail` — when `eval.py` emits more than 10 lines matching `RESULT|model=|final_content` (multi-turn scenarios print a `final_content`-bearing dict per turn), `head` exits after 10 lines and the upstream `tee`/`eval.py` get SIGPIPE, so the pipeline returns non-zero and `set -e` aborts the whole sweep mid-run — append `|| true` to the `head` (or drop `head`, e.g. `grep -E '...' | head -10 || true`).
- [low] tools/logpipe/ingest.sh:34 — gsutil output/error are redirected to fixed, predictable paths `/tmp/logpipe_ls.txt` and `/tmp/logpipe_rsync.err` — two concurrent `ingest.sh` runs clobber each other's files and mis-parse, and a local attacker can pre-create a symlink at those names to redirect the redirects onto an arbitrary writable file — use `mktemp` (with a `trap` cleanup) instead of hardcoded `/tmp` names.
- [low] tools/logpipe/ingest.sh:66 — dedup key is `log.stem`, but logs are recursed from per-device subfolders (`<device>/<name>.log`), so two devices uploading a log with the same stem (e.g. a timestamp-named file) silently drop the second device's conversation from the corpus — key on the path relative to the raw dir (e.g. `str(log.relative_to(raw_dir))`) or prepend the device folder name.
- [low] tools/logpipe/report.py:17 — `rows = [json.loads(l) for l in VERDICTS.open() if l.strip()]` calls `json.loads` with no guard — a single malformed/truncated line in `verdicts.jsonl` raises an uncaught `JSONDecodeError` and aborts the entire report — wrap the parse in try/except and skip bad lines (mirroring the defensive pattern `prep_judge.py` already uses).
- [nit] tools/logpipe/report.py:35 — `sorted(off, ...)[:12]` slices to 12 *before* filtering to high severity — with more than 12 high-severity failures the "worst offenders" list silently drops the extras — filter first, then slice: `[r for r in sorted(off, ...) if r.get("severity") == "high"][:12]`.
- [nit] tools/logpipe/parse_log.py:24 — `USER_RE = re.compile(r"^(.*)$")` is compiled but never referenced anywhere — dead code — delete it.
- [nit] tools/llama-smoke/grid.py:442 — `--scenarios` help text claims "empty = all 12", but `ALL_SCENARIOS` (lines 388-405) contains 16 entries and omits `bars_nb_ferry_heavy`, which `eval.py` does define — stale/incorrect doc that misleads users about grid coverage — update the help string (and add the missing scenario if it's meant to be swept).
- [nit] tools/llama-smoke/grid.py:420 — `fmt_markdown()` is defined but never called (main writes its own header/rows inline) — dead code — remove it.
- [nit] tools/llm-smoke/eval_gemma4.py:18 — `apply_chat_template` is imported from `mlx_vlm` but never used (the code calls `tokenizer.apply_chat_template` directly) — dead import — remove it.
- [nit] tools/llama-smoke/eval.py:118 — `from jinja2 import nodes` is imported inside `_install_generation_noop_extension` but never used — dead import — remove it.

## Coverage
tools/llama-smoke/eval.py — findings: 1
tools/llama-smoke/grid.py — findings: 2
tools/llama-smoke/sweep.sh — findings: 1
tools/llm-smoke/bench_kv.py — clean
tools/llm-smoke/bench_memory_gemma4.py — clean
tools/llm-smoke/bench_memory.py — clean
tools/llm-smoke/bench.py — clean
tools/llm-smoke/eval_gemma4_native.py — clean
tools/llm-smoke/eval_gemma4.py — findings: 1
tools/llm-smoke/eval.py — clean
tools/llm-smoke/gemma4_format.py — clean
tools/logpipe/ingest.sh — findings: 2
tools/logpipe/parse_log.py — findings: 1
tools/logpipe/prep_judge.py — clean
tools/logpipe/report.py — findings: 2
- [medium] tests/test_routing.py:21 — `test_zigzag_roundtrip` parametrizes only `|n| <= 1_000_000`, so it never exercises `n >= 2**31` where `mcpzim/routing.py:594` `_zigzag_encode` is actually broken: its positive branch uses `(n << 1) ^ (n >> 31)` (32-bit) while the negative branch uses `(n << 1) ^ (n >> 63)` (64-bit). For `n == 2**31` the roundtrip yields `-2147483649` instead of `2147483648` (verified). The test therefore certifies a codec that silently corrupts any positive value `>= 2**31` while the suite reports green. — smallest safe fix: add `2**31`, `2**32`, and `2**63 - 1` to the parametrize list and fix `_zigzag_encode` to use `n >> 63` in both branches (or `n >> 31` consistently).
- [low] tests/test_routing.py:148 — `test_geom_decoder_roundtrip` hand-builds the polyline blob instead of calling `encode_graph_v2(geoms=...)`, and no other test in the suite passes `geoms=`, so the geometry-encoding path in `encode_graph_v2` (mcpzim/routing.py:557-558) and the `geom_offsets`/`geom_blob` slicing in `Graph.parse` (mcpzim/routing.py:139-146) are never exercised end-to-end — a regression in geometry serialization or parse would pass CI unnoticed. — smallest safe fix: build the blob via `encode_graph_v2(nodes=..., edges=..., names=..., geoms=[pts])` and round-trip it through `Graph.parse`, asserting the decoded polyline.
- [low] tests/test_routing.py:26 — `test_varint_roundtrip` (and the rest of the suite) never exercises `_decode_varint`'s error branches (`ValueError("varint truncated ...")` at mcpzim/routing.py:44 and `ValueError("varint too long")` at routing.py:54), nor `Graph.parse`'s too-small-blob / bad-magic / unsupported-version guards — malformed `graph.bin` input gets its error handling run for the first time in production rather than in tests. — smallest safe fix: add `pytest.raises(ValueError)` cases feeding a truncated varint buffer, an over-long varint, and a short/garbage blob to `Graph.parse`.

## Coverage
tests/__init__.py — clean
tests/test_content.py — clean
tests/test_geocode.py — clean
tests/test_library.py — clean
tests/test_routing.py — findings: 3
- [low] tools/fine-tune/generate.py:581 — `_fill` singularizes the `{subtype}` placeholder with `random.choice(POIS).rstrip("s")`, which mangles "-ies"/"-easies" plurals (e.g. "breweries"→"brewerie", "bakeries"→"bakerie", "galleries"→"gallerie", "pharmacies"→"pharmacie", "speakeasies"→"speakeasie") — generated training queries (e.g. `"show me only the {subtype} ones"`, line 162) contain misspelled category names, degrading the fine-tune data — replace the naive strip with a real singularization rule (e.g. `-ies`→`-y`, `-easies`→`-easy`) or drop the placeholder.
- [nit] tools/fine-tune/generate.py:296 — module-level data constants `ASPECTS` (296 vs 414), `LANDMARKS` (302 vs 434), and `POIS` (309 vs 465) are each defined twice; the first definitions are immediately overwritten by the second and are dead code, and `SEED_QUERIES` (line 292) is computed but never referenced anywhere — duplicated/dead data invites divergence (editing one copy has no effect) — delete the first definitions and `SEED_QUERIES`, keep one source of truth.
- [low] tools/fine-tune/generate.py:672 — `_eval_preamble()` hard-codes `sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")` and, on `ImportError`, silently falls back to a generic preamble that does not match the eval harness's `SYSTEM_PREAMBLE` + tool block — on any machine without that path the training data is generated against a different prompt distribution (which this file itself warns "silently regresses on eval") with no error surfaced — derive the path relative to the repo/`__file__`, make it configurable, and fail loudly rather than silently degrading.
- [nit] ios/MCPZimEval/ProbeE2ECLI.swift:443 — `ProbeDiscussCLI.run` defaults `gguf` to the machine-specific absolute path `/Users/jasontitus/experiments/mcpzim/tools/fine-tune/ft-out-lfm2.5-8b-v7full/imx/lfm2.5-8b-a1b-ft.imx.IQ3_XS.gguf` — on any other checkout the default run fails at model load (no such file) with no hint beyond the path — make `--gguf` required (or default to a repo-relative path) instead of baking in a developer home directory.
- [low] ios/MCPZimChat/Views/RouteWebView.swift:668 — `webView.isInspectable = true` is set unconditionally (no `#if DEBUG`), so a shipping build exposes the offline ZIM content and the `mcpzim` JS bridge — which receives the user's live GPS coordinates via the injected `geoJSONCoords`/user-dot JS — to anyone who attaches Safari Web Inspector to the device/Mac — gate it behind `#if DEBUG` so production builds don't ship a live inspector surface.

## Coverage
swift/Tests/MCPZimKitTests/IntentRouterTests.swift — clean
ios/MCPZimChat/Views/RouteWebView.swift — findings: 1
ios/MCPZimEval/ProbeE2ECLI.swift — findings: 1
tools/fine-tune/generate.py — findings: 3
- [medium] swift/Sources/MCPZimKit/ReferenceResolver.swift:725 — The subjectless-elliptical binding path (resolve section 4, line 316) is silently defeated for several stems it explicitly enumerates. `introducesOwnSubject()` treats any non-functional token of 4+ chars as an "own subject", and the adjectives/words in the enumerated stems "how tall", "how long", "how come", "what year", "by whom" ("tall", "long", "come", "year", "whom") are all >=4 chars and absent from the functional set, so the function returns true and `!introducesOwnSubject(words)` short-circuits the bind — Concrete consequence: "how tall?", "how long?", "what year?" etc. after discussing a subject return `.none` (and isContinuation=false) and are dispatched as a fresh query instead of resolving to `primaryEntity` — exactly the weak-model multi-hop failure this resolver exists to prevent — Smallest safe fix: run the guard only on the tokens that remain after removing the matched elliptical stem (or add the question adjectives to the functional set), so a recognized stem always binds.

- [low] swift/Sources/MCPZimKit/ReferenceResolver.swift:428 — `descriptiveMatches()` filters tokens with the `stopwords` set (line 439) which contains every descriptive adjective ("older", "newer", "old", "new", "bigger", "smaller", "big", "small", "first", "last", "other"), so the descriptor is stripped before matching and can never disambiguate: "the older church" reduces to "church" and returns `.ambiguous` whenever two items share that noun — contradicting the comment that the descriptor is a disambiguation hint — Concrete consequence: a selector that does uniquely identify one list item still re-prompts the user instead of binding — Smallest safe fix: keep non-stopword descriptor tokens and prefer items whose label contains both the descriptor and the noun, falling back to noun-only matching.

- [low] ios/MCPZimChat/Sharing/ZimCatalog.swift:181 — `approximateBytes()` calls `Scanner.scanDouble()` without pinning a POSIX locale; on devices whose locale uses "," as the decimal separator, fractional labels such as "7.8 GB"/"2.1 GB" are not parsed as 7.8/2.1 (they yield the integer part or misparse), under-estimating `sizeBytes` in the unsafe direction for the free-space precheck the code otherwise deliberately over-estimates. Additionally `expandSize()` maps "P"→"PB" (line 169) but `approximateBytes()` has no P branch, so a PB-scale label falls through to a multiplier of 1 (treated as raw bytes) — Concrete consequence: the free-space precheck can approve a download that does not actually fit — Smallest safe fix: set `scanner.locale = Locale(identifier: "en_US_POSIX")` before scanning and add a "P" multiplier branch.
## Coverage
swift/Sources/MCPZimKit/ReferenceResolver.swift — findings: 2
tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift — clean
swift/Sources/MCPZimKit/ConversationThreads.swift — clean
ios/MCPZimChat/Sharing/ZimCatalog.swift — findings: 1
- [high] ios/MCPZimChatMacTests/ConversationalEvalTests.swift:408 — makeOrReuseSession() never calls runSetupIfNeeded() (and never sets setupState = .ready), but ChatSession.send(_:) early-returns when setupState != .ready (initial value is .pending, and loadSelectedModel() only loads the model — it does not advance setupState) — every scenario turn is silently dropped: isGenerating never flips, no assistant message is appended, no tools dispatch, so all 20 scenario tests FAIL (empty reply / empty tool list) whenever ZIMBLE_TEST_STREETZIM is set — add `await session.runSetupIfNeeded()` after `await session.loadSelectedModel()` (or set `session.setupState = .ready` the way the ChatSession test factory at ChatSession.swift:1957-1960 does).
- [nit] ios/MCPZimChatMacTests/ConversationalEvalTests.swift:538 — the test_00…test_19 methods hardcode Self.scenarios[N] indices, so inserting or reordering a scenario silently retargets every named test (name no longer matches the scenario it runs) — derive each test from a named lookup (e.g. scenarios.first { $0.name == "..." }) instead of positional indices.
- [low] tools/llm-smoke/eval.py:334 — _prefer_valid's docstring says it "Picks the LONGEST matching candidate" but the implementation returns valid[0] (the FIRST valid candidate) — in nested tool-call objects the wrong (outer/inner) candidate can be scored, producing misleading pass/fail results — either implement longest-candidate selection or correct the docstring to match the code.
- [low] tools/llm-smoke/eval.py:534 — `--mode bench` runs only CASES[0] (identical to `--mode one`), so the benchmark summary's "mean prefill / mean decode" is computed over a single case and is not representative — make bench run the full CASES list (or remove the redundant mode).
- [nit] tools/llm-smoke/eval.py:242 — _extract_between() is defined but never called anywhere (the tag-pair loop in extract_tool_call inlines its own find logic) — dead code that drifts from the real extraction logic — delete it or use it in the tag loop.

## Coverage
swift/Tests/MCPZimKitTests/DiscussRetrievalTests.swift — clean
ios/MCPZimChatMacTests/ConversationalEvalTests.swift — findings: 2
swift/Sources/MCPZimKit/SZRGGraph.swift — clean
tools/llm-smoke/eval.py — findings: 3
# Sweep findings — batch 23

## Findings

- [low] ios/MCPZimChat/Providers/FoundationModelsProvider.swift:145 — `ensureWarmSession()` (and `installNativeTools`/`setNativeInstructions`/`dropWarmSession`) mutate `warmSession`, `nativeTools`, and `nativeInstructions` without any synchronization, while the class is `@unchecked Sendable` and the mutations are reached from different execution contexts: `prewarmIfIdle()` is fired from the composer focus handler (main actor, ChatSession.swift:2296), while `setNativeInstructions` + `generateNativeTurn` run from the async `runNativeToolsTurn` task (ChatSession.swift:3028/3882). — A concurrent focus-prewarm and turn dispatch can both pass `if let existing = warmSession`/`warmSession = built`, creating two `LanguageModelSession`s or dropping the session mid-stream; this surfaces as Apple's `concurrentRequests` programmer-error or a lost native-tools conversation. — Serialize all warm-session/tool/instruction state behind the existing `queue` (or move it into an actor / `@MainActor`-bound helper) so read-check-write in `ensureWarmSession` is atomic.

- [low] ios/MCPZimChat/Views/NearbyShareView.swift:176 — `shareableVoiceSizeLabel` is a computed property evaluated during `body` rendering and calls `ZimSwarmController.shareableVoiceBytes`, which re-enumerates the voice-model directories on every access (`Supertonic3Assets.currentBytesOnDisk` runs a full `FileManager.enumerator` over `modelDirectory`, and `KokoroAssets.isDownloaded`/`currentBytesOnDisk` do per-file `attributesOfItem`; ZimSwarmController.swift:122-131). Because `manager` is `@ObservedObject` and its `@Published transfers` array updates continuously during a transfer, this body re-evaluates many times per second. — Repeated synchronous directory-tree enumeration on the main thread during active transfers, causing UI frame drops/stalls (LibraryView.swift:581-592 already offloads these same size computations to `Task.detached(priority: .utility)` for exactly this reason). — Compute the voice size once (e.g. in `onAppear`/a cached property or off-main-thread task) instead of inside `body`.

## Coverage
swift/Tests/MCPZimKitTests/SZRGSpatialTests.swift — clean
ios/MCPZimChat/Providers/FoundationModelsProvider.swift — findings: 1
swift/Tests/MCPZimKitTests/ArticleFactoidTests.swift — clean
ios/MCPZimChat/Views/NearbyShareView.swift — findings: 1
- [low] swift/Sources/MCPZimKit/QwenChatMLTemplate.swift:317 — `stripReasoning` cannot distinguish the final (non-streaming) message from a streaming buffer, so when Qwen's `<|im_end|>` stop marker clips generation inside an unclosed `<think>…` span the raw `<think>` tag and hidden chain-of-thought are left verbatim and land in the user's chat bubble — Qwen 3.x emits a reasoning scratchpad that is never meant for display, and the dangling-close branch only rescues the close-only form (`…</think>answer`), not the open-only form (`<think>…scratchpad` with no closer) — smallest safe fix: add a `final:` (or `stripUnclosedWhenComplete:`) parameter and, on the final-message call path only, also drop a trailing unclosed `<think>…` suffix; keep the leave-alone behavior for streaming callers.
- [nit] swift/Sources/MCPZimKit/QwenChatMLTemplate.swift:259 — `repairJSON` collapses comma runs with `while out.contains(",,")` + `replacingOccurrences(of: ",,")`, which rescans and reallocates the whole string once per comma pair (O(n²) on a comma-heavy body) — a long comma run in a model-emitted tool-call body makes the repair pass do quadratic work — smallest safe fix: use the single-pass regex `,{2,}` exactly as `Gemma3Template.repairJSON` (line 358) already does.
## Coverage
swift/Sources/MCPZimKit/QwenChatMLTemplate.swift — findings: 2
swift/Tests/MCPZimKitTests/ConversationContinuationTests.swift — clean
swift/Sources/MCPZimKit/Gemma3Template.swift — clean
ios/project.yml — clean
- [low] ios/MCPZimChat/Sharing/ZimSwarmController.swift:383 — `swarmID` from the LocalSwarm P2P engine callback is concatenated into a filesystem path (`stagingBase.appendingPathComponent(swarmID)`) and then enumerated and passed to `fm.removeItem(at: dir)` without validating it against the staging directory — a `swarmID` containing `../` (a malicious seeder controls the advertised swarm ID) makes `dir` resolve outside `Documents/Incoming`, so an empty directory (or one containing only `.DS_Store` / the two `.localswarm-*` sidecars) elsewhere in the app container can be deleted — resolve/standardize `dir` and reject it unless it is still under `stagingBase` (or reject any `swarmID` containing `/` or `..`) before enumerating/removing.
- [low] ios/MCPZimChat/Sharing/ZimSwarmController.swift:215 — `importCompletedSwarm` runs on the `@MainActor` and performs the entire import loop (per-file `moveItem`/`removeItem`/`attributesOfItem`/`createDirectory` plus the recursive `enumerator` walk in `cleanupStagingFolder`) synchronously on the main thread — importing a large voice-model bundle or many ZIMs issues hundreds–thousands of blocking filesystem syscalls and stalls the UI during the import — offload the move/cleanup work to a background task (the class is already actor-isolated; do the file I/O off the main actor and hop back to publish the summary).
- [low] tools/fine-tune/generate_places_diverse.py:39 — `sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")` hardcodes one developer's absolute home path — on any other machine the `eval` import in `_eval_preamble()` silently fails and every example falls back to a truncated preamble lacking the tool block/location context, silently diverging the generated training data — derive the path from the repo (e.g. `Path(__file__).resolve().parents[2] / "tools/llama-smoke"`) or make it configurable.
- [low] tools/fine-tune/generate_places_diverse.py:367 — resume counts completed examples by line count and then does `seeds = seeds[done:]`, but the `asyncio.gather` workers write in nondeterministic completion order, so line `i` of the output does not correspond to seed `i` — after an interrupted run, resuming skips/duplicates seeds (some examples written twice, others omitted) and a trailing partial line is counted as a full example — record the seed index in each written record (or write deterministically per-seed) and resume from the recorded index rather than the line count.
- [low] tools/fine-tune/generate_places_diverse.py:344 — `resp.choices[0].message.content` is dereferenced outside the `try` that guards the API call — a provider response with an empty `choices` list (some OpenAI-compatible local servers return `choices: []` on refused/malformed requests) raises `IndexError`, which propagates through `asyncio.gather` and aborts the entire batch run instead of logging and continuing — guard `resp.choices` before indexing (and move it into the error-handled block).
- [nit] tools/fine-tune/generate_places_diverse.py:289 — grounding validation matches result names with substring membership (`n in low`), so a reply that merely contains a name as a substring of an unrelated word (e.g. a place named "Bar" counted by the word "barbecue", "Cafe" by "cafeteria") passes the ≥2-name requirement without actually citing the place — the validator's stated purpose (force the student to cite real result names) is weakened — require token/word-boundary matching (e.g. regex `\b` + `re.escape(name)`) instead of bare substring search.

## Coverage
ios/MCPZimChat/Sharing/ZimSwarmController.swift — findings: 2
swift/Sources/MCPZimKit/Router.swift — clean
swift/Tests/MCPZimKitTests/ConversationThreadsTests.swift — clean
tools/fine-tune/generate_places_diverse.py — findings: 4

## Run stats

input 1183268 tok (+12352384 cached), output 558696 tok, cost $1.05 — 226 files in 41m (325.0 files/h, 1.7 min/batch)
