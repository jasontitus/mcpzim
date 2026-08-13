# Review verdicts — 2026-08-13

Independent verification (Claude subagents, adversarial re-read of every cited
site, with empirical recomputation where claims were checkable) of the two
DeepSeek-V4-Pro sweep reports:

- `PI_REVIEW_BUGS_DEEPSEEK_V4_PRO_20260813.md` — 96 findings:
  **42 confirmed / 2 refuted / 5 partial / 47 unsampled**
  (both highs + all 20 mediums verified; 27 lows sampled)
- `PI_REVIEW_PERF_DEEPSEEK_V4_PRO_20260813.md` — 59 findings:
  **39 confirmed / 2 refuted / 8 partial / 10 unsampled**
  (both highs + all 20 mediums verified; 27 lows sampled)

Note: `ios/LocalPackages/` (vendored third-party: FluidAudio, kokoro-ios,
mlx-swift-lm, llama.cpp-swift, LocalSwarm) was excluded from the sweep.

## Fix first (correctness/security)

1. **[high] ios/MCPZimChatMacTests/ConversationalEvalTests.swift:408** — the
   20-scenario conversational eval is entirely dead: `setupState` never leaves
   `.pending`, so every `send()` early-returns and assertions run on empty
   output. Compounded by `ios/tools/eval.sh:89`'s `|| true` hiding the failure
   from automation. (The ChatSession test factory at :1957 sets `.ready` with a
   comment naming this exact trap.)
2. **[medium] swift/Sources/MCPZimKit/SZRGSpatial.swift:388 +
   ZimService.swift:1869** — untrusted SZCI **v2** `numNodes` drives a pre-read
   `[Int32](count: numNodes*2)` (~34 GB at 0xFFFFFFFF) plus an
   overflow-trappable `shard*nodesPerShard*8`; ZIMs arrive via P2P nearby-share,
   so a crafted archive jetsams the app on the routing path. v1 validates; v2 is
   the gap.
3. **[medium] ios/MCPZimChat/Views/PlacesWebView.swift:680/664** — `mcpzim`
   script handler does no origin check and `decidePolicyFor` allows all `.other`
   navigations, so untrusted shared-ZIM content can load remote pages and drive
   `UIApplication.shared.open` with arbitrary schemes. Also ships
   `isInspectable = true` unconditionally (here, RouteWebView.swift:668,
   PlacesWebView.swift:451/892 — the report missed the latter two).
4. **[medium] ios/MCPZimChat/Providers/Gemma4Provider.swift:530 + :165** — user
   Stop/reset can't cancel MLX generation (no `onTermination`, no cancellation
   check, `cancelGeneration()` is the inherited no-op — other providers wire it),
   AND the KV-cache vars are mutated cross-thread by the background-reset path
   with no `isGenerating` guard on an `@unchecked Sendable` class — a real data
   race.
5. **[medium] ios/MCPZimChat/Sharing/ZimDownloadManager.swift:256** —
   pause/progress race wedges an item in "downloading" with no task, blocking
   resume and pinning `isIdleTimerDisabled` (battery drain) until relaunch.
6. **[medium] ios/MCPZimChat/Voice/VoiceChatController.swift:668** —
   `lastSubmittedTranscript` never resets, so repeating an utterance in a later
   turn ("yes", "next") is silently swallowed and the mic just re-arms.
7. **[low] ios/MCPZimChat/Providers/LlamaCppProvider.swift:598** — failure path
   frees the model and nils `self.model` but not `self.vocab`;
   `promptTokenCount` guards on non-nil vocab and would tokenize with a dangling
   pointer (narrow window, real UAF).

Also confirmed: native FoundationModels tools skip the limit/radius clamps the
text path has (`FoundationModelsNativeTools.swift:63/175`), reintroducing the
`limit*2` trap the adapter's own comment warns about; hero-image regex
misclassifies any 100–199 px dimension as a 1-px spacer (`HeroMediaView.swift:225`,
verified empirically) so hero media is dead for most articles; `routing.py:471`
raises a raw `KeyError` (not a clean "no route data") on a graph.json-only ZIM;
several fine-tune shell scripts abort under `set -euo pipefail` on unguarded
`ls | head` pipelines; `eval_ft_pcgaming.sh` parses the wrong `RESULT` line so
every row reports ✗.

## Perf: fix first

1. **[high] ios/MCPZimChat/AppIntents/ZimfoRunner.swift:69** — `@MainActor`
   static `buildFresh()` opens every ZIM synchronously on the main actor (own
   comment: "seconds"); paid on the first Siri/Shortcuts intent of a fresh
   process → intent-timeout/ANR risk.
2. **[medium] ChatView.swift:723 + MarkdownMessageText.swift:12** — the real
   streaming-jank pair: per 10 Hz UI push the whole growing message gets a
   ~11-pass regex strip AND a markdown re-parse on the main thread; O(n²) bytes
   over a long narration. (The memo at :725 is keyed to guarantee a miss every
   push.)
3. **[medium] FoundationModelsProvider.swift:352** — fresh
   `LanguageModelSession()` + full-transcript re-prefill on every `generate()`,
   once per tool iteration; multi-second TTFT tax per tool round-trip. File's
   own comment names the fix.
4. **[medium] RouteWebView.swift:54 (+PlacesWebView.swift:66)** — 10–100 KB JSON
   parse + downsample + ~400 `String(format:)` in `init`, re-run per
   `MessageRow.body` pass at streaming cadence.
5. **[medium] swift/.../ZimService.swift:1258** — up-to-500k-record `nearPlaces`
   scan pays a full haversine per record with no bbox reject; cheap prefilter
   (expect ~40-60% scan savings, not the whole "seconds" — per-record dict
   lookups remain).
6. **[medium] mcpzim/routing.py:103** — node/edge/geom arrays are boxed Python
   lists for million-node graphs (~4× memory, poor A* locality); `from array
   import array` is already imported and unused for these.
7. **[high] NearbyShareView.swift:175** — body-evaluated properties do recursive
   on-disk size walks (+`createDirectory` per access) every body pass of a
   per-tick-republishing view. (Real anti-pattern; "hangs" overstated — Kokoro
   is 2 files, so low-ms jank.)
8. **[medium] ConversationThreads.swift:619** — `re.matches(in:)` materializes
   every anchor match before a break-at-8 loop; `enumerateMatches` + stop.

## Refuted / corrected — do not "fix"

- **[high bugs] ios/scripts/testflight-assign-internal.rb:25** — REFUTED
  empirically. The claim assumed Ruby `Integer#to_s(2)` (a "0"/"1" digit
  string); the value is an `OpenSSL::BN`, whose `to_s(2)` returns raw
  big-endian bytes. Verified with a real P-256 signature: 32-byte ASCII-8BIT
  output, the `> 32` guard passes, `\0` left-pad is the standard DER→raw
  conversion. The JWT is well-formed; the script does not abort.
- **[medium bugs] SpeechRecognizerService.swift:258** — REFUTED. The SDK header
  declares `recognitionTaskWithRequest:` non-null (`NS_ASSUME_NONNULL`), so
  Swift imports a non-optional; the proposed `guard let` wouldn't compile.
  Unavailability surfaces via `isAvailable` (already checked) or the error
  callback.
- **[low perf] Gemma4Provider.swift:799** — REFUTED. Swift `String +=` on a
  uniquely-referenced local is amortized O(appended) → O(n) over the stream,
  not O(n²). The report's own batch-26 summary states this rule correctly.
- **[low perf] PastLogsView.swift:79** — REFUTED. The "unbounded log dir"
  premise is false: `LogArchive` prunes to `maxFiles = 20` at every session
  start, so `allFileInfos()` scans ≤20 entries (sub-ms). Sync-on-main is real
  but can never stall.
- **Report internal contradiction (perf):** batch 17 dismissed the
  `PromptExperiment.swift` per-token re-decode and `generate.py`
  `_build_tool_block` patterns as negligible and marked those files clean, then
  later batches re-reported them as medium. They are dev-tool cold paths —
  treat as the PARTIALs they are.
- Fix-advice corrections: `BARTModel.swift:166` — "synchronize only at EOS" is
  incoherent (the `.item()` IS the EOS check); the `asyncEval` half is valid.
  `Gemma4ToolCallParser.swift:50` — real, but harm is premature dispatch +
  KV-cache divergence (a perf bug), NOT transcript corruption (the host keeps
  the full buffer). `MCPToolAdapter.swift:984` — a `withTaskGroup` fan-out would
  fetch all candidates even though the serial loop usually stops at the first;
  use bounded prefetch / race-with-cancel.

## Duplicates (one fix each)

Bugs: `debug.report.githubToken` store+field ×2; hardcoded `/Users/jasontitus`
path ×3-4; unguarded `ls|head` under errexit ×2; missing-GPS-fallback sibling
tools ×2; MiB-labeled-MB ×2.
Perf: per-token full re-decode ×3 (PromptExperiment, self-dismissed in batch
17); linear nearest-node scan ×2 (SZRGGraph/Router); `_build_tool_block` ×2;
View-init JSON parse ×2 (Route/PlacesWebView); uncached `NSRegularExpression`
×4 (ZimCatalog/TTSService/ArticleHeuristics/IntentRouter); bench_memory
per-config load ×2.

## Provenance & cost

Model `deepseek/deepseek-v4-pro:high` (api.deepseek.com), 4-wide live sweeps
over 202 files + refeed. Bugs: $1.05 (1.2M in / 12.4M cached / 559k out).
Perf: $0.98 (1.53M in / 12.5M cached / 316k out). Repo total ≈ **$2.03** at
2026-08-13 list prices.
