# How to build and run

Durable notes on every working build + run incantation in this repo. Update this file whenever you add a target or discover a working invocation. The context window is not durable; this file is.

## iOS app (on physical iPhone)

```sh
cd ios
xcodegen generate
xcodebuild -project MCPZimChat.xcodeproj \
  -scheme MCPZimChat \
  -destination 'generic/platform=iOS' \
  -configuration Debug \
  -derivedDataPath build \
  DEVELOPMENT_TEAM=A6G8H8NGAM \
  build

# install to Jazzman 17 (edit UUID if device changes)
xcrun devicectl device install app \
  --device 5AE213CA-315A-532A-878B-2CC4EB051ABD \
  build/Build/Products/Debug-iphoneos/MCPZimChat.app

# terminate old process so the new build actually runs
PID=$(xcrun devicectl device info processes --device 5AE213CA-315A-532A-878B-2CC4EB051ABD 2>/dev/null \
      | grep MCPZimChat | awk '{print $1}' | head -1)
[ -n "$PID" ] && xcrun devicectl device process terminate \
  --device 5AE213CA-315A-532A-878B-2CC4EB051ABD --pid "$PID"

xcrun devicectl device process launch \
  --device 5AE213CA-315A-532A-878B-2CC4EB051ABD \
  org.mcpzim.MCPZimChat
```

Device UUIDs:
- Jazzman 17 (iPhone 17 Pro Max): `5AE213CA-315A-532A-878B-2CC4EB051ABD`
- Hardware serial: `00008150-000669303687801C` (what `idevice_id -l` returns — `devicectl` accepts it too)

List all attached devices: `xcrun devicectl list devices`.

## Vendored mlx-swift-lm fork (KV-cache quantization)

`ios/project.yml` pins `mlx-swift-lm` to `path: LocalPackages/mlx-swift-lm`. This is a copy of `mlx-swift-lm` 3.31.3 with a single-file patch: `Libraries/MLXLLM/Models/Gemma4Text.swift::Gemma4Attention.callAsFunction` branches on `QuantizedKVCacheProtocol` before `cache.update(...)` so MLX's 4-bit groupwise `QuantizedKVCache` runs end-to-end. Without the patch MLX asserts at `cache.update` on phones. The attention forward also dequantizes the post-update K/V on its way out so KV-shared layers still see full history.

Gate: `DeviceProfile.useQuantizedKVCache` (true on iPhone/iPad tiers, false on Mac). `Gemma4Provider.generate()` passes `kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0`. Full-attention layers swap on the first post-prefill step; sliding (`RotatingKVCache`) layers stay FP16 because upstream MLX doesn't support rotating-quantized yet.

Historical note: before 2026-04-21 the patch lived in a separate `Swift-gemma4-core` fork. `mlx-swift-lm` 3.x landed native Gemma 4 support, so we retired that fork and moved the patch into `mlx-swift-lm` instead. `KV_CACHE_COMPRESSION.md` "Status as of 2026-04-21" has the full migration notes.

If you bump the upstream version:

```sh
rm -rf ios/LocalPackages/mlx-swift-lm
rsync -a --exclude ".build" --exclude ".git" --exclude ".swiftpm" \
  ios/build/SourcePackages/checkouts/mlx-swift-lm/ \
  ios/LocalPackages/mlx-swift-lm/
# then re-apply the Gemma4Text.swift patch (same shape as KV_CACHE_COMPRESSION.md Option 1)
```

After bumping, force SPM to re-resolve against the vendored path:

```sh
rm -rf ios/build/SourcePackages && \
  xcodebuild -project ios/MCPZimChat.xcodeproj -scheme MCPZimChat \
  -derivedDataPath ios/build -resolvePackageDependencies
```

Build with `-skipMacroValidation` — `MLXHuggingFace` exposes Swift macros and Xcode wants explicit approval by default. Add that flag to any scripted `xcodebuild` invocation that touches the iOS target.

`tools/gemma-smoke/Package.swift` still uses the retired `Swift-gemma4-core` URL (the harness is FP16-only — LCP + tokenizer tests — so hasn't been migrated to the upstream stack yet).

## Device syslog (no more copy-paste from the in-window debug pane)

`ios/scripts/mcp-logs.sh` wraps `idevicesyslog -p MCPZimChat` and writes to `/tmp/mcp-syslog.log`. The app's `ChatSession.debug()` goes through OSLog so the streamer captures it.

```sh
ios/scripts/mcp-logs.sh start            # once per Mac boot
ios/scripts/mcp-logs.sh status           # check it's alive + bytes captured
ios/scripts/mcp-logs.sh tail             # last 60 app-filtered lines
ios/scripts/mcp-logs.sh recent 400       # last N lines
ios/scripts/mcp-logs.sh since '12:35'    # since timestamp
ios/scripts/mcp-logs.sh stop
```

Restart the streamer after a USB reconnect or app reinstall — `status` will still say "running" even if idevicesyslog stopped forwarding. Bytes-captured staying flat for > a minute is the tell.

## Crash logs

```sh
idevicecrashreport -e -k -f MCPZim ./crashes
```

**Always pass `-f MCPZim`.** Without the filter it copies thousands of system `.ips` files.

## mcp-crashes.sh — crash triage helper

`ios/scripts/mcp-crashes.sh` wraps the `idevicecrashreport` + `.ips` parsing loop we used to do by hand. Assumes one device attached; override with `MCPZIM_DEVICE_UUID=... mcp-crashes.sh ...` if there are several.

```sh
# Default: pull MCPZim + Jetsam, list newest 5 of each, summarise the top one.
ios/scripts/mcp-crashes.sh            # == `scan`

# Pull fresh (drops new .ips into /tmp/mcpzim-crash + /tmp/mcpzim-jetsam).
# Add --all to also sweep the unfiltered crash queue into /tmp/mcpzim-crash-all.
ios/scripts/mcp-crashes.sh pull
ios/scripts/mcp-crashes.sh pull --all

# List.
ios/scripts/mcp-crashes.sh mcpzim [N]     # app crashes, newest first
ios/scripts/mcp-crashes.sh jetsam [N]     # JetsamEvent-*.ips, newest first
ios/scripts/mcp-crashes.sh today          # just today

# Parse one .ips (captureTime, exception, faulting stack, jetsam RSS / peak).
ios/scripts/mcp-crashes.sh summary /tmp/mcpzim-jetsam/JetsamEvent-...ips

# Memory trajectory from the syslog buffer — requires mcp-logs.sh to be running.
ios/scripts/mcp-crashes.sh mem            # n/min/max/avg + last 15 samples
ios/scripts/mcp-crashes.sh peaks [N]      # top N peak mem readings (default 20)
```

The jetsam-summary line is the one to read first. A healthy MCPZim sample on an iPhone 17 Pro Max sits around RSS ≤ 5 GB; `rss=6[0-9]{3} MB` with `freeze_skip_reason: out-of-slots` means we crossed the hard 6144 MB process cap and iOS jetsammed us. `largestProcess: MCPZimChat` names us as the culprit; `largestProcess: Oura` (or similar third-party app) means we weren't the killer — the event is incidental system pressure, MCPZim may not have been killed at all.

If `mem` / `peaks` complain about no syslog buffer, check `mcp-logs.sh status` and restart the streamer (the `idevicesyslog` USB transport drops silently after a device disconnect — the script's `running` flag is stale; only `bytes` growing over time confirms forwarding).

## MCPZimEvalCLI (multi-model eval matrix)

Headless macOS executable that runs `EvalHarness` — the full multi-model × scenario scorecard — in its own clean process. Replaces the old `MCPZimChatMacTests/MultiModelEvalTests.swift` XCTest target: that setup injected the test bundle into `MCPZimChatMac.app`, stood up a second `ChatSession` inside the app's `@main` process, and crashed MLX's `slice_update` when both sessions raced on the GPU.

```sh
cd ios
xcodegen generate   # only if you edited project.yml
xcodebuild -project MCPZimChat.xcodeproj \
  -scheme MCPZimEvalCLI \
  -destination 'platform=macOS' \
  -configuration Debug \
  -skipMacroValidation \
  build

# Binary lands in DerivedData alongside MCPZimChatMac.
BIN=~/Library/Developer/Xcode/DerivedData/MCPZimChat-*/Build/Products/Debug/MCPZimEvalCLI
$BIN --help
$BIN                                    # full matrix (all cached variants × all scenarios)
$BIN --variant qwen3-4b                 # filter variants by substring (repeatable)
$BIN --scenario restaurants_in_sf       # filter scenarios by substring (repeatable)
$BIN --variant gemma4 --scenario compare_musk_bezos   # combine filters
```

Exit codes: 0 = every scenario had at least one passing variant; 1 = some scenario failed on every variant; 2 = no matching variant weights cached (or argv error); 3 = harness threw.

Scenarios + fixtures live in `ios/MCPZimEval/EvalHarness.swift`. Add a new scenario by extending `EvalHarness.scenarios` and dropping a fixture builder.

## gemma-smoke (headless macOS CLI)

A standalone CLI harness at `tools/gemma-smoke/` for iterating on the Gemma-4 + MCPZim prompt stack on Mac. Does NOT run via `swift run` — MLX can't find its `.metallib` because `swift build` skips the Metal compile step that Xcode's build system runs. Build with xcodebuild and run the produced binary directly:

```sh
cd tools/gemma-smoke

# Build once (slow — compiles MLX + metallib + Gemma core). Produces:
#   .build-xcode/Build/Products/Debug/GemmaSmoke
#   .build-xcode/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib
xcodebuild -scheme GemmaSmoke \
  -destination 'generic/platform=macOS' \
  -derivedDataPath .build-xcode \
  -configuration Debug build

# MUST run from the Debug dir so the metallib next to the binary resolves.
cd .build-xcode/Build/Products/Debug
./GemmaSmoke "your prompt here"
```

First run downloads Gemma-4-4b-it-4bit into `~/.cache/huggingface/` (~1.6 GB). Subsequent runs start in < 10 s.

Binary accepts a positional prompt arg; defaults to "tell me about baseball".

## cache-probe / prompt-experiment (KV-cache diagnostics)

Two env-var-selected modes on the same `GemmaSmoke` binary:

```sh
# Tokenizer stability + BPE round-trip check (single-turn scenario)
GEMMA_SMOKE_MODE=cache-probe ./GemmaSmoke "tell me about baseball"

# Multi-turn LCP comparison across preamble layouts — use this when
# you're about to change anything in composeSystemMessage /
# categoryHint / locationLineText. Saves ~1 min per iOS install cycle.
GEMMA_SMOKE_MODE=prompt-experiment ./GemmaSmoke "x"
```

`prompt-experiment` reports `LCP vs primed` / `LCP vs cached` / `hit=YES|no` for each layout in `tools/gemma-smoke/Sources/GemmaSmoke/PromptExperiment.swift`. Add a new `Layout<X>` function and register it in `run()`'s `for (label, runner)` list to test a candidate; no iOS rebuild needed.

## llm-smoke (Python cross-family model bench)

`tools/llm-smoke/` is the Python-mlx-lm / mlx-vlm harness suite used to
produce `ON_DEVICE_MODEL_REPORT_2026-04-23.md`. It runs any
`mlx-community` model (Gemma 3, Gemma 4, Qwen 3, Phi 4-mini, Nemotron,
gemma-3n, gemma-3-12b) against the 9 tool-selection scenarios from
`ios/MCPZimEval/EvalHarness.swift` and reports accuracy + prefill/decode
tok/s + peak memory at configurable preamble sizes. Everything runs on
mac — no iOS device needed. Managed via `uv` with Python 3.12.

### Setup

```sh
cd tools/llm-smoke
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install mlx-lm mlx-vlm psutil
```

### Scripts

| Script | Purpose |
|---|---|
| `eval.py MODEL_ID [all\|one]` | 9-scenario tool-call eval via mlx-lm (HF chat template + manual tool injection fallback for Phi-4-mini). |
| `eval_gemma4.py MODEL_ID` | Same eval via mlx-vlm — required for Gemma 4 (mlx-lm rejects its shared-KV layer weights). |
| `eval_gemma4_native.py MODEL_ID` | Gemma 4 eval using the Python port of `MCPZimKit.Gemma4PromptTemplate` + `Gemma4ToolFormat` — matches what ships in the iOS app. |
| `bench.py MODEL_ID --mode all` | Memory + warm-cache probe for one model (mlx-lm). |
| `bench_memory.py --sizes 7000,20000,40000` | Cross-model peak-memory bench at controlled preamble sizes. |
| `bench_memory_gemma4.py` | Same, but via mlx-vlm for Gemma 4. |
| `bench_kv.py MODEL_ID` | Gemma 3 KV-cache variant sweep (default / bounded_512 / kv_bits=8 / kv_bits=4). |

### Example runs

```sh
# 9-scenario eval for the app's default candidates
python eval.py mlx-community/Qwen3-4B-Instruct-2507-4bit
python eval.py mlx-community/gemma-3-4b-it-4bit
python eval_gemma4_native.py mlx-community/gemma-4-e2b-it-4bit

# Memory comparison 7k/20k/40k across the main three candidates
python bench_memory.py --which gemma3 --sizes 7000,20000,40000
python bench_memory.py --which qwen   --sizes 7000,20000,40000
python bench_memory_gemma4.py --sizes 7000,20000,40000

# KV cache variants on Gemma 3
python bench_kv.py mlx-community/gemma-3-4b-it-4bit
```

### Shared pieces

- `gemma4_format.py` — Python port of the Gemma 4 custom mini-format (`<|tool_call>call:NAME{k:v,…}<tool_call|>` + `<|"|>…<|"|>` quoted strings + nested objects). Used by `eval_gemma4_native.py` so the mac eval matches the Swift template byte-for-byte.
- Scenario + tool definitions live in `eval.py` (`CASES`, `TOOLS`). Every other script imports from there — change the tool surface once.

### Reproducing the scorecard

See `ON_DEVICE_MODEL_REPORT_2026-04-23.md` for full methodology. Quick
reproduction:

```sh
source .venv/bin/activate
for m in \
  mlx-community/Qwen3-4B-Instruct-2507-4bit \
  mlx-community/gemma-3-4b-it-4bit \
  mlx-community/gemma-3-4b-it-qat-4bit \
  mlx-community/gemma-3-4b-it-4bit-DWQ \
  mlx-community/gemma-3n-E2B-it-lm-4bit \
  mlx-community/gemma-3n-E4B-it-lm-4bit \
  mlx-community/Phi-4-mini-instruct-4bit \
  mlx-community/Phi-4-mini-instruct-6bit \
; do python eval.py "$m" all ; done
python eval_gemma4_native.py mlx-community/gemma-4-e2b-it-4bit
python eval_gemma4_native.py mlx-community/gemma-4-e4b-it-4bit
python bench_memory.py --sizes 7000,20000,40000
```

## Streetzim viewer hook (window.streetzimRouting)

`RouteWebView.swift` and `PlacesWebView.swift` inject JS that calls into `window.streetzimRouting` on the embedded streetzim viewer to drive routing, drive-mode, multi-place pins, coverage rings, and camera/chrome. The hook is exposed at the end of `initRouting()` in `../streetzim/resources/viewer/index.html`. All the backing state (`lastRoute`, `modeBtns`, `setOriginFromLatLon`, `graph`) lives inside that function, so the window object is the only seam.

Current hook surface (`version: 3`):

- `setOrigin(lat, lon, label)` / `setDest(lat, lon, label)` — geocode a coord to the nearest graph node, drop a marker, trigger route compute if both ends are set.
- `loadGraph()` — kick off the lazy graph fetch (the viewer ordinarily does this on first toggle click).
- `clickMode(mode)` — low-level, fires the routing-panel mode button click handler (toggles).
- `enterDriveMode(mode, origLat, origLon, destLat, destLon)` — atomic: loadGraph → set origin+dest → await route → `driveMode.enter()` directly, sync go-row chrome. Returns a Promise. **Preferred path for host containers.**
- `exitDriveMode()` / `switchDriveMode(newMode)` — explicit control.
- `clearRoute()` — removes markers, line, go-row.
- `showPlaces(places, opts)` / `clearPlaces()` — multi-pin display with popups + optional `fitBounds`. Each place: `{lat, lon, label, description?, color?}`.
- `showRadiusRing({lat, lon, radiusKm, color?})` / `clearRadiusRing()` — dashed coverage circle for "near me" queries.
- `flyTo({lat, lon, zoom, pitch, bearing, duration})` — camera control.
- `setChromeVisibility({search, controls, panel})` — hide/show the viewer's own UI when the host overlays its own.
- Getters: `graphReady`, `hasRoute`, `driveActive`, `lastEnteredMode`.

Updating the hook requires rebuilding the streetzim ZIM so the patched `index.html` ships inside the ZIM blob the app reads:

```sh
cd ../streetzim && ./build_world_and_us.sh          # or a narrower regional build script
```

Until the rebuilt ZIM replaces the one on the device, calls to missing-method branches time out silently (the iOS JS polls for 30 s and logs the timeout via the WebView console bridge — the plain map still renders). The iOS side always feature-detects (`typeof window.streetzimRouting.enterDriveMode === "function"`) and falls back to the older step-by-step path when a capability is missing, so the app never hard-fails on an older ZIM.

The drive PWA at https://streetzim.web.app ships the same viewer via the predeploy `scripts/sync-drive-viewer.sh` — `firebase deploy --only hosting` from `../streetzim/` refreshes both without rebuilding ZIMs.

## iOS chat-bubble map rendering (RouteWebView + PlacesWebView)

Two parallel `WebView`-backed SwiftUI views render tool results against the streetzim viewer:

- **`RouteWebView`** — triggered by `traceHasRoute` for `plan_driving_route`, `route_from_places`, `show_map`. Draws the polyline, exposes Drive/Walk/Bike pills that call `enterDriveMode` on tap.
- **`PlacesWebView`** — triggered by `traceHasPlaces` for `near_named_place`, `near_places`, `nearby_stories`, `nearby_stories_at_place`. Parses the tool's `results` / `stories` array into `{lat, lon, label, description}` and renders via `showPlaces` + `showRadiusRing` around the search origin.

Both live only for the **newest assistant message** (see `isLatestAssistant` in `ChatView.swift::MessageRow`). Older traces collapse to a `MapPlaceholder` chip — a live WKWebView + MapLibre instance is ~300–500 MB of Metal buffers, so stacking them across a dozen tool calls reliably trips the iPhone's 6 GB jetsam cap mid-generation.

## Qwen 3.5 hybrid-cache cannot reuse across turns

Qwen 3.5 4B forces a full prefill (~13 s on a 7k-token preamble) every
turn because its hybrid attention (MambaCache + KVCacheSimple) trips
`broadcast_shapes (128,256) vs (129,256)` on partial-prefix reuse. Root
cause + upstream status + mitigation options documented separately in
`QWEN35_HYBRID_CACHE.md`. The short answer: we're pinned at
`mlx-swift-lm` 3.31.3 which is HEAD; the upstream fix
(mlx-swift-lm#157) hasn't been written yet. For fast multi-turn on
device, prefer Qwen 3 4B (non-hybrid, same 9/9 eval score).

## Model picker — on-device candidates

`ChatSession.init()` builds the provider list every launch. The current
(2026-04-23) picker contents:

| id | Display | Weights HF repo | Template | Notes |
|---|---|---|---|---|
| `gemma4-e2b-it-4bit` | Gemma 4 E2B (4-bit · multimodal) | `mlx-community/gemma-4-e2b-it-4bit` | `Gemma4Template` | App default; vision+audio encoders stay resident (~750 MB tax). |
| `gemma4-e2b-it-4bit-text` | Gemma 4 E2B Text (4-bit · text-only) | `mlx-community/Gemma4-E2B-IT-Text-int4` | `Gemma4Template` | Weaker tool-calling under long preambles; kept for A/B. |
| `gemma3-4b-it-text-4bit` | Gemma 3 4B IT (4-bit · text) | `mlx-community/gemma-3-text-4b-it-4bit` | `Gemma3Template` | Benched 7/9 on mac eval; dense (no Qwen 3.5 hybrid bug). Text-only checkpoint — the multimodal `gemma-3-4b-it-4bit` weights mismatch mlx-swift-lm 3.31.3's `Gemma3TextModel` `o_proj` shape. |
| `qwen3-4b-4bit` | Qwen 3 4B (4-bit) | `mlx-community/Qwen3-4B-4bit` | `QwenChatMLTemplate` | Prior baseline. |
| `qwen35-4b-4bit` | Qwen 3.5 4B (4-bit) | `mlx-community/Qwen3.5-4B-MLX-4bit` | `QwenChatMLTemplate` | Forces full prefill per turn — see above. |
| `qwen3-1-7b-4bit` | Qwen 3 1.7B (4-bit) | `mlx-community/Qwen3-1.7B-4bit` | `QwenChatMLTemplate` | Small-slot fallback for ≤4 GB iPhones. |
| `gemma3-12b-it-text-4bit` *(mac only)* | Gemma 3 12B IT (4-bit · text · mac) | `mlx-community/gemma-3-text-12b-it-4bit` | `Gemma3Template` | 9/9 on eval at the non-text-only QAT variant; peak 9–13 GB on mac. Gated `#if os(macOS)` in ChatSession.init. Text-only checkpoint for mlx-swift-lm compatibility. |

`Gemma3Template` is a sibling of `QwenChatMLTemplate` — same JSON-in-tags
tool-call convention, different turn markers
(`<start_of_turn>` / `<end_of_turn>` vs `<|im_start|>` / `<|im_end|>`).
Gemma 3 isn't natively tool-tuned; the model learns the format from the
in-system demonstration the template emits.

When adding a new model: build a `ModelTemplate`-conforming struct for
its family if one doesn't exist, append a `Gemma4Provider` entry in
`ChatSession.init()` (despite the name, that class is the generic MLX
provider — the template slot is what differs across families), and pass
the `template:` argument. No other wiring changes — the adapter + router
are template-agnostic.

## Preemptive memory guard before MLX generate()

MLX's Metal backend doesn't surface command-buffer errors as Swift errors — when the GPU runs out of memory mid-eval the underlying C++ throws and the process terminates before any Swift `catch` can fire. `ChatSession.runGenerationLoop` checks `os_proc_available_memory()` at the top of each iter and refuses to start a new prefill/sample if headroom is below ~700 MB, surfacing a chat error message instead of an `abort_trap`.

## "Send Debug Report" button (debug pane)

The debug pane has a **Report** button (next to Copy / Clear) that emits the current chat messages + live debug log through `os.Logger` as a base64-encoded chunked stream. When `ios/scripts/mcp-logs.sh` is running (required — that's the transport), the chunks land in `/tmp/mcp-syslog.log` and the reassembler script pulls them back out:

```sh
# after the user taps Report on the phone
bash ios/scripts/mcp-report.sh latest
#   → writes /tmp/mcpzim-debug-reports/<hash>.json, prints a preview
```

Other commands:
- `mcp-report.sh list` — every `(hash, chunks, bytes, complete?)` tuple seen in the buffer.
- `mcp-report.sh pull <HASH>` — reassemble a specific (older) report by hash.

The button also clears `debugEntries` after emission so the next bad query produces a clean report. The button flashes `Sent · <HASH>` for ~2.5 s so the user can tell you which one to fetch ("crappy results, hash AB12").

Wire format lives in `ios/MCPZimChat/Chat/DebugReport.swift`. The payload type (`SerializedDebugReport`) is Codable + flat, so `jq` works out of the box on the reassembled JSON.

## Fast-path intent routing (skip the LLM entirely)

On Qwen 3.5 4B a single turn eats ~13 s of prefill before the first token streams. Queries that are already structurally identical to a tool call shouldn't pay that — we pattern-match them in `IntentRouter.classify()` (swift/Sources/MCPZimKit/IntentRouter.swift) and dispatch the tool directly. Current fast paths:

| Query shape | Tool | Notes |
|---|---|---|
| `what is here` / `where am I` / `what's around me` | `what_is_here` | no args — MCP adapter fills `lat`/`lon` from `hostStateProvider` at dispatch time |
| `<cat> near me / around here` | `near_places` | needs GPS; ordered AFTER `what_is_here` so "what is around me" doesn't land here with `kind="what is"` |
| `[polite]? directions/route/navigate to <X>` | `route_from_places` | `polite` accepts `give me`, `show me`, `get me`, `find me`, `tell me`, `please`, `can/could/would/will you`, `I need / I want / I'd like` |
| `[polite]? how do I get to <X>` / `take me to <X>` | `route_from_places` | same polite stripping |
| `<cat> in/near/at/around <Y>` | `near_named_place` | skips when the query starts with an interrogative (`what`, `where`, `how`, …) so `"where can I find bars in SF"` still falls to the LLM for phrasing |
| `compare <A> and/vs/versus/with/to <B>` | `compare_articles` | exactly the Case-2 dropped turn from the log (Qwen 3.5 emitted `" "` splice); fast-path sidesteps it |
| `tell me about <X>` / `what is X` / `who is/was X` / `give me an overview of X` | `article_overview` | last in the chain so specific patterns win first; guarded against navigational pronouns (`my`, `this`, `here`) so `"what is my next turn"` still falls to the LLM |

Each fast path also synthesises a reply in `IntentRouter.synthesize*` so the LLM can be skipped entirely — see `testSynthesize*` for the expected shapes. The caption alone isn't the full answer on article/compare queries (no narrated prose), but skipping 15 s of prefill + the malformed-JSON risk is worth the stylistic downgrade. Follow-up work: wire the article sheet to open automatically on `article_overview` / `compare_articles` so the user gets the full body alongside the caption.

Why polite prefixes matter: voice input routinely emits `"Give me directions to SF"`. Before the prefix strip, that fell to the LLM and Qwen 3.5 4B emitted malformed JSON (`{"name": "route_from_places",,"arguments": {...,,...,}}`) with 20% probability — a silent dropped turn after 15 s of waiting. See `dropped-request.log` for a capture.

## Defensive `zim` arg sanitization

`MCPToolAdapter.dispatch` strips a hallucinated `zim` filename
before any tool runs. When the model passes a `zim` value that
doesn't match (exact, then case-insensitive) any loaded ZIM
filename, the arg is removed and the underlying tool falls back to
"search all loaded ZIMs" — same behaviour as if the model had
omitted the arg entirely.

Real on-device capture (Qwen 3 4B 4-bit, see `dropped-request.log`
follow-up):

```
compare_articles({
  "titles": ["South Korea", "North Korea"],
  "zim":    "wikipediapedia_en_all maxi 2025-10.zim"   ← hallucinated
})
```

The actual loaded ZIM is `wikipedia_en_all_maxi_2025-10.zim`
(no duplicated "pedia", underscores not spaces). Without the
sanitiser, the tool returned an "unknown ZIM" error in 285 bytes,
and iter 1 mangled the entity names ("south kri", "north kri")
while writing prose about that error. With the sanitiser the same
emission falls through to a successful all-ZIMs search.

Pure logic in `MCPToolAdapter.sanitizeZim(_:loadedZimNames:)`,
exercised by `SanitizeZimArgTests` (no async or stub plumbing
required). The async wrapper `sanitizeOptionalZimArg` only awaits
`service.inventory()` when a `zim` arg is actually present, so
the typical no-`zim` dispatch path stays free.

We never *substitute* a near-miss. A "close" filename could
legitimately be a different ZIM (en vs es, full vs nopic, …) and
the wrong substitution would be worse than no pin. Strip-only is
the safer contract.

## Qwen tool-call JSON repair

`QwenChatMLTemplate.repairJSON` (swift/Sources/MCPZimKit) runs as a second-chance decode whenever strict `JSONSerialization` rejects a `<tool_call>` body. Repairs observed in the wild on Qwen 3.5 4B 4-bit:

1. **Double commas** → collapse to single. Real capture: `{"name": "route_from_places",,"arguments": …}`.
2. **Trailing comma** before `}` or `]`. Qwen leaks JavaScript habits (`{"kinds": ["bar",]}`).
3. **Whitespace-only string wedged before a bareword** (`["North Korea", " "South Korea"]` → `["North Korea", "South Korea"]`). The model spliced `" "` into the array at what should have been the next opening quote; collapsing the `" "` back to a single `"` re-joins the halves. Gated on the following char being a non-delimiter so intentional `" "` values survive.
4. **Brace-balance clip repair** — original behaviour, adds missing `}` when a turn was clipped by `<|im_end|>` mid-object.

All four are covered by `QwenClippedToolCallTests` with verbatim captures from `dropped-request.log` and a guard test to prove intentional whitespace-only values aren't clobbered.

## Fine-tune pipeline (Gemma 3 4B LoRA)

`tools/fine-tune/` turns a teacher's synthetic tool-calling trajectories into a LoRA-fine-tuned Q4_K_M GGUF for the iOS app.

```sh
cd tools/fine-tune

# 1. Generate training data (LM Studio running gemma-3-27b-it teacher
#    on this Mac, same GPU as MLX — they CAN'T run concurrently).
.venv/bin/python generate.py \
  --base-url http://192.168.68.68:1234/v1 \
  --model gemma-3-27b-it --n 550 --concurrency 4 \
  --out train.jsonl --fail-log train_fails.jsonl

# Optional: boost specific categories when tool coverage is uneven.
#   --boost narrate=8,current_place=3,wiki_search=4

# 2. Generate 2-turn chains (article_overview → get_article_section)
#    — the single-turn generator above emits zero get_article_section
#    calls, so chain scenarios in llama-smoke fail without these.
.venv/bin/python generate_chains.py --n 150 --out train_chains.jsonl

# 3. Train + fuse + convert + quantize.
cat train.jsonl train_chains.jsonl > train_combined.jsonl
bash finetune.sh train_combined.jsonl
# → ft-out/gemma3-4b-it-ft.Q4_K_M.gguf
```

Gotchas I burned time on:

1. **Prompt format must match eval/iOS exactly.** The student is trained in the distribution it'll see at inference, and the iOS Gemma3Template folds system + tool-block into the first user message (Gemma 3 has no `system` role). A training JSONL that uses `{"role": "system"}` renders to a different prompt after `apply_chat_template` and the FT silently regresses. `generate.py`'s `trajectory_to_jsonl` now emits the eval-matched fold-into-user preamble.

2. **LM Studio context exceeded with long prompts.** Teacher defaults to ~2k ctx; system prompt + tool descriptions + 1024 max_tokens blew the budget ~50% of the time. Trimmed SYSTEM_PROMPT to 334 tokens + `max_tokens=640`. Use `--fail-log` to catch future regressions.

3. **Metal GPU watchdog aborts long-sequence training.** `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` fires at peak mem ~35 GB when training on 700+ token examples at batch-size=4, layers=16. Workaround: lower `BATCH_SIZE=2` or `LORA_LAYERS=8`. Val loss plateaus by iter 200 anyway — iter-300 checkpoint is a fine stopping point if you crash at iter 340.

4. **mlx_lm fuse `--export-gguf` doesn't support gemma3.** Use the llama.cpp `convert_hf_to_gguf.py` path — but mlx's re-serialised tokenizer has a different hash than what the convert script recognises, throwing `NotImplementedError: BPE pre-tokenizer was not recognized`. `finetune.sh` copies the original tokenizer.{json,model,_config.json} from the HF base-model cache over the fused-hf copies after fuse. LoRA doesn't touch the tokenizer so this is safe.

5. **`--de-quantize` → `--dequantize`.** mlx-lm renamed the flag; old `finetune.sh` silently broke.

6. **`torch` missing for convert script.** llama.cpp's `convert_hf_to_gguf.py` imports torch at top-level but omits it from `requirements-convert_hf_to_gguf.txt` (platform-specific wheel). Install separately: `uv pip install torch`.

A/B eval against stock: `tools/llama-smoke/grid.py --models gemma3-4b --only Q4_K_M --kv q8_0/q8_0 --out GRID_RESULTS_FT_AB.md`. The fine-tuned model has a `ModelSpec` with `local_paths={"Q4_K_M": "…/ft-out/…"}`; `eval.py` grew `--local-path` for this.

## Why this exists

Every time the conversation compacts I forget how to invoke the macOS tools because the knowledge lives in scrolled-off chat, not in a file. If you add a new target or find a working incantation, add a section here.
