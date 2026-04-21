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

## Why this exists

Every time the conversation compacts I forget how to invoke the macOS tools because the knowledge lives in scrolled-off chat, not in a file. If you add a new target or find a working incantation, add a section here.
