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

## Vendored Gemma4 fork (KV-cache quantization)

`ios/project.yml` pins `Gemma4SwiftCore` to `path: LocalPackages/Swift-gemma4-core`. This is a fork of https://github.com/yejingyang8963-byte/Swift-gemma4-core.git with a single patch: `Sources/Gemma4SwiftCore/Layers/Gemma4TextAttention+Forward.swift` branches on `QuantizedKVCacheProtocol` so MLX's 4-bit groupwise `QuantizedKVCache` runs end-to-end. Without the fork, MLX asserts at `cache.update` on phones.

Gate: `DeviceProfile.useQuantizedKVCache` (true on iPhone/iPad tiers, false on Mac). Wired into `Gemma4Provider.generate()` via `kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0`. Full-attention layers swap after prefill; sliding (`RotatingKVCache`) layers stay FP16 because MLX doesn't yet support rotating-quantized.

If you bump the upstream version: `rm -rf ios/LocalPackages/Swift-gemma4-core && git clone https://github.com/yejingyang8963-byte/Swift-gemma4-core.git ios/LocalPackages/Swift-gemma4-core && rm -rf ios/LocalPackages/Swift-gemma4-core/.git`, then re-apply the attention-forward patch (see `KV_CACHE_COMPRESSION.md` Option 1 for the diff shape).

After bumping, force SPM to re-resolve against the vendored path:

```sh
rm -rf ios/build/SourcePackages && \
  xcodebuild -project ios/MCPZimChat.xcodeproj -scheme MCPZimChat \
  -derivedDataPath ios/build -resolvePackageDependencies
```

`tools/gemma-smoke/Package.swift` still uses the upstream URL (the harness only tests FP16 paths — LCP + tokenizer — so the fork patch isn't needed there).

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
