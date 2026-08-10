# Core ML TTS experiment

The reusable, project-independent comparison and phone validation plan lives
at [`docs/ON_PHONE_TTS_COMPARISON.md`](../../docs/ON_PHONE_TTS_COMPARISON.md).
This file stays focused on the mcpzim harness and its raw Mac runs.

This target compares FluidAudio 0.15.5 TTS backends in a process that does not
link the app's MLX runtime. It uses the same punctuation-heavy 92-character
utterance as `MCPZimTTSBenchCLI` and samples `task_vm_info.phys_footprint` every
10 ms during each measured phase.

## July 15, 2026 Mac Studio result

The Core ML rows are second, clean-process runs after one-time device
compilation. The MLX row is the range from two clean-process paired renders
with the same 10 ms footprint sampler. These are useful for choosing a backend,
but are not a substitute for iPhone 17 Pro Max jetsam and energy measurements.

| Backend | Audio | Init | First synthesis | Immediate repeat | Highest observed footprint | Resident after 1 s | After cleanup | Model storage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing Kokoro MLX | 8.43 s | 0.16–0.39 s | 1.09–6.30 s | 0.40–0.44 s | 4,204–4,254 MB | 423–450 MB | not unloaded | 326 MB |
| FluidAudio Kokoro ANE | 9.60 s | 1.63–1.75 s | 1.49–5.16 s | 0.33–0.35 s | 1,435–1,451 MB | 937 MB | 600–617 MB | 190 MB |
| Supertonic 3 ANE INT8 | 8.32 s | 0.17–0.19 s | 0.33–0.38 s | 0.28 s | 30–33 MB | 28 MB | 24–28 MB | about 253 MB |
| Supertonic 3 ANE INT4 | 8.32 s | 0.17–0.18 s | 0.35–0.37 s | 0.29–0.30 s | 34–36 MB | 32 MB | 28–31 MB | about 160 MB |

MLX's first synthesis is variable: a prior fully uncached run took 8.15 s. Its
footprint was much more repeatable: the first render peaked at 3,463–3,688 MB,
and an immediate second render raised the process peak to 4,204–4,254 MB before
MLX cache clearing returned it to 423–450 MB with the service still loaded.

The first ever Core ML synthesis also pays device compilation: 5.55 s for
Kokoro ANE, 10.10 s for Supertonic INT8, and 4.88 s for Supertonic INT4 in this
run. That cost should be staged after installation or during a non-speaking
setup phase.

Kokoro ANE repeatedly emitted an `MILCompilerForANE` diagnostic during
teardown even though synthesis succeeded, so it should not ship until that is
understood. Supertonic INT8 is the current candidate: INT4 saved about 93 MB of
storage but did not reduce runtime footprint and altered the punctuation pause
pattern more noticeably.

## Run

```sh
xcodegen generate
xcodebuild -project MCPZimChat.xcodeproj \
  -scheme MCPZimCoreMLTTSBenchCLI -configuration Release \
  -destination 'platform=macOS,arch=arm64' ONLY_ACTIVE_ARCH=YES build

build-tts-coreml/Build/Products/Release/MCPZimCoreMLTTSBenchCLI \
  --backend supertonic3-int8 --prepare-assets
build-tts-coreml/Build/Products/Release/MCPZimCoreMLTTSBenchCLI \
  --backend supertonic3-int8
```

The benchmark writes WAV comparisons to `/tmp/mcpzim-tts-output`.
