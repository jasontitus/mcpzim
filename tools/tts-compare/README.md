# Isolated speech comparison

This macOS Release harness compares the actual Swift/Core ML runtimes without
loading Bonsai, MLX, playback, or speech recognition. Run one backend per process
and one process at a time. It never opens audio hardware. `--save` writes WAV
files for a separate listening comparison.

The app's vendored FluidAudio is the default. To investigate a newer revision,
set `TTS_FLUIDAUDIO_PATH` to a separate checkout; this does not update the app.
Inflect is compiled in only when present in that checkout.

September 5 experiment: FluidAudio
`5c19d5e12320e22bbfb7a1877b089d2665a69add`, retaining the app's existing
`Supertonic3Constants.maxChunkLengthLatin = 96` patch. All other upstream
source is unchanged. See the [research and results](../../docs/ON_DEVICE_TTS_REVIEW_2026-09-05.md).

```sh
TTS_FLUIDAUDIO_PATH=/path/to/pinned/FluidAudio \
  swift build --package-path tools/tts-compare \
  --scratch-path /private/tmp/zimfo-tts-compare/build -c release

codesign --force \
  --sign 'Developer ID Application: Jason Titus (A6G8H8NGAM)' \
  --identifier org.mcpzim.ZimfoTTSCompare --timestamp=none \
  /private/tmp/zimfo-tts-compare/build/release/ZimfoTTSCompare
codesign --verify --strict \
  /private/tmp/zimfo-tts-compare/build/release/ZimfoTTSCompare

/private/tmp/zimfo-tts-compare/build/release/ZimfoTTSCompare kokoro-ane --prepare
/private/tmp/zimfo-tts-compare/build/release/ZimfoTTSCompare kokoro-ane --save
```

Backends: `kokoro-ane`, `kokoro-ane-cpu-tail`, `supertonic-int8`, `supertonic-int4`,
`pocket-fp16`, `pocket-int8`, `pocket-ane`, `pocket-ane-state`, `inflect-micro`, `inflect-nano`,
`inflect-micro-cpu`. Options: `--cache DIR`, `--output DIR`, `--turns 1..30`,
`--text TEXT`, `--voice NAME`, `--steps 1..16`, `--prepare`, `--save`.
Voice and step options apply only to the backends that support them.

For the separate staged Kokoro prototype, set `TTS_KOKORO_PIPELINE_PATH` to
a directory containing that repository's Swift package. Its basename must
differ from `swift` to avoid a package-identity collision with MCPZimKit.
The September experiment uses source revision
`66d8cf5108cce0991b8868b01b4d8a8b2e98881d`, copied unchanged to a temporary
`KokoroPipeline` directory. `kokoro-staged --fixture FILE --cache DIR`
expects the upstream seven-second input JSON, `DIR/hnsf_weights.json`, and
four model packages in `DIR/coreml` (duration t128, f0ntrain t280,
decoder-pre 7s, har-post 7s). It measures prepared phoneme tokens, not a full
text frontend, and repeats one input. Keep this diagnostic separate from
the multi-input raw-text table. Its current initializer recompiles packages.

## Measurement contract

- Separate asset preparation from measured runs. The first inference can still
  incur lazy Core ML compilation. Repeat in a fresh process to measure persistent
  compiled-cache behavior, and retain both results.
- Default ten-turn corpus spans five inputs, with an immediate first-input repeat.
  Results are a small engineering screen, not statistically representative quality
  or latency percentiles. Inputs are fixed benchmark prose, not generated app answers.
- `first_pcm_seconds` measures text API entry to the first nonempty PCM chunk.
  Supertonic uses the app's 94-character boundary policy and continuation comma;
  Pocket returns streamed 80 ms frames; Kokoro/Inflect return the full waveform.
- Pocket uses temperature 0.3, matching current upstream English preference.
  The `gpu` placement name allows Core ML `.all` for parts of the pipeline;
  the ANE variant pins FlowLM to `.cpuAndNeuralEngine`. Both retain CPU Mimi.
  These labels describe configuration, not a measured energy result or a guarantee
  that every operation executes on one processor.
- Kokoro's CPU-tail option keeps the five neural stages on CPU/ANE and puts
  noise plus iSTFT on CPU. It is equivalent to newer FluidAudio's `aneTailCpu`
  preset, expressed with the constructor so the app's older vendor can compile it.
- Physical footprint is sampled every 10 ms plus phase endpoints, in **MiB**.
  This is process `task_vm_info.phys_footprint`, not downloaded weights, RSS,
  total machine RAM, or isolated ANE compiler-service memory. Short peaks may be
  missed. Measure the full app separately to assess Bonsai coexistence.
- Every turn requires nonempty, finite, non-silent PCM. RMS, amplitude, and
  clipping are recorded; these checks do not prove pronunciation or naturalness.
- Streaming results include the extra initial buffer needed to avoid a gap at
  real-time playback speed. Timed runs generate as fast as possible without
  playback backpressure. The aggregate PCM buffer is retained until that turn
  is logged/saved, then released.
- `settled` and `cleanup` follow one-second waits. Upstream singleton caches may
  remain after manager cleanup, so process exit is the strongest isolation boundary.

Save stdout and stderr together to preserve warnings alongside JSON lines.
Do not compare timing runs while another model is compiling or synthesizing.
Audition samples afterward, at matched loudness, separately from timing.
