# On-Phone TTS Comparison for Apple Platforms

**Last updated:** July 15, 2026
**Target:** iPhone 17 Pro Max, with an on-device LLM resident
**Status:** Mac screening complete; iPhone latency, jetsam, thermal, and listening gates remain

This document is intentionally project-independent. It can be copied into any
Apple-platform project that uses Kokoro or needs low-latency, offline speech.
The measurements identify which engines deserve phone testing; they are not
presented as iPhone measurements.

## Decision

Use **Supertonic 3 ANE INT8** as the leading phone candidate when TTS must run
beside an on-device LLM. On the Mac screening host it generated an 8.32-second
utterance in 0.28 seconds when warm, peaked at 30–33 MB, and remained at about
28 MB while loaded.

Keep the existing Kokoro implementation available until listening confirms
that Supertonic's voice and prosody are acceptable. If preserving Kokoro's
voice identity is mandatory, the 7-stage Kokoro Core ML/ANE implementation is
worth further work, but its roughly 1.45 GB peak and repeated ANE compiler
diagnostic make it a poor co-resident partner for a large phone LLM today.

Use Supertonic INT4 only when saving about 93 MB of model storage is more
important than retaining INT8 precision. INT4 did not improve runtime latency
or memory in these tests.

## Compared implementations

| Name in this document | Runtime | Tested voice | Output |
|---|---|---|---|
| Current Kokoro MLX | Kokoro v1.0 through KokoroSwift/MLX and Metal | `af_heart` | 24 kHz mono |
| Kokoro Core ML/ANE | FluidAudio `KokoroAneManager`, 7-stage Core ML chain | `af_heart` | 24 kHz mono |
| Supertonic 3 INT8 | FluidAudio `Supertonic3Manager`, ANE-bucketed INT8 vector estimator | `F1` | 44.1 kHz mono |
| Supertonic 3 INT4 | Same pipeline with ANE-bucketed INT4 vector estimator | `F1` | 44.1 kHz mono |

The Core ML tests pin **FluidAudio 0.15.5**. Re-run the benchmark before
upgrading because model placement, chunking, quantization defaults, and public
APIs can change.

## Measurement result

### Host and workload

- Mac Studio, Apple M1 Ultra, 128 GB RAM
- macOS 26.5.2
- Release builds, arm64
- Model files already downloaded
- Fresh process per backend
- `task_vm_info.phys_footprint` sampled every 10 ms
- Full generated chunk returned before the synthesis timer stopped
- One-second idle sample with the manager still resident, then cleanup where supported
- Test text: `Dr. Vladimir V. Putin attended Leningrad State University; later, he joined the KGB in 1975.`

The produced clips are similar but not identical in duration, so both latency
and real-time factor matter.

### Latency

| Backend | Audio duration | Initialization | First synthesis in fresh process | Immediate repeat | Warm RTFx | First-ever preparation observed |
|---|---:|---:|---:|---:|---:|---:|
| Current Kokoro MLX | 8.43 s | 0.16–0.39 s | 1.09–6.30 s | 0.40–0.44 s | 19.3–21.3× | 8.15 s fully uncached |
| Kokoro Core ML/ANE | 9.60 s | 1.63–1.75 s | 1.49–5.16 s | 0.33–0.35 s | 27.2–29.6× | 5.55 s |
| Supertonic 3 INT8 | 8.32 s | 0.17–0.19 s | 0.33–0.38 s | 0.28 s | 29.4–30.2× | 10.10 s |
| Supertonic 3 INT4 | 8.32 s | 0.17–0.18 s | 0.35–0.37 s | 0.29–0.30 s | 28.0–29.4× | 4.88 s |

“First-ever preparation” includes lazy model loading, device specialization,
and other one-time work seen on the first execution. It should be staged after
model installation or during an explicit voice-setup step, not during the
first spoken answer.

MLX cold latency was notably variable even with model files present. Its peak
footprint was more repeatable.

### Memory and storage

| Backend | First-render peak | Highest paired-render peak | Resident after 1 s idle | After cleanup | Selected model storage |
|---|---:|---:|---:|---:|---:|
| Current Kokoro MLX | 3,463–3,688 MB | 4,204–4,254 MB | 423–450 MB | Not unloaded in harness | 326 MB |
| Kokoro Core ML/ANE | 1,378–1,386 MB | 1,435–1,451 MB | 937 MB | 600–617 MB | about 190 MB |
| Supertonic 3 INT8 | 27–32 MB | 30–33 MB | 28 MB | 24–28 MB | about 253 MB |
| Supertonic 3 INT4 | 30 MB | 34–36 MB | 32 MB | 28–31 MB | about 160 MB |

Storage is the selected variant plus shared model assets, not the combined test
cache containing both INT8 and INT4. Framework binary size, downloaded archive
overhead, and application assets are excluded.

The process footprint is useful for screening, but Core ML/ANE driver charging
can differ on iOS. The iPhone test must measure the full app with the LLM,
context cache, retrieval data, speech recognition, playback buffers, and TTS
resident together.

## Interpretation

### Supertonic 3 INT8

Relative to current Kokoro MLX in the paired Mac runs, INT8 was:

- about 30–36% faster for a warm chunk;
- roughly 127–140× lower at peak, depending on which run is compared;
- about 15–16× lower while resident and idle;
- about 22% smaller in selected model storage.

Its low resident footprint changes the lifecycle decision: keep the manager and
voice style loaded throughout a conversation, rather than repeatedly unloading
them. On a memory warning or long background transition, cleanup remains cheap.

The tested model is multilingual, produces 44.1 kHz audio, and publishes ten
reference voice styles (`F1`–`F5`, `M1`–`M5`). Performance should be essentially
the same across those small style files, but quality must be judged per voice.

### Supertonic 3 INT4

INT4 reduced selected storage from roughly 253 MB to 160 MB. It did not reduce
runtime memory or improve steady-state speed. A signal-level silence scan also
showed a more noticeably altered pause pattern than INT8 on the punctuation
sample. That is not by itself a quality verdict, but it is enough to prefer
INT8 until blind listening says otherwise.

### Kokoro Core ML/ANE

This path preserves the tested `af_heart` voice and lowered peak memory by
about 3× compared with current MLX. Warm inference was fast. However:

- resident idle memory was still about 937 MB;
- cleanup left about 600 MB in the process;
- cold-process synthesis remained variable;
- every run emitted an `MILCompilerForANE` diagnostic during teardown even
  though audio generation completed.

Do not ship this path until the teardown diagnostic and on-phone memory charge
are understood. It may still be useful in Kokoro-only projects that do not
keep a large LLM resident.

### Current Kokoro MLX

The current implementation remains the perceptual reference and provides the
broad Kokoro voice catalog, but synthesis can transiently consume several
gigabytes. Projects that retain it should serialize large LLM prefill and TTS,
enforce a memory-headroom gate, clear transient MLX allocations after every
chunk, and avoid preparing another utterance while the prior allocation spike
has not settled.

## Listening gate

Performance does not justify a voice regression. Use a blind, loudness-matched
comparison before changing the default backend.

### Preparation

1. Generate every sample from the exact text that the app passes to the engine,
   including normalization and chunking.
2. Convert copies to a common sample rate and approximately the same integrated
   loudness. The mcpzim comparison used 48 kHz mono at about -18 LUFS and -2 dBTP.
3. Randomize filenames so the listener does not know the backend.
4. Keep native-level exports too; they reveal gain, clipping, and noise-floor
   differences hidden by loudness matching.

Example listening-copy command:

```sh
ffmpeg -i input.wav \
  -af loudnorm=I=-18:TP=-2:LRA=11 \
  -ar 48000 -ac 1 -c:a pcm_s16le output-listening.wav
```

### Minimum prompt set

Use short chunks because conversational systems start speaking clause by
clause. Include at least:

1. `Dr. Vladimir V. Putin attended Leningrad State University; later, he joined the KGB in 1975.`
2. `He later worked in St. Petersburg. What happened next?`
3. `When was the Alamo? How many people died there?`
4. `On July 4, 1776, the declaration was adopted at approximately 2:00 p.m.`
5. `LIGO detects gravitational waves—ripples in spacetime—using laser interferometers.`
6. `Continue on Hamilton Dr. for 0.2 mi, then turn onto Main St.`
7. A 150–250 character multi-sentence answer with commas, parentheses, and a quotation.
8. Names and places drawn from the product's actual Wikipedia or map content.

The `Dr.` and `St.` cases are important. An application preprocessor can ruin
pronunciation before the TTS model sees the text; for example, a global road
abbreviation replacement can turn “Dr. Vladimir” into “Drive Vladimir” or
“St. Petersburg” into “Street Petersburg.” Test normalization separately from
the model.

### Score sheet

Rate each sample from 1–5 and retain comments:

| Criterion | Question |
|---|---|
| Naturalness | Does it sound like continuous human speech rather than joined phrases? |
| Punctuation | Are commas, semicolons, questions, and sentence endings paced correctly? |
| Pronunciation | Are names, initials, acronyms, dates, and numbers correct? |
| Stability | Are there skipped words, repetitions, clicks, or sudden timbre changes? |
| Conversational fit | Does it sound appropriate for short factual follow-ups? |
| Listening fatigue | Would several minutes of answers remain pleasant? |
| Voice preference | Is at least one available voice as acceptable as the current default? |

Do not reject Supertonic after testing only `F1`. If its model-level prosody is
acceptable but the timbre is not, evaluate all ten reference styles before
making the backend decision.

## Reusable integration pattern

### Pin the runtime

For the exact API exercised here:

```swift
dependencies: [
    .package(
        url: "https://github.com/FluidInference/FluidAudio.git",
        exact: "0.15.5"
    )
]
```

Do not silently float to a newer FluidAudio release in production. Upgrade in
a dedicated change with a device benchmark and listening pass.

### Supertonic INT8 setup

```swift
import FluidAudio

let manager = Supertonic3Manager(
    directory: modelDirectory,
    vectorEstimator: .aneBucketed(.int8)
)

try await manager.initialize()
let voice = try await Supertonic3ResourceDownloader.loadVoiceStyle(
    .f1,
    directory: modelDirectory
)

let result = try await manager.synthesize(
    text: clause,
    language: "en",
    style: voice
)
// result.samples is 44.1 kHz mono Float32 PCM.
```

### Lifecycle

1. Download and verify model assets before an offline session.
2. Run one representative synthesis during installation/setup to pay lazy
   device preparation outside the first conversation.
3. Keep the manager and chosen voice style resident while a conversation is
   active.
4. Synthesize one complete clause at a time; queue playback and then prepare
   the next clause serially.
5. Cancel queued synthesis immediately when the user interrupts.
6. Call cleanup on memory pressure or a long background transition.
7. Keep an OS TTS fallback for missing/corrupt assets and unsupported devices.

### Conversational chunking

- Start after a complete short clause, not after an arbitrary token count.
- Treat abbreviations such as `Dr.`, `St.`, `U.S.`, and initials as non-terminal
  periods when finding sentence boundaries.
- Prefer punctuation boundaries around 80–180 characters.
- Allow only one synthesis operation at a time; playback can overlap the next
  synthesis once memory measurements show it is safe.
- Cap very long spoken responses while leaving the full answer visible.
- Normalize Markdown, units, road names, dates, and decimals before chunking,
  with regression tests for both factual conversation and navigation.

## iPhone 17 Pro Max validation plan

Run the final build on the target phone. Mac numbers are not a release gate.

### Scenarios

- TTS alone, cold and warm
- TTS with the production LLM loaded but idle
- TTS overlapping LLM decoding
- TTS overlapping worst-case LLM prefill
- 8K, 16K, and maximum supported context/KV-cache states
- speech recognition and playback active
- repeated ten-turn conversation
- foreground, background transition, and interruption
- nominal, fair, and serious thermal states
- airplane mode after all assets are installed

### Record

- time from clause-ready to playback-ready;
- time from clause-ready to first audible buffer;
- synthesis RTFx and produced audio duration;
- process physical footprint before, peak, resident idle, and after cleanup;
- memory warnings and jetsam logs;
- ANE/GPU/CPU utilization and energy impact;
- model download and installed storage;
- thermal state and performance degradation over repeated turns;
- cancellation latency and whether interrupted audio leaks into the next turn.

### Suggested release gates

- No jetsam or memory warning in the largest supported LLM-context scenario.
- Warm short-clause synthesis under 500 ms.
- TTS incremental peak under 250 MB when the LLM is resident.
- First audible playback within one second of a complete clause becoming ready.
- No network access after assets are installed.
- No pronunciation regression on the product-specific prompt set.
- Blind-listening quality within 0.25 points of the current engine's average,
  with no individual category more than 0.5 points worse.

## Licensing and distribution

- Kokoro-82M identifies its weights as Apache 2.0.
- FluidAudio identifies its framework as Apache 2.0.
- Supertonic 3 weights use BigScience OpenRAIL-M. The license permits paid and
  commercial products and grants royalty-free rights to use, sell, sublicense,
  host, and distribute the model, but it is not restriction-free. Distribution
  requires a copy of the license, retained notices, modification notices where
  applicable, and enforceable propagation of the license's prohibited-use
  terms. Generated content must be intelligibly disclosed as machine-generated.
  See the [exact upstream model license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE).
- The built-in F1–F5 and M1–M5 styles ship with the upstream model. Treat any
  separately purchased or custom voice style as an additional licensed asset
  and review its terms independently.

Model and framework licenses are separate. Repeat the review whenever a model
revision, voice pack, or runtime version changes.

## Sources and provenance

- [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
- [FluidAudio repository](https://github.com/FluidInference/FluidAudio)
- [FluidAudio 0.15.5 model documentation](https://github.com/FluidInference/FluidAudio/blob/v0.15.5/Documentation/Models.md)
- [FluidAudio 0.15.5 API documentation](https://github.com/FluidInference/FluidAudio/blob/v0.15.5/Documentation/API.md)
- [Supertonic 3 Core ML model card](https://huggingface.co/FluidInference/supertonic-3-coreml)
- mcpzim raw benchmark notes: `ios/MCPZimCoreMLTTSBench/README.md`
- mcpzim Core ML harness: `ios/MCPZimCoreMLTTSBench/CoreMLTTSBenchMain.swift`
- mcpzim current-engine harness: `ios/MCPZimTTSBench/TTSBenchMain.swift`

When this document is copied to another project, preserve the benchmark date,
host, version pins, and “Mac screening” qualification. Add a new result section
rather than overwriting these measurements with a different device or runtime.
