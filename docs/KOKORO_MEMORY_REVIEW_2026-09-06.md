# Kokoro memory optimization and implementation review

## Result and scope

The existing MLX Kokoro pipeline now evaluates completed vocoder residuals
eagerly and releases reusable buffers between its two upsampling stages.
Weights, arithmetic precision, phonemes, voice style, speed, sample rate,
sentence boundaries and playback scheduling are unchanged. A service retains
only its selected voice embedding instead of the whole voice pack.

On this Mac Studio (M1 Ultra, macOS 26.6.2), peak process physical footprint
fell 48–60% across four real-text cases. Every Float32 sample was identical
between baseline and optimized synthesis in both first and warm calls.
These are measured engineering results, not a phone memory guarantee or a
perceptual evaluation of different engines.

The existing 2,800 MiB Kokoro allowance plus 700 MiB reserve remains in place.
This round does **not** make the phone with 2,409 MiB headroom bypass its
fallback, and is not uploaded to TestFlight. Lowering the guard needs a
physical-phone run with Bonsai, imported Wikipedia/StreetZIM, recognition,
playback and memory pressure. The earlier pending voice-startup/UI work is
documented separately in [its review](VOICE_STARTUP_REVIEW_2026-09-06.md).

## Design decisions and adversarial self-review

1. **Residual evaluation checkpoints — kept.** MLX's lazy graph retained
   intermediate vocoder activations across many convolution/residual blocks.
   `MLX.eval(result)` completes each residual, preserving its full temporal
   context. An additional evaluation completes each upsampling stage.
   Challenge: evaluation can change execution scheduling, fusion or random
   consumption. Fixed-seed Float32 comparisons, including first synthesis,
   passed exactly. The generation lock already serializes Kokoro's pipeline;
   this change introduces no new tasks, shared mutable tensors or lock order.
   Checkpoints alone reduced the initial short-case peak from 4,112 to
   2,512 MiB, but left a substantial allocator-cache peak.

2. **Cache release at upsampling boundaries — kept.** After evaluation,
   `Memory.clearCache()` releases unused allocator buffers before the next
   large stage. It does not unload live weights or tensors. Challenge: this
   is still a process-wide cache operation and can cost reallocation time.
   Only two releases are added per synthesis, rather than clearing at every
   convolution. An initial experiment showed warm latency increasing from
   0.47 to 1.01 seconds for 8.425 seconds of audio; later runs varied. Do not
   promise a latency improvement. Bonsai uses llama.cpp's allocator, so its
   live prefix/KV state is not evicted by MLX cache clearing. Other MLX-model
   configurations still need coexistence latency testing.

3. **Retain one immutable voice — kept.** The initializer validates the
   requested voice and retains exactly that embedding. The existing reader
   still temporarily decodes the complete NPZ; this is a residency reduction,
   not selective ZIP loading. Challenge: picker changes must create a new
   service, as they already did because `voiceName` was immutable. Both US
   and British voices passed real-model parity; an absent voice exits the
   benchmark cleanly with code 3 before synthesis. The fetched pack contains
   28 entries, approximately 14 MiB uncompressed, so this saves about 13.5 MiB
   of live embeddings—not the stale 45 MiB estimate in earlier asset notes.

4. **Global allocator cap — rejected for production.** A benchmark-only
   32 MiB cache cap reduced the short-case peak to about 2,073 MiB without
   changing WAV bytes. Setting/restoring a global cap from a TTS worker could
   override another model's allocator policy. No production cache-limit or
   memory-limit mutation was added.

5. **Quantization, shorter clauses, lower sample rate — not used.** These
   introduce precision or prosody changes. Inspecting the downloaded
   safetensors header showed all 548 tensors are F32 despite the upstream
   repository name containing `bf16`; no conversion was performed.

6. **Benchmark validation — strengthened during review.** The original
   harness saved only warm 16-bit WAVs, which could hide first-call differences
   and small numerical changes. It now saves first and warm Float32 sidecars,
   checks finite/nonempty samples, seeds only the isolated benchmark RNG and
   records MLX active-memory peaks separately from physical footprint. The
   comparison script requires exact PCM, equal duration and non-silent output,
   keeps failed-process logs, and runs engines serially in fresh processes.
   These are self-reviews, not an independent-agent review.

## Final measurements

Release benchmark; four cases, two implementations, two calls per process
(16 syntheses). Same harness, selected-voice policy, assets and seed on both
sides; the retained pre-change KokoroSwift framework isolates the vocoder
change. No build ran concurrently with this final pass. Persistent Metal
compiler caches were already populated: “first” means first synthesis in a
fresh process, **not** first-ever installation. Fixed run order and a small
sample make latency comparisons indicative rather than statistically robust.

Memory is `task_vm_info.phys_footprint`, sampled every 10 ms and at endpoints,
in MiB. Brief peaks may be missed. GPU accounting, scheduling and settling
vary; MLX peak active bytes are a different metric, retained in raw logs.

| Case / voice | Audio | Baseline peak | Optimized peak | Reduction | Warm synthesis before → after |
|---|---:|---:|---:|---:|---:|
| Putin punctuation / af_heart | 8.425 s | 4,097.5 MiB | 1,628.9 MiB | 60.2% | 1.041 → 0.753 s |
| Schwarzenegger / af_heart | 9.125 s | 4,239.8 MiB | 1,679.2 MiB | 60.4% | 0.694 → 0.491 s |
| Shaw / bf_emma | 6.600 s | 3,605.4 MiB | 1,517.4 MiB | 57.9% | 0.314 → 0.331 s |
| Longer Napoleon prose / af_heart | 20.750 s | 6,033.3 MiB | 3,119.8 MiB | 48.3% | 0.906 → 0.920 s |

Peak is the maximum of first and warm calls. The longer input exercises the
existing text chunker, not a proposed larger iPhone streaming window.
Eight first/warm comparisons had zero maximum and RMS Float32 error.
Synthesis remains faster than playback on this Mac in every measured case.

[Structured measurements](benchmarks/kokoro-memory-2026-09-06/measurements.json),
[exact PCM checks](benchmarks/kokoro-memory-2026-09-06/parity.json),
[regression harness](../tools/tts-compare/check-kokoro-parity.py).
Raw logs are alongside those JSON files. WAV/Float32 files remain in
`/private/tmp/kokoro-memory-final`; they are intentionally not committed.

Validation: signed iOS Debug build and signature gate passed for
`com.tiltastech.zimfo`, team `A6G8H8NGAM`; `git diff --check` passed. No phone
playback, jetsam, thermal or background-transition claim is made from this
Mac-only optimization round. Existing conversation routing/retrieval was not
modified by these Kokoro changes.

## Other implementations and engines

The earlier [on-device comparison](ON_DEVICE_TTS_REVIEW_2026-09-05.md) already
contains 230 real synthesis turns across Core ML engines. Those measurements
are historical evidence from this workspace, not new benchmarks in this round.
Current primary-source checks add the following conclusions:

| Candidate | Evidence and adversarial assessment | Decision |
|---|---|---|
| [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml) | Seven stages, reported 80 MB fp16/int8-palettized export and 16.9× iPhone 16 Pro throughput. The demo uses precomputed phonemes, excluding G2P. Noise and tail retain FP32 because reduced precision caused quality problems. Small weight files do not establish peak memory. Its claim that discarded audio eliminates palette artifacts is insufficient: intermediate activations still feed the audible tail. | Worth a parity/phone prototype, not an automatic replacement. |
| [mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Notes/iphone-performance-notes.md) | Published Release iPhone bucket measurements and explicit compute placement. Warmed timings exclude cold compilation. Our prior seven-second, pre-tokenized prototype peaked at 785 MiB, but initialization recompiled packages (~48 s) and first synthesis cost ~7.5 s. | Best separate Core ML prototype to investigate after persistent compilation and a real text frontend. |
| [FluidAudio Kokoro ANE](https://github.com/FluidInference/FluidAudio/pull/848) | Prior local CPU-tail configuration peaked at 1,099 MiB versus 1,983 MiB with GPU tail. The upstream advisory still covers iOS 26.6 BNNS crashes; Mac success does not clear it. [Tail v2](https://github.com/FluidInference/FluidAudio/pull/868) fixes an audio reconstruction/output-level issue. | Do not make the phone default while that crash path remains unresolved. |
| [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) / [sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html) | Full-precision and quantized Kokoro options; ONNX offers a CPU route that can avoid GPU contention. The Python wrapper is not an iOS integration; sherpa offers the native route. Different phonemizers/voice indexing can change pronunciation despite the same model family. Model size is not an iPhone peak measurement. | Next useful CPU comparison: same v1.0 weights/voice, full precision first, matched phonemes where possible. |
| [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) | 100M parameters, streaming audio, upstream ~200 ms first chunk / ~6× M4 CPU throughput. Prior local Core ML ANE screen peaked at 494 MiB with 92 ms subsequent first-frame median. These are different runtimes and timing boundaries. It does not preserve Kokoro's exact voice. | Strongest different-engine candidate for a listening trial and phone streaming test. |
| [KittenTTS](https://github.com/KittenML/KittenTTS) | CPU/ONNX developer preview, 15–80M models and 25–80 MB advertised storage. Upstream notes issues with Nano INT8 and lists mobile SDK as roadmap work. No matched evidence here proves Kokoro-level naturalness or iOS peak memory. | Secondary candidate; evaluate Mini/Micro quality before favoring the smallest quantized model. |

Prior Inflect Micro CPU measurements were also small (~293 MiB), but its
single male voice and pronunciation frontend make it a less direct substitute
for the user's preferred Kokoro sound. See the earlier review for sources.

Kokoro ONNX wrapper code is MIT and Kokoro weights Apache-2.0; Kitten's project
is Apache-2.0. A future integration must retain the exact conversion, frontend
and voice asset licenses—especially independently distributed Pocket voices.
No new backend or model asset was bundled in this round.
