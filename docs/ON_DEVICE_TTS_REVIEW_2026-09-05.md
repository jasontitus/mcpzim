# On-device speech review — September 5, 2026

The user values Kokoro's voice quality, but needs speech to coexist with Bonsai
on a phone. This review checks current upstream releases and runs the actual
Swift inference implementations on the Mac before considering an app switch.
The app's default speech engine and vendored FluidAudio have not been replaced.

## What changed upstream

**Supertonic:** the current app already uses Supertonic 3. I found no newer
model generation. Supertone's July 23 notice announces the end of development
and official support for its open-source models; VoiceBuilder's announced end
date is August 31. The GitHub API still reported `archived: false` when checked,
so the announcement and repository flag should not be conflated. This creates
a maintenance reason to evaluate replacements, not evidence that an existing
offline installation stops working.
[Primary repository and service notice](https://github.com/supertone-inc/supertonic).

**FluidAudio:** the app vendors v0.15.5 (`19600a48`). The latest release is
v0.15.6, August 19. Comparing the app's upstream base to current main found
72 commits and no changes under the Supertonic implementation. Newer code
adds Inflect Micro/Nano v2 and other backends, and improves Kokoro normalization.
The Kokoro Tail v2 amplitude correction landed later on August 19, after the
release, so updating only to the release tag would omit that fix.
[Release](https://github.com/FluidInference/FluidAudio/releases/tag/v0.15.6),
[exact source comparison](https://github.com/FluidInference/FluidAudio/compare/19600a48...5c19d5e12320e22bbfb7a1877b089d2665a69add),
[Tail v2 correction](https://github.com/FluidInference/FluidAudio/pull/868).

**Pocket TTS:** Kyutai released software v3.1.0 on September 3. That does not
mean new English voice weights: the current English configuration still pins
the April 2026 pack. It is a 100M-parameter model with genuine audio streaming,
which can preserve longer-text context while returning audio frames early.
The current FluidAudio conversion is described as Pocket v2.1; testing it is
not verification of every v3.1 training/runtime change.
[Release](https://github.com/kyutai-labs/pocket-tts/releases/tag/v3.1.0),
[model/runtime](https://github.com/kyutai-labs/pocket-tts),
[English configuration](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english_2026-04.yaml).

Pocket's mixed INT8 optimization is relevant, but backend-dependent. Its PR
reported 450 → 234 MB in PyTorch, with an ARM torchao speedup while an older
QNNPACK path slowed down. Those figures do not transfer automatically to
Core ML or an iPhone. This review therefore measures both available Core ML
precisions rather than treating the published reduction as a prediction.
[Quantization implementation and benchmarks](https://github.com/kyutai-labs/pocket-tts/pull/147).

**Inflect v2:** Micro has about 9.36M parameters and 37.53 MB FP32 weights;
Nano has about 3.97M and 15.97 MB. Both target a single English male voice.
Those are weight sizes, not measured app memory. Micro's small community
preference sample is not a broad proof that it sounds better than Kokoro.
[Micro model card](https://huggingface.co/owensong/Inflect-Micro-v2),
[Nano model card](https://huggingface.co/owensong/Inflect-Nano-v2).

The new FluidAudio port is beta. Its text frontend approximates the eSpeak
phonemes used during training with a shared lexicon/BART path; direct IPA is
the fidelity escape hatch. Pronunciation of Wikipedia names, numbers, and
abbreviations needs particular attention. The tiny acoustic model alone does
not establish the full frontend's memory cost.
[Core ML integration](https://github.com/FluidInference/FluidAudio/pull/823).

## Kokoro deserves another look, with two implementations distinguished

The user's preferred sound makes Kokoro worth prioritizing. The old app MLX
screen peaked near 4.2 GB, and the older FluidAudio Core ML screen around
1.45 GB. Those were July Mac measurements, not a current phone limit.
[Historical measurements and methodology](ON_PHONE_TTS_COMPARISON.md).

FluidAudio's newer Kokoro default distributes stages across ANE and GPU.
It is not an entirely ANE-resident engine. Its known BNNS crash is reported
fixed on macOS 26.6 but still reproduced on iOS 26.6 with A19 Pro. The current
phone is iOS 26.6.1, so a successful Mac test cannot clear that integration
risk. The upstream change corrects the OS advisory, not the underlying crash.
[iPhone crash report](https://github.com/FluidInference/FluidAudio/issues/844),
[advisory correction](https://github.com/FluidInference/FluidAudio/pull/848).

The separate `mattmireles/kokoro-coreml` pipeline uses fixed model buckets and
staged execution. Its July iPhone Release comparison found gains over MLX,
but only a subset of its stages ran on ANE on A17 Pro. Run ordering and heat
could reverse a speed comparison, reinforcing the need for separate processes
and matched conditions. It is a distinct implementation, so neither the
FluidAudio crash nor its memory result should be assumed to apply unchanged.
[Staged pipeline](https://github.com/mattmireles/kokoro-coreml),
[Release iPhone comparison](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Notes/iphone-release-build-mlx-comparison.md).

## Local measurement protocol

Mac Studio, M1 Ultra, 128 GiB RAM, macOS 26.6.2. Release compilation, one
backend at a time, separate processes, downloaded assets prepared separately,
then two ten-turn runs. The second process measures persistent Core ML caches;
the first run's compilation cost is retained separately. The app's existing
Supertonic 96-character library cap and 94-character dispatch policy are
preserved in the experimental checkout.

The corpus includes names, dates, units, conversational continuations, and
longer article prose. All synthesis runs validate nonempty, finite, non-silent
PCM, and record amplitude, time to first PCM, total synthesis, frame delivery,
10 ms sampled physical footprint, settled memory, and cleanup. Playback is
separate. This is a small engineering screen, not a naturalness study or a
population-level latency percentile.

Memory is **process physical footprint in MiB**, not RSS, model download size,
or whole-system RAM. ANE compiler services can consume memory outside this
process. Bonsai is not loaded into these isolated runs. Existing unrelated
local inference servers were observed, but no workload was started on them.
Mac results do not establish phone jetsam, energy, heat, or coexistence with
an actively decoding Bonsai model.

[Reproducible harness](../tools/tts-compare/README.md).

## Measured results

Second fresh process, with model compilation caches established. First PCM includes all backend work from raw text, but excludes manager initialization. The subsequent-turn median covers nine calls across the small corpus; it is not a general latency percentile.

| Backend | Initialize | First PCM in new process | Subsequent first PCM median | Peak MiB | Settled MiB | After cleanup MiB |
|---|---:|---:|---:|---:|---:|---:|
| Supertonic 3 INT8, ANE | 0.125 s | 0.183 s | 0.085 s | 60 | 44 | 38 |
| Supertonic 3 INT4, ANE | 0.127 s | 0.196 s | 0.083 s | 56 | 40 | 34 |
| Kokoro, ANE + GPU tail | 1.723 s | 1.676 s | 0.390 s | 1983 | 1269 | 640 |
| Kokoro, ANE + CPU tail | 1.683 s | 1.661 s | 0.432 s | 1099 | 963 | 542 |
| Pocket FP16, GPU-oriented | 0.443 s | 0.509 s | 0.130 s | 757 | 691 | 659 |
| Pocket INT8, GPU-oriented | 0.421 s | 0.513 s | 0.132 s | 686 | 628 | 587 |
| Pocket FP16, ANE | 0.363 s | 0.336 s | 0.092 s | 494 | 482 | 478 |
| Inflect Micro, GPU | 1.100 s | 0.335 s | 0.116 s | 477 | 407 | 93 |
| Inflect Nano, GPU | 1.093 s | 0.286 s | 0.086 s | 352 | 352 | 91 |
| Inflect Micro, CPU | 1.004 s | 0.265 s | 0.187 s | 293 | 197 | 69 |

**Kokoro:** ANE neural stages plus CPU noise/iSTFT reduced peak footprint by **45%**, from 1,983 to 1,099 MiB, for an approximately 11% increase in subsequent-turn median latency (390 → 432 ms). Both use the same `af_heart` voice and model family. This is the most promising Kokoro configuration from this screen, although it is still substantially larger than Supertonic and does not clear the known iOS crash risk. The FluidAudio Kokoro runs also emitted an ANE compiler diagnostic during teardown despite successful synthesis and exit status zero.

**Pocket:** selecting the ordinary ANE FlowLM export reduced peak footprint by **35%** (757 → 494 MiB) and improved the subsequent first-frame median by **29%** (130 → 92 ms). INT8 on the GPU-oriented path saved only about 9% here, far less than the separate PyTorch report. No extra startup buffer was required in the measured streaming frame traces after their first frame; this does not include real playback scheduling.

**Inflect:** CPU execution reduced Micro peak footprint by **39%** (477 → 293 MiB), at a median of 187 ms instead of 116 ms. The full frontend, runtime allocations, and bucket caches matter more than the small advertised weight file. Nano was faster and smaller than GPU Micro, but CPU Micro was the lower-memory configuration actually measured.

**Supertonic:** remains the strongest measured memory baseline. INT4 saved only about 4 MiB of peak footprint in this screen; that small difference and a few milliseconds of timing do not establish a meaningful quality/performance advantage over the current INT8 setup. The twelve-step listening sample is an exploration, not a new default.

### Additional diagnostic cases

- **Pocket fused ANE-state:** failed during initialization with Core ML reporting that `functionName` must be nil unless the model is an ML Program. No latency or low-memory success is claimed for that path. The ordinary ANE path succeeded.

- **Separate staged Kokoro:** on its upstream 104-character garden fixture, one seven-second bucket, and precomputed phoneme tokens, the second run reached **785 MiB peak**, with **333 ms** median warm synthesis. Its unmodified package initializer recompiled model packages on each process launch, taking **47.9 seconds**, followed by **7.5 seconds** for first synthesis in the second process. This prototype requires persistent compiled-model loading and a text frontend before an app integration; it is not directly comparable to the varied raw-text table.

The screen completed **230 validated synthesis turns** across the two-process backend runs and matched samples. One experimental backend failed initialization, as recorded above. Finite PCM and clipping checks do not establish correct pronunciation or naturalness.

[Full structured results, cold runs, diagnostics, and per-turn data](benchmarks/tts-mac-2026-09-05.json). Local WAV comparisons and a listening sheet are in `ios/build-tts-comparison/listening/` (ignored build artifacts). Samples were adjusted to matched whole-clip RMS with a peak cap; measured raw PCM remains separate.

## Recommendation

Keep the repaired **Supertonic 3 INT8** path as the deployed baseline while auditioning the alternatives. Given the user’s preference, prioritize **Kokoro with ANE neural stages and CPU waveform finishing** for the next real-app phone experiment. It materially reduces the memory problem; it does not eliminate it. **Pocket ANE** is the strongest measured streaming challenger with substantially less memory than either Kokoro prototype. **Inflect Micro on CPU** is an interesting compact fallback if its single voice and pronunciation are acceptable.

A wholesale FluidAudio update is not required merely to expose the Kokoro compute-unit configuration: the existing vendor has the per-stage constructor. The newer normalization and Tail v2 assets do require a reviewed source/assets update. Keep that distinction explicit when translating the experiment into the app.

### Adversarial review of the experiment

Reviewed metric boundaries, silent/invalid PCM rejection, model/cache isolation, backend-specific streaming granularity, text-versus-token inputs, retained singleton memory, actual stage placement settings, and cold-start labeling. A sampler phase-boundary race was tightened; the staged and CPU-tail measurements use that version. Replaced a newer convenience preset with its equivalent per-stage constructor to preserve old-vendor compatibility. The production dependency and default engine were not changed. This was an adversarial self-review, not an independent reviewer.

Release builds passed against both the current vendor and the experimental
checkout. The executed harness passed Developer ID signature verification;
four invalid-configuration checks passed, the listening WAVs were verified,
and `git diff --check` passed. Downloaded model and shared frontend files are
recorded in the [SHA-256 asset manifest](benchmarks/tts-model-assets-2026-09-05.json).

## Other options and next evaluation

The Rust/Candle Pocket iOS project advertises lower memory and approximately
200 ms startup, but those are maintainer claims from a different runtime,
not directly comparable to this Core ML harness. Its model/export parity
would need checking before treating it as a shortcut.
[Native iOS port](https://github.com/UnaMentis/pocket-tts-ios).

LuxTTS provides 48 kHz voice cloning and an ANE variant, but its own Core ML
documentation still reports hundreds of MB in steady state and a short
fixed-window limit. It is a lower-priority fit for this app's immediate
memory/latency requirement.
[LuxTTS integration and measurements](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/TTS/LuxTts.md).

ANE placement can reduce competition with Bonsai's GPU work, but ANE and GPU
share system memory. Placement requests are not proof of actual per-operation
placement or energy savings. Pocket's tested ANE option changes FlowLM placement;
its CPU Mimi decoder and other mixed stages remain. Final evaluation should
measure all of these together with the real app, not add isolated memory
numbers and call the result safe.

The phone is currently reachable, so no reconnect is needed. The next app
selection should be based on matched-loudness listening, current-OS stability,
offline startup, cancellation, and memory while Bonsai is resident and decoding.
Suggested engineering targets are sub-500 ms warm speech onset and preferably
under 250 MiB incremental footprint, with first-install compilation handled
explicitly. These are selection goals, not thresholds proved by this Mac screen.

The [Napoleon repair](NAPOLEON_SPEECH_2026-09-05.md) is already installed and
verified separately. Its warm-cache phone probe reached first Supertonic PCM
in 430 ms including preparation; the first run after installation still paid
15.4 seconds before PCM. This review does not hide that cold-start limitation
or attribute it to Apple's output voice.
