# Voice selection and first-audio delay review

## Reported behavior

Build 20260907051627, Arnold Schwarzenegger query: Kokoro was the saved
preference. The runtime constructed it, then the memory guard switched to
Supertonic with 2409 MiB available, below the 2800 + 700 MiB safety budget.
Retrieval and source excerpts finished in about 0.29 seconds. Supertonic
preparation took 3.38 seconds and its first synthesis 11.87 seconds; first
speech arrived 15.22 seconds after streaming began. Subsequent chunks took
roughly 40–70 ms. This is a voice cold-start problem, not slow ZIM answers.

## Changes

- Apply the existing Kokoro headroom/thermal gate before constructing it,
  including when it is explicitly preferred. Do not relax its safety budget.
- Label the setting “Preferred engine”. Voice chat initially says “Selecting
  voice…”, then displays the actual backend and an explicit fallback reason.
  Distinguish missing assets, heat, memory, and initialization failure.
- Share one Supertonic model runtime across playback-service instances. Voice
  styles are cached separately; changing voice does not allocate another model.
- After the language model and reranker are ready, silently warm an installed
  Supertonic runtime when it is likely to be used, with at least 800 MiB
  headroom and no serious/critical thermal pressure. This path checks all
  required local assets and voice style first; it does not initiate downloads.
  It creates no audio engine, accesses no microphone, and plays no audio.
- Serialize initialization, warm-up and real synthesis explicitly: actor
  isolation alone does not prevent overlapping work across `await` calls.
  Concurrent warm-up requests share one task. Failed warm-up remains retryable.
- Cancel and clear per-session preparation on stop, and clear stale engine
  status/fallback explanations before the next session.

## Adversarial review and validation

Self-review covered pre-allocation fallback, unavailable files, thermal limits,
unknown headroom, duplicate runtime allocation, actor reentrancy, cancellation,
stale session state, incidental downloads and capture/audio-engine interference.
All source edits are in app code; the vendored FluidAudio package is unchanged.

Signed Mac tests: four passed, including opt-in synthesis using real cached
Supertonic models without playback. Measured silent warm-up 19.61 s, first
post-warm utterance 2.57 s, new service using the same runtime 0.133 s. This is
Mac evidence of runtime reuse, not an iPhone first-audio claim. Different input
shapes can still initialize an additional ANE bucket. The runtime intentionally
retains its lightweight model allocation between voice sessions.

Run the opt-in Mac test with TEST_RUNNER_ZIMFO_TEST_SUPERTONIC_RUNTIME=1 and
TEST_RUNNER_MCPZIM_SUPERTONIC_MODEL_DIR pointing at a complete existing model
cache, selecting SupertonicRuntimeTests in MCPZimChatMacTests. Ordinary tests
skip the model-dependent test. No models, generated PCM, or personal logs are
committed.

Phone validation remains necessary, especially starting voice immediately after
launch or returning before background warm-up has completed. If assets are
absent or warm-up is skipped under resource pressure, a cold first turn remains
possible. These changes have not yet been uploaded to TestFlight.

Final signed iOS Debug build and bundle/team signature gate passed.
