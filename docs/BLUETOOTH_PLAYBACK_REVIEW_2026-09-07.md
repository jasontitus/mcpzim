# Bluetooth playback hang review

## Reported evidence

At 17:29:32 the phone queued the gravitational-wave response through Supertonic to Tesla Bluetooth. Eight chunks were synthesized (about 38.7 seconds of PCM). There was no TTS-done log before stop at 17:30:55. The user confirms none of this response was audible. Synthesis and scheduled-buffer metrics are therefore not evidence of successful playback.

The preceding Kokoro-to-Supertonic thermal fallback and 13-second cold start are separate issues. This change does not relax thermal safeguards or claim to fix cold synthesis latency.

## Confirmed code defects and changes

- Supertonic's final `.dataPlayedBack` continuation had no cancellation or deadline. `stop()` did not release it. Added a Foundation-only exactly-once completion gate with played, cancelled, interrupted, and timed-out outcomes.
- The deadline is estimated remaining queued PCM duration plus eight seconds for output latency. Expiry is an explicit playback failure, never a successful completion.
- Stop releases the waiter. Late callbacks cannot complete it twice. A superseded waiter cannot clear another waiter's state.
- Playback engine and player are stopped after completion, timeout, and stop so they do not remain active through the Bluetooth microphone-capture phase. The synthesis runtime remains loaded.
- Playback engine configuration changes during queued playback produce a visible, logged failure. Notification handling is deferred to avoid synchronous lock re-entry from engine operations.
- Speech chunk failures formerly retried every 75 milliseconds indefinitely. They now stop playback and enter the visible error state. Text remains in the conversation; no automatic replay of potentially partially heard audio.

## Adversarial self-review and tests

Tests cover completion before waiter registration, stop before registration, cancellation, a missing callback deadline, output-interruption signals, and a late callback after completion. Swift suite: 654 tests executed, one skipped, zero failures (653 runnable tests passed).

Reviewed callback-versus-stop ordering, double resume, retained continuations, stale waiter cleanup, timeout versus success reporting, and indefinite chunk retry. Gate tests exercise the completion lifecycle with simulated signals; they do not emulate AVAudioEngine or Tesla Bluetooth.

The log does not establish the precise Bluetooth failure mechanism. Restarting the playback graph between capture phases addresses a plausible stale-engine path; a real two-turn Tesla playback test is still needed to establish audible recovery. A completed AVAudioEngine callback still cannot prove the car's speakers were audible.
