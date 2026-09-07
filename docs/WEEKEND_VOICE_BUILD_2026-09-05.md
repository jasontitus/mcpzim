# Weekend voice build — September 5, 2026

The phone build retains Supertonic 3 INT8 as the default and offers established
Kokoro MLX in Library → Voice chat → Engine. End and restart voice chat after
changing the engine. Both engines' assets are present in Application Support
on the connected iPhone 17 Pro Max, including all ten Supertonic voice styles.
These are preloaded device assets, not extra model payloads in the signed app.
The development provisioning profile expires July 16, 2027.

## Shipping changes and adversarial review

- The voice bar reports the active backend, including a memory fallback.
- Insufficient memory or serious thermal pressure switches a high-memory
  backend to Supertonic instead of an unexpected Apple voice. The stored
  preference is preserved. Explicit System voice selection still works.
- Review caught that fallback must happen **before** reading the backend's
  chunk limit. Otherwise a Kokoro-sized chunk reaches Supertonic's smaller
  input window. The ordering is corrected.
- Ending voice chat releases the stored TTS service. Outstanding tasks can
  retain it until cancellation finishes; release is not claimed instantaneous.
- Missing/unloadable Kokoro assets also select Supertonic when available.
- Synthesis checks task cancellation before and after the Supertonic call.
- Experimental Kokoro Core ML was removed from the app, along with its
  temporary shared-adapter refactor and vendor model-name changes.

## Physical-device evidence

Bonsai remained loaded during the tests. The Napoleon answer came from local
ZIM excerpts. PCM-only probes exercised synthesis without the microphone.

| Backend / probe | Result |
| --- | --- |
| New Kokoro ANE + CPU | Crashed after four chunks; rejected |
| Established Kokoro MLX, 94-character probe | Ten chunks completed; 6.53 s preparation, 8.00 s to first PCM |
| Established Kokoro MLX, production 112-character cap | Nine chunks completed; 1.13 s preparation, 2.07 s to first PCM; later chunks roughly 0.7–1.2 s |
| Supertonic, first actual playback after install | 3.61 s preparation; four chunks queued at 25.77 s; playback completed at 39.53 s |
| Supertonic, fresh process with warm compilation cache | 0.192 s preparation; all four chunks queued at 0.981 s; playback completed at 24.67 s |

Kokoro's sampled app footprint reached approximately 2.64 GB with Bonsai
resident. This is sampled footprint, not a measurement of the allocation peak.
The conservative 2.8 GB synthesis estimate and memory guard remain in place.
Warm Supertonic sampled app footprint was approximately 943 MB.

The rejected Core ML crash is a native `EXC_BAD_ACCESS` / `SIGSEGV` in
`BNNSGraphContextExecute_v2`, not a catchable Swift error. The phone report is
`MCPZimChat-2026-09-05-113908.ips`, retained under
`/private/tmp/mcpzim-crash-wifi/`. A successful Mac benchmark did not establish
iPhone safety; the phone test directly reproduced the upstream risk.

The user heard a roughly 0.2-second glitch and unnatural mid-sentence pauses
during the first Supertonic playback. Cold compilation caused large synthesis
stalls, but persistent prosody defects must not be declared fixed merely
because warm synthesis is fast. Short 94-character windows can still split
phrases such as “the first / president” and “American / Revolutionary War.”
The input cap remains 94 spoken characters. A follow-up correction prefers an
available natural clause before a forced word wrap, recognizes a closing
parenthesis as a clause boundary, and ignores outer whitespace when deciding
whether a completed tail already fits. Regression tests preserve exact source
consumption and prevent premature streaming dispatch. This addresses those
specific mid-phrase splits; it does not establish that every prosody glitch is
fixed. The intermediate boundary playback measured zero queue starvation gaps
on all six chunks (later synthesis 61–110 ms); the final whitespace correction
is checked separately below.

## Validation and limits

- Full MCPZimKit suite after the boundary fixes: 604 tests passed.
- Targeted speech-policy suite: 22 tests passed as part of the full suite.
- Signed iOS build and signature gate passed; app installed on the phone.
- Both Kokoro synthesis probes completed; Supertonic playback completed.
- This is a short device acceptance test, not a weekend-length stress test.
- New benchmark hooks are DEBUG-only and require explicit launch environment
  flags. Normal launches do not run them or alter the selected engine.

Raw test logs are retained in `/private/tmp/zimfo-weekend-*.log`. See also
`ON_DEVICE_TTS_REVIEW_2026-09-05.md` for the broader Mac comparison and sources.


The audible Kokoro comparison completed all four Washington chunks: 1.160 s
preparation, all audio queued at 4.339 s, playback complete at 27.231 s. No crash.

Offline review caught a directory-name discrepancy: the actual downloader
uses `models/supertonic_3/supertonic-3`, despite the old adapter comment naming
`supertonic-3-coreml`. All ten styles were copied into the runtime-confirmed
`supertonic-3/voice_styles` folder; the model bundles and styles were enumerated
there before handoff. Merely populating the similarly named folder would not
have made voice switching work offline.


Final corrected Supertonic playback: five chunks, every queue-gap metric
0.000 s, later synthesis 83–131 ms, playback completed in 34.423 s. First
synthesis after this reinstall compiled for 8.536 s following 3.913 s model
preparation; this is explicitly a cold-install result, not warm response
latency. The already measured warm run queued the passage in under one second.
The final playback warmed the installed app before normal launch.

Final handoff: normal launch with no probe flags passed the 45-second device
health watch. Final iOS and macOS builds and their signature gates passed.

## Follow-up: end-of-sentence glitch report

The user's pasted 11:44–11:45 session selected Supertonic directly. The Kokoro
probe running earlier in that session was silent and did not change the stored
engine preference. Its unclean-session banner referred to the rejected 11:39
Core ML experiment. The reported real conversation had zero queue-gap metrics;
this does not demonstrate artifact-free PCM or a clean capture transition.

Review found both engines used AVAudioPlayerNode's default completion callback
for the final playback waiter. The installed Apple SDK's AVAudioPlayerNode.h
documents that callback as `DataConsumed`, which can precede audible completion.
Both waiters now explicitly request `dataPlayedBack`, accounting for downstream
and device latency before capture is reconfigured. The controller also checks
cancellation after the waiter returns so an interrupted reply does not start a
second listening cycle. These changes address a concrete end-of-reply race;
they are not proof that every reported sentence-boundary artifact has the same
cause. Signed iOS build and signature gate passed, and the correction installed
successfully. Device verification is recorded in
`/private/tmp/zimfo-playback-tail-phone.log` and the normal-launch health log.
