# Napoleon extraction and speech startup — September 5, 2026

The captured session explains both failures. The article answer was extracted
from the enabled Wikipedia ZIM, but a nested thumbnail map was treated as prose.
Speech used **Supertonic 3 ANE INT8**, with no system-voice fallback recorded.
The perceived voice quality remains a listening concern; the log does not
support attributing that utterance to Apple's output voice.

## Findings and repairs

| Finding | Repair | Adversarial check |
|---|---|---|
| A legacy `div.thumb` map contributed battle labels, distances, JSON errors, and its caption to the biography. | Remove complete nested prose widgets before discovering article sections. | Actual 809,954-byte Napoleon HTML retrieved from `wikipedia_en_all_maxi_2025-10.zim` confirms the markup. Regressions cover nested divs, widget headings, quoting, exact class tokens, and preservation of ordinary prose. |
| The paragraph-break branch bypassed the 94-character speech cap, returning a 664-character first chunk. | Apply the cap before accepting a paragraph boundary. | Streaming and completed replies remain bounded and drain without losing source characters. |
| Three dummy utterances ran before synthesizing the actual reply. | Prepare assets and voice style without dummy inference on the answer path. | First real synthesis is measured explicitly; initial Core ML compilation remains visible rather than hidden in preparation. |
| Backend attribution was ambiguous in the first-audio log. | Record the selected backend with first-audio readiness. | Both repaired runs explicitly identify Supertonic F1. |
| The diagnostics uploader could mistake its active file for a completed session and receive HTTP 400. | Compare the unique session filename within the log directory instead of Foundation URL object representations. | Final normal launch uploaded the previous completed session; no active-file HTTP 400 was observed. |

These are extraction, chunk scheduling, startup, and diagnostics changes.
Factual answer text still comes from the ZIM. The engine weights, eight-step
synthesis setting, and selected F1 voice remain the same.

## Physical-phone evidence

iPhone 17 Pro Max, iOS 26.6.1. The probe synthesizes production PCM and discards
it without starting playback or microphone capture. Its original no-audio
restriction was honored. These are PCM readiness measurements, not acoustic
latency or a listening-quality test.

| Measurement | Original user session | Repaired, first run after install | Repaired, new process with compiled model cache |
|---|---:|---:|---:|
| TTS preparation | 10.06 s | 3.419 s | 0.203 s |
| First PCM from speech start | 48.81 s | 15.405 s | **0.430 s** |
| First synthesis chunk | 664 characters | 66 characters | 66 characters |
| Whole ten-chunk repaired answer synthesis | Not comparable | 23.689 s | **1.206 s** |

The first installation still incurs substantial lazy Core ML compilation.
The repeat launched a fresh process, showing that the fast result does not
depend on keeping a previous conversation's TTS manager alive. A later chunk
also incurred a one-time bucket compilation during the first run.

The repaired Napoleon text arrived in **196 ms**, and its early-life follow-up
in **81 ms**. Both contain clean source excerpts. A separately observed user
selection of the legacy section completed in 10 ms; that was not one of the
two automated questions. Whole-app memory includes Bonsai and changing KV
cache residency, so it cannot establish isolated TTS memory use.

## Verification and review

- New regressions reproduced the defects before repair: seven failed assertions.
- Final MCPZimKit suite: **600 tests passed**, zero failures.
- Signed iOS Debug and macOS Release builds and repository signature gates passed.
- Final iOS build installed; normal launch survived all nine checks in a
  45-second health watch. Probe flags were removed for that normal launch.
- Adversarial self-review checked malformed/class-like attributes, nested
  markup, widget headings, text consumption, actual backend attribution, and
  cold-versus-cached timing claims. This was not an independent reviewer.
- `git diff --check` passed. No commit or distribution upload was performed.

The final iOS debug library UUID is
`5A9F2BC6-5978-33A9-B492-AAF75B60BCE5`; SHA-256 is
`2ef95d77843d95551dbe4358b0b73a7131dd035b99d0543c7fc000d00efd3a2c`.

[Sanitized machine-readable measurements](benchmarks/napoleon-speech-2026-09-05.json).
Private local evidence remains outside the repository:
`/private/tmp/zimfo-napoleon-user.log`,
`/private/tmp/zimfo-napoleon-phone-timed.log`,
`/private/tmp/zimfo-napoleon-repeat-timed.log`,
`/private/tmp/zimfo-napoleon-zim.html`,
`/private/tmp/zimfo-napoleon-swift-tests.log`, and
`/private/tmp/zimfo-napoleon-normal-watch.log`.
