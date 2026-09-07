# Adversarial review: conversation discovery and Bonsai latency

Reviewed the complete conversation and latency change set on September 5,
2026: deterministic routing, topic sampling, nearby stories, suggestion cards
and spoken selection, prompt recovery, DEBUG benchmarks, and their evidence.
The unrelated fine-tuning artifacts and Xcode's user-interface state are not
part of this change set. This was a source and regression review followed by
silent testing on the attached A19 Pro iPhone, not an independent reviewer.

## Findings corrected

| Priority | Failure condition | Correction and evidence |
| --- | --- | --- |
| P1 | A decode error could leave a partially mutated backend context while retaining its token mirror and recovery checkpoint. A later request could reuse state whose validity was unknown. | Decode failure now invalidates both. Normal cancellation still preserves completed state. The original device-buffer/512-token failure was real; wrong answers after such a failure were not reproduced. |
| P2 | Discovery offset was computed from the total number of user turns and clamped to 48. Follow-up questions skipped topics, and sufficiently long conversations could remain on the same page forever. | Separate per-library-kind cursors advance from successful discovery results. Pagination walks a stable bounded candidate list, skips unreadable entries, and explicitly announces the end before cycling. Tests cover 20 unrelated turns, sparse pages, archive changes, reset, and out-of-range cursors. |
| P2 | WikiMed topic cards discarded their archive when converted into a plain “tell me about” request. With Wikipedia also loaded, a selection could open the other source. | Cards retain the sampled archive; tapped and spoken selections dispatch the same archive-qualified intent. Stub replacement and miss handling honor that choice. A regression checks the intent built from a medical topic result. |
| P2 | Map suggestions advertised “show me X,” while their actual action required “show me X on the map.” A synthetic continuation could even advertise “show me Another nearby lead.” | Both the voice bar and appended cue use the exact card action. Selection recognizes that action. Regression checks round-trip section questions, map actions, and nearby continuation cues. No audio was played during testing. |
| P2 | Nearby exclusions compared only the map name, although the visible offer could use a different canonical Wikipedia title. The same place could be returned again, including through the map fallback. | Exclusions cover map names, wiki tags, and resolved article aliases. The regression excludes “Iris Center” and verifies that its differently named map place is not returned as a fallback. |
| P2 | The recovery probe ignored the interrupted run's outcome. It could claim interruption correctness even if cancellation never occurred. | The pass condition now requires an actual `CancellationError` after the requested output threshold, nonempty completed answers, cold reference caches, and equality after changing early evidence. |
| P2 | New welcome text promised that every answer stays on the device despite optional diagnostic log uploads. | Replaced the unconditional privacy promise with an accurate description of offline articles and maps. Existing diagnostic preferences are unchanged. |

Additional hardening: cancelled question autoruns stop their timer and return
instead of spinning on a cancelled sleep; discovery observes cancellation;
asynchronous suggestion ranking verifies that the original reply still exists
and remains current before writing; discovery cards preserve the same order as
their ordinal selection list and clear stale lists at the end.

## Validation

The signed iOS Debug build and repository signature gate passed. The focused
Swift suite passed 155 tests with zero failures, covering the prefix policy,
intent routing, suggestion selection, topic pagination, and nearby enrichment.
`git diff --check` passed.

Phone measurements and the final health check are recorded in the companion
[latency report](BONSAI_PHONE_LATENCY_2026-09-05.md) and sanitized benchmark
records. The strengthened probe checks retries, append, actual cancellation,
edited answers, changed evidence, and explicit reset. These are deterministic
cache-correctness checks, not a broad factual-quality evaluation.
All six strengthened checks passed on the phone. A six-turn real-content
smoke also completed, including two topic continuations around an article
discussion, archive-qualified option selection, nearby discovery, and a
fresh nearby batch. The article stub regression declares both archives in
the fixture inventory and verifies that only the unqualified request can
switch to the larger Wikipedia match.
The normal launch was restored without test flags; the repository watcher
confirmed that the installed app remained alive for 45 seconds.

## Limits and remaining work

- The real Wikipedia smoke still exposed a pre-existing extraction issue:
  “Tell me about his parents” returned a family-business sentence. This is
  outside the cache optimization; evidence selection still needs attention.
- New evidence still requires prefill. The subsecond results apply to reuse
  and recovery, not every cold question or follow-up.
- A fresh Duet answer made a questionable claim that duet performers take
  turns playing solos. Its prompt reused zero tokens; it was not a recovery
  restore. Factual faithfulness remains unproven by the cache suite. The
  sampler also depends on editorial variety: this phone's first two pages
  were dominated by links from its featured Bach/music article.
- The pinned runtime was retained. The upstream comparison did not establish
  a TTFT win, and thermal differences prevent precise speedup claims.
- Wider target verification on the pinned runtime has promising headroom,
  but no draft model or speculative decoder has been enabled. The MLX
  challenge findings remain candidates for a measured phone implementation.
- The checkpoint adds about 149.6 MiB for this model, capped at 256 MiB.
  Repeated successful hot runs do not prove behavior under every iOS memory
  pressure or thermal condition. The optional device-buffer experiment is
  not the production default.
- No TestFlight upload was performed. Audio recognition and playback were
  not exercised because the user requested silence.

## Source-grounding follow-up — September 5

The later physical-phone source audit confirmed that the loaded Wikipedia
ZIM itself contains the questionable duet “taking turns” sentence. That
example was a source error, not demonstrated model invention. Article
answers now use exact ZIM excerpts with an explicit missing-evidence response;
the parents-name selection was corrected as well. See
[SOURCE_GROUNDING_2026-09-05.md](SOURCE_GROUNDING_2026-09-05.md) and the
[full app review](APP_REVIEW_2026-09-05.md). The earlier raw-inference cache
measurements remain valid for that benchmark; ordinary article answers now
avoid that generation pass entirely.
