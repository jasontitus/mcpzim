# Legacy follow-up relevance — 2026-09-07

The phone answered “What is Grand Duchy of Lithuania's legacy?” with three sentences about Prussian tribes and early eastward expansion. The source label was accurate: the local June 2026 English Wikipedia ZIM has the same historical preface inside its Legacy section. The phone log also identifies Legacy as the selected section. This is sentence relevance failure, not evidence of a heading/parser mismatch.

`SourceBoundAnswer` previously accepted the heading as coverage of the question and preferred source order. Suggested section navigation could additionally fall back to a broad overview. Legacy questions now require a sentence expressing lasting effects, influence, reception, remembrance, or continuity. This check also applies to the section-overview fallback. Output remains complete, unchanged source sentences in source order; no inference or generated factual prose is added.

Adversarial self-review:

- Tested the misleading historical preface followed by actual legacy evidence, through both typed and suggested questions.
- Tested a misleading Legacy heading with background only: it cannot authorize an unrelated answer through the overview fallback.
- Preserved Lenin's mixed reception, including both reviled and revered, rather than extracting a favorable fragment.
- Ordinary early-history navigation retains its existing behavior. Article titles containing “legacy” do not activate this extra question filter.
- This is a bounded English relevance heuristic, not semantic proof. It can miss unusual descriptions of legacy or accept incidental influence language. A future broader relevance mechanism must preserve exact-source attribution and be evaluated against these cases; the heading alone is insufficient.

Validation: Swift suite executed 631 tests, one skipped, zero failures (630 passed). Signed iPhone build and real-archive replay recorded separately below when complete.

Signed iOS Debug build and signature gate passed; installed on the paired iPhone over Wi-Fi. Silent real-archive replay of the overview followed by the reported legacy question returned three Legacy excerpts about enduring regional differences, permanent divisions, and inspiration for national revival movements. User submission to answer: 0.676s; excerpt selection: 2.0ms. This replay used the typed retrieval path (the resolved title included “The”); suggested-section behavior is covered by the regression test. No model-generated factual prose or speech was used.

Local validation logs: `/private/tmp/legacy-tests.log`, `/private/tmp/legacy-ios-build.log`, `/private/tmp/legacy-signature.log`, `/private/tmp/legacy-phone-replay.log`, `/private/tmp/legacy-watch.log`. Conversation logs remain outside version control.
