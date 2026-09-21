# Date refinement latency and factoid source review — September 7, 2026

## Changes

A conversation-local `ValidatedDateEvidence` record retains at most three exact source passages (3,300 characters) from a direct factual exploration answer. It can answer only the exact short requests “What year?” or “Which year?” while the resolved question remains unchanged. This avoids repeating both LLM calls and archive searches for an already answered question.

The eligibility gate requires a historical-date question, a direct factual answer, and one distinct four-digit year in the retained evidence. Multiple dates, missing dates, negation, uncertainty, background-only answers, current-condition answers, and unsupported question shapes fall back to the existing path. “What date?” is intentionally not served from a year-only result. Output retains full source text and provenance rather than extracting a bare number. Its continuation cards come from those source identities.

The record is replaced or cleared after each exploration answer, cleared after ordinary fast answers, and owned by the discussion state. Existing conversation reset, topic replacement, and library rebuild paths discard that state. Nothing is persisted to disk and no extra model weights or embeddings are retained.

The deterministic `article_factoid` presentation path now attaches the actual returned ZIM article, section and library to the answer. Previously the displayed answer was grounded but had no source metadata, causing five existing source assertions to fail. The evidence sentences receive attribution. The host-computed approximate age does not receive an exact-excerpt label or fabricated source attribution.

## Adversarial self-review

- Same date in different countries: reuse requires equality of the full resolved question, not equality of the year or article alone.
- Same country, different organization: NATO/EU question changes reject reuse even when the dates happen to be equal.
- Multiple dates, date ranges or uncertain claims: no cached answer. Tests cover multiple years, missing dates, explicit negation and straight/curly-apostrophe negative contractions.
- More specific follow-up: “What date?”, “What year was that disputed?”, and “What year is it now?” cannot consume the cached answer.
- Topic reset or removed archive: the existing reset/rebuild logic destroys the discussion state and its record. Cache ownership was checked against these paths.
- Current opinion or historical background: only a non-contextual direct fact can create the record; the prior qualification cannot be lost by taking the fast path.
- Stale suggestions: continuation cards are built from the reused passages' archive titles, not the old anchor or arbitrary prior assistant text.
- Calculated age masquerading as a source quote: only the underlying evidence receives sentence attribution. The approximate age remains the existing host calculation.
- Missing/error factoid: source attachment requires a successful branch with both an actual title and nonempty evidence; it does not manufacture citations for a miss.

This cache does not certify the initial selector's semantic judgment. It only reuses that same bounded, source-bound answer for an unchanged historical-year request. Conservative rejection costs latency rather than changing the answer boundary.

## Validation

Shared Swift suite: 645 executed, one skipped, zero failures (644 passed), including the new adversarial reuse tests. The targeted actual-model replay covers country/facet transfer and the five existing age/founding-date source failures. It uses Bonsai Q1 with phone context settings on the Mac and the June 2026 Wikipedia ZIM, not a physical phone. Replay/build results are recorded below after completion.

## Actual-model review findings

The first replay returned a Lithuania–Poland passage containing both 1999 and 2004. Date reuse correctly refused that ambiguous record; “What year?” performed normal retrieval and produced a single-year answer. The repeated-year case exercises reuse only after that ambiguity has actually been resolved. This is intentionally more conservative than guessing which date belongs to which entity.

All five affected factoid answers now have source metadata. Four former source assertions pass. The Apple assertion expects `Apple Inc.` while this archive lookup returns the real redirect title `Apple, Inc.`; the citation is present and points to that archive identity. We did not rewrite an archive title simply to make the test green.

The replay also exposed stale generic continuation wording such as “tell me about Cupertino, California” after Apple's founding date. Factoid suggestions now explicitly target the returned source article. This uses the same identity as the citation and preserves the canonical/redirect lookup behavior; it does not infer a new topic from words in the answer.

## Results

- Actual Bonsai replay: the first “What year?” required retrieval (13.97 seconds); after its single-year answer, the repeated question reused exactly the same source evidence in **0.108 seconds**. The log confirms no LLM or search on the reused turn. This demonstrates the eligible fast path; it is not a promise that ambiguous follow-ups will skip retrieval.
- Targeted replay: **11/12 assertions passed**. The remaining assertion is the documented `Apple Inc.` / `Apple, Inc.` archive-title mismatch. All five factoid citations are present, and their continuation cards now target the cited articles.
- Final attribution review uses the actual returned section text for sentence matching, not a copy of the extracted answer as its own evidence. Host age arithmetic remains outside that attribution.
- Shared tests: **644 passed, one skipped, zero failures**. The reusable exploration suite now includes the repeated year follow-up.
- Sanitized replay: [date-reuse-and-sources.json](benchmarks/exploration-2026-09-07/date-reuse-and-sources.json).
- These measurements are from the Mac, with a different Wikipedia snapshot from the phone. No phone installation, TestFlight upload, commit or push was performed in this round.
- Signed iPhone build and repository signature gate passed for `com.tiltastech.zimfo`, team `A6G8H8NGAM` (`/private/tmp/improvement-ios-reviewed.log`).
