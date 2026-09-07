# Place follow-up fix and pending-change adversarial self-review

## Current release status

All changes in this review shipped in **Zimfo 1.0 (20260907051627)** and are
verified available to `InternalTesters`. Earlier “not uploaded” statements below
record the status at that investigation step; the release-completion section
records final acceptance. Physical-phone validation of this build remains open.

Related reviews: [source-grounded conversations](EVIDENCE_CONVERSATION_SYSTEM_2026-09-06.md)
and [model-download launch memory](LAUNCH_CRASH_REVIEW_2026-09-06.md).

## Field failure

Build 20260907041312 correctly ran `bar` searches at 5 km, then 10 km around
(36.840586643248692, -121.63241233439634). It returned The Pit Bar and drew
11 pins. “Show me The Pit Bar on the map” resolved the name from the list,
but generic continuation routing ran before the stateless map action and
opened a Wikipedia article. A second search then silently substituted
“Barrow pit”. The failure was action selection and identity preservation,
not missing map tiles or inference latency.

## Changes and adversarial findings

- **Fixed, high: explicit map actions were swallowed by encyclopedia continuation.**
  Map selection now precedes generic continuation and binds exact saved names,
  pronouns, supported ordinals, and bounded place descriptors. Fresh names and
  geographic qualifiers cannot borrow an old place's coordinates. Duplicate
  names require clarification.
- **Fixed, high: clarification always dispatched an article.** A selected place
  now reruns the original map action; article clarification retains its path.
- **Fixed during review, high: the initial coordinate shortcut could accept
  model-provided coordinates.** Removed it from the model-facing `locate`
  dispatcher. The host-only shortcut requires an exact matching name and
  coordinate pair in the saved place list, and uses the saved archive when
  supplied. Model tool calls still require a real geocode result.
- **Fixed, high: an older article-miss rescue bypassed strict identity checks.**
  Removed the second search-and-open-top-hit path. Existing title suggestions
  and not-found replies remain; search rank cannot authorize a different entity.
- **Fixed during review, low: new map grammar lowercased fresh place names.**
  Preserve user capitalization and canonical saved labels.
- **Fixed, low: cancelled evidence work could enter the new synchronous
  in-hand authorship scan.** Added an entry cancellation guard.
- An initial archive test used no loaded archives. The sanitizer correctly
  removed the nonexistent archive argument; that was a fixture defect, not a
  production archive-loss bug. The final test covers saved archive propagation
  through the trusted path and actual map-payload parsing.

## Full pending diff reviewed

Reviewed the map changes plus the previous uncommitted authorship contract,
source-bound answer selection, exact title policy, geographic qualification,
deferred missing-answer presentation, checksum autorelease pool, checksum memo
key, restored file URL handling, tests and review documentation. This was an
adversarial self-review, not an independent agent review.

The existing authorship path still emits source sentences, not model factual
prose. It retains required question facets and rejects covered third-party,
negative, hypothetical, and mismatched-work-category cases. Hash verification
retains streaming behavior and expected-digest-sensitive memoization. No model,
KV cache, inference, speech-engine or download scheduling changes were added in
this follow-up fix.

## Validation

- Swift suite: 623 tests executed, 1 opt-in replay skipped, 0 failures.
- Separate historical replay: 159 turns; identical results to the previously
  fixed build for the replayed routing/excerpt components.
- New stateful regression: bars at 5 km → search wider at the same center →
  named/ordinal map selection; self-resolved and clarified lists; duplicate
  names; unrelated topic; explicitly different city; fabricated coordinates.
- Trusted selected result parses into exactly one map pin with saved latitude.
- Signed iOS Debug build and signature gate passed.
- Signed Mac ModelDownloadTests: 10 passed. Firebase test-host initialization
  logged keychain errors; the download tests themselves passed. This does not
  validate production debug-log upload.
- Shipping checksum function, isolated 512 MiB fixture: peak RSS 8,683,520 bytes,
  correct SHA-256, under the 64 MiB regression limit.

## Limits and remaining issues

Historical replay covers logged text and source excerpts, not a full phone UI
session or every state transition. The previous replay missed this place-list
case; the new explicit stateful fixture covers it. No remote phone visual test
or new TestFlight upload was performed for these changes.

The English authorship contract remains bounded: passive voice, complex
pronouns, unusual work titles and tables can still cause false negatives.
Geographic qualification depends on location metadata in the archive. Exact
saved coordinates are preserved, but archive metadata is optional in older
place results. Existing bar-category results can include alcohol shops because
the archive's category index groups them; this patch does not change category
membership. The previously documented background download concurrency concerns
remain outside these changes.

## Follow-up: automatic Kokoro default

The default preference is now Automatic (prefer Kokoro). Existing explicit
Supertonic, Kokoro and System choices are preserved. Automatic checks installed
assets, thermal state and live memory **before** allocating Kokoro. It requires
3,500 MiB headroom (existing 2,800 MiB measured peak estimate plus 700 MiB margin).
Missing assets, unknown/nonfinite headroom, insufficient memory and serious or
critical thermal pressure select the lightweight fallback. Settings exposes
Kokoro download/voice controls under Automatic; no automatic download was added.

macOS now estimates reclaimable RAM from free plus inactive pages, excluding
swap/compressed memory, and shares that measurement with ongoing voice safety
checks. iOS retains its process-specific jetsam headroom measurement. These are
conservative runtime estimates, not a guarantee against concurrent allocations.
The host Mach port is released after each macOS measurement.

Adversarial review caught a proposed dependency that would have broken the
standalone TTS benchmark target. The pure selection policy instead lives in
TTSFactory, which that target already compiles. Tests cover the exact threshold,
missing files, thermal pressure, zero/negative/NaN/infinite headroom. Signed Mac
voice tests: 3 passed. iOS build and signature gate passed. No new phone listening
benchmark or TestFlight upload was performed for this default change.

## Reported termination during Belize lookup

Fetched the opted-in `2026-09-06_21-31-48.log`. The final request was “Where is
Belize”, routed to `locate`. Logging ends while loading be-1 after be-6 and be-0;
last sampled process footprint was 1162.9 MB. The subsequent launch successfully
loaded Bonsai. At initial investigation no iOS diagnostic was available. The subsequently
retrieved TestFlight report confirms an allocation-failure abort in this exact
lookup path; see the diagnostic update below.

Inspection of the exact local `osm-california-2026-05-29.zim` found sixteen `be`
shards, approximately 6.9 MB and 63,000 records each. The leaf cache budgeted raw
JSON bytes while retaining the substantially larger decoded dictionary arrays.

Fixed the concrete memory risks:

- Name lookup (split and unsplit) and category-name fallback now parse records
  in batches of at most 128, or roughly 256 KiB; individual records over 1 MiB
  fail explicitly. Only matching records survive a batch's autorelease pool.
- The search cache owns raw JSON, limited to 24 MiB/64 entries. It evicts before
  insertion and never retains an entry larger than the byte budget.
- Cache data is copied into exact-sized storage: libzim's no-copy Data slices
  may otherwise retain an entire decompressed cluster behind a small entry.
- Invalid arrays, malformed/trailing JSON, oversized records and more than
  5,000 matches in one shard throw explicit errors rather than returning a
  misleading partial result or allowing unbounded decoded accumulation.
- Cancellation is checked between shards and decoded batches.

Adversarial tests cover nested records, braces/quotes inside strings, escaped
Unicode, case folding, batch boundaries, large nonmatching arrays, truncated and
malformed JSON, oversized records and broad-query output limits. Existing
geocoder integration tests still pass. The broad-match cap intentionally makes
extremely vague searches fail with a request for a more specific name.

Reproducible benchmark: `tools/map-search/check-shard-memory.py DIRECTORY`, with
all sixteen extracted be-*.json files. Separate optimized processes scanned the
same 170 matching records: old peak RSS 200,671,232 bytes / 3.501 seconds; new
peak RSS 50,937,856 bytes / 3.955 seconds. Matching names were identical. These
measurements isolate parsing/cache overhead; they exclude the app, language
model and libzim cluster decompression. They demonstrate about 75% lower peak
memory for this component, not an on-phone crash reproduction or a guarantee
against every termination cause.

Final Swift suite: 627 executed, one opt-in replay skipped, zero failures.
Final signed iOS build and signature gate passed. No new TestFlight upload yet.
The existing country-versus-local-name ambiguity also remains: this California
archive contains a street named Belize and several businesses whose names
include Belize; reducing memory does not establish which the user intended.


## TestFlight diagnostic update: allocation failure confirmed

Retrieved the TestFlight crash feedback with comment “Where is Belize” for
build 20260907041312. Its crash time, 2026-09-06 21:51:53.4593 -0700, matches
the last shard-loading log. The symbolicated report identifies Thread 8:

```
EXC_CRASH (SIGABRT), SIGNAL 6
swift::swift_abortAllocationFailure
_DictionaryStorage.allocate
Dictionary._conditionallyBridgeFromObjectiveC
Array._conditionallyBridgeFromObjectiveC
DefaultZimService.loadLeafChunk(pair:leaf:) — ZimService.swift:2145
DefaultZimService.geocodeResolved(...) — ZimService.swift:902
MCPToolAdapter.dispatch(...) — MCPToolAdapter.swift:645
ChatSession.executeDirectIntent(...)
```

This confirms a failed memory allocation while bridging an entire decoded
search shard into Swift dictionaries. It is an allocation-failure abort, not
a recorded jetsam termination. No Kokoro synthesis or model generation is on
the crashing stack. The batch parser replaces that whole-shard conversion;
the raw cache and owned byte copies reduce additional retention. The previous
benchmark and tests apply to that changed path. No extra code change or test
rerun was needed merely to attach this diagnostic evidence. Phone validation
of the replacement build is still outstanding; it has not yet been uploaded.

Diagnostic retrieved through Apple's read-only App Store Connect beta-feedback
crash-log endpoint. Full report retained privately at
`/private/tmp/zimfo-belize-testflight.crash`; credentials and tester contact
information are not included in repository artifacts.

## Release completion

Uploaded Zimfo 1.0 (20260907051627) using `ios/scripts/testflight-upload.sh`.
The exact distribution IPA passed the bundle/team signature gate. Xcode
reported `Upload succeeded` and `EXPORT SUCCEEDED`; the script reported
`upload submitted`. A network timeout interrupted the subsequent processing
poll, so only `testflight-assign-internal.rb` was retried. No rebuild or second
upload occurred. Apple then reported `IN_BETA_TESTING`, and the script verified
the build's relationship to `InternalTesters` (build ID
2852b126-ec8e-4c8b-8cf6-0f20cb4ae21f).

This release includes the place-follow-up corrections, confirmed allocation
failure fix, and automatic Kokoro preference. On-phone validation remains the
next check, especially the original “Where is Belize” sequence.
