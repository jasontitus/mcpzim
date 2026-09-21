# Corrected-title recovery: Grand Duchy of Lithuania

The physical iPhone session at 07:59 transcribed “grand Duchy” as “grand
Dutch”, offered two war-list articles, then asked the same clarification even
after the user supplied the correct full title twice.

## Root causes and changes

- `ReferenceResolver.descriptiveMatches` accepted **any** shared substring
  after “the”. “Lithuania” matched both list items, ignoring “grand duchy”.
  Require every meaningful descriptor token, with token boundaries and simple
  plural support. Existing short selectors such as “the church” still work.
- Article lookup tried the spoken leading “the” as part of the title only.
  After all exact paths/index matches miss, retry once without that prefix.
  An actual title such as “The Who” retains precedence over “Who”. The retry
  preserves the selected archive and section and still requires an exact
  source identity; no fuzzy hit is automatically opened.

## Adversarial self-review

- Shared country tokens cannot override additional subject words.
- Substring “bar” cannot bind to “Barrow pit” or “Barbados”. A fixture that
  also included an actual “Bar” token was corrected: that is a legitimate
  selector, not a substring collision.
- A genuine leading-article title wins before the grammatical-prefix retry.
- Prefix removal is bounded to one retry through a private helper; repeated
  “the” inputs cannot trigger arbitrarily deep recursion.
- No change to fuzzy suggestion acceptance, factual grounding, saved place
  coordinates, Kokoro weights or its admission guard.
- The original miss still offers weak war-list suggestions. This fix makes
  correction possible; it does not claim improved speech recognition or
  typo-tolerant title ranking.

## Verification

Swift suite: 629 executed, one optional test skipped, zero failures. New tests
cover both corrected-title phrasings, clarification picks, substring negatives,
exact leading-article precedence and fallback retrieval. Signed iOS Debug
build and embedded-framework signature gate passed.

Installed directly on the paired iPhone 17 Pro Max and replayed these inputs
without microphone or TTS:

1. “Tell me about the grand Dutch of Lithuania” — still an unconfirmed-title
   response, with no fabricated factual answer.
2. “The grand Duchy of Lithuania” — fetched the correct article and returned
   exact source excerpts. About four seconds, including ordinary model tool
   routing for the bare title; not a claimed subsecond fast path.
3. “Tell me about the grand Duchy of Lithuania” — remained on that article,
   with source-bound excerpts rather than another clarification loop.

Replay completed at 08:09:12 in the phone's `2026-09-07_08-09-04.log`.
Local evidence: `/private/tmp/lithuania-phone-replay.log`,
`/private/tmp/lithuania-regression-tests.log`,
`/private/tmp/lithuania-phone-signature.log`.
Private conversation logs are not committed.

The complete prior phone log contains two memory warnings (including a later
Lennon voice turn) but no explicit voice error. Kokoro was the active engine;
this does not establish what visible error the user saw. The warning is not
evidence of a crash, and this change does not claim to fix that unidentified
message or certify sustained Kokoro memory safety.
