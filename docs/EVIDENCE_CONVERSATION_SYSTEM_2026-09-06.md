# Evidence-based conversation system

## Goal

Keep the fast, offline, source-excerpt path while separating three questions:
what the user wants, which entity they mean, and whether a retrieved sentence
actually answers them. Word overlap and a Wikipedia citation alone are not
sufficient evidence of a requested relationship.

## Design implemented

1. `EvidenceQuestion` parses a bounded authorship contract: a requested action
   (write/author/pen) and an optional work category (plays, novels, etc.).
   Irregular morphology such as write/wrote/written is normalized. This is the
   first supported relationship, not a universal semantic parser.
2. `SourceBoundAnswer` requires an asserted relationship between the current
   subject and identifiable works. The subject must be named, or a pronoun
   must occur in an unbroken, locally anchored run of source sentences. It
   rejects incidental mentions, third-party writing, non-asserted actions,
   and explicit conflicting work categories. Other question facets still
   require evidence. Output remains complete source sentences with provenance.
3. Authorship turns inspect the sections already in hand before the semantic
   top-k cut. A lead sentence listing Shaw's plays cannot disappear just
   because an influence section ranks highly. A computed selection is reused
   by the renderer rather than evaluated twice.
4. Retrieval attempts return an explicit evidence verdict. Failed attempts
   stay internal until section and linked-source retries have completed.
   Only the final failure is presented/spoken. No successful reply is inferred
   by matching words inside generated text.
5. Title-index suggestions cannot authorize identity. Automatic resolution
   accepts exact normalized titles and actual ZIM redirect paths. A fuzzy
   miss returns real title/path/archive choices with `unconfirmed_title`,
   stopping both direct and model-tool paths before older search rescue can
   adopt an unrelated entity.
6. Category-at-location questions outrank generic named-place lookup:
   “Where is a good coffee shop in Salinas” becomes coffee shops around
   Salinas. The geocoder never progressively drops arbitrary name tokens.
   When it separates an explicit geographic qualifier, the record's location
   must still satisfy that qualifier. A failed nearby search clears its saved
   search center instead of retaining stale coordinates.

## Adversarial self-review rounds

| Round | Attack / failure | Resolution |
| --- | --- | --- |
| Design | A citation proves source location, not that Shaw wrote the named work | Separate relationship checks from lexical retrieval |
| Initial implementation | `authored` and possessive lists still fail the literal `write` facet | Relationship proof satisfies the predicate facet; all other facets remain mandatory |
| Entity integration | A strict title miss could still trigger app-level search-and-open rescue | Return an explicit clarification result handled by both dispatch paths |
| Geographic review | Token drops turn a coffee request into “a good”; stripping city names selects another city | Remove arbitrary drops and enforce geographic qualifiers |
| Semantic review | A new named person captures “He”; book text under Plays borrows the heading's category | Break pronoun context on subject changes and reject conflicting explicit types |
| Presentation review | Refusals are displayed during retries; wording controls retry logic | Defer failed presentation and use an evidence boolean |
| Performance review | Full source scan recomputes the chosen answer; pronoun context grows with article size | Reuse selected reply; bound preceding-sentence context to four sentences |

No independent subagent review was used. These were repeated adversarial
self-reviews with executable regressions, not merely positive examples.

## Validation

The full Swift package suite passes **617 tests**. Added cases include both
reported Shaw follow-ups, other-person and negative examples, retained extra
facets, Salinas routing, real-service fuzzy Crockett clarification, and actual
redirect-path acceptance. Geocoder tests now reject the previously intentional
unsafe K1 Kart → K1 Speed shortcut and test city-constrained fallback. Test
counts therefore include replacement of the old token-drop behavior tests.

A signed iOS build is verified separately after final integration. No new
inference weights, KV settings, TTS settings, or launch-memory fixes are
changed by this system. No physical phone replay has been performed this turn.

## Limits and next extensions

The authorship recognizer is deliberately conservative English grammar. It
can miss passive voice, complex pronouns, tables, or unusual titles. A miss
must remain a retrieval/clarification result; it is not license to invent.
Additional relationships should get separate contracts and negative fixtures,
not increasingly broad global synonym lists. Full place-name ambiguity (two
cities with the exact same name), list/table extraction, and other biography
relations remain separate extensions. This patch does not claim complete
natural-language entailment or guarantee that the archive itself is accurate.

Suggested phone replay: Shaw overview → “What did he write?” → “What plays did
he write?” → “What novels did he write in prison?”; then a new conversation
for coffee in Salinas, a category follow-up, and “Who was David crocket?”.
Check source identity, exact excerpt provenance, a single final response, map
center, latency, and whether uncertain names produce choices rather than facts.

## Historical conversation replay

Retrieved the now-posted `2026-09-06_20-27-01.log` and reviewed all downloaded
September 5–6 sessions. Extracted **159 logged turns** and **77 recorded excerpt
answers**. Ran the same `ConversationLogReplayTests` harness against HEAD
(before this system) in an isolated temporary Swift package and against the
working tree. Inputs were identical: question, logged GPS, prior topic where
recoverable, recorded excerpt text, and logged section-choice status.

There were **four changed turns** across router output, argument values,
individual excerpt selection, and selection from pooled topic excerpts:

- Nearest coffee shop **in Carmel Valley**: previously `near_places` with the
  invalid category `coffee shop in carmel valley`; now `near_named_place` with
  category `coffee shop` and center name `carmel valley`.
- Good coffee shop **in Salinas**: previously `locate` for the whole phrase;
  now coffee shops around the named city.
- Shaw **“What did he write?”**: rejects the irrelevant recorded answer; when
  given previously quoted Shaw evidence, selects the exact sentence naming
  *Man and Superman*, *Pygmalion*, and *Saint Joan*.
- Shaw **“What plays did he write?”**: with that same topic evidence pool,
  selects the same source sentence instead of material about other writers.

The other **155 turns had identical replay outputs**. **76 of 77** individual
excerpt-answer fixtures were unchanged; the one newly rejected answer was the
reported Shaw error. Putin, Lenin, Napoleon, Einstein, Lincoln, and other
logged questions were included, along with the earlier deliberate unsupported
question probes. The only logged `geocode fallback` in these sessions was the
unsafe Salinas → “a good” match, not a useful result to preserve.

Replay limitations: source fixtures are previously emitted ZIM excerpts, not
complete original article bodies. The topic pool deliberately supplies
previously quoted evidence and is not a reconstruction of the app's exact
retrieval cache. Topic/GPS state is reconstructed; list selection, route state,
ASR, speech, WebKit, and complete turn-to-turn tool results are not simulated.
Crockett name confirmation and redirects are instead checked by the separate
real-service fixture tests. These results support compatibility but cannot
guarantee every physical-device conversation.

Local inputs/reports (not committed because they contain personal locations):
`/private/tmp/zimfo-conversation-replay.json`,
`/private/tmp/zimfo-replay-baseline.json`,
`/private/tmp/zimfo-replay-current.json`, and
`/private/tmp/zimfo-replay-diff.json`.
The opt-in replay test uses `ZIMFO_REPLAY_INPUT` and `ZIMFO_REPLAY_OUTPUT`.

Final iOS integration: signed Debug build succeeded and passed the bundle/team
signature gate. No new TestFlight upload or physical-device replay has been
performed for this conversation-system change.

Final suite after adding historical compatibility cases: **620 tests, one
opt-in diagnostic replay skipped, zero failures** (619 passed). The diagnostic
replay was run separately with its input/output environment variables against
both baseline and current code, and passed in both runs. Router benchmark:
3,000 classifications in 0.333 seconds (~111 microseconds each) on this Mac;
this is not a phone end-to-end latency measurement.

## TestFlight release

Uploaded **Zimfo 1.0 (20260907041312)** with the canonical repository script.
The exact distribution IPA passed the signature gate for `com.tiltastech.zimfo`
and team `A6G8H8NGAM`. Xcode reported `Upload succeeded` and `EXPORT SUCCEEDED`;
the script reported `upload submitted`, confirmed `IN_BETA_TESTING`, verified
assignment to `InternalTesters`, and exited successfully.

Evidence: `/private/tmp/zimfo-evidence-system-testflight.log` and
`/private/tmp/zimfo-evidence-distribution-signature.log`. Archive retained at
`ios/build-testflight/1.0-20260907041312/Zimfo.xcarchive`.
