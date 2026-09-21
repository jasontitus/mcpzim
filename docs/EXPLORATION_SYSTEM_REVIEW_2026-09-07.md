# Broader conversation exploration system — September 7, 2026

## Design

The earlier phrase-specific modern-effect repair did not handle modern geopolitics, the Ukraine war, or present public attitudes. A historical article title was being prepended to every query and new sources required links from the old article. Those constraints confused source provenance with relevance and prevented legitimate subject transitions.

The fast, source-bound path remains first. On an evidence miss, the on-device model proposes a bounded `ExplorationPlan`: a standalone resolved question, evidence need (fact/connection/opinion/exploration), temporal scope, and at most three short search queries. This is navigation data only. It is not displayed as an answer and cannot execute arbitrary tools. The host searches the loaded archive, tries exact article titles, then follows the exact archive path of a search hit through `discuss_article`. Actual ZIM redirects are authoritative; fuzzy model title substitutions are not.

The resolved question becomes the active exploration frame. Subsequent follow-ups resolve against it, while the original anchor remains available as historical context. Elliptical subject changes can retain the previous facet. Explicit ordinary topic changes continue through existing routing. The frame stores the resolved question, temporal scope and proposed subjects; it is not a persistent entity graph.

Candidate evidence consists of complete adjacent source sentences, with article, section and archive identity. Windows are ranked, diversified across articles, and bounded to 14 windows / 9,000 characters. The model selects IDs as direct evidence or related background. The host rejects malformed, duplicate, or nonexistent IDs and reconstructs output from archive text only. Model-written prose is never rendered or spoken.

Connections, current conditions and opinion questions do not become newly inferred factual claims. They receive an atomic host-written qualification plus source excerpts. Original-question checks for temporal and opinion requirements override model misclassification. Government policy is not public opinion; historical influence does not establish a cause of a modern war; an offline snapshot cannot confirm conditions today. A current-opinion miss can offer related real articles without pretending to have surveyed anyone.

## Bounds and failure behavior

- Two model calls maximum per exploration attempt: 220 planning tokens and 100 selection tokens, deterministic sampling.
- Three search queries, one top search hit considered per query, three new articles loaded maximum; exact-title probes precede search.
- Active source retention: original anchor plus four support sources.
- No model factual prose, search snippets or invented titles are accepted as evidence.
- Cancellation checked around generation and archive operations; the normal stop mechanism cancels the provider.
- Malformed plans use a bounded host-generated search plan; invalid selections fall back to existing grounded retrieval, with a diagnostic reason.
- Complete coherent source sentences retain dates and qualifications. Long windows over the cap are skipped, never truncated into claims.
- Normal direct answers keep their existing low latency. Exploration adds planning and selection latency that must be measured on the actual model/device.

## Adversarial self-review

1. Model erases “now” or relabels public sentiment as history: original-question host checks force the limitation regardless of its plan.
2. Historical ties or government support offered as public opinion: selection prompt distinguishes them; host always qualifies opinion output as context, never established current sentiment.
3. Model injects its own answer text: no free-text factual output field is used. Output is reconstructed by validated IDs.
4. Duplicate/out-of-range IDs: reject the entire selection, with regression tests.
5. Wrong title from fuzzy rescue: exact-title probes require matching identity. Search hits use their exact archive path, preserving real redirects and preventing broad-topic rewriting of a selected History/Relations article. A title is a retrieval hypothesis; the second stage still judges the passage.
6. Source contains instructions: data is separated from system instructions. The selector has no tool access; it can only return IDs. This reduces execution risk but does not prove semantic relevance.
7. Old anchor captures pronouns after a subject change: persistent exploration question is consulted before the old anchor's lexical fast path. Explicit topic changes still exit normally.
8. Stale exploration facet after an ordinary fast answer: clear the exploration frame when a normal source answer succeeds.
9. Single repetitive source consumes all evidence: cap each article at five windows and diversify the candidate pool.
10. Voice starts before caveat: caveat and excerpts are rendered atomically in the same message.

This is not a proof of entailment. The selector can still judge relevance incorrectly, and lexical candidate ranking can omit good passages. Actual-model replays are therefore required, including failures and negative controls. No claim that every historical conversation is solved follows from passing parser tests.

## Validation

Core regression tests cover bounded plan parsing and fallback, independent temporal/opinion requirements, exact source reconstruction, invalid IDs, inherited facets and date/medium transitions. Raw conversation logs remain outside git.

## Failures found in actual-model adversarial replays

The initial implementation failed semantic review despite reconstructing exact archive text. Bonsai sometimes repeated the prior question, labeled geopolitics as opinion, substituted EU for NATO, selected an unrelated NATO poll for attitudes toward Lithuania, and returned too many valid evidence IDs. The host now independently preserves current named subjects, temporal requirements and opinion intent; emphasizes the current question in model input; and caps valid selections to three after validating every ID. Duplicate or nonexistent IDs still reject the proposal. Current opinion candidates must contain explicit modern dates, and all named subjects must occur in the passage body. These checks are conservative and incomplete, not a semantic guarantee.

A second replay caught two routing issues: merely mentioning Lithuania in Bulgaria's article prevented a subject switch, and an unsuccessful exploration attempt could run twice in one turn. Historical date-facet substitutions now precede the article-mention shortcut. Short date refinements retain the resolved question. An exploration frame makes at most one exploration attempt per turn. Work medium switches retain the already resolved title and require an exact new archive match.

Retrieval also exposed deliberate title broadening in `discuss_article`: a selected History title could become the parent article. Exact search paths now bypass that broadening; a missing path returns an error instead of silently substituting the parent. Regression tests cover both cases. Bilateral article hypotheses use alphabetically ordered subjects; all hypotheses still require archive resolution.

Suggestions are restricted to retrieved sources used in the answer or covering the user's named subjects, reducing irrelevant search-result cards. Source excerpts and their limitation text are appended atomically so speech cannot start with an unqualified excerpt.

The evaluation CLI's source list was missing the production download manager and catalog types required by the LlamaCpp provider. Added those two files explicitly to its target, rather than importing unrelated sharing UI or changing production download behavior. The generated project change contains only their source entries.

### Validation history

- Swift regression suite after final evidence guard: 642 executed, one skipped, zero failures (641 passed).
- Real-model replays use the actual Bonsai Q1 weights with phone context settings on the Mac and the June 2026 no-picture Wikipedia archive. This differs from the phone's October 2025 archive and is not a phone latency or memory measurement.
- Earlier 10/12 assertion passes were **not** accepted: manual review found incorrect country/facet answers. Final replay and signed iOS build results follow below.
- Phone deployment is pending availability; the Mac's UI is locked, so validation uses the signed headless evaluation executable.

## Final review additions

- An evaluation-only disable switch (`MCPZIM_EVAL_DISABLE_EXPLORATION=1`, compiled only for the headless `MCPZIM_EVAL` target) allows control replays. It is absent from both apps.
- The 42-turn legacy control passed 20 assertions. The initial enabled run passed 21. Comparing individual turns exposed one new regression: a climate exploration frame captured “How about the Mongols?”. Explicit facets now reach the fast in-hand path; pronouns, short date refinements and chronological continuations consult the active exploration frame first.
- Malformed model output used `need: historical`, outside the declared schema. A host fallback now proposes bounded searches while preserving the question and archive-only evidence boundary. A replay recovered the Siberian expansion answer from Russia's article.
- The model retained Buddhism in its plan but dropped it during selection for “Then what happened in Soviet times”. The selector now receives the contextualized question, and the host requires inherited facet terms in the passage body. A unit test rejects unrelated Soviet events and accepts a Buddhist-monastery passage.
- The existing legacy suite is not a clean release gate. Some failures concern section/identity metadata despite responsive factual text; others are substantive pre-existing failures (parent identities, post-graduation activity, gravitational-wave questions, first-grand-duke selection and qualified Santa Rosa title resolution). The control comparison does not prove all pre-existing behavior is acceptable. These remain visible in the sanitized replay report.

Remaining limits: English lexical entity/facet guards are conservative and incomplete; lower-case names and unfamiliar demonyms may not receive the same guard coverage. The small model can misjudge relevance or retrieve a narrower event than requested. Cross-article causal answers are explicitly labeled background, not a newly proved historical explanation. This implementation broadens archive navigation; it does not solve general semantic entailment. Phone memory/latency and voice integration still require a physical-device replay.

The final review also excludes ordinary question-openers such as “Then” from the named-entity guard. They describe conversation flow, not entities which must appear verbatim in an answer. A regression test ensures “Then what happened in Soviet times” requires the Soviet term, not the word “Then”.

Sanitized actual-model replay artifacts are stored in `docs/benchmarks/exploration-2026-09-07/`. They contain the test questions, answers, source titles/sections and assertion failures, without local model/archive paths or device location. Intermediate failures are retained intentionally so the adversarial review is auditable.

## Combined replay results

The combined real-model replay executed 54 turns: 12 new exploration cases and 42 legacy conversation cases (the two StreetZIM-only conversations were skipped because this run loaded Wikipedia only). The new cases passed 12/12; legacy cases passed 24/42, compared with 20/42 with exploration disabled. No previously passing legacy turn became a failing turn in this comparison. This is assertion-level evidence, not proof of complete semantic correctness.

Manual review still caught a chronological country substitution that the assertions detected only as a missing source: a Buddhism follow-up about Mongolia quoted Buryatia. The final host guard now requires the inherited conversation subject in the resolved article identity or body as well as the inherited facet in the body. It cannot borrow Mongolia from the conversation to justify another country's article. The focused post-fix replay is recorded separately.

The 18 remaining legacy assertion failures include both real pre-existing answer gaps and restrictive metadata expectations. Do not represent this as an all-green conversational engine. The new system's source-only output prevents model factual prose, but citation provenance alone does not establish relevance or completeness. The retained reports include every failure.

Performance: direct overview/medium-switch examples were roughly 0.1–0.3 seconds on this Mac. Exploratory turns in the combined run were roughly 10–31 seconds. These are development replay timings, not controlled phone measurements; some runs overlapped builds/control work. The existing Bonsai and Kokoro inference implementations were not changed by this exploration feature.

## Completion evidence

- Focused chronology replay after the final guard: the Soviet-era follow-up quotes Mongolia's **Modern history** passage about destruction of Buddhist monasteries and Stalinist purges, and passes its content/source assertions. The only remaining failure in that three-turn case is the earlier Buddhism answer's section-label expectation, which also fails in the control.
- Final shared suite: **641 passed, one skipped, zero failures** (642 executed).
- Signed iPhone build: `BUILD SUCCEEDED`; repository signature gate passed for `com.tiltastech.zimfo`, team `A6G8H8NGAM`.
- Device tooling still reports the iPhone unavailable. This version is built but has **not** been installed on the phone or uploaded to TestFlight in this round. No commit or push was performed.
- Build/test logs remain in private temporary storage: `exploration-ios-reviewed.log`, `exploration-core-final8.log`, `exploration-combined-final.log`, `exploration-control.log`, `exploration-chronology-reviewed.log`.
