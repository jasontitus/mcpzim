# Conversation exploration review — September 7, 2026

## Evidence reviewed

Downloaded 117 shared session logs from Firebase Storage into a private temporary directory. Scanned 198 user turns in sessions without an autorun marker, plus the latest active phone log. These counts are not 198 distinct intentional questions: speech fragments and repeated corrections occur, and some diagnostic sessions may lack an autorun marker. Reviewed answer chains for Apple TV, Bulgaria/Lithuania, Nine to Five, and the current Grand Duchy conversation; scanned the remaining question chains to identify required exploration patterns. Old failures describe historical builds, not proof that every failure still occurs today.

The latest voice session asked “What about how it affects modern Ukraine?” and “How does it affect modern Lithuania?” after a legacy answer. Both were refused. Retrieval chose Christianity/Establishment sections and then searched related articles, despite having the relevant Legacy section. A literal keyword gate separately required wording such as “affect modern” to appear in the evidence. This combination made natural follow-ups feel blocked.

## What conversation needs to support

| Observed pattern | Real examples | Expected behavior |
|---|---|---|
| Keep the subject, change the facet | Putin → youth → first job; Shaw → what did he write → what plays | Keep the person and retrieve the new relationship; don't require button wording. |
| Keep the facet, change the subject | Bulgaria joining NATO → “How about Lithuania?” | Carry the NATO-membership question across the country change, rather than restarting with a country overview. |
| Refine the last answer | Legacy → effects on modern Ukraine → effects on modern Lithuania | Preserve the historical subject, change the affected country, retain source qualifications. |
| Resolve short follow-ups | “What year?”; photons → “When were they discovered?” | Resolve against the current question/entity, not unrelated date-bearing prose. |
| Follow an entity mentioned in an answer | Bob Dylan → Joan Baez → Beatles; Apple TV → “And the OS?” | Move to an identified related entity while retaining a way back to the original subject. |
| Switch medium or meaning | Nine to Five song → “And the movie” | Resolve the same work name with the requested medium; don't accept Number Nine or an actor as a substitute. |
| Shift from place to event | Pearl Harbor Museum → Japanese attack → casualties | Recognize the event as the new target and preserve the casualty question. |
| Keep a geographic search frame | coffee → museums; search wider; directions; show listed bar | Carry category, search center, radius, and selected result independently of encyclopedia context. |
| Correct speech or interpretation | grand Dutch → grand Duchy; “No, ...”; “not ... capital punishment” | Replace the mistaken interpretation without forcing a reset or unrelated choice. |
| Change topic outright | biography → nearest bar → another biography | Honor explicit switches; don't let an old pinned subject trap the request. |
| Ask for current information | Apple TV → most recent version; battery breakthroughs | Report what the loaded archive supports and its date, not pretend offline content establishes today's latest facts. |

## Changes in this round

Before section ranking or linked-article search, a substantive follow-up now checks every loaded source section using the same exact-excerpt evidence rules. This removes the top-k recall bottleneck without allowing generated factual prose. If no supported answer is found, existing retrieval/retry behavior continues. Generic content-free follow-ups retain existing behavior.

Modern-impact questions recognize a bounded relation and preserve the affected subject. Country adjectives can match their country names for the reported Lithuania/Ukraine cases. A legacy-effect sentence must still be present. Explicit causal constructions match the affected subject after the causal phrase, preventing the actor from being mistaken for the recipient. Sentence text, dates, and qualifications remain unchanged.

This is an incremental repair, not a claim that all exploration patterns above are solved. The English relation recognizer has deliberately bounded syntax and aliases; unusual paraphrases can still miss. All-section scanning also retains the existing lexical evidence gate's limitations. It must not be confused with general semantic entailment.

## Next system design

Maintain an explicit conversation frame containing the anchor article, active facet/relation, affected entity, selected mentioned entity, and independent map-search state. Each turn should produce a proposed transition (refine facet, substitute entity, follow link, correct, or reset), validated against loaded ZIM titles/links/results. Resolve “And the movie” or “How about Lithuania?” into that frame before retrieval. Keep the original user wording for display and log the resolved query plus transition reason.

Retrieval should first inspect current evidence, then an identified linked entity, then bounded corpus search. Distinguish an unresolved entity from absent evidence and from a supported partial answer. Source text about historical influence must not be presented as an inferred claim about current politics. Corrections must be able to replace an unresolved candidate set. A general relation verifier could broaden paraphrase support, but must be benchmarked for memory, latency, and false positives before replacing the current gate.

Build a replay matrix from the examples above at three levels: frame transitions; evidence selection against full real-archive sections; and physical-phone UI/voice flows. Synthetic unit tests alone cannot certify natural exploration. Keep raw logs and locations outside git.

## Adversarial self-review and validation

- Positive tests: both reported modern-impact phrasings; source qualifications remain intact.
- Negative tests: an unrelated country, an unsupported secret-password qualifier, unsupported year syntax, and an explicit topic-change request are not silently normalized into supported legacy questions.
- A test initially caught actor/recipient confusion. Fixed before deployment.
- Complete Swift suite: 632 executed, one skipped, zero failures (631 passed), including existing Shaw, map follow-up, clarification, and legacy regressions.
- Physical replay/build evidence will be appended after verification.

The first physical replay passed Ukraine and Shaw but exposed two false-positive Lithuania excerpts: historical Moscow influence and a trailing monument aside in a causal sentence about Ukrainians. Added both source shapes to the regression and tightened modern-effect evidence to enduring/reception markers and the causal object's clause. All 631 non-skipped tests passed again. This is why a positive answer alone is not a sufficient replay pass.

Final signed iPhone build and signature gate passed; installed over Wi-Fi. The second six-turn silent replay returned the Ukraine passage in 53ms, the Lithuania answer in 54ms, and Shaw's plays in 67ms. Lithuania now selects national revival, constitutional continuity, and modern commemorative naming; the unrelated historical influence and monument aside no longer qualify as Lithuania effects. All answers are exact ZIM excerpts. Local logs: `/private/tmp/wander-final-phone.log`, `/private/tmp/wander-final-watch.log`, `/private/tmp/wander-build-final.log`, `/private/tmp/wander-tests.log`. No TestFlight upload or commit was performed in this round.

## Follow-up audit: 10:40–10:42 phone session

The user returned to the Grand Duchy and asked, in order:

1. “How does it relate to modern geopolitics?” — refused after 1.323s. Retrieval selected etymology, religion, establishment and expansion; corpus search surfaced Lithuania–Poland relations and History of Belarus but the linked-title restriction rejected the candidates.
2. “How is it relevant to the war in Ukraine?” — refused after 0.650s. Queries retained the entire historical anchor title, retrieving old wars and nobility instead of resolving the contemporary war as a distinct target.
3. “How do Ukrainians feel about Lithuania now?” — refused after 1.781s. The app eventually pulled Lithuania but selected early history, ethnic groups and tourism, while continuing to report failure against the Grand Duchy anchor.

All three are legitimate exploratory transitions. The latest modern-effect grammar does not recognize these phrasings; its earlier success on two country-specific questions was not evidence of general conversation support. The full-section scan also retains literal relevance gates. No answer in this log was fabricated, but repeated generic refusals concealed a retrieval/interpretation failure and gave no useful continuation.

The third question additionally requests present public attitudes, which requires dated evidence of public opinion. Government policy or historical cultural ties cannot establish how Ukrainians feel. The loaded archive is the October 2025 snapshot, so the app cannot establish September 2026 opinion from that archive alone. It should distinguish this limitation from historical background or documented bilateral relations it can actually retrieve.

Required repair is the frame/transition system above, with relationship-aware evidence and temporal scope. Do not simply expand a keyword whitelist or remove grounding. Allow a newly resolved modern entity to become a source target; a hyperlink from the original historical article is not a necessary condition for a user-requested subject change. Validate the new title independently. Log the interpretation, temporal scope, searched source identities and actual rejection reason. This audit changed documentation only; these three failures are not yet fixed. Raw log: `/private/tmp/geopolitics-latest.log`.

## Broader implementation follow-up

The subsequent implementation and adversarial replay findings are documented in [EXPLORATION_SYSTEM_REVIEW_2026-09-07.md](EXPLORATION_SYSTEM_REVIEW_2026-09-07.md). It adds a bounded archive-search plan, an active exploration frame, exact-path source resolution and evidence-ID selection. The model's factual prose is never rendered. This supersedes the phrase-specific repair as the fallback for unsupported exploratory transitions; existing fast paths remain available.
