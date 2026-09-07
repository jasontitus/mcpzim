# Evolving Wikipedia and StreetZIM conversations

Repository investigation, September 4, 2026. This describes the working tree,
including existing uncommitted discovery, grounding, and interface changes.
It is a code review and implementation proposal, not a device usability study.

The main opportunity is continuity: keep the place, the subject, and the
user's question connected while moving between maps, stories, and directions.
The app already has retrieval, reference resolution, related-topic extraction,
semantic section ranking, source attribution, and streaming speech. Improving
their handoffs is the first priority.

## The experience to aim for

1. **Start with a useful choice.** “What's interesting around here?” returns
   a small selection of actual places, with a story hook where Wikipedia
   coverage exists. A map-only place remains useful without an invented story.
2. **Follow the user's interest.** “The church” selects a place. “Why was it
   built?” retrieves the relevant article passages and answers that question,
   without replaying the introduction.
3. **Move between evidence sources.** “Can we walk there?” keeps that place's
   coordinates. “What happened to it during the earthquake?” keeps its article.
   A route should only be offered when the loaded map supports the requested
   travel mode; driving results must not be described as walking directions.
4. **Allow deliberate drift.** “Tell me about the architect” follows a verified
   article link. “Back to the church” returns to the earlier subject and place.
5. **Leave room to talk.** Give a short answer, optionally one relevant spoken
   invitation, and keep other actions on screen. Interrupting should lead
   directly to listening, with no leftover audio from the abandoned response.

## Findings in the implementation

| Finding | Evidence | Consequence / next action |
| --- | --- | --- |
| Suggestions and spoken reference resolution use different inputs. | `ChatSession.appendThreadOfferIfUseful` filters and reranks cards; grounded answers install their own cards. `ReferenceResolver.resolve` accepts the first `focus.openThreads` entry and uses `lastList` for ordinals. | A spoken choice can differ from a tap. Resolve explicit card selections against the latest displayed cards. Implemented in this increment. |
| A card's label is not always its action. | `DiscoveryThread.prompt` carries section questions, full-article actions, and map-only actions; the older thread resolver constructs `tell me about <label>`. | Preserve the full action when a person names a card. Implemented in this increment. |
| Place identity and article identity can diverge. | `updateFocusAfterTool` remembers Wikipedia results as `.topic`; place results use `.place` and sometimes a `wiki_path`. `DiscussionState` stores articles separately. | Introduce a shared subject containing both a verified article reference and the resolved place, retaining the association after article follow-ups. |
| Old offers can survive unrelated results. | `updateFocusAfterTool` intentionally retains `openThreads` when extraction returns nothing. | Keep candidate history for ranking, but give offers a reply ID and an explicit lifecycle. Only offers actually presented in the current reply should be accepted by “yes.” |
| Ordinary answers repeatedly teach voice commands. | `suggestionVoiceCue` appends “You can say…” and generic alternatives; `appendThreadOfferIfUseful` also changes the displayed/spoken answer. | Separate answer text from optional voice invitations. Teach commands at onboarding or on request; otherwise offer one contextual next question. |
| Retrieval already has substantial depth. | `answerWithinDiscussion` contextualizes the last question, checks coverage, expands through linked articles, and retries different passages after a coverage failure. | Preserve these mechanisms. Instrument failed retrieval and latency before adding more retrieval stages or another embedding model. |
| Voice has interruption support, but timing needs field evidence. | `VoiceChatController.interruptAndListen` cancels playback/generation; end-of-utterance silence defaults to two seconds. | Measure completion-to-first-audio and interrupt-to-listening on actual devices. Do not simply lower the silence threshold and cut off mid-sentence pauses. |

## Implemented in this increment

`ConversationSuggestionSelection` converts an exact displayed label,
“the second suggestion,” or “option 2” to the same complete prompt as a tap.
A single displayed offer also accepts “yes, please.” It preserves map actions
and section questions instead of treating their labels as article titles.

`ChatSession.send` applies this before classification and discussion routing,
using only the latest assistant reply. Pending disambiguation and factoid
clarifications retain precedence. The dispatched prompt is recorded as the
user turn, matching tapping; the original spoken choice is retained in the
existing debug log when it differs.

Bare “the second one” remains a map/search-list reference because one reply
can show both numbered places and follow-up cards. “The second suggestion” is
unambiguous. Duplicate card labels and new questions fall through unchanged.

This increment does not replace the older resolver: ambiguous “yes” with
multiple cards can still reach its first-thread fallback. Unifying the offer
lifecycle below is needed to remove that remaining guess consistently.

## Next implementation sequence

1. **Unify presented offers.** Represent each action with its originating reply,
   source, stable target, display label, and dispatch payload. Use it for taps,
   spoken choices, and the one spoken invitation. Store clarification choices
   separately. A multi-choice acknowledgement asks a brief clarifying question;
   it never selects an invisible candidate. Expire offers on a new answer,
   source change, reset, or cancelled response.
2. **Keep one subject across maps and articles.** Carry place coordinates and
   archive-qualified article paths together, with evidence of the match.
   Preserve the selected entity when the user requests a facet. Resolve “there”
   against the selected place, while “here” remains current GPS. Use the existing
   entity history for explicit returns such as “back to the church.”
3. **Make answers conversational.** Use a short-answer default for voice, with
   deeper sections on request. Keep full-article narration an explicit action.
   Track already-covered facets so “more” adds information. Put several useful
   branches on screen but speak at most one, only when it advances the exchange.
4. **Evaluate complete exchanges.** Extend the existing evaluation executable
   in `ios/MCPZimEval/EvalHarness.swift` with the scenarios below. First record
   baseline results, then compare the same archive/model/device configuration.

## Conversation acceptance scenarios

| Exchange | Required observation |
| --- | --- |
| Nearby places → select church → history → directions there | Same verified place persists across article and route requests; coordinates are not invented. |
| Article answer → second suggestion → “why?” | The displayed second action executes, then the follow-up stays on its subject. |
| Map-only park → named card | Opens the map action; no fabricated Wikipedia article or historical explanation. |
| Related architect → “back to the church” | Returns to the earlier church rather than researching the phrase literally. |
| Several cards → “yes” | Clarifies rather than choosing an unseen or unintended topic. Pending work. |
| Article → unrelated topic → “yes” | Never resurrects an expired offer. Pending work. |
| Missing passage → useful related source | One bounded recovery gives supported information or clearly explains the local coverage gap. |
| Spoken answer → interrupt → new question | Old audio stops; microphone readiness is accurate; abandoned generation cannot overwrite the new reply. |

Record correct subject/action, evidence source, duplicate answer rate, repair
turns, time to first useful text/audio, total turn time, and peak memory. Report
retrieval/routing failures separately from unsupported generated claims. Device
voice trials should include mid-sentence pauses, noisy surroundings, movement,
and mixed Wikipedia/StreetZIM coverage.

## Validation

The pure Swift selection tests cover section actions, map actions, displayed
order, single-offer acceptance, duplicate labels, empty offers, and rejection
of unrelated questions and ambiguous list references. Host integration also
requires an app build and the device conversations above before release.
