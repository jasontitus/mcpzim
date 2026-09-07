# Source-only conversation answers

User requirement: factual answers must come from the loaded ZIM archives,
rather than model training knowledge or hallucination.

## Output boundary

Wikipedia discussion answers now select complete sentences from actual
retrieved article sections. Each excerpt retains article, section, and archive
identity. Selection can use semantic ranking and query synonyms, but ranking
does not authorize new prose. Only whitespace and citation markers are
normalized after HTML extraction. Document titles/h1 headings are excluded
from the lead body so they do not repeat at the beginning of a sentence.

When the retrieved passages do not cover the question, the app says it did
not find an answer **in those passages**. It does not claim the entire archive
lacks the answer. Existing bounded section retry and linked-article retrieval
remain available.

The former word-overlap attribution threshold is gone. Negation, actors,
numbers, and qualifications cannot be rearranged and still earn a source
match. The badge says **ZIM source excerpts**, not that every statement is
semantically verified or true. Source excerpts bypass model-markup removal
and Markdown interpretation in the answer bubble.

## Every answer path

- Discussion and direct article overview/comparison: host selects source
  sentences; no answer-generation call is needed.
- Generic model loop: raw drafts stay private. Only the parsed tool invocation
  enters history; surrounding prose is discarded. Successful article tool
  results enter the same source renderer. No-tool and exhausted-loop results
  finish from current-turn tool evidence or a missing-evidence response.
- Apple native tools: the dispatcher records actual adapter results before
  the model sees them. Final model prose is discarded. The native session is
  reset afterward so its private draft cannot seed the next turn; the host
  retains the source-backed conversation anchor.
- Maps, routes, inventory, and search choices: host formats actual structured
  tool results. Full-article reading copies extracted ZIM text. Generated
  map/reduce notes and forced unconstrained summaries are removed.
- Speech consumes the approved answer message. No unapproved draft is placed
  in that message or its model-history field.

This deliberately trades freely generated paraphrases for stronger provenance
and lower article-answer latency. It is a source-excerpt conversation mode,
not a semantic entailment model. The original 75% sentence-only query gate
rejected useful sections merely because their sentences did not repeat the
heading or the question's conversational wording. Section choices now retain
their article/archive/heading identity and navigate that source directly.
Free-form matching includes the section heading; any remaining question
facets still require evidence. Named-organization and parent-identity matches
do not inherit the broad aliases used for search. Retrieval can still miss
paraphrased or distributed evidence, and the model can choose a poor tool.
Those limitations affect usefulness, not permission to invent an answer.

The reported early-life, school, career, family, and NATO conversation and
the follow-up adversarial review are documented in
[conversation repairs](CONVERSATION_RELEVANCE_2026-09-05.md).

## Evidence and correction

The physical phone's Wikipedia ZIM contains the duet sentence claiming that
performers take turns playing solo sections. This corrects the interpretation
of the earlier latency smoke: that particular questionable answer repeated
a source error. Source fidelity cannot fix an incorrect or outdated archive.

Phone source checks found Einstein's parents in the actual article:
“His parents, secular Ashkenazi Jews, were Hermann Einstein, a salesman and
engineer, and Pauline Koch.” An adversarial “secret password” question now
returns missing evidence; the previous loose ranking selected a secret FBI
dossier sentence because of the shared word “secret.”

Unit regressions cover negation reversal, swapped actors and numbers, modality,
missing facts, wrong tool-result types, parent-name distractors, comparison
sources, short sentences, initials, citation cleanup, and whole-sentence
budgets. Silent phone probes stream hostile draft prose with and without a
real tool call, inspect both live messages and stored history, and cancel
mid-draft. They require an actual model invocation and a successful tool call
for the tool scenario, avoiding a false pass caused by bypassing generation.

Final build and device results are recorded in the
[app review](APP_REVIEW_2026-09-05.md). Native Foundation Models is compiled
and its dispatch/output boundary inspected; its runtime is not covered by
the Bonsai phone smoke. Microphone and playback are intentionally untested.
