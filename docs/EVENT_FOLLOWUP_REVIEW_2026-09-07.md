# Event follow-up relevance review

## Report and diagnosis

Latest phone session `2026-09-07_14-27-42.log`: after an overview of the Grand Duchy of Lithuania, “How did it end?” selected passages about Mongol rule, an attempted restoration, and a separate court. These were authentic quotations with the wrong event. The in-hand evidence shortcut accepted lexical coverage of “end” without establishing the affected subject.

## Changes

- Event-transition questions (how/why/when something ended, began, collapsed, or dissolved) require semantic evidence selection. The lexical answer path cannot reintroduce its unsupported matches when that selection fails, even through a section overview.
- Resolve the event subject before selection. “What year?” retains the preceding event instead of becoming a search for the word “end.”
- When the resolved event concerns the existing anchor article, skip search planning and external article searches. Prioritize its lead and allow sufficient windows from its own sections. Explicit different subjects retain the existing exploration search path.
- Select at most one direct event passage, avoiding an appended list of separate historical events. Returned factual text remains copied from retrieved ZIM passages. No generated factual narrative is authorized.

## Adversarial self-review

1. Rejected the first implementation: moving the question to the model was insufficient while “it” remained unresolved. Real-model replay selected a magnate-family article and later failed the year continuation.
2. Rejected the next answer: it identified the 1795 dissolution but appended an attempted restoration in 1812. Added a single strongest direct-passage limit and explicit selection instructions.
3. Checked failure paths: keyword matching cannot answer an event question after malformed/unavailable model selection. Existing source navigation, authored works, distance, age and career questions are not all forced through this event classification.
4. Preserved temporal qualifiers and existing current-information cautions; event selection does not automatically label every question historical. Named anchor matching checks the resolved anchor terms, including lower-case user text.

Tests: Swift suite executed 649 tests, one skipped, zero failures (648 runnable tests passed). The added fixture includes the three types of misleading event evidence and date-continuation checks. `eval/event_followup_qa_v1.json` also asserts against the misleading magnate, court, and restoration answers, not merely the presence of 1795.

The desktop replay uses Wikipedia June 2026; the phone has October 2025. Model selection is probabilistic and the replay does not establish general semantic correctness or validate every archive version. Source attribution proves origin, not relevance; this distinction remains central to review.

Final real-model replay: 3/3 turns passed with manual inspection. Both event and year questions select the lead's 1795 partitions passage; the sentence following it discusses the founding and is retained as part of the source window. Event selection took 10.76 seconds and year selection 10.25 seconds on the Mac, compared with the inaccurate instant keyword response. Ordinary overviews and location fast paths are unchanged. Signed iPhone compilation and signature gate passed.
