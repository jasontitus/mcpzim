# Off-the-rails judge — Claude workflow pass

The corpus (`eval/corpus/conversations.jsonl`) is judge-ready: each line is one
real on-device conversation with, per turn, the user text, routing, retrieved
passages, grounding sources, the assistant answer, the `[Perf]` row, any
disambiguation offer, and the TTS backend/deferral.

This pass has Claude read each un-judged conversation and score every turn, so
we accumulate a labelled record of *what* went wrong and *where the fix belongs*
— the same retrieval-vs-model split that decided the fine-tune question on
2026-08-02 (verdict: ~74% retrieval, ~11% model).

## How to run

1. `tools/logpipe/ingest.sh` — refresh the corpus from Firebase (or `--from`).
2. `python3 tools/logpipe/prep_judge.py` — writes `eval/corpus/_to_judge.jsonl`
   (conversations whose session id is not yet in `eval/corpus/verdicts.jsonl`).
3. Ask Claude Code: **"run the logpipe judge"** — it reads `_to_judge.jsonl`,
   applies the contract below (fanning out with subagents when the batch is
   large), and appends one verdict object per turn to
   `eval/corpus/verdicts.jsonl`.
4. `python3 tools/logpipe/report.py` — prints the current split and the worst
   offenders, ranked.

## Verdict contract (one JSON object per turn, appended to verdicts.jsonl)

```json
{
  "session": "2026-08-02_18-27-10",
  "turn_index": 0,
  "user": "Tell me about lead",
  "verdict": "off_rails",            // on_rails | off_rails
  "category": "retrieval",           // ok | retrieval | model | corpus | attribution | routing | tts
  "severity": "high",               // low | medium | high
  "reason": "Grounded on a 224-char disambiguation stub; answer confabulated element facts."
}
```

## Category definitions (keep these stable — they drive the trend line)

- **ok** — turn was correct and well-grounded. `verdict: on_rails`.
- **retrieval** — wrong/junk/empty passages, or the right section was never
  retrieved. The model was set up to fail. (Fix: ranking / section filter.)
- **routing** — dispatched to the wrong tool or wrong article (e.g. "Apple" →
  fruit; a factoid opener falling into the LLM tool loop). (Fix: IntentRouter.)
- **model** — the correct evidence WAS in the passages, but the answer omitted
  or garbled it. This is the only bucket a fine-tune can fix.
- **corpus** — the fact simply is not in this ZIM's article (content limit,
  not a bug). Common on simple-wiki.
- **attribution** — the answer was correct but ungrounded (answered from
  parametric knowledge with no source attached). Behavior/contract issue.
- **tts** — no audio / crash from the voice path (e.g. Kokoro deferral or the
  MLX synthesis abort), independent of answer quality.

## Judging guidance

- Judge the ANSWER against the RETRIEVED PASSAGES, not against ground truth you
  happen to know — the question is "given what it was handed, did it do the
  right thing," which is what separates retrieval from model failure.
- A correct answer with `grounding: []` and `passages: []` is **attribution**,
  not ok — flag it (the gravitational-waves turns are the canonical example).
- `tts_deferred: true` or an unclean-ended session with a heavy TTS backend →
  add a `tts` verdict for the turn even if the text was fine.
- Be terse and concrete in `reason`; it becomes the fix ticket.
