# Conversational QA suite

`conversational_qa_v1.json` is the shared source of truth for real mcpzim
conversation regressions. It deliberately stores answer *anchors* instead of
exact prose so a better model can phrase an answer naturally without failing
the test.

Each conversation starts with a clean transcript. Turns inside a conversation
share focus, clarification choices, and grounded article state. Every entry can
record:

- `expected_tool`: deterministic fast-path tool, when one is required.
- `anchor_groups`: every group must have at least one case-insensitive match.
- `must_not_contain`: hallucination or regression phrases that fail the turn.
- `must_not_suggest`: case-insensitive suggestion-chip fragments that fail the
  turn, for conversational UX regressions that do not affect answer prose.
- `expected_source_titles`: every listed article must occur in the answer's
  structured grounding records.
- `expected_source_sections`: at least one listed section must occur in the
  answer's structured grounding records. Multiple names express acceptable
  retrieval alternatives, not prose matches.
- `expected_clarification`: the answer must ask the user to choose.
- `minimum_answer_sentences`: lower bound for conversational depth; use this
  sparingly on questions where a bare fact is correct but not a useful reply.
- `max_seconds_advisory`: reported as a latency regression but not a content
  failure, because Mac and phone operating points differ.
- `requires_streetzim`: skip the conversation unless a StreetZIM is supplied.

List or run cases with the headless Mac binary:

```sh
cd ios
DYLD_FRAMEWORK_PATH=build-eval/Build/Products/Debug \
  build-eval/Build/Products/Debug/MCPZimEvalCLI --probe-discuss \
  --suite ../eval/conversational_qa_v1.json --list-suite

DYLD_FRAMEWORK_PATH=build-eval/Build/Products/Debug \
  build-eval/Build/Products/Debug/MCPZimEvalCLI --probe-discuss \
  --zim ~/Downloads/wikipedia_en_all_nopic_2026-06.zim \
  --gguf /path/to/model.gguf \
  --suite ../eval/conversational_qa_v1.json --case putin_biography_followups
```

Repeat `--case` to select several conversations. With no `--case`, the runner
executes every eligible conversation and skips StreetZIM cases unless
`--streetzim` is also provided.

## Prepared-discussion iteration loop

The runner can A/B the shipping semantic-section preparation against a
same-model lexical baseline without a source edit or app rebuild:

```sh
eval/run_prepared_discussion_ab.sh
```

That one-command wrapper uses the local full Wikipedia and Bonsai paths shown
below, writes the complete logs and JSON reports under
`/tmp/mcpzim-prepared-discussion-ab`, and prints a paired summary. Override
`ZIM`, `BONSAI_GGUF`, `CASE`, `SEED`, or `OUT_DIR` as environment variables.
The underlying commands are:

```sh
export ZIM="$HOME/Downloads/wikipedia_en_all_nopic_2026-06.zim"
export BONSAI_GGUF="/path/to/Bonsai-27B-Q1_0.gguf"

cd ios
DYLD_FRAMEWORK_PATH=build-eval/Build/Products/Debug \
  build-eval/Build/Products/Debug/MCPZimEvalCLI --probe-discuss \
  --zim "$ZIM" --gguf "$BONSAI_GGUF" \
  --suite ../eval/conversational_qa_v1.json \
  --case prepared_mongolia_topic_chat \
  --prep-mode semantic-sections \
  --seed 42 \
  --report-json /tmp/mongolia-semantic.json

DYLD_FRAMEWORK_PATH=build-eval/Build/Products/Debug \
  build-eval/Build/Products/Debug/MCPZimEvalCLI --probe-discuss \
  --zim "$ZIM" --gguf "$BONSAI_GGUF" \
  --suite ../eval/conversational_qa_v1.json \
  --case prepared_mongolia_topic_chat \
  --prep-mode none \
  --seed 42 \
  --report-json /tmp/mongolia-baseline.json
```

`semantic-sections` is the shipping default. `none` still loads and pins the
same article sections, but skips the prepared embeddings, isolating their
effect. Each run reports preparation time, section/vector counts, vector
bytes, actual grounding sources, per-turn latency, peak footprint, suggestion
chips, and content failures. The JSON report preserves those fields plus every
answer for later comparison or scoring. Bonsai defaults to the shipping sampler
(temperature 1.0, top-p 0.95, top-k 20), while the evaluator uses and records
seed 42 by default. Pass `--seed` to sample another paired run; explicit sampler
flags remain available for A/Bs.

Keep changes in this order so failures stay diagnosable:

1. Add or adjust conversation expectations from a real interaction.
2. Run `none` and `semantic-sections` with the same ZIM, GGUF, and sampler.
3. Inspect retrieval/source failures before judging generated prose.
4. Listen on the Mac only after the grounded/content run passes.
5. Use the phone for final audio, lifecycle, thermal, and memory confirmation.
