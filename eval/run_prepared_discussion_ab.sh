#!/bin/zsh

set -u

ROOT="${0:A:h:h}"
PRODUCTS="${MCPZIM_EVAL_PRODUCTS:-$ROOT/ios/build-eval/Build/Products/Debug}"
BIN="${MCPZIM_EVAL_BIN:-$PRODUCTS/MCPZimEvalCLI}"
ZIM="${ZIM:-$HOME/Downloads/wikipedia_en_all_nopic_2026-06.zim}"
BONSAI_GGUF="${BONSAI_GGUF:-$HOME/Library/Caches/huggingface/hub/models--prism-ml--Bonsai-27B-gguf/snapshots/main/Bonsai-27B-Q1_0.gguf}"
CASE="${CASE:-prepared_mongolia_topic_chat}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-/tmp/mcpzim-prepared-discussion-ab}"

if [[ ! -x "$BIN" ]]; then
  print -u2 "Missing evaluator: $BIN"
  print -u2 "Build MCPZimEvalCLI first; see eval/README.md."
  exit 2
fi
for input_path in "$ZIM" "$BONSAI_GGUF"; do
  if [[ ! -f "$input_path" ]]; then
    print -u2 "Missing input: $input_path"
    exit 2
  fi
done
if ! command -v jq >/dev/null; then
  print -u2 "jq is required to summarize the reports."
  exit 2
fi

mkdir -p "$OUT_DIR"
typeset -A statuses

run_mode() {
  local mode="$1"
  local report="$OUT_DIR/$mode.json"
  local log="$OUT_DIR/$mode.log"
  print "Running $CASE · $mode · seed $SEED"
  env DYLD_FRAMEWORK_PATH="$PRODUCTS" "$BIN" --probe-discuss \
    --zim "$ZIM" \
    --gguf "$BONSAI_GGUF" \
    --suite "$ROOT/eval/conversational_qa_v1.json" \
    --case "$CASE" \
    --prep-mode "$mode" \
    --seed "$SEED" \
    --report-json "$report" >"$log" 2>&1
  statuses[$mode]=$?
  print "  exit ${statuses[$mode]} · $report · $log"
}

run_mode semantic-sections
run_mode none

print
printf '%-18s %7s %8s %8s %9s %9s %9s\n' \
  mode passed prep_s peak_mb vectors turn_2_s turn_3_s
for mode in semantic-sections none; do
  jq -r '
    ([.turns[].preparation | select(. != null)][0]) as $prep |
    [.preparationStrategy,
     ((.passedTurns|tostring) + "/" + ((.passedTurns + .failedTurns)|tostring)),
     ($prep.elapsedSeconds|tostring),
     (.peakFootprintMB|tostring),
     ($prep.vectorCount|tostring),
     (.turns[1].elapsedSeconds|tostring),
     (.turns[2].elapsedSeconds|tostring)] | @tsv' "$OUT_DIR/$mode.json" |
    while IFS=$'\t' read -r m pass prep peak vectors turn2 turn3; do
      printf '%-18s %7s %8.3f %8.1f %9s %9.3f %9.3f\n' \
        "$m" "$pass" "$prep" "$peak" "$vectors" "$turn2" "$turn3"
    done
done

jq -n \
  --slurpfile semantic "$OUT_DIR/semantic-sections.json" \
  --slurpfile baseline "$OUT_DIR/none.json" '
  {
    paired_seed: $semantic[0].samplingSeed,
    answers_identical: ([range(0; $semantic[0].turns|length) as $i |
      $semantic[0].turns[$i].answer == $baseline[0].turns[$i].answer] | all),
    grounding_identical: ([range(0; $semantic[0].turns|length) as $i |
      $semantic[0].turns[$i].groundingSections == $baseline[0].turns[$i].groundingSections] | all)
  }'

if [[ "${statuses[semantic-sections]}" -ne 0 || "${statuses[none]}" -ne 0 ]]; then
  exit 1
fi
