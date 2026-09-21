#!/bin/bash
# Gate a partially trained GSQ chain: export a candidate database to GGUF and score
# it with llama-perplexity on the held-out text.
#
# Why this exists: the 64-block sweep runs for hours, and drift - the composed
# student/teacher norm ratio - is only a proxy for whether the artifact is any
# good. The number that decides it is perplexity on held-out app text, measured
# with the same instrument and the same bytes for every arm. D10 of
# docs/QUANTIZATION_QUALITY_PIPELINE_2026-09-20.md defines the instrument and the
# text; the plan generator (solver/tools/export_plan.py) and the exporter
# (packing/export_qwen.py) exist, but nothing wired export -> measure -> record,
# so a mid-sweep check meant running three commands by hand and interpreting the
# output.
#
# The held-out text is the concatenation of the `prompt` key from every
# invocation-*/input.json, in sorted order. That file is built once and then
# reused, because the whole point is identical bytes for every arm: rebuilding it
# differently between arms would make the numbers incomparable. The Bonsai arm is
# measured alongside as a control - it is a fixed artifact with a known score, so
# it catches a text-construction or instrument mistake immediately.
#
# Reference points (same instrument, same text):
#   Bonsai 27B Q1_0      3.80 GB   ~7.07      the bar
#   Qwen3.8-27B bf16     54.6 GB   13.397     the ceiling (F16 GGUF)
#   our pre-fix all-Q1   3.80 GB   ~73786     the failure the fix targets
#
# Usage: gate-partial-export.sh <candidate-database> [output-dir]
set -u

cd "$(dirname "$0")/../.." || exit 1   # tools/calibration

DB="${1:?usage: gate-partial-export.sh <candidate-database> [output-dir]}"
OUT="${2:-runs/native-mps/gate-partial}"
MODEL_DIR=runs/local-inputs/model
PRISM=cuda_runtime/.context/prism
HELDOUT=runs/native-mps/heldout.txt
BONSAI=/Users/jasontitus/.cache/huggingface/hub/models--prism-ml--Bonsai-27B-gguf/snapshots/0cf7e3d21581b169b4df1de8bf01316000e2fbb7/Bonsai-27B-Q1_0.gguf
PERPLEXITY=/opt/homebrew/bin/llama-perplexity
RECORD=runs/native-mps/gate-results.jsonl

[ -f "$DB" ] || { echo "=== no candidate database at $DB ==="; exit 1; }
[ -x "$PERPLEXITY" ] || { echo "=== no llama-perplexity at $PERPLEXITY ==="; exit 1; }
mkdir -p "$OUT"

# The text is built once and never rebuilt, so every arm scores identical bytes.
if [ ! -f "$HELDOUT" ]; then
  echo "=== building the held-out text from invocation prompts ==="
  python3 - "$HELDOUT" <<'PY'
import json, pathlib, sys
src = sorted(pathlib.Path("runs/mac-pipeline-v1-20260919/activations").glob("invocation-*/input.json"))
parts = [json.loads(p.read_text())["prompt"] for p in src]
pathlib.Path(sys.argv[1]).write_text("\n".join(parts) + "\n")
print(f"  {len(parts)} invocations, {pathlib.Path(sys.argv[1]).stat().st_size} bytes")
PY
fi

echo "=== export plan ==="
PYTHONPATH=. .venv/bin/python -m solver.tools.export_plan \
  --candidate-database "$DB" --model-dir "$MODEL_DIR" --output "$OUT/plan.json" || exit 1

echo "=== export ==="
PYTHONPATH=. .venv/bin/python packing/export_qwen.py \
  --plan "$OUT/plan.json" --output-dir "$OUT/out" --prism-source "$PRISM" || exit 1

GGUF="$OUT/out/model.gguf"
[ -f "$GGUF" ] || { echo "=== exporter reported success but wrote no model.gguf ==="; exit 1; }

score() {
  label="$1"; model="$2"
  echo "=== perplexity: $label ==="
  "$PERPLEXITY" -m "$model" -f "$HELDOUT" 2>&1 | tee "$OUT/ppl-$label.log" \
    | grep -E "Final estimate|calculating perplexity over" | tail -2
}

# Control first: if this does not reproduce ~7.07, the text or the instrument is
# wrong and the candidate's number means nothing.
CONTROL=""
if [ -f "$BONSAI" ]; then
  score bonsai "$BONSAI"
  CONTROL="$(grep -oE 'Final estimate: PPL = [0-9.]+' "$OUT/ppl-bonsai.log" | tail -1 | grep -oE '[0-9.]+')"
else
  echo "=== no Bonsai artifact; the candidate number is unvalidated ==="
fi

score candidate "$GGUF"
CANDIDATE="$(grep -oE 'Final estimate: PPL = [0-9.]+' "$OUT/ppl-candidate.log" | tail -1 | grep -oE '[0-9.]+')"

python3 - "$RECORD" "$DB" "$GGUF" "$CONTROL" "$CANDIDATE" <<'PY'
import json, pathlib, sys, hashlib, datetime
record = {
    "when": datetime.datetime.now().isoformat(timespec="seconds"),
    "candidate_database": sys.argv[2],
    "artifact": sys.argv[3],
    "artifact_bytes": pathlib.Path(sys.argv[3]).stat().st_size,
    "control_bonsai_ppl": float(sys.argv[4]) if sys.argv[4] else None,
    "candidate_ppl": float(sys.argv[5]) if sys.argv[5] else None,
}
path = pathlib.Path(sys.argv[1])
with path.open("a") as stream:
    stream.write(json.dumps(record) + "\n")
print("recorded:", json.dumps(record))
PY
