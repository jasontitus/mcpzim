#!/usr/bin/env bash
# Sequential Gemma 3 4B quant sweep — each eval runs alone so Metal
# memory contention doesn't muddy the peak-mem numbers the parallel
# run gave us. Rerun whenever we want clean comparable data.
set -euo pipefail
cd "$(dirname "$0")"
OUT=RESULTS_2026-04-23_SEQ.md
{
  echo "# Sequential Gemma 3 4B quant sweep — $(date +%F)"
  echo
  echo "\`\`\`"
} > "$OUT"
# The console preview below used `head -10`, which under `set -euo pipefail`
# aborted the whole sweep mid-run: multi-turn scenarios print more than 10
# matching lines, head exits early, upstream gets SIGPIPE and pipefail
# propagates it (bugs review, sweep.sh:21). awk truncates without closing the
# pipe, so a real eval.py failure still fails the sweep — `|| true` would have
# masked that, which is the exact anti-pattern the same review flags in
# ios/tools/eval.sh:89.
for q in Q4_K_M Q5_K_M Q6_K Q8_0; do
  echo "=== ${q} ==="
  .venv/bin/python eval.py \
    --repo bartowski/google_gemma-3-4b-it-GGUF \
    --file "google_gemma-3-4b-it-${q}.gguf" \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn 2>&1 \
    | tee -a "$OUT" \
    | grep -E 'RESULT|model=|final_content' \
    | awk 'NR<=10'
  echo >> "$OUT"
done
echo "\`\`\`" >> "$OUT"
echo "Wrote $OUT"
