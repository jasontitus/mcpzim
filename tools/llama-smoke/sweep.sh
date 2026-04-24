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
for q in Q4_K_M Q5_K_M Q6_K Q8_0; do
  echo "=== ${q} ==="
  .venv/bin/python eval.py \
    --repo bartowski/google_gemma-3-4b-it-GGUF \
    --file "google_gemma-3-4b-it-${q}.gguf" \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn 2>&1 \
    | tee -a "$OUT" \
    | grep -E 'RESULT|model=|final_content' \
    | head -10
  echo >> "$OUT"
done
echo "\`\`\`" >> "$OUT"
echo "Wrote $OUT"
