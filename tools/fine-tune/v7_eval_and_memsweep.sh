#!/usr/bin/env bash
# End-to-end: wait for v7 GGUF -> eval accuracy -> quantize Q3_K_M/Q2_K
# -> memory x accuracy sweep -> write all result .md files.
# Runs unattended; produces real numbers (no GPU contention — v7 training
# already stopped).
set -u
FT=/Users/jasontitus/experiments/mcpzim/tools/fine-tune
LS=/Users/jasontitus/experiments/mcpzim/tools/llama-smoke
QUANT="$FT/.llama.cpp-src/build/bin/llama-quantize"
GGUF="$FT/ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q4_K_M.gguf"
F16="$FT/ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.f16.gguf"
PY="$LS/.venv/bin/python"

echo "=== [$( date +%H:%M:%S )] waiting for v7 Q4_K_M GGUF + stable size ==="
prev=0
for i in $(seq 1 120); do
  if [ -f "$GGUF" ]; then
    sz=$(stat -f %z "$GGUF")
    # pipeline still running until no mlx/convert/quantize proc touches it
    if [ "$sz" = "$prev" ] && [ "$sz" -gt 4000000000 ] \
       && ! pgrep -f "finetune_lfm2|convert_hf_to_gguf|llama-quantize" >/dev/null; then
      echo "  GGUF stable at $(echo "scale=2;$sz/1073741824"|bc)GB"; break
    fi
    prev=$sz
  fi
  sleep 15
done

echo "=== [$( date +%H:%M:%S )] STEP 2: v7 Q4_K_M accuracy ==="
cd "$LS"
$PY grid.py --models lfm2.5-8b-a1b-ft --only Q4_K_M --kv q8_0/q8_0 \
    --out GRID_RESULTS_LFM25_V7.md
echo "--- v7 Q4_K_M result ---"
grep -c '✓' GRID_RESULTS_LFM25_V7.md | sed 's/^/  passes(approx ✓ count incl header): /'

echo "=== [$( date +%H:%M:%S )] STEP 3: quantize v7 -> Q3_K_M, Q2_K (CPU) ==="
cd "$FT"
[ -f ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q3_K_M.gguf ] || \
  "$QUANT" "$F16" ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q3_K_M.gguf Q3_K_M 2>&1 | tail -2
[ -f ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q2_K.gguf ] || \
  "$QUANT" "$F16" ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q2_K.gguf Q2_K 2>&1 | tail -2

echo "=== [$( date +%H:%M:%S )] STEP 4: memory x accuracy sweep ==="
cd "$LS"
# v7 weight-quant x KV-quant sweep + v6 low-quant comparison
$PY grid.py \
  --models lfm2.5-8b-a1b-ft,lfm2.5-8b-a1b-ft-q3km,lfm2.5-8b-a1b-ft-q2k,lfm2.5-v6-q3km,lfm2.5-v6-q2k \
  --only Q4_K_M,Q3_K_M,Q2_K \
  --kv q8_0/q8_0,q4_0/q4_0 \
  --out GRID_RESULTS_LFM25_MEM.md

echo "=== [$( date +%H:%M:%S )] DONE. Results:"
echo "  $LS/GRID_RESULTS_LFM25_V7.md"
echo "  $LS/GRID_RESULTS_LFM25_MEM.md"
