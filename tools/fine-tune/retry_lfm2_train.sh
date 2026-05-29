#!/usr/bin/env bash
# Retry wrapper for the LFM2.5 LoRA training step.
#
# mlx-swift has a stochastic Metal "Abort trap: 6" cold-start bug that
# fires right after the iter-1 validation pass — sometimes on the first
# launch, sometimes not at all (v5 + v6-final ran clean; v6 attempts 1-2
# and v7 attempts 1-2 died at iter 1). It is NOT deterministic and NOT
# memory-related (23 GB free when it fires). The only known mitigation
# is to relaunch until a run gets past iter 1; once past, runs reliably
# reach hundreds of iters.
#
# This loop reruns `mlx_lm lora` (training only) until it either
#   (a) completes all $ITERS, or
#   (b) leaves a checkpoint at >= $MIN_SALVAGE_ITER after an abort
# then hands off to finetune_lfm2.sh for the fuse->convert->quantize
# pipeline (which is idempotent and skips the already-done train step
# because adapters/adapters.safetensors exists).
#
# Usage:
#   ITERS=1000 BATCH_SIZE=2 MAX_SEQ_LEN=1792 LEARN_RATE=1.5e-5 \
#     bash retry_lfm2_train.sh train_v7_filtered.jsonl
set -u

TRAIN_DATA="${1:?usage: retry_lfm2_train.sh <train.jsonl>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$HERE/.venv/bin/python"
BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
OUT_DIR="${OUT_DIR:-$HERE/ft-out-lfm2.5-8b}"
ITERS="${ITERS:-1000}"
LORA_LAYERS="${LORA_LAYERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARN_RATE="${LEARN_RATE:-1.5e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1792}"
MAX_TRIES="${MAX_TRIES:-8}"
MIN_SALVAGE_ITER="${MIN_SALVAGE_ITER:-600}"

ADAPTERS_DIR="$OUT_DIR/adapters"
DATA_DIR="$OUT_DIR/data"

# One-time 95/5 split (reuse finetune_lfm2.sh's exact logic).
mkdir -p "$OUT_DIR" "$DATA_DIR"
if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
    "$VENV_PY" - "$TRAIN_DATA" "$DATA_DIR" <<'PY'
import json, random, sys
src, ddir = sys.argv[1], sys.argv[2]
random.seed(42)
rows = [l for l in open(src) if l.strip()]
random.shuffle(rows)
cut = max(1, int(len(rows) * 0.05))
open(f"{ddir}/valid.jsonl", "w").writelines(rows[:cut])
open(f"{ddir}/train.jsonl", "w").writelines(rows[cut:])
print(f"split: train={len(rows)-cut} valid={cut}")
PY
fi

highest_ckpt() {
    ls "$ADAPTERS_DIR"/[0-9]*_adapters.safetensors 2>/dev/null \
        | grep -oE '[0-9]{7}' | sed 's/^0*//' | sort -n | tail -1
}

for try in $(seq 1 "$MAX_TRIES"); do
    echo ">>> train attempt $try/$MAX_TRIES ($(date +%H:%M:%S))"
    mkdir -p "$ADAPTERS_DIR"
    "$VENV_PY" -m mlx_lm lora \
        --model "$BASE_MODEL" --train --data "$DATA_DIR" \
        --adapter-path "$ADAPTERS_DIR" \
        --iters "$ITERS" --num-layers "$LORA_LAYERS" \
        --batch-size "$BATCH_SIZE" --learning-rate "$LEARN_RATE" \
        --fine-tune-type lora --max-seq-length "$MAX_SEQ_LEN" \
        --val-batches 5 --save-every 50 --grad-checkpoint
    rc=$?
    ckpt=$(highest_ckpt); ckpt=${ckpt:-0}
    echo ">>> attempt $try exited rc=$rc, highest checkpoint=iter $ckpt"
    if [[ $rc -eq 0 ]]; then
        echo ">>> training COMPLETED cleanly"; break
    fi
    if [[ "$ckpt" -ge "$MIN_SALVAGE_ITER" ]]; then
        echo ">>> aborted but checkpoint iter $ckpt >= $MIN_SALVAGE_ITER — salvaging"
        cp "$ADAPTERS_DIR/$(printf '%07d' "$ckpt")_adapters.safetensors" \
           "$ADAPTERS_DIR/adapters.safetensors"
        break
    fi
    echo ">>> checkpoint too early (iter $ckpt) — wiping + retrying"
    rm -f "$ADAPTERS_DIR"/*.safetensors
    sleep 5
done

if [[ ! -f "$ADAPTERS_DIR/adapters.safetensors" ]]; then
    echo "!!! no usable adapter after $MAX_TRIES tries — giving up" >&2
    exit 1
fi

echo ">>> handing off to finetune_lfm2.sh for fuse/convert/quantize"
ITERS="$ITERS" BATCH_SIZE="$BATCH_SIZE" MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  LEARN_RATE="$LEARN_RATE" bash "$HERE/finetune_lfm2.sh" "$TRAIN_DATA"
