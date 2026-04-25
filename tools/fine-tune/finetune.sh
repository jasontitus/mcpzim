#!/usr/bin/env bash
# Full fine-tune pipeline for Gemma 3 4B IT on Mac M2 Max via mlx-lm.
#
# Pipeline:
#   1. Download full-precision Gemma 3 4B IT weights from HF (bf16,
#      not the 4-bit quant — LoRA needs FP weights)
#   2. LoRA fine-tune on train.jsonl via `mlx_lm.lora`
#   3. Fuse LoRA weights back into the base model
#   4. Convert fused HF weights to GGUF via llama.cpp's
#      `convert_hf_to_gguf.py`
#   5. Quantize the GGUF to Q4_K_M (our shipping format)
#
# Expected wall-clock on M2 Max 64 GB:
#   - Download: 5-15 min (~8 GB of bf16 weights)
#   - LoRA train 500 iters @ batch 4: ~45-60 min
#   - Fuse: ~1 min
#   - HF→GGUF convert: ~5 min
#   - Q4_K_M quantize: ~2 min
# Total end-to-end: ~1.5 hours.
#
# Usage:
#   bash finetune.sh train.jsonl                   # default settings
#   ITERS=1000 LORA_RANK=32 bash finetune.sh train.jsonl
#
# The script is idempotent: each step checks for its output file and
# skips if already done, so you can ctrl-C + resume without redoing
# earlier steps.
set -euo pipefail

TRAIN_DATA="${1:-train.jsonl}"
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "error: $TRAIN_DATA not found" >&2
    echo "usage: bash finetune.sh <train.jsonl>" >&2
    exit 1
fi

# ---------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-mlx-community/gemma-3-4b-it-bf16}"
# MODEL_TAG drives the GGUF filenames so multiple base models can
# share the same parent OUT_DIR without artifact collisions. Default
# is derived from BASE_MODEL's short name + "-ft".
MODEL_TAG="${MODEL_TAG:-$(basename "$BASE_MODEL" | sed -E 's/-bf16$//; s/-it$/-it/' )-ft}"
ITERS="${ITERS:-500}"
LORA_LAYERS="${LORA_LAYERS:-16}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARN_RATE="${LEARN_RATE:-1e-5}"
OUT_DIR="${OUT_DIR:-./ft-out}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HERE/../../ios/LocalPackages/llama.cpp-swift}"

ADAPTERS_DIR="$OUT_DIR/adapters"
FUSED_DIR="$OUT_DIR/fused-hf"
GGUF_F16="$OUT_DIR/${MODEL_TAG}.f16.gguf"
GGUF_Q4="$OUT_DIR/${MODEL_TAG}.Q4_K_M.gguf"
VAL_SPLIT="$OUT_DIR/valid.jsonl"
TRAIN_SPLIT="$OUT_DIR/train.jsonl"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------
# Venv — reuse generate.py's .venv; add mlx-lm if missing.
# ---------------------------------------------------------------
VENV_PY="$HERE/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo ">> creating venv"
    uv venv --python 3.12 "$HERE/.venv"
fi
echo ">> ensuring mlx-lm + huggingface-hub installed"
uv pip install --python "$VENV_PY" -q mlx-lm huggingface_hub

# ---------------------------------------------------------------
# Step 1 — split train.jsonl into train/valid (95/5)
# ---------------------------------------------------------------
if [[ ! -f "$TRAIN_SPLIT" || ! -f "$VAL_SPLIT" ]]; then
    echo ">> splitting $TRAIN_DATA into train/valid (95/5)"
    "$VENV_PY" - <<PY
import json, random
random.seed(42)
rows = [l for l in open("$TRAIN_DATA") if l.strip()]
random.shuffle(rows)
cut = max(1, int(len(rows) * 0.05))
open("$VAL_SPLIT", "w").writelines(rows[:cut])
open("$TRAIN_SPLIT", "w").writelines(rows[cut:])
print(f"  train={len(rows)-cut} valid={cut}")
PY
fi

# ---------------------------------------------------------------
# Step 2 — LoRA fine-tune with mlx-lm
# ---------------------------------------------------------------
if [[ ! -f "$ADAPTERS_DIR/adapters.safetensors" ]]; then
    echo ">> LoRA fine-tuning $BASE_MODEL (iters=$ITERS rank=$LORA_RANK layers=$LORA_LAYERS)"
    mkdir -p "$ADAPTERS_DIR"
    # mlx-lm expects train.jsonl + valid.jsonl in a data dir.
    DATA_DIR="$OUT_DIR/data"
    mkdir -p "$DATA_DIR"
    cp "$TRAIN_SPLIT" "$DATA_DIR/train.jsonl"
    cp "$VAL_SPLIT" "$DATA_DIR/valid.jsonl"
    "$VENV_PY" -m mlx_lm lora \
        --model "$BASE_MODEL" \
        --train \
        --data "$DATA_DIR" \
        --adapter-path "$ADAPTERS_DIR" \
        --iters "$ITERS" \
        --num-layers "$LORA_LAYERS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARN_RATE" \
        --fine-tune-type lora \
        --max-seq-length "${MAX_SEQ_LEN:-2048}" \
        --val-batches "${VAL_BATCHES:-5}" \
        ${GRAD_CKPT:+--grad-checkpoint}
fi

# ---------------------------------------------------------------
# Step 3 — fuse LoRA weights into a standalone FP HF checkpoint.
# mlx_lm fuse's --export-gguf doesn't support gemma3, so we take
# the HF→GGUF convert path. After fuse, we overwrite mlx's
# re-serialised tokenizer files with the originals from the base
# model cache — llama.cpp's convert_hf_to_gguf.py hashes the
# tokenizer and rejects mlx's re-encoding with NotImplementedError:
# "BPE pre-tokenizer was not recognized". LoRA doesn't modify the
# tokenizer so this is safe.
# ---------------------------------------------------------------
if [[ ! -f "$FUSED_DIR/config.json" ]]; then
    echo ">> fusing LoRA → $FUSED_DIR"
    "$VENV_PY" -m mlx_lm fuse \
        --model "$BASE_MODEL" \
        --adapter-path "$ADAPTERS_DIR" \
        --save-path "$FUSED_DIR" \
        --dequantize
    # Restore upstream tokenizer so convert_hf_to_gguf.py accepts it.
    BASE_SNAPSHOT=$(ls -d ~/.cache/huggingface/hub/models--${BASE_MODEL//\//--}/snapshots/*/ 2>/dev/null | head -1)
    if [[ -n "$BASE_SNAPSHOT" ]]; then
        echo ">> restoring upstream tokenizer into fused-hf"
        cp -L "$BASE_SNAPSHOT/tokenizer.json" "$BASE_SNAPSHOT/tokenizer_config.json" "$BASE_SNAPSHOT/tokenizer.model" "$FUSED_DIR/" 2>/dev/null || true
    fi
fi

# ---------------------------------------------------------------
# Step 4 — HF → GGUF convert via llama.cpp's python helper.
# ---------------------------------------------------------------
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-$HERE/.llama.cpp-src}"
if [[ ! -d "$LLAMA_CPP_SRC" ]]; then
    echo ">> cloning llama.cpp source for conversion helpers"
    git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_SRC"
    uv pip install --python "$VENV_PY" -q --index-strategy unsafe-best-match -r "$LLAMA_CPP_SRC/requirements/requirements-convert_hf_to_gguf.txt"
    # convert_hf_to_gguf.py imports torch at top-level but the
    # requirements file omits it (platform-specific wheel).
    uv pip install --python "$VENV_PY" -q torch
fi
if [[ ! -f "$GGUF_F16" ]]; then
    echo ">> converting fused HF → F16 GGUF"
    "$VENV_PY" "$LLAMA_CPP_SRC/convert_hf_to_gguf.py" \
        "$FUSED_DIR" \
        --outfile "$GGUF_F16" \
        --outtype f16
fi

# ---------------------------------------------------------------
# Step 5 — quantize F16 GGUF → Q4_K_M (ships to phone)
# ---------------------------------------------------------------
QUANTIZE_BIN="${QUANTIZE_BIN:-}"
if [[ -z "$QUANTIZE_BIN" ]]; then
    # Build the quantize binary from source if needed.
    if [[ ! -x "$LLAMA_CPP_SRC/build/bin/llama-quantize" ]]; then
        echo ">> building llama-quantize"
        (cd "$LLAMA_CPP_SRC" && cmake -B build -S . -DGGML_METAL=OFF -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null \
            && cmake --build build --target llama-quantize -j 8)
    fi
    QUANTIZE_BIN="$LLAMA_CPP_SRC/build/bin/llama-quantize"
fi
if [[ ! -f "$GGUF_Q4" ]]; then
    echo ">> quantizing $GGUF_F16 → $GGUF_Q4 (Q4_K_M)"
    "$QUANTIZE_BIN" "$GGUF_F16" "$GGUF_Q4" Q4_K_M
fi

echo
echo "=== done ==="
echo "Fused HF model:    $FUSED_DIR"
echo "F16 GGUF:          $GGUF_F16"
echo "Q4_K_M GGUF:       $GGUF_Q4"
echo
echo "Next: A/B eval on the llama-smoke grid:"
echo "  cd ../llama-smoke"
echo "  # add a ModelSpec in grid.py pointing at $GGUF_Q4"
echo "  .venv/bin/python grid.py --models gemma3-4b-ft"
echo
echo "Or load into LM Studio: drag $GGUF_Q4 into ~/.cache/lm-studio/models/"
