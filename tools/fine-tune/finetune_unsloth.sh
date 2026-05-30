#!/usr/bin/env bash
# CUDA pipeline using Unsloth's FastLanguageModel for the train+fuse
# step (instead of vanilla PEFT in finetune_cuda.sh). Same split →
# train+fuse → tokenizer-fix → HF→GGUF → Q4_K_M quantize chain;
# only Step 2's python differs.
#
# Use this for Qwen 3.5 / 3.6 27B QLoRA where Unsloth's optimizations
# fit on a 24 GB card AND its built-in thinking-mode toggle avoids the
# eval-loop issue we hit on Qwen3.5-9B with vanilla PEFT.
#
# Usage:
#   BASE_MODEL=Qwen/Qwen3.6-27B MODEL_TAG=qwen3.6-27b-it-ft \
#     OUT_DIR=./ft-out-qwen3.6-27b BATCH_SIZE=1 GRAD_ACCUM=4 \
#     bash finetune_unsloth.sh data/train_v4_combined.jsonl
set -euo pipefail

TRAIN_DATA="${1:-train.jsonl}"
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "error: $TRAIN_DATA not found" >&2
    exit 1
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
MODEL_TAG="${MODEL_TAG:-$(basename "$BASE_MODEL" | sed -E 's/-bf16$//')-ft}"
ITERS="${ITERS:-500}"
LORA_LAYERS="${LORA_LAYERS:-16}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARN_RATE="${LEARN_RATE:-1e-5}"
OUT_DIR="${OUT_DIR:-./ft-out}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-$HERE/.llama.cpp-src}"

ADAPTERS_DIR="$OUT_DIR/adapters"
FUSED_DIR="$OUT_DIR/fused-hf"
GGUF_F16="$OUT_DIR/${MODEL_TAG}.f16.gguf"
GGUF_Q4="$OUT_DIR/${MODEL_TAG}.Q4_K_M.gguf"
VAL_SPLIT="$OUT_DIR/valid.jsonl"
TRAIN_SPLIT="$OUT_DIR/train.jsonl"

mkdir -p "$OUT_DIR"

VENV_PY="${VENV_PY:-$HOME/mcpzim-ft/.venv/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "error: $VENV_PY not found. Create the venv first." >&2
    exit 1
fi

# bnb / cu13 jitlink fix (same as finetune_cuda.sh; harmless if unused).
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
CU13_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$CU13_LIB" ]]; then
    export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --- Step 1: 95/5 split (same logic as finetune_cuda.sh) ---
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

# --- Step 2: Unsloth LoRA train + fuse ---
DATA_DIR="$OUT_DIR/data"
mkdir -p "$DATA_DIR"
cp "$TRAIN_SPLIT" "$DATA_DIR/train.jsonl"
cp "$VAL_SPLIT"   "$DATA_DIR/valid.jsonl"

if [[ ! -f "$FUSED_DIR/config.json" ]]; then
    echo ">> Unsloth LoRA fine-tune + fuse $BASE_MODEL"
    echo "   (iters=$ITERS rank=$LORA_RANK bsz=$BATCH_SIZE x grad-accum=$GRAD_ACCUM = effective $((BATCH_SIZE * GRAD_ACCUM)))"
    "$VENV_PY" "$HERE/finetune_unsloth.py" \
        --model "$BASE_MODEL" \
        --data-dir "$DATA_DIR" \
        --adapter-path "$ADAPTERS_DIR" \
        --fused-path "$FUSED_DIR" \
        --iters "$ITERS" \
        --num-layers "$LORA_LAYERS" \
        --lora-rank "$LORA_RANK" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --learning-rate "$LEARN_RATE" \
        --max-seq-length "${MAX_SEQ_LEN:-2048}" \
        --save-every "${SAVE_EVERY:-200}"
fi

# --- Step 3.5: copy tokenizer.model into fused dir for sentencepiece-style models ---
"$VENV_PY" - "$BASE_MODEL" "$FUSED_DIR" <<'PY'
import os, shutil, sys
base_model, fused = sys.argv[1], sys.argv[2]
target = os.path.join(fused, "tokenizer.model")
if os.path.exists(target):
    print("  tokenizer.model already in fused dir; skipping")
    raise SystemExit(0)
try:
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=base_model, filename="tokenizer.model")
    shutil.copyfile(src, target)
    print(f">> copied tokenizer.model ({os.path.getsize(src)} bytes) into fused dir")
except Exception as e:
    print(f"  no tokenizer.model on hub for {base_model}: {type(e).__name__}")
PY

# --- Step 4: HF → GGUF F16 ---
if [[ ! -d "$LLAMA_CPP_SRC" ]]; then
    echo ">> cloning llama.cpp source"
    git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_SRC"
    "$HOME/.local/bin/uv" pip install --python "$VENV_PY" --index-strategy unsafe-best-match -q \
        -r "$LLAMA_CPP_SRC/requirements/requirements-convert_hf_to_gguf.txt"
fi
if [[ ! -f "$GGUF_F16" ]]; then
    echo ">> converting fused HF → F16 GGUF"
    "$VENV_PY" "$LLAMA_CPP_SRC/convert_hf_to_gguf.py" "$FUSED_DIR" \
        --outfile "$GGUF_F16" --outtype f16
fi

# --- Step 5: F16 → Q4_K_M ---
QUANTIZE_BIN="${QUANTIZE_BIN:-$LLAMA_CPP_SRC/build/bin/llama-quantize}"
if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo ">> building llama-quantize (CPU build)"
    export PATH="$(dirname "$VENV_PY"):$PATH"
    (cd "$LLAMA_CPP_SRC" && cmake -B build -S . -DGGML_CUDA=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF >/dev/null \
        && cmake --build build --target llama-quantize -j 8)
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
