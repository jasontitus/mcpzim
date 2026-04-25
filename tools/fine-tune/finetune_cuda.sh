#!/usr/bin/env bash
# CUDA-side full pipeline: LoRA train (PEFT) → fuse → HF→GGUF → Q4_K_M.
# Mirrors finetune.sh's stage layout but swaps the mlx-lm steps for a
# torch+peft training loop in finetune_cuda.py. Steps 1, 4, 5 are
# unchanged from the Mac version (they're platform-agnostic).
#
# Usage:
#   bash finetune_cuda.sh train.jsonl                   # gemma-3-4b-it
#   BASE_MODEL=google/gemma-3-1b-it MODEL_TAG=gemma3-1b-it-ft \
#     OUT_DIR=./ft-out-gemma3-1b bash finetune_cuda.sh train.jsonl
#
# Env knobs (all optional):
#   BASE_MODEL=google/gemma-3-4b-it          # any HF causal-LM
#   MODEL_TAG=gemma3-4b-it-ft                # GGUF filename root
#   OUT_DIR=./ft-out                         # holds adapters/, fused-hf/, gguf
#   ITERS=500
#   LORA_LAYERS=16   LORA_RANK=16
#   BATCH_SIZE=4     LEARN_RATE=1e-5
#   MAX_SEQ_LEN=2048 VAL_BATCHES=5
set -euo pipefail

TRAIN_DATA="${1:-train.jsonl}"
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "error: $TRAIN_DATA not found" >&2
    exit 1
fi

BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-it}"
MODEL_TAG="${MODEL_TAG:-$(basename "$BASE_MODEL" | sed -E 's/-bf16$//')-ft}"
ITERS="${ITERS:-500}"
LORA_LAYERS="${LORA_LAYERS:-16}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
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

# --- Step 1: 95/5 split (same logic as finetune.sh) ---
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

# --- Step 2: PEFT LoRA train + fuse-hf in one Python pass ---
DATA_DIR="$OUT_DIR/data"
mkdir -p "$DATA_DIR"
cp "$TRAIN_SPLIT" "$DATA_DIR/train.jsonl"
cp "$VAL_SPLIT"   "$DATA_DIR/valid.jsonl"

if [[ ! -f "$FUSED_DIR/config.json" ]]; then
    echo ">> LoRA fine-tuning + fusing $BASE_MODEL "
    echo "   (iters=$ITERS rank=$LORA_RANK layers=$LORA_LAYERS bsz=$BATCH_SIZE)"
    "$VENV_PY" "$HERE/finetune_cuda.py" \
        --model "$BASE_MODEL" \
        --data-dir "$DATA_DIR" \
        --adapter-path "$ADAPTERS_DIR" \
        --fused-path "$FUSED_DIR" \
        --iters "$ITERS" \
        --num-layers "$LORA_LAYERS" \
        --lora-rank "$LORA_RANK" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARN_RATE" \
        --max-seq-length "${MAX_SEQ_LEN:-2048}" \
        --val-batches "${VAL_BATCHES:-5}"
fi

# --- Step 3 (optional): restore upstream tokenizer into fused dir ---
# PEFT's save_pretrained already preserves the tokenizer files we need,
# but llama.cpp's convert_hf_to_gguf.py is finicky about tokenizer.model
# vs tokenizer.json. If conversion errors with "BPE pre-tokenizer was not
# recognized", uncomment and adapt the cp from finetune.sh's Mac block.

# --- Step 3.5: ensure tokenizer.model lands in fused dir for sentencepiece-style models.
# llama.cpp's Gemma3Model.set_vocab() takes the sentencepiece path iff
# tokenizer.model exists in dir_model — otherwise it falls back to
# _set_vocab_gpt2() which fails on Gemma 3's BPE-hash whitelist. PEFT's
# tokenizer.save_pretrained() drops only tokenizer.json + config, not
# the .model file. So fetch from the HF hub cache and copy it in.
# Idempotent: skips if tokenizer.model already exists in fused dir.
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
    # Many HF models genuinely don't have a tokenizer.model (Qwen, etc.)
    # — that's fine, those use the BPE path which works without it.
    print(f"  no tokenizer.model on hub for {base_model}: {type(e).__name__}")
PY

# --- Step 4: HF → GGUF F16 ---
if [[ ! -d "$LLAMA_CPP_SRC" ]]; then
    echo ">> cloning llama.cpp source"
    git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_SRC"
    # uv-managed venvs don't include pip by default; use uv pip instead.
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
    # System cmake isn't installed on this WSL box; venv's cmake works.
    # Add it to PATH so cmake's child cmake calls find each other.
    export PATH="$(dirname "$VENV_PY"):$PATH"
    (cd "$LLAMA_CPP_SRC" && cmake -B build -S . -DGGML_CUDA=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF >/dev/null \
        && cmake --build build --target llama-quantize -j 8)
fi
if [[ ! -f "$GGUF_Q4" ]]; then
    echo ">> quantizing F16 → Q4_K_M"
    "$QUANTIZE_BIN" "$GGUF_F16" "$GGUF_Q4" Q4_K_M
fi

echo
echo "=== done ==="
echo "Fused HF:    $FUSED_DIR"
echo "F16 GGUF:    $GGUF_F16"
echo "Q4_K_M GGUF: $GGUF_Q4"
