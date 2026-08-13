#!/usr/bin/env bash
# LFM2.5-8B-A1B fine-tune pipeline (M2 Max / mlx-lm).
#
# Diverges from finetune.sh in three places, all keyed off
# `lfm2moe`'s mlx-lm fuse output:
#
#   1. After mlx-lm fuse, `feed_forward.switch_mlp.{gate,up,down}_proj`
#      are stacked over the expert axis (shape `[n_experts, A, B]`).
#      llama.cpp's `lfm2.py` converter wants per-expert
#      `experts.{xid}.{w1,w2,w3}` — we unstack in-place.
#
#   2. mlx-lm reshapes `conv.conv.weight` from HF's `(channels,1,kernel)`
#      to `(channels,kernel,1)`. The converter's `squeeze(1)` then
#      stays 3-D instead of producing `(channels,kernel)`, and llama.cpp
#      trips `GGML_ASSERT(ggml_is_matrix(c))` at inference. We permute
#      back to HF order.
#
#   3. The convert script's BPE pre-tokenizer hash for LFM2.5 differs
#      from upstream `lfm2`; conversion/base.py has the alias patched
#      locally (see `c6952c9...` → `lfm2` in get_vocab_base_pre).
#
# Usage:
#   bash finetune_lfm2.sh train_v4_combined.jsonl   # 300 iters default
#   ITERS=500 bash finetune_lfm2.sh train_v4_combined.jsonl
set -euo pipefail

TRAIN_DATA="${1:-train_v4_combined.jsonl}"
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "error: $TRAIN_DATA not found" >&2
    exit 1
fi

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
MODEL_TAG="${MODEL_TAG:-lfm2.5-8b-a1b-ft}"
OUT_DIR="${OUT_DIR:-./ft-out-lfm2.5-8b}"
ITERS="${ITERS:-300}"
LORA_LAYERS="${LORA_LAYERS:-16}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARN_RATE="${LEARN_RATE:-1e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
GRAD_CKPT="${GRAD_CKPT:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-$HERE/.llama.cpp-src}"

ADAPTERS_DIR="$OUT_DIR/adapters"
FUSED_DIR="$OUT_DIR/fused-hf"
GGUF_F16="$OUT_DIR/${MODEL_TAG}.f16.gguf"
GGUF_Q4="$OUT_DIR/${MODEL_TAG}.Q4_K_M.gguf"
VAL_SPLIT="$OUT_DIR/valid.jsonl"
TRAIN_SPLIT="$OUT_DIR/train.jsonl"

mkdir -p "$OUT_DIR"
VENV_PY="$HERE/.venv/bin/python"

# --- Step 1: 95/5 split ---
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

# --- Step 2: LoRA fine-tune ---
if [[ ! -f "$ADAPTERS_DIR/adapters.safetensors" ]]; then
    echo ">> LoRA fine-tuning $BASE_MODEL (iters=$ITERS rank=$LORA_RANK layers=$LORA_LAYERS bsz=$BATCH_SIZE)"
    mkdir -p "$ADAPTERS_DIR"
    DATA_DIR="$OUT_DIR/data"
    mkdir -p "$DATA_DIR"
    cp "$TRAIN_SPLIT" "$DATA_DIR/train.jsonl"
    cp "$VAL_SPLIT" "$DATA_DIR/valid.jsonl"
    # `--save-every 50`: save adapter every 50 iters.
    # Background: prior 800-iter runs aborted with Metal "Abort trap: 6"
    # inside mlx-swift's command-buffer queue. First two runs crashed
    # AT the iter-100 save, so we tried `--save-every 9999` to skip
    # intermediate saves — that prevented the iter-100 abort but a
    # random later abort (iter ~150) killed the run with no checkpoint.
    # Compromise: small interval so we always have a recent saved
    # checkpoint to run the pipeline against if mlx-swift aborts again.
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
        --max-seq-length "$MAX_SEQ_LEN" \
        --val-batches 5 \
        --save-every 50 \
        ${GRAD_CKPT:+--grad-checkpoint}
fi

# --- Step 3: fuse ---
if [[ ! -f "$FUSED_DIR/config.json" ]]; then
    echo ">> fusing LoRA → $FUSED_DIR"
    "$VENV_PY" -m mlx_lm fuse \
        --model "$BASE_MODEL" \
        --adapter-path "$ADAPTERS_DIR" \
        --save-path "$FUSED_DIR" \
        --dequantize
fi

# --- Step 3.5: LFM2-specific post-fuse fixups ---
# Check if we've already done this (idempotent: per-expert tensor names
# present means we've unstacked).
if "$VENV_PY" -c "
from safetensors import safe_open
import sys
with safe_open('$FUSED_DIR/model-00001-of-00004.safetensors', framework='pt') as f:
    has_unstacked = any('experts.0.w1' in k for k in f.keys())
sys.exit(0 if has_unstacked else 1)
" 2>/dev/null; then
    echo ">> already unstacked/permuted; skipping"
else
    echo ">> unstacking experts + permuting conv.conv weights"
    export FUSED_DIR
    "$VENV_PY" - <<'PY'
import json
import os
import re
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ["FUSED_DIR"]
PROJ_TO_W = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
PAT_EXPERTS = re.compile(
    r"^(model\.layers\.\d+\.feed_forward\.)switch_mlp\.(gate_proj|down_proj|up_proj)\.weight$"
)
PAT_CONV = re.compile(r"^model\.layers\.\d+\.conv\.conv\.weight$")

shards = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors"))
weight_map: dict[str, str] = {}
for shard in shards:
    path = os.path.join(SRC, shard)
    print(f">>>> rewriting {shard}", flush=True)
    new_tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            t = f.get_tensor(k)
            m = PAT_EXPERTS.match(k)
            if m:
                base = m.group(1)
                w_name = PROJ_TO_W[m.group(2)]
                for xid in range(t.shape[0]):
                    nk = f"{base}experts.{xid}.{w_name}.weight"
                    new_tensors[nk] = t[xid].clone()
                    weight_map[nk] = shard
                continue
            if PAT_CONV.match(k):
                t = t.permute(0, 2, 1).contiguous()
            new_tensors[k] = t
            weight_map[k] = shard
    save_file(new_tensors, path, metadata=meta)

with open(os.path.join(SRC, "model.safetensors.index.json"), "w") as fh:
    json.dump({"metadata": {}, "weight_map": weight_map}, fh, indent=2)
PY
fi

# --- Step 3.6: restore upstream tokenizer (mlx fuse re-serialises) ---
# Step 3.6 is unconditional, so the old `ls -d … | head -1` aborted *every* run
# on a box without that HF snapshot: ls exits 2, pipefail propagates it, errexit
# kills the script before the convert step and the [[ -n ]] guard never gets to
# skip (bugs review, finetune_lfm2.sh:172). A glob array cannot fail; an
# unmatched pattern stays literal, so -d rejects it.
_base_snapshots=(~/.cache/huggingface/hub/models--${BASE_MODEL//\//--}/snapshots/*/)
BASE_SNAPSHOT=""
if [[ -d "${_base_snapshots[0]}" ]]; then
    BASE_SNAPSHOT="${_base_snapshots[0]%/}"
fi
if [[ -n "$BASE_SNAPSHOT" ]]; then
    echo ">> restoring upstream tokenizer into $FUSED_DIR"
    cp -L "$BASE_SNAPSHOT/tokenizer.json" \
          "$BASE_SNAPSHOT/tokenizer_config.json" \
          "$BASE_SNAPSHOT/chat_template.jinja" \
          "$FUSED_DIR/" 2>/dev/null || true
fi

# --- Step 4: HF → GGUF convert ---
if [[ ! -f "$GGUF_F16" ]]; then
    echo ">> converting fused HF → F16 GGUF"
    "$VENV_PY" "$LLAMA_CPP_SRC/convert_hf_to_gguf.py" \
        "$FUSED_DIR" --outfile "$GGUF_F16" --outtype f16
fi

# --- Step 5: Q4_K_M quantize ---
if [[ ! -f "$GGUF_Q4" ]]; then
    echo ">> quantizing → Q4_K_M"
    "$LLAMA_CPP_SRC/build/bin/llama-quantize" \
        "$GGUF_F16" "$GGUF_Q4" Q4_K_M
fi

echo
echo "=== done: $GGUF_Q4 ==="
ls -la "$GGUF_Q4"
