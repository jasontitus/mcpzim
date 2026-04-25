#!/usr/bin/env bash
# Linux/CUDA equivalent of train_all.sh.
#
# Maps the mlx-community tags from train_all.sh to their HF originals:
#   mlx-community/gemma-3-4b-it-bf16    → google/gemma-3-4b-it     (gated)
#   mlx-community/gemma-3-1b-it-bf16    → google/gemma-3-1b-it     (gated)
#   mlx-community/Qwen3.5-4B-Instruct   → Qwen/Qwen3-4B            (the
#                                         ".5" in the mlx tag was a typo;
#                                         Qwen3-4B is the real model)
#   mlx-community/Qwen3.5-1.7B-Instruct → Qwen/Qwen3-1.7B
#
# If you don't have license access to google/gemma-3-*, swap in the
# unsloth/* mirror — same weights, ungated.
#
# Usage:
#   bash train_all_cuda.sh /path/to/train_v4_combined.jsonl
set -euo pipefail

TRAIN_DATA="${1:-train_v4_combined.jsonl}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "error: $TRAIN_DATA not found" >&2
    exit 1
fi

# Per-row batch size: Qwen3-1.7B fits at bsz=4 (peak 20.5GB on a 24GB
# card). Gemma 3 thrashes at bsz=4 even at the 1B size — its sliding-
# window attention pattern blows up under SDPA + grad-checkpoint and
# pegs all 24GB → cudaMalloc thrashing → 7× slower than Qwen at the
# same params. So Gemma 1B drops to bsz=2 and Gemma 4B to bsz=1. The
# 4B Qwen stays at bsz=2 (text-only, no SWA bloat).
CANDIDATES=(
    "Qwen/Qwen3-1.7B|qwen3-1.7b-it-ft|$HERE/ft-out-qwen3-1.7b|4"
    "google/gemma-3-1b-it|gemma3-1b-it-ft|$HERE/ft-out-gemma3-1b|4"
    "Qwen/Qwen3-4B|qwen3-4b-it-ft|$HERE/ft-out-qwen3-4b|2"
    "google/gemma-3-4b-it|gemma3-4b-it-ft|$HERE/ft-out-gemma3-4b|2"
)

for spec in "${CANDIDATES[@]}"; do
    IFS='|' read -r base tag out bsz <<< "$spec"
    echo
    echo "===================================================================="
    echo "  $tag  ($base)  bsz=$bsz"
    echo "  → $out"
    echo "===================================================================="
    if [[ -f "$out/${tag}.Q4_K_M.gguf" ]]; then
        echo "  already built, skipping"
        continue
    fi
    BASE_MODEL="$base" \
    MODEL_TAG="$tag" \
    OUT_DIR="$out" \
    BATCH_SIZE="$bsz" \
    bash "$HERE/finetune_cuda.sh" "$TRAIN_DATA"
done

echo
echo "=== all candidates done ==="
for spec in "${CANDIDATES[@]}"; do
    IFS='|' read -r _ tag out _ <<< "$spec"
    if [[ -f "$out/${tag}.Q4_K_M.gguf" ]]; then
        size=$(du -h "$out/${tag}.Q4_K_M.gguf" | cut -f1)
        echo "  $tag: $out/${tag}.Q4_K_M.gguf ($size)"
    else
        echo "  $tag: BUILD FAILED — see $out/"
    fi
done
