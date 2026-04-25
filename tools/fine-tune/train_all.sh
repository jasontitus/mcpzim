#!/usr/bin/env bash
# Run finetune.sh against every shipping-tier base model in sequence.
#
# Each candidate gets its own OUT_DIR + MODEL_TAG so artifacts don't
# collide. Same training data goes to all of them — the eval-format
# preamble + fence-wrapped tool calls render correctly through any
# chat template that supports user/assistant role alternation.
#
# Caveat: Qwen models have a real `system` role and their chat
# template renders system separately, while our training data folds
# system + tool_block into the first user turn (Gemma 3 has no
# system role). Inference-time iOS uses QwenChatMLTemplate which
# DOES use system role — so there's a small format-mismatch the
# Qwen FT may need extra runs to absorb. Watch the val loss curve
# and the eval grid; if Qwen FT regresses harder than the Gemma
# variants, write a one-shot converter that re-emits the eval-format
# JSONL with system/user split for those candidates.
#
# Usage:
#   bash train_all.sh /path/to/train_v4_combined.jsonl
#
# Run on a Mac with reasonable GPU headroom (Mac Studio recommended).
# Expect ~30-90 min per model depending on dataset size.
set -euo pipefail

TRAIN_DATA="${1:-train_v4_combined.jsonl}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "error: $TRAIN_DATA not found" >&2
    echo "usage: bash train_all.sh /path/to/train_v4_combined.jsonl" >&2
    exit 1
fi

# Each row: BASE_MODEL | MODEL_TAG | OUT_DIR
#
# MODEL_TAG drives the final ${TAG}.Q4_K_M.gguf filename. OUT_DIR
# isolates adapters + fused-hf + GGUFs per candidate so we can A/B
# them without retraining.
CANDIDATES=(
    "mlx-community/gemma-3-4b-it-bf16|gemma3-4b-it-ft|$HERE/ft-out-gemma3-4b"
    "mlx-community/gemma-3-1b-it-bf16|gemma3-1b-it-ft|$HERE/ft-out-gemma3-1b"
    "mlx-community/Qwen3.5-4B-Instruct-bf16|qwen3.5-4b-it-ft|$HERE/ft-out-qwen3.5-4b"
    "mlx-community/Qwen3.5-1.7B-Instruct-bf16|qwen3.5-1.7b-it-ft|$HERE/ft-out-qwen3.5-1.7b"
)

for spec in "${CANDIDATES[@]}"; do
    IFS='|' read -r base tag out <<< "$spec"
    echo
    echo "======================================================================"
    echo "  $tag  ($base)"
    echo "  → $out"
    echo "======================================================================"
    if [[ -f "$out/${tag}.Q4_K_M.gguf" ]]; then
        echo "  already built, skipping"
        continue
    fi
    BASE_MODEL="$base" \
    MODEL_TAG="$tag" \
    OUT_DIR="$out" \
    bash "$HERE/finetune.sh" "$TRAIN_DATA"
done

echo
echo "=== all candidates done ==="
for spec in "${CANDIDATES[@]}"; do
    IFS='|' read -r base tag out <<< "$spec"
    if [[ -f "$out/${tag}.Q4_K_M.gguf" ]]; then
        size=$(du -h "$out/${tag}.Q4_K_M.gguf" | cut -f1)
        echo "  $tag: $out/${tag}.Q4_K_M.gguf ($size)"
    else
        echo "  $tag: BUILD FAILED — see $out/"
    fi
done
