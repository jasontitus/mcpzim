#!/usr/bin/env bash
# Run llama-smoke eval against the 4 FT'd GGUFs on pcgaming.
#
# Loops every (model × scenario) combo, runs eval.py one-shot per cell,
# captures the final RESULT line, aggregates into a markdown table.
#
# CPU inference via llama-cpp-python (already installed in the venv).
# Switch to CUDA-built llama-cpp-python later if eval throughput matters
# (~10× faster but rebuild is ~15 min on this box).
#
# Usage:
#   bash eval_ft_pcgaming.sh                    # all 4 models × 13 scenarios
#   MODELS="qwen3-1.7b-it-ft" bash eval_...     # subset of models
#   SCENARIOS="bars_sc_caltrain_chain" bash ... # subset of scenarios
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="${VENV_PY:-$HOME/mcpzim-ft/.venv/bin/python}"
EVAL_PY="${EVAL_PY:-$HERE/eval.py}"
FT_ROOT="${FT_ROOT:-$HOME/mcpzim-ft}"
OUT_DIR="${OUT_DIR:-$HERE/eval-results}"
mkdir -p "$OUT_DIR"

ALL_MODELS=(
    "qwen3-1.7b-it-ft|$FT_ROOT/ft-out-qwen3-1.7b/qwen3-1.7b-it-ft.Q4_K_M.gguf"
    "gemma3-1b-it-ft|$FT_ROOT/ft-out-gemma3-1b/gemma3-1b-it-ft.Q4_K_M.gguf"
    "qwen3-4b-it-ft|$FT_ROOT/ft-out-qwen3-4b/qwen3-4b-it-ft.Q4_K_M.gguf"
    "gemma3-4b-it-ft|$FT_ROOT/ft-out-gemma3-4b/gemma3-4b-it-ft.Q4_K_M.gguf"
)

ALL_SCENARIOS=(
    "bars_sc_caltrain_chain"
    "sky_is_blue_chain"
    "restaurants_in_sf"
    "nearby_stories_palo_alto"
    "tell_me_about_palo_alto"
    "compare_musk_bezos"
    "relations_us_iran"
    "narrate_hp_garage"
    "what_is_here_in_sf"
    "grav_waves_chain"
    "wwi_vs_wwii_chain"
    "french_revolution_chain"
    "crispr_chain"
)

# Optional filtering via env.
MODELS_FILTER="${MODELS:-}"
SCENARIOS_FILTER="${SCENARIOS:-}"

stamp=$(date +%Y%m%d-%H%M%S)
LOG="$OUT_DIR/eval-$stamp.log"
TBL="$OUT_DIR/eval-$stamp.md"

echo "# Eval results — $stamp" > "$TBL"
echo >> "$TBL"
echo "| model | scenario | pass | wall_s | peak_mb | error |" >> "$TBL"
echo "|---|---|---|---|---|---|" >> "$TBL"

total=0; passed=0
for m in "${ALL_MODELS[@]}"; do
    IFS='|' read -r mkey mpath <<< "$m"
    [[ -n "$MODELS_FILTER" && "$mkey" != *"$MODELS_FILTER"* ]] && continue
    if [[ ! -f "$mpath" ]]; then
        echo "[skip] $mkey: $mpath not found" | tee -a "$LOG"
        continue
    fi
    for s in "${ALL_SCENARIOS[@]}"; do
        [[ -n "$SCENARIOS_FILTER" && "$s" != *"$SCENARIOS_FILTER"* ]] && continue
        total=$((total+1))
        echo "==> $mkey / $s" | tee -a "$LOG"
        out=$("$VENV_PY" "$EVAL_PY" \
            --local-path "$mpath" --scenario "$s" \
            --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn \
            2>&1 | tee -a "$LOG" | grep "^RESULT " | tail -1 || true)
        if [[ -z "$out" ]]; then
            row="| $mkey | $s | ✗ | - | - | no RESULT line |"
        else
            # parse 'RESULT scenario=X passed=Y wall_s=Z peak_mb=W ...'
            pass=$(echo "$out" | grep -oE "passed=[A-Za-z]+" | cut -d= -f2)
            wall=$(echo "$out" | grep -oE "wall_s=[0-9.]+" | cut -d= -f2)
            peak=$(echo "$out" | grep -oE "peak_mb=[0-9]+" | cut -d= -f2)
            sym="✗"; [[ "$pass" == "True" ]] && { sym="✓"; passed=$((passed+1)); }
            row="| $mkey | $s | $sym | $wall | $peak |  |"
        fi
        echo "$row" >> "$TBL"
    done
done

echo >> "$TBL"
echo "**Total: $passed / $total passed**" >> "$TBL"
echo >> "$TBL"
echo "Done. $passed / $total passed."
echo "Markdown: $TBL"
echo "Full log: $LOG"
