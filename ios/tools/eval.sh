#!/usr/bin/env bash
#
# Run the Zimfo conversational eval (ConversationalEvalTests) and
# optionally sample Apple Neural Engine utilization / power for the
# whole run. ANE numbers matter for the Apple Foundation Models
# provider — GPU stays idle there, so Activity Monitor's GPU column
# misses the real compute.
#
# Usage:
#   tools/eval.sh                        # default scenarios, no ANE sample
#   tools/eval.sh --model apple-foundation-models-native
#   tools/eval.sh --ane                  # also sample ANE (needs sudo)
#   tools/eval.sh --ane --model gemma4-4b-it-4bit
#   tools/eval.sh --streetzim /abs/path.zim --wikipedia /abs/path.zim
#
# Defaults target the maintainer's DC test setup; override the two
# paths if you're pointing at different ZIMs.

set -euo pipefail

STREETZIM="${ZIMBLE_TEST_STREETZIM:-/Users/jasontitus/experiments/streetzim/osm-washington-dc-test.zim}"
WIKIPEDIA="${ZIMBLE_TEST_WIKIPEDIA:-/Users/jasontitus/Downloads/wikipedia_en_all_nopic_2026-03.zim}"
MODEL_ID="${ZIMBLE_TEST_MODEL_ID:-}"
SAMPLE_ANE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL_ID="$2"; shift 2 ;;
    --ane)        SAMPLE_ANE=1; shift ;;
    --streetzim)  STREETZIM="$2"; shift 2 ;;
    --wikipedia)  WIKIPEDIA="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

LOG_DIR="/tmp/zimfo-eval"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
TEST_LOG="$LOG_DIR/$STAMP-tests.log"
ANE_LOG="$LOG_DIR/$STAMP-powermetrics.log"

PM_PID=""
SUDO_KEEPER_PID=""
cleanup() {
  [[ -n "$PM_PID" ]] && sudo -n kill "$PM_PID" 2>/dev/null || true
  [[ -n "$SUDO_KEEPER_PID" ]] && kill "$SUDO_KEEPER_PID" 2>/dev/null || true
}
trap cleanup EXIT

if [[ $SAMPLE_ANE -eq 1 ]]; then
  # powermetrics needs root; prime sudo up front so the long eval run
  # doesn't block on a password prompt mid-stream.
  echo "Priming sudo for powermetrics…"
  if ! sudo -v; then
    echo "sudo unavailable; rerun without --ane (or configure sudoers)." >&2
    exit 1
  fi
  # Keep sudo timestamp alive across the run.
  ( while true; do sudo -n true 2>/dev/null || exit; sleep 30; done ) &
  SUDO_KEEPER_PID=$!
  # -s ane → ANE power + utilization
  # -s cpu_power,gpu_power → context, so we can see whether CPU/GPU
  #   backends are firing too
  # -i 1000 → sample every second
  sudo powermetrics -s ane,cpu_power,gpu_power -i 1000 > "$ANE_LOG" 2>&1 &
  PM_PID=$!
  echo "powermetrics → $ANE_LOG (pid=$PM_PID)"
fi

# Build the xcodebuild env. `TEST_RUNNER_` is the prefix xcodebuild
# strips when forwarding env vars into the test process.
ENV_ARGS=(
  "TEST_RUNNER_ZIMBLE_TEST_STREETZIM=$STREETZIM"
  "TEST_RUNNER_ZIMBLE_TEST_WIKIPEDIA=$WIKIPEDIA"
)
if [[ -n "$MODEL_ID" ]]; then
  ENV_ARGS+=("TEST_RUNNER_ZIMBLE_TEST_MODEL_ID=$MODEL_ID")
fi

echo "Running eval — model=${MODEL_ID:-default}  streetzim=$STREETZIM"
echo "Full test log: $TEST_LOG"
cd "$(dirname "$0")/.."
env "${ENV_ARGS[@]}" xcodebuild test \
  -scheme MCPZimChatMacTests \
  -destination 'platform=macOS' \
  -only-testing MCPZimChatMacTests/ConversationalEvalTests \
  2>&1 | tee "$TEST_LOG" > /dev/null || true

if [[ -n "$PM_PID" ]]; then
  sudo -n kill "$PM_PID" 2>/dev/null || true
  sleep 1
fi

echo
echo "=== Test results ==="
grep -E "Test Case.*passed|Test Case.*failed|Executed [0-9]+ tests" "$TEST_LOG" | tail -40

if [[ $SAMPLE_ANE -eq 1 && -s "$ANE_LOG" ]]; then
  echo
  echo "=== ANE / CPU / GPU power summary (from $ANE_LOG) ==="
  # Average ANE power across samples.
  awk '
    /ANE Power:/    { sum_ane += $3; n_ane++ }
    /CPU Power:/    { sum_cpu += $3; n_cpu++ }
    /GPU Power:/    { sum_gpu += $3; n_gpu++ }
    END {
      if (n_ane > 0) printf "ANE: mean %.1f mW over %d samples\n", sum_ane / n_ane, n_ane
      if (n_cpu > 0) printf "CPU: mean %.1f mW over %d samples\n", sum_cpu / n_cpu, n_cpu
      if (n_gpu > 0) printf "GPU: mean %.1f mW over %d samples\n", sum_gpu / n_gpu, n_gpu
    }' "$ANE_LOG"
  echo
  echo "(Full powermetrics output at $ANE_LOG — grep for ANE util %% etc.)"
fi
