#!/bin/bash
# Run GSQ one block per process until the model is fully calibrated.
#
# Why per-block: this port has a measured failure mode where a long-lived MPS
# process produces a non-finite loss entering a block -- block 3, reproducibly,
# after ~261 steps -- while a FRESH process restoring the same checkpoint trains
# that block normally (87/87 steps, loss 509 -> 38). Restarting per block costs
# one model load (~2 s here) and removes the failure entirely. The state a block
# ENTERS is bit-identical whether or not the process restarted -- six independent
# attempts produced byte-equal solver/optimizer/rng/scheduler -- and a two-layer CPU
# sweep reproduces a single-process run exactly, so this is not a change of science.
# The trained-block equivalence has NOT been measured on the real model, which would
# need the same block trained both ways.
#
# Each iteration:
#   1. writes a PER-ITERATION config next to the config of record -- the config of
#      record is never mutated -- with a fresh attempt_id, a fresh output
#      directory, and gsq_max_blocks=1
#   2. points resume_checkpoint at the previous attempt's store and passes that
#      store's block-boundary anchor to --resume
#   3. runs exactly one block
#   4. advances ONLY on `blocked_stop`. `completed` ends the sweep. A mid-block
#      `checkpointed_stop` resumes the same block from its own snapshot. Any other
#      status, a nonzero exit, a passed deadline, or a republished snapshot stops
#      the sweep loudly with both stores left in place.
#
# The stop is taken AFTER a block completes, with the next block's phase-start
# state published as an anchor first: `--resume` reads a phase-start commit, so a
# stop taken before the block would leave the next process with nothing to name.
#
# Usage: runs/native-mps/run-gsq-per-block.sh [max_blocks]
set -u

cd "$(dirname "$0")/../.." || exit 1      # tools/calibration

CONFIG=runs/native-mps/gsq-run.json
STORE_ROOT=runs/native-mps/checkpoints/gsq
LIMIT="${1:-64}"
# A non-numeric or zero limit makes the loop test below fail, and the `while` reads
# that as "done": the script would print the limit banner and exit 0 having launched
# nothing, recording a typo as a completed sweep.
case "$LIMIT" in
  ''|*[!0-9]*) echo "=== max_blocks must be a positive integer, got '$LIMIT' ==="; exit 2 ;;
esac
if [ "$LIMIT" -lt 1 ]; then
  echo "=== max_blocks must be at least 1, got $LIMIT ==="
  exit 2
fi
# Normalised to a whole decimal so the comparison below is numeric: jq can round-trip
# a float literal verbatim (1.8e9 stays 1.8e9), and truncating that at the first dot
# reads as "1" and refuses every block as past its deadline.
DEADLINE="$(jq -r 'if .deadline_unix then (.deadline_unix|floor|tostring) else empty end' "$CONFIG")"

BLOCK=0
ITERATIONS=0
STALLS=0
PREV_STORE=""
PREV_SNAPSHOT=""

# Optional seed, so a failed iteration can be retried with one command instead of a
# hand-built config: PB_RESUME_STORE=<attempt store> PB_RESUME_SNAPSHOT=<snapshot>.
# The block index is read from the snapshot name for the limit test and the log; the
# authoritative position still comes from the checkpoint itself.
PB_RESUME_STORE="${PB_RESUME_STORE:-}"
PB_RESUME_SNAPSHOT="${PB_RESUME_SNAPSHOT:-}"
if [ -n "$PB_RESUME_STORE" ] && [ -n "$PB_RESUME_SNAPSHOT" ]; then
  PREV_STORE="$PB_RESUME_STORE"
  PREV_SNAPSHOT="$PB_RESUME_SNAPSHOT"
  BLOCK="$(printf '%s' "$PB_RESUME_SNAPSHOT" | sed -n 's/.*-b0*\([0-9]\{1,\}\)-.*/\1/p')"
  BLOCK="${BLOCK:-0}"
  echo "=== seeded from $PREV_SNAPSHOT at block $BLOCK (store $PREV_STORE) ==="
fi

while [ "$BLOCK" -lt "$LIMIT" ]; do
  # A passed deadline would otherwise let `should_stop` end every process after a
  # single step, which reads as progress forever. Refuse to start instead.
  if [ -n "$DEADLINE" ]; then
    if [ "$(date +%s)" -ge "${DEADLINE%.*}" ]; then
      echo "=== deadline_unix ($DEADLINE) has passed; refusing to start block $BLOCK ==="
      exit 1
    fi
  fi

  ITERATIONS=$((ITERATIONS + 1))
  if [ "$ITERATIONS" -gt $((4 * LIMIT + 8)) ]; then
    echo "=== runaway: $ITERATIONS iterations without reaching block $LIMIT ==="
    exit 1
  fi

  # Refuse to start a block the volume cannot hold. Measured on the real model: a
  # block costs ~26 GiB of store plus up to ~29 GiB of output scratch, so budget
  # ~60 GiB. Filling the root volume takes the machine's other work down with it, so
  # stop cleanly instead and let the caller reclaim.
  FREE_GIB="$(df -g . | awk 'NR==2{print $4}')"
  if [ "${FREE_GIB:-0}" -lt "${MCPZIM_BLOCK_GIB:-60}" ]; then
    echo "=== ${FREE_GIB} GiB free, a block needs ~${MCPZIM_BLOCK_GIB} GiB: stopping before block $BLOCK ==="
    echo "=== reclaim superseded stores or output directories, or set MCPZIM_BLOCK_GIB to override ==="
    exit 3
  fi

  STAMP="$(date +%Y%m%d-%H%M%S)-$(openssl rand -hex 2)"
  OUT="runs/native-mps/gsq-pb-$STAMP"
  PER_CONFIG="runs/native-mps/pb-config-$STAMP.json"
  CUR_STORE="$STORE_ROOT/$STAMP"

  # `output` must not exist yet; the run refuses a non-empty one by design.
  if [ -n "$PREV_STORE" ]; then
    jq --arg a "$STAMP" --arg o "$OUT" --arg s "$PREV_STORE" \
       '.attempt_id=$a|.output=$o|.gsq_max_blocks=1|.resume_checkpoint={backend:"local",path:$s}' \
       "$CONFIG" > "$PER_CONFIG" || exit 1
  else
    jq --arg a "$STAMP" --arg o "$OUT" \
       '.attempt_id=$a|.output=$o|.gsq_max_blocks=1|del(.resume_checkpoint)' \
       "$CONFIG" > "$PER_CONFIG" || exit 1
  fi

  rc=0
  if [ -n "$PREV_STORE" ]; then
    echo "=== block $BLOCK -> $OUT (resume $PREV_SNAPSHOT from $PREV_STORE) ==="
    PYTHONPATH=. .venv/bin/python -m solver.run --config "$PER_CONFIG" --stage gsq --resume "$PREV_SNAPSHOT" || rc=$?
  else
    echo "=== block $BLOCK -> $OUT (fresh run) ==="
    PYTHONPATH=. .venv/bin/python -m solver.run --config "$PER_CONFIG" --stage gsq || rc=$?
  fi

  if [ "$rc" -ne 0 ]; then
    echo "=== FAILED at block $BLOCK (rc=$rc) ==="
    echo "=== output: $OUT   store: $CUR_STORE   config used: $PER_CONFIG ==="
    echo "=== resume with: set resume_checkpoint to $CUR_STORE and pass --resume its latest snapshot ==="
    exit 1
  fi

  REPORT="$OUT/gsq-report.json"
  STATUS="$(jq -r .status "$REPORT")"
  SNAP="$(jq -r .snapshot "$OUT/latest-checkpoint.json")"
  NEXT="$(jq -r '.next_block // -1' "$REPORT")"
  echo "=== block $BLOCK: status=$STATUS snapshot=$SNAP next_block=$NEXT ==="

  # The store still holds every snapshot, so the restored materialisation is scratch:
  # 16.61 GB per resumed run measured on the real model, and 9.14 GB of it (the
  # warmstart copies) is never read on this path. Reclaimed only after a successful
  # iteration, so a failure keeps everything for inspection.
  for restored in "$OUT"/restored-*; do
    if [ -e "$restored" ]; then rm -rf "$restored"; fi
  done

  case "$STATUS" in
    completed)
      echo "=== all blocks complete ==="
      rm -f "$PER_CONFIG"
      exit 0
      ;;
    blocked_stop)
      BLOCK="$NEXT"
      PREV_STORE="$CUR_STORE"
      PREV_SNAPSHOT="$SNAP"
      STALLS=0
      rm -f "$PER_CONFIG"
      ;;
    checkpointed_stop)
      # Mid-block controlled stop (signal, deadline, or --max-steps). The block is
      # NOT finished, so the block counter must not advance and the model must not
      # be treated as calibrated: resume the same block from this snapshot.
      # A `checkpointed_stop` always advances `global_step` and publishes a fresh
      # ordinal, so comparing snapshot names cannot detect a stall. What can go wrong
      # instead is a stop that keeps firing at the top of every process -- a repeated
      # signal, or a condition the operator has not cleared -- which creeps one step
      # per ~50 s model load. Bound the undirected case.
      STALLS=$((STALLS + 1))
      if [ "$STALLS" -gt 3 ]; then
        echo "=== $STALLS consecutive mid-block stops without finishing block $BLOCK; stopping ==="
        echo "=== output: $OUT   store: $CUR_STORE   config used: $PER_CONFIG ==="
        exit 1
      fi
      echo "=== block $BLOCK stopped mid-block; resuming it from $SNAP ==="
      PREV_STORE="$CUR_STORE"
      PREV_SNAPSHOT="$SNAP"
      rm -f "$PER_CONFIG"
      ;;
    *)
      echo "=== unexpected status '$STATUS' at block $BLOCK; stopping ==="
      echo "=== output: $OUT   store: $CUR_STORE   config used: $PER_CONFIG ==="
      exit 1
      ;;
  esac
done
echo "=== reached the block limit ($LIMIT) ==="
