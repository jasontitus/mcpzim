#!/bin/bash
# Reclaim superseded per-block roots so a long sweep can finish.
#
# Why this exists: each block leaves a store under `checkpoints/gsq/<stamp>` and an
# output directory `runs/native-mps/gsq-pb-<stamp>`, and neither is released as the
# sweep advances. Measured on the real model a block costs roughly 15 GiB of store
# plus up to 28 GiB of output, so an unpruned 64-block sweep exhausts the volume and
# the driver stops on its own free-space guard (exit 3). D8 of
# docs/QUANTIZATION_QUALITY_PIPELINE_2026-09-20.md states the requirement - "an
# unpruned run cannot finish, and the abandoned roots must be reclaimed first" -
# and this is the implementation of it.
#
# Why keeping two is safe. The store is content addressed, and every publish carries
# *every* completed block's archive: solver/run.py says of them "Every completed
# block's archive stays in `extras`, and so in every later payload, because it is the
# only record of that block's trained weights: `database` is rebuilt on resume by
# `restore_candidates` from the archives a snapshot carries". The newest store is
# therefore a complete record of the whole calibration, and the per-block driver only
# ever resumes from the *previous* attempt's store. Keeping the newest two of each
# leaves every root the next block can name.
#
# Confinement. `SINCE` is a stamp prefix (`YYYYMMDD-HHMMSS`); only roots whose stamp
# is >= SINCE are considered, so a run can be confined to the roots it created and
# pre-existing ones are never touched. Stamps sort chronologically as strings.
#
# Usage: reclaim-superseded.sh [keep] [SINCE] [--apply]
#   keep    how many newest of each kind to retain (default 2)
#   SINCE   stamp prefix; default empty means every root is eligible
#   --apply actually delete; without it this only reports
set -u

cd "$(dirname "$0")/../.." || exit 1

KEEP=2
SINCE=""
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    ''|*[!0-9]*) SINCE="$arg" ;;
    *) KEEP="$arg" ;;
  esac
done

STORE_ROOT=runs/native-mps/checkpoints/gsq
OUT_GLOB=runs/native-mps/gsq-pb-*
ARCHIVE=runs/native-mps/drift-archive
FREED=0
REMOVED=0

# Only roots the per-block driver created are eligible: its stamp is
# `YYYYMMDD-HHMMSS-xxxx`. Deliberately named roots (`validate-*`, `rco-*`) are not
# swept, because a name that is not a timestamp sorts after every timestamp and a
# keep-the-newest rule would otherwise treat an intentional artifact as the oldest
# thing in the directory and delete it - which is exactly what this check prevents,
# caught on a 54 GiB validation store.
is_stamp() {
  case "$1" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]) return 0 ;;
    *) return 1 ;;
  esac
}

process() {
  label="$1"
  shift
  kept=0
  # `ls -dt` sorts newest first, which is the order the keep rule needs.
  for path in $(ls -dt "$@" 2>/dev/null); do
    base="$(basename "$path")"
    # Output directories carry the driver's `gsq-pb-` prefix; stores do not.
    stamp="${base#gsq-pb-}"
    is_stamp "$stamp" || continue
    # A prefix match means SINCE names this very stamp; a '<' means it predates it.
    if [ -n "$SINCE" ] && [ "$stamp" \< "$SINCE" ]; then
      continue
    fi
    kept=$((kept + 1))
    if [ "$kept" -le "$KEEP" ]; then
      echo "keep   $label $base"
      continue
    fi
    size=$(du -sk "$path" 2>/dev/null | awk '{print $1}')
    size=${size:-0}
    # Preserve the small metadata before deleting the root. `drift.json` is
    # attempt-scoped and holds the per-block composed-stream measurement - the
    # metric this whole port is judged on - and each store records only its own
    # single commit, so a pruned output directory takes its drift history with it
    # and the value cannot be recovered afterwards. That happened to blocks 8-10
    # of the 2026-09-20 sweep before this archiving existed.
    if [ "$APPLY" -eq 1 ]; then
      dest="$ARCHIVE/$base"
      mkdir -p "$dest"
      for meta in drift.json gsq-report.json performance-report.json latest-checkpoint.json; do
        [ -f "$path/$meta" ] && cp "$path/$meta" "$dest/" 2>/dev/null
      done
    fi
    if [ "$APPLY" -eq 1 ]; then
      echo "remove $label $base ($((size / 1024)) MiB)"
      rm -rf "$path"
    else
      echo "would remove $label $base ($((size / 1024)) MiB)"
    fi
    FREED=$((FREED + size))
    REMOVED=$((REMOVED + 1))
  done
}

echo "=== reclaim: keep=$KEEP since='${SINCE:-<all>}' apply=$APPLY ==="
process store "$STORE_ROOT"/*
process output $OUT_GLOB

if [ "$APPLY" -eq 1 ]; then
  echo "=== removed $REMOVED root(s), reclaimed $((FREED / 1024 / 1024)) GiB ==="
else
  echo "=== $REMOVED root(s) eligible, $((FREED / 1024 / 1024)) GiB reclaimable (dry run) ==="
fi
