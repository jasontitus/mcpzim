#!/usr/bin/env bash
# Pull new Zimfo debug logs and fold them into the conversation corpus.
#
#   tools/logpipe/ingest.sh                 # pull from Firebase Storage, then parse
#   tools/logpipe/ingest.sh --from ~/Downloads/zimfo-logs   # ingest a local dump
#                                           # (AirDrop / Save-to-Files bulk export)
#
# Transport: the app auto-uploads each finished session log to
#   gs://tiltastech-zimfo.firebasestorage.app/debug-logs/<device>/<name>.log
# (works over cellular — a walking session lands without any fiddling). This
# script rsyncs that prefix down, so re-runs only fetch what's new. Requires
# `gcloud auth login` once. `--from` skips the cloud and ingests a directory
# of .log files instead.
#
# Output: eval/corpus/raw/*.log (verbatim) and eval/corpus/conversations.jsonl
# (one structured conversation per line, deduped by session id). Judge the new
# rows with tools/logpipe/judge.md.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/eval/corpus/raw"
CORPUS="$ROOT/eval/corpus/conversations.jsonl"
BUCKET="gs://tiltastech-zimfo.firebasestorage.app/debug-logs"
mkdir -p "$RAW"

if [ "${1:-}" = "--from" ]; then
  SRC="${2:?--from needs a directory}"
  echo "logpipe: copying *.log from $SRC"
  cp -n "$SRC"/*.log "$RAW"/ 2>/dev/null || true
else
  echo "logpipe: rsync $BUCKET → $RAW"
  if ! gsutil -m rsync -r "$BUCKET" "$RAW" 2>/tmp/logpipe_rsync.err; then
    echo "logpipe: gsutil rsync failed (run 'gcloud auth login', or use --from <dir>):" >&2
    tail -2 /tmp/logpipe_rsync.err >&2
    exit 1
  fi
fi

# Parse only sessions not already in the corpus (dedup by session id = log stem).
python3 - "$RAW" "$CORPUS" "$ROOT/tools/logpipe/parse_log.py" <<'PY'
import json, sys, subprocess
from pathlib import Path
raw_dir, corpus_path, parser = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

seen = set()
if corpus_path.exists():
    for line in corpus_path.open():
        try: seen.add(json.loads(line)["session"])
        except Exception: pass

new_convs, new_sessions = [], set()
for log in sorted(raw_dir.glob("*.log")):
    if log.stem in seen:
        continue
    out = subprocess.run(["python3", parser, str(log)],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        conv = json.loads(line)
        # A session can split into several conversations (reset); the first
        # keeps the stem, later ones are suffixed so dedup stays stable.
        new_convs.append(line)
        new_sessions.add(conv["session"])

with corpus_path.open("a") as f:
    for line in new_convs:
        f.write(line + "\n")

total = sum(1 for _ in corpus_path.open()) if corpus_path.exists() else 0
print(f"logpipe: +{len(new_convs)} conversations from {len(new_sessions)} new session(s); corpus now {total} conversations")
PY
