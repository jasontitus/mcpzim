#!/usr/bin/env bash
# mcp-report.sh — reassemble a debug report that the iOS app's
# "Send Debug Report" button emitted into the idevicesyslog stream.
#
# Pairs with `ChatSession.emitDebugReport()` on device — see
# ios/MCPZimChat/Chat/DebugReport.swift for the wire format.
#
# Usage:
#   mcp-report.sh latest       # reassemble the newest complete report
#   mcp-report.sh list         # print every (hash, time, chunk count) tuple seen
#   mcp-report.sh pull HASH    # reassemble the report with a specific hash
#
# Requires ios/scripts/mcp-logs.sh to be tailing (it populates
# /tmp/mcp-syslog.log, which is the buffer we scan). If your Mac
# hasn't been streaming the phone's logs the report line never
# arrived and no script can recover it — restart mcp-logs.sh and
# ask the user to tap Report again.

set -euo pipefail

BUF=/tmp/mcp-syslog.log
OUT=/tmp/mcpzim-debug-reports
mkdir -p "$OUT"

usage() {
  sed -n '2,18p' "$0"
  exit 1
}

scan_reports() {
  # Prints a line per report hash: BEGIN count, END count, total expected.
  awk '
    /\[DebugReport BEGIN hash=/ {
      h = gensub(/.*hash=([A-Fa-f0-9]+).*/, "\\1", 1, $0)
      t = gensub(/.*total=([0-9]+).*/, "\\1", 1, $0)
      sz = gensub(/.*size=([0-9]+).*/, "\\1", 1, $0)
      begin[h] = 1; total[h] = t + 0; size[h] = sz + 0
      first_seen[h] = $0
    }
    /\[DebugReport seq=/ {
      h = gensub(/.*hash=([A-Fa-f0-9]+).*/, "\\1", 1, $0)
      seq[h]++
    }
    /\[DebugReport END hash=/ {
      h = gensub(/.*hash=([A-Fa-f0-9]+).*/, "\\1", 1, $0)
      done[h] = 1
    }
    END {
      for (h in begin) {
        printf "%s\tchunks=%d/%d\tbytes=%d\tend=%s\n",
          h, seq[h]+0, total[h], size[h], (done[h] ? "yes" : "NO")
      }
    }
  ' "$BUF"
}

# Reassemble one hash's base64 payload from the buffer and emit JSON.
assemble() {
  local want="$1"
  python3 - "$BUF" "$want" <<'PY'
import re, sys, base64, os
buf_path, want = sys.argv[1], sys.argv[2]
chunks = {}
total = None
with open(buf_path, "r", errors="ignore") as f:
    for line in f:
        # Example line:
        #   Apr 22 10:42:01 MCPZimChat ...: [DebugReport seq=3/42 hash=AB12] BASE64…
        m = re.search(
            r"\[DebugReport seq=(\d+)/(\d+) hash=([A-Fa-f0-9]+)\]\s+(\S+)",
            line)
        if not m:
            continue
        idx, tot, h, body = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        if h != want:
            continue
        total = tot
        chunks[idx] = body
if not chunks:
    sys.stderr.write(f"no chunks for hash {want}\n")
    sys.exit(2)
missing = [i for i in range(1, (total or max(chunks))+1) if i not in chunks]
if missing:
    sys.stderr.write(
        f"warning: missing {len(missing)} of {total} chunks: "
        + f"{missing[:8]}{' …' if len(missing) > 8 else ''}\n")
ordered = [chunks[i] for i in sorted(chunks)]
raw = base64.b64decode("".join(ordered))
sys.stdout.buffer.write(raw)
PY
}

cmd="${1:-latest}"

case "$cmd" in
  list)
    scan_reports | sort -t$'\t' -k4,4
    ;;
  latest)
    last=$(scan_reports | awk '$4=="end=yes"{print}' | tail -n 1 | awk '{print $1}')
    if [ -z "$last" ]; then
      echo "no complete debug reports found in $BUF" >&2
      echo "hint: is ios/scripts/mcp-logs.sh running?" >&2
      exit 1
    fi
    out_json="$OUT/$last.json"
    assemble "$last" > "$out_json"
    echo "wrote: $out_json"
    echo
    # Quick preview — device info + messages count + last few debug entries.
    python3 - "$out_json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
print(f"generatedAt:     {j.get('generatedAt')}")
print(f"appBuild:        {j.get('appBuild')}  deviceTier: {j.get('deviceTier')}")
print(f"selectedModelId: {j.get('selectedModelId')}")
print(f"messages:        {len(j.get('messages', []))}")
print(f"debugEntries:    {len(j.get('debugEntries', []))}")
print()
print("— last 5 debug entries —")
for e in j.get('debugEntries', [])[-5:]:
    print(f"  [{e.get('category','?')}] {e.get('message','')[:120]}")
print()
print("— messages —")
for m in j.get('messages', []):
    txt = (m.get('text','') or '').replace('\n',' ')[:120]
    tools = ",".join(t.get('name','?') for t in m.get('toolCalls', []))
    print(f"  [{m.get('role','?')}] {txt}  {('['+tools+']') if tools else ''}")
PY
    ;;
  pull)
    [ $# -ge 2 ] || usage
    want="$2"
    out_json="$OUT/$want.json"
    assemble "$want" > "$out_json"
    echo "wrote: $out_json"
    ;;
  *)
    usage
    ;;
esac
