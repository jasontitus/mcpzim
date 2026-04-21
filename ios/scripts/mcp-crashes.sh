#!/bin/bash
# mcp-crashes.sh — pull, triage, and summarise MCPZim crash reports.
#
# Replaces the run-idevicecrashreport-then-eyeball-each-ips loop we used
# to do by hand after every reported crash. Specifically tailored for
# the iOS jetsam / voice-chat / Gemma memory-pressure failure modes
# mcpzim keeps tripping.
#
# usage:
#   mcp-crashes.sh               # full scan: pull latest, summary of today's MCPZim + jetsam events
#   mcp-crashes.sh pull          # pull fresh .ips from the device (MCPZim + Jetsam + rest)
#   mcp-crashes.sh mcpzim [N]    # list last N MCPZim app .ips files, newest first
#   mcp-crashes.sh jetsam [N]    # list last N JetsamEvent .ips files, newest first
#   mcp-crashes.sh summary <file>  # parse one .ips: proc, reason, faulting stack, killed procs
#   mcp-crashes.sh mem           # memory trajectory from the syslog buffer (n/min/max/avg + peaks)
#   mcp-crashes.sh peaks [N]     # top N peak mem readings (default 20)
#   mcp-crashes.sh today         # just what's dated today in both categories
#
# Assumes `idevice_id -l` returns a single attached device. If
# `mcp-logs.sh` is running, the `mem` / `peaks` commands share its
# `/tmp/mcp-syslog.log` buffer. The syslog streamer is separate —
# this script does NOT start or stop it.

set -euo pipefail

DEVICE_UUID="${MCPZIM_DEVICE_UUID:-}"
CRASH_DIR_MCPZIM=/tmp/mcpzim-crash
CRASH_DIR_JETSAM=/tmp/mcpzim-jetsam
CRASH_DIR_ALL=/tmp/mcpzim-crash-all
SYSLOG=/tmp/mcp-syslog.log

resolve_device() {
  if [ -n "$DEVICE_UUID" ]; then
    echo "$DEVICE_UUID"
    return
  fi
  local id
  id=$(idevice_id -l 2>/dev/null | head -1 || true)
  if [ -z "$id" ]; then
    echo "no attached device (idevice_id -l empty). Set MCPZIM_DEVICE_UUID=... to force." >&2
    exit 1
  fi
  echo "$id"
}

pull_cmd() {
  local uuid
  uuid=$(resolve_device)
  mkdir -p "$CRASH_DIR_MCPZIM" "$CRASH_DIR_JETSAM" "$CRASH_DIR_ALL"
  echo "pulling MCPZim-filtered crashes → $CRASH_DIR_MCPZIM"
  idevicecrashreport -u "$uuid" -f MCPZim "$CRASH_DIR_MCPZIM" 2>&1 | tail -3
  echo "pulling Jetsam events → $CRASH_DIR_JETSAM"
  idevicecrashreport -u "$uuid" -f Jetsam "$CRASH_DIR_JETSAM" 2>&1 | tail -3
  # Optional broader sweep (Siri, WiFi, etc) — usually noise, skipped unless asked.
  if [ "${1:-}" = "--all" ]; then
    echo "pulling every .ips (no filter) → $CRASH_DIR_ALL"
    idevicecrashreport -u "$uuid" "$CRASH_DIR_ALL" 2>&1 | tail -3
  fi
}

list_cmd() {
  local dir="$1" n="${2:-10}" label="$3"
  echo "=== $label (newest $n in $dir) ==="
  if ! compgen -G "$dir/*.ips" >/dev/null 2>&1; then
    echo "  (none)"
    return
  fi
  ls -lt "$dir"/*.ips 2>/dev/null | head -"$n" | awk '{print $6, $7, $8, $NF}'
}

summary_cmd() {
  local file="$1"
  if [ ! -f "$file" ]; then
    echo "file not found: $file" >&2
    exit 2
  fi
  echo "=== $file ==="
  tail -n +2 "$file" | python3 -c "
import json, sys, os
try:
    d = json.load(sys.stdin)
except Exception as e:
    print(f'  parse error: {e}')
    sys.exit(0)

# Minimal first-header line from the file's first line (app.proc-pid)
fname = os.path.basename('$file')
print(f'  file: {fname}')
cap = d.get('captureTime') or d.get('date') or d.get('incident_timestamp')
if cap: print(f'  captureTime: {cap}')

exc = d.get('exception', {}) or {}
if exc:
    print(f'  exception: type={exc.get(\"type\")} signal={exc.get(\"signal\")} subtype={exc.get(\"subtype\")}')
if d.get('termination'):
    t = d['termination']
    print(f'  termination: {t.get(\"indicator\")} by={t.get(\"byProc\")} code={t.get(\"code\")}')

lp = d.get('largestProcess')
if isinstance(lp, str):
    print(f'  largestProcess: {lp}')
elif isinstance(lp, dict):
    rss = (lp.get(\"rpages\", 0) or 0) * 16 / 1024
    print(f'  largestProcess: {lp.get(\"procname\") or lp.get(\"name\")} rss={rss:.0f} MB')

# Faulting stack (in-process crash)
ft = d.get('faultingThread')
threads = d.get('threads')
images = d.get('usedImages')
if ft is not None and threads and images:
    try:
        thr = threads[ft]
        frames = thr.get('frames', [])[:8]
        print('  faulting stack (top 8):')
        for fr in frames:
            im = images[fr.get('imageIndex', 0)]
            sym = fr.get('symbol', '')
            off = fr.get('imageOffset', 0)
            print(f'    {im.get(\"name\"):30s} +{off:7d} {sym}')
    except Exception:
        pass

# Jetsam-specific: our process
procs = d.get('processes', []) or []
mcp = [p for p in procs if isinstance(p, dict) and 'MCP' in (p.get('procname') or p.get('name') or '')]
for p in mcp:
    rss = (p.get('rpages', 0) or 0) * 16 / 1024
    peak = (p.get('lifetimeMax', 0) or 0) * 16 / 1024
    freeze = p.get('freeze_skip_reason:') or p.get('freeze_skip_reason')
    states = p.get('states')
    print(f'  MCPZim proc: rss={rss:.0f} MB peak={peak:.0f} MB states={states} freeze={freeze}')

# Any explicitly killed procs (jetsam)
killed = [p for p in procs if isinstance(p, dict) and p.get('killed')]
if killed:
    print('  killed procs:')
    for p in killed[:5]:
        rss = (p.get('rpages', 0) or 0) * 16 / 1024
        print(f'    {p.get(\"procname\",\"?\"):30s} rss={rss:.0f} MB reason={p.get(\"reason\",\"?\")}')
"
}

mem_cmd() {
  if [ ! -f "$SYSLOG" ]; then
    echo "no syslog buffer at $SYSLOG. Start mcp-logs.sh to capture memory data." >&2
    exit 2
  fi
  local count
  count=$(grep -oE "mem=[0-9.]+ MB" "$SYSLOG" | wc -l | tr -d ' ')
  if [ "$count" -eq 0 ]; then
    echo "buffer at $SYSLOG has no mem=... readings yet." >&2
    exit 2
  fi
  echo "=== memory trajectory in $SYSLOG ==="
  grep -oE "mem=[0-9.]+ MB" "$SYSLOG" | awk -F'[= ]' '
    BEGIN{min=1e9; max=0}
    {v=$2+0; s+=v; n++; if(v<min) min=v; if(v>max) max=v}
    END{printf "  samples=%d  min=%.0f MB  max=%.0f MB  avg=%.0f MB  range=%.0f MB\n", n, min, max, s/n, max-min}'
  echo "=== last 15 readings ==="
  grep -oE "^[A-Za-z]+ [0-9]+ [0-9:.]+ .*mem=[0-9.]+ MB" "$SYSLOG" 2>/dev/null | tail -15 | awk -F'mem=' '{
    # Trim everything after the mem= reading.
    ts=$1; sub(/[ ].*$/, "", ts); mb=$2; sub(/ MB.*/, "", mb); printf "  %s  %s MB\n", ts, mb
  }' || tail -15 "$SYSLOG"
}

peaks_cmd() {
  local n="${1:-20}"
  if [ ! -f "$SYSLOG" ]; then
    echo "no syslog buffer at $SYSLOG." >&2
    exit 2
  fi
  echo "=== top $n peak mem readings ==="
  grep -nE "mem=[0-9.]+ MB" "$SYSLOG" \
    | awk -F'mem=' '{
        line=$1; rest=$2;
        mb=rest; sub(/ MB.*/, "", mb);
        printf "%8.1f\t%s mem=%s MB\n", mb+0, line, mb
      }' \
    | sort -rn | head -"$n" | cut -f2-
}

today_cmd() {
  local today
  today=$(date +%Y-%m-%d)
  echo "=== today ($today) ==="
  echo "--- MCPZim ---"
  (ls -lt "$CRASH_DIR_MCPZIM"/*"$today"*.ips 2>/dev/null || echo "  (none)") | head -10
  echo "--- Jetsam ---"
  (ls -lt "$CRASH_DIR_JETSAM"/*"$today"*.ips 2>/dev/null || echo "  (none)") | head -10
}

full_scan() {
  pull_cmd
  echo
  list_cmd "$CRASH_DIR_MCPZIM" 5 "MCPZim app crashes"
  echo
  list_cmd "$CRASH_DIR_JETSAM" 5 "Jetsam events"
  echo
  # Summarise the newest of each, if any.
  local newest
  for dir in "$CRASH_DIR_MCPZIM" "$CRASH_DIR_JETSAM"; do
    newest=$(ls -t "$dir"/*.ips 2>/dev/null | head -1 || true)
    if [ -n "$newest" ]; then
      summary_cmd "$newest"
      echo
    fi
  done
  # Only run the mem report if the syslog buffer has samples.
  if [ -f "$SYSLOG" ] && grep -qE "mem=[0-9.]+ MB" "$SYSLOG" 2>/dev/null; then
    mem_cmd
  fi
}

case "${1:-scan}" in
  pull)    shift; pull_cmd "$@" ;;
  mcpzim)  list_cmd "$CRASH_DIR_MCPZIM" "${2:-10}" "MCPZim app crashes" ;;
  jetsam)  list_cmd "$CRASH_DIR_JETSAM" "${2:-10}" "Jetsam events" ;;
  summary) shift; summary_cmd "$@" ;;
  mem)     mem_cmd ;;
  peaks)   shift; peaks_cmd "$@" ;;
  today)   today_cmd ;;
  scan|"") full_scan ;;
  *)       sed -n '2,/^$/p' "$0"; exit 2 ;;
esac
