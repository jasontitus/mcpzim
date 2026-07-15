#!/usr/bin/env bash
#
# Deploy + VERIFY loop for MCPZimChat — works over WIFI (devicectl only,
# no usbmuxd). Born 2026-07-02 after a build was declared "great" while
# it was crash-looping on launch: install/launch alone proves nothing.
#
#   ios/scripts/mcp-deploy-verify.sh                 # install+launch+watch
#   ios/scripts/mcp-deploy-verify.sh watch           # just watch the running app
#   ios/scripts/mcp-deploy-verify.sh crashes         # list device crash reports
#   ios/scripts/mcp-deploy-verify.sh pull <name.ips> # copy one report to /tmp
#
# Exit codes: 0 = app alive after WATCH_SECS; 1 = app died (crash reports,
# if any, are pulled automatically); 2 = install/launch failed.

set -uo pipefail

DEVICE="${MCPZIM_DEVICE_UUID:-5AE213CA-315A-532A-878B-2CC4EB051ABD}"
BUNDLE="org.mcpzim.MCPZimChat"
APP="${MCPZIM_APP_PATH:-$(dirname "$0")/../build/Build/Products/Debug-iphoneos/MCPZimChat.app}"
WATCH_SECS="${MCPZIM_WATCH_SECS:-45}"
CRASH_DIR=/tmp/mcpzim-crash-wifi

list_crashes() {
  xcrun devicectl device info files --device "$DEVICE" \
    --domain-type systemCrashLogs 2>/dev/null \
    | grep -aiE "mcpzim|jetsam" | grep -av "Retired/" | awk '{print $1}'
}

pull_crash() {
  mkdir -p "$CRASH_DIR"
  xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type systemCrashLogs \
    --source "$1" --destination "$CRASH_DIR/$(basename "$1")" >/dev/null 2>&1 \
    && echo "pulled → $CRASH_DIR/$(basename "$1")"
}

alive() {
  xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
    | grep -c "$BUNDLE\|MCPZimChat.app" || true
}

case "${1:-deploy}" in
  crashes)
    list_crashes
    exit 0 ;;
  pull)
    pull_crash "$2"; exit 0 ;;
  watch) ;;  # fall through to the watch loop below
  deploy)
    echo "== install =="
    for i in $(seq 1 20); do
      out=$(xcrun devicectl device install app --device "$DEVICE" "$APP" 2>&1)
      echo "$out" | grep -q "Complete\|installed" && { echo "install ok"; break; }
      echo "$out" | grep -q "locked" || { echo "$out" | tail -3; exit 2; }
      echo "  device locked — retrying in 15s ($i/20)"; sleep 15
    done
    PID=$(xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
          | grep MCPZimChat | awk '{print $1}' | head -1)
    [ -n "${PID:-}" ] && xcrun devicectl device process terminate \
        --device "$DEVICE" --pid "$PID" >/dev/null 2>&1
    echo "== launch =="
    for i in $(seq 1 12); do
      lout=$(xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" 2>&1)
      echo "$lout" | grep -q "Launched" && { echo "launch ok"; break; }
      echo "$lout" | grep -q "Locked" || { echo "$lout" | tail -2; exit 2; }
      echo "  device locked — retrying in 10s ($i/12)"; sleep 10
    done ;;
  *) echo "usage: $0 [deploy|watch|crashes|pull <name.ips>]"; exit 2 ;;
esac

echo "== watch (${WATCH_SECS}s) =="
BEFORE=$(list_crashes)
DEAD=0
for t in $(seq 5 5 "$WATCH_SECS"); do
  sleep 5
  if [ "$(alive)" -eq 0 ]; then
    echo "✗ app NOT RUNNING at t=${t}s"
    DEAD=1
    break
  fi
  echo "  t=${t}s alive"
done

# Crash reports can lag the kill by ~30s — give them a beat when dead.
[ "$DEAD" -eq 1 ] && sleep 30
AFTER=$(list_crashes)
NEW=$(comm -13 <(echo "$BEFORE" | sort) <(echo "$AFTER" | sort))
if [ -n "$NEW" ]; then
  echo "== NEW crash reports =="
  echo "$NEW"
  while IFS= read -r f; do [ -n "$f" ] && pull_crash "$f"; done <<< "$NEW"
  for f in "$CRASH_DIR"/*.ips; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") ---"
    grep -aoE '"(exception|terminationReason|bug_type|procName|largestProcess)"[^,]*' "$f" | head -8
  done
fi

if [ "$DEAD" -eq 1 ]; then
  echo "RESULT: FAILED — app died within ${WATCH_SECS}s"
  exit 1
fi
echo "RESULT: OK — app alive after ${WATCH_SECS}s"
