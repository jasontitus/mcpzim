#!/bin/bash
# mcp-logs.sh — quick access to the running device syslog.
#
# The helper assumes a background idevicesyslog is writing to
# /tmp/mcp-syslog.log via `mcp-logs.sh start`. Once running it stays
# up until you call `stop` or reboot the Mac.
#
# usage:
#   mcp-logs.sh start            start the persistent streamer
#   mcp-logs.sh stop             stop the streamer
#   mcp-logs.sh status           is the streamer running?
#   mcp-logs.sh tail             last 60 app lines
#   mcp-logs.sh recent [N]       last N app lines (default 200)
#   mcp-logs.sh since '11:35'    app lines at/after that timestamp
#   mcp-logs.sh raw [N]          last N RAW (unfiltered) syslog lines

LOGFILE=/tmp/mcp-syslog.log
PIDFILE=/tmp/mcp-syslog.pid
DEVICE=00008150-000669303687801C

# Only keep lines coming from our app's debug dylib (where `print()`
# + Logger() land) or the Gemma4 / Kokoro Swift logs. Drop system
# CoreFoundation / Network / backboardd noise.
APP_FILTER='MCPZimChat\(MCPZimChat.debug.dylib\)|MCPZimChat\(KokoroSwift\)|MCPZimChat\(Gemma4SwiftCore\)|MCPZimChat\(MLX'

case "${1:-tail}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"
      exit 0
    fi
    : > "$LOGFILE"
    nohup idevicesyslog -u "$DEVICE" -p MCPZimChat > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "started (pid $(cat "$PIDFILE")) → $LOGFILE"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null
      rm -f "$PIDFILE"
      echo stopped
    else
      echo "not running"
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      bytes=$(wc -c < "$LOGFILE" 2>/dev/null || echo 0)
      echo "running (pid $(cat "$PIDFILE")), log=$LOGFILE ($bytes bytes)"
    else
      echo "not running"
    fi
    ;;
  tail)
    grep -E "$APP_FILTER" "$LOGFILE" 2>/dev/null | tail -60
    ;;
  recent)
    N="${2:-200}"
    grep -E "$APP_FILTER" "$LOGFILE" 2>/dev/null | tail -"$N"
    ;;
  since)
    [ -z "$2" ] && { echo "usage: $0 since '11:35'"; exit 1; }
    grep -E "$APP_FILTER" "$LOGFILE" 2>/dev/null | awk -v t="$2" '$3 >= t'
    ;;
  raw)
    N="${2:-200}"
    tail -"$N" "$LOGFILE" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {start|stop|status|tail|recent [N]|since 'HH:MM'|raw [N]}"
    exit 1
    ;;
esac
