#!/usr/bin/env bash
#
# Bonsai 27B operating-point A/B: run the SAME conversation through any
# combination of quant × runtime legs and merge the uniform [Perf] rows
# into one side-by-side markdown table.
#
#   tools/bonsai-ab/compare.sh [--legs q1-gguf,ternary-gguf,ternary-mlx] \
#       --zim <wiki.zim> [--turn "..." ...] [extra probe-discuss args]
#
# Legs (default: all three runnable ones):
#   q1-gguf      Bonsai-27B-Q1_0.gguf        · llama.cpp   (phone class)
#   ternary-gguf Ternary-Bonsai-27B-Q2_0.gguf · llama.cpp  (Mac class)
#   ternary-mlx  prism-ml/Ternary-Bonsai-27B-mlx-2bit · MLX (Mac class)
#   q1-mlx       BLOCKED: stock mlx-c rejects bits=1 — needs
#                PrismML-Eng/mlx-swift branch `prism`.
#
# Each leg runs in its OWN process so Metal pool state from one can't
# contaminate another's numbers. Logs land in /tmp/bonsai-ab/<leg>.log;
# the merged table prints to stdout and /tmp/bonsai-ab/TABLE.md.

set -uo pipefail

BIN="${MCPZIM_EVALCLI:-$(ls -d "$HOME"/Library/Developer/Xcode/DerivedData/MCPZimChat-*/Build/Products/Debug/MCPZimEvalCLI 2>/dev/null | head -1)}"
[ -x "$BIN" ] || { echo "MCPZimEvalCLI not found — build the MCPZimEvalCLI scheme first (or set MCPZIM_EVALCLI)"; exit 2; }

# The CLI links llama.framework via @rpath; a fresh DerivedData build
# doesn't carry the macOS slice. Self-heal (HOW_TO_BUILD gotcha).
DD="$(dirname "$BIN")"
if [ ! -d "$DD/PackageFrameworks/llama.framework" ]; then
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  mkdir -p "$DD/PackageFrameworks"
  cp -R "$REPO_ROOT/ios/LocalPackages/llama.cpp-swift/llama.xcframework/macos-arm64_x86_64/llama.framework" \
    "$DD/PackageFrameworks/" 2>/dev/null && echo "(copied llama.framework beside CLI)"
fi

HF_HUB="$HOME/Library/Caches/huggingface/hub"
Q1_GGUF="$HF_HUB/models--prism-ml--Bonsai-27B-gguf/snapshots/main/Bonsai-27B-Q1_0.gguf"
TERNARY_GGUF="$HF_HUB/models--prism-ml--Ternary-Bonsai-27B-gguf/snapshots/main/Ternary-Bonsai-27B-Q2_0.gguf"

LEGS="q1-gguf,ternary-gguf,ternary-mlx"
if [ "${1:-}" = "--legs" ]; then
  LEGS="$2"; shift 2
fi

OUT=/tmp/bonsai-ab
mkdir -p "$OUT"
RC=0
RAN_LOGS=()
RAN_LABELS=()

for leg in ${LEGS//,/ }; do
  case "$leg" in
    q1-gguf)      extra=(--runtime llamacpp --gguf "$Q1_GGUF") ;;
    ternary-gguf) extra=(--runtime llamacpp --gguf "$TERNARY_GGUF") ;;
    ternary-mlx)  extra=(--runtime mlx --mlx-repo prism-ml/Ternary-Bonsai-27B-mlx-2bit) ;;
    q1-mlx)       echo "== $leg SKIPPED: needs PrismML-Eng/mlx-swift branch prism (stock mlx-c rejects bits=1)"; continue ;;
    *) echo "unknown leg: $leg"; exit 2 ;;
  esac
  echo "== $leg run =="
  "$BIN" --probe-discuss "${extra[@]}" "$@" > "$OUT/$leg.log" 2>&1
  rc=$?
  echo "   exit=$rc · log=$OUT/$leg.log"
  [ $rc -gt $RC ] && RC=$rc
  RAN_LOGS+=("$OUT/$leg.log")
  RAN_LABELS+=("$leg")
done

python3 - "${RAN_LOGS[@]}" <<'EOF' | tee "$OUT/TABLE.md"
import re, sys, os

def rows(path):
    out = []
    perf = re.compile(
        r"\[Perf\] (\S+(?: \d+)?) · runtime=(\S+) model=(\S+) prompt=(\d+)tok "
        r"reused=(\d+) prefill=([\d.]+)s ttft=([\d.-]+)s out=(\d+)tok "
        r"decode=([\d.]+)tok/s total=([\d.]+)s footprint=(\d+)MB stop=(\S+)")
    turn = None
    for line in open(path, errors="replace"):
        m = re.search(r"YOU: (.*)", line)
        if m:
            turn = m.group(1).strip()
        p = perf.search(line)
        if p and turn is not None:
            out.append((turn, p.groups()))
    return out

legs = [(os.path.basename(p)[:-4], rows(p)) for p in sys.argv[1:]]
print("# Bonsai 27B — operating-point comparison (same turns)\n")
print("| turn | leg | reused/prompt tok | prefill s | TTFT s | out tok | decode tok/s | total s | footprint MB | stop |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, rowset in legs:
    for turn, g in rowset:
        _, rt, model, prompt, reused, prefill, ttft, out_t, dec, total, foot, stop = g
        label = turn if len(turn) <= 42 else turn[:39] + "…"
        print(f"| {label} | {name} | {reused}/{prompt} | {prefill} | {ttft} | {out_t} | {dec} | {total} | {foot} | {stop} |")

for name, rowset in legs:
    if not rowset:
        print(f"\n**{name}: no [Perf] rows — check the log**")
        continue
    ttfts = sorted(float(g[6]) for _, g in rowset if float(g[6]) >= 0)
    decs = sorted(float(g[8]) for _, g in rowset)
    foots = [int(g[10]) for _, g in rowset]
    print(f"\n**{name}** · turns={len(rowset)} · "
          f"median TTFT={ttfts[len(ttfts)//2]:.2f}s · "
          f"median decode={decs[len(decs)//2]:.1f} tok/s · "
          f"peak footprint={max(foots)} MB")
EOF

exit $RC
