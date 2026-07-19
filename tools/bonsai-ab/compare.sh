#!/usr/bin/env bash
#
# Bonsai 27B cross-runtime A/B: run the SAME conversation through the
# llama.cpp GGUF runtime and the MLX runtime (same 1-bit weights, same
# ChatML template, same sampling profile), then merge the uniform
# [Perf] rows into one side-by-side markdown table.
#
#   tools/bonsai-ab/compare.sh --zim <wiki.zim> --gguf <Bonsai-27B-Q1_0.gguf> \
#       [--streetzim <osm.zim>] [--turn "..." ...] [extra probe-discuss args]
#
# Each runtime runs in its OWN process so Metal pool state from one
# can't contaminate the other's memory numbers. Full logs land in
# /tmp/bonsai-ab/<runtime>.log; the merged table prints to stdout and
# /tmp/bonsai-ab/TABLE.md.

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

OUT=/tmp/bonsai-ab
mkdir -p "$OUT"

echo "== llamacpp run =="
"$BIN" --probe-discuss --runtime llamacpp "$@" > "$OUT/llamacpp.log" 2>&1
RC1=$?
echo "   exit=$RC1 · log=$OUT/llamacpp.log"

echo "== mlx run =="
"$BIN" --probe-discuss --runtime mlx "$@" > "$OUT/mlx.log" 2>&1
RC2=$?
echo "   exit=$RC2 · log=$OUT/mlx.log"

python3 - "$OUT/llamacpp.log" "$OUT/mlx.log" <<'EOF' | tee "$OUT/TABLE.md"
import re, sys

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

a, b = rows(sys.argv[1]), rows(sys.argv[2])
print("# Bonsai 27B — llama.cpp vs MLX (same weights, same turns)\n")
print("| turn | runtime | reused/prompt tok | prefill s | TTFT s | out tok | decode tok/s | total s | footprint MB | stop |")
print("|---|---|---|---|---|---|---|---|---|---|")
for rowset in (a, b):
    for turn, g in rowset:
        it, rt, model, prompt, reused, prefill, ttft, out_t, dec, total, foot, stop = g
        label = turn if len(turn) <= 42 else turn[:39] + "…"
        print(f"| {label} | {rt} | {reused}/{prompt} | {prefill} | {ttft} | {out_t} | {dec} | {total} | {foot} | {stop} |")

def agg(rowset, name):
    if not rowset:
        print(f"\n**{name}: no [Perf] rows — check the log**")
        return
    ttfts = [float(g[6]) for _, g in rowset if float(g[6]) >= 0]
    decs = [float(g[8]) for _, g in rowset]
    foots = [int(g[10]) for _, g in rowset]
    print(f"\n**{name}** · turns={len(rowset)} · "
          f"median TTFT={sorted(ttfts)[len(ttfts)//2]:.2f}s · "
          f"median decode={sorted(decs)[len(decs)//2]:.1f} tok/s · "
          f"peak footprint={max(foots)} MB")

agg(a, "llamacpp")
agg(b, "mlx")
EOF

exit $(( RC1 > RC2 ? RC1 : RC2 ))
