#!/usr/bin/env python3
"""Parse a Zimfo device debug log into structured conversation records.

The app writes crash-surviving logs to Documents/debug-logs/*.log; each line is
`HH:MM:SS.mmm [Category] message`. This turns one raw log into JSON conversation
records (one per session, split on `conversation reset`) with per-turn user text,
routing, retrieved passages, grounding, the assistant answer, the [Perf] row, any
disambiguation offer, and the TTS backend/deferral — everything the Claude judge
needs to decide whether a turn went off the rails and whether the cause was
retrieval or the model.

Usage:
  parse_log.py <raw.log> [<raw.log> ...]        # prints JSONL, one conv per line
  parse_log.py --dir eval/corpus/raw            # parse every *.log in a dir
"""
import json
import re
import sys
from pathlib import Path

BUILD_RE = re.compile(r"=== Zimfo (.+?) ===")
UNCLEAN_RE = re.compile(r"PREVIOUS SESSION ENDED UNCLEANLY")
LINE_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d+)\s+\[([^\]]+)\]\s+(.*?)(?:\s+·\s+mem=[\d.]+ MB)?$")
USER_RE = re.compile(r"^(.*)$")
COMPLEXITY_RE = re.compile(r"query complexity:\s+(\S+)")
FASTPATH_RE = re.compile(r"fast-path intent:\s+(\S+)")
TOOLLOOP_RE = re.compile(r"runGenerationLoop: entered")
PASSAGES_RE = re.compile(r"discuss .*?: passages = (.+)$")
GROUNDING_RE = re.compile(r"grounding sources:\s+(.+)$")
DISAMBIG_RE = re.compile(r"disambiguation offered:\s+(.+)$")
TTS_BACKEND_RE = re.compile(r"tts=(.+?)(?:\s+\(deferred\))?$")
TTS_DEFER_RE = re.compile(r"TTS eager synthesis deferred")
PERF_RE = re.compile(
    r"(?:grounded|iter \d+) · runtime=(\S+) model=(\S+) prompt=(\d+)tok "
    r"reused=(\d+) prefill=([\d.]+)s ttft=([\d.-]+)s out=(\d+)tok "
    r"decode=([\d.]+)tok/s total=([\d.]+)s footprint=(\d+)MB stop=(\S+)")


def parse(path: Path) -> list[dict]:
    build = None
    unclean = False
    tts_backend = None
    tts_deferred = False
    convs: list[dict] = []
    turns: list[dict] = []
    cur: dict | None = None

    def flush_turn():
        nonlocal cur
        if cur and (cur.get("user") or cur.get("assistant")):
            turns.append(cur)
        cur = None

    def flush_conv():
        nonlocal turns
        flush_turn()
        if turns:
            convs.append({
                "session": path.stem,
                "build": build,
                "ended_uncleanly": unclean,
                "turns": turns,
            })
        turns = []

    for raw in path.read_text(errors="replace").splitlines():
        if (m := BUILD_RE.search(raw)):
            build = m.group(1)
            continue
        if UNCLEAN_RE.search(raw):
            unclean = True
            continue
        m = LINE_RE.match(raw)
        if not m:
            continue
        ts, cat, msg = m.group(1), m.group(2), m.group(3).strip()

        if cat == "User":
            flush_turn()
            cur = {"t": ts, "user": msg, "assistant": None, "route": None,
                   "complexity": None, "passages": [], "grounding": [],
                   "disambiguation": [], "perf": None,
                   "tts_backend": tts_backend, "tts_deferred": False}
            continue
        if cur is None:
            # Pre-first-turn lines (load, voice backend) — capture the backend.
            if cat == "Voice" and (mm := TTS_BACKEND_RE.search(msg)):
                tts_backend = mm.group(1).strip()
            continue

        if cat == "Router":
            if (mm := COMPLEXITY_RE.search(msg)): cur["complexity"] = mm.group(1)
            elif (mm := FASTPATH_RE.search(msg)): cur["route"] = f"fast-path:{mm.group(1)}"
            elif (mm := DISAMBIG_RE.search(msg)):
                cur["disambiguation"] = [s.strip() for s in mm.group(1).split("|")]
        elif cat == "Chat":
            if TOOLLOOP_RE.search(msg) and not cur["route"]:
                cur["route"] = "llm-tool-loop"
            elif (mm := PASSAGES_RE.search(msg)):
                cur["passages"] = [s.strip() for s in mm.group(1).split("|")]
            elif (mm := GROUNDING_RE.search(msg)):
                cur["grounding"] = [s.strip() for s in mm.group(1).split("|")]
            elif "conversation reset" in msg:
                flush_conv()
        elif cat == "Assistant":
            cur["assistant"] = msg
        elif cat == "Perf":
            if (mm := PERF_RE.search(msg)):
                (rt, model, prompt, reused, prefill, ttft, out, dec,
                 total, foot, stop) = mm.groups()
                cur["perf"] = {
                    "runtime": rt, "model": model, "prompt_tok": int(prompt),
                    "reused_tok": int(reused), "prefill_s": float(prefill),
                    "ttft_s": float(ttft), "out_tok": int(out),
                    "decode_tps": float(dec), "total_s": float(total),
                    "footprint_mb": int(foot), "stop": stop}
        elif cat == "Voice":
            if (mm := TTS_BACKEND_RE.search(msg)):
                tts_backend = mm.group(1).strip()
                cur["tts_backend"] = tts_backend
            elif TTS_DEFER_RE.search(msg):
                cur["tts_deferred"] = True

    flush_conv()
    return convs


def main(argv: list[str]) -> int:
    paths: list[Path] = []
    if argv and argv[0] == "--dir":
        paths = sorted(Path(argv[1]).glob("*.log"))
    else:
        paths = [Path(a) for a in argv]
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    for p in paths:
        for conv in parse(p):
            print(json.dumps(conv, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
