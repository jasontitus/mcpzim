#!/usr/bin/env python3
"""Emit corpus conversations not yet judged, for the Claude judge pass.

Writes eval/corpus/_to_judge.jsonl = every conversation in conversations.jsonl
whose session id has no verdict in verdicts.jsonl. See judge.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "eval/corpus/conversations.jsonl"
VERDICTS = ROOT / "eval/corpus/verdicts.jsonl"
OUT = ROOT / "eval/corpus/_to_judge.jsonl"

judged = set()
if VERDICTS.exists():
    for line in VERDICTS.open():
        try:
            judged.add(json.loads(line)["session"])
        except Exception:
            pass

n = 0
with OUT.open("w") as out:
    if CORPUS.exists():
        for line in CORPUS.open():
            try:
                conv = json.loads(line)
            except Exception:
                continue
            if conv.get("session") not in judged:
                out.write(line)
                n += 1
print(f"prep_judge: {n} un-judged conversation(s) → {OUT}")
