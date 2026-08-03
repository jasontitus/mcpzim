#!/usr/bin/env python3
"""Summarize the judged corpus: the retrieval-vs-model split and worst offenders.

The split is the number that decides where engineering effort goes (and whether
a fine-tune is worth it). See judge.md for category definitions.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERDICTS = ROOT / "eval/corpus/verdicts.jsonl"

if not VERDICTS.exists():
    raise SystemExit("no verdicts yet — run the judge (see judge.md)")

rows = [json.loads(l) for l in VERDICTS.open() if l.strip()]
total = len(rows)
off = [r for r in rows if r.get("verdict") == "off_rails"]
cats = Counter(r.get("category", "?") for r in off)
sessions = len({r["session"] for r in rows})

print(f"corpus: {total} judged turns across {sessions} sessions")
print(f"off-rails: {len(off)} ({100*len(off)//max(1,total)}%)\n")
print("failure categories (off-rails only):")
for cat, n in cats.most_common():
    bar = "█" * n
    print(f"  {cat:12} {n:3}  {bar}")

fixable_by_ft = cats.get("model", 0)
print(f"\nfine-tune-addressable (category=model): {fixable_by_ft}"
      f" of {len(off)} failures ({100*fixable_by_ft//max(1,len(off))}%)")

print("\nworst offenders (high severity):")
for r in sorted(off, key=lambda r: r.get("severity") != "high")[:12]:
    if r.get("severity") == "high":
        print(f"  [{r['category']}] {r['session']} #{r['turn_index']}: "
              f"{r['user'][:44]!r} — {r.get('reason','')[:70]}")
