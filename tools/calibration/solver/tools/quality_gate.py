"""Measure one quantization arm on held-out app text and record it against the bar.

Why this exists
---------------
The app suite cannot rank arms: of its 66 frozen turns, 49 invoke no language model in
either arm, and the only paired difference it has ever produced is 4 wins against 1 loss
(exact McNemar p = 0.375). So "is our Q1 better than Bonsai" needs a continuous metric, and
upstream's own is perplexity - WikiText2 there, our held-out invocations here.

`zimfo-q1-v1` uses the same `(128, 18)` block layout as llama.cpp's `Q1_0`, so one binary
measures Bonsai, our exports and any ternary arm on identical bytes. This wraps that into a
ledger, so a number is never quoted from memory or from a different text.

Usage (from tools/calibration):
    # measure an arm and append it to the ledger
    PYTHONPATH=. .venv/bin/python solver/tools/quality_gate.py measure \
        --gguf runs/q1-layout-probe-20260919/export/model.gguf --label rtn-all-q1

    # add the incumbent once, then see everything ranked
    PYTHONPATH=. .venv/bin/python solver/tools/quality_gate.py measure \
        --gguf ~/Library/Caches/huggingface/hub/models--prism-ml--Bonsai-27B-gguf/snapshots/main/Bonsai-27B-Q1_0.gguf \
        --label bonsai-27b-q1_0 --bar
    PYTHONPATH=. .venv/bin/python solver/tools/quality_gate.py compare

The held-out text is built from invocations excluded from training (every 11th of 87); see
`--text`, which defaults to the file this tool expects to exist.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEXT = Path("/tmp/gsq-heldout.txt")
DEFAULT_LEDGER = REPO_ROOT / "tools" / "calibration" / "runs" / "quality-arms.json"
PERPLEXITY = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
BUILD_LINE = re.compile(r"^build:\s*(.+)$", re.MULTILINE)


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def measure(args) -> dict:
    gguf, text = Path(args.gguf).expanduser(), Path(args.text).expanduser()
    if not gguf.is_file():
        raise SystemExit(f"No such GGUF: {gguf}")
    if not text.is_file():
        raise SystemExit(f"No such held-out text: {text}")
    binary = Path(args.binary or (os.environ.get("LLAMA_PERPLEXITY") or "llama-perplexity"))
    command = [str(binary), "-m", str(gguf), "-f", str(text), "-c", str(args.context),
               "-ngl", str(args.ngl)]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - started
    output = completed.stdout + completed.stderr
    match = PERPLEXITY.search(output)
    if completed.returncode != 0 or match is None:
        tail = "\n".join(output.strip().splitlines()[-12:])
        raise SystemExit(f"Perplexity failed (exit {completed.returncode}):\n{tail}")
    build = BUILD_LINE.search(output)
    arm = {
        "label": args.label,
        "status": "completed",
        "perplexity": float(match.group(1)),
        "perplexity_stderr": float(match.group(2)),
        "gguf": str(gguf.resolve()),
        "gguf_bytes": gguf.stat().st_size,
        "gguf_sha256": sha256_of(gguf),
        "text": str(text.resolve()),
        "text_sha256": sha256_of(text),
        "context": args.context,
        "ngl": args.ngl,
        "command": command,
        "elapsed_seconds": round(elapsed, 1),
        "tool": build.group(1).strip() if build else binary.name,
        "is_bar": bool(args.bar),
        "notes": args.notes or "",
        "measured_unix": int(time.time()),
    }
    return arm


def load_ledger(path: Path) -> list:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"Ledger is not valid JSON: {path}: {error}")
    if not isinstance(value, list):
        raise SystemExit(f"Ledger must be a list of arms: {path}")
    return value


def write_ledger(path: Path, arms: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(arms, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def record(args) -> int:
    ledger = load_ledger(args.ledger)
    arm = measure(args)
    for existing in ledger:
        if existing.get("label") == arm["label"]:
            raise SystemExit(
                f"Label {arm['label']!r} is already in the ledger with PPL "
                f"{existing.get('perplexity')}. Use a new label rather than overwriting a "
                f"measurement.")
    ledger.append(arm)
    if arm["is_bar"]:
        for existing in ledger[:-1]:
            existing["is_bar"] = False
    write_ledger(args.ledger, ledger)
    print(f"{arm['label']:<24} PPL {arm['perplexity']:.4f} +/- {arm['perplexity_stderr']:.4f}"
          f"  ({arm['gguf_bytes']/1e9:.2f} GB, {arm['elapsed_seconds']}s)")
    return 0


def compare(args) -> int:
    ledger = load_ledger(args.ledger)
    if not ledger:
        raise SystemExit(f"No arms recorded yet in {args.ledger}")
    bars = [arm for arm in ledger if arm.get("is_bar")]
    bar = bars[-1] if bars else None
    texts = {arm["text_sha256"] for arm in ledger}
    if len(texts) > 1:
        print("WARNING: arms were measured on different text; PPL is not comparable.\n")
    print(f"{'arm':<24} {'PPL':>9} {'+/-':>7} {'GB':>6} {'vs bar':>9}  note")
    for arm in sorted(ledger, key=lambda a: a["perplexity"]):
        delta = "" if bar is None or arm is bar else f"{arm['perplexity'] - bar['perplexity']:+.4f}"
        mark = "  <- BAR" if arm is bar else ""
        print(f"{arm['label']:<24} {arm['perplexity']:9.4f} {arm['perplexity_stderr']:7.4f} "
              f"{arm['gguf_bytes']/1e9:6.2f} {delta:>9}  {arm['notes']}{mark}")
    if bar is None:
        print("\nNo arm is marked as the bar; re-measure Bonsai with --bar.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    record_parser = sub.add_parser("measure", help="measure an arm and append it to the ledger")
    record_parser.add_argument("--gguf", required=True)
    record_parser.add_argument("--label", required=True)
    record_parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    record_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    record_parser.add_argument("--binary", default=None,
                               help="defaults to $LLAMA_PERPLEXITY or llama-perplexity on PATH")
    record_parser.add_argument("--context", type=int, default=4096)
    record_parser.add_argument("--ngl", type=int, default=99)
    record_parser.add_argument("--bar", action="store_true",
                               help="mark this arm as the one others are compared against")
    record_parser.add_argument("--notes", default="")
    record_parser.set_defaults(handler=record)

    compare_parser = sub.add_parser("compare", help="rank every recorded arm against the bar")
    compare_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    compare_parser.set_defaults(handler=compare)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
