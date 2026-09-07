#!/usr/bin/env python3
"""Extract performance-only records from Zimfo's opt-in phone latency log.

Does not copy questions, replies, coordinates, or unrelated startup logs.
Usage: summarize-phone-latency.py LOG --runtime prism-b9591 > results.json
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def summarize(log, runtime):
    # An unclean-launch diagnostic can repeat the previous log's last lines.
    # Only the suite belonging to this launch is evidence for this run.
    marker = "latency suite start · "
    if marker not in log:
        raise ValueError("No latency suite start found")
    log = log[log.rindex(marker):]
    cases, failures, curves, captures, restores = [], [], [], [], []
    thermal = "unavailable"
    for line in log.splitlines():
        if "perf start ·" in line:
            match = re.search(r"thermal=(\w+)", line)
            thermal = match[1] if match else "unavailable"
        match = re.search(r"latency case (.+?) · runtime=", line)
        if match:
            fields = dict(re.findall(r"(\w+)=([^ ·]+)", line))
            cases.append({
                "case": match[1], "thermal_start": thermal,
                "prompt_tokens": int(fields["prompt"].removesuffix("tok")),
                "reused_tokens": int(fields["reused"]),
                "prefill_seconds": float(fields["prefill"].removesuffix("s")),
                "ttft_seconds": float(fields["ttft"].removesuffix("s")),
                "output_tokens": int(fields["out"].removesuffix("tok")),
                "decode_tokens_per_second": float(fields["decode"].removesuffix("tok/s")),
                "total_seconds": float(fields["total"].removesuffix("s")),
                "footprint_mb": float(fields["footprint"].removesuffix("MB")),
                "stop": fields["stop"],
            })
        if "latency case " in line and "stopped after" in line:
            failures.append(line.split("latency case ", 1)[1].split(" · mem=")[0])
        match = re.search(r"verify curve · trial=(\d+) · width=(\d+) · seconds=([\d.]+) · thermal=(\w+) · warmup=(\w+)", line)
        if match:
            curves.append(dict(trial=int(match[1]), width=int(match[2]),
                seconds=float(match[3]), thermal=match[4], warmup=match[5] == "yes"))
        if "recovery captured ·" in line:
            captures.append(line.split("recovery captured · ", 1)[1].split(" · mem=")[0])
        if "recovery restored ·" in line:
            restores.append(line.split("recovery restored · ", 1)[1].split(" · mem=")[0])
    medians = {str(width): statistics.median(r["seconds"] for r in curves
        if r["width"] == width and not r["warmup"])
        for width in sorted({r["width"] for r in curves if not r["warmup"]})}
    verdict = re.search(r"latency correctness · ([^\n]+)", log)
    return dict(runtime=runtime, completed="latency suite complete" in log,
        correctness=verdict[1].split(" · mem=")[0] if verdict else None,
        cases=cases, stopped_cases=failures, captures=captures, restores=restores,
        verification_trials=curves, verification_median_seconds=medians)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log.read_text(), args.runtime), indent=2))
