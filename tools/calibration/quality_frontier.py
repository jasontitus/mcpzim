"""Compare scored, paired Zimfo runs. Does not generate answers or grade them.

Protocol hashes must describe frozen tasks, source snapshots, rubric, split,
sampling and context policies. Case keys describe identical logical inputs for
model-only runs, or identical scenarios for closed-loop app conversations.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, low=0, high=math.inf):
    require(type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high, f"Invalid {label}")


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate(run):
    require(run.get("schema_version") == 1 and run.get("status") == "completed",
            "Run must be completed schema version 1")
    require(digest(run.get("protocol_sha256")), "Missing protocol hash")
    require(run.get("mode") in ("model-only", "app-conversation"), "Invalid comparison mode")
    model = run["model"]
    for key in ("id", "revision", "precision", "runtime", "machine"):
        require(isinstance(model.get(key), str) and bool(model[key].strip()), f"Missing model {key}")
    require(digest(model.get("artifact_manifest_sha256")), "Missing artifact manifest hash")
    require(type(model.get("deployment_bytes")) is int and model["deployment_bytes"] > 0,
            "Require measured complete deployment bytes")
    require(isinstance(run.get("cases"), list) and bool(run["cases"]), "Empty cases")
    cases = {}
    for case in run["cases"]:
        for key in ("id", "conversation_id", "category"):
            require(isinstance(case.get(key), str) and bool(case[key]), f"Missing case {key}")
        require(case["id"] not in cases, "Duplicate case id")
        require(digest(case.get("case_key_sha256")), "Missing frozen case key")
        require(case.get("status") in ("scored", "failed"), "Unscored/skipped case")
        number(case.get("score"), "score", high=1)
        require(case["status"] != "failed" or case["score"] == 0, "Failed case must score zero")
        critical = case.get("critical_failures")
        require(isinstance(critical, list) and all(isinstance(x, str) and x for x in critical),
                "Require explicit critical failure list")
        if run["mode"] == "model-only":
            require(case.get("model_invoked") is True, "Deterministic route in model-only results")
        for key in ("ttft_seconds", "decode_tokens_per_second", "peak_memory_bytes"):
            if case.get(key) is not None:
                number(case[key], key)
        cases[case["id"]] = case
    return cases


def score(cases):
    """Each conversation has equal weight; turns within it have equal weight."""
    conversations = {}
    for case in cases:
        conversations.setdefault(case["conversation_id"], []).append(case["score"])
    return statistics.mean(statistics.mean(v) for v in conversations.values())


def compare(reference, bonsai, candidates):
    runs = [reference, bonsai, *candidates]
    indexed = [validate(run) for run in runs]
    require(reference["model"]["id"] == "Qwen/Qwen3.8-27B"
            and reference["model"]["precision"] in ("BF16", "FP16"),
            "Reference must be original full-precision Qwen3.8-27B")
    require(reference["model"].get("original_weights") is True,
            "Reference must attest original weights, not dequantized weights")
    for candidate in candidates:
        require(candidate["model"].get("source_model") == reference["model"]["id"]
                and candidate["model"].get("source_revision") == reference["model"]["revision"],
                "Candidate must derive from the same original Qwen revision")
    for run, cases in zip(runs[1:], indexed[1:]):
        require(run["protocol_sha256"] == reference["protocol_sha256"]
                and run["mode"] == reference["mode"], "Mismatched evaluation protocol")
        require(cases.keys() == indexed[0].keys(), "Missing or extra paired cases")
        for key, baseline in indexed[0].items():
            for field in ("conversation_id", "category", "case_key_sha256"):
                require(cases[key][field] == baseline[field], f"Mismatched {field}: {key}")
    reference_score = score(indexed[0].values())
    bonsai_score = score(indexed[1].values())
    bonsai_bytes = bonsai["model"]["deployment_bytes"]
    rows = []
    for index, (run, cases) in enumerate(zip(runs, indexed)):
        value = score(cases.values())
        categories = {}
        for category in sorted({c["category"] for c in cases.values()}):
            subset = [c for c in cases.values() if c["category"] == category]
            keys = [c["id"] for c in subset]
            cat_score = score(subset)
            categories[category] = {
                "score": cat_score,
                "loss_vs_qwen_pp": 100 * (score(indexed[0][k] for k in keys) - cat_score),
                "gain_vs_bonsai_pp": 100 * (cat_score - score(indexed[1][k] for k in keys)),
            }
        row = {
            "role": "reference" if index == 0 else "bonsai" if index == 1 else "candidate",
            "model": run["model"], "score": value,
            # Bonsai is a distinct comparator, not a compressed Qwen checkpoint.
            "compression_loss_pp": None if index == 1 else 100 * (reference_score - value),
            "gain_vs_bonsai_pp": 100 * (value - bonsai_score),
            "size_change_vs_bonsai_percent": 100 * (run["model"]["deployment_bytes"] / bonsai_bytes - 1),
            "headroom_retained_percent": (100 * (value - bonsai_score) / (reference_score - bonsai_score)
                                          if index >= 2 and reference_score > bonsai_score else None),
            "failed_cases": [k for k, c in cases.items() if c["status"] == "failed"],
            "critical_failures": {k: c["critical_failures"] for k, c in cases.items() if c["critical_failures"]},
            "new_critical_failures_vs_bonsai": {
                k: sorted(set(c["critical_failures"]) - set(indexed[1][k]["critical_failures"]))
                for k, c in cases.items()
                if set(c["critical_failures"]) - set(indexed[1][k]["critical_failures"])},
            "categories": categories,
        }
        for field in ("ttft_seconds", "decode_tokens_per_second", "peak_memory_bytes"):
            values = [c[field] for c in cases.values() if c.get(field) is not None]
            row[field] = {"measured_cases": len(values), "total_cases": len(cases),
                          "median": statistics.median(values) if values else None,
                          "maximum": max(values) if values else None}
        rows.append(row)
    return {"schema_version": 1, "protocol_sha256": reference["protocol_sha256"],
            "mode": reference["mode"], "paired_cases": len(indexed[0]),
            "score_weighting": "equal conversations; equal turns within each conversation",
            "rows": rows, "automatic_promotion": False,
            "limitations": ["Scores supplied by evaluation rubric; not independently graded by this reporter",
                            "No statistical significance claim; review paired conversation-level uncertainty",
                            "Compare performance only on matching machines and runtime conditions",
                            "Artifact hashes and original-weight provenance must be verified by the runner"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--bonsai", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.reference.read_text()), json.loads(args.bonsai.read_text()),
                     [json.loads(p.read_text()) for p in args.candidate])
    # Exclusive creation preserves previous comparisons rather than overwriting.
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
