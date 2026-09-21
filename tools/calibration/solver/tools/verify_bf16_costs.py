"""Preflight the BF16 leg of the RCO byte budget against the pinned converter.

Why this exists
---------------
RCO allocates 1-bit vs 16-bit tensors under the exact byte budget in
`packing/evidence/qwen-cost-manifest.json`, and only the Q1 leg of that budget has
ever been exercised by a measured export (3,803,453,760 bytes serialized; every one
of the 498 choices costs exactly 1.125 bits/param). The BF16 leg is arithmetic over
the *source* shape: `bf16_bytes = align(expected_nbytes(shape,30))` (export_qwen.py:92),
two bytes per element of the inventory shape.

The exporter does not serialize a bf16 choice that way. It runs
`converter.modify_tensors` and then writes GGML type 0 (FP32) for any output whose
`ndim <= 1` (export_qwen.py:162 planning, export_qwen.py:194 dtype), so the real
serialized cost of a bf16 choice is one tensor per *output*, at FP32 quadruple cost
whenever an output collapses to a vector. If `modify_tensors` ever returns more than
one tensor for a 2-D source, or an output with `ndim <= 1` or a different shape, the
budget RCO enforces is not the budget the export produces -- and a real export cannot
discover that, because it needs the 52 GiB checkpoint, while this needs neither the
weights nor a GPU: the converter is driven on `torch.empty(shape,device='meta')`,
the same meta-input call the cost manifest itself makes for the protected tensors
(export_qwen.py:97).

Checked per eligible 2-D source tensor: the pinned converter returns exactly one
output, named the manifest's `runtime_name`, with the manifest's shape, `ndim == 2`
(so the exporter writes bf16, not FP32) and contiguous. The contiguity check below catches
non-contiguity the *converter* introduces: the probe input is a meta placeholder, which is
always contiguous, so for the suffixes whose converter yields the input unchanged the check
cannot see an input-layout problem. The export feeds dense safetensors, so that is a
coverage statement rather than a live defect (an adversarial review measured it: 9 of the 14
eligible suffixes behave this way). The exporter's `converted.view(torch.uint16)`
requires contiguity (export_qwen.py:197), and the exporter's
byte rule `align(expected_nbytes(shape, 30))` equals the manifest's `bf16_bytes`.
All 498 eligible tensors are inspected; the whole set is affordable on meta tensors
(seconds, headers only), so there is no sampled path. Measured 2026-09-20 on the
pinned checkout: 498/498 agree, 53,786,705,920 bf16 bytes by the manifest's rule and
by the exporter's, under a 53,808,281,920 byte all-bf16 container.

Not checked here: the value-level conversion itself (meta tensors carry no values),
the Q1 leg (already measured end to end), RCO's chosen q1/bf16 split, and the non-2D
protected tensors, whose own `modify_tensors` expansions the manifest already records.

Usage (from tools/calibration):
    PYTHONPATH=. .venv/bin/python solver/tools/verify_bf16_costs.py

Exits non-zero, naming the first disagreement, if any eligible tensor's real
serialized bf16 cost differs from its manifest `bf16_bytes`.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# .../tools/calibration/solver/tools/verify_bf16_costs.py -> tools/calibration
CALIBRATION = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]

DOCS = REPO_ROOT / "docs" / "benchmarks" / "quantization-2026-09-19"
INVENTORY = DOCS / "encoding-inventory.json"
COSTS = CALIBRATION / "packing" / "evidence" / "qwen-cost-manifest.json"
MODEL_DIR = CALIBRATION / "runs" / "qwen3.8-27b-original"
RUNS = CALIBRATION / "runs"
# The converter checkout the exporter is given (cuda_runtime/Dockerfile exports the
# same variable for /opt/prism); the local pinned checkout is the default.
PRISM = Path(os.environ.get("ZIMFO_PRISM_SOURCE") or CALIBRATION / "cuda_runtime" / ".context" / "prism")

if str(CALIBRATION) not in sys.path:
    sys.path.insert(0, str(CALIBRATION))
from packing.export_qwen import load_converter
from packing.gguf import align, expected_nbytes


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selectable_tensors(inventory: dict) -> dict:
    """The exporter's own eligibility rule, replicated so the check covers it exactly."""
    return {name: info for name, info in inventory["tensors"].items()
            if (name.startswith("model.language_model.") or name == "lm_head.weight")
            and len(info["shape"]) == 2}


def inspect(converter, original: str, entry: dict):
    """One source tensor's real bf16 serialization under the exporter's own rules."""
    import torch
    shape = entry["shape"]
    normalized = original.replace("language_model.", "")
    bid = next((int(part) for part in normalized.split(".") if part.isdecimal()), None)
    outputs = list(converter.modify_tensors(torch.empty(shape, device="meta"), normalized, bid))
    row = {"source": original, "runtime_name": entry["runtime_name"], "shape": shape,
           "manifest_bf16_bytes": entry["bf16_bytes"], "exporter_bytes": 0, "outputs": []}
    details = []
    for name, data in outputs:
        # export_qwen.py:162/194: FP32 for ndim<=1, bf16 otherwise, per produced tensor.
        kind = 0 if data.ndim <= 1 else 30
        row["outputs"].append({"name": name, "shape": list(data.shape), "type": kind,
                               "contiguous": bool(data.is_contiguous()),
                               "bytes": align(expected_nbytes(list(data.shape), kind))})
    row["exporter_bytes"] = sum(output["bytes"] for output in row["outputs"])
    if len(outputs) != 1:
        details.append(f"{original}: pinned converter returns {len(outputs)} tensors for one 2D source "
                       f"({[output['name'] for output in row['outputs']]}), the manifest budgets one bf16 tensor "
                       f"of {entry['bf16_bytes']} bytes, the exporter writes {row['exporter_bytes']} bytes")
        return row, details
    name, data = outputs[0]
    if name != entry["runtime_name"] or list(data.shape) != list(shape):
        details.append(f"{original}: pinned converter maps it to {name} {list(data.shape)}, "
                       f"the manifest names {entry['runtime_name']} {shape} "
                       f"(real {row['exporter_bytes']} bytes, manifest {entry['bf16_bytes']})")
    elif data.ndim <= 1:
        details.append(f"{original}: output {name} has ndim {data.ndim}, so the exporter writes FP32 type 0 "
                       f"({row['exporter_bytes']} bytes), not bf16 ({entry['bf16_bytes']} bytes)")
    elif row["exporter_bytes"] != entry["bf16_bytes"]:
        details.append(f"{original}: output {name} {list(data.shape)} costs {row['exporter_bytes']} bytes, "
                       f"the manifest budgets {entry['bf16_bytes']}")
    if not data.is_contiguous():
        details.append(f"{original}: output {name} is not contiguous, the exporter's "
                       f"view(torch.uint16) would fail on {row['exporter_bytes']} bytes of bf16")
    return row, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY,
                        help="pinned tensor inventory the manifest shapes come from")
    parser.add_argument("--cost-manifest", type=Path, default=COSTS,
                        help="the RCO byte budget under test (packing/evidence/qwen-cost-manifest.json)")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="ORIGINAL checkpoint dir whose config.json hparams the exporter uses; "
                             "only safetensors headers are read, no weights are loaded")
    parser.add_argument("--prism-source", type=Path, default=PRISM,
                        help="pinned converter checkout (default $ZIMFO_PRISM_SOURCE, else the "
                             "archived cuda_runtime/.context/prism)")
    parser.add_argument("--report", type=Path, default=RUNS / "bf16-cost-preflight.json",
                        help="JSON report output path")
    args = parser.parse_args()

    if not args.inventory.is_file():
        raise SystemExit(f"Inventory not found: {args.inventory}")
    if not args.cost_manifest.is_file():
        raise SystemExit(f"Cost manifest not found: {args.cost_manifest}")
    inventory = json.loads(args.inventory.read_text())
    manifest = json.loads(args.cost_manifest.read_text())
    selectable = selectable_tensors(inventory)
    if set(selectable) != set(manifest["tensors"]):
        missing = sorted(set(selectable) - set(manifest["tensors"]))[:4]
        extra = sorted(set(manifest["tensors"]) - set(selectable))[:4]
        raise SystemExit(f"Manifest does not cover the eligible 2D tensors exactly "
                         f"({len(manifest['tensors'])} manifest, {len(selectable)} eligible; "
                         f"missing={missing} extra={extra})")
    if not (args.model_dir / "config.json").is_file():
        raise SystemExit(f"Model directory without config.json: {args.model_dir}")

    import time
    import torch
    started = time.monotonic()
    try:
        cls, gguf, revision = load_converter(args.prism_source)
        cls.no_mtp = True
        config = json.loads((args.model_dir / "config.json").read_text())
        # fname_out is stored and never opened (conversion/base.py:134); no GGUF is produced.
        converter = cls(args.model_dir, gguf.LlamaFileType.MOSTLY_Q1_0,
                        args.report.with_name("unused-metadata.gguf"), hparams=config)
    except Exception as error:
        raise SystemExit(f"The pinned converter cannot be exercised on CPU "
                         f"({args.prism_source}): {type(error).__name__}: {error} -- "
                         f"the BF16 leg of the budget stays unverified")
    instantiated = time.monotonic()

    disagreements = []
    if manifest.get("prism_revision") != revision:
        disagreements.append(f"manifest prism_revision {manifest.get('prism_revision')} is not the loaded "
                             f"converter revision {revision}; its costs describe a different converter")
    rows = []
    for original in sorted(selectable):
        entry = manifest["tensors"][original]
        if entry["shape"] != selectable[original]["shape"]:
            # Report the tool's own disagreement rather than handing the converter a
            # shape the manifest and the inventory disagree about, which would surface
            # as a converter traceback instead of this tool's verdict.
            disagreements.append(f"{original}: manifest shape {entry['shape']} is not the pinned inventory "
                                 f"shape {selectable[original]['shape']}")
            rows.append({"source": original, "runtime_name": entry["runtime_name"],
                         "shape": entry["shape"], "manifest_bf16_bytes": entry["bf16_bytes"],
                         "exporter_bytes": 0, "outputs": [],
                         "skipped": "manifest shape disagrees with the pinned inventory"})
            continue
        row, details = inspect(converter, original, entry)
        rows.append(row)
        disagreements.extend(details)
    inspected = time.monotonic()

    manifest_total = sum(row["manifest_bf16_bytes"] for row in rows)
    exporter_total = sum(row["exporter_bytes"] for row in rows)
    report = {
        "schema_version": 1,
        "purpose": "BF16 leg of the RCO byte budget vs the pinned converter",
        "prism_revision": revision,
        "inputs": {"inventory": {"path": str(args.inventory), "sha256": sha256_of(args.inventory)},
                   "cost_manifest": {"path": str(args.cost_manifest), "sha256": sha256_of(args.cost_manifest)},
                   "model_dir": str(args.model_dir),
                   "config_sha256": sha256_of(args.model_dir / "config.json"),
                   "prism_source": str(args.prism_source)},
        "eligible_tensors": len(selectable),
        "inspected": len(rows),
        "aggregate": {"manifest_bf16_bytes": manifest_total, "exporter_bf16_bytes": exporter_total},
        "seconds": {"instantiate": round(instantiated - started, 2), "inspect": round(inspected - instantiated, 2)},
        "verdict": "agrees" if not disagreements else "disagrees",
        "disagreements": disagreements,
        "unverified": [
            "output values and dtypes: meta tensors carry none, so the exporter's .to(bfloat16) and its "
            "finiteness check (export_qwen.py:195-196) are not run",
            "the Q1 leg of the budget: it is fixed by a measured export (3,803,453,760 bytes), not re-derived here",
            "RCO's chosen q1/bf16 split: this prices every eligible tensor as bf16, whatever the allocation picks",
            "the non-2D protected tensors, whose own modify_tensors expansions the manifest already records",
            "per-tensor tensor-info headers and the final alignment: they land in fixed_container_bytes, which "
            "stays valid only because this check requires the 1:1 name/shape preservation they were counted under"],
        "tensors": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if disagreements:
        raise SystemExit(f"BF16 budget disagreement ({len(disagreements)} of {len(rows)} inspected): "
                         f"{disagreements[0]}; report {args.report}")
    print(f"BF16 budget agrees: {len(rows)} eligible 2D sources, each 1:1, name/shape/ndim-preserving and "
          f"contiguous, {exporter_total} bytes == manifest {manifest_total} bytes; report {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
