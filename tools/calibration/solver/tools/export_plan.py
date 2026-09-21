"""Build a packing export plan from the solver's candidate database.

Why this exists
---------------
`packing/export_qwen.py` turns a plan into a GGUF and runs no quality checks;
`solver/` turns calibration data into candidates and reports no quality number either.
This is the missing link between them, and it is what makes a quantization
*measurable*: it can emit the all-Q1 artifact, an all-original **baseline arm**, and any
prefix-quantized arm in between, so a quality curve can be built over exactly one
variable.

Arms this expresses:

    --layers 0 --embedding bf16 --head bf16   every tensor bf16      -> the baseline control
    --layers k                                layers 0..k-1 Q1, rest bf16 -> progressive curve
    (default: every layer)                    every tensor Q1       -> the shipping artifact

`--embedding`/`--head` default to `q1` because the shipping Q1 layout quantizes both
(`docs/QUANTIZATION_PACKING.md`), so a pure control arm must set them to `bf16`
explicitly; `--layers 0` alone still emits one-bit embeddings and head.

Un-quantized tensors are written `bf16`, which preserves the original bits exactly
(`docs/QUANTIZATION_PACKING.md`: GGML type 30 for reference choices), so a partial arm
differs from the baseline only in the layers it actually quantized.

Names
-----
The solver's database is keyed `model.layers.<i>.<module>` (plus `model.embed_tokens`
and `lm_head`); the exporter's plan is keyed by the pinned inventory's HF names
(`model.language_model.layers.<i>.<module>.weight`). Both hold exactly the 498 eligible
2D tensors, and this tool asserts that bijection rather than trusting it.

Usage (from tools/calibration):
    PYTHONPATH=. .venv/bin/python solver/tools/export_plan.py \
        --candidate-database runs/native-mps/gsq-pb-<stamp>/candidate-database.json \
        --model-dir runs/qwen3.8-27b-original \
        --layers 8 --output runs/native-mps/plan-layers-8.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# .../tools/calibration/solver/tools/export_plan.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]

DOCS = REPO_ROOT / "docs" / "benchmarks" / "quantization-2026-09-19"
COSTS = REPO_ROOT / "tools" / "calibration" / "packing" / "evidence" / "qwen-cost-manifest.json"
RESERVE_BYTES = 10 * 1024**3


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def to_inventory_name(solver_key: str):
    """Map a solver candidate key to the pinned inventory's HF tensor name."""
    if solver_key == "lm_head":
        return "lm_head.weight"
    if solver_key == "model.embed_tokens":
        return "model.language_model.embed_tokens.weight"
    if solver_key.startswith("model.layers."):
        return solver_key.replace("model.layers.", "model.language_model.layers.", 1) + ".weight"
    return None


def layer_index(inventory_name: str):
    """Layer number for a per-layer tensor, or None for embeddings/head."""
    parts = inventory_name.split(".")
    if "layers" not in parts:
        return None
    try:
        return int(parts[parts.index("layers") + 1])
    except (IndexError, ValueError):
        return None


def selectable_tensors(inventory: dict) -> set:
    """The exporter's own eligibility rule, replicated so plans cover it exactly."""
    return {name for name, info in inventory["tensors"].items()
            if (name.startswith("model.language_model.") or name == "lm_head.weight")
            and len(info["shape"]) == 2}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a packing export plan from a solver candidate database.")
    parser.add_argument("--candidate-database", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="the ORIGINAL unquantized model the provenance documents describe")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=None,
                        help="quantize layers 0..N-1; unset means every available candidate")
    parser.add_argument("--allocation", type=Path, default=None,
                        help="RCO allocation.json from the rco stage: per-tensor q1/bf16 "
                             "chosen under an exact byte budget. Mutually exclusive with --layers.")
    parser.add_argument("--embedding", choices=["q1", "bf16"], default="q1")
    parser.add_argument("--head", choices=["q1", "bf16"], default="q1")
    parser.add_argument("--purpose", default="gsq candidate calibration")
    parser.add_argument("--max-gguf-bytes", type=int, default=None,
                        help="optional size gate; omit for control arms, set it for the shipping arm")
    parser.add_argument("--disk-reserve-bytes", type=int, default=RESERVE_BYTES)
    args = parser.parse_args()

    if not DOCS.is_dir():
        raise SystemExit(f"Pinned provenance documents not found at {DOCS}")
    database = json.loads(args.candidate_database.read_text())
    inventory = json.loads((DOCS / "encoding-inventory.json").read_text())
    selectable = selectable_tensors(inventory)
    costs = json.loads(COSTS.read_text())

    # The bijection is the risky part of this tool, so check it rather than assume it.
    mapping = {}
    for solver_key in database:
        inventory_name = to_inventory_name(solver_key)
        if inventory_name is None:
            raise SystemExit(f"Unmapped solver candidate key: {solver_key}")
        mapping[inventory_name] = solver_key
    if set(mapping) != selectable:
        missing = sorted(selectable - set(mapping))[:4]
        extra = sorted(set(mapping) - selectable)[:4]
        raise SystemExit(
            f"Candidate database does not cover the eligible tensors exactly "
            f"({len(mapping)} mapped, {len(selectable)} eligible; missing={missing} extra={extra})")

    # RCO's allocation is the whole point of the rco stage, and until now nothing
    # consumed `allocation.json`: a successful allocation was written and dropped,
    # so the mechanism's second half could not reach an artifact. --allocation maps
    # its per-tensor q1/bf16 choices onto the exporter's plan; --layers stays for
    # the depth curve.
    allocation = None
    if args.allocation is not None:
        if args.layers is not None:
            raise SystemExit("--allocation and --layers are alternatives; pass one")
        allocation = json.loads(args.allocation.read_text())
        choices_by_tensor = allocation.get("choices")
        if not isinstance(choices_by_tensor, dict) or not choices_by_tensor:
            raise SystemExit(f"No per-tensor choices in {args.allocation}")
        unknown = sorted(set(choices_by_tensor) - set(database))
        if unknown:
            raise SystemExit(f"Allocation names not in the candidate database: {unknown[:4]}")
        missing = sorted(set(database) - set(choices_by_tensor))
        if missing:
            raise SystemExit(f"Allocation misses {len(missing)} tensors, e.g. {missing[:4]}")

    quantized = 0
    choices = {}
    for inventory_name in sorted(selectable):
        index = layer_index(inventory_name)
        solver_key = mapping[inventory_name]
        candidate = Path(database[solver_key])

        if allocation is not None:
            # The allocation covers every eligible tensor, embedding and head
            # included, so it wins over the --embedding/--head defaults.
            mode = choices_by_tensor[solver_key]
            if mode not in ("q1", "bf16"):
                raise SystemExit(f"Unknown allocation mode {mode!r} for {solver_key}")
        elif inventory_name == "lm_head.weight":
            mode = args.head
        elif index is None:
            mode = args.embedding
        elif args.layers is not None and index >= args.layers:
            mode = "bf16"
        else:
            mode = "q1"

        if mode == "q1":
            if not candidate.is_file():
                raise SystemExit(f"Missing candidate for {solver_key}: {candidate}")
            choices[inventory_name] = {"mode": "q1", "candidate": str(candidate.resolve()),
                                       "sha256": sha256_of(candidate)}
            quantized += 1
        else:
            choices[inventory_name] = {"mode": "bf16"}

    # Preflight the exact serialized size from the pinned cost manifest, and verify the
    # tool against the one layout whose total is already known.
    per_tensor = costs["tensors"]
    total = costs["fixed_container_bytes"]
    for inventory_name, choice in choices.items():
        entry = per_tensor.get(inventory_name)
        if entry is None:
            raise SystemExit(f"Cost manifest lacks {inventory_name}")
        total += entry["q1_bytes"] if choice["mode"] == "q1" else entry["bf16_bytes"]
    if quantized == len(selectable):
        recorded = costs["all_q1_serialized_bytes"]
        if total != recorded:
            raise SystemExit(f"All-Q1 estimate {total} disagrees with the pinned {recorded}")
    required = 2 * total + int(args.disk_reserve_bytes)
    if args.max_gguf_bytes is not None and total > args.max_gguf_bytes:
        raise SystemExit(f"Estimated {total} bytes exceeds --max-gguf-bytes")

    plan = {
        "model_dir": str(args.model_dir.resolve()),
        "weight_hashes": str(DOCS / "original-weight-hashes.json"),
        "tensor_inventory": str(DOCS / "encoding-inventory.json"),
        "metadata_manifest": str(DOCS / "baseline-metadata-manifest.json"),
        "choices": choices,
        "purpose": args.purpose,
        "algorithm": "GSQ candidate export plan",
        "disk_reserve_bytes": int(args.disk_reserve_bytes),
    }
    if args.max_gguf_bytes is not None:
        plan["max_gguf_bytes"] = int(args.max_gguf_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"plan       {args.output}")
    print(f"eligibility {len(selectable)} 2D tensors, {quantized} quantized, "
          f"{len(selectable) - quantized} left at bf16")
    if allocation is not None:
        print(f"allocation {args.allocation} (RCO: {quantized} q1, "
              f"{len(selectable) - quantized} bf16)")
    print(f"estimated  {total/1e9:.3f} GB serialized "
          f"(needs ~{required/2**30:.1f} GiB free during export)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
