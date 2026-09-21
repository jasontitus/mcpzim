"""Bind the local Mac artifact layout to the layout the solver expects.

Why this exists
---------------
``run.validate_restored_inputs`` demands one root containing three siblings::

    <inputs>/restore-validation.json
    <inputs>/calibration/manifest.json     (+ the invocation directories)
    <inputs>/model/                        (the original BF16 shards)

That is the *cloud* layout produced by ``restore_inputs.py`` when it downloads a
GCS input package. On this Mac the same artifacts exist but in two separate
places::

    runs/mac-pipeline-v1-20260919/activations/   <- the calibration package
    runs/qwen3.8-27b-original/                   <- the original weights

The handoff is explicit that the fix is an adapter, not a fabricated receipt:
``restore-validation.json`` is inside the frozen checkpoint identity, so a
made-up one would produce checkpoints that cannot be resumed or migrated.

What this adapter does — and does not — do
------------------------------------------
It **symlinks** the two artifact trees into a single root and writes a receipt
whose contents are *derived*, not asserted:

* ``manifest_sha256`` is recomputed from the calibration manifest it points at.
* ``input_commit_sha256`` is recomputed from the GCS input manifest when one is
  available locally, so the identity still describes the real upstream package.
* ``model`` file coverage and small-file hashes are recomputed from the weights
  directory.

It does **not** hash the 52 GiB of weight shards by default — that is exactly what
``restore_inputs.py`` does on the cloud path, and it takes a long time. Pass
``--hash-shards`` to do it properly. Without it the receipt records
``shards_hashed: false`` so a reader knows what was and was not verified; the
solver's own size check still runs either way.

Usage::

    cd tools/calibration
    PYTHONPATH=. .venv/bin/python solver/tools/prepare_local_inputs.py \\
        --activations runs/mac-pipeline-v1-20260919/activations \\
        --model       runs/qwen3.8-27b-original \\
        --out         runs/local-inputs \\
        [--gcs-manifest runs/gcs-staging-v1-20260919/inputs-manifest.json] \\
        [--hash-shards]

Then point the stage config at ``--out`` as ``inputs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

#: The baseline the whole pipeline is pinned to. A mismatch here means the
#: artifacts are not the model this solver was characterized against.
EXPECTED_MODEL = "Qwen/Qwen3.8-27B"
EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def digest(path: Path, chunk: int = 8 * 1024**2) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def link_tree(source: Path, destination: Path) -> None:
    """Symlink ``source`` at ``destination`` without touching the source.

    A symlink rather than a copy: these are 21 GiB and 52 GiB trees, and the
    point of the adapter is to describe what already exists, not to duplicate it.
    ``run.validate_restored_inputs`` explicitly rejects a *symlinked model file*
    (``path.is_symlink()``), so for the model directory the files are hard-linked
    where the filesystem allows and copied where it does not.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"refusing to overwrite existing {destination}")
    os.symlink(source.resolve(), destination, target_is_directory=True)


def hardlink_tree(source: Path, destination: Path, only: set[str] | None = None) -> list[str]:
    """Recreate ``source`` at ``destination`` with hard links (no data copy).

    Hard links keep the solver's ``is_symlink()`` check satisfied while costing
    no additional space on the same filesystem.

    ``only``, when given, restricts the tree to those top-level names. This
    matters: ``run.validate_restored_inputs`` compares the *directory listing*
    against the calibration manifest's file list, so any extra file in the model
    directory — a downloader's ``download-status.json``, a ``.cache`` directory —
    fails the coverage check even though the model itself is complete. Those are
    bookkeeping, not model artifacts, and are excluded rather than linked and
    then filtered from the receipt: the directory must be clean, not just the
    record of it.

    Returns the names that were skipped.
    """
    destination.mkdir(parents=True, exist_ok=False)
    skipped: list[str] = []
    for entry in sorted(source.iterdir()):
        if only is not None and entry.name not in only:
            skipped.append(entry.name)
            continue
        if entry.is_dir():
            hardlink_tree(entry, destination / entry.name)
        else:
            try:
                os.link(entry, destination / entry.name)
            except OSError:
                shutil.copy2(entry, destination / entry.name)
    return skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gcs-manifest", type=Path, default=None,
                        help="the GCS inputs manifest, if present locally")
    parser.add_argument("--hash-shards", action="store_true",
                        help="hash the weight shards too (slow; the cloud path does this)")
    args = parser.parse_args()

    activations = args.activations.resolve()
    model_src = args.model.resolve()
    out = args.out.resolve()

    manifest_path = activations / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no calibration manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    # Verify the baseline identity before writing anything.
    if manifest.get("model") != EXPECTED_MODEL:
        raise SystemExit(
            f"calibration manifest is for {manifest.get('model')!r}, "
            f"expected {EXPECTED_MODEL!r}")
    if manifest.get("revision") != EXPECTED_REVISION:
        raise SystemExit(
            f"calibration manifest revision {manifest.get('revision')!r} does not "
            f"match the pinned {EXPECTED_REVISION!r}")
    if manifest.get("status") != "completed":
        raise SystemExit(f"calibration package status is {manifest.get('status')!r}")

    sequences = manifest.get("sequences") or []
    if len(sequences) != manifest.get("source_invocation_count"):
        raise SystemExit(
            f"incomplete corpus: {len(sequences)} sequences vs "
            f"{manifest.get('source_invocation_count')} declared")
    missing = [s["directory"] for s in sequences if not (activations / s["directory"]).is_dir()]
    if missing:
        raise SystemExit(f"{len(missing)} invocation directories missing, e.g. {missing[:3]}")

    out.mkdir(parents=True, exist_ok=True)
    calibration = out / "calibration"
    model = out / "model"
    if calibration.exists() or calibration.is_symlink():
        raise SystemExit(f"refusing to overwrite {calibration}")
    link_tree(activations, calibration)

    # The model directory must contain exactly what the calibration manifest
    # names, so the solver's coverage check passes. Compute that set first.
    expected_names = set((manifest.get("model_artifacts") or {}).get("files") or {})
    if model.exists():
        raise SystemExit(f"refusing to overwrite {model}")
    print(f"hard-linking {model_src} -> {model} (no data copy)")
    skipped = hardlink_tree(model_src, model, only=expected_names or None)
    if skipped:
        print(f"  excluded {len(skipped)} non-model entr(ies): {sorted(skipped)}")

    # Coverage and hashes are derived from the artifact tree, but the tree may
    # carry files that are not part of the model. `download-status.json` is the
    # downloader's own bookkeeping and is present here; counting it would make
    # `run.validate_restored_inputs`'s coverage check fail, because that check
    # compares the directory listing against the *calibration manifest's* file
    # list, which describes the model and nothing else.
    #
    # Keep only files the model actually claims. If the manifest has no file list
    # to check against, fall back to the whole tree and say so in the receipt.
    files = {}
    for path in sorted(model.iterdir()):
        if not path.is_file():
            continue
        entry = {"bytes": path.stat().st_size}
        if args.hash_shards or not path.name.endswith(".safetensors"):
            entry["sha256"] = digest(path)
        files[path.name] = entry
    if not files:
        raise SystemExit(f"no model files found under {model}")
    if expected_names:
        absent = expected_names - set(files)
        if absent:
            raise SystemExit(
                f"{len(absent)} model file(s) named by the calibration manifest are "
                f"missing from {model}: {sorted(absent)[:5]}"
            )

    input_commit = None
    if args.gcs_manifest and args.gcs_manifest.is_file():
        input_commit = digest(args.gcs_manifest)
        print(f"input_commit_sha256 from {args.gcs_manifest.name}: {input_commit[:16]}…")

    receipt = {
        "status": "validated",
        "input_commit_sha256": input_commit,
        "manifest_sha256": digest(manifest_path),
        "model_artifacts": {
            "files": files,
            "shards_hashed": bool(args.hash_shards),
        },
        "provenance": {
            "adapter": "solver/tools/prepare_local_inputs.py",
            "activations_source": str(activations),
            "model_source": str(model_src),
            "invocations": len(sequences),
            "note": (
                "Local-layout adapter. Hashes are recomputed from the artifacts "
                "rather than copied from a cloud receipt."
                + ("" if args.hash_shards else
                   " Weight shards were NOT hashed; only their sizes were checked.")
            ),
        },
    }
    (out / "restore-validation.json").write_text(json.dumps(receipt, indent=2))

    print(f"\nwrote {out}/restore-validation.json")
    print(f"  invocations : {len(sequences)}")
    print(f"  model files : {len(files)}  (shards hashed: {bool(args.hash_shards)})")
    print(f"  manifest    : {receipt['manifest_sha256'][:16]}…")
    print(f"\nPoint the stage config's 'inputs' at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())