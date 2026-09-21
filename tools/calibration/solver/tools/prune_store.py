"""Reclaim superseded objects from a local checkpoint store, keeping the chain usable.

Why this exists
---------------
`LocalCheckpointStore` is append-only by design: `publish` writes objects and then a
commit manifest, and nothing ever deletes either. That is the right default -- a
manifest that cannot be re-read is not a checkpoint -- but a per-block GSQ sweep
publishes three commits per block, so a 64-block run leaves ~1.6 TB of which most is
superseded. Measured on one real attempt (`20260920-100739-35a4`, 3 commits):

    phase-start  13 objects  16.71 GB  -> 1.54 GB dead once the next commit lands
    block-end    14 objects  19.86 GB  -> 6.78 GB dead
    anchor       14 objects  19.72 GB  -> retained
    on disk 28.05 GB, so 8.32 GB (30%) is reclaimable per attempt

What it deletes, and what it deliberately does not
--------------------------------------------------
It deletes only OBJECTS that no retained commit references. It keeps every commit
manifest, which preserves the store's immutability contract: re-publishing a pruned
snapshot re-uploads the objects it needs (an existing name with different bytes still
fails, because the manifest is still there to compare against).

A pruned commit is no longer restorable, and `verify` will say so loudly rather than
returning something wrong. That is the intended trade: a per-block sweep only ever
resumes from the newest anchor.

Reclamation takes the store lock (`checkpoints.exclusive_store_lock`), the same one
`publish` holds across its upload-plus-manifest window. Without it, a prune racing a
live publish would either delete objects the manifest about to land references -- the
newest checkpoint then cannot be restored -- or unlink an in-flight `.upload-` temp
inside `objects/` and abort the run mid-block. If the lock is held, prune waits, and
then refuses rather than deleting.

Defaults to a dry run. `--apply` is required to delete anything, and after applying it
re-verifies every retained commit.

Usage (from tools/calibration):
    PYTHONPATH=. .venv/bin/python solver/tools/prune_store.py runs/native-mps/checkpoints/gsq/<attempt>
    PYTHONPATH=. .venv/bin/python solver/tools/prune_store.py --apply runs/native-mps/checkpoints/gsq/<attempt>
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from checkpoints import CheckpointError, exclusive_store_lock


def commit_objects(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    return {entry["sha256"]: entry["bytes"] for entry in manifest["payloads"].values()}


def order_key(path: Path):
    """Sort by the cursor then the publish ordinal embedded in the snapshot name."""
    name = path.stem
    try:
        global_step = int(name.split("-s", 1)[1].split("-", 1)[0])
        ordinal = int(name.rsplit("-p", 1)[1])
    except (IndexError, ValueError):
        return (0, 0, name)
    return (global_step, ordinal, name)


def prune(store: Path, apply: bool, keep: int, timeout_seconds: float = 300.0) -> int:
    """Reclaim superseded objects while holding the store lock.

    The lock spans both reading the retained set and deleting, because a publish
    that starts between those steps installs objects this run would already have
    classified as dead. Waiting for it is the safe direction: its manifest is the
    newest checkpoint, and a run resumes from exactly that. On timeout it refuses.
    """
    with exclusive_store_lock(store, timeout_seconds=timeout_seconds):
        return _prune_locked(store, apply, keep)


def _prune_locked(store: Path, apply: bool, keep: int) -> int:
    objects = store / "objects"
    manifests = sorted((store / "commits").glob("*.json"), key=order_key)
    if not manifests:
        print(f"{store}: no commits")
        return 0
    retained = manifests[-keep:]
    retained_hashes = set()
    for path in retained:
        retained_hashes |= set(commit_objects(path))
    print(f"{store}")
    print(f"  commits: {len(manifests)}, retaining {len(retained)} (newest by cursor)")
    for path in retained:
        print(f"    keep {path.name}")

    # Dotfiles under objects/ are in-flight `.upload-` temporaries, never
    # published payloads: an object is installed under its digest name by atomic
    # link. Unlinking one would abort the publishing process mid-copy.
    on_disk = {p.name: p.stat().st_size for p in objects.iterdir()
               if p.is_file() and not p.name.startswith(".")}
    dead = {h: size for h, size in on_disk.items() if h not in retained_hashes}
    freed = sum(dead.values())
    print(f"  objects on disk: {len(on_disk)} ({sum(on_disk.values())/1e9:.2f} GB)")
    print(f"  live: {len(retained_hashes)} "
          f"({sum(on_disk.get(h, 0) for h in retained_hashes)/1e9:.2f} GB)")
    print(f"  reclaimable: {len(dead)} objects, {freed/1e9:.2f} GB")

    missing = [h for h in retained_hashes if h not in on_disk]
    if missing:
        print(f"  REFUSING: {len(missing)} objects referenced by a retained commit are missing")
        return 1
    if not apply:
        print("  dry run; pass --apply to delete")
        return 0

    for digest in dead:
        (objects / digest).unlink()
    print(f"  deleted {len(dead)} objects, freed {freed/1e9:.2f} GB")

    from checkpoints import LocalCheckpointStore
    instance = LocalCheckpointStore(store)
    for path in retained:
        manifest = json.loads(path.read_text())
        instance.verify(path.stem, manifest["identity"])
        print(f"  verified {path.name} still restorable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reclaim objects no retained commit references, keeping the chain usable.")
    parser.add_argument("stores", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="delete; default is a dry run")
    parser.add_argument("--keep", type=int, default=1,
                        help="how many newest commits to retain (default 1)")
    args = parser.parse_args()
    if args.keep < 1:
        raise SystemExit(
            '--keep below 1 retains no commits, so every object becomes dead and the '
            'whole store is unlinked while its manifests remain; pass --keep 1 or more')
    status = 0
    for store in args.stores:
        try:
            status |= prune(store, args.apply, args.keep)
        except CheckpointError as error:
            print(f"{store}: refusing to prune: {error}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
