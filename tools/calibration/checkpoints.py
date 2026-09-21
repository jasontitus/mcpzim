"""Manifest-last, immutable checkpoints. No solver or cloud integration is implied.

Call only at a completed optimizer-update boundary, with frozen source files.
Payload copying/hashing uses bounded buffers; tensor serialization is the caller's
responsibility. This store assumes a trusted directory on a local POSIX filesystem.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from typing import Mapping

CHUNK_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
LOCK_NAME = ".store-lock"
REQUIRED_ROLES = frozenset({"solver", "optimizer", "scheduler", "rng", "progress"})
IDENTITY_KEYS = frozenset({"baseline_repo", "baseline_revision", "calibration_sha256",
                           "solver_revision", "solver_config_sha256",
                           "candidate_database_sha256", "runtime_sha256"})
HASH = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")


class CheckpointError(ValueError):
    pass


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _name(value: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise CheckpointError("Invalid checkpoint or role name")
    return value


@contextmanager
def exclusive_store_lock(root, timeout_seconds: float = 300.0):
    """Serialize publication against object reclamation, across processes.

    `publish` installs payload objects and only then writes the commit manifest,
    which is the order crash-safety requires: a manifest never names bytes that
    were not durable first. The cost is a window in which those objects are
    referenced by no commit at all, and a concurrent `prune` -- which deletes
    every object no *retained* commit references -- would delete them, leaving the
    manifest that lands next permanently unrestorable. Prune would equally unlink
    an in-flight `.upload-` temp inside `objects/` and abort the run mid-block.

    Both are avoided by taking this lock for the whole file-level window rather
    than by reordering publish, which would trade a recoverable state for an
    unrecoverable one. The lock is advisory and per-file-descriptor, so a killed
    process releases it; prune refuses rather than deleting when it cannot get it.
    """
    path = Path(root) / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CheckpointError(f"Store lock is held by another process: {path}")
                time.sleep(0.2)
        yield
    finally:
        # Closing the descriptor releases the lock, including on error.
        os.close(descriptor)


def _identity(identity: Mapping) -> dict:
    if set(identity) != IDENTITY_KEYS:
        raise CheckpointError("Identity must bind baseline, data, solver, candidates and runtime")
    result = dict(identity)
    for key, value in result.items():
        if not isinstance(value, str) or not value:
            raise CheckpointError(f"Invalid identity field: {key}")
        if key.endswith("sha256") and not HASH.fullmatch(value):
            raise CheckpointError(f"Invalid SHA256: {key}")
    # Require immutable git revisions, not 'main', for baseline and solver.
    for key in ("baseline_revision", "solver_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", result[key]):
            raise CheckpointError(f"{key} must be a pinned commit")
    return result


class LocalCheckpointStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(exist_ok=True)
        _sync_directory(self.root)

    def _install(self, temp: Path, destination: Path):
        """Atomic create-if-absent; never replace a published file."""
        try:
            os.link(temp, destination)
        except FileExistsError:
            if digest_file(temp) != digest_file(destination):
                raise CheckpointError(f"Immutable object conflict: {destination.name}")
        _sync_directory(destination.parent)

    def _upload(self, source: Path) -> dict:
        fd, name = tempfile.mkstemp(prefix=".upload-", dir=self.objects)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as output, Path(source).open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=CHUNK_BYTES)
                output.flush()
                os.fsync(output.fileno())
            digest, size = digest_file(temp)
            # Detect ordinary source modification during copying. The caller must
            # still freeze files; this cannot make concurrent tensor writes safe.
            if digest_file(Path(source)) != (digest, size):
                raise CheckpointError("Source changed while checkpoint was copied")
            self._install(temp, self.objects / digest)
            if digest_file(self.objects / digest) != (digest, size):
                raise CheckpointError("Uploaded payload failed verification")
            return {"sha256": digest, "bytes": size}
        finally:
            temp.unlink(missing_ok=True)

    def publish(self, snapshot: str, identity: Mapping, payloads: Mapping[str, Path],
                timeout_seconds: float = 300.0) -> dict:
        """Return only after all objects and then the commit manifest are durable.

        Repeating the same snapshot and exact bytes is idempotent. Reusing its
        name with other bytes fails. Interrupted uploads may leave orphan objects,
        which are not resumable snapshots and are never selected as checkpoints.

        Holds the store lock across upload and manifest so a concurrent
        reclamation cannot delete objects this snapshot is about to reference.
        ``timeout_seconds`` bounds the wait for another holder.
        """
        with exclusive_store_lock(self.root, timeout_seconds=timeout_seconds):
            return self._publish_locked(snapshot, identity, payloads)

    def _publish_locked(self, snapshot: str, identity: Mapping, payloads: Mapping[str, Path]) -> dict:
        _name(snapshot)
        identity = _identity(identity)
        if not REQUIRED_ROLES <= set(payloads) or len(payloads) > 256:
            raise CheckpointError("Missing required state roles or too many payloads")
        for role, path in payloads.items():
            _name(role)
            if not Path(path).is_file():
                raise CheckpointError(f"Missing payload: {role}")
        manifest = {"schema": 1, "snapshot": snapshot, "identity": identity,
                    "payloads": {role: self._upload(Path(path))
                                 for role, path in sorted(payloads.items())}}
        encoded = canonical_json(manifest)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise CheckpointError("Manifest too large")
        fd, name = tempfile.mkstemp(prefix=".commit-", dir=self.commits)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._install(temp, self.commits / f"{snapshot}.json")
        finally:
            temp.unlink(missing_ok=True)
        return manifest

    def committed_snapshots(self) -> list[str]:
        """Names only, not a validity guarantee; restore rechecks every payload."""
        return sorted(path.stem for path in self.commits.glob("*.json"))

    def verify(self, snapshot: str, expected_identity: Mapping) -> dict:
        path = self.commits / f"{_name(snapshot)}.json"
        expected_identity = _identity(expected_identity)
        with path.open("rb") as stream:
            encoded = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise CheckpointError("Manifest too large")
        try:
            manifest = json.loads(encoded)
            if (set(manifest) != {"schema", "snapshot", "identity", "payloads"}
                    or manifest["schema"] != 1 or manifest["snapshot"] != snapshot
                    or manifest["identity"] != expected_identity):
                raise CheckpointError("Checkpoint identity or schema mismatch")
            payloads = manifest["payloads"]
            if not isinstance(payloads, dict) or not REQUIRED_ROLES <= set(payloads) or len(payloads) > 256:
                raise CheckpointError("Invalid checkpoint state roles")
            for role, entry in payloads.items():
                _name(role)
                if (set(entry) != {"sha256", "bytes"} or not HASH.fullmatch(entry["sha256"])
                        or type(entry["bytes"]) is not int or entry["bytes"] < 0):
                    raise CheckpointError("Invalid payload descriptor")
                try:
                    actual = digest_file(self.objects / entry["sha256"])
                except OSError as error:
                    # A pruned store reports this rather than a bare FileNotFoundError:
                    # every other failure of this store is a CheckpointError, and callers
                    # branch on it.
                    raise CheckpointError(f"Missing payload: {role}") from error
                if actual != (entry["sha256"], entry["bytes"]):
                    raise CheckpointError(f"Corrupt payload: {role}")
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise CheckpointError("Malformed checkpoint manifest") from error
        return manifest

    def restore(self, snapshot: str, expected_identity: Mapping, destination: Path,
                commit_generation=None, timeout_seconds: float = 300.0) -> dict:
        """Verify everything, stage copies, then expose one complete directory.

        Destination must not exist and must have a single restoring owner.
        No pickle/tensor deserialization is done here.

        ``commit_generation`` is accepted for interface parity with the
        generation-addressed backends and refused when supplied: this store is
        snapshot-addressed and its manifests carry no generation, so honouring the
        flag is impossible and ignoring it would silently restore a different point
        than the caller asked for. Without this the keyword arrived as an
        unexpected argument and surfaced as a bare ``TypeError`` from the CLI.

        Holds the store lock across verification and staging, because a reclamation
        running in that window would delete an object between the check that it
        exists and the copy that reads it. ``timeout_seconds`` bounds the wait for a
        concurrent publish.
        """
        if commit_generation is not None:
            raise CheckpointError(
                "The local checkpoint store is snapshot-addressed and has no commit "
                "generations; drop --commit-generation or use the gcs backend")
        with exclusive_store_lock(self.root, timeout_seconds=timeout_seconds):
            return self._restore_locked(snapshot, expected_identity, destination)

    def _restore_locked(self, snapshot: str, expected_identity: Mapping, destination: Path) -> dict:
        destination = Path(destination)
        if destination.exists():
            raise CheckpointError("Restore destination already exists")
        manifest = self.verify(snapshot, expected_identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination.parent))
        try:
            for role, entry in manifest["payloads"].items():
                output = staging / role
                with (self.objects / entry["sha256"]).open("rb") as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=CHUNK_BYTES)
                    target.flush()
                    os.fsync(target.fileno())
                if digest_file(output) != (entry["sha256"], entry["bytes"]):
                    raise CheckpointError("Payload changed during restore")
            _sync_directory(staging)
            if destination.exists():
                raise CheckpointError("Restore destination already exists")
            os.rename(staging, destination)
            _sync_directory(destination.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return manifest


def _mps_rng_state():
    """MPS generator state, or ``None`` when this run has not used MPS.

    ``torch.mps.get_rng_state`` returns a CPU ByteTensor and
    ``torch.mps.set_rng_state`` restores it exactly. Measured on this machine
    (torch 2.8.0): capture, draw, restore, draw again reproduces the same six
    values bit-for-bit, and ``torch.manual_seed`` does position the MPS
    generator. An earlier version of this module refused to restore the MPS
    state on the assumption that replay was unproven, which made **every**
    checkpoint from an MPS run unresumable. That assumption was wrong.

    Captured only when MPS has actually been used (non-zero live allocation).
    Keying on availability instead would add an MPS entry to a pure-CPU run
    wherever the machine happens to support MPS, making an otherwise valid CPU
    checkpoint unresumable for no reason.
    """
    import torch
    if not torch.backends.mps.is_available():
        return None
    if torch.mps.current_allocated_memory() == 0:
        return None
    return torch.mps.get_rng_state()


def capture_rng_state() -> dict:
    """Global Python, NumPy and torch RNGs; callers save explicit Generators too."""
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    state = {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    # Present only on a machine where an MPS generator exists; a CPU-only or
    # CUDA-only checkpoint keeps exactly the schema it had before.
    mps = _mps_rng_state()
    if mps is not None:
        state["mps"] = mps
    return state


def restore_rng_state(state: dict):
    import numpy as np
    import torch
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["torch_cuda"]) != cuda_count:
        raise CheckpointError("CUDA RNG device count changed; exact resume unavailable")
    # Restore the MPS generator too when the checkpoint carries one. This was
    # previously refused on the belief that replay was unproven; it is not —
    # capture -> draw -> restore -> draw is bit-exact on this machine, and
    # skipping the restore would leave MPS draws wherever a freshly built Metal
    # generator happened to sit, which looks like a successful resume and
    # silently perturbs every later draw.
    if state.get("mps") is not None:
        import torch
        if not torch.backends.mps.is_available():
            raise CheckpointError(
                "Checkpoint carries an MPS RNG state but MPS is unavailable in this "
                "process; exact resume is impossible here.")
        torch.mps.set_rng_state(state["mps"])
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch_cpu"])
    if cuda_count:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
