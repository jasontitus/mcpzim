"""Where the pinned upstream sources live.

Why this module exists
----------------------
Ten call sites hardcoded ``upstream='/opt/upstream'``. That path is the *container*
layout the CUDA image used; it does not exist on a workstation, and the same
sources are already checked in locally at
``tools/calibration/cuda_runtime/.context/sources`` (with a manifest recording
their provenance). The failure mode of the hardcoded default is a bare
``ModuleNotFoundError: No module named 'manifold'`` from deep inside a stage,
naming neither the path nor the fix.

Resolution order, first hit wins:

1. an explicit argument (a config value), if the directory exists
2. ``GSQ_UPSTREAM_DIR`` from the environment
3. the checked-in ``cuda_runtime/.context/sources`` next to this package
4. ``/opt/upstream`` (the container layout)

A path that does not contain both ``gsq/src/quantization`` and ``rco/src/manifold.py``
is rejected with an error naming what is missing, rather than being returned and
failing later.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
#: tools/calibration/solver/ -> tools/calibration/
_PACKAGE_ROOT = _HERE.parents[1]
_CHECKED_IN = _PACKAGE_ROOT / "cuda_runtime" / ".context" / "sources"

_REQUIRED = (
    Path("gsq") / "src" / "quantization",
    Path("rco") / "src" / "manifold.py",
)


def _is_valid(root: Path) -> bool:
    return root.is_dir() and all((root / part).exists() for part in _REQUIRED)


#: Pinned revisions, from the same manifest that records the file hashes.
PINNED_REVISIONS = {
    "gsq": "03fc16484c369e3127225615d5e03e8d3a6043e3",
    "rco": "9a1e09c07d468109cbe60a1b87d5036034a79d10",
}

_MANIFEST = _CHECKED_IN / "manifest.json"


def verify_revision(root: Path) -> list[str]:
    """Check ``root`` against the checked-in manifest. Returns problems, if any.

    The pinned revisions are load-bearing: the port's Gumbel quantizer was
    validated against GSQ at a specific commit, and a different revision could
    change the quantizer algebra or the RCO manifold under it. Without this check
    a stale or hand-set ``GSQ_UPSTREAM_DIR`` would be used silently, and the run
    would record an identity that does not describe what actually ran.

    Verification is by **content hash** where the manifest has one, and by
    revision where the tree carries git metadata. A directory that matches
    neither is reported, not rejected — the environment may legitimately point at
    an un-hashed working copy — but the caller can then decide.
    """
    problems: list[str] = []
    if not _MANIFEST.exists():
        problems.append(f"no manifest at {_MANIFEST}; cannot verify revisions")
        return problems
    try:
        manifest = json.loads(_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"manifest at {_MANIFEST} unreadable: {exc}")
        return problems

    for name, expected_revision in PINNED_REVISIONS.items():
        tree = root / name
        expected_files = manifest.get(name, {}).get("files")
        if tree.is_dir() and expected_files:
            mismatched = []
            missing = []
            for rel, want in expected_files.items():
                path = tree / rel
                if not path.is_file():
                    missing.append(rel)
                    continue
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != want:
                    mismatched.append(rel)
            if mismatched or missing:
                problems.append(
                    f"{name} at {tree} does not match the pinned manifest "
                    f"({len(mismatched)} changed, {len(missing)} missing; "
                    f"expected revision {expected_revision[:12]})"
                )
    return problems


def resolve(explicit: str | os.PathLike | None = None, verify: bool = True) -> str:
    """Return a directory holding both upstream trees, or raise naming the fix.

    With ``verify`` (the default) the chosen root is checked against the pinned
    manifest, and a mismatch raises rather than silently running against
    different quantizer source than the port was validated with.
    """
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("GSQ_UPSTREAM_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(_CHECKED_IN)
    candidates.append(Path("/opt/upstream"))

    for candidate in candidates:
        if not _is_valid(candidate):
            continue
        if verify:
            problems = verify_revision(candidate)
            if problems:
                raise RuntimeError(
                    f"Upstream sources at {candidate} do not match the pinned "
                    f"revisions:\n  " + "\n  ".join(problems) + "\n"
                    "The port's Gumbel quantizer and RCO manifold were validated "
                    "against these exact revisions. Point GSQ_UPSTREAM_DIR at a "
                    "clean checkout, or pass verify=False if you intend to run "
                    "against a different revision and will record that in the "
                    "run identity."
                )
        return str(candidate)

    attempted = "\n".join(f"    {c}" for c in candidates)
    missing = "\n".join(f"    {c / part}" for c in candidates for part in _REQUIRED)
    raise FileNotFoundError(
        "Could not locate the pinned upstream GSQ/RCO sources.\n"
        f"Attempted roots:\n{attempted}\n"
        f"Each must contain {' and '.join(str(p) for p in _REQUIRED)}.\n"
        "Checked for:\n"
        f"{missing}\n"
        "Fix: set GSQ_UPSTREAM_DIR, or pass `upstream` in the stage config."
    )

def runtime_digest() -> str:
    """A content digest identifying the solver code that is running.

    On the cloud path this was a container image digest
    (``runtime_image.split('@sha256:')[1]``), which is a sound identity for a
    frozen image. There is no image here: the code is a working tree that the
    port modifies. Using a stale container digest would be wrong in a specific
    way — a checkpoint written by the *ported* solver would carry the identity of
    the *CUDA* runtime, so a resume could believe it was reading a run it cannot
    reproduce.

    So the digest is computed from the source itself: the SHA-256 of every
    ``solver/*.py`` file, of the checkpoint-format modules that live beside the
    package, plus the pinned upstream revisions. Any edit to the port changes it,
    which is the correct behaviour — a checkpoint is resumable under the code
    that wrote it and nothing else.

    ``checkpoints.py`` is included because the port **changed** it: the MPS RNG
    capture and the matching restore branch live there, and it defines the
    manifest schema, the required roles and the snapshot naming. A digest over
    ``solver/*.py`` alone would admit a checkpoint written before that change to
    a process running after it, which is exactly the misidentification this
    function exists to prevent. Files absent on a given machine are skipped so
    the digest stays computable on a partial checkout.

    Returns a 64-character hex digest, matching the shape an image digest had.
    """
    h = hashlib.sha256()
    solver = _HERE.parent
    sources = sorted(solver.glob("*.py"))
    # Checkpoint format modules live at the package root, not inside solver/.
    for name in ("checkpoints.py", "checkpoint_bridge.py", "checkpoint_plan.py"):
        candidate = _PACKAGE_ROOT / name
        if candidate.is_file():
            sources.append(candidate)
    for path in sources:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    for name, revision in sorted(PINNED_REVISIONS.items()):
        h.update(f"{name}:{revision}".encode())
    return h.hexdigest()
