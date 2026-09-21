"""Bisect `run.main`'s prologue for the block-3 failure, once the CLI reproduces it.

`repro_block3_cli.py` shows the failure comes from `main` and not from the update: a
direct `gsq_run` call with the same config, model, corpus, restored state, restore
materialisation and publish trains the first update fine (loss 487.858). `main`'s
prologue calls three module-level steps that a direct call skips, and stubbing them --
of which `configure` alone is sufficient -- makes block 3 train all 87 steps and export
its candidates.

    validate_restored_inputs(config)
    bind_identity(config, manifest)
    reproducibility.configure(seed, compute_device)

`configure` is what `main` calls before the model is loaded; on MPS it installs
`torch.use_deterministic_algorithms(True, warn_only=True)`, which `reproducibility.py`
itself says buys nothing on Metal. Setting that flag AFTER the model load and the state
restore does not reproduce the failure, so what matters is the flag being in effect
while the model is loaded and the block prepared.

Usage (from tools/calibration):
    STUB=configure PYTHONPATH=. .venv/bin/python solver/tools/bisect_main.py [frozen] [snapshot]

STUB is a comma-separated subset of the three names; empty means "stub nothing", which
should reproduce the failure.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FROZEN = "runs/native-mps/gsq-pb-20260920-102029-b216/frozen-config.json"
DEFAULT_SNAPSHOT = "gsq-b003-s00000261-e000-q00000-p00002"
STUB = {name.strip() for name in os.environ.get("STUB", "").split(",") if name.strip()}


def main() -> int:
    import solver.run as R

    if "validate_restored_inputs" in STUB:
        R.validate_restored_inputs = lambda config: {}
    if "bind_identity" in STUB:
        R.bind_identity = lambda config, manifest: config["identity"]
    if "configure" in STUB:
        # `main` imports this locally (`from .reproducibility import configure`), so the
        # patch has to land on the source module rather than on run's namespace.
        import solver.reproducibility as REP
        REP.configure = lambda seed, device=None: {"stubbed": True, "seed": seed,
                                                   "device": str(device)}
    print(f"stubbed: {sorted(STUB) or 'nothing'}", flush=True)

    frozen = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FROZEN)
    snapshot = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SNAPSHOT
    config = json.loads(frozen.read_text())
    stamp = time.strftime("probe-bisect-%Y%m%d-%H%M%S")
    config["output"] = f"runs/native-mps/{stamp}"
    config["attempt_id"] = stamp
    config["checkpoint"] = {"backend": "local", "path": "runs/native-mps/probe-stores"}
    config_path = Path(f"runs/native-mps/probe-cfg-{stamp}.json")
    config_path.write_text(json.dumps(config, indent=2))
    print(f"config={config_path}\noutput={config['output']}", flush=True)

    sys.argv = ["solver.run", "--config", str(config_path), "--stage", "gsq",
                "--resume", snapshot]
    try:
        R.main()
        print("VERDICT: main returned without raising (the stubbed steps are implicated)")
        return 0
    except Exception as error:
        print(f"VERDICT: RAISED {type(error).__name__}: {str(error)[:200]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
