"""Reproduce the block-3 failure in ~3 minutes through the CLI's own entry point.

Why this exists
---------------
A per-block sweep trains blocks 0-2 and then dies entering block 3 with
`FloatingPointError: Nonfinite/missing GSQ gradient at block 3 sequence 0
(loss=nonfinite)`, reproducibly, in fresh processes, with a cache that is finite in
every one of its 87 files. Reconstructing that same update outside the CLI -- same
config, model, corpus, restored state, same 19.72 GB restore materialisation, same
19.56 GB phase-start publish -- gives a FINITE loss (487.858) and a clean update, which
is what pointed the search at `run.main` rather than at the block.

This script calls `solver.run.main` with a synthetic argv, so it reproduces the failure
in ~165 s instead of a 12-minute block, and it is the harness `bisect_main.py` uses to
bisect the prologue.

Usage (from tools/calibration):
    PYTHONPATH=. .venv/bin/python solver/tools/repro_block3_cli.py [frozen-config.json]
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# The failing attempt's own frozen config: it carries the resolved model/corpus paths,
# the identity, and the `resume_checkpoint` prefix that holds the block-3 anchor.
DEFAULT_FROZEN = "runs/native-mps/gsq-pb-20260920-102029-b216/frozen-config.json"
DEFAULT_SNAPSHOT = "gsq-b003-s00000261-e000-q00000-p00002"


def main() -> int:
    frozen = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FROZEN)
    snapshot = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SNAPSHOT
    config = json.loads(frozen.read_text())
    stamp = time.strftime("repro-block3-%Y%m%d-%H%M%S")
    config["output"] = f"runs/native-mps/{stamp}"
    config["attempt_id"] = stamp
    config["checkpoint"] = {"backend": "local", "path": "runs/native-mps/repro-stores"}
    config_path = Path(f"runs/native-mps/repro-cfg-{stamp}.json")
    config_path.write_text(json.dumps(config, indent=2))
    print(f"config={config_path}\noutput={config['output']}\n"
          f"source store={config['resume_checkpoint']['path']}\nsnapshot={snapshot}",
          flush=True)

    from solver.run import main as run_main
    sys.argv = ["solver.run", "--config", str(config_path), "--stage", "gsq",
                "--resume", snapshot]
    try:
        run_main()
        print("RESULT: main returned without raising")
        return 0
    except Exception as error:
        print(f"RESULT: RAISED {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
