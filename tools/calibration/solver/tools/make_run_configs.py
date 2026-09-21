#!/usr/bin/env python3
"""Build the frozen stage configs for a native MPS GSQ+RCO run.

Why this exists
---------------
The cloud path generates stage configs from ``solver/job.py``, which assumes a
container image, a GCS checkpoint bucket, and the ``/mnt/zimfo-inputs`` layout.
None of those exist here. This script writes the same *config contract* —
the keys ``solver.run`` reads positionally — against local paths, with the
identity material derived rather than copied:

* ``runtime_sha256``  from the solver source itself (see ``upstream_paths.runtime_digest``)
* ``input_commit_sha256``  from the GCS input manifest when present locally
* ``identity.calibration_sha256``  from the calibration manifest
* ``cost_manifest``  path + hash from the checked-in packing evidence

It does **not** fabricate a receipt or invent an identity. If a required artifact
is missing it says which one and stops.

Usage::

    cd tools/calibration
    PYTHONPATH=. .venv/bin/python solver/tools/make_run_configs.py \\
        --stage gsq --out runs/native-mps
    PYTHONPATH=. .venv/bin/python solver/tools/make_run_configs.py \\
        --stage rco --out runs/native-mps --gsq-output runs/native-mps/gsq

Stages are written one file per stage under ``<out>/<stage>/frozen-config.json``,
mirroring what ``solver.job`` produces, so the same resume machinery applies. The
``rco`` stage is the one exception, in both ways: it continues a chain named by
``--gsq-output`` rather than starting from the inputs alone, and its config is
written beside the output directory it names, because ``run.main`` refuses a
non-empty output directory and the config sitting in it would be that directory's
only entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: tools/calibration/solver/tools/ -> tools/calibration/
CALIBRATION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CALIBRATION))

ALL_STAGES = ["initialize", "smoke_gsq", "smoke_rco", "embedding", "gsq", "head", "rco"]

#: Byte budget per stage, both legs inside the ~5.5 GB phone budget.
DEFAULT_TARGET_BYTES = 4_000_000_000
#: The pinned knapsack rounds every group's cost to 1/500 bit-param units, so with
#: the 498 groups this model presents (one per candidate-database entry) its all-Q1
#: floor is 1.228 bits/param against the true 1.125: below this the dp is never
#: overwritten, argmax returns 0 and the backtrack reads wrapped entries, so the
#: allocation comes back SILENTLY uniform-Q1 rather than failing (and just below
#: it, an IndexError). Only the rco stage turns the budget into an allocation, so
#: only the rco stage is held to the floor.
RCO_MIN_TARGET_BYTES = 4_146_400_000
RCO_TARGET_BYTES = 4_200_000_000


def digest(path: Path, chunk: int = 8 * 1024**2) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rco_inputs_from_gsq(gsq_output: Path) -> dict:
    """The rco stage's inputs, derived from the gsq stage it continues.

    ``rco_run`` reads ``candidate_database`` (run.py:765), ``candidate_archives``
    (run.py:794) and ``boundary_reports`` (run.py:756); the shared ``base`` below
    carries none of them, and a config without them dies of a KeyError after the
    whole 27B model is loaded. Nothing here is invented: the database, its
    archives and the learned boundary reports are the ones the named chain
    already produced, and each is checked to exist before any config is written.
    """
    gsq_output = Path(gsq_output)
    database = gsq_output / "candidate-database.json"
    if not database.is_file():
        raise SystemExit(
            f"{gsq_output} holds no candidate-database.json; --gsq-output must name "
            f"a gsq stage output directory"
        )
    entries = json.loads(database.read_text())

    # `read_database` (run.py:286-289) resolves a relative entry against the
    # DATABASE FILE's directory. That is only right for the absolute entries a
    # gsq stage with an absolute `output` writes: a stage run with a relative
    # `output` writes entries that `read_database` turns into absolute paths
    # that do not exist, and rco then cannot open a single candidate. Resolve
    # exactly the way it will and refuse such a chain here, by name.
    def resolved(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else database.parent / path

    unresolved = [name for name, value in entries.items() if not resolved(value).is_file()]
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} of {len(entries)} candidates named by {database} do not resolve "
            f"the way run.read_database resolves them (relative entries against {database.parent}); "
            f"first: {unresolved[0]} -> {entries[unresolved[0]]}. A gsq stage whose own `output` was "
            f"relative writes entries only that working directory could resolve."
        )
    uncovered = [name for name in ("model.embed_tokens", "lm_head") if name not in entries]
    if uncovered:
        # RCOTrainer needs a candidate for every eligible projection plus the
        # embedding and the head (rco.py:38); the rest of the coverage check is
        # its job, not this script's.
        raise SystemExit(f"{database} names no {uncovered}")

    # Roles, not paths, are what `restore_candidates` (run.py:292-309) reads back
    # out of a checkpoint, so the gsq stage's own role names carry over: one
    # archive per trained block, written as `block-<n>.tar` (run.py:622). The
    # `cache-*` archives that stage also carries are the per-block input
    # activations the NEXT gsq block trains on (run.py:510), and every archive
    # listed here is copied into every rco checkpoint, so they stay behind.
    archives = {f"candidate_block_{block}": path
                for path in (gsq_output.glob("block-*.tar"))
                if (block := path.stem.removeprefix("block-")).isdigit()}
    # A resumed rco rebuilds its candidate database from the archives its own
    # checkpoint carries (run.py:769), so the chain's initial archive -- the one
    # place the embedding and head candidates live -- has to come with it. The
    # gsq stage recorded which archive that was: `solver.job` chains it into the
    # gsq config (job.py:213-215) and gsq_run reads it as `initial_candidates`.
    chain = gsq_output / "frozen-config.json"
    inherited = json.loads(chain.read_text()).get("candidate_archives", {}).get("initial_candidates") if chain.is_file() else None
    if not inherited:
        raise SystemExit(
            f"{gsq_output} records no initial_candidates archive, so a resumed rco would "
            f"rebuild a database with no embedding/head candidates"
        )
    initial = Path(inherited)
    initial = initial if initial.is_absolute() else CALIBRATION / initial
    if not initial.is_file():
        raise SystemExit(f"{gsq_output} was chained from {initial}, which does not exist")
    archives["initial_candidates"] = initial

    # The learned boundary reports come from the `embedding` and `head` stages,
    # which this script writes beside the gsq stage (one `<out>/<stage>` per
    # stage), and the descriptor shape -- path plus hash -- is the one
    # `solver.job` appends for those two stages (job.py:229).
    reports = []
    for stage in ("embedding", "head"):
        path = gsq_output.parent / stage / f"{stage}-report.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "completed" or report.get("updates", 0) <= 0 or report.get("stage") != stage:
            # Handing this to rco only moves the refusal into the stage
            # ('Boundary training did not complete', run.py:759-760), and
            # dropping it silently would trade a learned boundary for an RTN
            # seed. Neither is this script's decision to make.
            raise SystemExit(
                f"{path} is not a completed {stage} boundary report (status={report.get('status')!r}, "
                f"updates={report.get('updates')!r}, stage={report.get('stage')!r}); finish that stage "
                f"or remove the report"
            )
        reports.append({"path": str(path), "sha256": digest(path)})

    keys = {"candidate_database": str(database),
            "candidate_archives": {role: str(path) for role, path in archives.items()}}
    if len(reports) == 2:
        return {**keys, "boundary_reports": reports}
    # `rco_run` refuses an RTN-only embedding/head outright: the seeds are not a
    # final recipe, so it demands a bounded <=2-step run AND a config that says
    # the seed is known to be one (run.py:763-764). The bound is the caller's to
    # pass; the acknowledgement is the config's to make.
    return {**keys, "allow_rtn_boundary_smoke": True}


def build(stage: str, inputs: Path, out: Path, target_bytes: int,
          checkpoint_seconds: int, deadline_seconds: int, seed: int,
          upstream: str | None, checkpoint_root: Path, rco_inputs: dict | None = None) -> dict:
    from solver.upstream_paths import resolve, runtime_digest, PINNED_REVISIONS

    calibration_manifest = inputs / "calibration" / "manifest.json"
    if not calibration_manifest.is_file():
        raise SystemExit(
            f"no calibration manifest at {calibration_manifest}; run "
            f"solver/tools/prepare_local_inputs.py first"
        )
    manifest = json.loads(calibration_manifest.read_text())

    cost_manifest = CALIBRATION / "packing" / "evidence" / "qwen-cost-manifest.json"
    if not cost_manifest.is_file():
        raise SystemExit(f"no cost manifest at {cost_manifest}")

    receipt = json.loads((inputs / "restore-validation.json").read_text())

    base = {
        "inputs": str(inputs),
        "input_commit_sha256": receipt.get("input_commit_sha256"),
        "runtime_sha256": runtime_digest(),
        "largest_first": True,
        "checkpoint_seconds": checkpoint_seconds,
        "deadline_unix": __import__("time").time() + deadline_seconds,
        "target_bytes": target_bytes,
        "seed": seed,
        "cost_manifest": {"path": str(cost_manifest), "sha256": digest(cost_manifest)},
        "upstream": resolve(upstream),
        "stage": stage,
        "output": str(out / stage),
        "checkpoint": {"backend": "local", "path": str(checkpoint_root / stage)},
        "gsq_execution_device": "mps",
        # Every measured number this port has (per-block cost, peak memory, the
        # 75 s propagation) comes from `block_cpu_offload`, which keeps one block
        # on the accelerator and the model on CPU. Omitting the key silently runs
        # a different configuration: measured tonight, the same validation stage
        # built its input cache roughly 11x slower with no memory mode set, at
        # 0.3% CPU, on an otherwise idle machine.
        "gsq_memory_mode": "block_cpu_offload",
        # D3.1. Both bounds, because they catch different shapes of the same
        # failure: `growth` fires on the per-block inflation the port measured
        # (2.94x after one quantized block, so it trips at the first block with a
        # predecessor), and `ratio` is the backstop for a single block that is
        # catastrophic on its own without a dramatic step change. Without these
        # keys the guard in `gsq_run` is inert, which is how a chain reaches a
        # 60,800 loss before anyone sees it.
        "gsq_max_drift_growth": 2.0,
        "gsq_max_drift_ratio": 4.0,
    }

    # NOTE: do NOT write an `identity` block here. `run.bind_identity` COMPUTES
    # it from the frozen config plus the corpus and candidate database, and
    # raises if a supplied block disagrees. Pre-baking one — as an earlier
    # version of this script did — makes every run fail its own identity check.
    # What this script must supply is the *input* to that computation:
    # `runtime_sha256` (the solver-source digest), the corpus path, and a stable
    # candidate database. Everything else is derived at run time.
    if stage == "rco":
        # The base above describes a GSQ block-residency run, and `rco` is not
        # one. `run.main` calls `memory_mode` on every stage before the model is
        # loaded (run.py:930), and that refuses `block_cpu_offload` for any stage
        # but gsq (gsq_residency.py:24-26) -- which is the refusal a generated
        # rco config used to die on. RCO holds no per-block residency: the whole
        # model is resident, which is `full`. That is the default for a missing
        # key, but it is stated here rather than left to a default, because for
        # every other stage in this file the missing key silently means something
        # else.
        base["gsq_memory_mode"] = "full"
        if rco_inputs is None:
            raise SystemExit(
                "--stage rco continues a chain: pass --gsq-output <gsq stage output "
                "directory>. Its candidate database, candidate archives and boundary "
                "reports are derived from that directory; none of them can be guessed"
            )
        base.update(rco_inputs)
    return base

    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path,
                        default=CALIBRATION / "runs" / "local-inputs")
    parser.add_argument("--out", type=Path,
                        default=CALIBRATION / "runs" / "native-mps")
    parser.add_argument("--stage", default=None, choices=[None] + ALL_STAGES,
                        help="one stage; omit for all")
    parser.add_argument("--target-bytes", type=int, default=None,
                        help=f"exact byte budget; {DEFAULT_TARGET_BYTES} normally, "
                             f"{RCO_TARGET_BYTES} for --stage rco, whose solver floor is "
                             f"{RCO_MIN_TARGET_BYTES}")
    parser.add_argument("--checkpoint-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=CALIBRATION / "runs" / "native-mps" / "checkpoints")
    parser.add_argument("--gsq-output", type=Path, default=None,
                        help="gsq stage output the rco stage continues; required for --stage rco")
    args = parser.parse_args()

    inputs = args.inputs.resolve()
    if not (inputs / "restore-validation.json").is_file():
        raise SystemExit(
            f"{inputs} has no restore-validation.json; run "
            f"solver/tools/prepare_local_inputs.py first"
        )

    stages = [args.stage] if args.stage else ALL_STAGES
    out_root = args.out.resolve()
    if "rco" in stages and args.target_bytes is not None and args.target_bytes < RCO_MIN_TARGET_BYTES:
        raise SystemExit(
            f"--target-bytes {args.target_bytes} is below the rco solver's floor of "
            f"{RCO_MIN_TARGET_BYTES}: the pinned knapsack rounds each group's cost to 1/500 "
            f"bit-param units, so with 498 groups its all-Q1 floor is 1.228 bits/param against "
            f"the true 1.125, the budget becomes infeasible, and the allocation comes back "
            f"silently uniform-Q1 instead of saying so"
        )
    # Derive the rco stage's chain before anything is written, so an unavailable
    # chain costs nothing rather than leaving a half-written set of configs.
    rco_inputs = None
    if "rco" in stages:
        if args.gsq_output is None:
            raise SystemExit(
                "--stage rco continues a gsq chain: pass --gsq-output <gsq stage output "
                "directory>"
            )
        rco_inputs = rco_inputs_from_gsq(args.gsq_output.resolve())
    for stage in stages:
        # Only the rco stage turns `target_bytes` into an allocation, so only it
        # takes the floor-clearing default above the `initialize`/`gsq`/`head`
        # budget those stages have always carried.
        stage_bytes = (args.target_bytes if args.target_bytes is not None
                       else (RCO_TARGET_BYTES if stage == "rco" else DEFAULT_TARGET_BYTES))
        cfg = build(stage, inputs, out_root, stage_bytes,
                    args.checkpoint_seconds, int(args.deadline_hours * 3600),
                    args.seed, args.upstream, args.checkpoint_root.resolve(),
                    rco_inputs=rco_inputs)
        # The rco config is written BESIDE the output directory it names, never
        # inside it: `run.main` refuses a non-empty output directory (run.py:926)
        # and this file alone is enough to trip that before the stage reaches its
        # own validation. `solver.job` writes its stage configs the same way, at
        # the job root beside the stage directory each one names (job.py:181).
        # Every other stage keeps the layout it has always had.
        target = (out_root / (stage + "-config.json") if stage == "rco"
                  else out_root / stage / "frozen-config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(cfg, indent=2))
        print(f"  {stage:12} -> {target}")

    print(f"\nRun one stage with:")
    print(f"  cd {CALIBRATION} && PYTHONPATH=. .venv/bin/python -m solver.run \\")
    print(f"      --config {out_root}/<stage>/frozen-config.json --stage <stage>")
    if "rco" in stages:
        print(f"      --config {out_root}/rco-config.json --stage rco   # rco is never run from <out>/rco")
        if rco_inputs.get("allow_rtn_boundary_smoke"):
            print(f"\nNote: no completed embedding+head boundary reports were found beside")
            print(f"{args.gsq_output}; the rco config carries allow_rtn_boundary_smoke, which")
            print("rco_run accepts only for a run bounded to --max-steps 1 or 2.")
    print("\nNote: `run.main` refuses an existing non-empty output directory, so")
    print("each stage needs its own --out or a cleared directory between attempts.")
    print("A config sitting inside the output directory it names is itself that")
    print("directory's first entry, which is exactly why the rco config does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())