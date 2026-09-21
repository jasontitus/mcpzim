#!/usr/bin/env python3
"""Sweep the D4/D5 budget curve: quality against depth, and quality against epochs.

Why this exists
---------------
`docs/QUANTIZATION_QUALITY_PIPELINE_2026-09-20.md` §3 D4 (:333) makes a
quality-vs-depth curve the first end-to-end signal this pipeline has ever had, and D5
(:339) makes it the decision instrument: 1 epoch is 87 updates = 465 s of training per
block, so 64 blocks costs ~11.4 h at one epoch, ~19 h at two and ~33 h at four (§5,
:708-716). Nothing else in the repo can choose between those budgets, because the app
suite cannot rank arms at all - 49 of its 66 frozen turns invoke no model in either arm
and its only paired result is 4 wins against 1 loss (exact McNemar p = 0.375, §2.0.1) -
so §2.4 (:224) lists "budget chosen from a measured quality/cost curve" as MISSING and
§4 item 12 (:706) names this runner.

A point is one artifact plus one number, and the curve is two cuts through that:

  depth arm   `export_plan.py --layers k` over the completed `gsq` chain's candidate
              database - layers 0..k-1 Q1, every other tensor bf16, which preserves the
              original bits exactly - exported to a GGUF by `packing.export_qwen.py`,
              then measured by `quality_gate.py` on the held-out app text.
  epochs arm  a fresh `gsq` run over the shallow subset at `gsq_epochs = e`, then the
              same export and the same measurement at that subset's depth.

Each point is appended to one JSON results file with its label, layers, epochs, artifact
bytes, perplexity and stderr, the chain's per-block composed-stream drift record (D3.1)
and its own wall-clock cost. The sweep is serial, resumable and loud:

* **serial.** D10 (:460) measured that two concurrent model loads hang at ~0% CPU rather
  than fail, which is how a 60 s measurement becomes a 25 minute hang. Measurement arms
  run strictly one at a time here, and `<results>.lock` refuses a second sweep against
  the same results file, because two sweeps are two concurrent loads.
* **resumable.** A point whose label is recorded as completed (or as the skip below) is
  never redone, and a duplicate label is refused rather than overwritten the way
  `quality_gate.py:118` refuses it.
* **loud.** A failed stage, a `gsq` chain that stopped on its drift guard
  (`run.py:703-711`), or an export that does not fit the disk stops the sweep with a
  message and a non-zero exit. D8 (:378, :725) is explicit that an unpruned run cannot
  finish, so free space is checked with `shutil.disk_usage` for the whole next stage -
  blocks x the per-block budget, plus the exporter's own `2 x estimate + reserve`
  (`export_plan.py:50,195`) - before the stage starts rather than mid-block.

`run.main` refuses a non-empty output directory (`run.py:926`), so every point gets a
fresh one; and a `gsq` config without `gsq_memory_mode: block_cpu_offload` builds its
input cache ~11x slower (`make_run_configs.py:199-204`), so the epoch arms derive from
the chain's own config of record instead of being rebuilt. D8 (:414) also means a curve
point wants a second full model in memory: do not run this beside a training process.

What it deliberately does not do: an epoch arm changes exactly one key of the chain's
`gsq` config (`gsq_epochs`), it never deletes an artifact, it never ranks the curve (that
is `quality_gate.py compare`, over the shared ledger whose arms this sweep appends to) and
it never runs the app suite (`run_mac_pipeline.py` owns that half of D10).

Usage (from tools/calibration):

    # what the default sweep would execute, executing nothing:
    PYTHONPATH=. .venv/bin/python solver/tools/budget_curve.py \\
        --chain-output runs/native-mps/validate-new-objective/stage-run-4 --dry-run

    # the sweep itself (hours, one arm at a time):
    PYTHONPATH=. .venv/bin/python solver/tools/budget_curve.py \\
        --chain-output runs/native-mps/validate-new-objective/stage-run-4

    # rank what it recorded, against the bar already in the ledger:
    PYTHONPATH=. .venv/bin/python solver/tools/quality_gate.py compare

A depth cutoff past the chain's trained prefix would measure the chain's *RTN initial*
candidates for the layers beyond it, not trained ones, so such a point is refused unless
`--allow-untrained-layers` says that is the arm you want.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: tools/calibration/solver/tools/ -> tools/calibration/ (the cwd every command runs in)
CALIBRATION = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = CALIBRATION / "runs" / "native-mps" / "budget-curve"
#: quality_gate.DEFAULT_TEXT (:39): the 8 training-excluded invocations' rendered prompts.
DEFAULT_TEXT = Path("/tmp/gsq-heldout.txt")
#: quality_gate.DEFAULT_LEDGER: the house ledger, which already holds Bonsai's 7.0653 bar.
DEFAULT_LEDGER = CALIBRATION / "runs" / "quality-arms.json"
#: The pinned Prism converter source: the Dockerfile stages `.context/prism/` at
#: `/opt/prism/`, and `export_qwen.py:29` verifies the receipt in its root.
DEFAULT_PRISM = CALIBRATION / "cuda_runtime" / ".context" / "prism"
PRISM_REVISION = "62061f91088281e65071cc38c5f69ee95c39f14e"  # export_qwen.PRISM (:19)
PER_BLOCK_GIB = 60.0
EXPORT_RESERVE_BYTES = 10 * 1024**3  # export_plan.RESERVE_BYTES (:50)
DEFAULT_DEPTHS = "1,2,4,8"
DEFAULT_EPOCHS = "1,2,4"
SCHEMA = "budget-curve/1"
GIB = 1024**3
PLAN_ESTIMATE = re.compile(r"estimated\s+([0-9.]+)\s*GB serialized")
DONE_STATUSES = ("completed", "skipped")


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_list(value: str) -> list[int]:
    """A comma-separated list of positive ints, for --layers/--epochs."""
    try:
        items = [int(part) for part in value.replace(" ", "").split(",") if part != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {value!r}")
    if not items:
        raise argparse.ArgumentTypeError("empty list")
    if any(item < 1 for item in items):
        # A zero cutoff is the all-bf16 control arm, which the depth curve is not: it
        # shares no quantized tensor with any other point.
        raise argparse.ArgumentTypeError(f"every value must be >= 1, got {value!r}")
    return items


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SystemExit(f"No such file: {path}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"Not valid JSON: {path}: {error}")


def write_json_atomic(path: Path, value):
    """The write pattern quality_gate.write_ledger uses: temp file, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def load_results(path: Path) -> dict:
    if not Path(path).is_file():
        return {"points": []}
    document = read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("points"), list):
        raise SystemExit(f"Results file must be an object with a 'points' list: {path}")
    return document


def load_ledger(path: Path) -> list:
    if not Path(path).is_file():
        return []
    arms = read_json(path)
    if not isinstance(arms, list):
        raise SystemExit(f"Perplexity ledger must be a list of arms: {path}")
    return arms


def free_bytes(anchor: Path) -> int:
    """Free space on the volume that will hold `anchor` (walk up to something existing)."""
    probe = Path(anchor)
    while not probe.exists():
        if probe.parent == probe:
            raise SystemExit(f"Cannot find an existing directory above {anchor}")
        probe = probe.parent
    return shutil.disk_usage(probe).free


def count_block_dirs(output: Path) -> int:
    """Trained blocks in a chain output, for a report that does not carry the count.

    `gsq_run` writes `blocks_completed` (:740) only on the blocked_stop path; the
    `completed` path (:743) has none, and its output holds one `block-<i>/` per exported
    block.
    """
    return sum(1 for child in Path(output).glob("block-*")
               if child.is_dir() and any(child.glob("*.safetensors")))


def epochs_patch(label: str, point_dir: Path, shallow: int, epochs: int,
                 deadline_hours: float) -> dict:
    """The operational keys one epoch arm changes; nothing scientific is touched."""
    return {
        "attempt_id": label,
        "output": str(Path(point_dir) / "stage"),
        "checkpoint": {"backend": "local", "path": str(Path(point_dir) / "checkpoints")},
        "gsq_max_blocks": shallow,
        "gsq_epochs": epochs,
        "deadline_unix": int(time.time() + deadline_hours * 3600),
    }


def derived_gsq_config(chain_config: dict, patch: dict) -> dict:
    """The chain's config of record with `patch` applied, as a fresh `gsq` attempt.

    `identity` is dropped because `bind_identity` COMPUTES it and raises if a supplied one
    disagrees with the new `gsq_epochs` (see the NOTE at the end of make_run_configs.build),
    and `resume_checkpoint` because a fresh point must not read another attempt's store.
    `gsq_memory_mode: block_cpu_offload` therefore survives by inheritance, which is the
    point: a config built from scratch without it builds its input cache ~11x slower
    (make_run_configs.py:199-204).
    """
    derived = {key: value for key, value in chain_config.items()
               if key not in ("identity", "resume_checkpoint")}
    derived.update(patch)
    return derived


class Chain:
    """A completed `gsq` chain: what the depth arms cut, and what the epoch arms seed from."""

    def __init__(self, args):
        self.output = args.chain_output
        self.config_path = args.chain_config or self.output / "frozen-config.json"
        self.config = read_json(self.config_path)
        self.report_path = self.output / "gsq-report.json"
        self.report = read_json(self.report_path)
        self.status = self.report.get("status")
        self.stop_reason = self.report.get("stop_reason")
        blocks = self.report.get("blocks_completed")
        self.trained_blocks = int(blocks) if isinstance(blocks, int) else count_block_dirs(self.output)
        # `gsq_epochs` is absent from a config that used the default, and it is bound into
        # the chain identity (run.py:187-190), so this is the chain's own recipe.
        self.epochs = int(self.config.get("gsq_epochs") or 1)
        self.database = args.candidate_database or self.output / "candidate-database.json"
        if not self.database.is_file():
            raise SystemExit(
                f"No candidate database at {self.database}; the depth arms cut the chain's "
                f"trained candidates and there is nothing to cut")
        configured = self.config.get("model_dir")
        if not configured and self.config.get("inputs"):
            configured = Path(self.config["inputs"]) / "model"
        self.model_dir = Path(args.model_dir or configured) if (args.model_dir or configured) else None
        if self.model_dir is None or not self.model_dir.is_dir():
            raise SystemExit(
                f"No original model directory (--model-dir, or model_dir in {self.config_path})")
        # The epoch arms start from the chain's own seed: the RTN initial database that
        # `initialize` wrote and that every attempt derived from this config used.
        self.seed_database = self.config.get("candidate_database")
        if not self.seed_database or not Path(self.seed_database).is_file():
            raise SystemExit(
                f"The chain config {self.config_path} names no usable seed database "
                f"(candidate_database={self.seed_database!r}); the epoch arms need one")

    def drift(self) -> list | None:
        """D3.1's per-block composed-stream record, written by `run.py:666-671`."""
        path = self.output / "drift.json"
        return read_json(path) if path.is_file() else None


class Driver:
    """Prints (and in a dry run only prints) one ordered sequence of steps per point."""

    def __init__(self, args, dry: bool):
        self.args = args
        self.dry = dry
        self.env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                ["."] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
        }
        self.counter = 0
        self.journal: list[dict] = []
        self.steps: list[dict] = []
        self.costs: dict[str, float] = {}
        self.estimates: dict[str, int | None] = {}

    # -- plumbing ---------------------------------------------------------
    def begin(self, label: str):
        self.journal = []
        self.costs = {}
        print(f"\n=== {label} ===", flush=True)

    def step(self, text: str, argv=None, cost: str | None = None):
        self.counter += 1
        entry = {"index": self.counter, "text": text, "argv": argv, "cost": cost}
        self.steps.append(entry)
        self.journal.append(entry)
        print(f"{self.counter:>4}  {text}", flush=True)

    def command(self, argv, what: str, cost: str | None = None, capture: bool = False):
        rendered = [str(part) for part in argv]
        self.step(shlex.join(rendered), argv=rendered, cost=cost)
        if self.dry:
            return None
        started = time.monotonic()
        completed = subprocess.run(rendered, cwd=CALIBRATION, env=self.env,
                                   capture_output=capture, text=True)
        if cost:
            self.costs[cost] = round(time.monotonic() - started, 1)
        if completed.returncode != 0:
            if capture:
                output = (completed.stdout or "") + (completed.stderr or "")
                detail = ":\n" + "\n".join(output.strip().splitlines()[-12:])
            else:
                detail = "; its output is on the terminal above"
            raise SystemExit(f"{what} failed (exit {completed.returncode}){detail}")
        return completed

    def check_disk(self, need_bytes: int | None, what: str, anchor: Path):
        """Refuse to start a stage the volume cannot hold (D8 :725).

        `need_bytes is None` is the dry run's export case: the requirement is the plan's own
        `2 x estimate + reserve`, which only the (unexecuted) plan step could report.
        """
        free = free_bytes(anchor)
        if need_bytes is None:
            self.step(f"check-disk [need = 2 x the plan's serialized estimate + "
                      f"{EXPORT_RESERVE_BYTES / GIB:.1f} GiB reserve, from its stdout] "
                      f"free={free / GIB:.1f} GiB", argv=["check-disk", str(anchor)])
            return
        verdict = "ok" if free >= need_bytes else "INSUFFICIENT"
        self.step(f"check-disk need={need_bytes / GIB:.1f} GiB free={free / GIB:.1f} GiB "
                  f"{verdict} ({what})", argv=["check-disk", str(anchor)])
        if self.dry or free >= need_bytes:
            return
        raise SystemExit(
            f"{what}: {need_bytes / GIB:.1f} GiB needed, {free / GIB:.1f} GiB free on the "
            f"volume holding {anchor}.\nD8 (:725): an unpruned run cannot finish, and a full "
            f"root volume takes the machine's other work down with it. Reclaim output or "
            f"superseded checkpoint stores, or point --work-root at a larger volume.")

    def fresh_directory(self, path: Path, what: str):
        """`run.main` (:926) and the exporter (:57) both refuse a non-empty output dir."""
        occupied = Path(path).exists() and any(Path(path).iterdir())
        self.step(f"check-fresh {path}" + (" (OCCUPIED)" if occupied else ""),
                  argv=["check-fresh", str(path)])
        if occupied and not self.dry:
            raise SystemExit(
                f"{what} {path} is not empty, and both `solver.run` and "
                f"`packing.export_qwen` refuse one. That point is half-written and was never "
                f"recorded, so delete {path} and re-run to redo that point.")

    # -- steps ------------------------------------------------------------
    def plan_step(self, label: str, database: Path, cutoff: int, plan_path: Path) -> int | None:
        completed = self.command(
            [sys.executable, "solver/tools/export_plan.py",
             "--candidate-database", database,
             "--model-dir", self.chain.model_dir,
             "--layers", cutoff,
             "--output", plan_path,
             "--disk-reserve-bytes", self.args.export_reserve_bytes,
             "--purpose", f"D4/D5 budget curve point {label}: layers 0..{cutoff - 1} Q1"],
            "export plan", cost="plan", capture=True)
        if completed is None:
            return None
        match = PLAN_ESTIMATE.search(completed.stdout or "")
        if match is None:
            raise SystemExit(
                f"export_plan printed no serialized estimate, so the export's disk "
                f"requirement cannot be sized:\n{(completed.stdout or '')[-600:]}")
        return math.ceil(float(match.group(1)) * 1e9)

    def export_step(self, plan_path: Path, export_dir: Path) -> Path:
        self.fresh_directory(export_dir, "export directory")
        self.command([sys.executable, "-m", "packing.export_qwen",
                      "--plan", plan_path, "--output-dir", export_dir,
                      "--prism-source", self.args.prism_source],
                     "export", cost="export")
        return export_dir / "model.gguf"

    def measure_step(self, label: str, gguf: Path) -> dict:
        """One arm, one process: D10 (:460) on why these never overlap."""
        existing = next((arm for arm in self.ledger if arm.get("label") == label), None)
        if existing is not None:
            # Crash between `measure` and the results write: adopt the measured arm rather
            # than re-measuring it, which the ledger's duplicate-label refusal forbids.
            if existing.get("text") != str(self.args.text):
                print(f"     WARNING: the ledger's {label} was measured on "
                      f"{existing.get('text')}, not {self.args.text}", flush=True)
            self.step(f"[adopt] {label} is already in {self.args.ledger} with PPL "
                      f"{existing.get('perplexity')}", argv=["adopt", label])
            return existing
        if not self.dry and not Path(gguf).is_file():
            raise SystemExit(f"Export wrote no GGUF at {gguf}")
        self.command([sys.executable, "solver/tools/quality_gate.py", "measure",
                      "--gguf", gguf, "--label", label,
                      "--ledger", self.args.ledger, "--text", self.args.text,
                      "--context", self.args.context, "--ngl", self.args.ngl],
                     "perplexity measurement", cost="measure")
        if self.dry:
            return {}
        arms = load_ledger(self.args.ledger)
        arm = next((item for item in arms if item.get("label") == label), None)
        if arm is None:
            raise SystemExit(f"quality_gate recorded no arm labelled {label} in {self.args.ledger}")
        return arm

    # -- points -----------------------------------------------------------
    def assemble(self, label: str, kind: str, cutoff: int, epochs: int, plan_path: Path,
                 export_dir: Path, database: Path, trained_blocks: int,
                 drift: list | None, drift_source: Path | None, gsq: dict | None) -> dict:
        arm = self.measure_step(label, export_dir / "model.gguf")
        return {
            "label": label,
            "kind": kind,
            "layers": cutoff,
            "epochs": epochs,
            "status": "completed",
            "candidate_database": str(database),
            "trained_blocks": trained_blocks,
            "untrained_layers": max(0, cutoff - trained_blocks),
            "plan": str(plan_path),
            "plan_estimated_bytes": self.estimates.get(label),
            "gguf": arm.get("gguf") or str(export_dir / "model.gguf"),
            "gguf_bytes": arm.get("gguf_bytes"),
            "gguf_sha256": arm.get("gguf_sha256"),
            "perplexity": arm.get("perplexity"),
            "perplexity_stderr": arm.get("perplexity_stderr"),
            "context": self.args.context,
            "ngl": self.args.ngl,
            "text": arm.get("text"),
            "text_sha256": arm.get("text_sha256"),
            "drift": drift,
            "drift_source": str(drift_source) if drift_source else None,
            "drift_stop": (drift[-1].get("stopped_by") if drift else None),
            "gsq": gsq,
            "cost_seconds": {**self.costs, "total": round(sum(self.costs.values()), 1)},
            "commands": [entry["argv"] for entry in self.journal if entry["argv"]],
            "recorded_unix": int(time.time()),
        }

    def expose(self, label: str, database: Path, cutoff: int, point_dir: Path) -> tuple[Path, Path]:
        """The plan and export of one point: fresh plan, fresh export directory."""
        plan_path = point_dir / "plan.json"
        self.estimates[label] = self.plan_step(label, database, cutoff, plan_path)
        estimate = self.estimates[label]
        self.check_disk(None if estimate is None else 2 * estimate + self.args.export_reserve_bytes,
                        "export", point_dir)
        return plan_path, self.export_step(plan_path, point_dir / "export")

    def depth_arm(self, cutoff: int) -> dict:
        label = f"{self.args.label_prefix}-d{cutoff}"
        chain = self.chain
        self.begin(label)
        untrained = max(0, cutoff - chain.trained_blocks)
        if untrained and not self.args.allow_untrained_layers:
            if self.dry:
                self.step(f"[REFUSE] {label}: {untrained} layer(s) past the chain's "
                          f"{chain.trained_blocks} trained blocks hold RTN initial candidates, "
                          f"not trained ones; pass --allow-untrained-layers to measure that arm",
                          argv=["refuse", label])
                return {"label": label, "kind": "depth", "layers": cutoff,
                        "epochs": chain.epochs, "status": "refused", "untrained_layers": untrained,
                        "reason": "cutoff exceeds the chain's trained prefix"}
            raise SystemExit(
                f"{label}: cutoff {cutoff} exceeds the chain's {chain.trained_blocks} trained "
                f"blocks, so layers {chain.trained_blocks}..{cutoff - 1} would be the untrained "
                f"RTN initial candidates and the point would not measure training at depth.\n"
                f"Train the chain to at least {cutoff} blocks, cut at {chain.trained_blocks} or "
                f"less, or pass --allow-untrained-layers to measure the RTN arm deliberately.")
        point_dir = self.args.work_root / label
        if not self.dry:
            point_dir.mkdir(parents=True, exist_ok=True)
        plan_path, gguf = self.expose(label, chain.database, cutoff, point_dir)
        return self.assemble(label, "depth", cutoff, chain.epochs, plan_path, gguf.parent,
                             chain.database, chain.trained_blocks, chain.drift(),
                             chain.output, None)

    def epochs_arm(self, epochs: int) -> dict:
        label = f"{self.args.label_prefix}-e{epochs}"
        chain, shallow = self.chain, self.args.shallow_layers
        self.begin(label)
        same_as = f"{self.args.label_prefix}-d{shallow}"
        if (chain.epochs == epochs and shallow in self.args.layers
                and chain.trained_blocks >= shallow):
            # The chain IS this point: `gsq_epochs` is bound into the chain identity
            # (run.py:187-190) along with the seed and the seed database, so re-running it
            # would reproduce the depth arm's own artifact hours later, under a second
            # label, and put one artifact twice on the curve.
            self.step(f"[skip] {label}: the chain already trained {chain.trained_blocks} blocks "
                      f"at gsq_epochs={chain.epochs}, so this point is {same_as}",
                      argv=["skip", label])
            return {"label": label, "kind": "epochs", "layers": shallow, "epochs": epochs,
                    "status": "skipped", "same_as": same_as,
                    "reason": f"the chain is this point (gsq_epochs={chain.epochs}, "
                              f"trained_blocks={chain.trained_blocks})",
                    "recorded_unix": int(time.time())}
        point_dir = self.args.work_root / label
        if not self.dry:
            point_dir.mkdir(parents=True, exist_ok=True)
        config_path = point_dir / "config.json"
        output = point_dir / "stage"
        patch = epochs_patch(label, point_dir, shallow, epochs, self.args.deadline_hours)
        derived = derived_gsq_config(chain.config, patch)
        # The refusals come before the write: a sweep that stops on space, or on a
        # half-written directory, leaves no config behind to look like a started point.
        self.fresh_directory(output, "stage output directory")
        need = int(shallow * self.args.per_block_gib * GIB) + self.args.export_reserve_bytes
        self.check_disk(need, f"{shallow} gsq blocks at {self.args.per_block_gib:.0f} GiB each",
                        point_dir)
        self.step(f"write-config {config_path} " + json.dumps(patch, sort_keys=True),
                  argv=["write-config", str(config_path), json.dumps(patch, sort_keys=True)])
        if not self.dry:
            write_json_atomic(config_path, derived)
        self.command([sys.executable, "-m", "solver.run", "--config", config_path,
                      "--stage", "gsq"], "gsq stage", cost="gsq")
        if self.dry:
            report, gsq = None, None
        else:
            report = read_json(output / "gsq-report.json")
            self.report_guard(label, report, shallow)
            gsq = {"status": report.get("status"), "stop_reason": report.get("stop_reason"),
                   "blocks_completed": report.get("blocks_completed"),
                   "report": str(output / "gsq-report.json")}
        # `gsq_run` writes the arm's candidates here (the per-block atomic_json and :743), so
        # the plan step's input is deterministic and the dry run prints the same sequence a
        # real run executes.
        database = output / "candidate-database.json"
        plan_path, gguf = self.expose(label, database, shallow, point_dir)
        drift = None
        if not self.dry:
            drift_path = output / "drift.json"
            drift = read_json(drift_path) if drift_path.is_file() else None
        return self.assemble(label, "epochs", shallow, epochs, plan_path, gguf.parent, database,
                             shallow, drift, output, gsq)

    def report_guard(self, label: str, report: dict, shallow: int):
        """A `gsq` stage that did not train its subset, or tripped its guard, ends the sweep."""
        status, stop, nxt = report.get("status"), report.get("stop_reason"), report.get("next_block")
        if stop == "drift":
            drift = report.get("drift") or {}
            raise SystemExit(
                f"{label}: the chain's drift guard tripped ({drift.get('stopped_by')}) at block "
                f"{drift.get('block')} - norm_ratio {drift.get('norm_ratio')} (max "
                f"{drift.get('norm_ratio_max')}), growth {drift.get('growth')}, cosine "
                f"{drift.get('cosine')}.\nD3.1 stops a chain at a block boundary because the "
                f"composed stream's inflation is not a number worth exporting; measuring this "
                f"arm anyway would put a known-bad artifact on the curve.")
        if status == "completed" or (status == "blocked_stop" and nxt == shallow):
            return
        raise SystemExit(
            f"{label}: gsq stopped with status {status!r} stop_reason {stop!r} next_block {nxt} "
            f"(expected blocked_stop at block {shallow}, or completed).\nThe stage's output, "
            f"store and frozen config are left in place for inspection; the sweep stops here "
            f"rather than measuring an arm whose training did not finish.")

    def record(self, point: dict):
        shown = point.get("perplexity")
        if shown is not None:
            summary = (f"PPL {shown:.4f} +/- {point['perplexity_stderr']:.4f}, "
                       f"{point['gguf_bytes'] / 1e9:.2f} GB, "
                       f"{point['cost_seconds']['total']}s")
        elif point.get("reason"):
            summary = f"{point['status']}: {point['reason']}"
        else:
            summary = "planned (PPL, bytes and cost come from the executed steps)"
        if self.dry:
            print(f"     would record {point['label']} ({summary})", flush=True)
            return
        if any(item["label"] == point["label"] for item in self.results["points"]):
            raise SystemExit(
                f"Label {point['label']!r} is already in {self.args.results}. Use a new "
                f"--label-prefix or a new --results file rather than overwriting a measurement.")
        self.results["points"].append(point)
        write_json_atomic(self.args.results, self.results)
        print(f"     recorded {point['label']}: {summary}", flush=True)

    # -- the sweep --------------------------------------------------------
    def run(self):
        args = self.args
        self.chain = Chain(args)
        self.results = load_results(args.results)
        self.ledger = load_ledger(args.ledger)
        text_sha = sha256_of(args.text) if args.text.is_file() else None
        self.instrument_guard(text_sha)
        if "schema" not in self.results:
            self.results.update({
                "schema": SCHEMA,
                "design": "docs/QUANTIZATION_QUALITY_PIPELINE_2026-09-20.md section 3 "
                          "D4/D5/D8, section 5",
                "created_unix": int(time.time()),
                "curve": {"layers": args.layers, "epochs": args.epochs,
                          "shallow_layers": args.shallow_layers,
                          "text": str(args.text), "text_sha256": text_sha,
                          "context": args.context, "ngl": args.ngl,
                          "ledger": str(args.ledger), "label_prefix": args.label_prefix,
                          "per_block_gib": args.per_block_gib,
                          "export_reserve_bytes": args.export_reserve_bytes,
                          "prism_revision": PRISM_REVISION},
                "chain": {"output": str(self.chain.output),
                          "config": str(self.chain.config_path),
                          "report": str(self.chain.report_path),
                          "status": self.chain.status,
                          "stop_reason": self.chain.stop_reason,
                          "trained_blocks": self.chain.trained_blocks,
                          "epochs": self.chain.epochs,
                          "candidate_database": str(self.chain.database),
                          "seed_database": str(self.chain.seed_database)},
                "points": [],
            })
        plan = ([("depth", cutoff) for cutoff in args.layers]
                + [("epochs", epochs) for epochs in args.epochs])
        print(f"chain      {self.chain.output}  ({self.chain.trained_blocks} blocks trained, "
              f"gsq_epochs={self.chain.epochs}, status={self.chain.status}"
              f"{', stop_reason=' + str(self.chain.stop_reason) if self.chain.stop_reason else ''})")
        print(f"candidate  {self.chain.database}   seed {self.chain.seed_database}")
        print(f"model      {self.chain.model_dir}")
        print(f"text       {args.text}  "
              + (f"sha256 {text_sha[:16]}..." if text_sha else "MISSING - D10 (:426) measures "
                 "every arm on the held-out invocations' rendered prompts"))
        print(f"results    {args.results}   ledger {args.ledger}")
        print(f"work root  {args.work_root}   prism {args.prism_source}")
        print(f"limits     drift growth<={self.chain.config.get('gsq_max_drift_growth')} "
              f"ratio<={self.chain.config.get('gsq_max_drift_ratio')}, "
              f"{args.per_block_gib:.0f} GiB per gsq block, "
              f"export reserve {args.export_reserve_bytes / GIB:.1f} GiB")
        print(f"order      " + ", ".join(f"{kind}:{value}" for kind, value in plan)
              + f"   (shallow subset {args.shallow_layers} layers for the epoch arms)")
        print(f"cwd        all commands run in {CALIBRATION} with PYTHONPATH=.")
        if self.chain.stop_reason == "drift":
            print(f"WARNING: the chain itself stopped on its drift guard; only points inside its "
                  f"{self.chain.trained_blocks} trained blocks are curve points.")
        done = {item["label"]: item for item in self.results["points"]}
        for kind, value in plan:
            label = f"{args.label_prefix}-{'d' if kind == 'depth' else 'e'}{value}"
            if label in done and done[label].get("status") in DONE_STATUSES:
                print(f"\n=== {label} ===\n     already recorded "
                      f"({done[label].get('perplexity', done[label].get('status'))}): "
                      f"not redone", flush=True)
                continue
            self.record(self.depth_arm(value) if kind == "depth" else self.epochs_arm(value))
        print(f"\n{len(self.results['points'])} point(s) in {args.results}; rank them with "
              f"`PYTHONPATH=. .venv/bin/python solver/tools/quality_gate.py compare"
              f"{' --ledger ' + str(args.ledger) if args.ledger != DEFAULT_LEDGER else ''}`")
        if self.dry:
            commands = [entry for entry in self.steps if entry["argv"]
                        and str(entry["argv"][0]) == sys.executable]
            print(f"\n=== dry run: {len(self.steps)} steps, {len(commands)} commands, in order ===")
            for position, entry in enumerate(commands, start=1):
                print(f"{position:>4}  {shlex.join(entry['argv'])}")

    def instrument_guard(self, text_sha: str | None):
        """One curve is measured on one text at one context; mixing them is not a curve."""
        recorded = self.results.get("curve")
        if recorded is None:
            return
        for key, value in (("text_sha256", text_sha), ("text", str(self.args.text)),
                           ("context", self.args.context), ("ngl", self.args.ngl)):
            if recorded.get(key) == value:
                continue
            complaint = (
                f"{self.args.results} was measured with {key}={recorded.get(key)!r}, this run "
                f"has {value!r}. Perplexity across different text or context is not comparable "
                f"(quality_gate.compare warns about exactly this), so use a separate --results "
                f"file for a different instrument.")
            if not self.dry:
                raise SystemExit(complaint)
            # A dry run exists to be read, so it prints the verdict and the sequence a real
            # run would refuse to execute.
            self.step(f"[REFUSE] {complaint}", argv=["refuse", key])


def acquire_lock(results: Path):
    """One sweep per results file: two sweeps are two concurrent model loads (D10 :460)."""
    path = Path(str(results) + ".lock")
    if path.exists():
        holder = path.read_text().strip()
        pid = int(holder) if holder.isdigit() else None
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit(
                f"Another budget-curve sweep (pid {pid}) holds {path}. Two sweeps measure "
                f"concurrently, which is the load that hangs rather than fails (D10 :460).")
        print(f"reclaiming stale lock {path} from dead pid {holder}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n")


def release_lock(results: Path):
    path = Path(str(results) + ".lock")
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain-output", required=True, type=Path,
                        help="a completed `gsq` chain's output directory: its "
                             "candidate-database.json feeds the depth arms and its config "
                             "seeds the epoch arms")
    parser.add_argument("--chain-config", type=Path, default=None,
                        help="the chain's frozen config of record; defaults to "
                             "<chain-output>/frozen-config.json. The epoch arms derive from it "
                             "so that gsq_epochs is the only scientific key they change.")
    parser.add_argument("--candidate-database", type=Path, default=None,
                        help="the trained candidates the depth arms cut; defaults to "
                             "<chain-output>/candidate-database.json")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="the original unquantized model the plan must reference; "
                             "defaults to the chain config's model_dir")
    parser.add_argument("--layers", type=positive_list, default=positive_list(DEFAULT_DEPTHS),
                        help=f"depth cutoffs, quantizing layers 0..k-1 (default {DEFAULT_DEPTHS})")
    parser.add_argument("--epochs", type=positive_list, default=positive_list(DEFAULT_EPOCHS),
                        help="gsq_epochs values for the shallow-subset arms "
                             f"(default {DEFAULT_EPOCHS})")
    parser.add_argument("--shallow-layers", type=int, default=None,
                        help="the shallow subset D5's epoch arms train; defaults to the deepest "
                             "depth cutoff so both cuts meet at one point")
    parser.add_argument("--allow-untrained-layers", action="store_true",
                        help="measure cutoffs past the chain's trained prefix anyway, which "
                             "quantizes that chain's RTN initial candidates - the RTN-only arm, "
                             "not a point on the training curve")
    parser.add_argument("--results", type=Path, default=None,
                        help="the one JSON results file, appended atomically; defaults to "
                             "<work-root>/curve.json")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT,
                        help="where point directories, exports and checkpoint stores live")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                        help="quality_gate's perplexity ledger; the default is the house one, "
                             "so `quality_gate.py compare` ranks the curve against Bonsai's bar")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT,
                        help="the held-out text every arm is measured on (D10, :426)")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--prism-source", type=Path,
                        default=Path(os.environ.get("ZIMFO_PRISM_SOURCE") or DEFAULT_PRISM),
                        help=f"pinned Prism converter source (revision {PRISM_REVISION})")
    parser.add_argument("--per-block-gib", type=float, default=PER_BLOCK_GIB,
                        help=f"disk budget per gsq block (default {PER_BLOCK_GIB:.0f}): D8 "
                             f"(:387-389) measured 9.5 GB/block of store growth, and the "
                             f"per-block driver budgets ~26 GiB of store plus ~29 GiB of "
                             f"scratch per block")
    parser.add_argument("--export-reserve-bytes", type=int, default=EXPORT_RESERVE_BYTES,
                        help="the exporter's own reserve (export_plan.RESERVE_BYTES)")
    parser.add_argument("--deadline-hours", type=float, default=12.0,
                        help="deadline written into each epoch arm's derived gsq config")
    parser.add_argument("--label-prefix", default="budget-curve",
                        help="prefix for every point label; quality_gate refuses a duplicate "
                             "label, so use a new prefix for a different chain")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact ordered command sequence and execute nothing")
    args = parser.parse_args()

    args.chain_output = args.chain_output.resolve()
    args.work_root = args.work_root.resolve()
    args.results = (args.results or (args.work_root / "curve.json")).resolve()
    args.text = args.text.resolve()
    args.prism_source = args.prism_source.resolve()
    if args.shallow_layers is None:
        args.shallow_layers = max(args.layers)
    if args.shallow_layers < 1:
        raise SystemExit("--shallow-layers must be >= 1")
    for name, values in (("--layers", args.layers), ("--epochs", args.epochs)):
        if len(set(values)) != len(values):
            # A repeated value repeats its point label, and a label is measured once
            # (quality_gate.py:118). Refusing here costs nothing; discovering it after the
            # second arm has trained its shallow subset costs hours.
            raise SystemExit(f"{name} repeats a value ({values}); each value is one labelled "
                             f"point on the curve")
    for name in ("chain_config", "candidate_database", "model_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    if not args.dry_run:
        # The dry run prints what would happen even where the machine cannot do it yet;
        # a real run refuses to start on a missing instrument.
        if not args.text.is_file():
            raise SystemExit(f"No held-out text at {args.text} (D10 :426 measures every arm on it)")
        receipt = args.prism_source / "zimfo-prism-source.json"
        if not receipt.is_file():
            raise SystemExit(f"No Prism source receipt at {receipt}; pass --prism-source")
        revision = read_json(receipt).get("revision")
        if revision != PRISM_REVISION:
            raise SystemExit(f"Prism source is revision {revision}, the pinned export needs "
                             f"{PRISM_REVISION} (export_qwen.py:19)")
        acquire_lock(args.results)
    else:
        print("dry run: nothing is executed and nothing is written\n")
    try:
        Driver(args, dry=args.dry_run).run()
    finally:
        if not args.dry_run:
            release_lock(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
