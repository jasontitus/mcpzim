"""Full Mac grid runner — models × weight-quant × KV-quant × scenarios.

Runs each (model, weight-quant, kv-type, scenario) combo as a SUBPROCESS
invocation of eval.py so each run gets a fresh Python process with a
clean Metal pool. Parses the `RESULT ...` lines from the child and
aggregates into a markdown scorecard.

Why subprocess and not in-process: llama.cpp's `llama_backend_init()`
bumps a backend refcount, and `llama_free`/`llama_model_free` calls
leave the Metal heap fragmented across multiple Llama() instances.
Running each combo as its own process means peak-RSS numbers are
actually comparable — no carry-over from a prior load.

Usage:
  .venv/bin/python grid.py                        # full matrix
  .venv/bin/python grid.py --only Q4_K_M          # filter weight quants
  .venv/bin/python grid.py --scenarios bars_sc_caltrain_chain,sky_is_blue_chain
  .venv/bin/python grid.py --models gemma         # model substring
"""

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
VENV_PYTHON = HERE / ".venv" / "bin" / "python"
EVAL_SCRIPT = HERE / "eval.py"


@dataclasses.dataclass
class ModelSpec:
    key: str         # short id used in scorecard rows
    repo: Optional[str] = None   # HF repo; None for local-only
    prefix: Optional[str] = None # GGUF filename prefix (suffix = f"{prefix}-{quant}.gguf")
    quants: list[str] = dataclasses.field(default_factory=list)
    local_paths: Optional[dict[str, str]] = None  # {quant: absolute_path}


MODELS: list[ModelSpec] = [
    # Gemma 4 — heterogeneous iSWA with shorter SWA window (~1024)
    # than Gemma 3. swa_full=false engages the PR #21513 attention-
    # rotation path here (unique to Gemma 4 / hetero-iSWA models).
    # Text-only GGUFs (mmproj-* skipped at load time).
    ModelSpec(
        key="gemma4-e4b",
        repo="bartowski/google_gemma-4-E4B-it-GGUF",
        prefix="google_gemma-4-E4B-it",
        quants=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    ),
    ModelSpec(
        key="gemma4-e2b",
        repo="bartowski/google_gemma-4-E2B-it-GGUF",
        prefix="google_gemma-4-E2B-it",
        quants=["Q4_K_M", "Q5_K_M", "Q8_0"],
    ),
    ModelSpec(
        key="gemma3-4b",
        repo="bartowski/google_gemma-3-4b-it-GGUF",
        prefix="google_gemma-3-4b-it",
        quants=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    ),
    ModelSpec(
        key="qwen3.5-4b",
        repo="unsloth/Qwen3.5-4B-GGUF",
        prefix="Qwen3.5-4B",
        quants=["Q4_K_M", "Q5_K_M", "Q8_0"],
    ),
    ModelSpec(
        key="qwen3.5-2b",
        repo="unsloth/Qwen3.5-2B-GGUF",
        prefix="Qwen3.5-2B",
        quants=["Q4_K_M", "Q5_K_M", "Q8_0"],
    ),
    # LoRA-fine-tuned variants — produced by tools/fine-tune/train_all.sh
    # against the v4 dataset (train_v4_combined.jsonl, ~3000 examples
    # mixing single-turn + chains + grounded near_places). Each candidate
    # uses its own OUT_DIR so the GGUFs sit side-by-side.
    ModelSpec(
        key="gemma3-4b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-gemma3-4b/gemma3-4b-it-ft.Q4_K_M.gguf",
        },
    ),
    ModelSpec(
        key="gemma3-1b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-gemma3-1b/gemma3-1b-it-ft.Q4_K_M.gguf",
        },
    ),
    ModelSpec(
        key="qwen3-4b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-qwen3-4b/qwen3-4b-it-ft.Q4_K_M.gguf",
        },
    ),
    ModelSpec(
        key="qwen3-1.7b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-qwen3-1.7b/qwen3-1.7b-it-ft.Q4_K_M.gguf",
        },
    ),
    ModelSpec(
        key="qwen3.5-4b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-qwen3.5-4b/qwen3.5-4b-it-ft.Q4_K_M.gguf",
        },
    ),
    ModelSpec(
        key="qwen3-8b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-qwen3-8b/qwen3-8b-it-ft.Q4_K_M.gguf",
        },
    ),
    ModelSpec(
        key="qwen3.5-9b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-qwen3.5-9b/qwen3.5-9b-it-ft.Q4_K_M.gguf",
        },
    ),
    # Mac mlx-lm fine-tunes (single-machine experiments, ship-tier).
    # 2026-04-26: replaced the Mac mlx-lm 27B (was broken, 0/13) with the
    # pcgaming Unsloth iter-100 build. Run with
    # CHAT_TEMPLATE=/tmp/qwen36_patched_chat_template.jinja TOOL_ITER_BUDGET=8
    # in the env so eval.py disables thinking-mode and gives the model
    # enough tool-call budget to recover from fixture errors.
    ModelSpec(
        key="qwen3.6-27b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-qwen3.6-27b-unsloth/"
                      "qwen3.6-27b-it-ft-iter100.Q4_K_M.gguf",
        },
    ),
]


# (K type, V type) pairs. f16/f16 is the unquantized baseline.
KV_QUANTS: list[tuple[str, str]] = [
    ("f16",  "f16"),
    ("q8_0", "q8_0"),
    ("q4_0", "q4_0"),
]


# All 12 scenarios currently defined in eval.py.
ALL_SCENARIOS = [
    "bars_sc_caltrain_chain",
    "sky_is_blue_chain",
    "restaurants_in_sf",
    "nearby_stories_palo_alto",
    "tell_me_about_palo_alto",
    "compare_musk_bezos",
    "relations_us_iran",
    "narrate_hp_garage",
    "what_is_here_in_sf",
    "grav_waves_chain",
    "wwi_vs_wwii_chain",
    "french_revolution_chain",
    "crispr_chain",
]


RESULT_RE = re.compile(r"^RESULT (.+)$", re.MULTILINE)


@dataclasses.dataclass
class Row:
    model: str
    quant: str
    kv: str
    scenario: str
    passed: bool
    peak_mb: int
    wall_s: float
    ge5gb: int
    ge6gb: int
    error: Optional[str] = None


def parse_result(stdout: str) -> Optional[Row]:
    """Pull the RESULT lines from eval.py's output."""
    kv = {}
    for m in RESULT_RE.finditer(stdout):
        for tok in m.group(1).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
    if "peak_mb" not in kv:
        return None
    return Row(
        model="", quant="", kv="", scenario=kv.get("scenario", "?"),
        passed=kv.get("passed", "False") == "True",
        peak_mb=int(float(kv["peak_mb"])),
        wall_s=float(kv.get("wall_s", "0")),
        ge5gb=int(kv.get("ge5gb", "0")),
        ge6gb=int(kv.get("ge6gb", "0")),
    )


def run_one(model: ModelSpec, quant: str, kv: tuple[str, str],
            scenario: str, timeout_s: int = 600) -> Row:
    cmd = [str(VENV_PYTHON), str(EVAL_SCRIPT)]
    if model.local_paths and quant in model.local_paths:
        cmd += ["--local-path", model.local_paths[quant]]
    else:
        fname = f"{model.prefix}-{quant}.gguf"
        cmd += ["--repo", model.repo, "--file", fname]
    cmd += [
        "--scenario", scenario,
        "--cache-type-k", kv[0], "--cache-type-v", kv[1],
        "--flash-attn",
        "--swa-full", "false",   # engage iSWA rotation-pruning — our
                                   # shipping config lever for both
                                   # Gemma 3 (homogeneous iSWA) and
                                   # Gemma 4 (heterogeneous iSWA via
                                   # PR #21513 attention rotation).
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=HERE)
    except subprocess.TimeoutExpired:
        return Row(model.key, quant, "/".join(kv), scenario,
                   False, 0, timeout_s, 0, 0, error="timeout")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).splitlines()[-5:]
        return Row(model.key, quant, "/".join(kv), scenario,
                   False, 0, dt, 0, 0,
                   error=f"rc={proc.returncode}: {' / '.join(tail)}")
    row = parse_result(proc.stdout)
    if row is None:
        return Row(model.key, quant, "/".join(kv), scenario,
                   False, 0, dt, 0, 0, error="no RESULT in output")
    row.model = model.key
    row.quant = quant
    row.kv = "/".join(kv)
    return row


def fmt_markdown(rows: list[Row]) -> str:
    lines = [
        "| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pass_cell = "✓" if r.passed else ("·" if r.error else "✗")
        peak = str(r.peak_mb) if r.peak_mb else (r.error or "—")
        lines.append(
            f"| {r.model} | {r.quant} | {r.kv} | {r.scenario} | "
            f"{pass_cell} | {peak} | {r.ge5gb} | {r.ge6gb} | {r.wall_s:.1f} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="Comma-sep substring filter; empty = all")
    ap.add_argument("--only", default="",
                    help="Comma-sep quant filter, e.g. Q4_K_M,Q8_0")
    ap.add_argument("--scenarios", default="",
                    help=f"Comma-sep scenario filter; empty = all 12")
    ap.add_argument("--kv", default="",
                    help="Comma-sep KV filter (e.g. 'q8_0/q8_0')")
    ap.add_argument("--out", default="GRID_RESULTS.md")
    args = ap.parse_args()

    model_filt = [m.strip() for m in args.models.split(",") if m.strip()]
    quant_filt = [q.strip() for q in args.only.split(",") if q.strip()]
    scen_filt = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    kv_filt = [k.strip() for k in args.kv.split(",") if k.strip()]

    models = [m for m in MODELS
              if not model_filt or any(f in m.key for f in model_filt)]
    scenarios = [s for s in ALL_SCENARIOS
                 if not scen_filt or any(f in s for f in scen_filt)]
    kv_opts = [(k, v) for k, v in KV_QUANTS
               if not kv_filt or f"{k}/{v}" in kv_filt]

    combos = []
    for m in models:
        for q in m.quants:
            if quant_filt and q not in quant_filt:
                continue
            for kv in kv_opts:
                for s in scenarios:
                    combos.append((m, q, kv, s))
    print(f"grid: {len(combos)} combos "
          f"({len(models)} models × {len(kv_opts)} KV × {len(scenarios)} scenarios)")

    results: list[Row] = []
    out_path = HERE / args.out
    with open(out_path, "w") as fh:
        fh.write(f"# llama.cpp grid — {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        fh.write("Running sequentially — each combo is its own python "
                 "subprocess so peak-RSS numbers don't carry over.\n\n")
        fh.write("| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        fh.flush()

        for i, (m, q, kv, s) in enumerate(combos, 1):
            t0 = time.perf_counter()
            row = run_one(m, q, kv, s)
            dt = time.perf_counter() - t0
            results.append(row)
            pass_cell = "✓" if row.passed else ("·" if row.error else "✗")
            peak = str(row.peak_mb) if row.peak_mb else (row.error or "—")
            line = (f"| {row.model} | {row.quant} | {row.kv} | {row.scenario} | "
                    f"{pass_cell} | {peak} | {row.ge5gb} | {row.ge6gb} | "
                    f"{row.wall_s:.1f} |")
            fh.write(line + "\n"); fh.flush()
            print(f"[{i}/{len(combos)} · {dt:.0f}s] {line}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
