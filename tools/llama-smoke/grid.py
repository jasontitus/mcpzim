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
  .venv/bin/python grid.py --server-url http://127.0.0.1:8091 \\
    --server-model bonsai-ternary-27b --out GRID_RESULTS_BONSAI.md
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
    tool_format: str = "json"  # "json" (Gemma/Qwen) or "pythonic" (LFM2)


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
    # Official Google QAT release (Q4_0) — quantization-aware-trained, so the
    # 4-bit keeps near-BF16 quality (Python mlx eval: 9/9 vs PTQ's 5/9). Local
    # path = the HF-cache blob of google/gemma-4-E4B-it-qat-q4_0-gguf. See
    # GEMMA4_QAT_MTP.md for the full QAT/MTP findings.
    ModelSpec(
        key="gemma4-e4b-qat",
        quants=["Q4_0"],
        local_paths={
            "Q4_0": "/Users/jasontitus/.cache/huggingface/hub/"
                    "models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/"
                    "bb3b92e6f031fa438b409f898dd9f14f499a0cb0/"
                    "gemma-4-E4B_q4_0-it.gguf",
        },
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
                      "fine-tune/ft-out-qwen3.6-27b-unsloth-bsz2/"
                      "qwen3.6-27b-it-ft-iter500.Q4_K_M.gguf",
        },
    ),
    # Liquid LFM2.5 — 8.3B total / 1.5B active MoE (hybrid LIV-conv + GQA).
    # Native ChatML-ish turn markers + Pythonic tool-call body:
    #   <|tool_call_start|>[fn(arg='v', ...)]<|tool_call_end|>
    # llama.cpp surfaces the body without the markers, so the parser
    # accepts both forms.
    ModelSpec(
        key="lfm2.5-8b-a1b",
        repo="LiquidAI/LFM2.5-8B-A1B-GGUF",
        prefix="LFM2.5-8B-A1B",
        quants=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        tool_format="pythonic",
    ),
    ModelSpec(
        # train_v4_combined.jsonl uses Gemma-style "emit JSON block" tool
        # call format folded into the first user turn (Gemma 3 has no
        # system role). LFM2.5 was trained on that distribution here, so
        # at inference it emits JSON-fenced tool calls — match the eval
        # format to what we trained for.
        key="lfm2.5-8b-a1b-ft",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q4_K_M.gguf",
        },
        tool_format="json",
    ),
    ModelSpec(
        # Same dataset re-emitted by convert_to_lfm2_native.py: real
        # `system` role + Pythonic tool calls (`<|tool_call_start|>
        # [fn(arg='v')]<|tool_call_end|>`). Trained on pcgaming with
        # PEFT (3.74M adapter, attn + router + dense FFN; MoE experts
        # not adapted because PEFT's standard target_modules can't
        # reach into LFM2's stacked-expert representation).
        key="lfm2.5-8b-a1b-ft-native",
        quants=["Q4_K_M"],
        local_paths={
            "Q4_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-lfm2.5-native-pulled/"
                      "lfm2.5-8b-a1b-ft-native.Q4_K_M.gguf",
        },
        tool_format="pythonic",
    ),
    # Memory-tuning sweep: the same fine-tuned LFM2.5 at smaller weight
    # quants. LFM2.5's RSS is dominated by resident weights (8.3B total
    # MoE params — every expert stays in memory even though only 1.5B
    # are active per token), so the weight quant is the biggest memory
    # lever. Q4_K_M=4.9GB, Q3_K_L=~4.0GB, Q3_K_M=~3.7GB. Combine with
    # q4_0 KV to shave the cache too. Eval each (quant × KV) to map the
    # memory/accuracy frontier vs the 10/13 Gemma 3 4B FT baseline.
    # Paths produced by mem_sweep quantization of the v7 f16 GGUF.
    ModelSpec(
        key="lfm2.5-8b-a1b-ft-q3km",
        quants=["Q3_K_M"],
        local_paths={
            # 2026-06-10: ft-out-lfm2.5-8b/ was removed in the fine-tune
            # cleanup; the shipping v7-full artifact lives in -v7full.
            "Q3_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-lfm2.5-8b-v7full/"
                      "lfm2.5-8b-a1b-ft.Q3_K_M.gguf",
        },
        tool_format="json",
    ),
    # v7-full requantized WITH an importance matrix computed on our own
    # tool-call transcripts (llama-imatrix; "poor man's QAT"). Sweep of the
    # Q3_K_M→Q2_K gap where plain PTQ collapses (Q2_K = 3/13). 2026-06-10.
    ModelSpec(
        key="lfm2.5-v7-imx",
        quants=["IQ3_XXS", "IQ3_XS", "Q3_K_S", "IQ2_M", "Q3_K_M"],
        local_paths={
            q: "/Users/jasontitus/experiments/mcpzim/tools/fine-tune/"
               f"ft-out-lfm2.5-8b-v7full/imx/lfm2.5-8b-a1b-ft.imx.{q}.gguf"
            for q in ["IQ3_XXS", "IQ3_XS", "Q3_K_S", "IQ2_M", "Q3_K_M"]
        },
        tool_format="json",
    ),
    # v8hist = v7 dataset + 224 history-event chains (the `history` template,
    # targeting the french_revolution_chain miss — v7-full's only failing
    # scenario). Same 800-iter recipe, 2026-06-10.
    ModelSpec(
        key="lfm2.5-8b-a1b-ft-v8hist-q3km",
        quants=["Q3_K_M"],
        local_paths={
            "Q3_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-lfm2.5-8b-v8hist/"
                      "lfm2.5-8b-a1b-ft.Q3_K_M.gguf",
        },
        tool_format="json",
    ),
    ModelSpec(
        key="lfm2.5-8b-a1b-ft-q3kl",
        quants=["Q3_K_L"],
        local_paths={
            "Q3_K_L": "/Users/jasontitus/experiments/mcpzim/tools/"
                      "fine-tune/ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q3_K_L.gguf",
        },
        tool_format="json",
    ),
    ModelSpec(
        key="lfm2.5-8b-a1b-ft-q2k",
        quants=["Q2_K"],
        local_paths={
            "Q2_K": "/Users/jasontitus/experiments/mcpzim/tools/"
                    "fine-tune/ft-out-lfm2.5-8b/lfm2.5-8b-a1b-ft.Q2_K.gguf",
        },
        tool_format="json",
    ),
    # v6 memory variants (already CPU-quantized, in ft-out-lfm2.5-8b-v6/).
    # v6 = 10/13 baseline; these map its memory/accuracy frontier.
    ModelSpec(
        key="lfm2.5-v6-q3km",
        quants=["Q3_K_M"],
        local_paths={"Q3_K_M": "/Users/jasontitus/experiments/mcpzim/tools/"
                     "fine-tune/ft-out-lfm2.5-8b-v6/lfm2.5-8b-a1b-ft.Q3_K_M.gguf"},
        tool_format="json",
    ),
    ModelSpec(
        key="lfm2.5-v6-q2k",
        quants=["Q2_K"],
        local_paths={"Q2_K": "/Users/jasontitus/experiments/mcpzim/tools/"
                     "fine-tune/ft-out-lfm2.5-8b-v6/lfm2.5-8b-a1b-ft.Q2_K.gguf"},
        tool_format="json",
    ),
]


# (K type, V type) pairs. f16/f16 is the unquantized baseline.
KV_QUANTS: list[tuple[str, str]] = [
    ("f16",  "f16"),
    ("q8_0", "q8_0"),
    ("q4_0", "q4_0"),
]


# All scenarios currently defined in eval.py.
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
    "putin_biography_chain",
    "alamo_history_chain",
    "gravity_waves_creation",
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
            scenario: str, timeout_s: int = 600,
            server_url: Optional[str] = None,
            server_model: str = "local-model",
            max_turn_tokens: int = 2048,
            seed: int = 42,
            native_tools: bool = False,
            force_expected_tools: bool = False) -> Row:
    cmd = [str(VENV_PYTHON), str(EVAL_SCRIPT)]
    if server_url:
        cmd += ["--server-url", server_url, "--server-model", server_model]
    elif model.local_paths and quant in model.local_paths:
        cmd += ["--local-path", model.local_paths[quant]]
    else:
        fname = f"{model.prefix}-{quant}.gguf"
        cmd += ["--repo", model.repo, "--file", fname]
    cmd += [
        "--scenario", scenario,
        "--max-turn-tokens", str(max_turn_tokens),
        "--seed", str(seed),
        "--cache-type-k", kv[0], "--cache-type-v", kv[1],
        "--flash-attn",
        "--swa-full", "false",   # engage iSWA rotation-pruning — our
                                   # shipping config lever for both
                                   # Gemma 3 (homogeneous iSWA) and
                                   # Gemma 4 (heterogeneous iSWA via
                                   # PR #21513 attention rotation).
        "--tool-format", model.tool_format,
    ]
    if native_tools:
        cmd.append("--native-tools")
    if force_expected_tools:
        cmd.append("--force-expected-tools")
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
    ap.add_argument("--server-url", default="",
                    help="Evaluate one model already hosted by llama.cpp.")
    ap.add_argument("--server-model", default="local-model",
                    help="Display/API model id for --server-url.")
    ap.add_argument("--max-turn-tokens", type=int, default=2048,
                    help="Maximum tokens for each model response.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed for an external llama.cpp server.")
    ap.add_argument("--native-tools", action="store_true",
                    help="Use native OpenAI-style tool call round trips.")
    ap.add_argument("--force-expected-tools", action="store_true",
                    help="Require retrieval on scenario turns that expect it.")
    args = ap.parse_args()

    model_filt = [m.strip() for m in args.models.split(",") if m.strip()]
    quant_filt = [q.strip() for q in args.only.split(",") if q.strip()]
    scen_filt = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    kv_filt = [k.strip() for k in args.kv.split(",") if k.strip()]

    if args.server_url:
        models = [ModelSpec(key=args.server_model, quants=["server"])]
    else:
        models = [m for m in MODELS
                  if not model_filt or any(f in m.key for f in model_filt)]
    scenarios = [s for s in ALL_SCENARIOS
                 if not scen_filt or any(f in s for f in scen_filt)]
    if args.server_url:
        kv_opts = [("server", "server")]
    else:
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
        if args.server_url:
            fh.write("Peak RSS is the Python client only; model/server memory "
                     "is excluded.\n\n")
        fh.write("| model | quant | KV | scenario | pass | peak MB | ≥5GB | ≥6GB | wall s |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        fh.flush()

        for i, (m, q, kv, s) in enumerate(combos, 1):
            t0 = time.perf_counter()
            row = run_one(
                m, q, kv, s,
                server_url=args.server_url or None,
                server_model=args.server_model,
                max_turn_tokens=args.max_turn_tokens,
                seed=args.seed,
                native_tools=args.native_tools,
                force_expected_tools=args.force_expected_tools,
            )
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
