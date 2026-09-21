#!/usr/bin/env python3
"""Reproduce a captured invocation from token IDs and diff against the capture.

Why this exists
---------------
The captured package contains two boundary tensors per invocation:
``first_block_input`` (the embedding output, before block 0's input layer norm)
and ``final_normalized_hidden``. They are **not** training input — the solver
generates its own teacher/student streams at run time from token IDs
(``run.py:321-335``). They are a **parity oracle**: if the ported model, loaded
from the original weights on MPS, reproduces these two tensors from the token IDs
alone, then the port's forward path matches whatever produced the capture.

That check is worth running before any multi-hour optimization, because it is the
only available test that validates the whole chain — weight loading, dtype,
attention implementation, rotary convention, hybrid block dispatch — against
known-good numbers.

Usage
-----
    cd tools/calibration
    PYTHONPATH=. .venv/bin/python solver/tools/check_capture_parity.py \\
        runs/mac-pipeline-v1-20260919/activations/invocation-000000 \\
        runs/qwen3.8-27b-original \\
        [--device mps|cpu]

Reports per-tensor max-abs and relative error for both boundaries, and exits
non-zero if either exceeds its tolerance. Tolerances are stated and justified
rather than tuned to pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

#: Tolerances in bf16 ULPs, not absolute units — see `compare`.
#:
#: `first_block_input` is a single embedding lookup with no arithmetic, so it
#: must be **bit-exact** (measured: 0.000e+00). Any nonzero value there means a
#: weight, dtype or tokenisation problem, which is why it gets almost no slack.
#:
#: `final_normalized_hidden` is 64 blocks of bf16 arithmetic through different
#: kernels. Measured on invocation-000000: mean 0.80 ulp, and the error grows
#: with token position (0.159 -> 0.401 across the sequence), which is the
#: signature of accumulation rather than a wrong path. Allow a small multiple of
#: one ulp for that accumulation; a genuine path error (wrong norm placement,
#: wrong rope convention) produces tens of ulps, as the double-norm variant of
#: this check did.
TOL_FIRST_BLOCK_ULPS = 0.01
TOL_FINAL_HIDDEN_ULPS = 3.0


def load_capture(invocation: Path):
    seq = json.loads((invocation / "sequence.json").read_text())
    inp = json.loads((invocation / "input.json").read_text())

    tokens = inp["tokenIDs"]
    if len(tokens) != seq["tokens"]:
        raise SystemExit(
            f"token count mismatch: input.json has {len(tokens)}, "
            f"sequence.json declares {seq['tokens']}"
        )

    boundaries = {}
    for name in ("first_block_input", "final_normalized_hidden"):
        entry = seq["files"].get(name)
        if entry is None:
            continue
        path = invocation / os.path.basename(entry["file"])
        with safe_open(str(path), framework="pt") as f:
            boundaries[name] = f.get_tensor("hidden")
    return tokens, boundaries, seq


def compare(name, got, want, tol_ulps, device):
    """Compare in units of bfloat16 ULPs at the tensor's own scale.

    An absolute tolerance is the wrong instrument here and would have to be
    tuned. The capture ran on **MLX** kernels; this runs on **PyTorch**, which
    falls back to reference implementations for the gated-delta and causal-conv
    paths (neither `flash-linear-attention` nor `causal_conv1d` is installed).
    Different kernels round differently in bf16, and the error accumulates
    through 64 blocks.

    So the meaningful question is not "is it close in absolute terms" but "is it
    within the precision the comparison can possibly achieve". bfloat16 has 8
    mantissa bits; one ULP at a value of magnitude m is ~m*2^-8. A mean error
    below one ULP means the two implementations agree to the last representable
    step, and any further agreement is not attainable without running the
    capture's kernels.
    """
    if got.shape != want.shape:
        print(f"  {name}: SHAPE MISMATCH got {tuple(got.shape)} want {tuple(want.shape)}")
        return False
    g = got.float().cpu()
    w = want.float().cpu()
    diff = (g - w).abs()
    scale = w.abs().max().clamp_min(1e-6).item()
    ulp = float(torch.finfo(torch.bfloat16).eps) * scale
    mean_ulps = diff.mean().item() / ulp
    max_ulps = diff.max().item() / ulp
    ok = mean_ulps <= tol_ulps
    flag = "OK  " if ok else "FAIL"
    print(
        f"  {flag} {name}: mean={diff.mean().item():.3e} max={diff.max().item():.3e}  "
        f"| mean={mean_ulps:.2f} ulp  max={max_ulps:.2f} ulp  "
        f"| 1 ulp = {ulp:.3e}  (tol {tol_ulps:.1f} ulp mean)"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("invocation", type=Path, help="an invocation-* directory")
    parser.add_argument("model_dir", type=Path, help="original BF16 weight directory")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    tokens, boundaries, seq = load_capture(args.invocation)
    print(f"invocation     : {args.invocation.name}")
    print(f"  conversation : {seq.get('conversation_id')}  turn {seq.get('turn')}")
    print(f"  tokens       : {len(tokens)}")
    print(f"  boundaries   : {sorted(boundaries)}")
    print(f"model          : {args.model_dir}")
    print(f"device         : {args.device}")
    print()

    # The solver's own device policy is authoritative for what is usable here.
    try:
        from solver.device_policy import resolve_device, describe

        device = resolve_device(args.device)
    except Exception:
        device = args.device
        describe = lambda d: d  # noqa: E731
    print(f"resolved device: {describe(device)}")

    from solver.qwen import load_original

    print("loading original weights (this reads ~52 GiB of shards)...")
    model = load_original(str(args.model_dir), torch.device(device))
    model.eval()

    ids = torch.tensor([tokens], device=device, dtype=torch.long)
    ok = True
    with torch.no_grad():
        if "first_block_input" in boundaries:
            # The capture saves `body.embed_tokens(token_array)` — the embedding
            # output BEFORE block 0's input layer norm.
            got = model.model.embed_tokens(ids)
            print("\nfirst_block_input (embedding output, pre-norm):")
            ok &= compare("first_block_input", got, boundaries["first_block_input"],
                          TOL_FIRST_BLOCK_ULPS, device)

        if "final_normalized_hidden" in boundaries:
            # `Qwen3_5TextModel.forward` applies `self.norm` itself — verified by
            # inspecting it: `Qwen3_5TextModel.forward applies self.norm: True`,
            # `Qwen3_5Model`/`Qwen3_5ForCausalLM`: False. `model.model` is the
            # TextModel, so its `last_hidden_state` is already the normalized
            # output. Applying the norm again here (as an earlier version of this
            # check did) gives std 1.97 instead of the capture's 1.83 — a
            # double-norm artifact in the CHECK, not in the model.
            out = model.model(inputs_embeds=model.model.embed_tokens(ids), use_cache=False)
            hidden = out.last_hidden_state
            print("\nfinal_normalized_hidden (all blocks, norm applied inside the model):")
            ok &= compare("final_normalized_hidden", hidden,
                          boundaries["final_normalized_hidden"],
                          TOL_FINAL_HIDDEN_ULPS, device)

    print()
    if ok:
        print("PARITY OK — the port reproduces the capture from token IDs alone.")
        print("Caveat: the capture ran on MLX 0.32.2 with `use_kernel=not training`.")
        print("A difference here is kernel-level unless it exceeds the stated tolerance.")
        return 0
    print("PARITY FAILED — do not start a production run on this port until resolved.")
    print("Most likely causes, in order: wrong norm placement, wrong rotary convention,")
    print("wrong attention implementation, or a dtype that is not bf16 end to end.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())