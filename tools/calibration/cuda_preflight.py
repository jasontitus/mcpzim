"""Synthetic CUDA feasibility checks. Never downloads model or app data.

The CPU tiny profile exercises report/failure handling and numerical code only.
A passed target profile establishes projection-level feasibility, not full RCO.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import torch

from streamed_linear_probe import streamed_linear


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dense(x, p, reference, deltas):
    weight = reference.to(x.device) + sum(p[i] * d.to(x.device) for i, d in enumerate(deltas))
    return x @ weight.T


def relative_error(actual, expected):
    a, b = actual.double(), expected.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite output or gradient")
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def parity(device, dtype):
    torch.manual_seed(320)
    reference = torch.randn(48, 32, dtype=dtype) / math.sqrt(32)
    deltas = tuple(torch.randn_like(reference) * .02 for _ in range(3))
    inputs = torch.randn(2, 5, 32, dtype=dtype)
    alpha_init = torch.tensor([.2, -.3, .5], dtype=torch.float32)
    results = []
    for streamed in (False, True):
        x = inputs.to(device).detach().requires_grad_()
        alpha = alpha_init.to(device).detach().requires_grad_()
        soft = alpha.softmax(-1)
        hard = torch.tensor([0., 1., 0.], device=device)
        p = hard - soft.detach() + soft
        y = streamed_linear(x, p, reference, deltas, 13) if streamed else dense(x, p, reference, deltas)
        loss = (y.float().tanh() - .2).square().mean()
        loss.backward()
        if alpha.grad is None or torch.count_nonzero(alpha.grad) != 3:
            raise AssertionError("Missing candidate gradient")
        results.append([y.detach().cpu(), x.grad.cpu(), alpha.grad.cpu()])
    errors = {name: relative_error(a, b) for name, a, b in zip(
        ("output", "input_gradient", "allocation_gradient"), results[1], results[0])}
    # Different tile reductions need a BF16 tolerance, but must not hide gross errors.
    limit = .05 if dtype == torch.bfloat16 else 1e-4
    if max(errors.values()) > limit:
        raise AssertionError(f"Parity failed: {errors}, limit={limit}")
    return {"dtype": str(dtype), "relative_l2_errors": errors, "limit": limit}


def measure_projection(device, tokens, width, output, iterations, tile_rows, limit_bytes):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(27)
    reference = torch.randn(output, width, dtype=dtype) / math.sqrt(width)
    deltas = tuple(torch.randn_like(reference) * (.02 / math.sqrt(width)) for _ in range(6))
    inputs = torch.randn(1, tokens, width, dtype=dtype)
    steps = []
    for iteration in range(iterations):
        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        x = inputs.to(device).detach().requires_grad_()
        alpha = torch.zeros(6, dtype=torch.float32, device=device, requires_grad=True)
        soft = alpha.softmax(-1)
        hard = torch.zeros_like(soft); hard[1] = 1
        p = hard - soft.detach() + soft
        y = streamed_linear(x, p, reference, deltas, tile_rows)
        loss = y.float().square().mean()
        loss.backward()
        if not torch.isfinite(x.grad).all() or not torch.isfinite(alpha.grad).all():
            raise AssertionError("Nonfinite BF16 backward")
        if not torch.count_nonzero(alpha.grad):
            raise AssertionError("All allocation gradients are zero")
        if device == "cuda": torch.cuda.synchronize()
        step = {"iteration": iteration, "seconds": time.monotonic() - started, "loss": loss.item()}
        if device == "cuda":
            step.update(peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            if max(step["peak_allocated"], step["peak_reserved"]) > limit_bytes:
                raise AssertionError(f"Exceeded memory ceiling: {step}")
        del x, alpha, soft, hard, p, y, loss
        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            step["live_after_release"] = torch.cuda.memory_allocated()
        steps.append(step)
    if device == "cuda" and steps[-1]["live_after_release"] > steps[0]["live_after_release"] + 16 * 1024**2:
        raise AssertionError("Live GPU allocations accumulated across steps")
    return {"tokens": tokens, "input_width": width, "output_width": output,
            "candidate_count": 7, "dtype": str(dtype), "tile_rows": tile_rows,
            "host_weight_bytes": reference.numel() * reference.element_size() * 7, "steps": steps}


def run(args):
    if args.profile == "target" and args.device != "cuda":
        raise ValueError("Target profile requires CUDA; CPU is not a substitute")
    report = {"schema": 1, "status": "running", "profile": args.profile, "device": args.device,
              "torch": torch.__version__, "platform": platform.platform(), "checks": [],
              "limitations": ["Synthetic projections, not full attention/recurrent blocks",
                              "No GSQ optimization, budget solver or quantized model produced",
                              "Host candidate storage is not yet disk-backed"]}
    write_report(args.output, report)
    try:
        if args.device == "cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
            if not torch.cuda.is_bf16_supported(): raise RuntimeError("CUDA BF16 unavailable")
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": props.name, "total_memory": props.total_memory,
                             "capability": list(torch.cuda.get_device_capability()), "cuda": torch.version.cuda}
            # Enforce allocator ceiling too; this does not bound non-PyTorch allocations.
            torch.cuda.set_per_process_memory_fraction(min(.90, args.max_gpu_gib * 1024**3 / props.total_memory))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout
        for dtype in ([torch.float32, torch.bfloat16] if args.device == "cuda" else [torch.float32]):
            report["checks"].append({"parity": parity(args.device, dtype)})
            write_report(args.output, report)
        cases = [(11, 32, 48), (17, 48, 32)] if args.profile == "tiny" else [
            (tokens, width, out) for tokens in (2048, 4096, 7330)
            for width, out in ((5120, 17408), (17408, 5120))]
        for tokens, width, out in cases:
            if args.device == "cuda": torch.cuda.empty_cache()
            measurement = measure_projection(args.device, tokens, width, out, args.iterations,
                                             args.tile_rows, args.max_gpu_gib * 1024**3)
            report["checks"].append({"projection": measurement})
            report["peak_host_rss_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["host_rss_unit"] = "bytes" if platform.system() == "Darwin" else "KiB"
            write_report(args.output, report)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--profile", choices=("tiny", "target"), default="target")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--tile-rows", type=int, default=256)
    parser.add_argument("--max-gpu-gib", type=float, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations < 3 or args.tile_rows < 1 or not 0 < args.max_gpu_gib <= 20:
        parser.error("Require at least three iterations, positive tiles, and a GPU ceiling at most 20 GiB")
    run(args)
