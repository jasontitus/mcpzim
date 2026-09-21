"""Feasibility probe, not a production RCO adapter.

Keep immutable weight/candidate tensors on CPU and tile their device transfers.
Preserve the upstream W_ref + sum(p_i * delta_i) surrogate, including gradients
for candidates whose straight-through forward probability is zero.
Only first-order differentiation is supported. Host storage is not disk-backed.
"""

import torch
from torch.autograd.function import once_differentiable


class _StreamedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probabilities, reference, deltas, tile_rows):
        if reference.device.type != "cpu" or any(d.device.type != "cpu" for d in deltas):
            raise ValueError("Weight storage must be on CPU")
        if tile_rows <= 0 or reference.ndim != 2 or not deltas:
            raise ValueError("Need a matrix, candidates, and positive tile size")
        if probabilities.shape != (len(deltas),) or any(d.shape != reference.shape for d in deltas):
            raise ValueError("Candidate/probability shape mismatch")
        if x.shape[-1] != reference.shape[1]:
            raise ValueError("Input width mismatch")
        if any(t.dtype != x.dtype for t in (reference, *deltas)):
            raise ValueError("Weights and inputs must have matching dtypes")
        if probabilities.dtype not in (x.dtype, torch.float32):
            raise ValueError("Probabilities must use input dtype or FP32")
        if probabilities.device != x.device:
            raise ValueError("Probabilities must share the input device")
        ctx.tile_rows = tile_rows
        # save_for_backward keeps version checks; these saved tensors are on CPU.
        ctx.save_for_backward(x.detach().to("cpu"), probabilities.detach().to("cpu"), reference, *deltas)
        out = x.new_empty((*x.shape[:-1], reference.shape[0]))
        for start in range(0, reference.shape[0], tile_rows):
            end = start + tile_rows
            weight = reference[start:end].to(x.device).clone()
            for p, delta in zip(probabilities, deltas):
                weight.add_(p * delta[start:end].to(x.device))
            out[..., start:end] = x @ weight.T
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x_cpu, p_cpu, reference, *deltas = ctx.saved_tensors
        x = x_cpu.to(grad_output.device)
        p = p_cpu.to(grad_output.device)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_g = grad_output.reshape(-1, grad_output.shape[-1])
        accumulation_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        grad_x = torch.zeros_like(flat_x, dtype=accumulation_dtype)
        grad_p = torch.zeros_like(p)
        for start in range(0, reference.shape[0], ctx.tile_rows):
            end = start + ctx.tile_rows
            g = flat_g[:, start:end]
            # Bound the temporary weight gradient to a row tile.
            grad_weight = g.T @ flat_x
            weight = reference[start:end].to(x.device).clone()
            for i, delta_cpu in enumerate(deltas):
                delta = delta_cpu[start:end].to(x.device)
                weight.add_(p[i] * delta)
                grad_p[i] += (grad_weight.to(accumulation_dtype) * delta.to(accumulation_dtype)).sum()
            grad_x.add_((g @ weight).to(accumulation_dtype))
        return grad_x.to(x.dtype).reshape(x.shape), grad_p, None, None, None


def streamed_linear(x, probabilities, reference, deltas, tile_rows=64):
    """Frozen CPU weights; differentiable input and candidate probabilities."""
    if reference.requires_grad or any(d.requires_grad for d in deltas):
        raise ValueError("Candidate weights must be frozen")
    return _StreamedLinear.apply(x, probabilities, reference, tuple(deltas), tile_rows)
