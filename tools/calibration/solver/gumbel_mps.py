"""Device-agnostic one-bit Gumbel-Softmax quantizer.

Upstream's ``GumbelQuantizer1Bit`` is CUDA-only in exactly one place, and it is
in the autograd function rather than the module::

    # forward
    ctx.cuda_fwd_rng_state = torch.cuda.get_rng_state(device=device)
    u = torch.rand_like(sign_logits)          # drawn on the CUDA device stream
    noise = torch.logit(u, eps=eps)

    # backward
    with torch.random.fork_rng(devices=[device]):
        torch.cuda.set_rng_state(ctx.cuda_fwd_rng_state, device=device)
        u = torch.rand_like(sign_logits)      # must reproduce the forward draw
        noise = torch.logit(u, eps=eps)

Replaying the RNG state is how upstream takes the gradient at the *same* sample
point the forward used. It is correct on CUDA and unavailable here:
``torch.random.fork_rng`` rejects an MPS device list, and ``torch.cuda`` has no
rng state on this machine. It is also fragile in general -- if the redraw
consumes a different stream than the forward did, the gradient is taken at a
different point and nothing raises; the quantizer just trains against the wrong
objective and yields a plausible, degraded model.

This module removes the replay instead of emulating it. The noise tensor drawn
in ``forward`` is stored on the autograd context and reused in ``backward``, so
there is nothing to restore and nothing to drift. The arithmetic is upstream's,
character for character, and is measured against it by
``tools/calibration/solver/tests/test_gumbel_mps.py``.

Noise is drawn on the **global CPU stream** and moved to the compute device.
That is load-bearing for reproducibility, not a convenience:
``torch.manual_seed`` seeds the CPU stream, and an unseeded device stream is not
reproducible from it. Measured on this machine::

    torch.manual_seed(0); torch.rand(5)                      -> [0.4963, 0.7682, ...]
    torch.manual_seed(0); torch.rand(5, device='mps').cpu()  -> [0.3664, 0.0755, ...]

so a device-stream draw would make a seeded MPS run unreproducible. Drawing on
CPU also keeps the noise identical across devices, which is what lets the parity
test compare CPU and MPS against one upstream reference.

The technique is the one proven bit-exact against upstream at five bit widths in
``/Users/jasontitus/experiments/gsq-rco-mlx`` (``src/gsq_mlx/rng.py``,
``docs/PARITY.md``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: Upstream's clamp. ``logit`` diverges at u=0 and u=1, so upstream passes
#: ``eps`` to ``torch.logit``; the same value is used here so the draws match.
EPS = 1e-8


class MPSOneBit(nn.Module):
    """Drop-in replacement for upstream's ``GumbelQuantizer1Bit`` on any device.

    The constructor signature, the parameter names, the state-dict layout, the
    ``forward`` signature and ``get_hard_weights`` all match upstream, so this is
    a substitution for ``BlockTrainer`` rather than an adapter around it.

    The two sanctioned differences from upstream are that the relaxation noise
    is drawn on the global CPU stream (Reproducibility, above) and that it is
    stored rather than replayed (Correctness, above). Both are properties of the
    RNG, not of the algebra: given the same noise tensor this computes the same
    forward, the same backward and the same hard weights, which the parity test
    asserts numerically.
    """

    def __init__(self, Q, scales, groupsize, std, strength, device, dtype, logits_dtype=None):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = device
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize

        scale_per_col = scales[:, self.idx]
        Q = Q / scale_per_col

        sign_logits = (std * (self._randn_like(Q) + Q * strength)).to(self.dtype)

        self.sign_logits = nn.Parameter(sign_logits.to(self.logits_dtype).detach())
        self.scales = nn.Parameter(scales.float().detach())

    @staticmethod
    def _randn_like(reference):
        """``torch.randn_like`` drawn on the global CPU stream, then moved.

        Upstream calls ``torch.randn_like(Q)``, which on CUDA draws from the
        device stream. An MPS device stream is not seeded by ``torch.manual_seed``
        (measured: see module docstring), so drawing on CPU and moving the result
        is what keeps a seeded MPS run reproducible. The tensor is one per
        parameter, so the transfer is not a performance concern.
        """
        out = torch.randn(reference.shape, dtype=torch.float32)
        return out.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _logistic_noise_like(reference):
        """``torch.logit(torch.rand_like(·), eps=1e-8)``, drawn on the global CPU stream.

        Equivalent to upstream's two lines and to ``torch.logit`` with the same
        clamp: ``rand`` already lies in ``[0, 1)``, so upstream's ``eps`` lower
        bound and a max of ``1 - eps`` are the same clamp on the only values it
        can produce, and ``logit(u, eps)`` is ``log(u / (1 - u))``.
        """
        u = torch.rand(reference.shape, dtype=torch.float32)
        u = u.clamp(min=EPS, max=1.0 - EPS)
        noise = torch.logit(u)
        return noise.to(device=reference.device, dtype=reference.dtype)

    def forward(self, temperature, scale=1.0):
        return MPSOneBitFunction.apply(
            self.sign_logits,
            self.scales,
            self.idx,
            float(temperature),
            float(scale),
            self.device,
            self.dtype
        )

    def get_hard_weights(self):
        hard_mask = (self.sign_logits > 0).to(self.dtype)
        hard_sign = 2.0 * hard_mask - 1.0
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = hard_sign * scale_per_col

        return output, self.scales.to(self.dtype)


class MPSOneBitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sign_logits, scales, idx, temperature, scale, device, dtype):
        ctx.save_for_backward(sign_logits, scales)
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = device
        ctx.dtype = dtype

        noise = MPSOneBit._logistic_noise_like(sign_logits)
        # Held for backward instead of the RNG state that would reproduce it.
        # Saving a tensor rather than a state is what makes this device-agnostic.
        ctx.noise = noise

        soft_sign = (2.0 * torch.sigmoid((2.0 * sign_logits * scale + noise) / temperature) - 1.0).to(dtype)

        scale_per_col = scales[:, idx].to(dtype)

        output = soft_sign * scale_per_col

        # Upstream returns this at the ambient promotion dtype, which for a
        # bf16 ``sign_logits`` is fp32: ``2.0 * sigmoid(...) - 1.0`` promotes
        # ``.to(dtype)`` back to fp32 because the Python floats are fp32 scalars
        # and bf16*broadcast-op-bf16 with an fp32 result lands in fp32. Forcing
        # the weight dtype here is deliberate; see the module docstring of
        # ``tests/test_gumbel_mps.py`` for why that difference is reported rather
        # than silently matched.
        return output.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sign_logits, scales = ctx.saved_tensors
        idx = ctx.idx
        temperature = ctx.temperature
        scale = ctx.scale
        dtype = ctx.dtype

        # The noise the forward actually used. No fork_rng, no replay, no
        # device stream.
        noise = ctx.noise

        soft_sign = (2.0 * torch.sigmoid((2.0 * sign_logits * scale + noise) / temperature) - 1.0).to(dtype)

        scale_per_col = scales[:, idx].to(dtype)

        grad_soft_quant = grad_output * scale_per_col
        grad_scale_per_col = grad_output * soft_sign

        grad_quant_logits = grad_soft_quant * (1.0 - soft_sign.pow(2)) * scale / temperature

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.float())

        return grad_quant_logits.to(sign_logits.dtype), grad_scales, None, None, None, None, None