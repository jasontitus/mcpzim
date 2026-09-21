"""Numerical parity: ``gumbel_mps.MPSOneBit`` against upstream's CUDA quantizer.

Upstream's ``GumbelQuantizer1Bit`` is loaded from the pinned source tree at
``tools/calibration/cuda_runtime/.context/sources/gsq/`` and its two
``torch.cuda.*_rng_state`` calls are shimmed onto the CPU stream, which is the
only CUDA dependency in that file. Everything the shim touches is *which stream
the noise comes from*; none of the forward, backward or hard-weight algebra is
altered, so the comparison is of arithmetic and not of RNG plumbing.

The shim is required for the comparison to be meaningful at all: both
implementations must draw the *same* logistic noise for the two `soft_sign`
tensors to be comparable, and upstream's replay only reproduces its forward draw
if the shimmed state round-trips. ``CpuRngShim`` below is the technique from
``/Users/jasontitus/experiments/gsq-rco-mlx/scripts/compare_upstream.py``.

Recipes and results are documented in the module-level ``RESULTS`` string of
``tools/calibration/solver/tests/``'s sibling report, and measured numbers are
asserted in this file rather than quoted from elsewhere.

Two honest differences, both measured and reported rather than tolerance-washed:

1. **Noise source.** Upstream draws on the device stream; this port draws on the
   global CPU stream. That is the point of the port (see the module docstring
   of ``gumbel_mps.py``) and is what makes the two reproducible from one
   ``torch.manual_seed`` on any backend. The shim above removes this difference
   for the comparison, so what the numbers measure is arithmetic only.

2. **The forward dtype caveat did not reproduce in this configuration.** At
   ``logits_dtype=float32`` -- which is what ``create_quantizer`` passes --
   ``sign_logits`` is fp32, so upstream's ``(2.0 * sigmoid(...) - 1.0).to(dtype)``
   already yields bf16 and its ``scale_per_col`` is bf16. ``MPSOneBit``'s
   explicit ``.to(dtype)`` on the returned product is therefore a no-op *here*,
   not a dtype change, and the measured forward delta is ``0.0`` with no dtype
   mismatch. The cast is kept because the sibling port hit a real fp32-vs-bf16
   mismatch at ``logits_dtype=bfloat16``, where ``sign_logits`` is bf16 and the
   Python float literals in the sigmoid expression promote the result to fp32.
   The parameterised ``logits_dtype`` test below covers that configuration and
   is the one that would catch it.
"""
from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest
import torch

SOLVER = Path(__file__).resolve().parents[1]
REPO = SOLVER.parents[1]
UPSTREAM_Q = SOLVER.parent / 'cuda_runtime' / '.context' / 'sources' / 'gsq' / 'src' / 'quantization'

if str(SOLVER.parent) not in sys.path:
    sys.path.insert(0, str(SOLVER.parent))

from solver.gsq import create_quantizer  # noqa: E402
from solver.gumbel_mps import MPSOneBit  # noqa: E402

#: Weight shape. ``K % 128 == 0`` so a groupsize of 128 gives two groups per row.
N, K = 48, 512
GROUPSIZE = 128
STD = 0.2
STRENGTH = 1.0
TEMPERATURES = (1.0, 0.5, 0.1)
SCALE = 100.0
DTYPE = torch.bfloat16
LOGITS_DTYPE = torch.float32
SEED = 0

#: Both implementations run identical arithmetic in the same order on the same
#: device, so the only legitimate forward/gradient difference is which logistic
#: noise sample each drew -- and the shim makes that the same sample. What is
#: left is bf16 rounding of the same operations, measured below.
TOL_FORWARD = 4e-4
TOL_GRAD = 4e-3
TOL_SCALES = 1e-5


@contextmanager
def cpu_cuda_rng_shim():
    """Run upstream's ``torch.cuda.*_rng_state`` calls on the CPU stream."""
    original_get = torch.cuda.get_rng_state
    original_set = torch.cuda.set_rng_state
    torch.cuda.get_rng_state = lambda device=None: torch.get_rng_state()
    torch.cuda.set_rng_state = lambda state, device=None: torch.set_rng_state(state)
    try:
        yield
    finally:
        torch.cuda.get_rng_state = original_get
        torch.cuda.set_rng_state = original_set


@contextmanager
def aligned_seed(seed):
    """Position the global CPU stream identically for both implementations."""
    saved = torch.get_rng_state().clone()
    saved_seed = int(torch.initial_seed())
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.manual_seed(saved_seed)
        torch.set_rng_state(saved)


def load_upstream():
    """Load the pinned upstream one-bit quantizer by path."""
    path = UPSTREAM_Q / 'gumbel_quantizer_1bit.py'
    spec = importlib.util.spec_from_file_location('upstream_gumbel_quantizer_1bit', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load upstream quantizer from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GumbelQuantizer1Bit


def synthetic_inputs(seed=SEED):
    """A fixed ``(Q, scales)`` pair in the solver's own convention.

    ``create_quantizer`` builds ``Q = sign(weight) * scales.repeat_interleave``,
    so ``Q / scales[:, idx]`` is exactly ``±1``. Reproducing that here keeps the
    comparison on the same inputs the solver actually feeds the quantizer.
    """
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(N, K, generator=generator, dtype=torch.float32) * 0.02
    scales = weight.reshape(N, -1, 128).abs().mean(-1).clamp_min(1e-8)
    initial = torch.where(weight > 0, 1.0, -1.0) * scales.repeat_interleave(128, 1)
    return initial, scales


def build_pair(device, logits_dtype=LOGITS_DTYPE):
    """Construct upstream's and this port's quantizer from the same stream position.

    Seeding before *each* constructor, rather than once around both, gives each
    side a declared starting position instead of one inherited from how many
    draws the other consumed. Upstream's ``__init__`` never touches
    ``torch.cuda``, so no shim is needed for construction.
    """
    upstream_cls = load_upstream()
    Q, scales = synthetic_inputs()
    Q, scales = Q.to(device), scales.to(device)
    with aligned_seed(SEED):
        up = upstream_cls(Q.clone(), scales.clone(), GROUPSIZE, STD, STRENGTH,
                          device, DTYPE, logits_dtype)
    with aligned_seed(SEED):
        port = MPSOneBit(Q.clone(), scales.clone(), GROUPSIZE, STD, STRENGTH,
                         device, DTYPE, logits_dtype)
    return up, port


def test_initialisation_matches_upstream_quantizer():
    """``create_quantizer`` must start from the same logits as upstream.

    Upstream (``gumbel_quantizer_1bit.py:19``) initialises from the *normalised*
    weight, ``std * (randn_like(Q) + Q * strength)`` with ``Q = W / scale``,
    ``std=0.01`` and ``strength=6``. The port previously passed a sign-only tensor
    with ``std=0.2`` and ``strength=1``: that keeps no magnitude information and
    flips roughly 16% of the starting signs off the RTN initialisation it is meant
    to start from, which is a first-order difference in what the optimizer is
    searching around. This pins the two constructions together exactly, rather
    than re-deriving the formula in the test.
    """
    upstream_quantizer = load_upstream()
    weight = torch.randn(8, 256) * .05
    scales = weight.abs().reshape(8, -1, 128).mean(-1).clamp_min(1e-8)
    # Upstream's *call sites* pass the RTN initialisation, not the raw weight
    # (`src/trainer.py:47`, `src/prior/gptq.py:236`, with the RTN grid coming from
    # `src/prior/quant.py`'s maxq==-2 branch). Matching the class is not enough:
    # an earlier revision of this change passed the raw weight, which is a
    # different distribution of starting logits.
    initial = torch.where(weight > 0, 1., -1.) * scales.repeat_interleave(128, 1)

    torch.manual_seed(SEED)
    reference = upstream_quantizer(initial, scales, 128, .01, 6,
                                   torch.device('cpu'), torch.float32,
                                   logits_dtype=torch.float32)
    torch.manual_seed(SEED)
    ours = create_quantizer(weight, None)

    torch.testing.assert_close(ours.sign_logits, reference.sign_logits, rtol=0, atol=0)
    torch.testing.assert_close(ours.scales, reference.scales, rtol=0, atol=0)


def loss_weights(device):
    """Non-uniform per-element weights, so every element gets a real gradient."""
    return (1.0 + 0.25 * torch.arange(N, dtype=torch.float32, device=device)).unsqueeze(1)


def _nan_to_num(tensor):
    return torch.nan_to_num(tensor.float(), nan=float('inf'))


@contextmanager
def pinned_noise(sign_logits, uniform):
    """Force ``torch.rand_like``/``torch.randn_like`` to return fixed tensors.

    Needed because upstream's forward draws its relaxation noise with
    ``torch.rand_like`` on the *device* stream, which on MPS is a generator
    ``torch.manual_seed`` does not position (measured: ``torch.rand(5)`` gives
    ``[0.4963, 0.7682, ...]`` under CPU and ``[0.3664, 0.0755, ...]`` under MPS
    from the same seed). Seeding alone therefore cannot put upstream and this
    port on the same sample point on MPS, and any comparison made without
    pinning measures the two noise draws rather than the two implementations.

    Pinning the *draw* rather than the noise keeps upstream's own
    ``torch.logit(u, eps=1e-8)`` in the path, so the clamp is still upstream's.
    """
    original_rand_like = torch.rand_like
    original_randn_like = torch.randn_like
    torch.rand_like = lambda t, *a, **k: uniform.to(t.device)
    torch.randn_like = lambda t, *a, **k: sign_logits.to(t.device)
    try:
        yield
    finally:
        torch.rand_like = original_rand_like
        torch.randn_like = original_randn_like


def run_coupled(device, temperature, steps=3, logits_dtype=LOGITS_DTYPE, pin=None, uniform=None):
    """Run coupled forward/backward steps and return the measured deltas.

    Both sides are seeded immediately before the op they are about to run, so a
    divergence in one step cannot accumulate into the next.

    On a device whose ``randn`` stream is not the CPU stream -- MPS -- upstream's
    ``__init__`` draws its ``sign_logits`` from a different point than the port
    does, so the two would be compared from different starting states and every
    delta would be the constructor's, not the algebra's. The init is therefore
    copied from upstream onto the port on such devices, which is the same
    technique ``align_int_init`` uses in the sibling port's harness: the
    comparison is of forward/backward/hard-weight arithmetic, and the
    constructor difference is reported separately by
    :func:`test_constructor_divergence_on_mps_is_upstreams_not_the_ports`.

    ``pin`` and ``uniform`` optionally force the ``randn_like``/``rand_like``
    draws to fixed tensors. On MPS that is required for any comparison at all;
    see :func:`pinned_noise`.
    """
    pinning = pinned_noise(pin, uniform) if pin is not None else nullcontext()
    with pinning:
        up, port = build_pair(device, logits_dtype)
        constructor_diverges = not torch.equal(up.sign_logits.cpu(), port.sign_logits.cpu())
        with torch.no_grad():
            port.sign_logits.copy_(up.sign_logits)
        weight = loss_weights(device)
        forward_delta = 0.0
        grad_logits_delta = 0.0
        grad_scales_delta = 0.0
        dtype_mismatch = 0
        outputs_agree = True

        for _ in range(steps):
            for module in (up, port):
                for parameter in module.parameters():
                    parameter.grad = None
            with cpu_cuda_rng_shim():
                with aligned_seed(SEED):
                    out_up = up(temperature, SCALE)
                with aligned_seed(SEED):
                    out_port = port(temperature, SCALE)
                if out_up.dtype != out_port.dtype:
                    dtype_mismatch += 1
                forward_delta = max(forward_delta,
                                    _nan_to_num((out_up.float() - out_port.float()).abs()).max().item())
                outputs_agree = outputs_agree and torch.isfinite(out_port).all().item()
                (out_up * weight).sum().backward()
                (out_port * weight).sum().backward()
            grad_logits_delta = max(
                grad_logits_delta,
                _nan_to_num((up.sign_logits.grad.float() - port.sign_logits.grad.float()).abs()).max().item(),
            )
            grad_scales_delta = max(
                grad_scales_delta,
                _nan_to_num((up.scales.grad.float() - port.scales.grad.float()).abs()).max().item(),
            )

        with torch.no_grad():
            hard_up, scales_up = up.get_hard_weights()
            hard_port, scales_port = port.get_hard_weights()
        hard_equal = torch.equal(hard_up.float().cpu(), hard_port.float().cpu())
        scales_equal = torch.equal(scales_up.float().cpu(), scales_port.float().cpu())
        hard_delta = (hard_up.float() - hard_port.float()).abs().max().item()
    return {
        'forward_delta': forward_delta,
        'grad_logits_delta': grad_logits_delta,
        'grad_scales_delta': grad_scales_delta,
        'hard_equal': hard_equal,
        'scales_equal': scales_equal,
        'hard_delta': hard_delta,
        'dtype_mismatch_steps': dtype_mismatch,
        'outputs_agree': outputs_agree,
        'constructor_diverges': constructor_diverges,
        'up_dtype': str(up.get_hard_weights()[0].dtype),
        'port_dtype': str(port.get_hard_weights()[0].dtype),
        'forward_up_dtype': str(out_up.dtype),
        'forward_port_dtype': str(out_port.dtype),
    }


def assert_parity(result, tol_forward, tol_grad, tol_scales):
    assert result['outputs_agree'], 'port produced non-finite forward output'
    assert result['forward_delta'] <= tol_forward, result
    assert result['grad_logits_delta'] <= tol_grad, result
    assert result['grad_scales_delta'] <= tol_scales, result
    # Hard weights are discrete and lie on the same scale grid; they must agree
    # exactly, not approximately.
    assert result['hard_equal'], result
    assert result['scales_equal'], result


@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_cpu_matches_upstream_across_temperatures(temperature):
    result = run_coupled('cpu', temperature)
    assert_parity(result, TOL_FORWARD, TOL_GRAD, TOL_SCALES)
    # Measured, not assumed: at logits_dtype=fp32 upstream's forward is already
    # bf16, so the port's extra .to(dtype) changes nothing and there is no dtype
    # mismatch to report. Pinned so a future change to either side shows up here.
    assert result['forward_up_dtype'] == 'torch.bfloat16', result
    assert result['forward_port_dtype'] == 'torch.bfloat16', result
    assert result['dtype_mismatch_steps'] == 0, result
    assert result['forward_delta'] == 0.0, result
    assert result['grad_logits_delta'] == 0.0, result
    # On CPU the constructors agree, so nothing was aligned to get here.
    assert result['constructor_diverges'] is False, result


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
def test_constructor_divergence_on_mps_is_upstreams_not_the_ports():
    """The MPS constructor difference is upstream's, and it is measured.

    Upstream's ``__init__`` calls ``torch.randn_like(Q)``, which on MPS draws
    from the Metal generator ``torch.manual_seed`` does not position, so
    upstream's MPS ``sign_logits`` differ from its own CPU ``sign_logits`` under
    the same seed. This port draws on the global CPU stream, so its MPS and CPU
    ``sign_logits`` are identical. The test asserts both directions, which is
    what makes the port's cross-device claim rest on a measurement rather than
    on the constructor happening to agree.
    """
    upstream_cls = load_upstream()
    Q, scales = synthetic_inputs()

    def build(cls, device):
        with aligned_seed(SEED):
            return cls(Q.clone().to(device), scales.clone().to(device), GROUPSIZE,
                       STD, STRENGTH, device, DTYPE, LOGITS_DTYPE)

    upstream_cpu, upstream_mps = build(upstream_cls, 'cpu'), build(upstream_cls, 'mps')
    port_cpu, port_mps = build(MPSOneBit, 'cpu'), build(MPSOneBit, 'mps')

    assert not torch.equal(upstream_cpu.sign_logits, upstream_mps.sign_logits.cpu())
    assert torch.equal(port_cpu.sign_logits, port_mps.sign_logits.cpu())
    # And the port's CPU init is upstream's CPU init, so this is a device-draw
    # difference in upstream rather than a different initialisation scheme.
    assert torch.equal(port_cpu.sign_logits, upstream_cpu.sign_logits)


@pytest.mark.parametrize('logits_dtype', [torch.float32, torch.bfloat16])
def test_cpu_matches_upstream_for_every_logits_dtype(logits_dtype):
    """The dtype the sibling port broke on.

    With ``logits_dtype=bfloat16``, ``sign_logits`` is bf16 and the Python float
    literals in ``2.0 * sigmoid(...) - 1.0`` promote the result to fp32, so
    upstream's ``.to(dtype)`` is a real narrowing and its ``soft_sign *
    scale_per_col`` is bf16 only because of that cast. The port applies the same
    cast in the same place; this asserts the two agree on values *and* on the
    returned dtype rather than leaving it to inspection.
    """
    result = run_coupled('cpu', 1.0, logits_dtype=logits_dtype)
    assert_parity(result, TOL_FORWARD, TOL_GRAD, TOL_SCALES)
    assert result['forward_up_dtype'] == result['forward_port_dtype'], result
    assert result['forward_up_dtype'] == 'torch.bfloat16', result


def test_cpu_noise_is_reproducible_from_manual_seed():
    """The port's noise is a pure function of the global CPU stream position."""
    Q, scales = synthetic_inputs()
    first = None
    for _ in range(2):
        with aligned_seed(SEED):
            module = MPSOneBit(Q.clone(), scales.clone(), GROUPSIZE, STD, STRENGTH,
                               'cpu', DTYPE, LOGITS_DTYPE)
            with aligned_seed(SEED + 7):
                out = module(1.0, SCALE)
        first = out if first is None else first
        assert torch.equal(out, first)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_mps_matches_upstream_within_stated_tolerance(temperature):
    """Upstream and this port, both on MPS, on a pinned noise sample.

    Upstream cannot be driven by ``manual_seed`` on MPS at all: its forward draws
    on the Metal stream. So the two implementations are put on the same sample
    point by pinning the ``rand_like`` draw, after which both run upstream's own
    ``torch.logit(u, eps=1e-8)`` on the same ``u``. What remains is the two
    forward/backward implementations, and the measured delta is ``0.0``.
    """
    zeros = torch.zeros(N, K, dtype=torch.float32)
    uniform = torch.rand(N, K, dtype=torch.float32)
    result = run_coupled('mps', temperature, pin=zeros, uniform=uniform)
    assert_parity(result, tol_forward=2e-3, tol_grad=2e-2, tol_scales=2e-4)
    # Zero measured, not merely within tolerance.
    assert result['forward_delta'] == 0.0, result
    assert result['grad_logits_delta'] == 0.0, result
    assert result['grad_scales_delta'] == 0.0, result


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_mps_matches_upstream_across_devices_on_pinned_noise(temperature):
    """Upstream-on-CPU vs the port-on-MPS, on one pinned noise sample.

    The cross-device statement, which is what the port is for: the port's MPS
    arithmetic equals upstream's CPU arithmetic when both see the same noise.
    """
    upstream_cls = load_upstream()
    Q, scales = synthetic_inputs()
    uniform = torch.rand(N, K, dtype=torch.float32)
    zeros = torch.zeros(N, K, dtype=torch.float32)

    with pinned_noise(zeros, uniform):
        with cpu_cuda_rng_shim():
            up = upstream_cls(Q.clone(), scales.clone(), GROUPSIZE, STD, STRENGTH,
                              'cpu', DTYPE, LOGITS_DTYPE)
            port = MPSOneBit(Q.clone().to('mps'), scales.clone().to('mps'), GROUPSIZE,
                             STD, STRENGTH, 'mps', DTYPE, LOGITS_DTYPE)
            with torch.no_grad():
                port.sign_logits.copy_(up.sign_logits)
            weight = loss_weights('cpu')
            out_up = up(temperature, SCALE)
            out_port = port(temperature, SCALE)
            (out_up * weight).sum().backward()
            (out_port * weight.cpu().to('mps')).sum().backward()

    assert out_port.dtype == out_up.dtype
    assert torch.equal(out_up.cpu(), out_port.cpu())
    assert torch.equal(up.sign_logits.grad.cpu(), port.sign_logits.grad.cpu())
    assert torch.equal(up.scales.grad.cpu(), port.scales.grad.cpu())
    assert torch.equal(up.get_hard_weights()[0].cpu(), port.get_hard_weights()[0].cpu())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_mps_hard_weights_are_exact_on_the_scale_grid(temperature):
    module = build_pair('mps')[1]
    with torch.no_grad():
        hard, scales = module.get_hard_weights()
        pattern = 2.0 * (module.sign_logits > 0).to(DTYPE) - 1.0
        expected = pattern * module.scales[:, module.idx].to(DTYPE)
    assert torch.equal(hard, expected)
    assert torch.equal(scales, module.scales.to(DTYPE))
    assert hard.dtype == scales.dtype == DTYPE


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
def test_works_where_upstream_raises():
    """Upstream's own function raises on MPS; the port runs.

    This is the port's reason to exist, asserted directly instead of inferred
    from the parity numbers.
    """
    upstream_cls = load_upstream()
    Q, scales = synthetic_inputs()
    Q, scales = Q.to('mps'), scales.to('mps')
    with aligned_seed(SEED):
        up = upstream_cls(Q.clone(), scales.clone(), GROUPSIZE, STD, STRENGTH,
                          'mps', DTYPE, LOGITS_DTYPE)
    with pytest.raises(Exception):
        up(1.0, SCALE)

    with aligned_seed(SEED):
        port = MPSOneBit(Q.clone(), scales.clone(), GROUPSIZE, STD, STRENGTH,
                         'mps', DTYPE, LOGITS_DTYPE)
    out = port(1.0, SCALE)
    assert out.device.type == 'mps'
    out.sum().backward()
    assert port.sign_logits.grad is not None and torch.isfinite(port.sign_logits.grad).all()
    assert port.scales.grad is not None and torch.isfinite(port.scales.grad).all()


if __name__ == '__main__':
    # MPS is reported with pinned noise, because *_without* pinning upstream draws
    # from the Metal stream that torch.manual_seed does not position, and the
    # resulting numbers measure the two noise draws rather than the two
    # implementations. See ``pinned_noise``.
    pinned = (torch.zeros(N, K, dtype=torch.float32), torch.rand(N, K, dtype=torch.float32))
    for device in ('cpu', 'mps'):
        if device == 'mps' and not torch.backends.mps.is_available():
            continue
        for temperature in TEMPERATURES:
            options = {'pin': pinned[0], 'uniform': pinned[1]} if device == 'mps' else {}
            measured = run_coupled(device, temperature, **options)
            print(device, f'T={temperature}', measured)