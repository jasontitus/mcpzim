"""Input→output parity for a whole GSQ block: upstream's quantizer vs this port.

``test_gumbel_mps.py`` compares the two *quantizers*; the CUDA canary compares
*updates* (provenance, scope, restore-equality) without ever putting a number from
one side next to a number from the other. Neither answers the question this file
answers: with the same weights, the same student stream, the same teacher stream
and the same noise draws, does a whole ``BlockTrainer.forward`` produce upstream's
loss and upstream's gradients - on CPU and on MPS?

Method
------
Upstream's ``GumbelQuantizer1Bit`` is loaded from the pinned tree at
``tools/calibration/cuda_runtime/.context/sources/gsq`` and its sole CUDA
dependency - the two ``torch.cuda.*_rng_state`` calls in the autograd function -
is shimmed onto the CPU stream, exactly as ``test_gumbel_mps.py`` does. The whole
block is then driven twice over one student stream and one teacher stream:

* the port through the real code path under test,
  ``BlockTrainer.forward(student, teacher, temperature=1., scale=1.)`` on the
  device being tested - which means ``create_quantizer``'s own device policy
  decides the quantizer, i.e. ``CPUOneBit`` on CPU and ``MPSOneBit`` on MPS, so
  what is measured is what the port actually runs rather than a hand-picked
  substitute;
* upstream through :func:`block_objective`, which is the same objective written
  out independently: the target is the unquantized block on the *teacher* stream,
  under ``no_grad``; the subject is the same block on the *student* stream with
  each eligible projection's weight replaced by its quantizer's output; the
  objective is ``MSELoss(subject, target)`` (upstream's ``self.loss_fn``,
  ``loss_fn = torch.nn.MSELoss()`` at ``base.py:49``). Both sides use the same
  expression, so the comparison isolates the quantizer.

The input and the target are two different draws (``INPUT_SEED``, ``TEACHER_SEED``)
of the same shape and scale. That is the port's contract - the drifted student
stream in, the clean target stream out of the unquantized block - and it is also
what makes the equality below able to see a driver that fed one stream into both
forwards: the superseded same-input form is a *different* number, not a rounding
difference.

``loss.backward()`` on both, then the *gradients* are compared, not the updates:
the port optimises with Adam where upstream uses Lion, which is a sanctioned
deviation, so a post-step comparison would diverge for reasons that have nothing
to do with the port's numerics. Gradients are the invariant that must hold. The
gradients that exist here are the quantizer parameters (``sign_logits``,
``scales``, one pair per eligible projection) and the student input, which is a
leaf, so the input→output relation is measured end to end. The teacher is not one
of them: both sides compute their target under ``no_grad``, so a gradient reaching
it would itself be a deviation.

Noise, pinned exactly
---------------------
``torch.rand``/``randn``/``rand_like``/``randn_like`` are replaced (see
:func:`pinned_noise`) by draws from a generator keyed by ``(kind, shape, seed)``.
That is what makes "the same noise draws" literally true rather than
approximately seeded:

* upstream draws its forward noise with ``torch.rand_like`` on the *device*
  stream, which on MPS is a generator ``torch.manual_seed`` does not position
  (measured in ``gumbel_mps.py``'s module docstring);
* the port draws on the global CPU stream;
* a shape-keyed pin removes both as a variable, and keeps upstream's own
  ``torch.logit(u, eps=1e-8)`` and the port's own
  ``torch.logit(u.clamp(1e-8, 1 - 1e-8))`` in the path, which were measured to be
  bit-identical to each other (max ``|difference|`` ``0.0`` over 4096 draws on both
  CPU and MPS, i.e. the clamp form is not a residual source).

The pin covers the constructors too, so the RTN initialisation
(``sign(w) * mean|w|`` per 128-group, ``std=0.01``, ``strength=6`` - what
``create_quantizer`` and upstream's call sites at ``trainer.py:47`` /
``prior/gptq.py:236`` pass) is drawn from the same numbers on both sides. The
initial parameters are then *asserted* bit-identical for the on-device axes, and
copied from upstream for the cross-device axis, where the group mean differs by
one fp32 ulp because MPS reduces it differently: measured ``3.7e-09`` on group
means of order ``0.02``, with the initial logits identical (``0.0``) because they
depend only on ``sign(w)``. Copying is what makes the cross-device axis measure
forward/backward arithmetic instead of the two devices' reduction order.

Measured deviations (torch 2.8.0, Apple M1 Ultra, ``T=1.0``, ``scale=1.0``,
``tiny_model(layers=2)``, block 0, 8 eligible projections, 2026-09-20)
-------------------------------------------------------------------------
======================  =================  ===================  ==================
axis                    loss                student-input grad    max param grad
======================  =================  ===================  ==================
upstream-CPU vs port-CPU  ``0.0`` (exact)   ``0.0`` (exact)      ``5.68e-14`` logits
                                                               ``7.276e-11`` scales
upstream-MPS vs port-MPS  ``0.0`` (exact)   ``5.821e-11``        ``1.42e-13`` logits
                                             (``7.55e-08`` rel)   ``5.82e-11`` .. ``1.16e-10``
                                                                 scales
upstream-CPU vs port-MPS  ``1`` ulp         ``8.731e-11``        ``2.84e-13`` logits
                          (``1.863e-09``)    (``1.13e-07`` rel)   ``8.73e-11`` .. ``1.31e-10``
                                                                 scales
======================  =================  ===================  ==================

The loss is ``0.02051140`` on every axis, and it is larger than the same-input
form's ``7.06e-05`` by construction: the two forwards now see *different* streams,
so what is measured is the block's response to both rather than the quantization
error on one. Nothing above is loosened by that - every bound is either exact or
quoted in ulps of the quantity it bounds, and the relative ones came out tighter
than the ones the same-input form carried.

Worst parameter-gradient deviation relative to the largest upstream gradient of
that parameter: ``2.94e-07`` (CPU), ``4.48e-07``..``1.17e-06`` (MPS),
``9.15e-07``..``1.26e-06`` (cross-device), i.e. 2.5, 3.8-9.9 and 7.7-10.6 fp32
unit roundoffs. The tolerances below are those measurements with stated headroom,
and where a deviation measured exactly zero it is asserted to be exactly zero.

The ranges are the honest part. Every quantity above is fixed from run to run
*except* the ``scales`` gradient on the MPS axes, which moved between ``5.82e-11``
and ``1.31e-10`` over eight repeats: that gradient sums a group's 128 columns
through ``scatter_add_``, and MPS does not fix the order of that sum (measured
directly - twenty ``scatter_add_`` calls on identical inputs are not bit-stable on
MPS, while the CPU's are). The ``scales`` bounds on those axes are therefore set
above the spread rather than at the measurement, and the tight, order-independent
statement is the relative one: a handful of unit roundoffs.

Where the port legitimately differs
-----------------------------------
1. **The port's CPU counterpart differentiates by autograd.** ``CPUOneBit``
   (``gsq.py``) is the port's CPU branch and backpropagates through the same
   forward expression with ``torch.autograd``, while upstream - and the port's
   own ``MPSOneBit``, whose backward is upstream's analytic form - computes
   ``grad * (1 - soft_sign**2) * scale / temperature`` by hand. Measured at the
   quantizer level, on one module with identical parameters and one pinned draw
   on CPU, where both sides' ``logit`` of that draw is the same number so the
   comparison isolates the backward form: the forward is bit-identical,
   ``MPSOneBit``'s gradient is bit-identical to upstream's, and ``CPUOneBit``'s
   differs in the last bits (``1.9e-09`` absolute on the probe's gradients). That
   difference - two algebraically equal forms of the same derivative - is the
   whole of the CPU axis' gradient residual.
2. **The logistic noise is computed on the CPU stream.** Upstream draws
   ``u`` and takes ``logit`` on the device; the port draws ``u`` on the global CPU
   stream and moves it (``gumbel_mps.py``'s module docstring, the first sanctioned
   difference), so its ``logit`` runs on the CPU kernel. Measured directly:
   ``torch.logit`` of the same ``u`` is *not* identical on CPU and MPS - one ulp,
   ``4.8e-07`` absolute on draws spanning ``1e-08`` to 1. That is the whole of the
   MPS axis' residual. The second sanctioned difference - noise stored rather than
   replayed - is not observable here, because the pin makes the replay land on the
   same draw.

Neither of those is tolerance-washed: the bounds below are a handful of fp32 unit
roundoffs, which is what those two mechanisms can produce, not a loose-enough
number to hide an algebra error.

Not compared, and why
---------------------
* **Optimizer updates / post-step parameters.** Adam versus Lion; gradients are
  the invariant (see above).
* **A real CUDA device.** This machine has no CUDA, so "upstream" here is
  upstream's source driven under the CPU RNG shim, not a compile-and-run on the
  hardware it ships for. The algebra is upstream's; the shimmed RNG plumbing is
  the port's own sanctioned difference.
* **The bf16 production configuration at block level.** The comparison runs at
  the tiny model's fp32 native dtype. In bf16 the port's CPU counterpart rounds
  differently *by construction*: ``CPUOneBit`` returns ``soft(fp32) *
  scales(fp32)`` while upstream narrows the group scale to ``self.dtype`` before
  the multiply, so a bf16 block run would measure cast placement as much as the
  quantizer. The ``logits_dtype``/bf16 axis is pinned at quantizer level by
  ``test_gumbel_mps.py``.
* **The RCO stage, and hard weights.** Out of scope here; the RCO stage and
  ``get_hard_weights`` have their own suites.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

SOLVER = Path(__file__).resolve().parents[1]
UPSTREAM_Q = (SOLVER.parent / 'cuda_runtime' / '.context' / 'sources' / 'gsq'
              / 'src' / 'quantization')

if str(SOLVER.parent) not in sys.path:
    sys.path.insert(0, str(SOLVER.parent))

from solver.gsq import BlockTrainer  # noqa: E402
from solver.qwen import block_kwargs, tiny_model  # noqa: E402

#: Block 0 of ``tiny_model(layers=2)`` is a ``linear_attention`` layer, whose eight
#: projections are all eligible (``in_features % 128 == 0``), so one block covers
#: both a grouped projection (``in_proj_qkv``, 256x128) and a square one.
BLOCK_INDEX = 0
LAYERS = 2
HIDDEN = 128
BATCH, SEQ = 2, 5
#: ``create_quantizer``'s constants, which are upstream's call-site constants.
GROUPSIZE, STD, STRENGTH = 128, 0.01, 6
#: The trainer's defaults, which is how the port's callers drive it.
TEMPERATURE, SCALE = 1.0, 1.0
BUILD_SEED, NOISE_SEED, INPUT_SEED = 0, 0, 11
#: The teacher stream is a second draw of the same shape and scale, distinct from
#: the student's: the port pairs a drifted input with a clean target, and a driver
#: that collapsed the two would be a different objective (see the module docstring).
TEACHER_SEED = 23

#: One fp32 unit roundoff. Tolerances that are not zero are quoted in these units,
#: because "one ulp" is the only size a rounding difference can legitimately have.
EPS = float(torch.finfo(torch.float32).eps)

#: One fp32 ulp of the parity loss itself. Both implementations compute the loss
#: ``0.02051140``, which lies in ``[2**-6, 2**-5)``, so its spacing is ``2**-29``;
#: a discrepancy that is a rounding of the mean rather than of the algebra can be
#: no larger than that. This is the only bound in this file stated in the loss's
#: own units, because the loss is a single reduced scalar and not a graded tensor.
LOSS_ULP = 2 ** -29

#: The ``scales`` gradient sums the 128 columns of a group into one number through
#: ``scatter_add_``, and MPS does not fix that summation's order: twenty repeated
#: ``scatter_add_`` calls on identical inputs were measured *not* to be bit-stable
#: on this machine, while the CPU's are. Over eight repeats of this comparison the
#: ``scales`` deviation moved between 5.82e-11 and 1.31e-10 while every other
#: quantity in the table below stayed fixed, so the MPS ``scales`` bounds are set
#: above that spread. They are still a statement about the port - the deviation is
#: a couple of the largest gradient's ulps - rather than about a summation order,
#: which is what the relative bound (a handful of unit roundoffs) is for.
SCALES_GRAD_TOL_CPU = 4e-10
SCALES_GRAD_TOL_MPS = 5e-10

# --- upstream-on-CPU vs the port's BlockTrainer-on-CPU (create_quantizer -> CPUOneBit)
CPU_LOSS_DELTA = 0.0            # measured: bit-identical, fixed over eight repeats
CPU_STUDENT_GRAD_DELTA = 0.0    # measured: bit-identical, fixed over eight repeats
CPU_LOGITS_GRAD_TOL = 1e-13     # measured 5.68e-14
CPU_GRAD_REL_TOL = 20 * EPS     # measured worst 2.94e-07 = 2.5 eps
# --- upstream-on-MPS vs the port's BlockTrainer-on-MPS (MPSOneBit)
MPS_LOSS_DELTA = 0.0            # measured: bit-identical, fixed over eight repeats.
                                # Not luck: the two sides' outputs differ by ~1e-11
                                # relative, which moves the mean of squares by ~1e-12,
                                # a thousandth of this loss's ulp (2**-29) - the
                                # same-input form's smaller loss (7.06e-05, ulp
                                # 2**-37) sat on the same rounding step's doorstep.
MPS_STUDENT_GRAD_REL_TOL = 20 * EPS  # measured 7.55e-08 = 0.63 eps
MPS_LOGITS_GRAD_TOL = 1e-12     # measured 1.42e-13, fixed over eight repeats
MPS_GRAD_REL_TOL = 40 * EPS     # measured worst 4.48e-07 .. 1.17e-06 = 3.8 .. 9.9 eps
# --- upstream-on-CPU vs the port's BlockTrainer-on-MPS
CROSS_LOSS_TOL = LOSS_ULP       # measured 1.863e-09 = exactly 1 ulp of the loss
CROSS_STUDENT_GRAD_REL_TOL = 40 * EPS  # measured 1.13e-07 = 0.95 eps
CROSS_LOGITS_GRAD_TOL = 1e-12   # measured 2.84e-13
CROSS_GRAD_REL_TOL = 80 * EPS   # measured worst 9.15e-07 .. 1.26e-06 = 7.7 .. 10.6 eps
CROSS_INIT_SCALES_TOL = 1e-8    # measured 3.73e-09: MPS's group mean, one ulp off CPU's

_MODELS: dict[str, torch.nn.Module] = {}


def load_upstream():
    """Load the pinned upstream one-bit quantizer by path."""
    path = UPSTREAM_Q / 'gumbel_quantizer_1bit.py'
    spec = importlib.util.spec_from_file_location('upstream_gumbel_quantizer_1bit', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load upstream quantizer from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GumbelQuantizer1Bit


@contextmanager
def cpu_cuda_rng_shim():
    """Run upstream's ``torch.cuda.*_rng_state`` calls on the CPU stream.

    The same technique as ``test_gumbel_mps.py``'s shim, which is the only CUDA
    dependency in upstream's quantizer: everything else it does is device
    arithmetic. This is what lets upstream's own class be driven on a machine
    without CUDA, and what forces its backward replay onto the pinned draw.
    """
    original_get = torch.cuda.get_rng_state
    original_set = torch.cuda.set_rng_state
    torch.cuda.get_rng_state = lambda device=None: torch.get_rng_state()
    torch.cuda.set_rng_state = lambda state, device=None: torch.set_rng_state(state)
    try:
        yield
    finally:
        torch.cuda.get_rng_state = original_get
        torch.cuda.set_rng_state = original_set


def _pinned_generator(kind, shape, seed):
    """A generator keyed by kind, shape and seed - and by nothing else.

    Keying on the shape is what lets one patch serve a block's eight projections
    of four different shapes: each projection gets a real (non-degenerate) draw,
    and the same projection gets the same draw on either implementation and on
    either device, whatever stream position or RNG each of them would have used.
    """
    payload = f'{kind}:{tuple(int(s) for s in shape)}'.encode()
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')
    return torch.Generator().manual_seed((seed * 1_000_003 + digest) % (2 ** 63))


@contextmanager
def pinned_noise(seed=NOISE_SEED):
    """Force every random draw to a fixed tensor keyed by ``(kind, shape, seed)``.

    Covers the four entry points the two implementations use - upstream's
    ``torch.randn_like`` in its constructor and ``torch.rand_like`` in its
    forward; the port's ``torch.randn``/``torch.rand`` (``MPSOneBit._randn_like``
    and ``_logistic_noise_like`` draw on the global CPU stream, and ``CPUOneBit``
    uses ``torch.rand_like``/``torch.randn_like``). Pinning the *draw* rather than
    the resulting noise keeps each side's own ``torch.logit`` in the path.
    """
    original = torch.rand, torch.randn, torch.rand_like, torch.randn_like

    def _shape(args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
            return tuple(args[0])
        return tuple(args)

    def rand(*args, **kwargs):
        shape = _shape(args)
        return original[0](shape, generator=_pinned_generator('rand', shape, seed),
                           dtype=kwargs.get('dtype') or torch.float32)

    def randn(*args, **kwargs):
        shape = _shape(args)
        return original[1](shape, generator=_pinned_generator('randn', shape, seed),
                           dtype=kwargs.get('dtype') or torch.float32)

    torch.rand = rand
    torch.randn = randn
    # Dtype-faithful, as ``*_like`` is: the draw is the one each implementation
    # asked for, at the dtype it asked for.
    torch.rand_like = lambda t, *a, **k: rand(*t.shape, dtype=t.dtype).to(t.device)
    torch.randn_like = lambda t, *a, **k: randn(*t.shape, dtype=t.dtype).to(t.device)
    try:
        yield
    finally:
        torch.rand, torch.randn, torch.rand_like, torch.randn_like = original


def tiny_model_on(device):
    """The same tiny model on any device: built once on CPU, then copied.

    Built once so the two sides of a comparison - and the two devices - start
    from bit-identical block weights, which is a precondition of the whole test
    rather than something to hope for from two seeded constructions.
    """
    if not _MODELS:
        torch.manual_seed(BUILD_SEED)
        _MODELS['cpu'] = tiny_model(layers=LAYERS)
    if device not in _MODELS:
        _MODELS[device] = copy.deepcopy(_MODELS['cpu']).to(device)
    return _MODELS[device]


def upstream_quantizers(model, device):
    """Upstream's quantizer per eligible projection, initialised as upstream is.

    Upstream's call sites pass ``Q = the RTN grid``, ``sign(w) * mean|w|`` per
    128-group (``trainer.py:47``, ``prior/gptq.py:236``, the grid itself from
    ``prior/quant.py``'s ``maxq == -2`` branch) - not the raw weight - which the
    quantizer then divides by the group scale, so the logits start as
    ``std * (randn + sign(w) * strength)``. This is the same construction
    ``create_quantizer`` performs, keyed by name so it can be checked against the
    trainer's own projection list.
    """
    quantizer_cls = load_upstream()
    block = model.model.layers[BLOCK_INDEX]
    quantizers = {}
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight.shape[1] % GROUPSIZE == 0:
            weight = module.weight
            scales = (weight.float().reshape(weight.shape[0], -1, GROUPSIZE).abs()
                      .mean(-1).clamp_min(1e-8))
            initial = torch.where(weight > 0, 1., -1.) * scales.repeat_interleave(GROUPSIZE, 1)
            quantizers[name] = quantizer_cls(initial.to(weight.dtype), scales, GROUPSIZE,
                                             STD, STRENGTH, device, weight.dtype,
                                             logits_dtype=torch.float32)
    return quantizers


def block_objective(model, student, teacher, replacements):
    """The port's objective written out independently, on both streams.

    ``BlockTrainer.forward`` takes the drifted student stream as its input and
    pairs it with ``block(teacher)`` - the *unquantized* block on the clean
    teacher stream - as its target, and this driver does the same thing by hand:
    the target under ``no_grad`` from the teacher stream, the subject from the
    student stream with every eligible projection's weight replaced by its
    quantizer's output, and ``MSELoss(subject, target)``. The on-device tests
    assert the two implementations agree bit-exactly, so this driver cannot be a
    different objective from the port's.

    **This pairing is a deliberate deviation from upstream, not a reproduction of
    it.** Upstream's ``calculate_mse`` (``src/models/base.py:444-453``) computes
    ``out_fp = forward_with_quantized(batch, None)`` and ``out_q =
    forward_with_quantized(batch, quantized_weights)`` on the *same* ``batch``, and
    ``main.py:391-402`` propagates that buffer through each *quantized* layer
    before training the next - so upstream's target sits on the drifted stream as
    well, and its loss rewards reproducing drift faithfully rather than opposing
    it. The port keeps the drifted input (what the block sees at inference) and
    moves only the target to the clean stream, which is the one form whose
    gradient opposes the drift measured here. An earlier revision of this port
    read upstream's *validation* helper ``get_loss``/``data_all['output']`` as the
    training target and "restored fidelity" by moving the target onto the student
    stream; that is the bug this file's assertions now guard.

    Read docs/QUANTIZATION_GSQ_DRIFT_REVIEW.md before changing either half.
    """
    block = model.model.layers[BLOCK_INDEX]
    kwargs = block_kwargs(model, student, BLOCK_INDEX)
    with torch.no_grad():
        target = block(teacher, **block_kwargs(model, teacher, BLOCK_INDEX))
    output = torch.func.functional_call(block, replacements, (student,), kwargs, strict=False)
    return (output.float() - target.float()).square().mean()


def _max_delta(ours, reference):
    """Largest absolute and relative deviation of two tensors of the same shape."""
    ours = ours.float().cpu()
    reference = reference.float().cpu()
    delta = (ours - reference).abs().max().item()
    scale = reference.abs().max().item()
    return delta, delta / max(scale, 1e-30)


def run_pair(device_up, device_port):
    """Drive the block both ways on one student stream, one teacher stream, one
    weight set and one noise sample."""
    model_up = tiny_model_on(device_up)
    model_port = tiny_model_on(device_port)
    weights_up = model_up.state_dict()
    weights_port = model_port.state_dict()
    assert list(weights_up) == list(weights_port), 'the two sides must hold the same tensors'
    assert all(torch.equal(weights_up[name].cpu(), weights_port[name].cpu())
               for name in weights_up), 'the two sides must start from the same block weights'

    # A real input, not a synthetic one: the block is driven through its own
    # attention/MLP path with real position embeddings and mask. The teacher is a
    # second draw of the same shape and scale, so the two streams are distinct
    # whatever the block does to them.
    student = (torch.randn(BATCH, SEQ, HIDDEN, generator=torch.Generator().manual_seed(INPUT_SEED))
               * 0.1)
    teacher = (torch.randn(BATCH, SEQ, HIDDEN, generator=torch.Generator().manual_seed(TEACHER_SEED))
               * 0.1)
    assert not torch.equal(student, teacher), 'the two streams must be different draws'
    student_up = student.to(device_up).detach().requires_grad_(True)
    student_port = student.to(device_port).detach().requires_grad_(True)
    # No grad on the teacher: the port's target is computed under ``no_grad``, and
    # upstream's is too, so nothing in either direction should reach it.
    teacher_up = teacher.to(device_up)
    teacher_port = teacher.to(device_port)
    assert torch.equal(student_up.cpu(), student_port.cpu()), \
        'the two sides must see the same input'
    assert torch.equal(teacher_up.cpu(), teacher_port.cpu()), \
        'the two sides must see the same target stream'

    with pinned_noise(), cpu_cuda_rng_shim():
        upstream = upstream_quantizers(model_up, device_up)
        trainer = BlockTrainer(model_port, BLOCK_INDEX)
        # The port's own projection selection has to line up with upstream's, by
        # name and by order, or the comparison would be of different weights.
        assert list(upstream) == list(trainer.names), (list(upstream), trainer.names)

        def port_replacements():
            return {name + '.weight': quantizer(TEMPERATURE, SCALE)
                    for name, quantizer in zip(trainer.names, trainer.quantizers)}

        with torch.no_grad():
            init_logits_delta = max(
                _max_delta(q.sign_logits, upstream[name].sign_logits)[0]
                for name, q in zip(trainer.names, trainer.quantizers))
            init_scales_delta = max(
                _max_delta(q.scales, upstream[name].scales)[0]
                for name, q in zip(trainer.names, trainer.quantizers))
            # Same weights on both sides is the premise, so the initial parameters
            # are aligned exactly. On one device the pinned construction already
            # agrees (asserted by the tests); across devices the group mean differs
            # by an ulp of the *reduction*, which is not the quantizer's algebra.
            for name, quantizer in zip(trainer.names, trainer.quantizers):
                quantizer.sign_logits.copy_(upstream[name].sign_logits)
                quantizer.scales.copy_(upstream[name].scales)
            # Pins this file's driver to the port's real forward.
            driver_equal = torch.equal(
                block_objective(model_port, student_port, teacher_port, port_replacements()),
                trainer.forward(student_port, teacher_port, TEMPERATURE, SCALE))

        loss_port = trainer.forward(student_port, teacher_port, TEMPERATURE, SCALE)
        loss_up = block_objective(model_up, student_up, teacher_up, {
            name + '.weight': quantizer(TEMPERATURE, SCALE)
            for name, quantizer in upstream.items()})
        loss_port.backward()
        loss_up.backward()

        logits_grad_delta = 0.0
        scales_grad_delta = 0.0
        grad_relative = 0.0
        for name, quantizer in zip(trainer.names, trainer.quantizers):
            reference = upstream[name]
            delta, relative = _max_delta(quantizer.sign_logits.grad, reference.sign_logits.grad)
            logits_grad_delta = max(logits_grad_delta, delta)
            grad_relative = max(grad_relative, relative)
            delta, relative = _max_delta(quantizer.scales.grad, reference.scales.grad)
            scales_grad_delta = max(scales_grad_delta, delta)
            grad_relative = max(grad_relative, relative)
        student_delta, student_relative = _max_delta(student_port.grad, student_up.grad)

    return {
        'axis': f'upstream-on-{device_up} vs port-on-{device_port}',
        'names': list(trainer.names),
        'init_logits_delta': init_logits_delta,
        'init_scales_delta': init_scales_delta,
        'init_equal': init_logits_delta == 0.0 and init_scales_delta == 0.0,
        'driver_equal': driver_equal,
        'loss_up': loss_up.item(),
        'loss_port': loss_port.item(),
        'loss_delta': abs(loss_up.item() - loss_port.item()),
        'student_grad_delta': student_delta,
        'student_grad_relative': student_relative,
        'logits_grad_delta': logits_grad_delta,
        'scales_grad_delta': scales_grad_delta,
        'grad_relative': grad_relative,
    }


def test_pinned_noise_is_one_draw_for_every_path_and_device():
    """The pinning technique itself, asserted rather than described.

    If a draw stopped going through the pinned entry points, the two sides would
    be compared on different noise and every number below would be meaningless, so
    the property the rest of the file rests on is checked directly: repeated draws
    agree, the four entry points agree with each other, and MPS gets the same
    numbers as CPU.
    """
    shape = (4, 8)
    reference = torch.zeros(shape)
    with pinned_noise():
        first = (torch.rand(shape), torch.randn(shape), torch.rand_like(reference),
                 torch.randn_like(reference))
        second = (torch.rand(shape), torch.randn(shape), torch.rand_like(reference),
                  torch.randn_like(reference))
    assert all(torch.equal(a, b) for a, b in zip(first, second)), first
    # upstream's path (rand_like) and the port's path (rand) are one draw.
    assert torch.equal(first[0], first[2]), first
    assert torch.equal(first[1], first[3]), first
    assert not torch.equal(first[0], torch.zeros(shape))
    if torch.backends.mps.is_available():
        on_mps = torch.zeros(shape, device='mps')
        with pinned_noise():
            assert torch.equal(torch.rand_like(on_mps).cpu(), first[2])
            assert torch.equal(torch.randn_like(on_mps).cpu(), first[3])
    saved = torch.get_rng_state().clone()
    with pinned_noise(seed=NOISE_SEED + 1):
        assert not torch.equal(torch.rand(shape), first[0])
    assert torch.equal(torch.get_rng_state(), saved), 'the pin must not disturb the stream'


def test_cpu_block_loss_and_gradients_match_upstream():
    """Upstream vs the port on CPU, through the port's own CPU quantizer.

    ``create_quantizer`` returns ``CPUOneBit`` for a CPU weight, so that is the
    port's CPU path and the thing worth comparing. Its backward is autograd
    through the same expression where upstream's is analytic (see the module
    docstring), which is exactly what the gradient tolerance here measures: a few
    fp32 ulps, not a loose bound.
    """
    result = run_pair('cpu', 'cpu')
    assert result['init_equal'], result
    assert result['driver_equal'], result
    assert result['loss_delta'] == CPU_LOSS_DELTA, result
    assert result['student_grad_delta'] == CPU_STUDENT_GRAD_DELTA, result
    assert result['logits_grad_delta'] <= CPU_LOGITS_GRAD_TOL, result
    assert result['scales_grad_delta'] <= SCALES_GRAD_TOL_CPU, result
    assert result['grad_relative'] <= CPU_GRAD_REL_TOL, result


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
def test_mps_block_loss_and_gradients_match_upstream_on_mps():
    """Upstream and the port, both on MPS, on one pinned noise sample.

    Upstream runs on MPS only under the shim: without it its forward raises,
    because ``torch.cuda.get_rng_state`` has no CUDA to talk to -
    ``test_gumbel_mps.py`` pins that. What is left after the shim and the pin is
    the port's own CPU-stream noise (one ``torch.logit`` on the other kernel), and
    that is the size of the deviation asserted here. The loss itself reduces
    identically on both sides - its perturbation is a thousandth of its own ulp -
    so it stays an exact equality and the gradients carry the deviation.
    """
    result = run_pair('mps', 'mps')
    assert result['init_equal'], result
    assert result['driver_equal'], result
    assert result['loss_delta'] == MPS_LOSS_DELTA, result
    assert result['student_grad_relative'] <= MPS_STUDENT_GRAD_REL_TOL, result
    assert result['logits_grad_delta'] <= MPS_LOGITS_GRAD_TOL, result
    assert result['scales_grad_delta'] <= SCALES_GRAD_TOL_MPS, result
    assert result['grad_relative'] <= MPS_GRAD_REL_TOL, result


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
def test_mps_block_matches_upstream_across_devices():
    """Upstream-on-CPU vs the port-on-MPS: the cross-device statement itself.

    The block's own kernels now differ as well as the quantizer's, so this axis
    carries the widest bound of the three, and it is still fp32 rounding - the loss
    agrees to one ulp of itself and the gradients to 7.7-10.6 unit roundoffs. The
    initial parameters are copied from upstream here, so what is measured is
    forward/backward arithmetic and not the two devices' reduction order.
    """
    result = run_pair('cpu', 'mps')
    assert result['init_logits_delta'] == 0.0, result
    assert result['init_scales_delta'] <= CROSS_INIT_SCALES_TOL, result
    assert result['driver_equal'], result
    assert result['loss_delta'] <= CROSS_LOSS_TOL, result
    assert result['student_grad_relative'] <= CROSS_STUDENT_GRAD_REL_TOL, result
    assert result['logits_grad_delta'] <= CROSS_LOGITS_GRAD_TOL, result
    assert result['scales_grad_delta'] <= SCALES_GRAD_TOL_MPS, result
    assert result['grad_relative'] <= CROSS_GRAD_REL_TOL, result


if __name__ == '__main__':
    # Re-measure on another machine; the tolerances above are this machine's.
    for up, port in (('cpu', 'cpu'), ('mps', 'mps'), ('cpu', 'mps')):
        if 'mps' in (up, port) and not torch.backends.mps.is_available():
            continue
        measured = run_pair(up, port)
        print(measured['axis'], {k: v for k, v in measured.items()
                                 if k not in ('axis', 'names')})
