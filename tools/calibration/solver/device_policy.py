"""Single owner of every device decision in the solver.

WHY THIS EXISTS
---------------
Device decisions in this solver were scattered: ``run.main`` decided the compute
device with its own CUDA checks, ``gsq_residency.execution_device`` validated the
configured execution device with a different allowlist, and each stage function
(``gsq_run``, ``rco_run``, ``embedding``/``head``) made its own
``device.type == 'cuda'`` comparisons for synchronization, cache flushing and
memory accounting. Porting to Apple Silicon therefore meant finding and editing
every one of those sites, and any site missed would silently keep a
CUDA-only assumption — a condition that executes on exactly one machine and
never runs in CI.

This module is the one place that changes. Everything else asks it:

    device = device_policy.resolve_device(config.get('gsq_execution_device'))

The rules encoded here:

* ``auto`` prefers MPS, then CUDA, then CPU.
* An explicit request that this machine cannot satisfy (``mps`` without MPS,
  ``cuda`` without a GPU, a misspelled device) falls back — with a warning —
  rather than raising, so a mis-set config key does not make the tool unusable.
  The fallback is always announced; it never silently claims the requested
  device.
* Every query (``is_available``, ``supports_bf16``, ``memory_stats``) is total:
  it answers for an unavailable device instead of raising, because it is called
  from logging paths inside the training loop.

Backends are probed, never version-sniffed: ``torch.backends.mps.is_built()``
and a real bf16 kernel launch say more than a macOS version string.
"""
from __future__ import annotations

import warnings

import torch

#: Devices this solver can run on. ``cuda`` is retained so an NVIDIA machine can
#: still validate the port; nothing here is written against CUDA-only behaviour.
SUPPORTED_DEVICES = ('mps', 'cpu', 'cuda')

_WARNED: set[str] = set()


def same_device(a, b):
    """Compare devices by type, ignoring an index that one side leaves implicit.

    ``torch.device('mps') != torch.device('mps', 0)``, and the same holds for
    CUDA, yet ``torch.zeros(1, device=torch.device('mps')).device`` reports
    ``mps:0``. Configs and execution policy name an accelerator by type while
    tensors report the index, so a bare ``!=`` rejects a baseline that is in
    fact resident where it should be. Type must always agree; when both sides
    state an index, it must agree too, so ``cuda:0`` and ``cuda:1`` stay
    distinct.
    """
    a, b = _device(a), _device(b)
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


def _warn_once(key, message):
    """Warn at most once per ``key`` so a per-step caller cannot spam logs."""
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, stacklevel=3)


def _device(value):
    """Coerce to ``torch.device`` without raising; unparseable -> ``cpu``."""
    try:
        return torch.device(value)
    except (TypeError, RuntimeError, ValueError):
        return torch.device('cpu')


def mps_available():
    """True if MPS is built and usable on this machine."""
    try:
        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except Exception:
        return False


def cuda_available():
    """True if CUDA is usable on this machine."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_available(device):
    """Whether ``device`` can actually be used right now.

    ``cpu`` is always available; ``meta`` never is.
    """
    return _device(device).type in ('cpu', 'cuda', 'mps') and {
        'cpu': lambda: True,
        'cuda': cuda_available,
        'mps': mps_available,
    }[_device(device).type]()


def resolve_device(preferred):
    """Resolve a requested device to one this machine can actually use.

    ``auto`` (and ``None``) prefers MPS, then CUDA, then CPU. An explicit
    request that cannot be satisfied warns and falls back to CPU (or MPS, if
    CUDA was requested on a Mac) instead of raising: a mis-set configuration key
    should not make the solver unusable, but it must never silently claim to be
    on the device it was asked for.
    """
    requested = (preferred or 'auto')
    try:
        requested = requested.type
    except AttributeError:
        requested = str(requested)
    requested = requested.lower()

    if requested == 'auto':
        if mps_available():
            return 'mps'
        if cuda_available():
            return 'cuda'
        return 'cpu'

    if requested not in SUPPORTED_DEVICES:
        _warn_once('unknown-device:' + requested,
                   'Unknown device %r; falling back to auto-detection. Supported: %s.'
                   % (requested, ', '.join(SUPPORTED_DEVICES)))
        return resolve_device('auto')

    if requested == 'mps' and not mps_available():
        _warn_once('no-mps',
                   'MPS requested but not available; using CPU. On an Apple Silicon '
                   'machine this means the torch build lacks a working MPS backend.')
        return 'cpu'

    if requested == 'cuda' and not cuda_available():
        _warn_once('no-cuda',
                   'CUDA requested but not available; falling back to %s.'
                   % ('mps' if mps_available() else 'cpu'))
        return 'mps' if mps_available() else 'cpu'

    return requested


def sync(device):
    """Block until queued work on ``device`` completes. No-op on CPU.

    MPS work is asynchronous and lazily scheduled, so an unsynchronized timing
    measurement excludes the work being timed. A device whose backend is absent
    is a no-op rather than an error: this is called from timing paths that must
    stay total, and ``torch.cuda.synchronize`` raises ``AssertionError`` (not
    ``RuntimeError``) on a build compiled without CUDA.
    """
    dev = _device(device)
    try:
        if dev.type == 'mps':
            if mps_available():torch.mps.synchronize()
        elif dev.type == 'cuda':
            if cuda_available():torch.cuda.synchronize(dev)
    except Exception as error:  # pragma: no cover - backend dependent
        _warn_once('sync:' + dev.type, 'sync(%s) failed: %s' % (dev.type, error))


def empty_cache(device=None):
    """Release cached allocations. No-op on CPU and for unknown devices."""
    dev = _device(device) if device is not None else None
    if dev is None or dev.type == 'cuda':
        if cuda_available():
            torch.cuda.empty_cache()
        return
    if dev.type == 'mps':
        torch.mps.empty_cache()


def supports_bf16(device):
    """Whether bf16 arithmetic is usable on ``device``.

    Probed, not version-sniffed: MPS bf16 depends on the installed macOS *and*
    the torch build, and CUDA bf16 depends on compute capability. A failed probe
    means fp16 or fp32 must be used instead.
    """
    dev = _device(device)
    if dev.type == 'cpu':
        return True
    if dev.type == 'cuda':
        if not cuda_available():
            return False
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False
    if dev.type == 'mps':
        if not mps_available():
            return False
        try:
            a = torch.zeros(2, 2, dtype=torch.bfloat16, device=dev)
            b = torch.ones(2, 2, dtype=torch.bfloat16, device=dev)
            return bool((a + b).sum().item() == 4.0)
        except Exception:
            return False
    return False


def memory_stats(device):
    """``(allocated_gb, reserved_gb)`` for ``device``. Never raises.

    CPU and unavailable devices report ``(0.0, 0.0)``: these numbers are written
    into progress logs inside the training loop, where raising would lose a run.
    MPS reports current and driver allocations, since it exposes no separate
    reserved pool; CUDA reports the peak counters that its allocator maintains.
    """
    dev = _device(device)
    try:
        if dev.type == 'mps':
            return (torch.mps.current_allocated_memory() / 1024**3,
                    torch.mps.driver_allocated_memory() / 1024**3)
        if dev.type == 'cuda' and cuda_available():
            return (torch.cuda.max_memory_allocated(dev) / 1024**3,
                    torch.cuda.max_memory_reserved(dev) / 1024**3)
    except Exception as error:  # pragma: no cover - backend dependent
        _warn_once('memory_stats:' + dev.type, 'memory_stats(%s) failed: %s' % (dev.type, error))
    return (0.0, 0.0)


def peak_memory_stats(device):
    """``(allocated_gb, reserved_gb)`` at their PEAK, or ``None`` if unavailable.

    Separate from :func:`memory_stats` because the two answer different questions
    and only one of them has an answer on every backend:

    * **CUDA** maintains allocator high-water marks, so a true peak is available.
    * **MPS** exposes only *current* allocation. Reading that into a field named
      ``peak`` produces a number that is neither a peak nor wrong-looking: it is
      sampled when the caller asks, which after a stage has released its training
      state is systematically *below* the real high-water mark. A reader would
      treat it as the run's peak and understate memory.

    Returning ``None`` is the honest answer where no peak exists. Callers record
    the absence rather than a substitute — see the receipt writer in ``run.py``.
    """
    dev = _device(device)
    if dev.type == 'cuda' and cuda_available():
        try:
            return (torch.cuda.max_memory_allocated(dev) / 1024**3,
                    torch.cuda.max_memory_reserved(dev) / 1024**3)
        except Exception:
            return None
    return None


def total_memory_bytes(device):
    """Advertised usable bytes for ``device``, or ``None`` when not exposed.

    ``None`` is a statement of absence, not a zero: MPS on Apple Silicon has no
    published device-memory figure because the accelerator draws on the same
    unified pages as the host. Callers that guard against overcommit must treat
    ``None`` as "unknown" and decline to decide, never as "fits".
    """
    dev = _device(device)
    if dev.type == 'cuda' and cuda_available():
        try:
            return int(torch.cuda.get_device_properties(dev).total_memory)
        except Exception:
            return None
    return None


def describe(device):
    """Human-readable one-liner for logs and bug reports."""
    dev = _device(device)
    if dev.type == 'mps':
        return 'mps (built=%s, bf16=%s)' % (mps_available(), supports_bf16(dev))
    if dev.type == 'cuda':
        if not cuda_available():
            return 'cuda (unavailable)'
        try:
            return 'cuda:%s' % torch.cuda.get_device_name(dev)
        except Exception:  # pragma: no cover - driver dependent
            return 'cuda'
    return 'cpu'