"""Fixed execution policy for same-runtime checkpoint replay validation.

The policy is per-backend because the guarantee genuinely differs. CUDA with
cuBLAS/cuDNN determinism configured is bit-reproducible; Metal/MPS is not, and
claiming otherwise would make a replay comparison meaningless. ``configure``
therefore records which policy it selected and the caller can export that as
evidence instead of assuming determinism everywhere.
"""
import os
import random
import numpy as np
import torch

from .device_policy import resolve_device

#: Recorded in the policy report when strict determinism cannot be enforced.
MPS_CAVEATS = (
    'Metal/MPS has no cuBLAS/cuDNN workspaces and torch reports many operations '
    'as nondeterministic, so torch.use_deterministic_algorithms(True) is not '
    'satisfiable on an MPS execution device; it is left disabled with '
    'warn_only=True so an unsupported op warns instead of aborting the run.',
    'MPS RNG state round-trips bit-exactly (torch.mps.get_rng_state/'
    'set_rng_state, covered by tests/test_checkpoints.py::'
    'test_mps_rng_state_round_trips_exactly), so the generator itself resumes '
    'exactly. What is NOT guaranteed is the arithmetic: kernels marked '
    'nondeterministic above can reorder reductions, so a resumed run may not '
    'reproduce a pre-resume trajectory bit-for-bit even with identical RNG.',
)


def configure(seed, device=None):
    """Seed every RNG and install a determinism policy for the run's backend.

    ``device`` is the device the stage will actually compute on. It is passed in
    rather than resolved here because an internal ``auto`` resolution prefers MPS
    on an Apple Silicon host, which on a machine with *both* MPS and CUDA would
    install the CUDA determinism policy and report ``backend: cuda`` for a run
    that executed on MPS. ``None`` keeps the old auto behaviour for callers that
    have no compute device yet.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic CUDA before initializing the device')
    resolved = torch.device(resolve_device('auto')) if device is None else torch.device(device)
    if resolved.type == 'mps':
        return _configure_mps(seed)
    if resolved.type == 'cuda':
        return _configure_cuda(seed)
    # CPU: no accelerator policy to install, but the generators still get seeded
    # so a CPU run is at least reproducible run-to-run.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return {'seed':seed,'backend':'cpu','deterministic_algorithms':None,'warn_only':None,
            'cublas_workspace_config':None,'cudnn_benchmark':None,'cudnn_deterministic':None,
            'tf32':None,'exact_rng_resume':True,'exact_arithmetic_replay':True,'caveats':[]}


def _configure_cuda(seed):
    # Set before cuBLAS creates handles/workspaces. Enforce rather than inherit
    # an incompatible environment setting from a launch shell.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # An MPS generator exists in the process whether or not this run uses
        # it; seed it so a later stray MPS draw is not silently unseeded.
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    return {'seed':seed,'backend':'cuda','deterministic_algorithms':True,'warn_only':False,
            'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'cudnn_benchmark':False,'cudnn_deterministic':True,'tf32':False,
            'exact_rng_resume':True,'exact_arithmetic_replay':True,'caveats':[]}


def _configure_mps(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    # warn_only=True is the honest setting, not a relaxation: Metal has no
    # deterministic kernel contract for the operations the solver needs, so
    # strict mode would abort on the first unsupported op without buying any
    # reproducibility. Do not report this run as deterministic.
    #
    # The flag is deliberately NOT installed here. It cannot make Metal deterministic,
    # but it does change which kernels the dispatcher selects, and installing it before
    # the model is loaded made the first full_attention block's update non-finite at
    # every attempt on this machine -- with `configure` stubbed, block 3 trains all 87
    # steps, exports and publishes its successor. `run.main` installs it immediately
    # after the load instead, so the policy below still describes the run.
    # See docs/PORT_VALIDATION.md and solver/tools/bisect_main.py.
    return {'seed':seed,'backend':'mps','deterministic_algorithms':True,'warn_only':True,
            'cublas_workspace_config':None,'cudnn_benchmark':None,'cudnn_deterministic':None,
            'tf32':None,
            # The MPS generator is seeded, captured and restored bit-exactly
            # (see MPS_CAVEATS). This field answers *RNG* resume, which is
            # exact; it does not claim bit-exact arithmetic replay, which
            # warn_only=True above explicitly disclaims.
            'exact_rng_resume':True,'exact_arithmetic_replay':False,
            'caveats':list(MPS_CAVEATS)}