"""The determinism policy must describe the device the run actually computes on.

The defect these tests defend against: ``configure`` resolved its backend with an
internal ``resolve_device('auto')``, which prefers MPS, while ``run.main``
resolved the stage's compute device separately and the BF16 gate probed CUDA
whenever CUDA was visible. On a host with both accelerators, the exported
execution-policy report could name ``cuda`` for a run that executed on MPS, and
strict determinism would be installed on a backend the run never touched.

The contract is therefore: whatever device is handed to ``configure`` is the
backend it reports and the policy it installs.
"""
import pytest
import torch

from solver.reproducibility import configure


def test_policy_backend_matches_the_supplied_device():
    """A device passed in is the backend reported back, regardless of auto-order."""
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        policy = configure(42, torch.device(device))
        assert policy["backend"] == device, (
            f"configure({device!r}) reported backend={policy['backend']!r}; "
            "the policy must describe the device the run uses"
        )


def test_strict_determinism_is_only_claimed_where_it_holds():
    """warn_only=True and exact_arithmetic_replay=False travel together on MPS.

    Metal has no deterministic-kernel contract for the ops the solver needs, so a
    policy that silently claimed exact arithmetic replay on MPS would be the
    false-confidence bug this whole module exists to avoid.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    policy = configure(42, torch.device("mps"))
    assert policy["warn_only"] is True
    assert policy["exact_arithmetic_replay"] is False
    assert policy["caveats"], "a policy that cannot guarantee replay must say so"


def test_mps_policy_claims_rng_resume_and_says_why_it_is_safe():
    """exact_rng_resume is True on MPS, and the caveat explains the distinction.

    MPS RNG state round-trips bit-exactly (test_checkpoints.py::
    test_mps_rng_state_round_trips_exactly). Reporting False here contradicted a
    code path that runs and a test that passes.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    policy = configure(42, torch.device("mps"))
    assert policy["exact_rng_resume"] is True
    joined = " ".join(policy["caveats"])
    assert "round-trips bit-exactly" in joined, (
        "the caveat must state that RNG resume is exact and that the unguaranteed "
        "part is arithmetic, not the generator"
    )


def test_configure_without_a_device_keeps_working():
    """Callers with no compute device yet (tiny CPU tests) still get a policy."""
    policy = configure(42)
    assert policy["backend"] in ("cpu", "mps", "cuda")
    assert isinstance(policy["seed"], int)