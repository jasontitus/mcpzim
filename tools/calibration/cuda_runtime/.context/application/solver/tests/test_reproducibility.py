import random
import numpy as np
import pytest
import torch
from solver.reproducibility import configure


def test_policy_rejects_late_setup(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(RuntimeError,match='before initializing'):configure(42)


def test_policy_seeds_all_rngs_and_enforces_strict_mode(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','incompatible')
    enabled=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn=(torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,torch.backends.cudnn.allow_tf32)
    tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        policy=configure(19)
        expected=(random.random(),np.random.rand(),torch.rand(3))
        configure(19)
        assert random.random()==expected[0] and np.random.rand()==expected[1]
        torch.testing.assert_close(torch.rand(3),expected[2],atol=0,rtol=0)
        assert policy['cublas_workspace_config']==':4096:8'
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn)
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,torch.backends.cudnn.allow_tf32=cudnn
        torch.backends.cuda.matmul.allow_tf32=tf32
