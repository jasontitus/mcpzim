"""Fixed execution policy for same-runtime checkpoint replay validation."""
import os
import random
import numpy as np
import torch


def configure(seed):
    if torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic CUDA before initializing the device')
    # Set before cuBLAS creates handles/workspaces. Enforce rather than inherit
    # an incompatible environment setting from a launch shell.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    return {'seed':seed,'deterministic_algorithms':True,'warn_only':False,
            'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'cudnn_benchmark':False,'cudnn_deterministic':True,'tf32':False}
