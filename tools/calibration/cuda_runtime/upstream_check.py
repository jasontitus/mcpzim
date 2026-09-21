"""Actual pinned upstream imports and small numerical integration checks.

Run GSQ and RCO in separate processes: GSQ imports src.*, RCO uses bare
common/metrics/search names. This is NOT a whole-model feasibility test.
"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from verify_sources import verify


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_budget(costs, weights, target):
    require(torch.isfinite(costs).all().item(), 'nonfinite costs')
    require(torch.isfinite(weights).all().item(), 'nonfinite weights')
    require((weights >= 0).all().item() and weights.sum().item() > 0, 'invalid weights')
    require(math.isfinite(target), 'nonfinite target')
    require(costs.min().item() * weights.sum().item() <= target <=
            costs.max().item() * weights.sum().item(), 'unattainable budget')


def gsq_check(root, device):
    sys.path.insert(0, str(root / 'gsq'))
    from src.quantization import GumbelQuantizer1Bit
    from src.models.qwen35 import Qwen35Wrapper
    from src.trainer import QuantizationTrainer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(71)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    q = torch.randn(8, 256, device=device, dtype=dtype)
    scale = torch.full((8, 2), .125, device=device)
    quant = GumbelQuantizer1Bit(q, scale, 128, .2, 1., device, dtype,
                              logits_dtype=torch.float32)
    hard, scales = quant.get_hard_weights()
    require(hard.shape == q.shape and scales.shape == scale.shape, 'wrong hard export shape')
    require(torch.equal(hard.abs(), scale.repeat_interleave(128, dim=1).to(dtype)),
            'one-bit export is not +/- group scale')
    result = {'imports': ['GumbelQuantizer1Bit', 'QuantizationTrainer', 'Qwen35Wrapper',
                          'Qwen3_5TextModel'], 'hard_export': 'passed',
              'forward_backward': 'not_run_cuda_required'}
    if device.type == 'cuda':
        optimizer = torch.optim.Adam(quant.parameters(), lr=.01)
        before = quant.sign_logits.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            loss = (quant(temperature=1.).float() - q.float()).square().mean()
            loss.backward()
            require(torch.isfinite(loss).item(), 'nonfinite GSQ loss')
            for parameter in quant.parameters():
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                        and parameter.grad.abs().sum().item() > 0, 'missing GSQ gradient')
            optimizer.step()
        require(not torch.equal(before, quant.sign_logits), 'GSQ optimizer did not update')
        result['forward_backward'] = 'passed'
    return result


def rco_check(root, device):
    sys.path.insert(0, str(root / 'rco/src'))
    from search.quant import WeightInterpolation
    from manifold import budget_normal, project_gradient, retraction, vector_transport
    from metrics import compute_kl_loss
    torch.manual_seed(83)
    # Three choices, two distinct candidate deltas; keep CPU double precision
    # for the independent autograd oracle rather than assuming BF16 parity.
    dtype = torch.float64
    alpha = torch.tensor([[.2, -.1, .4]], device=device, dtype=dtype, requires_grad=True)
    base = torch.randn(7, 5, device=device, dtype=dtype)
    deltas = [torch.randn_like(base) * .1, torch.randn_like(base) * .2]
    interpolation = WeightInterpolation(deltas, alpha, 0, cpu_deltas=True)
    x = torch.randn(2, 4, 5, device=device, dtype=dtype)
    class SmallModel(torch.nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.nn.functional.linear(x, interpolation(base)))

    ref_lp = torch.log_softmax(torch.randn(2, 3, 7, device=device), dim=-1)
    ids = torch.zeros(2, 4, dtype=torch.int64, device=device)
    mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], device=device)
    loss = compute_kl_loss(SmallModel(), ids, ref_lp, mask=mask, topk=0)
    loss.backward()
    observed = alpha.grad.detach().clone()
    oracle_alpha = alpha.detach().clone().requires_grad_()
    probs = oracle_alpha.softmax(-1)[0]
    mixed = base + probs[0] * deltas[0] + probs[1] * deltas[1]
    logits = torch.nn.functional.linear(x, mixed)[:, :-1].float()
    per_token = (ref_lp.exp() * (ref_lp - logits.log_softmax(-1))).sum(-1)
    oracle_loss = (per_token * mask[:, 1:]).sum() / mask[:, 1:].sum()
    oracle_loss.backward()
    torch.testing.assert_close(loss, oracle_loss, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(observed, oracle_alpha.grad, atol=1e-7, rtol=1e-6)
    require(torch.isfinite(observed).all().item() and observed.abs().min().item() > 0,
            'lost RCO candidate gradients')
    costs = torch.tensor([1.125, 2.125, 16.], dtype=dtype, device=device)
    weights = torch.ones(1, dtype=dtype, device=device)
    target = 2.
    validate_budget(costs, weights, target)
    retraction(alpha, costs, target, weights, tol=1e-8)
    optimizer = torch.optim.Adam([alpha], lr=.01)
    project_gradient(alpha, costs, weights)
    require(abs((alpha.grad * budget_normal(alpha, costs, weights)).sum().item()) < 1e-8,
            'RCO tangent projection failed')
    optimizer.step()
    achieved = retraction(alpha, costs, target, weights, tol=1e-8)
    vector_transport(optimizer, alpha, costs, weights)
    require(abs(achieved - target) < 1e-7, 'RCO budget update failed')
    return {'masked_full_kl_gradient_oracle': 'passed', 'candidate_gradients': 'passed',
            'project_adam_retract_transport': 'passed', 'achieved_budget': achieved,
            'scope': 'one_linear_upstream_components_not_full_model_search'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['gsq', 'rco'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--sources', type=Path, default=Path('/opt/upstream'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'stage': args.stage, 'device': args.device,
              'torch': torch.__version__, 'cuda_toolkit': torch.version.cuda,
              'whole_model_validated': False, 'gpu_validated': False}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    try:
        result['source_revisions'] = verify(args.sources)
        require(torch.__version__.split('+')[0] == '2.11.0', 'wrong torch version')
        require(torch.version.cuda == '13.0', 'wrong CUDA toolkit')
        result['compiled_architectures'] = torch._C._cuda_getArchFlags()
        require('sm_120' in result['compiled_architectures'],
                'base torch lacks RTX PRO 6000 Blackwell native architecture')
        if args.device == 'cuda':
            require(torch.cuda.is_available(), 'CUDA requested but unavailable')
            require(torch.cuda.is_bf16_supported(), 'BF16 unsupported')
            result['gpu'] = torch.cuda.get_device_name()
            result['capability'] = torch.cuda.get_device_capability()
        result['checks'] = (gsq_check if args.stage == 'gsq' else rco_check)(
            args.sources, torch.device(args.device))
        result['status'] = 'passed'
        result['gpu_validated'] = args.device == 'cuda'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
