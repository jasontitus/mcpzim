"""The optimiser the quantizer is trained with, checked against its reference rule.

Upstream trains GSQ with Lion, and the port now does too. Getting this wrong is silent:
the run still trains, still reports falling loss, and lands somewhere different. So the
update rule is asserted against a hand-computed step, including the ordering that
distinguishes Lion from Adam (the sign is taken on ``beta1 * m + (1 - beta1) * update``,
so ``beta1`` weights the *previous* moment).
"""

import torch

from solver.optim import Lion, quantizer_optimizer


def test_lion_matches_the_reference_update_rule():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad = torch.tensor([0.5, -0.25])
    optimiser = Lion([parameter], lr=0.1, betas=(0.9, 0.95), weight_decay=1.0)

    # update = grad + weight_decay * param
    update = torch.tensor([0.5 + 1.0 * 1.0, -0.25 + 1.0 * -2.0])

    # param -= lr * sign(beta1 * m + (1 - beta1) * update), with m starting at zero
    expected = torch.tensor([1.0, -2.0]) - 0.1 * torch.sign(0.1 * update)
    optimiser.step()
    assert torch.equal(parameter.detach(), expected)

    # and the moment advances to (1 - beta2) * update
    assert torch.allclose(optimiser.state[parameter]['exp_avg'], 0.05 * update)


def test_lion_is_a_sign_step_not_a_magnitude_step():
    # The distinguishing property: a ten-times larger gradient at the same moment gives
    # the same step. Adam would not, which is why substituting one for the other changes
    # the trajectory rather than only its speed.
    small = torch.nn.Parameter(torch.tensor([1.0]))
    large = torch.nn.Parameter(torch.tensor([1.0]))
    small.grad = torch.tensor([0.5])
    large.grad = torch.tensor([5.0])
    Lion([small], lr=0.1, weight_decay=0.0).step()
    Lion([large], lr=0.1, weight_decay=0.0).step()
    assert torch.equal(small.detach(), large.detach())


def test_quantizer_optimizer_splits_scales_from_logits():
    # Upstream's two groups: logits at lr1 with weight decay, scales at lr2 without.
    logits = torch.nn.Parameter(torch.zeros(3))
    scales = torch.nn.Parameter(torch.ones(2))
    named = [('quantizer.sign_logits', logits), ('quantizer.scales', scales)]
    optimiser = quantizer_optimizer(named, lr1=2e-4, lr2=1e-4, weight_decay=1.0)
    groups = optimiser.param_groups
    assert len(groups) == 2
    assert groups[0]['params'][0] is logits
    assert groups[0]['lr'] == 2e-4 and groups[0]['weight_decay'] == 1.0
    assert groups[1]['params'][0] is scales
    assert groups[1]['lr'] == 1e-4 and groups[1]['weight_decay'] == 0.0


def test_quantizer_optimizer_refuses_parameters_with_no_logits():
    import pytest

    with pytest.raises(ValueError):
        quantizer_optimizer([('quantizer.scales', torch.nn.Parameter(torch.ones(2)))])
