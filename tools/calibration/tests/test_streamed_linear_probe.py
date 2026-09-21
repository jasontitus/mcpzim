"""Numerical checks against independent dense autograd; no model download."""

import pytest
import torch

from streamed_linear_probe import streamed_linear


def weights(out_features, in_features, count=3, dtype=torch.float64):
    return (
        torch.randn(out_features, in_features, dtype=dtype) * 0.2,
        tuple(torch.randn(out_features, in_features, dtype=dtype) * 0.1 for _ in range(count)),
    )


def dense(x, p, ref, deltas):
    w = ref.to(x.device) + sum(p[i] * d.to(x.device) for i, d in enumerate(deltas))
    return x @ w.T


@pytest.mark.parametrize("tile", [1, 3, 32])
@pytest.mark.parametrize("hard", [False, True])
def test_two_block_loss_gradients_and_adam_match(tile, hard):
    torch.manual_seed(914)
    layers = [weights(7, 5), weights(4, 7)]
    original_x = torch.randn(2, 3, 5, dtype=torch.float64)
    original_alpha = torch.randn(2, 3, dtype=torch.float64)
    noise = torch.randn_like(original_alpha)
    results = []
    for stream in [False, True]:
        x = original_x.clone().requires_grad_()
        alpha = original_alpha.clone().requires_grad_()
        optimizer = torch.optim.Adam([alpha], lr=0.03)
        h = x
        for i, (ref, deltas) in enumerate(layers):
            soft = torch.softmax((alpha[i] + noise[i]) / 0.7, dim=-1)
            # Fixed hard selection isolates the STE derivative from the DP solver.
            onehot = torch.tensor([0., 1., 0.], dtype=soft.dtype)
            p = onehot - soft.detach() + soft if hard else soft
            h = streamed_linear(h, p, ref, deltas, tile) if stream else dense(h, p, ref, deltas)
            h = torch.tanh(h)
        loss = (h.sin() - 0.3).square().mean()
        loss.backward()
        assert torch.isfinite(alpha.grad).all()
        assert torch.count_nonzero(alpha.grad) == alpha.numel()
        before = (h.detach(), loss.detach(), x.grad.clone(), alpha.grad.clone())
        optimizer.step()
        results.append((*before, alpha.detach().clone()))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_probability_and_input_gradcheck():
    torch.manual_seed(22)
    ref, deltas = weights(5, 3)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    p = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: streamed_linear(a, b, ref, deltas, 2), (x, p))


def test_zero_forward_probability_still_receives_gradient():
    ref = torch.zeros(2, 2, dtype=torch.float64)
    deltas = (torch.ones_like(ref), 2 * torch.ones_like(ref))
    p = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
    streamed_linear(torch.ones(1, 2, dtype=p.dtype), p, ref, deltas, 1).sum().backward()
    torch.testing.assert_close(p.grad, torch.tensor([4., 8.], dtype=p.dtype))


def test_rejects_trainable_weights():
    ref, deltas = weights(2, 3)
    with pytest.raises(ValueError, match="frozen"):
        streamed_linear(torch.ones(1, 3, dtype=ref.dtype), torch.ones(3, dtype=ref.dtype), ref.requires_grad_(), deltas)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal unavailable")
def test_metal_forward_and_backward_with_cpu_storage():
    torch.manual_seed(25)
    ref, deltas = weights(9, 5, dtype=torch.float32)
    original_x = torch.randn(2, 5)
    original_p = torch.tensor([0.2, 0.0, 0.8])
    results = []
    for stream in [False, True]:
        x = original_x.to("mps").requires_grad_()
        p = original_p.to("mps").requires_grad_()
        y = streamed_linear(x, p, ref, deltas, 4) if stream else dense(x, p, ref, deltas)
        y.square().sum().backward()
        results.append((y.detach().cpu(), x.grad.cpu(), p.grad.cpu()))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
