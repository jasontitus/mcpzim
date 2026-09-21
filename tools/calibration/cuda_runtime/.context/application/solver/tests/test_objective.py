from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from solver.candidates import Q1Candidate, StreamedLinear, save_candidate
from solver.objective import full_kl


def candidate(tmp_path, rows=17):
    torch.manual_seed(12)
    signs = torch.randn(rows, 128)
    scales = torch.rand(rows, 1) * .1
    path = tmp_path / 'q1.safetensors'
    save_candidate(path, signs, scales)
    return Q1Candidate(path)


@pytest.mark.parametrize('chunks', [(1,1), (3,7), (200,200)])
def test_full_kl_and_gradients_match_dense(tmp_path, chunks):
    q = candidate(tmp_path)
    torch.manual_seed(3)
    reference = torch.randn(17, 128) * .1
    teacher = torch.randn(9, 128)
    student = torch.randn(9, 128, requires_grad=True)
    p = torch.tensor(.7, requires_grad=True)
    loss = full_kl(student, teacher, reference, p, q, *chunks)
    loss.backward()
    x = student.detach().clone().requires_grad_()
    pp = p.detach().clone().requires_grad_()
    mixed = reference + pp * (q.rows(0,17,'cpu',torch.float32)-reference)
    rp = (teacher @ reference.T).log_softmax(-1)
    sp = (x @ mixed.T).log_softmax(-1)
    oracle = (rp.exp() * (rp-sp)).sum(-1).mean()
    oracle.backward()
    torch.testing.assert_close(loss, oracle, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(student.grad, x.grad, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(p.grad, pp.grad, atol=2e-6, rtol=2e-6)


def test_streamed_linear_gradient_matches_dense(tmp_path):
    q = candidate(tmp_path)
    reference = torch.randn(17,128) * .1
    x = torch.randn(2,3,128,requires_grad=True)
    p = torch.tensor(.3,requires_grad=True)
    output = StreamedLinear.apply(x, reference, p, q, 5)
    output.square().sum().backward()
    xx = x.detach().clone().requires_grad_()
    pp = p.detach().clone().requires_grad_()
    dense = torch.nn.functional.linear(xx, reference + pp*(q.rows(0,17,'cpu',torch.float32)-reference))
    dense.square().sum().backward()
    torch.testing.assert_close(output,dense)
    torch.testing.assert_close(x.grad,xx.grad,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(p.grad,pp.grad,atol=1e-5,rtol=1e-5)


def test_q1_ties_and_effective_bf16_scales(tmp_path):
    path=tmp_path/'q1.safetensors'
    logits=torch.zeros(2,128);logits[:,1]=1
    scales=torch.tensor([[.123456],[.234567]])
    save_candidate(path,logits,scales)
    q=Q1Candidate(path)
    expected=torch.where(logits>0,1.,-1.)*scales.bfloat16().float()
    torch.testing.assert_close(q.rows(0,2,'cpu',torch.float32),expected,rtol=0,atol=0)


def test_full_kl_reference_identity_zero(tmp_path):
    x=torch.randn(7,128,requires_grad=True)
    head=torch.randn(19,128)*.1
    p=torch.tensor(0.,requires_grad=True)
    loss=full_kl(x,x.detach(),head,p,None,2,5)
    loss.backward()
    assert abs(loss.item())<1e-6
    assert x.grad.abs().max()<1e-6


def test_reject_empty_or_unfrozen_reference():
    with pytest.raises(ValueError):
        full_kl(torch.empty(0,128),torch.empty(0,128),torch.zeros(2,128),torch.tensor(.5))


@pytest.mark.parametrize("row_chunk", [1, 5, 41])
def test_streamed_bf16_linear_preserves_weight_gradient_cast(tmp_path, row_chunk):
    """Allocation gradients must differentiate the actual BF16 weight cast."""
    torch.manual_seed(3)
    path = tmp_path / 'bf16-q1.safetensors'
    save_candidate(path, torch.randn(41, 128), torch.rand(41, 1) * .1)
    q = Q1Candidate(path)
    reference = (torch.randn(41, 128) * .1).bfloat16()
    x = torch.randn(2, 7, 128).bfloat16().requires_grad_()
    probability = torch.tensor(.3, requires_grad=True)
    actual = StreamedLinear.apply(x, reference, probability, q, row_chunk)
    actual.float().square().sum().backward()

    dense_x = x.detach().clone().requires_grad_()
    dense_probability = probability.detach().clone().requires_grad_()
    delta = q.rows(0, 41, 'cpu', torch.float32) - reference.float()
    mixed = (reference.float() + dense_probability * delta).bfloat16()
    expected = torch.nn.functional.linear(dense_x, mixed)
    expected.float().square().sum().backward()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, dense_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(probability.grad, dense_probability.grad, atol=1e-5, rtol=1e-6)
