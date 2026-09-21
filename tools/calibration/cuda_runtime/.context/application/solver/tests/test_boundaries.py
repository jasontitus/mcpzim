from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from solver.boundaries import HeadKLLoss,head_weights,EmbeddingTrainer
from solver.qwen import tiny_model


def test_streamed_stochastic_head_kl_matches_dense_gradients():
    torch.manual_seed(88)
    student=torch.randn(5,128);teacher=torch.randn(5,128)
    reference=torch.randn(13,128)*.1
    logits=torch.randn(13,128,requires_grad=True)
    scales=torch.full((13,1),.07,requires_grad=True)
    loss=HeadKLLoss.apply(student,teacher,reference,logits,scales,.8,73,2,4)
    loss.backward()
    dense_logits=logits.detach().clone().requires_grad_()
    dense_scales=scales.detach().clone().requires_grad_()
    weights=torch.cat([head_weights(dense_logits,dense_scales,start,min(start+4,13),.8,73,torch.float32)[0]
                       for start in range(0,13,4)])
    rp=(teacher@reference.T).log_softmax(-1);sp=(student@weights.T).log_softmax(-1)
    expected=(rp.exp()*(rp-sp)).sum(-1).mean();expected.backward()
    torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(logits.grad,dense_logits.grad,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(scales.grad,dense_scales.grad,atol=1e-6,rtol=1e-5)


def test_embedding_observed_rows_learn_through_first_hybrid_block():
    torch.manual_seed(5)
    model=tiny_model()
    records=[[1,3,7,2],[1,2,7,11]]
    trainer=EmbeddingTrainer(model,records)
    assert trainer.observed_ids.tolist()==[1,2,3,7,11]
    assert trainer.quantizer.sign_logits.shape[0]<model.config.vocab_size
    loss=trainer(torch.tensor([records[0]]))
    loss.backward()
    assert loss>0 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in trainer.parameters())


def test_bf16_head_backward_matches_actual_upstream(monkeypatch):
    import contextlib,importlib.util
    source=Path('/opt/upstream/gsq/src/quantization/gumbel_quantizer_1bit.py')
    spec=importlib.util.spec_from_file_location('gsq_head_oracle',source)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    fork=torch.random.fork_rng
    @contextlib.contextmanager
    def cpu_fork(*args,**kwargs):
        with fork(devices=[]):yield
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device=None:torch.get_rng_state())
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state,device=None:torch.set_rng_state(state))
    monkeypatch.setattr(torch.random,'fork_rng',cpu_fork)
    torch.manual_seed(182)
    student=torch.randn(5,128).bfloat16();teacher=torch.randn(5,128).bfloat16()
    reference=(torch.randn(13,128)*.1).bfloat16()
    logits=torch.randn(13,128,requires_grad=True);scales=torch.full((13,1),.07,requires_grad=True)
    loss=HeadKLLoss.apply(student,teacher,reference,logits,scales,.8,73,2,4);loss.backward()
    dl=logits.detach().clone().requires_grad_();ds=scales.detach().clone().requires_grad_()
    weights=[]
    for start in range(0,13,4):
        torch.manual_seed(73+start)
        weights.append(upstream.GumbelSoftmaxFunction.apply(dl[start:start+4],ds[start:start+4],
            torch.zeros(128,dtype=torch.long),.8,1.,torch.device('cpu'),torch.bfloat16))
    rp=(teacher.float()@reference.float().T).log_softmax(-1)
    sp=(student.float()@torch.cat(weights).float().T).log_softmax(-1)
    expected=(rp.exp()*(rp-sp)).sum(-1).mean();expected.backward()
    torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(logits.grad,dl.grad,atol=0,rtol=0)
    torch.testing.assert_close(scales.grad,ds.grad,atol=0,rtol=0)
