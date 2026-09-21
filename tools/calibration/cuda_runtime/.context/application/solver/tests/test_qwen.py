from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from solver.qwen import tiny_model,run_block
from solver.gsq import BlockTrainer


def test_tiny_real_hybrid_blocks_match_model_and_train():
    torch.manual_seed(90)
    model=tiny_model()
    ids=torch.tensor([[1,5,9,7,3]])
    with torch.no_grad():
        hidden=model.model.embed_tokens(ids)
        first=hidden.clone()
        for index in range(2):
            hidden=run_block(model,index,hidden)
        expected=model.model(input_ids=ids,use_cache=False).last_hidden_state
        torch.testing.assert_close(model.model.norm(hidden),expected)
    for index in range(2):
        trainer=BlockTrainer(model,index)
        optimizer=torch.optim.Adam(trainer.parameters(),lr=.001)
        before=[p.detach().clone() for p in trainer.parameters()]
        loss=trainer(first,first)
        loss.backward()
        assert torch.isfinite(loss) and loss>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.parameters())
        optimizer.step()
        assert any(not torch.equal(a,b) for a,b in zip(before,trainer.parameters()))
        first=run_block(model,index,first).detach()


def test_native_sdpa_matches_eager_hybrid_forward():
    torch.manual_seed(19)
    model=tiny_model();ids=torch.tensor([[1,8,3,5,9]])
    with torch.no_grad():
        sdpa=model(input_ids=ids,use_cache=False).logits
        model.config._attn_implementation='eager'
        eager=model(input_ids=ids,use_cache=False).logits
    torch.testing.assert_close(sdpa,eager,rtol=1e-5,atol=1e-6)


def test_original_bf16_loader_does_not_inherit_fp32_meta_dtype():
    from accelerate import init_empty_weights
    from solver.qwen import install_original_tensor
    with init_empty_weights():model=torch.nn.Linear(128,32,bias=False)
    assert model.weight.dtype==torch.float32 and model.weight.device.type=='meta'
    original=torch.randn(32,128).bfloat16()
    install_original_tensor(model,'weight',original,torch.device('cpu'))
    assert model.weight.dtype==torch.bfloat16
    torch.testing.assert_close(model.weight,original,rtol=0,atol=0)
