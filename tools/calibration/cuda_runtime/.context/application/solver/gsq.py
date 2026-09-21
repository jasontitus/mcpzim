"""Resumable GSQ block training using upstream CUDA one-bit quantizers.

CPU mathematical counterpart is for tiny tests only. Teacher targets come from
original block inputs; student inputs include the already quantized prefix.
"""
import contextlib
import sys
from pathlib import Path
import torch
from .qwen import block_kwargs
from .candidates import save_candidate


class CPUOneBit(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        scales=weight.float().reshape(weight.shape[0],-1,128).abs().mean(-1).clamp_min(1e-8)
        self.scales=torch.nn.Parameter(scales.float())
        self.sign_logits=torch.nn.Parameter(.2*(torch.randn_like(weight.float())+
                                                torch.where(weight>0,1.,-1.)))
    def forward(self, temperature, scale=1.):
        u=torch.rand_like(self.sign_logits)
        soft=2*torch.sigmoid((2*self.sign_logits*scale+torch.logit(u,eps=1e-8))/temperature)-1
        return soft*self.scales.repeat_interleave(128,1)
    def get_hard_weights(self):
        scales=self.scales.bfloat16().float()
        return torch.where(self.sign_logits>0,1.,-1.)*scales.repeat_interleave(128,1),scales


def create_quantizer(weight, upstream):
    if weight.device.type=='cpu':
        return CPUOneBit(weight)
    sys.path.insert(0,str(Path(upstream)/'gsq'))
    from src.quantization import GumbelQuantizer1Bit
    scales=weight.float().reshape(weight.shape[0],-1,128).abs().mean(-1).clamp_min(1e-8)
    initial=torch.where(weight>0,1.,-1.)*scales.repeat_interleave(128,1)
    return GumbelQuantizer1Bit(initial.to(weight.dtype),scales,128,.2,1.,weight.device,
                              weight.dtype,logits_dtype=torch.float32)


class BlockTrainer(torch.nn.Module):
    def __init__(self, model, block_index, upstream='/opt/upstream'):
        super().__init__()
        # Do not register the full 54GB frozen model in this state_dict.
        object.__setattr__(self,'model',model)
        self.block_index=block_index
        self.names=[]
        quantizers=[]
        block=model.model.layers[block_index]
        for name,module in block.named_modules():
            if isinstance(module,torch.nn.Linear) and module.weight.shape[1]%128==0:
                self.names.append(name)
                quantizers.append(create_quantizer(module.weight,upstream))
        if not self.names:
            raise ValueError('No eligible block projections')
        self.quantizers=torch.nn.ModuleList(quantizers)

    def forward(self, student, teacher, temperature=1.):
        block=self.model.model.layers[self.block_index]
        with torch.no_grad():
            target=block(teacher,**block_kwargs(self.model,teacher,self.block_index))
        replacements={name+'.weight':quantizer(temperature).to(student.dtype)
                      for name,quantizer in zip(self.names,self.quantizers)}
        output=torch.func.functional_call(block,replacements,(student,),
                                         block_kwargs(self.model,student,self.block_index),strict=False)
        return (output.float()-target.float()).square().mean()

    @torch.no_grad()
    def hard_forward(self, student):
        block=self.model.model.layers[self.block_index]
        replacements={name+'.weight':q.get_hard_weights()[0].to(student.dtype)
                      for name,q in zip(self.names,self.quantizers)}
        return torch.func.functional_call(block,replacements,(student,),
                   block_kwargs(self.model,student,self.block_index),strict=False)

    @torch.no_grad()
    def export(self, directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        records={}
        for name,quantizer in zip(self.names,self.quantizers):
            path=directory/(name+'.safetensors')
            if path.exists():
                from .candidates import Q1Candidate
                candidate=Q1Candidate(path)
                expected=quantizer.get_hard_weights()[0]
                for row in range(0,len(expected),512):
                    end=min(row+512,len(expected))
                    torch.testing.assert_close(candidate.rows(row,end,expected.device,expected.dtype),
                                               expected[row:end],rtol=0,atol=0)
            else:
                save_candidate(path,quantizer.sign_logits,quantizer.scales)
            records[f'model.layers.{self.block_index}.{name}']=str(path)
        return records
