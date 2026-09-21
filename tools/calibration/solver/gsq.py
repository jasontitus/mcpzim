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


#: Upstream's initialisation, from ``gumbel_quantizer_1bit.py:19``:
#: ``sign_logits = std * (randn_like(Q) + Q * strength)`` where ``Q = W / scale``.
#: The port previously used ``std=0.2, strength=1`` on a sign-only ``Q``, which
#: discards the magnitude information upstream keeps and flips roughly 16% of the
#: starting signs off the RTN initialisation it is meant to start from.
INIT_STD = 0.01
INIT_STRENGTH = 6


class CPUOneBit(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        scales=weight.float().reshape(weight.shape[0],-1,128).abs().mean(-1).clamp_min(1e-8)
        self.scales=torch.nn.Parameter(scales.float())
        normalized=torch.where(weight>0,1.,-1.)      # RTN: the sign grid, as upstream passes it
        self.sign_logits=torch.nn.Parameter(INIT_STD*(torch.randn_like(normalized)+
                                                      normalized*INIT_STRENGTH))
    def forward(self, temperature, scale=1.):
        u=torch.rand_like(self.sign_logits)
        soft=2*torch.sigmoid((2*self.sign_logits*scale+torch.logit(u,eps=1e-8))/temperature)-1
        return soft*self.scales.repeat_interleave(128,1)
    def get_hard_weights(self):
        scales=self.scales.bfloat16().float()
        return torch.where(self.sign_logits>0,1.,-1.)*scales.repeat_interleave(128,1),scales


def create_quantizer(weight, upstream):
    """Build the one-bit quantizer for ``weight``'s device.

    Upstream's ``GumbelQuantizer1Bit`` is CPU-inexact (``CPUOneBit``) or
    CUDA-only (the RNG replay); ``MPSOneBit`` is the same algebra with the noise
    drawn on the global CPU stream and stored rather than replayed, so it runs
    on every backend. CUDA keeps upstream's implementation unchanged -- the
    device policy of this port is to add a backend, not to replace a working
    one.
    """
    scales=weight.float().reshape(weight.shape[0],-1,128).abs().mean(-1).clamp_min(1e-8)
    if weight.device.type=='cpu':
        return CPUOneBit(weight)
    # Upstream's call sites pass Q = the RTN initialisation, i.e. sign(w) * mean|w|
    # per group (trainer.py:47, prior/gptq.py:236, with quant.py's maxq==-2 branch
    # producing it), not the raw weight. The quantizer then divides by the group
    # scale, so the logits start as INIT_STD * (randn + sign(w) * INIT_STRENGTH).
    # An earlier revision of this change passed the raw weight instead, which is a
    # different distribution; the deviation that mattered was the constants, not
    # the form.
    initial=torch.where(weight>0,1.,-1.)*scales.repeat_interleave(128,1)
    if weight.device.type!='cuda':
        from .gumbel_mps import MPSOneBit
        return MPSOneBit(initial.to(weight.dtype),scales,128,INIT_STD,INIT_STRENGTH,weight.device,
                         weight.dtype,logits_dtype=torch.float32)
    sys.path.insert(0,str(Path(upstream)/'gsq'))
    from src.quantization import GumbelQuantizer1Bit
    return GumbelQuantizer1Bit(initial.to(weight.dtype),scales,128,INIT_STD,INIT_STRENGTH,weight.device,
                              weight.dtype,logits_dtype=torch.float32)


class BlockTrainer(torch.nn.Module):
    def __init__(self, model, block_index, upstream=None):
        from .upstream_paths import resolve as _resolve_upstream
        upstream=_resolve_upstream(upstream)
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

    def forward(self, student, teacher, temperature=1., scale=1.):
        """Reconstruction loss against the *clean* stream's output.

        The input is the composed quantized prefix - ``student``, what this block
        actually sees at inference - and the target is the unquantized block
        applied to ``teacher``, the *original* stream's input to this block. The
        pair is therefore "drifted input, clean output", which is the only form
        whose gradient opposes composition drift: a target computed from the same
        drifted input as the student's costs the calibration outright, because the
        block is then rewarded for reproducing the drift faithfully and the loss
        stays small however far the composed model walks from the original.

        Upstream (src/models/base.py:321-353, src/trainer.py) precomputes
        activations through the unquantized cascade once and never updates them, so
        its blocks train on the clean stream with a clean target and never observe
        the drift they cause at inference. Feeding the drifted stream in while
        keeping the clean target is the strictly stronger form of the same
        objective, and the drift this port measures is why it is needed.

        ``scale`` and ``temperature`` are upstream's annealed schedules; the
        callers pass them per optimizer step (see the update loop in run.py).
        """
        block=self.model.model.layers[self.block_index]
        with torch.no_grad():
            target=block(teacher,**block_kwargs(self.model,teacher,self.block_index))
        replacements={name+'.weight':quantizer(temperature,scale).to(student.dtype)
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
