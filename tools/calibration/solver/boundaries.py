"""Learned Q1 boundary stages: observed-token embedding GSQ and full-KL head GSQ."""
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .gsq import create_quantizer
from .qwen import run_block
from .candidates import pack_signs,save_candidate


class EmbeddingTrainer(torch.nn.Module):
    def __init__(self,model,records,upstream=None):
        from .upstream_paths import resolve as _resolve_upstream
        upstream=_resolve_upstream(upstream)
        super().__init__();object.__setattr__(self,'model',model)
        ids=torch.tensor(sorted(set(t for tokens in records for t in tokens)),
                         device=model.model.embed_tokens.weight.device,dtype=torch.long)
        if not len(ids) or ids.min()<0 or ids.max()>=model.config.vocab_size:
            raise ValueError('Invalid full-corpus embedding token IDs')
        self.register_buffer('observed_ids',ids)
        self.quantizer=create_quantizer(model.model.embed_tokens.weight[ids],upstream)
    def forward(self,ids,temperature=1.,scale=1.):
        offsets=torch.searchsorted(self.observed_ids,ids)
        if (offsets>=len(self.observed_ids)).any():raise ValueError('Unknown observed token')
        if not torch.equal(self.observed_ids[offsets],ids):raise ValueError('Unknown observed token')
        with torch.no_grad():
            target=run_block(self.model,0,self.model.model.embed_tokens(ids))
        weights=self.quantizer(temperature,scale).to(self.model.model.embed_tokens.weight.dtype)
        embedded=torch.nn.functional.embedding(offsets,weights)
        actual=run_block(self.model,0,embedded)
        return (actual.float()-target.float()).square().mean()
    def export(self,initial,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        from .candidates import Q1Candidate
        candidate=Q1Candidate(initial)
        if candidate.shape!=tuple(self.model.model.embed_tokens.weight.shape):raise ValueError('Embedding seed shape differs')
        candidate.check_immutable()
        with safe_open(str(initial),framework='pt') as f:
            codes=f.get_tensor('codes').clone();scales=f.get_tensor('scales').clone()
        ids=self.observed_ids.cpu()
        codes[ids]=pack_signs(self.quantizer.sign_logits).cpu()
        scales[ids]=self.quantizer.scales.detach().bfloat16().cpu()
        save_file({'codes':codes,'scales':scales},str(path),metadata={
            'format':'zimfo-q1-v1','group_size':'128','bit_order':'little',
            'effective_scale_dtype':'BF16','zero_logit_sign':'negative',
            'optimization':'first-hybrid-block-GSQ','observed_rows':str(len(ids)),
            'unobserved_rows':'RTN initialization'})


def head_weights(logits,scales,start,end,temperature,seed,dtype):
    generator=torch.Generator(device=logits.device).manual_seed(seed+start)
    u=torch.rand(logits[start:end].shape,device=logits.device,generator=generator)
    soft=(2*torch.sigmoid((2*logits[start:end]+torch.logit(u,eps=1e-8))/temperature)-1).to(dtype)
    weights=(soft*scales[start:end].to(dtype).repeat_interleave(128,1)).float()
    return weights,soft


class HeadKLLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,student,teacher,reference,logits,scales,temperature,seed,tokens,rows):
        import math
        if student.ndim!=2 or student.shape!=teacher.shape or not len(student) or reference.ndim!=2 or logits.shape!=reference.shape or reference.shape[1]!=student.shape[1] or scales.shape!=(len(reference),reference.shape[1]//128):
            raise ValueError('Invalid head shapes')
        if student.requires_grad or teacher.requires_grad or reference.requires_grad:
            raise ValueError('Head stage requires frozen full-corpus hidden states/reference')
        if not math.isfinite(temperature) or temperature<=0 or tokens<=0 or rows<=0:raise ValueError('Invalid head chunk/temperature')
        if not torch.isfinite(scales).all() or not torch.isfinite(logits).all():raise ValueError('Nonfinite head parameters')
        norms=[];loss=torch.zeros((),device=student.device)
        for t in range(0,len(student),tokens):
            s=student[t:t+tokens].float();r=teacher[t:t+tokens].float()
            zs=torch.full((len(s),),-torch.inf,device=s.device);zr=zs.clone()
            for v in range(0,len(reference),rows):
                end=min(v+rows,len(reference))
                w,_=head_weights(logits,scales,v,end,temperature,seed,reference.dtype)
                zs=torch.logaddexp(zs,(s@w.T).logsumexp(-1))
                zr=torch.logaddexp(zr,(r@reference[v:end].float().T).logsumexp(-1))
            norms.append(torch.stack((zs,zr)))
            for v in range(0,len(reference),rows):
                end=min(v+rows,len(reference))
                w,_=head_weights(logits,scales,v,end,temperature,seed,reference.dtype)
                sp=s@w.T-zs[:,None];rp=r@reference[v:end].float().T-zr[:,None]
                loss+=(rp.exp()*(rp-sp)).sum()
        ctx.save_for_backward(student,teacher,reference,logits,scales,torch.cat(norms,1))
        ctx.temperature,ctx.seed,ctx.tokens,ctx.rows=temperature,seed,tokens,rows
        return loss/len(student)
    @staticmethod
    def backward(ctx,gradient):
        student,teacher,reference,logits,scales,norms=ctx.saved_tensors
        gl=torch.zeros_like(logits);gs=torch.zeros_like(scales)
        for v in range(0,len(reference),ctx.rows):
            end=min(v+ctx.rows,len(reference))
            weights,soft=head_weights(logits,scales,v,end,ctx.temperature,ctx.seed,reference.dtype)
            gw=torch.zeros_like(weights)
            for t in range(0,len(student),ctx.tokens):
                s=student[t:t+ctx.tokens].float();r=teacher[t:t+ctx.tokens].float()
                zs,zr=norms[:,t:t+len(s)]
                ps=(s@weights.T-zs[:,None]).exp()
                pr=(r@reference[v:end].float().T-zr[:,None]).exp()
                gw.add_(((ps-pr)*(gradient.float()/len(student))).T@s)
            rounded_gw=gw.to(reference.dtype)
            # Match upstream GSQ1Bit's cast boundaries and FP32 scale reduction.
            gl[v:end]=(rounded_gw*scales[v:end].to(reference.dtype).repeat_interleave(128,1)*(1-soft.square())/ctx.temperature).to(logits.dtype)
            gs[v:end]=(rounded_gw*soft).float().reshape(end-v,-1,128).sum(-1)
        return None,None,None,gl,gs,None,None,None,None,None


class HeadTrainer(torch.nn.Module):
    def __init__(self,reference):
        super().__init__();object.__setattr__(self,'reference',reference)
        scales=reference.float().reshape(reference.shape[0],-1,128).abs().mean(-1).clamp_min(1e-8)
        self.scales=torch.nn.Parameter(scales)
        self.sign_logits=torch.nn.Parameter(.2*(torch.randn_like(reference.float())+
                                                torch.where(reference>0,1.,-1.)))
    def forward(self,student,teacher,temperature=1.,tokens=32,rows=1024):
        seed=int(torch.randint(0,2**30,()).item())
        return HeadKLLoss.apply(student,teacher,self.reference,self.sign_logits,self.scales,
                                temperature,seed,tokens,rows)
    def export(self,path):save_candidate(path,self.sign_logits,self.scales)
