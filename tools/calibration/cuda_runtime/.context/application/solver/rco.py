"""Memory-bounded candidate interpolation with upstream RCO budget operations."""
import contextlib
import sys
from pathlib import Path
import types
import torch
from .qwen import projection_modules
from .candidates import StreamedLinear,Q1Candidate
from .objective import full_kl


def gather_candidate(candidate, ids, reference, rows=512):
    flat=ids.flatten()
    result=reference.new_empty((flat.numel(),reference.shape[1]))
    # Read each touched row tile once, even when tokens repeat.
    for tile in (flat//rows).unique().tolist():
        start=int(tile)*rows;end=min(start+rows,candidate.shape[0])
        selected=(flat>=start)&(flat<end)
        decoded=candidate.rows(start,end,reference.device,reference.dtype)
        result[selected]=decoded[flat[selected]-start]
    return result.reshape(*ids.shape,reference.shape[1])


class RCOTrainer(torch.nn.Module):
    def __init__(self, model, database, target_bytes, fixed_bytes=0, upstream='/opt/upstream',
                 row_chunk=512, token_chunk=32, vocab_chunk=2048, cost_manifest=None):
        super().__init__()
        object.__setattr__(self,'model',model)
        sys.path.insert(0,str(Path(upstream)/'rco/src'))
        from manifold import project_gradient,retraction,vector_transport
        from search.quant import budget_constrained_argmax
        self.project_gradient=project_gradient;self.retraction=retraction;self.vector_transport=vector_transport
        self.budget_constrained_argmax=budget_constrained_argmax
        self.modules_by_name=projection_modules(model)
        if set(database)!=set(self.modules_by_name):
            raise ValueError('Candidate database does not cover every eligible projection/embedding/head')
        self.names=sorted(database)
        self.candidates={name:Q1Candidate(database[name]) for name in self.names}
        device=next(model.parameters()).device
        self.alpha=torch.nn.Parameter(torch.tensor([[2.,-2.]]*len(self.names),device=device))
        counts=torch.tensor([self.modules_by_name[n].weight.numel() for n in self.names],device=device,dtype=torch.float64)
        self.weights=(counts/counts.sum()).float()
        self.byte_costs={name:[self.candidates[name].packed_weight_bytes,
                              self.modules_by_name[name].weight.numel()*2] for name in self.names}
        self.serialized_costs_verified=False
        if cost_manifest is not None:
            mapped={('model.'+name.removeprefix('model.language_model.') if name.startswith('model.language_model.') else name).removesuffix('.weight'):item
                    for name,item in cost_manifest['tensors'].items()}
            if set(mapped)!=set(self.names):raise ValueError('Serialized cost manifest coverage mismatch')
            for name,item in mapped.items():
                if item['shape']!=list(self.modules_by_name[name].weight.shape):
                    raise ValueError('Serialized cost manifest shape mismatch')
                self.byte_costs[name]=[item['q1_bytes'],item['bf16_bytes']]
            if fixed_bytes!=cost_manifest['fixed_container_bytes']:
                raise ValueError('Protected/metadata byte budget mismatch')
            self.serialized_costs_verified=True
        self.costs=torch.tensor([self.byte_costs[name] for name in self.names],device=device).float()*8/counts.float()[:,None]
        self.nominal_costs=torch.tensor([1.125,16.],device=device)
        self.target=(target_bytes-fixed_bytes)*8/counts.sum().item()
        if not (self.weights*self.costs[:,0]).sum().item()-1e-6 <= self.target <= (self.weights*self.costs[:,1]).sum().item()+1e-6:
            raise ValueError('Requested byte budget is infeasible with Q1+BF16 candidates and protected tensors')
        self.retraction(self.alpha,self.costs,self.target,self.weights,tol=1e-6)
        self.row_chunk,self.token_chunk,self.vocab_chunk=row_chunk,token_chunk,vocab_chunk
        self.group_index={n:i for i,n in enumerate(self.names)}
        self.probabilities=None

    def assignment(self, logits):
        assignment=self.budget_constrained_argmax(logits,self.weights,self.nominal_costs,self.target)
        # Upstream DP rounds costs; repair rounding overflow against exact costs.
        def cost():return (self.weights*self.costs.gather(1,assignment[:,None]).squeeze(1)).sum().item()
        current=cost()
        if current>self.target+1e-7:
            upgrades=[i for i in range(len(self.names)) if assignment[i].item()==1]
            upgrades.sort(key=lambda i: ((logits[i,1]-logits[i,0])/self.weights[i]).item())
            for index in upgrades:
                assignment[index]=0
                current=cost()
                if current<=self.target+1e-7:
                    break
        if current>self.target+1e-6:
            raise ValueError('Discrete assignment exceeds exact weighted tensor budget')
        return assignment

    @contextlib.contextmanager
    def interpolated(self):
        previous=[]
        try:
            for name,module in self.modules_by_name.items():
                if name=='lm_head':
                    continue  # Full-vocabulary objective owns bounded head interpolation.
                candidate=self.candidates[name]
                index=self.group_index[name]
                previous.append((module,module.forward))
                if isinstance(module,torch.nn.Embedding):
                    def forward(current,ids,_index=index,_candidate=candidate):
                        reference=torch.nn.functional.embedding(ids,current.weight)
                        quant=gather_candidate(_candidate,ids,current.weight,self.row_chunk)
                        p=self.probabilities[_index,0]
                        return reference+p.to(reference.dtype)*(quant-reference)
                else:
                    def forward(current,x,_index=index,_candidate=candidate):
                        p=self.probabilities[_index,0]
                        value=StreamedLinear.apply(x,current.weight,p,_candidate,self.row_chunk)
                        return value if current.bias is None else value+current.bias
                module.forward=types.MethodType(forward,module)
            yield
        finally:
            for module,forward in previous:
                module.forward=forward

    def step(self, ids, optimizer, temperature=1.):
        if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<2:
            raise ValueError('Each full invocation must contain >=2 tokens; no padding/truncation')
        optimizer.zero_grad(set_to_none=True)
        u=torch.rand_like(self.alpha).clamp(1e-7,1-1e-7)
        logits=(self.alpha-torch.log(-torch.log(u)))/temperature
        soft=logits.softmax(-1)
        selected=self.assignment(logits)
        self.probabilities=torch.nn.functional.one_hot(selected,2).float()-soft.detach()+soft
        with torch.no_grad():
            reference=self.model.model(input_ids=ids,use_cache=False).last_hidden_state
        with self.interpolated():
            hidden=self.model.model(input_ids=ids,use_cache=False).last_hidden_state
            p=self.probabilities[self.group_index['lm_head'],0]
            loss=full_kl(hidden[0,:-1],reference[0,:-1],self.model.lm_head.weight,p,
                         self.candidates['lm_head'],self.token_chunk,self.vocab_chunk)
            loss.backward()  # Checkpoint recomputation must remain inside interpolation context.
        if not torch.isfinite(loss) or self.alpha.grad is None or not torch.isfinite(self.alpha.grad).all():
            raise FloatingPointError('Nonfinite/missing full-model RCO allocation gradient')
        raw_gradient_norm=self.alpha.grad.norm().item()
        self.project_gradient(self.alpha,self.costs,self.weights)
        optimizer.step()
        budget=self.retraction(self.alpha,self.costs,self.target,self.weights,tol=1e-6)
        self.vector_transport(optimizer,self.alpha,self.costs,self.weights)
        self.probabilities=None
        return {'loss':loss.item(),'raw_gradient_norm':raw_gradient_norm,'budget_bits':budget}

    def hard_allocation(self, target_bytes, fixed_bytes=0):
        selected=self.assignment(self.alpha.detach())
        choices={name:'bf16' if selected[i].item() else 'q1' for i,name in enumerate(self.names)}
        used=fixed_bytes+sum(self.byte_costs[n][1 if choices[n]=='bf16' else 0] for n in self.names)
        if used>target_bytes:
            raise ValueError('Hard allocation exceeds exact tensor byte budget')
        return {'choices':choices,'estimated_tensor_bytes':used,
                'extraction':'upstream_dynamic_programming_with_exact_cost_overflow_repair',
                'serialized_container_overhead_included':self.serialized_costs_verified}
