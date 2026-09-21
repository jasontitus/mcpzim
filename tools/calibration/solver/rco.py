"""Memory-bounded candidate interpolation with upstream RCO budget operations."""
import contextlib
import sys
from pathlib import Path
import types
import torch
from .qwen import projection_modules
from .candidates import StreamedLinear,Q1Candidate
from .objective import full_kl

#: The pinned knapsack's cost-grid resolution (`search/quant.py:434` prices each
#: group at `round(frac * bits * resolution)` units). Load-bearing: it is *not*
#: neutral, because 498 groups of ~0.002 weight each round up, putting the solver's
#: all-Q1 floor above the true 1.125 bits/param.
KNAPSACK_RESOLUTION = 500


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
    def __init__(self, model, database, target_bytes, fixed_bytes=0, upstream=None,
                 row_chunk=512, token_chunk=32, vocab_chunk=2048, cost_manifest=None):
        super().__init__()
        object.__setattr__(self,'model',model)
        from .upstream_paths import resolve as _resolve_upstream
        upstream=_resolve_upstream(upstream)
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
        # float64 for exact byte-cost arithmetic, but NOT on the device: MPS has no
        # float64 at all. These are ~number-of-projections-sized, computed once, and
        # only their normalized float result is used, so CPU fp64 -> float on the
        # compute device is both exact and off the hot path.
        counts=torch.tensor([self.modules_by_name[n].weight.numel() for n in self.names],
                            device='cpu',dtype=torch.float64)
        self.weights=(counts/counts.sum()).float().to(device)
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
        # `counts` is CPU fp64 by design (MPS has no fp64), so the normalized
        # value must be moved back to the compute device before it multiplies a
        # device-resident tensor. An earlier version hoisted only `self.weights`
        # and left this expression mixing mps:0 with cpu, which raises at
        # construction -- and was invisible in CI because every rco_run test uses
        # a CPU model, where the two operands coincidentally share a device.
        counts_dev=counts.float().to(device)
        self.costs=torch.tensor([self.byte_costs[name] for name in self.names],device=device).float()*8/counts_dev[:,None]
        self.nominal_costs=torch.tensor([1.125,16.],device=device)
        self.target=(target_bytes-fixed_bytes)*8/counts.sum().item()
        if not (self.weights*self.costs[:,0]).sum().item()-1e-6 <= self.target <= (self.weights*self.costs[:,1]).sum().item()+1e-6:
            raise ValueError('Requested byte budget is infeasible with Q1+BF16 candidates and protected tensors')
        self.retraction(self.alpha,self.costs,self.target,self.weights,tol=1e-6)
        # The knapsack quantizes each group's cost to 1/KNAPSACK_RESOLUTION units, so
        # its all-Q1 floor sits above the true 1.125 bits/param and any budget below it
        # is infeasible for the solver. That is not benign: `assignment` is called once
        # per Gumbel sample *during training*, and there the unwritten sentinel dp makes
        # it return uniform Q1 for any logits - a whole stage would train against an
        # allocation the solver never chose, while reporting a converging loss. Refuse
        # the budget at construction so every path sees a feasible target.
        self.parameter_count=sum(self.modules_by_name[name].weight.numel() for name in self.names)
        floor_units=sum(int(round(float(weight)*float(cost)*KNAPSACK_RESOLUTION))
                        for weight,cost in zip(self.weights,self.costs[:,0]))
        if int(round(self.target*KNAPSACK_RESOLUTION))<floor_units:
            floor_bits=floor_units/KNAPSACK_RESOLUTION
            raise ValueError(
                f'The knapsack is infeasible at {self.target:.6f} bits/param '
                f'({target_bytes} bytes): its 1/{KNAPSACK_RESOLUTION}-bit grid rounds each '
                f'of the {len(self.names)} groups up, so its all-Q1 floor is '
                f'{floor_bits:.6f} bits/param. Raise target_bytes by at least '
                f'{int((floor_bits-self.target)*self.parameter_count/8)+1} bytes.')
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

    def step(self, ids, optimizer, temperature=1., n_gumbel_samples=4):
        if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<2:
            raise ValueError('Each full invocation must contain >=2 tokens; no padding/truncation')
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            reference=self.model.model(input_ids=ids,use_cache=False).last_hidden_state
        # Upstream draws one independent Gumbel noise per sample, gives each sample
        # its own budget-feasible hard assignment, and averages the *gradients* by
        # scaling each sample's loss by 1/n_evals before backward
        # (rco/src/search/quant.py:669-704, n_evals at :669); the update is then one
        # projection, one Adam step and one retraction of that averaged gradient
        # (:718-728), not four sequential single-sample updates. The pinned driver
        # passes 4 samples at batch_size 1, batches-per-step 1
        # (scripts/run_search_quant.sh:19,29-31), so n_evals is the sample count.
        total_loss=0.
        with self.interpolated():
            for _ in range(n_gumbel_samples):
                u=torch.rand_like(self.alpha).clamp(1e-7,1-1e-7)
                logits=(self.alpha-torch.log(-torch.log(u)))/temperature
                soft=logits.softmax(-1)
                selected=self.assignment(logits)
                self.probabilities=torch.nn.functional.one_hot(selected,2).float()-soft.detach()+soft
                hidden=self.model.model(input_ids=ids,use_cache=False).last_hidden_state
                p=self.probabilities[self.group_index['lm_head'],0]
                loss=full_kl(hidden[0,:-1],reference[0,:-1],self.model.lm_head.weight,p,
                             self.candidates['lm_head'],self.token_chunk,self.vocab_chunk)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-model RCO allocation loss')
                # Checkpoint recomputation must remain inside interpolation context.
                (loss/n_gumbel_samples).backward()
                total_loss+=loss.item()
        if self.alpha.grad is None or not torch.isfinite(self.alpha.grad).all():
            raise FloatingPointError('Nonfinite/missing full-model RCO allocation gradient')
        raw_gradient_norm=self.alpha.grad.norm().item()
        self.project_gradient(self.alpha,self.costs,self.weights)
        optimizer.step()
        budget=self.retraction(self.alpha,self.costs,self.target,self.weights,tol=1e-6)
        self.vector_transport(optimizer,self.alpha,self.costs,self.weights)
        self.probabilities=None
        return {'loss':total_loss/n_gumbel_samples,'raw_gradient_norm':raw_gradient_norm,'budget_bits':budget}

    def hard_allocation(self, target_bytes, fixed_bytes=0):
        # __init__ has already refused an infeasible budget, so the knapsack here always
        # has a feasible assignment to find. What this function must not do is solve one
        # budget and report another: it validates the bytes it was handed, so require
        # them to describe the target the solver was constructed with.
        requested=(target_bytes-fixed_bytes)*8/self.parameter_count
        if abs(requested-self.target)>1e-6:
            raise ValueError(
                f'hard_allocation was handed {target_bytes} bytes ({requested:.6f} bits/param) '
                f'but this trainer solves for {self.target:.6f} bits/param; construct the trainer '
                f'with the budget you intend to allocate')
        # `self.assignment` runs the knapsack and then repairs a discretized-feasible
        # assignment that overspends the *exact* byte budget by a bounded amount. An
        # earlier revision of this change called the knapsack directly, which changed
        # nothing about the DP call and silently dropped the repair while `extraction`
        # still advertised it.
        selected=self.assignment(self.alpha.detach())
        choices={name:'bf16' if selected[i].item() else 'q1' for i,name in enumerate(self.names)}
        used=fixed_bytes+sum(self.byte_costs[n][1 if choices[n]=='bf16' else 0] for n in self.names)
        if used>target_bytes:
            raise ValueError('Hard allocation exceeds exact tensor byte budget')
        # A uniform Q1 answer is right only when the budget cannot afford a single
        # bf16 group. Above that floor it means the allocation failed, and returning
        # it silently would ship a model whose budget was never spent while looking
        # like a considered allocation. Measured on this port: target_bytes=4.0e9
        # yields 0 bf16 groups, 4.15e9 yields 96+, and nothing in between warned -
        # which is why this refuses rather than reports.
        cheapest=min(self.byte_costs[name][1]-self.byte_costs[name][0] for name in self.names)
        if used+cheapest<=target_bytes and all(mode=='q1' for mode in choices.values()):
            raise ValueError(
                f'RCO returned a uniform Q1 allocation although the budget ({target_bytes} '
                f'bytes) affords at least one bf16 group (cheapest upgrade {cheapest} bytes); '
                f'refusing to treat an unallocated model as an allocation')
        return {'choices':choices,'estimated_tensor_bytes':used,
                'extraction':'upstream_dynamic_programming_with_exact_cost_overflow_repair',
                'serialized_container_overhead_included':self.serialized_costs_verified}
