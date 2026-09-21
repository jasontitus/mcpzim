"""Explicit GSQ block residency; no change to GSQ/RCO optimization mathematics.

CPU-offload mode requires the full immutable baseline in host RAM. Only one
transformer block and its rotary buffers may reside on the execution accelerator.
The embedding cache is built on CPU before any block activation. This policy and
execution device remain part of the scientific checkpoint identity.
"""
from contextlib import contextmanager
import torch

from . import device_policy

FULL = 'full'
BLOCK_CPU_OFFLOAD = 'block_cpu_offload'

#: Device comparison lives in device_policy, the single owner of device decisions.
same_device = device_policy.same_device


def memory_mode(config):
    mode=config.get('gsq_memory_mode',FULL)
    if mode not in (FULL,BLOCK_CPU_OFFLOAD):
        raise ValueError('Unknown GSQ memory mode')
    if mode==BLOCK_CPU_OFFLOAD:
        if config.get('stage','gsq')!='gsq':
            raise ValueError('Block CPU offload is implemented only for GSQ')
        if 'gsq_execution_device' not in config:
            raise ValueError('Block CPU offload requires explicit gsq_execution_device')
    return mode


def require_baseline_fits(mode, device, baseline_bytes, usable_bytes=None):
    """Refuse a full residency that cannot fit, rather than dying silently.

    ``full`` puts the whole baseline on the accelerator. On a unified-memory
    machine that competes with the host for the same pages, and the failure is
    an external kill with no traceback: a run of this port died exactly that way
    with 50.1 GiB resident and swap exhausted at 42.1 of 43 GB. A refusal at
    startup is strictly better than a kill three hours in.

    ``usable_bytes`` is the caller's advertised budget. When it is unknown (MPS
    exposes no such figure) only a hard overcommit is refused, so this cannot
    turn an unknown into a false pass.
    """
    if mode!=FULL or device.type=='cpu':
        return
    if usable_bytes is None:
        return
    if baseline_bytes>usable_bytes:
        raise ValueError(
            f'GSQ full residency needs {baseline_bytes/2**30:.1f} GiB on {device.type} '
            f'but only {usable_bytes/2**30:.1f} GiB is advertised usable; set '
            f'gsq_memory_mode=block_cpu_offload to keep the baseline in host RAM')


def execution_device(config, default):
    value=config.get('gsq_execution_device',default)
    try:device=torch.device(value)
    except (TypeError,RuntimeError) as error:raise ValueError('Invalid GSQ execution device') from error
    if device.type not in device_policy.SUPPORTED_DEVICES:
        raise ValueError('GSQ execution device must be CPU test, CUDA or MPS')
    if device.type=='cpu' and device.index is not None:
        raise ValueError('CPU test execution device must not have an index')
    if device.type=='cuda':
        if not device_policy.is_available('cuda'):raise ValueError('Requested GSQ CUDA device is unavailable')
        index=torch.cuda.current_device() if device.index is None else device.index
        if not 0<=index<torch.cuda.device_count():raise ValueError('Requested GSQ CUDA index is unavailable')
        device=torch.device('cuda',index)
    elif device.type=='mps':
        # Apple Silicon port: the accelerator is MPS. Availability and the
        # warning-and-fall-back contract live in device_policy, the one module
        # that owns device decisions; this function only maps config to a
        # concrete torch.device and keeps its role as a checkpoint identity field.
        if device_policy.resolve_device('mps')!='mps':
            raise ValueError('Requested GSQ MPS device is unavailable')
        if device.index is not None:
            raise ValueError('MPS execution device must not have an index')
    return device


def _tensors(module):
    return list(module.parameters())+list(module.buffers())


class GSQResidency:
    """Use ``with residency.block(index)`` around trainer lifetime for one block."""
    def __init__(self,model,config):
        self.model=model;self.mode=memory_mode(config);self.active=None
        self.device=execution_device(config,model.model.embed_tokens.weight.device)
        if not len(model.model.layers):raise ValueError('GSQ requires transformer blocks')
        tensors=_tensors(model)
        if any(t.is_meta for t in tensors):raise ValueError('GSQ baseline has unmaterialized meta tensors')
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError('GSQ baseline must be frozen with no parameter gradients')
        expected=torch.device('cpu') if self.mode==BLOCK_CPU_OFFLOAD else self.device
        if any(not same_device(t.device,expected) for t in tensors):
            raise ValueError('Baseline residency does not match requested GSQ memory mode/device')
        baseline_bytes=sum(t.numel()*t.element_size() for t in tensors)
        require_baseline_fits(self.mode,self.device,baseline_bytes,
                              device_policy.total_memory_bytes(self.device))

    @property
    def cache_device(self):
        return self.model.model.embed_tokens.weight.device

    def assert_layout(self):
        if self.mode==FULL:
            if any(not same_device(t.device,self.device) for t in _tensors(self.model)):
                raise ValueError('Full GSQ baseline moved off execution device')
            return
        active_ids=set()
        if self.active is not None:
            active_ids={id(t) for module in (self.model.model.layers[self.active],self.model.model.rotary_emb)
                        for t in _tensors(module)}
        for tensor in _tensors(self.model):
            expected=self.device if id(tensor) in active_ids else torch.device('cpu')
            if not same_device(tensor.device,expected):
                raise ValueError('Inactive GSQ baseline tensor escaped CPU residency')

    @contextmanager
    def block(self,index):
        if type(index) is not int or not 0<=index<len(self.model.model.layers):
            raise ValueError('Invalid GSQ block index')
        if self.active is not None:raise ValueError('Only one GSQ block can be active')
        self.assert_layout()
        self.active=index
        block=self.model.model.layers[index];rotary=self.model.model.rotary_emb
        try:
            if self.mode==BLOCK_CPU_OFFLOAD:
                block.to(device=self.device);rotary.to(device=self.device)
            self.assert_layout()
            yield block
        finally:
            if self.mode==BLOCK_CPU_OFFLOAD:
                # Synchronous copies complete outstanding work on the owning
                # stream before releasing CUDA weights. Never change dtype.
                block.to(device='cpu');rotary.to(device='cpu')
            self.active=None
            # The memory win here is block.to('cpu') above, not this call.
            # MPS reclaims a freed block at `del` time rather than pooling it
            # the way the CUDA caching allocator does, so empty_cache has
            # nothing left to release on this build (measured: a 2 GiB
            # transient returns to base at `del`, 0 GiB further on
            # empty_cache). It is kept because it is correct and free on
            # CUDA, where the pooling is real.
            device_policy.empty_cache(self.device)
            self.assert_layout()

    def release_training_state(self,trainer,optimizer):
        """Release live gradients/Adam moments even when caller retains traceback."""
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
        if trainer is not None:
            trainer.zero_grad(set_to_none=True)
            if self.mode==BLOCK_CPU_OFFLOAD:trainer.to(device='cpu')
