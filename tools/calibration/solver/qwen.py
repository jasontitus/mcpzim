"""Pinned Transformers Qwen3.5 hybrid text architecture for original Qwen3.8 weights."""
import contextlib
import json
from pathlib import Path
import torch
from safetensors import safe_open


@contextlib.contextmanager
def init_empty_weights():
    """Construct modules on the ``meta`` device, allocating no storage.

    Local replacement for ``accelerate.init_empty_weights``. That package is
    absent from the native Mac environment and its only use here is this one
    context: new parameters land on ``meta``, so building a 27B model costs no
    memory and the real BF16 tensors are installed afterwards by
    :func:`install_original_tensor`. ``torch.device('meta')`` is a context
    manager in the same way and is what accelerate itself wraps.
    """
    with torch.device('meta'):
        yield


def set_module_tensor_to_device(model, name, device, value, dtype=None, clear_cache=False):
    """Move one named tensor into ``model``'s module tree.

    Local replacement for ``accelerate.utils.set_module_tensor_to_device``,
    kept argument-compatible with the call site in
    :func:`install_original_tensor` so callers do not change.

    ``accelerate`` walks its own device maps and hooks to do this and is not
    installed here; this implementation is the subset actually used — a
    parameter addressed by dotted module path, installed at a requested device
    and dtype. Two accelerate behaviours are load-bearing and preserved:

    * ``dtype`` is applied to the value *before* installation, so the stored
      parameter's dtype is exactly ``dtype``;
    * ``clear_cache`` is accepted for signature compatibility. accelerate uses
      it to release CUDA caching-allocator blocks between shards; on a unified
      memory Mac there is no separate device heap to release, so no cache is
      cleared and the argument has no effect. It is never silently repurposed.
    """
    if not isinstance(name, str) or not name:
        raise ValueError('Tensor name must be a non-empty string')
    if value is None:
        raise ValueError('set_module_tensor_to_device requires a value')
    module_path, _, tensor_name = name.rpartition('.')
    parent = model
    for component in filter(None, module_path.split('.')):
        if not hasattr(parent, component):
            raise ValueError(f'{name} does not address a module in this model')
        parent = getattr(parent, component)
    current = parent._parameters.get(tensor_name)
    if current is not None:
        is_param, existing = True, current
    elif tensor_name in parent._buffers:
        is_param, existing = False, parent._buffers[tensor_name]
    else:
        raise ValueError(f'{name} is not a parameter or buffer of {type(parent).__name__}')
    value = value.to(device=device, dtype=dtype)
    if is_param:
        # A meta parameter has no storage to copy into, and torch refuses a
        # set_data swap between meta and real storage. Replace the entry the
        # module actually reads, preserving the existing requires_grad flag, so
        # what is installed is exactly the requested dtype on the requested
        # device.
        parent._parameters[tensor_name] = torch.nn.Parameter(value, requires_grad=existing.requires_grad)
    else:
        parent._buffers[tensor_name] = value


def tiny_model(layers=2):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    kinds=['linear_attention','full_attention']
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        layer_types=[kinds[i%2] for i in range(layers)], linear_conv_kernel_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, tie_word_embeddings=False,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,
                         'mrope_section':[6,5,5],'mrope_interleaved':True})
    config._attn_implementation='sdpa'
    return Qwen3_5ForCausalLM(config).eval().requires_grad_(False)


def load_original(directory, device):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    directory = Path(directory)
    raw = json.loads((directory/'config.json').read_text())
    config = Qwen3_5TextConfig(**raw['text_config'])
    # Architecture guard. Two independent requirements, and both are needed:
    #
    # 1. **Production geometry is pinned exactly.** A scaled-down model with the
    #    same *ratios* is permitted for CPU tests; a production-width config must
    #    match the published numbers.
    # 2. **The hybrid layout is mandatory at every width.** This is the check that
    #    actually prevents a wrong model, and an earlier version was vacuous: it
    #    only tested that `layer_types` contained no *unknown* type, which an
    #    all-`full_attention` stack satisfies trivially. A synthetic
    #    all-full-attention Qwen3_5 loaded successfully through that guard —
    #    the exact case the guard is meant to exclude — and nothing downstream
    #    re-checks, so the guard is the last line of defence for tensor layout.
    if config.hidden_size == 5120 and config.num_hidden_layers == 64:
        if (config.num_attention_heads, config.num_key_value_heads, config.head_dim) != (24, 4, 256):
            raise ValueError('Expected original Qwen3.8-27B attention configuration')
        if (config.linear_num_value_heads, config.linear_num_key_heads,
                config.linear_key_head_dim, config.linear_value_head_dim) != (48, 16, 128, 128):
            raise ValueError('Expected original Qwen3.8-27B linear attention configuration')
    if config.linear_num_value_heads % config.linear_num_key_heads:
        raise ValueError('Expected linear value heads divisible by linear key heads')
    layer_types = list(config.layer_types)
    unknown = set(layer_types) - {'linear_attention', 'full_attention'}
    if unknown:
        raise ValueError(
            f'Expected a Qwen3.8-27B mixed linear/full attention stack; found '
            f'unsupported layer type(s) {sorted(unknown)}'
        )
    # The model is a *hybrid*: it must actually contain both kinds. Without this,
    # an all-full_attention stack — a different architecture with the same config
    # class — passes, and would be quantized by this adapter with linear-attention
    # tensors absent and nothing noticing.
    if 'linear_attention' not in layer_types or 'full_attention' not in layer_types:
        raise ValueError(
            f'Expected both linear_attention and full_attention layers (this adapter '
            f'quantizes a hybrid stack); got only '
            f'{sorted(set(layer_types))} across {len(layer_types)} layers'
        )
    config._attn_implementation='sdpa'
    with init_empty_weights():
        model=Qwen3_5ForCausalLM(config).eval()
    expected=set(model.state_dict())
    seen=set()
    index=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    for filename in sorted(set(index.values())):
        with safe_open(str(directory/filename), framework='pt') as f:
            for source in f.keys():
                if source.startswith('model.language_model.'):
                    target='model.'+source.removeprefix('model.language_model.')
                elif source=='lm_head.weight':
                    target=source
                else:
                    continue  # Original vision/MTP not used by the text-only application.
                if target not in expected or target in seen:
                    raise ValueError(f'Unexpected/duplicate original tensor {target}')
                tensor=f.get_tensor(source)
                if tensor.dtype!=torch.bfloat16:
                    raise ValueError(f'Original tensor is not BF16: {source}')
                install_original_tensor(model,target,tensor,device)
                seen.add(target)
    if seen!=expected:
        raise ValueError(f'Missing original text tensors: {expected-seen}')
    # Nonpersistent rotary buffers are not in the checkpoint, so the meta context
    # left them on `meta`; `Module.to()` cannot copy out of meta storage. Build
    # them in FP32 at their declared shape, then move devices without changing
    # that FP32 precision.
    for name,buffer in list(model.named_buffers()):
        if buffer.device.type=='meta':
            path,_,leaf=name.rpartition('.')
            parent=model
            for component in filter(None,path.split('.')):parent=getattr(parent,component)
            parent._buffers[leaf]=torch.empty(buffer.shape,dtype=buffer.dtype,device=device)
    if any(buffer.device.type=='meta' for buffer in model.buffers()):
        raise ValueError('Nonpersistent buffers were not materialized before device move')
    model.to(device=device)
    if any(t.device.type!=torch.device(device).type for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('Original parameters/buffers did not reach execution device')
    if any(p.dtype!=torch.bfloat16 for p in model.parameters()):
        raise ValueError('Original model loading must preserve BF16 for every parameter')
    return model.requires_grad_(False)


def install_original_tensor(model,name,tensor,device):
    if tensor.dtype!=torch.bfloat16:raise ValueError('Original tensor must be BF16')
    # Meta initialization defaults to FP32; dtype=None would silently double
    # all loaded weights. Explicit dtype preserves original checkpoint bits.
    return set_module_tensor_to_device(model,name,device,value=tensor,dtype=torch.bfloat16,clear_cache=False)


def block_kwargs(model, hidden, block_index):
    from transformers.masking_utils import create_causal_mask
    positions=torch.arange(hidden.shape[1],device=hidden.device).unsqueeze(0)
    rope_positions=positions.unsqueeze(0).expand(3,hidden.shape[0],-1)
    embeddings=model.model.rotary_emb(hidden,rope_positions)
    mask=None
    if model.config.layer_types[block_index]=='full_attention':
        mask=create_causal_mask(config=model.config,inputs_embeds=hidden,attention_mask=None,
                               past_key_values=None,position_ids=positions)
    return {'position_embeddings':embeddings,'attention_mask':mask,'position_ids':positions,
            'past_key_values':None,'use_cache':False}


def run_block(model, block_index, hidden):
    return model.model.layers[block_index](hidden,**block_kwargs(model,hidden,block_index))


def projection_modules(model):
    return {name: module for name,module in model.named_modules()
            if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)) and
            module.weight.ndim==2 and module.weight.shape[1]%128==0}
