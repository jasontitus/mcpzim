"""Pinned Transformers Qwen3.5 hybrid text architecture for original Qwen3.8 weights."""
import json
from pathlib import Path
import torch
from safetensors import safe_open


def tiny_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        layer_types=['linear_attention','full_attention'], linear_conv_kernel_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, tie_word_embeddings=False,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,
                         'mrope_section':[6,5,5],'mrope_interleaved':True})
    config._attn_implementation='sdpa'
    return Qwen3_5ForCausalLM(config).eval().requires_grad_(False)


def load_original(directory, device):
    from accelerate import init_empty_weights
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    directory = Path(directory)
    raw = json.loads((directory/'config.json').read_text())
    config = Qwen3_5TextConfig(**raw['text_config'])
    if config.hidden_size != 5120 or config.num_hidden_layers != 64:
        raise ValueError('Expected original Qwen3.8-27B hybrid configuration')
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
    # Nonpersistent rotary buffers are not in the checkpoint and meta init
    # leaves them on CPU. Move devices without changing their FP32 precision.
    model.to(device=device)
    if any(t.device.type!=torch.device(device).type for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('Original parameters/buffers did not reach execution device')
    if any(p.dtype!=torch.bfloat16 for p in model.parameters()):
        raise ValueError('Original model loading must preserve BF16 for every parameter')
    return model.requires_grad_(False)


def install_original_tensor(model,name,tensor,device):
    from accelerate.utils import set_module_tensor_to_device
    if tensor.dtype!=torch.bfloat16:raise ValueError('Original tensor must be BF16')
    # Meta initialization defaults to FP32; dtype=None would silently double
    # all loaded weights. Explicit dtype preserves original checkpoint bits.
    set_module_tensor_to_device(model,name,device,value=tensor,dtype=torch.bfloat16,clear_cache=False)


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
