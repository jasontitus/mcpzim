"""Synthetic complete 1-layer Llama fixture to test shipped CPU Q1 kernels.

Not a Qwen architecture/quality test. Contains actual serialized Q1 embeddings,
head and projection weights, F32 norms, vocabulary and standard GGUF metadata.
"""
import argparse
import hashlib
from pathlib import Path
import struct
import numpy as np
from packing.gguf import write,string
from packing.q1 import from_sign_logits


def fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    def text(key,value): return string(key)+struct.pack('<I',8)+string(value)
    def integer(key,value): return string(key)+struct.pack('<II',4,value)
    def number(key,value): return string(key)+struct.pack('<If',6,value)
    def strings(key,values): return string(key)+struct.pack('<IIQ',9,8,len(values))+b''.join(string(v) for v in values)
    metadata=[text('general.architecture','llama'),text('general.name','Zimfo Q1 compatibility fixture'),
        integer('general.alignment',32),integer('general.file_type',40),
        integer('llama.context_length',32),integer('llama.embedding_length',128),integer('llama.block_count',1),
        integer('llama.feed_forward_length',128),integer('llama.attention.head_count',4),
        integer('llama.attention.head_count_kv',4),integer('llama.rope.dimension_count',32),
        number('llama.attention.layer_norm_rms_epsilon',1e-5),text('tokenizer.ggml.model','llama'),
        strings('tokenizer.ggml.tokens',['<unk>','<s>','</s>']+[f'token{i}' for i in range(29)]),
        integer('tokenizer.ggml.unknown_token_id',0),integer('tokenizer.ggml.bos_token_id',1),integer('tokenizer.ggml.eos_token_id',2)]
    entries=[];rng=np.random.default_rng(31)
    shapes={'token_embd.weight':[32,128],'output.weight':[32,128],
        **{f'blk.0.{name}.weight':[128,128] for name in ('attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up','ffn_down')}}
    for name,shape in shapes.items():
        data,_=from_sign_logits(rng.normal(size=shape),np.full((shape[0],shape[1]//128),.03125,np.float32))
        path=root/(name+'.bin');path.write_bytes(data.tobytes())
        entries.append({'name':name,'shape':shape,'type':41,'payload':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for name in ('output_norm.weight','blk.0.attn_norm.weight','blk.0.ffn_norm.weight'):
        path=root/(name+'.bin');path.write_bytes(np.ones(128,dtype='<f4').tobytes())
        entries.append({'name':name,'shape':[128],'type':0,'payload':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return write(root/'tiny-q1.gguf',b''.join(metadata),len(metadata),entries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();print(fixture(a.output))
