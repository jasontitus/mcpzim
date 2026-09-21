"""Export a complete text-only Qwen checkpoint from explicit solver choices.

Uses pinned Prism converter for tokenizer/model metadata and protected tensor
transforms; one-bit candidates are permuted and packed directly, without another
quantization pass. Vision/MTP are deliberately omitted. No GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import shutil
import sys
import numpy as np
from packing.q1 import pack, reorder_qwen_candidate
from packing.gguf import write, serialized_size, validate_metadata, expected_nbytes, align

PRISM = '62061f91088281e65071cc38c5f69ee95c39f14e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_prism_source(source):
    source=Path(source).resolve()
    receipt=source/'zimfo-prism-source.json'
    if receipt.exists():
        document=json.loads(receipt.read_text())
        files=document.get('files',{})
        required={'conversion/base.py','conversion/qwen.py','conversion/__init__.py',
                  'gguf-py/gguf/constants.py','gguf-py/gguf/__init__.py','LICENSE'}
        if document.get('revision')!=PRISM or not required<=set(files):
            raise ValueError('Incomplete or wrong Prism source receipt')
        for relative,entry in files.items():
            path=source/relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.resolve().is_relative_to(source):
                raise ValueError('Unsafe Prism source path')
            if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise ValueError(f'Prism source hash mismatch: {relative}')
        python_files={str(p.relative_to(source)) for p in source.rglob('*.py') if '__pycache__' not in p.parts}
        if not python_files<=set(files): raise ValueError('Unverified Python code in Prism source tree')
        return PRISM
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision != PRISM: raise ValueError('Wrong Prism converter revision')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Modified Prism converter checkout')
    return revision


def load_converter(source):
    source=Path(source).resolve()
    revision=verify_prism_source(source)
    sys.path.insert(0,str(source/'gguf-py'));sys.path.insert(0,str(source))
    from conversion.qwen import Qwen3_5TextModel
    import gguf
    import inspect
    if not Path(gguf.__file__).resolve().is_relative_to(source/"gguf-py") or Path(inspect.getfile(Qwen3_5TextModel)).resolve()!=source/"conversion/qwen.py":
        raise ValueError("Another converter/gguf package is already imported")
    return Qwen3_5TextModel,gguf,revision


def load_candidate(path, expected_shape, expected_sha):
    from safetensors import safe_open
    import torch
    if sha(path)!=expected_sha: raise ValueError('Candidate hash mismatch')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        meta=f.metadata() or {}
        if (meta.get('format'),meta.get('group_size'),meta.get('bit_order'),meta.get('effective_scale_dtype')) != ('zimfo-q1-v1','128','little','BF16'):
            raise ValueError('Candidate format contract mismatch')
        bits,scales=f.get_tensor('codes'),f.get_tensor('scales')
        if bits.dtype!=torch.uint8 or scales.dtype!=torch.bfloat16:
            raise ValueError('Wrong candidate storage dtype')
        if list(bits.shape)!=[expected_shape[0],expected_shape[1]//8] or list(scales.shape)!=[expected_shape[0],expected_shape[1]//128]:
            raise ValueError('Candidate dimensions mismatch')
        return bits.numpy(),scales.float().numpy()


def cost_manifest(converter, metadata_bytes, metadata_count, originals):
    """Exact additive RCO cost: fixed container/protected bytes + selected tensor bytes."""
    import torch
    planned=[]; choices={}; protected=[]
    for original,info in originals.items():
        name=original.replace('language_model.','');shape=info['shape']
        if len(shape)==2:
            if shape[-1]%128: raise ValueError('Unmodeled candidate width')
            mapped=converter.map_tensor_name(name)
            choices[original]={'runtime_name':mapped,'shape':shape,
                'q1_bytes':align(expected_nbytes(shape,41)),
                'bf16_bytes':align(expected_nbytes(shape,30)),
                'f32_bytes':align(expected_nbytes(shape,0))}
            planned.append({'name':mapped,'shape':shape,'type':41})
        else:
            bid=next((int(x) for x in name.split('.') if x.isdecimal()),None)
            for mapped,data in converter.modify_tensors(torch.empty(shape,device='meta'),name,bid):
                t={'name':mapped,'shape':list(data.shape),'type':0}
                protected.append({'source':original,**t,'bytes':align(expected_nbytes(t['shape'],0))})
                planned.append(t)
    total=serialized_size(metadata_bytes,metadata_count,planned)
    return {'schema_version':1,'prism_revision':PRISM,'metadata_sha256':hashlib.sha256(metadata_bytes).hexdigest(),
        'fixed_container_bytes':total-sum(t['q1_bytes'] for t in choices.values()),
        'tensors':choices,'protected':protected,'all_q1_serialized_bytes':total,
        'cost_rule':'fixed_container_bytes + sum(selected padded q1_bytes or bf16_bytes)',
        'precision_rule':'non2D source tensors protected FP32 after pinnedconverter transforms; reference2D BF16'}


def export(plan_path, output_dir, prism_source):
    import torch
    from safetensors import safe_open
    from verify_baseline import verify
    if sys.byteorder!='little': raise ValueError('Exporter requires little-endian host')
    plan=json.loads(Path(plan_path).read_text())
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    report={'status':'running','purpose':plan.get('purpose','quantized_model_export'),
            'algorithm':plan.get('algorithm','explicit_solver_choice_manifest'),'scope':'text-only; vision and MTP omitted','plan_sha256':sha(plan_path),'tensors':[]}
    def save():
        p=root/'report.json.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(root/'report.json')
    save()
    try:
        model_dir=Path(plan['model_dir']).resolve()
        expected=json.loads(Path(plan['weight_hashes']).read_text())
        inventory=json.loads(Path(plan['tensor_inventory']).read_text())
        metadata=json.loads(Path(plan['metadata_manifest']).read_text())
        report['baseline_provenance']={'model':expected['model'],'revision':expected['revision'],
            'weight_hashes_sha256':sha(plan['weight_hashes']),'tensor_inventory_sha256':sha(plan['tensor_inventory']),
            'metadata_manifest_sha256':sha(plan['metadata_manifest'])}
        if metadata['model']!=expected['model'] or metadata['revision']!=expected['revision']:
            raise ValueError('Metadata/weight model mismatch')
        for name,entry in metadata['files'].items():
            if Path(name).name!=name or (model_dir/name).stat().st_size!=entry['bytes'] or sha(model_dir/name)!=entry['sha256']:
                raise ValueError(f'Metadata mismatch: {name}')
        verify(model_dir,expected,inventory,root/'source-validation.json')
        if {p.name for p in model_dir.glob('*.safetensors')}!=set(expected['files']):
            raise ValueError('Unverified extra source shards')
        originals={n:v for n,v in inventory['tensors'].items() if n.startswith('model.language_model.') or n=='lm_head.weight'}
        selectable={n for n,v in originals.items() if len(v['shape'])==2}
        if set(plan['choices'])!=selectable:
            raise ValueError('Every 2D tensor, including embeddings/head, needs an explicit choice')
        cls,gguf,revision=load_converter(prism_source)
        cls.no_mtp=True
        config=json.loads((model_dir/'config.json').read_text())
        converter=cls(model_dir,gguf.LlamaFileType.MOSTLY_Q1_0,root/'metadata.gguf',hparams=config)
        converter.write_vocab()
        template=(root/'metadata.gguf').read_bytes()
        magic,version,count,kv=struct.unpack('<4sIQQ',template[:24])
        if (magic,version,count)!=(b'GGUF',3,0): raise ValueError('Unexpected metadata template')
        if validate_metadata(template[24:],kv).get('general.architecture')!='qwen35':
            raise ValueError('Wrong converter architecture')
        costs=cost_manifest(converter,template[24:],kv,originals)
        (root/'cost-manifest.json').write_text(json.dumps(costs,indent=2)+'\n')
        planned=[]
        for original,info in originals.items():
            normalized=original.replace('language_model.','')
            mode=plan['choices'].get(original,{'mode':'f32'})['mode']
            if mode=='q1':
                planned.append({'name':converter.map_tensor_name(normalized),'shape':info['shape'],'type':41})
            elif mode in ('f16','bf16','f32'):
                bid=next((int(part) for part in normalized.split('.') if part.isdecimal()),None)
                for name,data in converter.modify_tensors(torch.empty(info['shape'],device='meta'),normalized,bid):
                    planned.append({'name':name,'shape':list(data.shape),'type':0 if mode=='f32' or data.ndim<=1 else (30 if mode=='bf16' else 1)})
            else: raise ValueError('Unknown precision choice')
        estimate=serialized_size(template[24:],kv,planned)
        report['exact_serialized_byte_budget']=estimate
        report['embedding_strategy']=plan['choices']['model.language_model.embed_tokens.weight']['mode']
        report['head_strategy']=plan['choices']['lm_head.weight']['mode']
        save()
        if estimate>int(plan.get('max_gguf_bytes',estimate)):
            raise ValueError('Exact serialized model exceeds requested size budget')
        if shutil.disk_usage(root).free < 2*estimate+len(template)+int(plan.get('disk_reserve_bytes',10*1024**3)):
            raise ValueError('Insufficient disk for staged payloads and final GGUF plus reserve')
        weight_map=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
        entries=[]
        for original,info in originals.items():
            normalized=original.replace('language_model.','')
            choice=plan['choices'].get(original,{'mode':'f32'})
            mode=choice['mode']; error=None
            if mode=='q1':
                if info['shape'][1]%128: raise ValueError('Q1 unsupported row width')
                bits,scales=load_candidate(choice['candidate'],info['shape'],choice['sha256'])
                tc=config['text_config']
                bits,scales=reorder_qwen_candidate(original,bits,scales,k_heads=tc['linear_num_key_heads'],
                    v_heads=tc['linear_num_value_heads'],head_k_dim=tc['linear_key_head_dim'],head_v_dim=tc['linear_value_head_dim'])
                data,error=pack(bits,scales,allow_scale_loss=bool(plan.get('allow_scale_loss',False)))
                produced=[(converter.map_tensor_name(normalized),data,info['shape'],41)]
            elif mode in ('f16','bf16','f32'):
                with safe_open(str(model_dir/weight_map[original]),framework='pt',device='cpu') as f:
                    tensor=f.get_tensor(original).float()
                bid=next((int(part) for part in normalized.split('.') if part.isdecimal()),None)
                produced=[]
                for name,data in converter.modify_tensors(tensor,normalized,bid):
                    # Norms and non-matrix parameters always retain converter FP32 semantics.
                    dtype=torch.float32 if mode=='f32' or data.ndim<=1 else (torch.bfloat16 if mode=='bf16' else torch.float16)
                    converted=data.to(dtype)
                    if not torch.isfinite(converted.float()).all(): raise ValueError('Nonfinite protected tensor')
                    value=converted.view(torch.uint16).numpy() if dtype==torch.bfloat16 else converted.numpy()
                    produced.append((name,value,list(value.shape),0 if dtype==torch.float32 else (30 if dtype==torch.bfloat16 else 1)))
            else: raise ValueError('Unknown precision choice')
            for name,data,shape,kind in produced:
                payload=root/(hashlib.sha256(name.encode()).hexdigest()[:24]+'.bin')
                with payload.open('xb') as stream: data.tofile(stream)
                entry={'name':name,'shape':shape,'type':kind,'payload':str(payload),'sha256':sha(payload)}
                entries.append(entry)
                report['tensors'].append({**entry,'source':original,'candidate':choice,'packing_error':error})
            save()
        if serialized_size(template[24:],kv,entries)!=estimate:
            raise ValueError('Actual tensor layout disagrees with preflight byte budget')
        result=write(root/'model.gguf',template[24:],kv,entries)
        report.update(status='completed',serialization=result,prism_revision=revision,gguf_sha256=sha(root/'model.gguf'))
        save()
        return report
    except BaseException as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--prism-source',required=True)
    args=parser.parse_args();export(args.plan,args.output_dir,args.prism_source)
