"""CPU-only row-streamed RTN candidate fixture: export/runtime compatibility ONLY.

This is not GSQ optimization, calibration, quality evidence, or a replacement
baseline. Frozen original BF16 weights -> FP32meanabs(group128) -> BF16 scale;
positive weight=1, zero and negative=0. Candidate files use solver's schema.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import numpy as np
import torch
from safetensors import safe_open

from packing.export_qwen import sha

ALGORITHM='rtn-meanabs-fp32-scale-bf16-group128-positive-strict-v1'
RESERVE=24*1024**3


def require_space(root, more=0):
    if shutil.disk_usage(root).free < RESERVE+more:
        raise RuntimeError('24GiB disk reserve would be consumed')


def atomic_json(path,data):
    temp=path.with_suffix('.json.pending');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def write_candidate(source,tensor,path,shape,rows=128):
    if path.exists():raise FileExistsError(path)
    out,cols=shape
    if cols%128:raise ValueError('Unsupported group width')
    code_bytes=out*(cols//8);scale_bytes=out*(cols//128)*2
    metadata={'format':'zimfo-q1-v1','group_size':'128','bit_order':'little','effective_scale_dtype':'BF16','zero_logit_sign':'negative',
              'algorithm':ALGORITHM,'purpose':'EXPORT_RUNTIME_COMPATIBILITY_ONLY_NOT_GSQ'}
    header=json.dumps({'__metadata__':metadata,
        'codes':{'dtype':'U8','shape':[out,cols//8],'data_offsets':[0,code_bytes]},
        'scales':{'dtype':'BF16','shape':[out,cols//128],'data_offsets':[code_bytes,code_bytes+scale_bytes]}},separators=(',',':')).encode()
    header+=b' '*((-len(header))%8);start_data=8+len(header)
    require_space(path.parent,start_data+code_bytes+scale_bytes)
    partial=path.with_suffix('.pending');peak=0
    with partial.open('xb') as target, safe_open(str(source),framework='pt',device='cpu') as original:
        target.write(struct.pack('<Q',len(header)));target.write(header)
        target.truncate(start_data+code_bytes+scale_bytes)
        weight=original.get_slice(tensor)
        if weight.get_dtype()!='BF16' or weight.get_shape()!=shape:raise ValueError('Wrong original tensor')
        for lo in range(0,out,rows):
            hi=min(out,lo+rows)
            x=weight[lo:hi].float()
            if not torch.isfinite(x).all():raise ValueError('Nonfinite original weight')
            scales=x.abs().reshape(hi-lo,-1,128).mean(-1).to(torch.bfloat16)
            bits=np.packbits((x>0).numpy(),axis=-1,bitorder='little')
            target.seek(start_data+lo*(cols//8));target.write(bits.tobytes())
            target.seek(start_data+code_bytes+lo*(cols//128)*2);target.write(scales.view(torch.uint16).numpy().tobytes())
            peak=max(peak,x.numel()*4+bits.nbytes+scales.numel()*2)
        target.flush();os.fsync(target.fileno())
    partial.replace(path)
    return {'bytes':path.stat().st_size,'sha256':sha(path),'peak_explicit_row_tensor_bytes':peak}


def run(args):
    torch.set_num_threads(4)
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=False)
    require_space(root,12*1024**3)
    candidates=root/'candidates';candidates.mkdir()
    model=Path(args.model_dir).resolve()
    docs=Path(__file__).resolve().parents[3]/'docs/benchmarks/quantization-2026-09-19'
    inventory=json.loads((docs/'encoding-inventory.json').read_text())
    validation=json.loads((docs/'baseline-weight-validation.json').read_text())
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    if validation['status']!='validated' or validation['model']!=inventory['model'] or validation['revision']!=inventory['revision']:
        raise ValueError('Missing pinned baseline validation')
    sources={};choices={}
    report={'status':'running','purpose':'EXPORT_RUNTIME_COMPATIBILITY_ONLY_NOT_GSQ','algorithm':ALGORITHM,
        'model':inventory['model'],'revision':inventory['revision'],'source_validation_sha256':sha(docs/'baseline-weight-validation.json'),
        'source_shards':sources,'candidates':choices,'completed':0}
    atomic_json(root/'rtn-report.json',report);started=time.monotonic()
    try:
        for tensor,info in inventory['tensors'].items():
            if not(tensor.startswith('model.language_model.') or tensor=='lm_head.weight') or len(info['shape'])!=2:continue
            require_space(root)
            source=model/index[tensor]
            if source.name not in sources:
                expected=validation['verified_files'][source.name]
                if source.stat().st_size!=expected['bytes'] or sha(source)!=expected['sha256']:raise ValueError('Source hash mismatch')
                sources[source.name]=expected
            path=candidates/(hashlib.sha256(tensor.encode()).hexdigest()[:24]+'.safetensors')
            detail=write_candidate(source,tensor,path,info['shape'],args.rows)
            choices[tensor]={'mode':'q1','candidate':str(path.resolve()),'sha256':detail['sha256']}
            report['completed']=len(choices);report['elapsed_seconds']=time.monotonic()-started
            report['candidate_file_bytes']=sum(p.stat().st_size for p in candidates.glob('*.safetensors'))
            atomic_json(root/'rtn-report.json',report)
            print(f"RTN layout only {len(choices)}/498 {tensor} elapsed={report['elapsed_seconds']:.1f}s",flush=True)
        if len(choices)!=498:raise ValueError('Incomplete eligible tensor coverage')
        plan={'model_dir':str(model),'weight_hashes':str(docs/'original-weight-hashes.json'),
            'tensor_inventory':str(docs/'encoding-inventory.json'),'metadata_manifest':str(docs/'baseline-metadata-manifest.json'),
            'choices':choices,'purpose':report['purpose'],'algorithm':ALGORITHM,'disk_reserve_bytes':RESERVE,
            'allow_scale_loss':True,'max_gguf_bytes':4*1024**3}
        atomic_json(root/'export-plan.json',plan)
        report['status']='completed';atomic_json(root/'rtn-report.json',report)
    except BaseException as error:
        report.update(status='failed',error=str(error));atomic_json(root/'rtn-report.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--rows',type=int,default=128)
    run(p.parse_args())
