"""Verify source artifacts for an explicit GSQ runtime migration.

No ordinary resume identity checks are relaxed. This module imports no torch,
loads no model, and never declares a migration applied or CUDA validation passed.
"""
import hashlib
import json
import math
from pathlib import Path
import re

OLD_RUNTIME = 'f13a4e99a85f4f650a8e3e130757fb4ddee891eeb8e34e6cea973e99dc56c3cd'
SOURCE_SNAPSHOT = 'gsq-b002-s00000174-e000-q00000-p00004'
MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
HASH = re.compile(r'[a-f0-9]{64}\Z')
ROLE = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z')
STATE_ROLES = {'solver','optimizer','scheduler','rng','progress'}
REQUIRED = STATE_ROLES | {'configuration','cache_block_2','candidate_block_0',
    'candidate_block_1','embedding_candidates','initial_candidates','packing_cost_manifest',
    'warmstart_block_0','warmstart_block_3'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(data):return hashlib.sha256(data).hexdigest()


def descriptor(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Missing or symlinked source payload')
    digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(chunk);size+=len(chunk)
    return {'sha256':digest.hexdigest(),'bytes':size}


def validate_source_receipt(source_receipt):
    receipt=source_receipt.get('receipt',{});manifest=receipt.get('manifest',{});commit=receipt.get('commit',{})
    identity=manifest.get('identity',{})
    if (source_receipt.get('snapshot')!=SOURCE_SNAPSHOT or manifest.get('snapshot')!=SOURCE_SNAPSHOT
            or manifest.get('schema')!=3 or identity.get('runtime_sha256')!=OLD_RUNTIME
            or identity.get('baseline_repo')!=MODEL or identity.get('baseline_revision')!=REVISION):
        raise ValueError('Unexpected original block2 source identity')
    raw=canonical(manifest)
    if (type(commit.get('generation')) is not int or commit['generation']<=0
            or commit.get('sha256')!=sha(raw) or commit.get('bytes')!=len(raw)
            or commit.get('object')!=manifest.get('prefix','')+'/commits/'+SOURCE_SNAPSHOT+'.json'
            or manifest.get('bucket')!='tiltastech-zimfo-quantization-us-central1'
            or manifest.get('prefix')!='runs/zimfo-gpu-6583ec8c4659/gsq'):
        raise ValueError('Source commit hash/generation/namespace mismatch')
    payloads=manifest.get('payloads',{})
    if set(payloads)!=REQUIRED:raise ValueError('Source payload inventory differs from reviewed block2 checkpoint')
    for role,item in payloads.items():
        if (not ROLE.fullmatch(role) or type(item.get('bytes')) is not int or item['bytes']<0
                or not isinstance(item.get('sha256'),str) or not HASH.fullmatch(item['sha256'])):
            raise ValueError('Invalid source payload descriptor')
        if role in STATE_ROLES:
            if set(item)!={'member','bytes','sha256'} or item['member']!=role:
                raise ValueError('Invalid state bundle member')
        else:_object(item,manifest['prefix'])
    _object(manifest.get('state_bundle',{}),manifest['prefix'])
    return manifest


def _object(item,prefix):
    if (set(item)!={'object','generation','bytes','sha256'}
            or not isinstance(item.get('sha256'),str) or not HASH.fullmatch(item['sha256'])
            or type(item.get('bytes')) is not int or item['bytes']<0
            or type(item.get('generation')) is not int or item['generation']<=0
            or item.get('object')!=prefix+'/objects/'+item['sha256']):
        raise ValueError('Invalid source object descriptor')


def validate_source_directory(directory,source_receipt):
    manifest=validate_source_receipt(source_receipt);directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():raise ValueError('Expected ordinary restored source directory')
    if {p.name for p in directory.iterdir()}!=set(manifest['payloads']):
        raise ValueError('Restored source directory inventory mismatch')
    checked={}
    for role,item in manifest['payloads'].items():
        checked[role]=descriptor(directory/role)
        if checked[role]!={key:item[key] for key in ('sha256','bytes')}:
            raise ValueError('Restored source payload changed: '+role)
    if checked['configuration']['bytes']>2*1024**2:raise ValueError('Oversized source configuration')
    config=json.loads((directory/'configuration').read_text())
    if (config.get('identity')!=manifest['identity'] or config.get('runtime_sha256')!=OLD_RUNTIME
            or config.get('stage')!='gsq' or config.get('largest_first') is not True
            or config.get('warmstart_sha256')!={str(i):checked['warmstart_block_'+str(i)]['sha256'] for i in (0,3)}
            or config.get('cost_manifest',{}).get('sha256')!=checked['packing_cost_manifest']['sha256']):
        raise ValueError('Frozen source configuration/auxiliary identity mismatch')
    return {'schema_version':1,'status':'source_payloads_verified','source_snapshot':SOURCE_SNAPSHOT,
            'source_commit':source_receipt['receipt']['commit'],'source_identity':manifest['identity'],
            'source_manifest_sha256':sha(canonical(manifest)),'payloads':checked,
            'configuration':config,'cuda_validation_passed':False,'migration_applied':False}


def validate_trainer_state(trainer,state,expected_projection_names):
    """Check positional quantizer state against independently known source names.

    expected_projection_names must come from the old trainer/original block, not
    be inferred from the destination trainer itself. Adam parameter order follows
    this exact quantizer order; matching tensor dimensions alone is insufficient.
    """
    names=list(trainer.names)
    if names!=list(expected_projection_names) or len(names)!=len(set(names)):
        raise ValueError('Projection name/order mismatch')
    actual=trainer.state_dict()
    if set(actual)!=set(state):raise ValueError('Solver state key inventory mismatch')
    inventory={}
    for key,tensor in state.items():
        target=actual[key]
        if tuple(tensor.shape)!=tuple(target.shape) or tensor.dtype!=target.dtype:
            raise ValueError('Solver state shape/dtype mismatch: '+key)
        inventory[key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
    return {'projection_names':names,'state_tensors':inventory,
            'parameter_names':[name for name,_ in trainer.named_parameters()]}


def build_migration_receipt(source_validation,new_identity,validation_report):
    """Record an explicitly supplied proof; never apply/relabel a checkpoint.

    The harness remains responsible for producing this report using actual CUDA
    computations. Merely creating this receipt is not a resume admission gate.
    """
    from checkpoints import _identity
    new_identity=_identity(new_identity)
    old=source_validation.get('source_identity',{})
    if source_validation.get('status')!='source_payloads_verified':raise ValueError('Source not verified')
    if (new_identity['runtime_sha256']==old.get('runtime_sha256')
            or any(new_identity[k]!=old.get(k) for k in ('baseline_repo','baseline_revision','calibration_sha256'))):
        raise ValueError('Migration must bind new runtime and unchanged baseline/calibration')
    if (validation_report.get('status')!='passed' or validation_report.get('actual_cuda') is not True
            or validation_report.get('source_commit')!=source_validation['source_commit']
            or validation_report.get('new_identity')!=new_identity):
        raise ValueError('Missing explicit matching CUDA validation report')
    cuda=validation_report.get('cuda',{})
    total=cuda.get('total_memory_bytes')
    peaks=[cuda.get(key) for key in ('peak_allocated_bytes','peak_reserved_bytes')]
    if (type(total) is not int or not 20*1024**3 <= total <= 24*1024**3
            or not cuda.get('device') or not cuda.get('capability')
            or any(type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=total for value in peaks)):
        raise ValueError('Missing measured real24GB device memory evidence')
    required={'linear_attention','full_attention'}
    blocks=validation_report.get('blocks',{})
    checks=('source_state_equal','update_equal','propagation_equal','next_update_after_restore_equal')
    if set(blocks)!=required or any(b.get('tokens')!=4673 or any(b.get(k) is not True for k in checks) for b in blocks.values()):
        raise ValueError('Incomplete longest-sequence hybrid validation')
    return {'schema_version':1,'status':'migration_proposal_validated_not_applied',
            'source_commit':source_validation['source_commit'],'source_identity':old,
            'source_payloads':source_validation['payloads'],'new_identity':new_identity,
            'validation_report_sha256':sha(canonical(validation_report)),
            'migration_applied':False,'ordinary_resume_identity_checks_unchanged':True}
