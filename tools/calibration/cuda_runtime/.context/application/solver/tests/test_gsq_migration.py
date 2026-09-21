import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from solver import gsq_migration as migration
from prepare_gsq24_source import prepare


def fixture(tmp_path):
    root=tmp_path/'payloads';root.mkdir()
    identity={'baseline_repo':migration.MODEL,'baseline_revision':migration.REVISION,
        'calibration_sha256':'c'*64,'solver_revision':'a'*40,'solver_config_sha256':'d'*64,
        'candidate_database_sha256':'b'*64,'runtime_sha256':migration.OLD_RUNTIME}
    prefix='runs/zimfo-gpu-6583ec8c4659/gsq'
    for name in migration.REQUIRED:(root/name).write_bytes(name.encode())
    config={'identity':identity,'runtime_sha256':migration.OLD_RUNTIME,'stage':'gsq','largest_first':True,
        'warmstart_sha256':{str(i):migration.descriptor(root/f'warmstart_block_{i}')['sha256'] for i in (0,3)},
        'cost_manifest':{'sha256':migration.descriptor(root/'packing_cost_manifest')['sha256']}}
    (root/'configuration').write_bytes(migration.canonical(config))
    payloads={}
    for name in migration.REQUIRED:
        item=migration.descriptor(root/name)
        if name in migration.STATE_ROLES:item['member']=name
        else:item.update(object=prefix+'/objects/'+item['sha256'],generation=8)
        payloads[name]=item
    manifest={'schema':3,'snapshot':migration.SOURCE_SNAPSHOT,'identity':identity,
        'bucket':'tiltastech-zimfo-quantization-us-central1','prefix':prefix,'payloads':payloads,
        'state_bundle':{'object':prefix+'/objects/'+'e'*64,'generation':9,'bytes':100,'sha256':'e'*64}}
    receipt={'snapshot':migration.SOURCE_SNAPSHOT,'receipt':{'manifest':manifest,'commit':{
        'object':prefix+'/commits/'+migration.SOURCE_SNAPSHOT+'.json','generation':10,
        'bytes':len(migration.canonical(manifest)),'sha256':migration.sha(migration.canonical(manifest))}}}
    return root,receipt


def test_validate_source_preserves_identity_and_does_not_claim_cuda(tmp_path):
    root,receipt=fixture(tmp_path);saved=copy.deepcopy(receipt)
    result=migration.validate_source_directory(root,receipt)
    assert receipt==saved and result['source_identity']==receipt['receipt']['manifest']['identity']
    assert not result['cuda_validation_passed'] and not result['migration_applied']


@pytest.mark.parametrize('change',['missing','changed','extra','symlink'])
def test_payload_inventory_and_hash_fail_closed(tmp_path,change):
    root,receipt=fixture(tmp_path)
    if change=='missing':(root/'rng').unlink()
    elif change=='changed':(root/'solver').write_bytes(b'changed')
    elif change=='extra':(root/'extra').write_bytes(b'extra')
    else:
        (root/'rng').unlink();(root/'rng').symlink_to(root/'solver')
    with pytest.raises(ValueError):migration.validate_source_directory(root,receipt)


@pytest.mark.parametrize('change',['hash','generation','identity'])
def test_receipt_rejects_changed_commit_or_runtime(tmp_path,change):
    _,receipt=fixture(tmp_path)
    if change=='hash':receipt['receipt']['commit']['sha256']='f'*64
    elif change=='generation':receipt['receipt']['commit']['generation']=True
    else:receipt['receipt']['manifest']['identity']['runtime_sha256']='f'*64
    with pytest.raises(ValueError):migration.validate_source_receipt(receipt)


def test_projection_order_and_tensor_inventory_are_independent(tmp_path):
    import torch
    class Trainer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.names=['attn.q','mlp.up']
            self.quantizers=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in self.names])
    trainer=Trainer();state=trainer.state_dict()
    assert migration.validate_trainer_state(trainer,state,['attn.q','mlp.up'])['parameter_names']
    with pytest.raises(ValueError,match='order'):migration.validate_trainer_state(trainer,state,['mlp.up','attn.q'])
    bad=dict(state);bad[next(iter(bad))]=torch.zeros(2,2,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='dtype'):migration.validate_trainer_state(trainer,bad,trainer.names)
    with pytest.raises(ValueError,match='inventory'):migration.validate_trainer_state(trainer,{},trainer.names)


def test_cpu_prepare_pins_old_generation_checks_returned_receipt_and_atomic_output(tmp_path,monkeypatch):
    root,receipt=fixture(tmp_path);source=tmp_path/'receipt.json';source.write_bytes(migration.canonical(receipt))
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    calls=[]
    class Store:
        def __init__(self,*args,**kwargs):pass
        def restore(self,snapshot,identity,destination,*,commit_generation):
            calls.append((snapshot,identity,commit_generation));shutil.copytree(root,destination)
            return receipt['receipt']
    output=tmp_path/'prepared'
    result=prepare(source,migration.sha(source.read_bytes()),output,store_factory=Store)
    assert calls==[(migration.SOURCE_SNAPSHOT,receipt['receipt']['manifest']['identity'],10)]
    assert result['gpu_validation_passed'] is False and (output/'source-manifest.json').is_file()
    assert {p.name for p in (output/'payloads').iterdir()}==migration.REQUIRED
    with pytest.raises(FileExistsError):prepare(source,migration.sha(source.read_bytes()),output,store_factory=Store)


def test_cpu_prepare_rejects_space_before_network_and_changed_remote_commit(tmp_path,monkeypatch):
    root,receipt=fixture(tmp_path);source=tmp_path/'receipt.json';source.write_bytes(migration.canonical(receipt))
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    with pytest.raises(ValueError,match='headroom'):
        prepare(source,migration.sha(source.read_bytes()),tmp_path/'out',store_factory=lambda *_a,**_k:pytest.fail('remote'))
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    class Store:
        def __init__(self,*args,**kwargs):pass
        def restore(self,*args,**kwargs):return {'changed':True}
    with pytest.raises(ValueError,match='Remote'):
        prepare(source,migration.sha(source.read_bytes()),tmp_path/'out',store_factory=Store)
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.gsq24-source-*'))


def test_migration_proposal_requires_explicit_complete_cuda_proof(tmp_path):
    root,receipt=fixture(tmp_path);source=migration.validate_source_directory(root,receipt)
    identity={**source['source_identity'],'runtime_sha256':'f'*64,'solver_config_sha256':'e'*64}
    report={'status':'passed','actual_cuda':True,'source_commit':source['source_commit'],'new_identity':identity,
        'cuda':{'total_memory_bytes':24*1024**3,'peak_allocated_bytes':12*1024**3,
                'peak_reserved_bytes':13*1024**3,'device':'test proof','capability':[8,9]},
        'blocks':{kind:{'tokens':4673,**{name:True for name in (
            'source_state_equal','update_equal','propagation_equal','next_update_after_restore_equal')}}
            for kind in ('linear_attention','full_attention')}}
    result=migration.build_migration_receipt(source,identity,report)
    assert result['migration_applied'] is False and result['ordinary_resume_identity_checks_unchanged']
    for key in ('actual_cuda','status'):
        bad=copy.deepcopy(report);bad[key]=False
        with pytest.raises(ValueError):migration.build_migration_receipt(source,identity,bad)
    bad=copy.deepcopy(report);bad['cuda']['total_memory_bytes']=96*1024**3
    with pytest.raises(ValueError,match='24GB'):migration.build_migration_receipt(source,identity,bad)
    report['blocks']['full_attention']['next_update_after_restore_equal']=False
    with pytest.raises(ValueError):migration.build_migration_receipt(source,identity,report)
