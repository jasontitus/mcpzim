"""Negative proof-contract checks; never pretend these are actual GPU evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
import torch

from solver import gsq24


def hardware(monkeypatch,*,available=True,bf16=True,count=1,memory=24*1024**3):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:available)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:bf16)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:count)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(total_memory=memory))


@pytest.mark.parametrize('options',[{'available':False},{'bf16':False},{'count':0},{'count':2},
                                    {'memory':19*1024**3},{'memory':48*1024**3},{'memory':96*1024**3}])
def test_wrong_hardware_never_admitted(monkeypatch,options):
    hardware(monkeypatch,**options)
    with pytest.raises((RuntimeError,ValueError)):gsq24.checked_hardware()


def test_24gb_bf16_single_device_admitted(monkeypatch):
    hardware(monkeypatch)
    assert gsq24.checked_hardware()==torch.device('cuda:0')


def source_pair(tmp_path,*,progress=None,dtype=torch.bfloat16,tokens=4,role='teacher',largest_first=False):
    progress=progress or {'stage':'gsq','block':2,'global_step':174,'epoch':0,'sequence':0}
    torch.save(progress,tmp_path/'progress')
    pair={role:torch.ones(1,tokens,5120,dtype=dtype),'student':torch.zeros(1,tokens,5120,dtype=dtype)}
    stream=io.BytesIO();torch.save(pair,stream)
    with tarfile.open(tmp_path/'cache_block_2','w') as archive:
        data=stream.getvalue();member=tarfile.TarInfo('000000.pt' if largest_first else '000001.pt')
        member.size=len(data);archive.addfile(member,io.BytesIO(data))
    return {'largest_first':largest_first},[[1,2],[3,4,5,6]]


@pytest.mark.parametrize('largest_first',[False,True])
def test_full_longest_pair_obeys_source_sequence_order(tmp_path,largest_first):
    config,records=source_pair(tmp_path,largest_first=largest_first)
    pair,index,tokens=gsq24.pair_from_source(tmp_path,config,records)
    assert index==(0 if largest_first else 1) and tokens==4
    assert set(pair)=={'teacher','student'}
    assert all(value.shape==(1,4,5120) and value.dtype==torch.bfloat16 for value in pair.values())


@pytest.mark.parametrize('change',[{'stage':'rco'},{'block':3},{'global_step':175},{'epoch':1},{'sequence':1}])
def test_changed_source_progress_rejected(tmp_path,change):
    progress={'stage':'gsq','block':2,'global_step':174,'epoch':0,'sequence':0,**change}
    config,records=source_pair(tmp_path,progress=progress)
    with pytest.raises(ValueError,match='block2-start'):gsq24.pair_from_source(tmp_path,config,records)


@pytest.mark.parametrize('change',[{'tokens':3},{'dtype':torch.float32},{'role':'wrong_teacher'}])
def test_truncated_or_wrong_precision_source_pair_rejected(tmp_path,change):
    config,records=source_pair(tmp_path,**change)
    with pytest.raises(ValueError):gsq24.pair_from_source(tmp_path,config,records)


def test_non_file_cache_member_rejected(tmp_path):
    config,records=source_pair(tmp_path)
    with tarfile.open(tmp_path/'cache_block_2','w') as archive:
        member=tarfile.TarInfo('000001.pt');member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd'
        archive.addfile(member)
    with pytest.raises(ValueError,match='cache member'):gsq24.pair_from_source(tmp_path,config,records)


def test_candidate_fingerprints_bind_tensor_bits_independent_of_metadata_order(tmp_path):
    from safetensors.torch import save_file
    weights={'codes':torch.tensor([[1,4,7]],dtype=torch.uint8),'scales':torch.tensor([[1.]],dtype=torch.bfloat16)}
    first=tmp_path/'first.safetensors';second=tmp_path/'second.safetensors';changed=tmp_path/'changed.safetensors'
    save_file(weights,str(first),metadata={'one':'1','two':'2'})
    save_file(weights,str(second),metadata={'two':'2','one':'1'})
    assert gsq24.candidate_fingerprints({'p':str(first)})==gsq24.candidate_fingerprints({'p':str(second)})
    other={**weights,'codes':torch.tensor([[1,4,6]],dtype=torch.uint8)}
    save_file(other,str(changed),metadata={'two':'2','one':'1'})
    assert gsq24.candidate_fingerprints({'p':str(first)})!=gsq24.candidate_fingerprints({'p':str(changed)})


def report_fixture(block=2,phase='streamed'):
    config={'runtime_sha256':'a'*64,'source_receipt_sha256':'b'*64,'expected_longest_tokens':4673}
    check={'loss':0.001,'update_seconds':0.2,'propagation_seconds':0.1,
           'all_gradients_and_states_compared':phase!='reference'}
    report={'status':'passed','block':block,'phase':phase,'kind':{2:'linear_attention',3:'full_attention'}[block],
            **config,'sequence_tokens':4673,'source_sequence_index':0,
            'diagnostic_prefix':block==3,'production_progress_advanced':False,
            'cuda':{'available':True,'device':'Test L4','capability':[8,9],
                    'total_memory_bytes':24*1024**3,'peak_allocated_bytes':8*1024**3,'peak_reserved_bytes':9*1024**3},
            'checks':[{**check,'step':step} for step in ([2] if phase=='cold' else [1,2])],
            'projections':[{'name':'attn.q_proj','shape':[128,128],'dtype':'torch.float32'}],
            'candidate_sha256':{f'model.layers.{block}.attn.q_proj':{
                'metadata':{'format':'zimfo-q1-v1'},'tensors':{
                'codes':{'shape':[128,16],'dtype':'torch.uint8','sha256':'1'*64},
                'scales':{'shape':[128,1],'dtype':'torch.bfloat16','sha256':'2'*64}}}}}
    return report,config


@pytest.mark.parametrize('block',[2,3])
@pytest.mark.parametrize('phase',['reference','streamed','cold'])
def test_complete_worker_report_contract(block,phase):
    report,config=report_fixture(block,phase)
    assert gsq24.validate_worker_report(report,config,block,phase)==report


@pytest.mark.parametrize('key,value',[
    ('status','failed'),('block',3),('phase','reference'),('kind','full_attention'),
    ('runtime_sha256','9'*64),('source_receipt_sha256','9'*64),('sequence_tokens',4096),
    ('diagnostic_prefix',True),('production_progress_advanced',True),('source_sequence_index',-1),
    ('source_sequence_index',True),('checks',[]),('projections',[]),('candidate_sha256',{})])
def test_wrong_worker_proof_rejected(key,value):
    report,config=report_fixture();report[key]=value
    with pytest.raises(ValueError):gsq24.validate_worker_report(report,config,2,'streamed')


@pytest.mark.parametrize('key,value',[('available',False),('total_memory_bytes',96*1024**3),
    ('peak_allocated_bytes',0),('peak_reserved_bytes',25*1024**3),('device',''),('capability',[])])
def test_missing_actual_gpu_measurement_rejected(key,value):
    report,config=report_fixture();report['cuda'][key]=value
    with pytest.raises(ValueError):gsq24.validate_worker_report(report,config,2,'streamed')


@pytest.mark.parametrize('key,value',[('step',3),('all_gradients_and_states_compared',False),
    ('loss',float('nan')),('update_seconds',float('inf')),('propagation_seconds',-1)])
def test_unchecked_or_nonfinite_worker_update_rejected(key,value):
    report,config=report_fixture();report['checks'][0][key]=value
    with pytest.raises(ValueError):gsq24.validate_worker_report(report,config,2,'streamed')


@pytest.mark.parametrize('mismatch',['input_commit','corpus'])
def test_worker_rejects_current_input_mismatch_before_loading_model(tmp_path,monkeypatch,mismatch):
    source=tmp_path/'source';payloads=source/'payloads';payloads.mkdir(parents=True)
    receipt=source/'source-receipt.json';receipt.write_text('{}')
    corpus=tmp_path/'corpus';corpus.mkdir();manifest=corpus/'manifest.json';manifest.write_text('{}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    original={'runtime_sha256':gsq24.PRIOR_RUNTIME,'input_commit_sha256':'c'*64,
              'identity':{'calibration_sha256':sha(manifest)}}
    (payloads/'configuration').write_text(json.dumps(original))
    config={'source_checkpoint':str(source),'source_receipt_sha256':sha(receipt),
            'input_commit_sha256':'d'*64 if mismatch=='input_commit' else 'c'*64,'corpus':str(corpus)}
    if mismatch=='corpus':manifest.write_text('{"different":true}')
    monkeypatch.setattr(gsq24,'configure',lambda _:None)
    monkeypatch.setattr(gsq24,'checked_hardware',lambda:torch.device('cpu'))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda _:None)
    monkeypatch.setattr(gsq24,'validate_restored_inputs',lambda _:None)
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before reading pair or loading model')
    monkeypatch.setattr(gsq24,'corpus_inputs',forbidden);monkeypatch.setattr(gsq24,'load_original',forbidden)
    with pytest.raises(ValueError,match='calibration inputs differ'):
        gsq24.worker(config,tmp_path/'output',2,'reference')


def pin_runtime(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'accelerate',SimpleNamespace(__version__='1.13.0'))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(__version__='5.7.0'))
    monkeypatch.setattr(torch,'__version__','2.11.0+cu130')
    monkeypatch.setattr(gsq24,'digest',lambda p:gsq24.PRIOR_GSQ_SHA256 if Path(p).name=='gsq.py' else gsq24.PRIOR_QWEN_SHA256)


def test_reference_runtime_pins_complete_layout_stack(monkeypatch):
    pin_runtime(monkeypatch)
    gsq24.verify_reference_runtime()


@pytest.mark.parametrize('change',['gsq','qwen','torch','transformers','accelerate'])
def test_reference_runtime_rejects_changed_code_or_dependencies(monkeypatch,change):
    import sys
    pin_runtime(monkeypatch)
    if change in ('gsq','qwen'):
        original=gsq24.digest
        monkeypatch.setattr(gsq24,'digest',lambda p:'0'*64 if Path(p).stem==change else original(p))
    elif change=='torch':monkeypatch.setattr(torch,'__version__','2.11.0+cpu')
    else:sys.modules[change].__version__='different'
    with pytest.raises(ValueError,match='dependency versions changed'):gsq24.verify_reference_runtime()


def scratch_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(gsq24.shutil,'disk_usage',lambda _:SimpleNamespace(free=200*1024**3))
    for block in (2,3):
        work=tmp_path/f'block-{block}';work.mkdir()
        for phase in ('reference','streamed','cold'):
            (work/f'{phase}-report.json').write_text('{"status":"passed"}')
            candidate=work/f'{phase}-candidates';candidate.mkdir();(candidate/'keep').write_text('candidate')
        for name in ('reference-1.pt','reference-2.pt'):(work/name).write_bytes(b'fixture')
        for name in ('checkpoint','cold-restored'):
            directory=work/name;directory.mkdir();(directory/'state').write_bytes(b'fixture')
    return {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_prune_discards_only_reproducible_state_preserves_reports_and_candidates(tmp_path,monkeypatch):
    scratch_fixture(tmp_path,monkeypatch)
    removed=gsq24.prune_validation_scratch(tmp_path)
    assert len(removed)==8
    for block in (2,3):
        work=tmp_path/f'block-{block}'
        for phase in ('reference','streamed','cold'):
            assert (work/f'{phase}-report.json').is_file()
            assert (work/f'{phase}-candidates'/'keep').read_text()=='candidate'


def test_prune_validates_all_proofs_before_any_delete(tmp_path,monkeypatch):
    scratch_fixture(tmp_path,monkeypatch)
    (tmp_path/'block-3/cold-report.json').write_text('{"status":"failed"}')
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='unverified'):gsq24.prune_validation_scratch(tmp_path)
    after={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert after==before


def test_prune_rejects_symlink_without_touching_target(tmp_path,monkeypatch):
    scratch_fixture(tmp_path,monkeypatch)
    target=tmp_path/'original-source';target.write_bytes(b'never delete original')
    path=tmp_path/'block-2/reference-1.pt';path.unlink();path.symlink_to(target)
    with pytest.raises(ValueError,match='Unsafe'):gsq24.prune_validation_scratch(tmp_path)
    assert target.read_bytes()==b'never delete original'


def test_canary_calls_real_gsq_loop_and_verifies_new_diagnostic_checkpoint(tmp_path,monkeypatch):
    from checkpoints import capture_rng_state
    from solver.qwen import tiny_model
    from solver.gsq import BlockTrainer
    from solver.run import initialize_candidates, archive_directory, digest
    torch.set_num_threads(1)
    small=tiny_model();layout=copy.deepcopy(small.config);layout.num_hidden_layers=4;layout.layer_types*=2
    model=type(small)(layout).eval().requires_grad_(False)
    initial=tmp_path/'initial';initial.mkdir();initialize_candidates(model,initial)
    source=tmp_path/'payloads';source.mkdir()
    import shutil
    shutil.copyfile(initial/'initial-candidates.tar',source/'initial_candidates')
    records=[[1,3,7,9],[2,5,6]]
    cache=tmp_path/'cache';cache.mkdir()
    for index,tokens in enumerate(records):
        hidden=model.model.embed_tokens(torch.tensor([tokens])).detach()
        torch.save({'teacher':hidden,'student':hidden.clone()},cache/f'{index:06d}.pt')
    archive_directory(cache,source/'cache_block_2')
    trainer=BlockTrainer(model,2)
    optimizer=torch.optim.Adam(trainer.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,len(records))
    values={'solver':trainer.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'rng':capture_rng_state(),'progress':{'stage':'gsq','block':2,'global_step':174,'epoch':0,'sequence':0}}
    for role,value in values.items():torch.save(value,source/role)
    for block in (0,3):torch.save({'block':block,'updates':1},source/f'warmstart_block_{block}')
    (source/'packing_cost_manifest').write_text('{}')
    corpus=tmp_path/'corpus';corpus.mkdir()
    manifest={'model':'Qwen/Qwen3.8-27B','revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'}
    (corpus/'manifest.json').write_text(json.dumps(manifest))
    source_config={'largest_first':True,'checkpoint_seconds':120,'runtime_sha256':gsq24.PRIOR_RUNTIME,
                   'candidate_database':'/old-host/missing-database.json','candidate_archives':{},
                   'warmstart_sha256':{str(block):digest(source/f'warmstart_block_{block}') for block in (0,3)},
                   'cost_manifest':{'path':'/old-host/costs.json','sha256':digest(source/'packing_cost_manifest')}}
    source_receipt={'receipt':{'commit':{'object':'runs/old/commits/source.json','generation':1,'bytes':1,'sha256':'b'*64}}}
    config={'inputs':str(tmp_path),'model_dir':str(tmp_path),'corpus':str(corpus),'runtime_sha256':'a'*64,
            'deadline_unix':float('inf'),'source_receipt_sha256':'c'*64}
    # JSON configs cannot contain inf; the canary deadline is explicitly future.
    import time
    config['deadline_unix']=time.time()+3600
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    output=tmp_path/'output';output.mkdir()
    result=gsq24.production_canary(model,records,source_config,source,source_receipt,config,output,torch.device('cpu'))
    assert result['new_diagnostic_updates']==2 and result['checkpoint_verified'] is True
    assert result['diagnostic_only'] is True and result['production_progress_advanced'] is False
    assert result['progress']['global_step']==176 and result['progress']['sequence']==2
    assert result['identity']['runtime_sha256']=='a'*64
    frozen=json.loads((output/'production-canary/frozen-config.json').read_text())
    assert frozen['diagnostic_only'] is True
    assert frozen['diagnostic_source_commit']==source_receipt['receipt']['commit']
    updates=[g for g in result['performance']['groups'].values() if g['operation']=='update_with_input_load']
    assert sum(g['count'] for g in updates)==2
    assert sum(g['tokens'] for g in updates)==7
    assert result['cuda']=={'available':False}  # CPU fixture cannot be real24GB proof.


def canary_report_fixture():
    worker,config=report_fixture()
    old={'baseline_repo':'Qwen/Qwen3.8-27B','baseline_revision':'c'*40,
         'calibration_sha256':'d'*64,'runtime_sha256':gsq24.PRIOR_RUNTIME,
         'solver_revision':'e'*40,'solver_config_sha256':'f'*64,'candidate_database_sha256':'1'*64}
    identity={**old,'runtime_sha256':config['runtime_sha256'],'solver_config_sha256':'2'*64}
    source={'source_identity':old,'source_commit':{'generation':23,'object':'runs/old/source','sha256':'3'*64,'bytes':1}}
    tokens=[4673,4096]
    report={'status':'passed','scope':'actual_gsq_run_two_updates_from_source_block2',
            'new_diagnostic_updates':2,'checkpoint_verified':True,'diagnostic_only':True,
            'production_progress_advanced':False,'runtime_sha256':config['runtime_sha256'],
            'source_receipt_sha256':config['source_receipt_sha256'],'source_commit':source['source_commit'],
            'identity':identity,'input_tokens':tokens,
            'progress':{'stage':'gsq','block':2,'epoch':0,'sequence':2,'global_step':176},
            'cuda':worker['cuda'],'performance':{'groups':{'update':{'stage':'gsq',
            'operation':'update_with_input_load','status':'completed','count':2,'seconds':1.0}}}}
    return report,config,source,identity,tokens


def test_canary_proof_binds_diagnostic_identity_real_gpu_and_two_updates():
    report,config,source,identity,tokens=canary_report_fixture()
    assert gsq24.validate_canary_report(report,config,source,identity,tokens)==report


@pytest.mark.parametrize('damage',['source_commit','runtime','identity','production','cursor','tokens','cpu','large_gpu',
                                    'missing_updates','wrong_count','failed_update','nan_time','zero_time'])
def test_canary_proof_rejects_incomplete_or_mismatched_evidence(damage):
    report,config,source,identity,tokens=canary_report_fixture()
    report=copy.deepcopy(report)
    if damage=='source_commit':report['source_commit']['generation']+=1
    elif damage=='runtime':report['runtime_sha256']='9'*64
    elif damage=='identity':report['identity']['calibration_sha256']='9'*64
    elif damage=='production':report['production_progress_advanced']=True
    elif damage=='cursor':report['progress']['global_step']=175
    elif damage=='tokens':report['input_tokens']=[4096,4096]
    elif damage=='cpu':report['cuda']={'available':False}
    elif damage=='large_gpu':report['cuda']['total_memory_bytes']=96*1024**3
    elif damage=='missing_updates':report['performance']['groups']={}
    elif damage=='wrong_count':report['performance']['groups']['update']['count']=1
    elif damage=='failed_update':report['performance']['groups']['update']['status']='failed'
    elif damage=='nan_time':report['performance']['groups']['update']['seconds']=float('nan')
    elif damage=='zero_time':report['performance']['groups']['update']['seconds']=0
    with pytest.raises(ValueError):gsq24.validate_canary_report(report,config,source,identity,tokens)
