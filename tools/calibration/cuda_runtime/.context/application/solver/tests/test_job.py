"""Orchestration tests inject stage processes, never claim actual GPU feasibility."""
import argparse
import copy
import json
from pathlib import Path
import pytest
from solver import job


def checkpoint():
    return {'snapshot': 'stage-b000-s00000002-e000-q00002',
            'receipt': {'commit': {'object': 'runs/test/rco/snapshots/commit.json',
                         'generation': 123, 'bytes': 512, 'sha256': 'a'*64}}}


def partial(component):
    smoke = {'model': job.MODEL, 'revision': job.REVISION, 'sequence_tokens': 4096,
             'cuda': {'available': True, 'device': 'test GPU', 'capability': [12, 0],
                      'total_memory_bytes': 96000, 'peak_allocated_bytes': 60000,
                      'peak_reserved_bytes': 65000}}
    if component == 'gsq':
        smoke.update({name: {'passed': True, 'loss': .5, 'gradient_norm': .3}
                      for name in ('linear_attention_gsq', 'full_attention_gsq')})
    else:
        smoke.update(full_model_rco={'passed': True, 'full_vocabulary': True, 'gradient_norm': .4},
                     checkpoint_resume={'passed': True, 'next_update_matches': True,
                         'snapshot': checkpoint()['snapshot'], 'commit': checkpoint()['receipt']['commit']})
    return {'status': 'completed', 'smoke': smoke, 'optimizer_updates': 2,
            'durable_checkpoint': checkpoint(), 'warmstart_states': {}, 'cpu_test_only': False}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    inputs = tmp_path/'prepared'; (inputs/'calibration').mkdir(parents=True)
    manifest = {'model':job.MODEL, 'revision':job.REVISION, 'status':'completed',
                'sequences':[{'tokens':2048},{'tokens':4096}], 'source_invocation_count':2}
    job.save(inputs/'calibration/manifest.json',manifest)
    corpus_sha = job.sha(inputs/'calibration/manifest.json')
    job.save(inputs/'restore-validation.json', {'status':'validated','input_commit_sha256':'b'*64,'manifest_sha256':corpus_sha})
    costs = tmp_path/'costs.json'; job.save(costs, {'all_q1_serialized_bytes':100})
    monkeypatch.setattr(job,'COSTS',costs)
    args = argparse.Namespace(inputs=str(inputs),output=str(tmp_path/'job'),bucket='bucket',prefix='runs/test',
        input_commit_sha256='b'*64,runtime_sha256='c'*64,target_bytes=1000,deadline_seconds=2700,checkpoint_seconds=120)
    calls=[]
    def runner(stage,config_path,output,deadline):
        config=job.read(config_path); calls.append((stage,config))
        output.mkdir()
        job.save(output/'frozen-config.json',config)
        result={'status':'completed','runtime_sha256':args.runtime_sha256,'calibration_sha256':corpus_sha}
        if stage=='initialize':
            job.save(output/'initial-database.json',{'model.embed_tokens':'seed'})
            (output/'initial-candidates.tar').write_bytes(b'fake fixture archive')
        elif stage=='smoke':
            result.update(partial(config['smoke_component']))
            if config['smoke_component']=='gsq':result['warmstart_states']={'0':'test-warmstart.pt'}
            else:job.save(output/'latest-checkpoint.json',checkpoint())
        else:
            job.save(output/'latest-checkpoint.json',checkpoint())
            result.update(stage=stage,updates=2,candidate_database=str(output/'database.json'),
                candidate_archives={'initial_candidates':'archive'},candidate_sha256='d'*64)
            job.save(output/'database.json',{'fixture':'candidate'})
            if stage=='gsq':result['final_cache']=str(output/'cache-64')
            if stage=='rco':result['allocation']={'choices':{'fixture':'q1'},'estimated_tensor_bytes':1000}
        job.save(output/(stage+'-report.json'),result)
        return result
    return args,calls,runner


def test_full_sequence_and_boundary_contract(setup):
    args,calls,runner=setup
    result=job.execute(args,stage_runner=runner)
    assert result['status']=='ready_for_packaging'
    assert result['packaged_model_validated'] is False
    assert [s for s,c in calls]==['initialize','smoke','smoke','embedding','gsq','head','rco']
    assert calls[1][1]['smoke_component']=='gsq' and calls[2][1]['smoke_component']=='rco'
    assert calls[4][1]['warmstart_states']=={'0':'test-warmstart.pt'}
    assert calls[5][1]['final_cache'].endswith('cache-64')
    assert len(calls[-1][1]['boundary_reports'])==2
    for entry in calls[-1][1]['boundary_reports']:assert job.sha(entry['path'])==entry['sha256']
    assert len({c['checkpoint']['prefix'] for s,c in calls})==7
    assert len({c['output'] for s,c in calls})==7
    assert result['progress']['optimizer_updates']==12
    assert result['smoke']['passed'] is True
    assert result['progress']['resume']['commit_generation']==123


def test_controlled_stop_requires_real_checkpoint(setup):
    args,calls,runner=setup
    def stopped(stage,path,out,deadline):
        result=runner(stage,path,out,deadline)
        if stage=='embedding':
            result.update(status='checkpointed_stop',progress={'global_step':2})
            job.save(out/(stage+'-report.json'),result)
        return result
    result=job.execute(args,stage_runner=stopped)
    assert result['status']=='checkpointed'
    assert len(calls)==4
    assert result['progress']['phase']=='embedding'


def test_stopped_without_checkpoint_is_failed(setup):
    args,calls,runner=setup
    def stopped(stage,path,out,deadline):
        result=runner(stage,path,out,deadline)
        if stage=='embedding':
            (out/'latest-checkpoint.json').unlink()
            result['status']='checkpointed_stop';job.save(out/(stage+'-report.json'),result)
        return result
    with pytest.raises(ValueError,match='checkpoint'):job.execute(args,stage_runner=stopped)
    assert job.read(Path(args.output)/'status.json')['status']=='failed'


def test_failed_subprocess_prevents_later_training(setup):
    args,calls,runner=setup
    def failing(stage,path,out,deadline):
        if stage=='smoke':raise RuntimeError('CUDA out of memory')
        return runner(stage,path,out,deadline)
    with pytest.raises(RuntimeError,match='memory'):job.execute(args,stage_runner=failing)
    result=job.read(Path(args.output)/'status.json')
    assert result['status']=='failed' and len(calls)==1
    assert 'smoke' not in result


def test_wrong_stage_identity_rejected(setup):
    args,calls,runner=setup
    def changed(stage,path,out,deadline):
        result=runner(stage,path,out,deadline);result['runtime_sha256']='f'*64
        return result
    with pytest.raises(ValueError,match='identity'):job.execute(args,stage_runner=changed)
    assert len(calls)==1


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(cpu_test_only=True),
    lambda p:p['smoke'].update(sequence_tokens=100),
    lambda p:p['smoke'].update(model='Bonsai'),
    lambda p:p['smoke']['cuda'].update(available=False),
    lambda p:p['smoke']['cuda'].update(peak_reserved_bytes=float('nan')),
    lambda p:p['smoke']['full_model_rco'].update(full_vocabulary=False),
    lambda p:p['smoke']['full_model_rco'].update(gradient_norm=0),
    lambda p:p['smoke']['checkpoint_resume'].update(next_update_matches=False),
    lambda p:p['durable_checkpoint']['receipt']['commit'].update(generation=None),
])
def test_smoke_evidence_fails_closed(mutation):
    gsq,rco=partial('gsq'),partial('rco');mutation(rco)
    with pytest.raises(ValueError):job.merge_smoke(gsq,rco,4096)


def test_merge_uses_peak_across_processes():
    gsq,rco=partial('gsq'),partial('rco')
    gsq['smoke']['cuda']['peak_allocated_bytes']=70000
    result=job.merge_smoke(gsq,rco,4096)
    assert result['cuda']['peak_allocated_bytes']==70000


def test_budget_rejected_before_stage(setup):
    args,calls,runner=setup;args.target_bytes=99
    with pytest.raises(ValueError,match='budget'):job.execute(args,stage_runner=runner)
    assert not calls


def test_preexisting_solver_log_allowed_but_no_other_output(setup):
    args,calls,runner=setup
    output=Path(args.output);output.mkdir();(output/'solver.log').touch()
    job.execute(args,stage_runner=runner)
    with pytest.raises(FileExistsError):job.execute(args,stage_runner=runner)


def test_deadline_after_smoke_returns_useful_committed_progress(setup):
    args,calls,runner=setup
    def clock():return 2600 if len(calls)>=3 else 0
    result=job.execute(args,stage_runner=runner,clock=clock)
    assert result['status']=='checkpointed' and len(calls)==3
    assert result['progress']['optimizer_updates']==4


def test_deadline_before_smoke_is_failure(setup):
    args,calls,runner=setup
    def clock():return 2600 if len(calls)>=1 else 0
    with pytest.raises(TimeoutError):job.execute(args,stage_runner=runner,clock=clock)
    assert job.read(Path(args.output)/'status.json')['status']=='failed'


def test_cold_stage_recovery_after_source_directory_loss(setup, tmp_path, monkeypatch):
    """Real Adam+RNG state through production Checkpointer/GCS codec, empty host."""
    import io
    import shutil
    import torch
    from checkpoints import restore_rng_state
    from gcs_checkpoints import GCSCheckpointStore
    from solver import run
    args, _, _ = setup
    class Backend:
        def __init__(self): self.objects = {}; self.counter = 0
        def stat(self, key, generation=None):
            gen, data = self.objects[key]
            if generation is not None and generation != gen: raise FileNotFoundError(key)
            return {'object':key,'generation':gen,'bytes':len(data)}
        def create(self, key, stream, size):
            if key not in self.objects:
                self.counter += 1; data = stream.read(); assert len(data) == size
                self.objects[key] = (self.counter,data)
            return self.stat(key)
        def open(self,key,generation):
            self.stat(key,generation); return io.BytesIO(self.objects[key][1])
    backend = Backend()
    store = GCSCheckpointStore('bucket','runs/test/rco',backend=backend)
    monkeypatch.setattr(run,'make_store',lambda config:GCSCheckpointStore(
        'bucket',config['checkpoint']['prefix'],backend=backend))
    source = tmp_path/'lost-worker'; source.mkdir()
    costs = source/'costs.json';job.save(costs,{'costs':'fixture'})
    boundary = source/'report.json';job.save(boundary,{'stage':'embedding','status':'completed','updates':1})
    warm = source/'warm.pt';torch.save({'fixture':torch.ones(1)},warm)
    manifest=job.read(Path(args.inputs)/'calibration/manifest.json')
    config={'inputs':args.inputs,'corpus':str(Path(args.inputs)/'calibration'),
            'output':str(source),'runtime_sha256':args.runtime_sha256,
            'input_commit_sha256':args.input_commit_sha256,'checkpoint_seconds':120,
            'checkpoint':{'backend':'gcs','bucket':'bucket','prefix':'runs/test/rco'},
            'cost_manifest':{'path':str(costs),'sha256':job.sha(costs)},
            'boundary_reports':[{'path':str(boundary),'sha256':job.sha(boundary)}],
            'warmstart_states':{'0':str(warm)}}
    run.bind_identity(config,manifest)
    job.save(source/'frozen-config.json',config)
    torch.manual_seed(42)
    model=torch.nn.Linear(3,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
    def step():
        optimizer.zero_grad(); model(torch.randn(2,3)).square().sum().backward()
        optimizer.step();scheduler.step()
    step()
    checkpointer=run.Checkpointer(config,source)
    checkpointer.save(model,optimizer,scheduler,{'stage':'rco','epoch':0,'sequence':1,'global_step':1},{},force=True)
    durable=job.read(source/'latest-checkpoint.json')
    step(); expected={k:v.clone() for k,v in model.state_dict().items()}
    status={'runtime_sha256':args.runtime_sha256,'input_commit_sha256':args.input_commit_sha256,
            'progress':{'durable_checkpoint':durable,'resume':{'kind':'production_stage','stage':'rco',
                'bucket':'bucket','prefix':'runs/test/rco','commit_generation':durable['receipt']['commit']['generation']}}}
    status_path=tmp_path/'retained-status.json';job.save(status_path,status)
    shutil.rmtree(source)
    recovered=tmp_path/'empty-host'
    with monkeypatch.context() as limited:
        limited.setattr(job.shutil,'disk_usage',lambda _:type('Space',(),{'free':1024})())
        with pytest.raises(ValueError,match='disk space'):
            job.prepare_resume(status_path,args.inputs,recovered,args.runtime_sha256,store_factory=lambda _:store)
        assert not recovered.exists()
    result=job.prepare_resume(status_path,args.inputs,recovered,args.runtime_sha256,store_factory=lambda _:store)
    restored_config=job.read(result['configuration'])
    assert result['status']=='verified_stage_recovery_prepared'
    assert '--resume' in result['command']
    assert restored_config['output']==str(recovered/'stage')
    assert run.bind_identity(restored_config,manifest)==config['identity']
    assert Path(restored_config['boundary_reports'][0]['path']).exists()
    assert Path(restored_config['warmstart_states']['0']).exists()
    assert not source.exists()
    run.restore_state(recovered/'verified-checkpoint',model,optimizer,scheduler)
    step()
    for key,value in model.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    # Relocating configuration changes payload bytes. Republish at the same
    # optimizer cursor in a new branch, leaving the old immutable commit intact.
    stage_dir=recovered/'stage';stage_dir.mkdir()
    job.save(stage_dir/'frozen-config.json',restored_config)
    resumed_checkpointer=run.Checkpointer(restored_config,stage_dir)
    resumed_directory=resumed_checkpointer.restore(durable['snapshot'],durable['receipt']['commit']['generation'])
    assert (resumed_directory/'solver').is_file()
    resumed_checkpointer.save(model,optimizer,scheduler,{'stage':'rco','epoch':0,'sequence':1,'global_step':1},{},force=True)
    new_commit=job.read(stage_dir/'latest-checkpoint.json')['receipt']['commit']
    assert new_commit['object']!=durable['receipt']['commit']['object']
    assert backend.stat(durable['receipt']['commit']['object'])['generation']==durable['receipt']['commit']['generation']
    with pytest.raises(FileExistsError):job.prepare_resume(status_path,args.inputs,recovered,args.runtime_sha256,store_factory=lambda _:store)


def test_diagnostic_smoke_is_not_advertised_as_production_resume(setup,tmp_path):
    args,calls,runner=setup
    def clock():return 2600 if len(calls)>=3 else 0
    result=job.execute(args,stage_runner=runner,clock=clock)
    assert result['progress']['production_optimizer_updates']==0
    assert result['progress']['validation_optimizer_updates']==4
    assert result['progress']['resume']['kind']=='diagnostic_checkpoint'
    with pytest.raises(ValueError,match='Diagnostic'):
        job.prepare_resume(Path(args.output)/'status.json',args.inputs,tmp_path/'resume',args.runtime_sha256)
