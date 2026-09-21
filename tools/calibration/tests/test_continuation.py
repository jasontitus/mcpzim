import json
from pathlib import Path
import subprocess
import sys

import pytest
import continuation as module

RUNTIME = 'a'*64
IMAGE = 'us-central1-docker.pkg.dev/project/repo/runtime@sha256:'+RUNTIME


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2)+'\n').encode()
    path.write_bytes(data)
    return data


def add_checkpoint(root, stage, status):
    directory = root/module.DIRECTORIES[stage]
    identity = {'baseline_repo':module.MODEL, 'baseline_revision':module.REVISION,
                'runtime_sha256':RUNTIME, 'calibration_sha256':'b'*64,
                'solver_config_sha256':'c'*64, 'candidate_database_sha256':'d'*64,
                'solver_revision':'e'*40}
    config = {'stage':stage, 'identity':identity, 'runtime_sha256':RUNTIME,
              'input_commit_sha256':status['input_commit_sha256'],
              'checkpoint':{'backend':'gcs', 'bucket':'test-bucket', 'prefix':'runs/test/'+stage}}
    data = write(directory/'frozen-config.json',config)
    snapshot = stage+'-b000-s00000002-e001-q00000-p00002'
    roles = {k:{'member':k,'sha256':'f'*64,'bytes':10} for k in ('solver','optimizer','scheduler','rng','progress')}
    roles['configuration'] = {'sha256':module.digest(data),'bytes':len(data)}
    manifest = {'schema':3, 'snapshot':snapshot, 'identity':identity,
                'bucket':'test-bucket','prefix':config['checkpoint']['prefix'],'payloads':roles}
    encoded = json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    checkpoint = {'snapshot':snapshot,'receipt':{'manifest':manifest,
        'commit':{'object':manifest['prefix']+'/commits/'+snapshot+'.json',
                  'sha256':module.digest(encoded),'bytes':len(encoded),'generation':123}}}
    write(directory/'latest-checkpoint.json',checkpoint)
    return checkpoint


def boundary(root,status,stage):
    report = {'stage':stage,'status':'completed','updates':87,'candidate_sha256':'1'*64,
              'runtime_sha256':RUNTIME,'calibration_sha256':'b'*64}
    path = root/module.DIRECTORIES[stage]/(stage+'-report.json')
    data = write(path,report)
    status['stages'].append({'phase':stage,'status':'completed','report':'/old-host/'+path.name,
                             'sha256':module.digest(data)})
    return path


@pytest.fixture
def setup(tmp_path):
    root = tmp_path/'retained'; root.mkdir()
    status = {'runtime_sha256':RUNTIME,'input_commit_sha256':'2'*64,
              'status':'failed','smoke':{'passed':True},'stages':[],
              'progress':{'phase':'gsq','durable_checkpoint':{'snapshot':'stale-smoke'}}}
    boundary(root,status,'embedding')
    add_checkpoint(root,'embedding',status)
    newest = add_checkpoint(root,'gsq',status)
    write(root/'status.json',status)
    kwargs = dict(job_dir=root, output=tmp_path/'plan',runtime_image=IMAGE,
                  inputs='/future-host/inputs',recovery_output='/future-host/new-recovery')
    return root,status,newest,kwargs


def test_stale_root_reconciles_published_stage_and_preserves_report(setup):
    root,status,newest,kwargs = setup
    before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = module.prepare(**kwargs)
    assert result['selected_stage']=='gsq'
    assert result['root_checkpoint_reconciled'] is True
    reconciled = json.loads((kwargs['output']/'reconciled-status.json').read_text())
    assert reconciled['progress']['durable_checkpoint']==newest
    assert reconciled['progress']['resume']['stage']=='gsq'
    assert result['remaining_stages_after_selected_completes']==['head','rco']
    assert result['remote_checkpoint_verified'] is False
    assert result['automatic_remaining_phases'] is False
    command = result['preparation_command_inside_identical_runtime']
    assert '--execute' not in command
    assert result['runtime_image']==IMAGE
    for descriptor in result['completed_boundary_reports']:
        assert module.digest((kwargs['output']/descriptor['path']).read_bytes())==descriptor['sha256']
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage',['missing','changed','unbound','wrong_identity','zero_updates'])
def test_boundary_evidence_fail_closed(setup,damage):
    root,status,_,kwargs = setup
    path = root/'03-embedding/embedding-report.json'
    if damage=='missing':path.unlink()
    elif damage=='changed':path.write_text(path.read_text()+' ')
    elif damage=='unbound':status['stages']=[];write(root/'status.json',status)
    else:
        report=json.loads(path.read_text())
        if damage=='wrong_identity':report['runtime_sha256']='9'*64
        else:report['updates']=0
        data=write(path,report)
        status['stages'][0]['sha256']=module.digest(data);write(root/'status.json',status)
    with pytest.raises(ValueError,match='boundary|Boundary'):
        module.prepare(**kwargs)
    assert not kwargs['output'].exists()


def test_completed_stage_plans_terminal_resume_without_demanding_new_updates(setup):
    root,status,_,kwargs = setup
    report={'status':'completed','optimizer_updates':0,'runtime_sha256':RUNTIME,'calibration_sha256':'b'*64}
    data=write(root/'04-gsq/gsq-report.json',report)
    status['stages'].append({'phase':'gsq','status':'completed','sha256':module.digest(data)})
    write(root/'status.json',status)
    result = module.prepare(**kwargs)
    assert result['resume_terminal_stage_even_if_no_new_updates'] is True
    assert result['remaining_stages_after_selected_completes']==['head','rco']


def test_rco_requires_both_boundary_reports(setup):
    root,status,_,kwargs = setup
    add_checkpoint(root,'rco',status)
    with pytest.raises(ValueError,match='Missing completed boundary report: head'):
        module.prepare(**kwargs)
    boundary(root,status,'head');write(root/'status.json',status)
    result=module.prepare(**kwargs)
    assert result['selected_stage']=='rco'
    assert result['remaining_stages_after_selected_completes']==[]
    assert len(result['completed_boundary_reports'])==2


@pytest.mark.parametrize('damage',['runtime','mutable_image','frozen_config','receipt_hash','generation','stage','smoke'])
def test_identity_and_generation_rejections(setup,damage):
    root,status,newest,kwargs=setup
    path=root/'04-gsq/latest-checkpoint.json'
    if damage=='runtime':kwargs['runtime_image']=IMAGE.replace(RUNTIME,'9'*64)
    elif damage=='mutable_image':kwargs['runtime_image']='runtime:latest'
    elif damage=='frozen_config':
        p=root/'04-gsq/frozen-config.json';p.write_text(p.read_text()+' ')
    elif damage=='receipt_hash':newest['receipt']['commit']['sha256']='0'*64;write(path,newest)
    elif damage=='generation':newest['receipt']['commit']['generation']=True;write(path,newest)
    elif damage=='stage':newest['snapshot']='head-wrong';write(path,newest)
    elif damage=='smoke':status['smoke']['passed']=False;write(root/'status.json',status)
    with pytest.raises(ValueError):module.prepare(**kwargs)
    assert not kwargs['output'].exists()


def test_no_production_checkpoint_does_not_promote_smoke(tmp_path):
    root=tmp_path/'job';root.mkdir()
    write(root/'status.json',{'runtime_sha256':RUNTIME,'smoke':{'passed':True}})
    with pytest.raises(ValueError,match='No committed production'):
        module.prepare(root,tmp_path/'plan',IMAGE,'/inputs','/recovery')


def test_output_cannot_overwrite_or_modify_retained_job(setup):
    root,_,_,kwargs=setup
    with pytest.raises(ValueError,match='outside'):module.prepare(**{**kwargs,'output':root/'new-plan'})
    module.prepare(**kwargs)
    with pytest.raises(FileExistsError):module.prepare(**kwargs)


def test_symlinked_stage_rejected(setup,tmp_path):
    root,_,_,kwargs=setup
    target=tmp_path/'moved';(root/'04-gsq').rename(target);(root/'04-gsq').symlink_to(target)
    with pytest.raises(ValueError,match='Symlinked'):module.prepare(**kwargs)


def test_cli_prepares_only(setup):
    _,_,_,kwargs=setup
    command=[sys.executable,str(Path(module.__file__))]
    for key,value in kwargs.items():command.extend(['--'+key.replace('_','-'),str(value)])
    process=subprocess.run(command,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    result=json.loads(process.stdout)
    assert result['status']=='continuation_preparation_only'
    assert result['commands_executed'] is False
    assert (kwargs['output']/'continuation-plan.json').is_file()


def test_mutation_during_snapshot_is_rejected(setup, monkeypatch):
    root,_,_,kwargs=setup
    original=module.load
    path=root/'04-gsq/latest-checkpoint.json'
    count=0
    def changing(source):
        nonlocal count
        result=original(source)
        if Path(source)==path:
            count+=1
            if count==1:path.write_bytes(result[1]+b' ')
        return result
    monkeypatch.setattr(module,'load',changing)
    with pytest.raises(ValueError,match='changed during'):module.prepare(**kwargs)
    assert not kwargs['output'].exists()


def test_completed_root_does_not_bypass_report_hash(setup):
    root,status,_,kwargs=setup
    write(root/'04-gsq/gsq-report.json',{'status':'completed'})
    status['stages'].append({'phase':'gsq','status':'completed','sha256':'0'*64})
    write(root/'status.json',status)
    with pytest.raises(ValueError,match='Completed stage report'):module.prepare(**kwargs)


def committed_config(directory,config):
    data=write(directory/'frozen-config.json',config)
    stage=config['stage'];snapshot=stage+'-b000-s00000087-e001-q00000-p00001'
    manifest={'schema':3,'snapshot':snapshot,'identity':config['identity'],
        'prefix':config['checkpoint']['prefix'],'bucket':config['checkpoint']['bucket'],
        'payloads':{k:{'member':k,'sha256':'f'*64,'bytes':10} for k in ('solver','optimizer','scheduler','rng','progress')}}
    manifest['payloads']['configuration']={'sha256':module.digest(data),'bytes':len(data)}
    encoded=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    checkpoint={'snapshot':snapshot,'receipt':{'manifest':manifest,'commit':{
        'object':manifest['prefix']+'/commits/'+snapshot+'.json','generation':456,
        'bytes':len(encoded),'sha256':module.digest(encoded)}}}
    write(directory/'latest-checkpoint.json',checkpoint)
    return checkpoint


@pytest.fixture
def continuation_setup(setup,tmp_path):
    root,status,checkpoint,kwargs=setup
    inputs=tmp_path/'inputs';inputs.mkdir()
    kwargs.update(inputs=str(inputs),recovery_output=str(tmp_path/'unused-recovery'))
    module.prepare(**kwargs)
    calls=[]
    original=json.loads((root/'04-gsq/frozen-config.json').read_text())
    def runner(spec,deadline):
        calls.append(spec)
        destination=Path(spec['output']);destination.mkdir(parents=True,exist_ok=True)
        if spec['action']=='prepare':
            config=json.loads(json.dumps(original))
            config.update(resume_checkpoint=config['checkpoint'].copy())
            write(destination/'resume-config.json',config)
            write(destination/'recovery-receipt.json',{
                'status':'verified_stage_recovery_prepared','stage':'gsq',
                'source_commit':checkpoint['receipt']['commit'],'identity':config['identity'],
                'configuration':str(destination/'resume-config.json')})
            return
        config=json.loads(Path(spec['config']).read_text())
        if 'identity' not in config:config['identity']={**original['identity'],'solver_config_sha256':'8'*64}
        committed_config(destination,config)
        stage=spec['stage']
        report={'status':'completed','runtime_sha256':RUNTIME,'calibration_sha256':'b'*64,
                'optimizer_updates':0}
        if stage in ('embedding','head'):
            report.update(stage=stage,updates=87,candidate_sha256='1'*64)
        if stage in ('embedding','head','gsq'):
            write(destination/'database.json',{'model.embed_tokens':'fixture-only'})
            (destination/'candidates.tar').write_bytes(b'fixture archive; fake runner only')
            report.update(candidate_database=str(destination/'database.json'),
                          candidate_archives={'initial_candidates':str(destination/'candidates.tar')})
            if stage=='gsq':
                (destination/'final-cache').mkdir();report['final_cache']=str(destination/'final-cache')
        else:report['allocation']={'choices':{'fixture':'q1'}}
        write(destination/(stage+'-report.json'),report)
    arguments=dict(plan_dir=kwargs['output'],output=tmp_path/'continued',workspace=tmp_path,inputs=inputs)
    return arguments,runner,calls,original


def test_continue_completed_stage_zero_new_updates_to_remaining_stages(continuation_setup):
    arguments,runner,calls,original=continuation_setup
    result=module.continue_plan(**arguments,runner=runner,clock=lambda:1000)
    assert result['status']=='quantization_stages_completed'
    assert [v['stage'] for v in calls]==['gsq','gsq','head','rco']
    assert all(IMAGE in v['argv'] for v in calls)
    assert '--gpus=all' not in calls[0]['argv']
    assert all('--gpus=all' in v['argv'] for v in calls[1:])
    configs=[json.loads(Path(v['config']).read_text()) for v in calls[1:]]
    assert configs[0]['identity']==original['identity']
    assert 'identity' not in configs[1]
    assert configs[0]['resume_checkpoint']['prefix']==original['checkpoint']['prefix']
    assert len({v['checkpoint']['prefix'] for v in configs})==3
    assert all('/continue-' in v['checkpoint']['prefix'] for v in configs)
    assert all(v['deadline_unix']==2800 for v in configs)
    assert len(configs[1]['boundary_reports'])==1
    assert len(configs[2]['boundary_reports'])==2
    assert result['packaged_model_validated'] is False
    assert (arguments['output']/'03-embedding/embedding-report.json').is_file()


def test_stop_receipt_can_prepare_next_continuation(continuation_setup,tmp_path):
    arguments,runner,calls,_=continuation_setup
    def stopping(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='stage':
            path=Path(spec['output'])/(spec['stage']+'-report.json')
            report=json.loads(path.read_text());report.update(status='checkpointed_stop',progress={'global_step':5})
            write(path,report)
    result=module.continue_plan(**arguments,runner=stopping,clock=lambda:1000)
    assert result['status']=='checkpointed'
    assert len(calls)==2
    assert result['progress']['resume']['stage']=='gsq'
    next_plan=module.prepare(arguments['output'],tmp_path/'next-plan',IMAGE,arguments['inputs'],tmp_path/'next-recovery')
    assert next_plan['selected_checkpoint']['generation']==456
    assert next_plan['completed_boundary_reports'][0]['stage']=='embedding'


def test_failure_after_checkpoint_publication_keeps_latest_resume(continuation_setup):
    arguments,runner,_,_=continuation_setup
    def failing(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='stage':raise RuntimeError('simulated post-publication failure')
    with pytest.raises(RuntimeError,match='post-publication'):
        module.continue_plan(**arguments,runner=failing,clock=lambda:1000)
    status=json.loads((arguments['output']/'status.json').read_text())
    assert status['status']=='failed'
    assert status['progress']['durable_checkpoint']['receipt']['commit']['generation']==456
    assert status['progress']['resume']['stage']=='gsq'


def test_failed_prepare_preserves_original_checkpoint_fallback(continuation_setup,tmp_path):
    arguments,_,_,_=continuation_setup
    def failing(*_):raise RuntimeError('restore unavailable')
    with pytest.raises(RuntimeError,match='restore unavailable'):
        module.continue_plan(**arguments,runner=failing,clock=lambda:1000)
    next_plan=module.prepare(arguments['output'],tmp_path/'retry-plan',IMAGE,arguments['inputs'],tmp_path/'retry-recovery')
    assert next_plan['selected_checkpoint']['generation']==123


def test_deadline_stops_before_next_stage(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    times=iter([1000,1000,3500,3500])
    result=module.continue_plan(**arguments,runner=runner,clock=lambda:next(times))
    assert result['status']=='checkpointed'
    assert [x['action'] for x in calls]==['prepare','stage']
    assert result['stages'][-1]['phase']=='gsq'


def test_changed_prepared_evidence_blocks_all_execution(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    path=arguments['plan_dir']/'04-gsq/frozen-config.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='evidence changed'):
        module.continue_plan(**arguments,runner=runner)
    assert calls==[]
    assert not arguments['output'].exists()


def test_failed_stage_outcome_is_not_promoted(continuation_setup):
    arguments,runner,_,_=continuation_setup
    def invalid(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='stage':
            path=Path(spec['output'])/(spec['stage']+'-report.json')
            report=json.loads(path.read_text());report['status']='failed';write(path,report)
    with pytest.raises(ValueError,match='outcome'):
        module.continue_plan(**arguments,runner=invalid,clock=lambda:1000)
    assert json.loads((arguments['output']/'status.json').read_text())['status']=='failed'


def test_container_failure_cleanup_cannot_mask_original(tmp_path,monkeypatch):
    calls=[]
    def subprocess_run(argv,**kwargs):
        calls.append(argv)
        if argv[1]=='run':raise subprocess.TimeoutExpired(argv,10)
        raise OSError('Docker stop itself failed')
    monkeypatch.setattr(module.subprocess,'run',subprocess_run)
    spec={'argv':['docker','run','fixture'], 'container_name':'continue-owned-gsq','log':str(tmp_path/'log')}
    with pytest.raises(subprocess.TimeoutExpired):module.run_container(spec,1000)
    assert calls[-1]==['docker','stop','--time','90','continue-owned-gsq']


def test_successful_continuation_evidence_prepares_terminal_rco_resume(continuation_setup,tmp_path):
    arguments,runner,_,_=continuation_setup
    module.continue_plan(**arguments,runner=runner,clock=lambda:1000)
    result=module.prepare(arguments['output'],tmp_path/'terminal-plan',IMAGE,arguments['inputs'],tmp_path/'terminal-recovery')
    assert result['selected_stage']=='rco'
    assert result['resume_terminal_stage_even_if_no_new_updates'] is True
    assert [v['stage'] for v in result['completed_boundary_reports']]==['embedding','head']


def test_future_input_mount_is_read_only_and_no_unpinned_image(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    module.continue_plan(**arguments,runner=runner,clock=lambda:1000)
    for spec in calls:
        assert '--pull=never' in spec['argv']
        assert f"type=bind,src={arguments['inputs']},dst={arguments['inputs']},readonly" in spec['argv']
        assert not any(v.startswith('--privileged') for v in spec['argv'])


def test_failed_remote_preparation_receipt_blocks_stage_execution(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    def wrong_receipt(spec,deadline):
        runner(spec,deadline)
        path=Path(spec['output'])/'recovery-receipt.json'
        receipt=json.loads(path.read_text());receipt['source_commit']['generation']=789
        write(path,receipt)
    with pytest.raises(ValueError,match='verify the selected'):
        module.continue_plan(**arguments,runner=wrong_receipt,clock=lambda:1000)
    assert len(calls)==1
    assert calls[0]['action']=='prepare'


def test_real_cpu_solver_reports_through_gsq_head_rco_controller(tmp_path):
    """Real tiny solver math/artifacts and return schemas; cloud transport is fake.

    Run with pinned upstream RCO's src on PYTHONPATH. This is not CUDA validation.
    """
    pytest.importorskip('manifold',reason='Pinned RCO src must be on PYTHONPATH for real solver contract test')
    import torch
    from solver.qwen import tiny_model,projection_modules
    from solver.run import initialize_candidates,boundary_run,gsq_run,rco_run,Checkpointer,digest,bind_identity
    torch.set_num_threads(1)
    records=[[1,3,7,9]]
    def model():
        torch.manual_seed(21)
        return tiny_model()
    root=tmp_path/'retained';root.mkdir()
    inputs=tmp_path/'inputs';inputs.mkdir()
    baseline_manifest={'model':module.MODEL,'revision':module.REVISION}
    corpus=inputs/'calibration';corpus.mkdir()
    corpus_bytes=write(corpus/'manifest.json',baseline_manifest)
    corpus_sha=module.digest(corpus_bytes)
    initial=tmp_path/'initial';initial.mkdir()
    initialize_candidates(model(),initial)
    status={'runtime_sha256':RUNTIME,'input_commit_sha256':'2'*64,'status':'checkpointed',
            'smoke':{'passed':True},'stages':[],'progress':{'phase':'gsq'}}
    # Establish the same schema/identity envelope as a frozen production stage.
    add_checkpoint(root,'gsq',status)
    original=json.loads((root/'04-gsq/frozen-config.json').read_text())
    original['corpus']=str(corpus)
    original['identity']['calibration_sha256']=corpus_sha
    original.update(candidate_database=str(initial/'initial-database.json'),
        candidate_archives={'initial_candidates':str(initial/'initial-candidates.tar')},checkpoint_seconds=120)
    tensors={name+'.weight':{'shape':list(module.weight.shape),'q1_bytes':module.weight.numel()//128*18,
                             'bf16_bytes':module.weight.numel()*2}
             for name,module in projection_modules(model()).items()}
    cost_path=tmp_path/'costs.json';write(cost_path,{'fixed_container_bytes':0,'tensors':tensors})
    original.update(cost_manifest={'path':str(cost_path),'sha256':digest(cost_path)},
                    target_bytes=sum(v['q1_bytes'] for v in tensors.values())*2)
    def local_config(config,output):
        return {**config,'checkpoint':{'backend':'local','path':str(output/'local-store')}}
    embedding_out=root/'03-embedding';embedding_out.mkdir()
    conf=local_config(original,embedding_out)
    embedding=boundary_run('embedding',model(),records,conf,embedding_out,Checkpointer(conf,embedding_out))
    embedding.update(runtime_sha256=RUNTIME,calibration_sha256=corpus_sha)
    data=write(embedding_out/'embedding-report.json',embedding)
    committed_config(embedding_out,{**original,'stage':'embedding','checkpoint':{**original['checkpoint'],'prefix':'runs/test/embedding'}})
    status['stages'].append({'phase':'embedding','status':'completed','sha256':module.digest(data)})
    original.update({k:embedding[k] for k in ('candidate_database','candidate_archives')})
    original.pop('identity')
    bind_identity(original,baseline_manifest)
    gsq_out=root/'04-gsq'
    conf=local_config(original,gsq_out)
    cp=Checkpointer(conf,gsq_out)
    gsq=gsq_run(model(),records,conf,gsq_out,cp)
    snapshot=json.loads((gsq_out/'latest-checkpoint.json').read_text())['snapshot']
    restored=cp.restore(snapshot)
    checkpoint=committed_config(gsq_out,original)  # fake cloud envelope only
    gsq.update(runtime_sha256=RUNTIME,calibration_sha256=corpus_sha)
    data=write(gsq_out/'gsq-report.json',gsq)
    status['stages'].append({'phase':'gsq','status':'completed','sha256':module.digest(data)})
    write(root/'status.json',status)
    plan=tmp_path/'plan';module.prepare(root,plan,IMAGE,inputs,tmp_path/'recovery-unused')
    actual_reports={}
    def runner(spec,deadline):
        output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
        if spec['action']=='prepare':
            config=json.loads(json.dumps(original));config['resume_checkpoint']=config['checkpoint'].copy()
            write(output/'resume-config.json',config)
            write(output/'recovery-receipt.json',{'status':'verified_stage_recovery_prepared','stage':'gsq',
                'source_commit':checkpoint['receipt']['commit'],'identity':config['identity'],
                'configuration':str(output/'resume-config.json')})
            return
        config=json.loads(Path(spec['config']).read_text())
        bind_identity(config,baseline_manifest)
        if spec['stage']=='gsq':assert config['identity']==original['identity']
        else:assert config['identity']['solver_config_sha256']!=original['identity']['solver_config_sha256']
        local=local_config(config,output);local.pop('resume_checkpoint',None)
        checkpointer=Checkpointer(local,output)
        stage=spec['stage']
        if stage=='gsq':result=gsq_run(model(),records,local,output,checkpointer,resume=restored)
        elif stage=='head':result=boundary_run(stage,model(),records,local,output,checkpointer)
        else:result=rco_run(model(),records,local,output,checkpointer)
        result.update(runtime_sha256=RUNTIME,calibration_sha256=corpus_sha)  # exact solver.run.main envelope
        actual_reports[stage]=result
        write(output/(stage+'-report.json'),result)
        committed_config(output,config)  # controller sees generation-pinned fake cloud transport
    result=module.continue_plan(plan,tmp_path/'continued',tmp_path,inputs,runner=runner)
    assert result['status']=='quantization_stages_completed'
    assert actual_reports['gsq']['optimizer_updates']==0
    assert actual_reports['head']['updates']==1
    assert actual_reports['rco']['optimizer_updates']==1
    assert actual_reports['rco']['allocation']['serialized_container_overhead_included'] is True


def test_sigterm_stops_owned_container_restores_handlers_and_raises(tmp_path,monkeypatch):
    import signal
    handlers={signal.SIGINT:object(),signal.SIGTERM:object()};before=handlers.copy();calls=[]
    def register(sig,handler):
        old=handlers[sig];handlers[sig]=handler;return old
    def subprocess_run(argv,**kwargs):
        calls.append(argv)
        if argv[1]=='run':handlers[signal.SIGTERM](signal.SIGTERM,None)
    monkeypatch.setattr(module.signal,'signal',register)
    monkeypatch.setattr(module.subprocess,'run',subprocess_run)
    with pytest.raises(InterruptedError,match='Continuation interrupted'):
        module.run_container({'argv':['docker','run','fixture'],'container_name':'continue-owned-head',
                              'log':str(tmp_path/'log')},1000)
    assert calls[-1]==['docker','stop','--time','90','continue-owned-head']
    assert handlers==before


def test_insufficient_disk_blocks_continuation_before_any_container(continuation_setup,monkeypatch):
    from types import SimpleNamespace
    arguments,runner,calls,_=continuation_setup
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=1024))
    with pytest.raises(ValueError,match='disk headroom'):
        module.continue_plan(**arguments,runner=runner)
    assert calls==[]
    assert not arguments['output'].exists()


def test_five_hour_budget_keeps_recovery_bounded_and_stage_cleanup_inside_cap(continuation_setup):
    arguments, runner, calls, original = continuation_setup
    deadlines = []
    def record(spec, deadline):
        deadlines.append(deadline)
        return runner(spec, deadline)
    result = module.continue_plan(**arguments, deadline_seconds=18000, runner=record, clock=lambda:1000)
    assert result['runtime_budget']['absolute_deadline'] == 19000
    assert result['runtime_budget']['work_deadline'] == 18100
    assert result['runtime_budget']['automatic_extension'] is False
    assert deadlines[0] == 3700  # pinned prepare-resume remains <=45min
    assert all(value == 18880 for value in deadlines[1:])
    command = calls[0]['argv']
    assert command[command.index('--deadline-seconds')+1] == '2700'
    configs = [json.loads(Path(spec['config']).read_text()) for spec in calls[1:]]
    assert all(config['deadline_unix'] == 18100 for config in configs)
    assert configs[0]['identity'] == original['identity']


@pytest.mark.parametrize('kwargs', [
    {'deadline_seconds': True}, {'deadline_seconds': 86401}, {'deadline_seconds': 1000},
    {'checkpoint_reserve_seconds': 180}, {'checkpoint_reserve_seconds': True},
    {'checkpoint_reserve_seconds': 2600},
])
def test_runtime_budget_rejects_unbounded_or_inadequate_reserve(continuation_setup, kwargs):
    arguments, runner, calls, _ = continuation_setup
    with pytest.raises(ValueError):
        module.continue_plan(**arguments, runner=runner, **kwargs)
    assert not calls


def test_new_run_prefix_and_owned_container_supervision(continuation_setup):
    arguments, runner, calls, _ = continuation_setup
    owner = 'zimfo-gpu-123456789abc'
    ledger = arguments['workspace']/'supervision'/'active-container.json'
    module.continue_plan(**arguments, runner=runner, write_prefix='runs/'+owner,
                         owner_run_id=owner, owned_container_ledger=ledger)
    assert all(call['owner_run_id']==owner and call['owned_container_ledger']==str(ledger) for call in calls)
    for call in calls[1:]:
        config=json.loads(Path(call['config']).read_text())
        assert config['checkpoint']['prefix'].startswith('runs/'+owner+'/continue-')


def test_ledger_is_written_before_container_and_cleared_only_after_success(tmp_path,monkeypatch):
    from types import SimpleNamespace
    ledger=tmp_path/'ledger.json'
    ownership={'container_name':'continue-owned-gsq','owner_run_id':'zimfo-gpu-123456789abc'}
    def run(argv,**kwargs):
        assert json.loads(ledger.read_text())==ownership
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    module.run_container({'argv':['docker','run','fixture'],**ownership,
        'owned_container_ledger':str(ledger),'log':str(tmp_path/'log')},1000)
    assert not ledger.exists()


def test_cleanup_failure_retains_ledger_for_outer_supervisor(tmp_path,monkeypatch):
    ledger=tmp_path/'ledger.json'
    def run(argv,**kwargs):
        raise RuntimeError('owned container cleanup unavailable')
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='cleanup unavailable'):
        module.run_container({'argv':['docker','run','fixture'],'container_name':'continue-owned-gsq',
            'owner_run_id':'zimfo-gpu-123456789abc','owned_container_ledger':str(ledger),
            'log':str(tmp_path/'log')},1000)
    assert json.loads(ledger.read_text())['container_name']=='continue-owned-gsq'


def current_plan_runner(runner):
    """The test's fake prepare process follows whichever new plan it is given."""
    def prepared(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='prepare':
            argv=spec['argv'];status=json.loads(Path(argv[argv.index('--status')+1]).read_text())
            config=json.loads(Path(status['progress']['resume']['config']).read_text())
            config['resume_checkpoint']=config['checkpoint'].copy()
            destination=Path(spec['output'])
            write(destination/'resume-config.json',config)
            write(destination/'recovery-receipt.json',{
                'status':'verified_stage_recovery_prepared','stage':spec['stage'],
                'source_commit':status['progress']['durable_checkpoint']['receipt']['commit'],
                'identity':config['identity'],'configuration':str(destination/'resume-config.json')})
    return prepared


def prepare_offload_source(arguments,original):
    original.update(gsq_memory_mode='block_cpu_offload',gsq_execution_device='cuda:0')
    root=arguments['workspace']/'retained'
    committed_config(root/'04-gsq',original)
    plan=arguments['workspace']/'offload-plan'
    module.prepare(root,plan,IMAGE,arguments['inputs'],arguments['workspace']/'unused-recovery-offload')
    return {**arguments,'plan_dir':plan}


def test_explicit_gsq_handoff_never_launches_head_or_rco(continuation_setup):
    arguments,runner,calls,original=continuation_setup
    arguments=prepare_offload_source(arguments,original)
    result=module.continue_plan(**arguments,runner=current_plan_runner(runner),clock=lambda:1000,stop_after_stage='gsq')
    assert result['status']=='checkpointed' and result['stop_reason']=='requested_stage_boundary'
    assert [v['stage'] for v in calls]==['gsq','gsq']
    handoff=result['stage_handoff']
    assert handoff['completed_stage']=='gsq' and handoff['next_stage']=='head'
    assert handoff['remaining_stages']==['head','rco'] and handoff['terminal_recovery_required'] is True
    assert handoff['commit']==result['progress']['durable_checkpoint']['receipt']['commit']
    assert Path(handoff['candidate_database']).is_file() and Path(handoff['final_cache']).is_dir()
    assert all(Path(v).is_file() for v in handoff['candidate_archives'].values())
    config=json.loads(Path(calls[-1]['config']).read_text())
    assert config['identity']==original['identity']
    assert config['gsq_memory_mode']=='block_cpu_offload' and config['gsq_execution_device']=='cuda:0'
    assert 'stop_after_stage' not in config
    assert not (arguments['output']/'head-config.json').exists()
    assert not (arguments['output']/'rco-config.json').exists()


def test_completed_gsq_handoff_prepares_terminal_resume_then_larger_gpu_stages(continuation_setup):
    arguments,runner,calls,original=continuation_setup
    arguments=prepare_offload_source(arguments,original)
    dynamic_runner=current_plan_runner(runner)
    module.continue_plan(**arguments,runner=dynamic_runner,clock=lambda:1000,stop_after_stage='gsq')
    again_plan=arguments['workspace']/'handoff-plan'
    plan=module.prepare(arguments['output'],again_plan,IMAGE,arguments['inputs'],arguments['workspace']/'unused-later-recovery')
    assert plan['selected_stage']=='gsq' and plan['resume_terminal_stage_even_if_no_new_updates'] is True
    later={**arguments,'plan_dir':again_plan,'output':arguments['workspace']/'larger-gpu-attempt'}
    calls.clear()
    result=module.continue_plan(**later,runner=dynamic_runner,clock=lambda:2000)
    assert result['status']=='quantization_stages_completed'
    assert [v['stage'] for v in calls]==['gsq','gsq','head','rco']
    assert 'stop_reason' not in result and 'stage_handoff' not in result
    assert result['requested_stop_after_stage'] is None
    configurations=[json.loads(Path(v['config']).read_text()) for v in calls if v['action']=='stage']
    assert configurations[0]['identity']==original['identity']
    assert configurations[0]['gsq_memory_mode']=='block_cpu_offload'
    for config in configurations[1:]:
        assert 'gsq_memory_mode' not in config and 'gsq_execution_device' not in config
        assert 'identity' not in config
    assert json.loads((later['output']/'04-gsq/gsq-report.json').read_text())['optimizer_updates']==0


def test_gsq_stop_requires_completed_artifact_handoff(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    def broken(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='stage':
            path=Path(spec['output'])/'final-cache';path.rmdir()
    with pytest.raises(ValueError,match='Missing propagated final cache'):
        module.continue_plan(**arguments,runner=broken,clock=lambda:1000,stop_after_stage='gsq')
    status=json.loads((arguments['output']/'status.json').read_text())
    assert status['status']=='failed' and 'stage_handoff' not in status
    assert all(v['stage']=='gsq' for v in calls)


def test_mid_gsq_checkpoint_is_not_completed_handoff(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    def stopped(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='stage':
            path=Path(spec['output'])/'gsq-report.json';report=json.loads(path.read_text());report['status']='checkpointed_stop'
            write(path,report)
    result=module.continue_plan(**arguments,runner=stopped,clock=lambda:1000,stop_after_stage='gsq')
    assert result['status']=='checkpointed' and 'stage_handoff' not in result
    assert 'stop_reason' not in result
    assert all(v['stage']=='gsq' for v in calls)


@pytest.mark.parametrize('stop_after_stage',['head','rco',False,7])
def test_stop_policy_rejects_unsupported_stage_before_launch(continuation_setup,stop_after_stage):
    arguments,runner,calls,_=continuation_setup
    with pytest.raises(ValueError,match='explicit GSQ'):
        module.continue_plan(**arguments,runner=runner,stop_after_stage=stop_after_stage)
    assert not calls and not arguments['output'].exists()


def test_stop_policy_cannot_select_head_or_later_checkpoint(continuation_setup):
    arguments,runner,calls,_=continuation_setup
    root=arguments['workspace']/'retained';status=json.loads((root/'status.json').read_text())
    add_checkpoint(root,'head',status)
    later_plan=arguments['workspace']/'head-plan'
    module.prepare(root,later_plan,IMAGE,arguments['inputs'],arguments['workspace']/'unused-head-recovery')
    with pytest.raises(ValueError,match='precedes'):
        module.continue_plan(**{**arguments,'plan_dir':later_plan},runner=runner,stop_after_stage='gsq')
    assert not calls and not arguments['output'].exists()


def test_execute_cli_exposes_explicit_gsq_stop(monkeypatch,capsys):
    captured={}
    def execute(*args,**kwargs):captured.update(kwargs);return {'status':'checkpointed'}
    monkeypatch.setattr(module,'continue_plan',execute)
    monkeypatch.setattr(sys,'argv',['continuation.py','execute','--plan-dir','/plan','--output','/out',
                                   '--workspace','/work','--inputs','/inputs','--stop-after-stage','gsq'])
    module.main()
    assert captured['stop_after_stage']=='gsq'
    assert json.loads(capsys.readouterr().out)['status']=='checkpointed'
