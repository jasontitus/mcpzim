"""Host continuation contracts: no cloud, Docker, or GPU execution."""
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cloud_gpu import bootstrap, launch
from cloud_gpu.test_launch import receipt, successful_status


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inputs(tmp_path):
    ready=receipt(); ready_path=tmp_path/'ready.json';ready_path.write_text(json.dumps(ready))
    root=tmp_path/'source';root.mkdir()
    status=json.loads(json.dumps(successful_status()))
    status.update(runtime_sha256='a'*64,input_commit_sha256='c'*64)
    status['progress']['optimizer_update_totals_known']=False
    source=root/'reconciled-status.json';source.write_text(json.dumps(status))
    plan={'schema_version':1,'status':'continuation_preparation_only','runtime_image':ready['production_image'],
          'runtime_sha256':'a'*64,'selected_checkpoint':status['progress']['durable_checkpoint']['receipt']['commit'],
          'recovery_required_free_bytes':11*1024**3,
          'files':{'reconciled-status.json':{'sha256':launch.sha(source),'bytes':source.stat().st_size}}}
    plan_path=root/'continuation-plan.json';plan_path.write_text(json.dumps(plan))
    return ready_path,plan_path,plan,status


def prepared(tmp_path,staged=True):
    ready_path,plan_path,plan,status=inputs(tmp_path)
    staging={'status':'continuation_staged','data_disk_id':'2','plan_path':'/mnt/zimfo-inputs/continuation-plans/third',
             'plan_sha256':launch.sha(plan_path),'files':plan['files'],'free_bytes':20*1024**3}
    staging_path=tmp_path/'staging.json';staging_path.write_text(json.dumps(staging))
    out=tmp_path/'launch'
    config=launch.prepare(ready_path,out,max_seconds=18000,continuation_plan=plan_path,
                         staged_plan_path=staging['plan_path'],continuation_staging=staging_path if staged else None)
    return out,config,status


def test_controller_plan_and_staging_are_frozen_before_any_cloud_call(tmp_path,monkeypatch):
    out,config,_=prepared(tmp_path)
    assert config['max_seconds']==18000
    assert config['continuation']['controller_sha256']==launch.sha(out/'continuation.py')
    args=launch.create_command(config,out,'2026-09-20T10:00:00+00:00')
    assert 'zimfo-continuation-controller='+str(out/'continuation.py') in next(v for v in args if v.startswith('--metadata-from-file='))
    (out/'continuation.py').write_text('changed')
    monkeypatch.setattr(launch,'run',lambda _:pytest.fail('cloud call before changed controller rejected'))
    with pytest.raises(ValueError,match='controller or plan changed'): launch.execute(out)


def test_no_staging_receipt_no_paid_launch(tmp_path,monkeypatch):
    out,_,_=prepared(tmp_path,staged=False)
    monkeypatch.setattr(launch,'run',lambda _:pytest.fail('paid call before staging proof'))
    with pytest.raises(ValueError,match='CPU-staged'):launch.execute(out)


def test_runtime_ancestry_and_input_rejected_before_preparation(tmp_path):
    ready_path,plan_path,plan,status=inputs(tmp_path)
    plan['runtime_image']=plan['runtime_image'].replace('a'*64,'b'*64)
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='binding'):
        launch.prepare(ready_path,tmp_path/'out',continuation_plan=plan_path,staged_plan_path='/mnt/zimfo-inputs/continuation-plans/a')
    assert not (tmp_path/'out').exists()


def test_all_continuation_deadlines_consume_startup_and_preserve_cleanup():
    start=datetime(2026,9,20,tzinfo=timezone.utc).timestamp()
    end=datetime.fromtimestamp(start+18000,timezone.utc).isoformat()
    a=bootstrap.continuation_deadlines(end,now=start+300)
    assert a=={'guest_seconds':17520,'bootstrap_seconds':17340,'controller_seconds':17100}
    assert a['controller_seconds']+120 < a['bootstrap_seconds'] < a['guest_seconds'] < 17700
    with pytest.raises(ValueError,match='Insufficient'):bootstrap.continuation_deadlines(end,now=start+17000)


def continuation_status(config,source):
    status=copy.deepcopy(source)
    status.update(runtime_image=config['continuation']['runtime_image'],packaged_model_validated=False)
    return status


def test_unknown_totals_and_exact_ancestry_allowed_without_inventing_updates(tmp_path):
    _,config,source=prepared(tmp_path)
    status=continuation_status(config,source)
    status['progress'].pop('optimizer_updates')
    assert bootstrap.validate_continuation_status(config,status) is status
    changed=copy.deepcopy(status)
    changed['progress']['durable_checkpoint']['receipt']['commit']['generation']='999'
    with pytest.raises(ValueError,match='ancestry'):bootstrap.validate_continuation_status(config,changed)
    changed=copy.deepcopy(status);changed['smoke']['cuda']['peak_allocated_bytes']=1
    with pytest.raises(ValueError,match='identity'):bootstrap.validate_continuation_status(config,changed)


def test_stage_completion_never_means_packaged_model(tmp_path):
    _,config,source=prepared(tmp_path)
    status=continuation_status(config,source)
    status.update(status='quantization_stages_completed',allocation={'layer':'q1'})
    status['progress']['phase']='rco'
    assert bootstrap.validate_continuation_status(config,status) is status
    status['packaged_model_validated']=True
    with pytest.raises(ValueError):bootstrap.validate_continuation_status(config,status)


def test_ancestor_remote_reads_require_exact_generation_and_digest(tmp_path,monkeypatch):
    _,config,source=prepared(tmp_path)
    status=continuation_status(config,source)
    status['progress']['durable_checkpoint']['receipt']['commit']['generation']='999'
    # Keep smoke ancestor valid: reject altered durable before any of its reads.
    monkeypatch.setattr(bootstrap,'metadata',lambda _:b'{"access_token":"test"}')
    seen=[]
    def remote(request,**kwargs):
        seen.append(request.full_url)
        raise AssertionError('Should reject status before remote verification')
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',remote)
    with pytest.raises(ValueError,match='ancestry'):bootstrap.validate_continuation_status(config,status)
    assert not seen


def test_supervisor_log_does_not_precreate_controller_output(tmp_path):
    output=tmp_path/'new-output';supervisor=tmp_path/'supervisor.log'
    child="from pathlib import Path;import sys;p=Path(sys.argv[1]);assert not p.exists();p.mkdir();print('created')"
    result=bootstrap.run_solver([sys.executable,'-c',child,str(output)],output,
                               log_path=supervisor,poll_seconds=.01,emit=lambda _:None)
    assert result.returncode==0 and supervisor.read_text()=='created\n'
    assert not (output/'solver.log').exists()


@pytest.mark.parametrize('owner',['zimfo-gpu-abcdef123456','wrong'])
def test_owned_container_cleanup_never_targets_unowned(tmp_path,monkeypatch,owner):
    ledger=tmp_path/'active.json';ledger.write_text(json.dumps({'owner_run_id':owner,
      'container_name':'continue-123456abcdef-gsq','publisher_container_name':'continue-123456abcdef-gsq-publisher'}))
    calls=[]
    monkeypatch.setattr(bootstrap.subprocess,'run',lambda args,**kw:calls.append(args))
    monkeypatch.setattr(bootstrap,'serial_summary',lambda _:False)
    bootstrap.cleanup_owned_container({'run_id':'zimfo-gpu-abcdef123456'},ledger)
    if owner=='wrong':assert not calls
    else:assert calls==[['docker','stop','--time=90','continue-123456abcdef-gsq','continue-123456abcdef-gsq-publisher']]


def test_async_is_opt_in_and_requires_cpu_publisher_capacity_receipt(tmp_path,monkeypatch):
    ready_path,plan_path,plan,source=inputs(tmp_path)
    out=tmp_path/'async-launch'
    config=launch.prepare(ready_path,out,max_seconds=18000,continuation_plan=plan_path,
             staged_plan_path='/mnt/zimfo-inputs/continuation-plans/a',checkpoint_mode='local-spool',
             spool_min_free_bytes=100*1024**3)
    c=config['continuation']
    assert c['publisher_sha256']==launch.sha(out/'checkpoint_bridge.py')
    metadata=next(a for a in launch.create_command(config,out,'deadline') if a.startswith('--metadata-from-file='))
    assert 'zimfo-checkpoint-publisher=' in metadata
    c['staging_receipt']={'status':'continuation_staged','data_disk_id':'2','plan_path':c['plan_path'],
                        'plan_sha256':c['plan_sha256'],'files':c['files'],'free_bytes':101*1024**3}
    with pytest.raises(ValueError,match='publisher/spool'):bootstrap.validate_staging_receipt(config)
    c['staging_receipt'].update(checkpoint_mode='local-spool',publisher_sha256=c['publisher_sha256'],
                               spool_min_free_bytes=c['spool_min_free_bytes'])
    bootstrap.validate_staging_receipt(config)
    c['staging_receipt']['free_bytes']=99*1024**3
    with pytest.raises(ValueError,match='publisher/spool'):bootstrap.validate_staging_receipt(config)


def test_execute_freezes_same_five_hour_provider_deadline_in_guest_config(tmp_path,monkeypatch):
    out,config,_=prepared(tmp_path)
    monkeypatch.setattr(launch,'ensure_ingress_denied',lambda:None)
    calls=[]
    def run(args):
        calls.append(args)
        if args[2:4]==['disks','describe']:
            role='boot_disk' if args[4].endswith('-boot') else 'data_disk'
            return {'id':config['ready'][role]['id'],'status':'READY','labels':{'zimfo-run':config['ready']['run_id']}}
        if args[2:4]==['instances','list']:return []
        if args[2:4]==['instances','create']:return [{'id':'new'}]
        pytest.fail(str(args))
    monkeypatch.setattr(launch,'run',run)
    before=datetime.now(timezone.utc).timestamp()
    result=launch.execute(out)
    frozen=json.loads((out/'config.json').read_text())
    hard=datetime.fromisoformat(frozen['absolute_deadline']).timestamp()
    assert 17998 <= hard-before <= 18001
    assert frozen==result['config']
    expected=digest({k:v for k,v in frozen.items() if k!='config_sha256'})
    assert frozen['config_sha256']==expected
    create=calls[-1]
    assert '--termination-time='+frozen['absolute_deadline'] in create
    assert (out/'absolute-deadline.txt').read_text()==frozen['absolute_deadline']


def test_cleanup_rejects_publisher_from_different_attempt(tmp_path,monkeypatch):
    ledger=tmp_path/'active.json';ledger.write_text(json.dumps({'owner_run_id':'zimfo-gpu-abcdef123456',
      'container_name':'continue-123456abcdef-gsq','publisher_container_name':'continue-abcdef123456-gsq-publisher'}))
    monkeypatch.setattr(bootstrap.subprocess,'run',lambda *a,**kw:pytest.fail('Unrelated publisher targeted'))
    monkeypatch.setattr(bootstrap,'serial_summary',lambda _:False)
    bootstrap.cleanup_owned_container({'run_id':'zimfo-gpu-abcdef123456'},ledger)
