import json
from pathlib import Path
import pytest
from cloud_gpu import gsq24_prep as prep
from cloud_gpu.test_launch import receipt


def ready(tmp_path):
    path=tmp_path/'ready.json';path.write_text(json.dumps(receipt()));return path


def test_cpu_plan_uses_compatible_new_pd_disks_and_plain_os(tmp_path):
    plan=prep.prepare(ready(tmp_path),tmp_path/'plan')
    boot,data=plan['disk_commands']
    assert '--type=pd-balanced' in boot and '--size=80GB' in boot
    assert '--type=pd-ssd' in data and '--size=256GB' in data
    assert '--image-project=ubuntu-os-cloud' in boot
    assert not any('hyperdisk' in item or 'deeplearning' in item for cmd in (boot,data) for item in cmd)
    command=prep.create_command(plan['config'],tmp_path/'plan','deadline')
    assert '--machine-type=n2-standard-4' in command
    assert '--termination-time=deadline' in command and '--instance-termination-action=DELETE' in command
    assert all('auto-delete=no' in arg for arg in command if arg.startswith('--disk='))
    assert not any('accelerator=' in arg for arg in command)


def test_mutated_cpu_script_rejected_before_cloud_calls(tmp_path,monkeypatch):
    out=tmp_path/'plan';prep.prepare(ready(tmp_path),out)
    (out/'gsq24_prep_guest.py').write_text('changed')
    monkeypatch.setattr(prep,'run',lambda _:pytest.fail('Cloud call before script binding'))
    with pytest.raises(ValueError,match='script'):prep.execute(out)


def test_execute_uses_only_new_owned_disks_and_refuses_retry(tmp_path,monkeypatch):
    out=tmp_path/'plan';plan=prep.prepare(ready(tmp_path),out);name=plan['config']['run_id'];calls=[]
    monkeypatch.setattr(prep,'ensure_ingress_denied',lambda:None)
    def run(args):
        calls.append(args)
        if args[2:4]==['images','describe']:return {'id':prep.OS_IMAGE_ID,'status':'READY'}
        if args[2:4]==['instances','list']:return []
        if args[2:4]==['disks','create']:
            disk=args[4];assert disk.startswith(name)
            return [{'name':disk,'id':'123','selfLink':'projects/test/disks/'+disk,'type':'types/pd-balanced' if disk.endswith('boot') else 'types/pd-ssd'}]
        if args[2:4]==['instances','create']:return [{'name':name,'id':'42'}]
        pytest.fail(str(args))
    monkeypatch.setattr(prep,'run',run)
    result=prep.execute(out)
    assert result['status']=='launched' and len(calls)==5
    with pytest.raises(ValueError,match='attempted'):prep.execute(out)


def test_cpu_startup_has_bounded_deadline_and_no_gpu_logic():
    script=Path(prep.HERE/'gsq24_prep_startup.sh').read_text()
    assert '--on-active=55m' in script and '3150s' in script
    assert 'shutdown -h now' in script and 'cpu_common.py' in script
    guest=Path(prep.HERE/'gsq24_prep_guest.py').read_text()
    assert "'--runtime=runc'" in guest and "'--pull=never'" in guest
    assert 'nvidia-driver-pinning-580' in guest
    assert "n2-standard-4" in guest and 'prepare_gsq24_source' in guest


def test_cpu_retry_reuses_verified_disks_without_format_or_creation(tmp_path):
    out=tmp_path/'first';plan=prep.prepare(ready(tmp_path),out)
    owner=plan['config']['run_id'];plan['status']='finished_failed'
    plan['config']['disks']={role:{'name':owner+suffix,'id':str(index),'self_link':'disk','type':kind}
        for index,(role,suffix,kind) in enumerate((('boot_disk','-boot','pd-balanced'),('data_disk','-inputs','pd-ssd')),1)}
    (out/'plan.json').write_text(json.dumps(plan))
    second=prep.prepare_resume(out,tmp_path/'second')
    assert second['disk_commands']==[] and second['config']['reuse_disks'] is True
    assert second['config']['disk_owner']==owner and second['config']['run_id']!=owner
    args=prep.create_command(second['config'],tmp_path/'second','deadline')
    assert any('--disk=name='+owner+'-boot,' in arg for arg in args)
    assert not (tmp_path/'second/final-runtime.json').exists()


def test_public_apt_trust_material_readable_under_strict_startup_umask():
    source=(prep.HERE/'gsq24_prep_guest.py').read_text()
    assert "Path('/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg').chmod(0o644)" in source
    assert "Path('/etc/apt/sources.list.d/nvidia-container-toolkit.list').chmod(0o644)" in source
    assert "if config.get('reuse_disks'):" in source
    assert "else:common.mount_inputs()" in source


def l4_ready():
    from cloud_gpu.gsq24 import SOURCE_IDENTITY,SOURCE_COMMIT,SOURCE_RECEIPT_SHA256
    result=receipt();result.update(status='gsq24_prepared_disks_detached',run_id='zimfo-gsq24prep-abcdef123456',
        source_receipt_sha256=SOURCE_RECEIPT_SHA256,free_bytes=140*1024**3,
        runtime_proof={'driver':'580.178.04','kernel':'kernel','driver_module':'/lib/modules/kernel/nvidia.ko'})
    result['restore_validation'].update(manifest_sha256=SOURCE_IDENTITY['calibration_sha256'])
    result['source_validation']={'status':'source_payloads_verified','source_commit':SOURCE_COMMIT,
        'source_manifest_sha256':SOURCE_COMMIT['sha256'],'source_identity':SOURCE_IDENTITY,'migration_applied':False,
        'configuration':{'identity':SOURCE_IDENTITY,'runtime_sha256':SOURCE_IDENTITY['runtime_sha256'],'stage':'gsq'}}
    for role,suffix in (('boot_disk','boot'),('data_disk','inputs')):
        result[role].update(name=result['run_id']+'-'+suffix,type='pd-balanced' if role=='boot_disk' else 'pd-ssd')
        result[role]['self_link']='https://www.googleapis.com/compute/v1/projects/tiltastech-zimfo/zones/us-central1-b/disks/'+result[role]['name']
    return result


def report(config):
    ready=config['ready'];runtime=ready['production_image'].split('@sha256:')[1]
    cuda={'available':True,'device':'NVIDIA L4','capability':[8,9],'total_memory_bytes':24*1024**3,
          'peak_allocated_bytes':1024**3,'peak_reserved_bytes':2*1024**3}
    workers=[]
    for block in (2,3):
        for phase in ('reference','streamed','cold'):
            workers.append({'status':'passed','runtime_sha256':runtime,'source_receipt_sha256':ready['source_receipt_sha256'],
                'block':block,'phase':phase,'kind':'linear_attention' if block==2 else 'full_attention',
                'sequence_tokens':4673,'diagnostic_prefix':block==3,'production_progress_advanced':False,'cuda':dict(cuda),
                'checks':[{'step':step,'loss':.2,'update_seconds':2.,'propagation_seconds':.1,'all_gradients_and_states_compared':phase!='reference'}
                    for step in ([2] if phase=='cold' else [1,2])],
                'projections':[{'name':'q_proj'}],'candidate_sha256':{f'model.layers.{block}.q_proj':{'metadata':{'format':'q1'},
                    'tensors':{key:{'shape':[1],'dtype':'float32','sha256':'a'*64} for key in ('codes','scales')}}}})
    return {'status':'passed','actual_24gb_validated':True,'numerical_parity_passed':True,'cold_checkpoint_replay_passed':True,
       'migration_applied':False,'production_progress_advanced':False,'production_gsq_loop_cuda_validated':True,
       'full_corpus_gsq_completed':False,'runtime_sha256':runtime,'inherited_gsq_updates':174,'inherited_embedding_updates':87,
       'complete_job_eta_seconds':None,'workers':workers,'peak_allocated_bytes':1024**3,'peak_reserved_bytes':2*1024**3,
       'production_canary':{'status':'passed','runtime_sha256':runtime,'source_receipt_sha256':ready['source_receipt_sha256'],
          'new_diagnostic_updates':2,'checkpoint_verified':True,'diagnostic_only':True,'production_progress_advanced':False,
          'source_commit':ready['source_validation']['source_commit'],
          'identity':dict(ready['source_validation']['source_identity'],runtime_sha256=runtime,solver_config_sha256='b'*64),
          'progress':{'stage':'gsq','block':2,'epoch':0,'global_step':176,'sequence':2},
          'performance':{'groups':{'measured':{'stage':'gsq','operation':'update_with_input_load',
              'status':'completed','count':2,'seconds':5.,'dimensions':{'block':2}}}},'cuda':dict(cuda)}}


def test_gpu_ready_rejects_hyperdisk_wrong_ancestry_and_small_host_scratch():
    import copy
    from cloud_gpu.gsq24 import validate_ready
    ready=l4_ready();validate_ready(ready)
    for mutate in (lambda r:r['boot_disk'].update(type='hyperdisk-balanced'),
                   lambda r:r['source_validation'].update(source_commit={}),
                   lambda r:r['restore_validation'].update(input_commit_sha256='x'),
                   lambda r:r.update(free_bytes=109*1024**3)):
        changed=copy.deepcopy(ready);mutate(changed)
        with pytest.raises(ValueError):validate_ready(changed)


def test_gpu_report_requires_all_real_hybrid_updates_canary_and_24gb_evidence():
    import copy
    from cloud_gpu.gsq24_guest import validate_report
    config={'ready':l4_ready()};valid=report(config)
    assert validate_report(config,valid) is valid
    mutations=[lambda r:r['workers'].pop(),lambda r:r['workers'][0]['cuda'].update(total_memory_bytes=96*1024**3),
       lambda r:r['workers'][2]['checks'][0].update(all_gradients_and_states_compared=False),
       lambda r:r['workers'][3].update(source_receipt_sha256='b'*64),
       lambda r:r['production_canary'].update(checkpoint_verified=False),
       lambda r:r['production_canary']['progress'].update(global_step=999),
       lambda r:r.pop('production_canary'),
       lambda r:r['production_canary']['performance']['groups']['measured'].update(seconds=float('nan')),
       lambda r:r['production_canary']['performance']['groups']['measured'].update(operation='checkpoint'),
       lambda r:r['production_canary']['performance']['groups']['measured'].update(count=1),
       lambda r:r['production_canary']['performance']['groups']['measured']['dimensions'].update(block=3),
       lambda r:r['production_canary']['progress'].update(stage='rco'),
       lambda r:r['production_canary'].update(identity=config['ready']['source_validation']['source_identity']),
       lambda r:r['production_canary']['identity'].update(calibration_sha256='c'*64),
       lambda r:r.update(full_corpus_gsq_completed=True),lambda r:r.update(peak_reserved_bytes=1)]
    for mutate in mutations:
        changed=copy.deepcopy(valid);mutate(changed)
        with pytest.raises(ValueError):validate_report(config,changed)


def test_gpu_container_never_installs_pulls_or_writes_source():
    from cloud_gpu.gsq24_guest import docker_command
    args=docker_command({'ready':l4_ready(),'run_id':'zimfo-gsq24-abcdef123456'},'/host/config.json','/host/output')
    assert '--pull=never' in args and '--network=none' in args
    assert 'type=bind,src=/mnt/zimfo-inputs/gsq24-source,dst=/inputs/gsq24-source,readonly' in args
    assert 'type=bind,src=/mnt/zimfo-inputs/prepared,dst=/inputs/prepared,readonly' in args
    assert args[-2:]==['--output','/output/results']


def test_gpu_bootstrap_changed_before_cloud_is_rejected(tmp_path,monkeypatch):
    from cloud_gpu import gsq24
    r=l4_ready();ready_path=tmp_path/'r.json';ready_path.write_text(json.dumps(r))
    harness={'inputs':'/inputs/prepared','source_checkpoint':'/inputs/gsq24-source',
       'runtime_sha256':r['production_image'].split('@sha256:')[1],'source_receipt_sha256':r['source_receipt_sha256'],
       'input_commit_sha256':r['input_commit']['sha256'],'expected_longest_tokens':4673}
    harness_path=tmp_path/'h.json';harness_path.write_text(json.dumps(harness))
    out=tmp_path/'gpu';plan=gsq24.prepare(ready_path,harness_path,out)
    config=plan['config'];config['absolute_deadline']='deadline'
    args=gsq24.create_command(config,out)
    assert '--machine-type=g2-standard-32' in args and '--provisioning-model=SPOT' in args
    assert all('auto-delete=no' in arg for arg in args if arg.startswith('--disk='))
    (out/'gsq24_guest.py').write_text('changed')
    monkeypatch.setattr(gsq24,'run',lambda _:pytest.fail('Cloud call before integrity check'))
    with pytest.raises(ValueError,match='bootstrap'):gsq24.execute(out)


def test_gpu_progress_only_exposes_bounded_completed_workers(tmp_path,monkeypatch):
    from cloud_gpu import gsq24_guest as guest
    observed=[];monkeypatch.setattr(guest.common,'serial_summary',observed.append)
    guest.emit_progress(tmp_path,{'elapsed_seconds':1})
    assert observed[-1]=={'elapsed_seconds':1}
    (tmp_path/'results').mkdir();path=tmp_path/'results/gsq24-report.json'
    path.write_text(json.dumps({'status':'running','workers':[{'status':'passed','secret':'not emitted'}]}))
    guest.emit_progress(tmp_path,{'elapsed_seconds':2})
    assert observed[-1]=={'elapsed_seconds':2,'completed_validation_workers':1,'validation_status':'running'}
    path.write_text(json.dumps({'status':'passed','workers':[{'status':'passed'}]*7}))
    guest.emit_progress(tmp_path,{})
    assert observed[-1]=={}
    for value in (None,[],42):
        path.write_text(json.dumps(value));guest.emit_progress(tmp_path,{})
        assert observed[-1]=={}
