"""CPU math + real immutable local/GCS bridge protocol; no CUDA/Docker/cloud.

MemoryBackend replaces only GCS object I/O. Container execution is injected;
models, optimizers, checkpoints, cloud restore, controller and re-preparation run.
Set PYTHONPATH to tools/calibration and the pinned RCO checkout's src directory.
"""
import copy
import json
from pathlib import Path
import pytest

import continuation as controller
from checkpoint_bridge import CheckpointBridge, read_latest
from gcs_checkpoints import GCSCheckpointStore
from test_gcs_checkpoints import MemoryBackend

RUNTIME='a'*64
IMAGE='us-central1-docker.pkg.dev/project/repo/runtime@sha256:'+RUNTIME


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2)+'\n').encode();path.write_bytes(data);return data


@pytest.mark.parametrize('stop_after_gsq',[False,True])
def test_real_cpu_local_spool_pipeline_and_cold_continuation(tmp_path,stop_after_gsq):
    pytest.importorskip('manifold',reason='Run with pinned RCO src on PYTHONPATH')
    import torch
    from solver.qwen import tiny_model, projection_modules
    from solver.run import (initialize_candidates, boundary_run, gsq_run, rco_run,
                            Checkpointer, bind_identity, digest)
    torch.set_num_threads(1)
    records=[[1,3,7,9]]
    def model():
        torch.manual_seed(21);return tiny_model()
    backend=MemoryBackend()
    root=tmp_path/'retained';root.mkdir()
    inputs=tmp_path/'inputs';inputs.mkdir()
    corpus=inputs/'calibration';corpus.mkdir()
    baseline_manifest={'model':controller.MODEL,'revision':controller.REVISION}
    corpus_sha=controller.digest(write(corpus/'manifest.json',baseline_manifest))
    initial=tmp_path/'initial';initial.mkdir();initialize_candidates(model(),initial)
    tensors={name+'.weight':{'shape':list(layer.weight.shape),
               'q1_bytes':layer.weight.numel()//128*18,'bf16_bytes':layer.weight.numel()*2}
               for name,layer in projection_modules(model()).items()}
    cost_path=tmp_path/'costs.json';write(cost_path,{'fixed_container_bytes':0,'tensors':tensors})
    local=root/'spool'/'embedding'
    config={'stage':'embedding','runtime_sha256':RUNTIME,'input_commit_sha256':'2'*64,
            'corpus':str(corpus),'checkpoint_seconds':120,
            'checkpoint':{'backend':'local','path':str(local)},
            'candidate_database':str(initial/'initial-database.json'),
            'candidate_archives':{'initial_candidates':str(initial/'initial-candidates.tar')},
            'cost_manifest':{'path':str(cost_path),'sha256':digest(cost_path)},
            'target_bytes':sum(v['q1_bytes'] for v in tensors.values())*2}
    bind_identity(config,baseline_manifest)
    directory=root/'03-embedding';directory.mkdir()
    write(directory/'frozen-config.json',config)
    checkpointer=Checkpointer(config,directory)
    stopped=boundary_run('embedding',model(),records,config,directory,checkpointer,max_steps=1)
    assert stopped['status']=='checkpointed_stop'
    target={'backend':'gcs','bucket':'test-bucket','prefix':'runs/source/production'}
    published=CheckpointBridge(local,root/'cloud'/'embedding',target,stage='embedding',backend=backend).publish_latest()
    assert published['status']=='published'
    status={'runtime_sha256':RUNTIME,'input_commit_sha256':'2'*64,'status':'checkpointed',
            'checkpoint_mode':'local-spool','smoke':{'passed':True},'stages':[],
            'progress':{'phase':'embedding','durable_checkpoint':published['checkpoint']}}
    write(root/'status.json',status)
    plan=tmp_path/'plan'
    controller.prepare(root,plan,IMAGE,inputs,tmp_path/'unused-recovery')
    restored_by_stage={};reports=[]
    def runner(spec,deadline):
        output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
        stage=spec['stage']
        if spec['action']=='prepare':
            argv=spec['argv'];source_status=json.loads(Path(argv[argv.index('--status')+1]).read_bytes())
            source_checkpoint=source_status['progress']['durable_checkpoint']
            source_config=json.loads(Path(source_status['progress']['resume']['config']).read_bytes())
            source=source_config['checkpoint']
            cloud=GCSCheckpointStore(source['bucket'],source['prefix'],backend=backend)
            restored=output/'restored'
            actual=cloud.restore(source_checkpoint['snapshot'],source_config['identity'],restored,
                                commit_generation=source_checkpoint['receipt']['commit']['generation'])
            assert actual==source_checkpoint['receipt']
            assert json.loads((restored/'configuration').read_bytes())==source_config
            recovered=copy.deepcopy(source_config);recovered['resume_checkpoint']=source.copy()
            write(output/'resume-config.json',recovered)
            write(output/'recovery-receipt.json',{'status':'verified_stage_recovery_prepared','stage':stage,
                'source_commit':actual['commit'],'identity':source_config['identity'],
                'configuration':str(output/'resume-config.json')})
            restored_by_stage[stage]=restored
            return
        actual_config=json.loads(Path(spec['config']).read_bytes())
        bind_identity(actual_config,baseline_manifest)
        write(output/'frozen-config.json',actual_config)
        cp=Checkpointer(actual_config,output)
        resume=restored_by_stage.pop(stage,None) if '--resume' in spec['argv'] else None
        if stage in ('embedding','head'):
            report=boundary_run(stage,model(),records,actual_config,output,cp,resume=resume)
        else:
            report=(gsq_run if stage=='gsq' else rco_run)(model(),records,actual_config,output,cp,resume=resume)
        report.update(runtime_sha256=RUNTIME,calibration_sha256=corpus_sha)
        reports.append((stage,report))
        write(output/(stage+'-report.json'),report)
        publisher_argv=spec['publisher']['argv']
        receipt_root=publisher_argv[publisher_argv.index('--receipt-root')+1]
        cloud_target=json.loads(Path(publisher_argv[publisher_argv.index('--target-json')+1]).read_bytes())
        bridge=CheckpointBridge(actual_config['checkpoint']['path'],receipt_root,cloud_target,stage=stage,backend=backend)
        result=bridge.publish_latest()
        local_latest=json.loads((output/'latest-checkpoint.json').read_bytes())
        assert result['checkpoint']['snapshot']==local_latest['snapshot']
        assert result['configuration']['identity']==actual_config['identity']
        write(spec['producer_done'],{'producer_returncode':0})
    publisher=Path(__file__).resolve().parents[1]/'checkpoint_bridge.py'
    options={'checkpoint_mode':'local-spool','publisher_script':publisher,
             'publisher_sha256':controller.digest(publisher.read_bytes()),'spool_min_free_bytes':1}
    continued=tmp_path/'continued'
    first_attempt=continued
    result=controller.continue_plan(plan,continued,tmp_path,inputs,runner=runner,
        stop_after_stage='gsq' if stop_after_gsq else None,**options)
    if stop_after_gsq:
        assert result['status']=='checkpointed' and result['stop_reason']=='requested_stage_boundary'
        assert [stage for stage,_ in reports]==['embedding','gsq']
        assert not (continued/'head-config.json').exists() and not (continued/'rco-config.json').exists()
        handoff_plan=tmp_path/'handoff-plan'
        handoff=controller.prepare(continued,handoff_plan,IMAGE,inputs,tmp_path/'unused-handoff-recovery')
        assert handoff['selected_stage']=='gsq' and handoff['resume_terminal_stage_even_if_no_new_updates'] is True
        continued=tmp_path/'after-handoff'
        result=controller.continue_plan(handoff_plan,continued,tmp_path,inputs,runner=runner,**options)
        assert [stage for stage,_ in reports]==['embedding','gsq','gsq','head','rco']
        assert reports[2][1]['optimizer_updates']==0
        assert 'stop_reason' not in result and 'stage_handoff' not in result
    else:
        assert [stage for stage,_ in reports]==['embedding','gsq','head','rco']
    assert result['status']=='quantization_stages_completed'
    assert reports[0][1]['updates']==1
    assert reports[1][1]['optimizer_updates']==2
    assert reports[-2][1]['updates']==1
    assert reports[-1][1]['optimizer_updates']==1
    for stage in controller.STAGES:
        owner=first_attempt if stop_after_gsq and stage=='embedding' else continued
        cloud=read_latest(owner/'cloud'/stage)
        assert cloud['configuration']['checkpoint']['backend']=='gcs'
        local_config=json.loads((owner/controller.DIRECTORIES[stage]/'frozen-config.json').read_bytes())
        assert local_config['checkpoint']['backend']=='local'
        assert cloud['configuration']['identity']==local_config['identity']
    second_plan=tmp_path/'second-plan'
    prepared=controller.prepare(continued,second_plan,IMAGE,inputs,tmp_path/'unused-second-recovery')
    assert prepared['selected_stage']=='rco'
    assert prepared['resume_terminal_stage_even_if_no_new_updates'] is True
    assert {x['stage'] for x in prepared['completed_boundary_reports']}=={'embedding','head'}
    again=controller.continue_plan(second_plan,tmp_path/'again',tmp_path,inputs,runner=runner,**options)
    assert again['status']=='quantization_stages_completed'
    assert reports[-1][0]=='rco' and reports[-1][1]['optimizer_updates']==0
    # The host reads receipts using only stdlib, without image SDK/model imports.
    import subprocess,sys
    command=('import runpy; ns=runpy.run_path('+repr(str(Path(controller.__file__).resolve()))+'); '
             'r=ns["read_cloud_publication"]('+repr(str(continued/'cloud'/'rco'))+'); '
             'assert r["checkpoint"]["receipt"]["commit"]["generation"]>0')
    subprocess.run([sys.executable,'-I','-S','-c',command],check=True,capture_output=True)

    def corrupted_source_receipt(spec,deadline):
        runner(spec,deadline)
        if spec['action']=='stage':
            path=Path(spec['output'])/'latest-checkpoint.json'
            local=json.loads(path.read_bytes());local['receipt']['payloads']['solver']['bytes']+=1
            write(path,local)
    with pytest.raises(ValueError,match='has not drained'):
        controller.continue_plan(second_plan,tmp_path/'corrupted',tmp_path,inputs,
                                 runner=corrupted_source_receipt,**options)
