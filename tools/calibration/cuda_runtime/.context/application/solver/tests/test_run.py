from pathlib import Path
import sys
import json
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from solver.qwen import tiny_model,projection_modules
from solver.run import initialize_candidates,gsq_run,rco_run,Checkpointer,read_database
from solver.candidates import Q1Candidate


def config(tmp_path):
    return {'checkpoint':{'backend':'local','path':str(tmp_path/'durable')},
            'identity':{'baseline_repo':'Qwen/Qwen3.8-27B','baseline_revision':'1'*40,
                        'calibration_sha256':'2'*64,'solver_revision':'3'*40,
                        'solver_config_sha256':'4'*64,'candidate_database_sha256':'5'*64,
                        'runtime_sha256':'6'*64},'checkpoint_seconds':120,'gsq_epochs':1,
            'rco_epochs':1,'upstream':'/opt/upstream','allow_rtn_boundary_smoke':True}


def new_model():
    torch.manual_seed(21)
    return tiny_model()


def test_actual_hybrid_gsq_interrupt_resume_and_rco(tmp_path):
    records=[[1,3,7,9],[2,5,8,11]]
    conf=config(tmp_path)
    full=tmp_path/'full';full.mkdir()
    model=new_model();initialize_candidates(model,full)
    torch.manual_seed(91)
    result=gsq_run(model,records,conf,full,Checkpointer(conf,full))
    assert result['status']=='completed' and result['final_quality_recipe_ready'] is False
    expected=read_database(full/'candidate-database.json')

    partial=tmp_path/'partial';partial.mkdir()
    # Separate durable root avoids deliberate immutable snapshot name collision
    # with the already completed uninterrupted oracle run.
    conf['checkpoint']['path']=str(tmp_path/'durable-resume')
    model=new_model();initialize_candidates(model,partial)
    torch.manual_seed(91)
    result=gsq_run(model,records,conf,partial,Checkpointer(conf,partial),max_steps=1)
    assert result['status']=='checkpointed_stop'
    snapshot=json.loads((partial/'latest-checkpoint.json').read_text())['snapshot']
    resumed=tmp_path/'resumed';resumed.mkdir()
    checkpointer=Checkpointer(conf,resumed)
    restored=checkpointer.restore(snapshot)
    model=new_model()
    result=gsq_run(model,records,conf,resumed,checkpointer,resume=restored)
    assert result['status']=='completed'
    actual=read_database(resumed/'candidate-database.json')
    for name in expected:
        a,b=Q1Candidate(expected[name]),Q1Candidate(actual[name])
        torch.testing.assert_close(a.rows(0,a.shape[0],'cpu',torch.float32),
                                   b.rows(0,b.shape[0],'cpu',torch.float32),atol=0,rtol=0)
    count=sum(m.weight.numel() for m in projection_modules(model).values())
    conf.update(candidate_database=str(resumed/'candidate-database.json'),
                target_tensor_bytes=int(count*4/8),fixed_tensor_bytes=0,
                candidate_archives={'initial_candidates':str(resumed/'initial_candidates.tar')})
    smoke=rco_run(model,records,conf,resumed,checkpointer,max_steps=1)
    assert smoke['status']=='checkpointed_stop'
    assert smoke['progress']['raw_gradient_norm']>0


def test_complete_learned_boundary_pipeline_and_cpu_smoke(tmp_path):
    from solver.run import boundary_run,digest
    from solver.smoke import smoke_run
    records=[[1,3,7,9],[2,5,8,11]]
    conf=config(tmp_path);model=new_model()
    initial=tmp_path/'initial';initial.mkdir();initialize_candidates(model,initial)
    count=sum(m.weight.numel() for m in projection_modules(model).values())
    conf.update(candidate_database=str(initial/'initial-database.json'),
                candidate_archives={'initial_candidates':str(initial/'initial-candidates.tar')},
                target_tensor_bytes=int(count*4/8),fixed_tensor_bytes=0)
    smoke=tmp_path/'smoke';smoke.mkdir()
    result=smoke_run(model,records,conf,smoke,Checkpointer(conf,smoke),allow_cpu_test=True)
    assert result['smoke']['passed'] is False
    assert result['smoke']['checkpoint_resume']['next_update_matches'] is True
    conf['warmstart_states']=result['warmstart_states']
    conf['warmstart_sha256']={k:digest(v) for k,v in result['warmstart_states'].items()}
    reports=[]
    for stage in ('embedding','gsq','head'):
        out=tmp_path/stage;out.mkdir()
        cp=Checkpointer(conf,out)
        if stage=='gsq':result=gsq_run(new_model(),records,conf,out,cp)
        else:result=boundary_run(stage,new_model(),records,conf,out,cp)
        assert result['status']=='completed'
        conf.update({k:result[k] for k in ('candidate_database','candidate_archives','final_cache') if k in result})
        if stage!='gsq':
            report=out/(stage+'-report.json');report.write_text(json.dumps(result))
            reports.append({'path':str(report),'sha256':digest(report)})
    conf['boundary_reports']=reports
    out=tmp_path/'rco';out.mkdir()
    result=rco_run(new_model(),records,conf,out,Checkpointer(conf,out),max_steps=1)
    assert result['progress']['raw_gradient_norm']>0
    snapshot=json.loads((out/'latest-checkpoint.json').read_text())['snapshot']
    resumed=tmp_path/'resumed-rco';resumed.mkdir();cp=Checkpointer(conf,resumed)
    restored=cp.restore(snapshot)
    for report in reports:Path(report['path']).unlink()
    result=rco_run(new_model(),records,conf,resumed,cp,resume=restored,max_steps=1)
    assert result['progress']['global_step']==2


def test_final_block_and_next_phase_checkpoint_resume(tmp_path):
    import pytest
    records=[[1,3,7,9]];conf=config(tmp_path)
    full=tmp_path/'full';full.mkdir();initialize_candidates(new_model(),full)
    (full/'frozen-config.json').write_text(json.dumps({**conf,'output':str(full)}))
    result=gsq_run(new_model(),records,conf,full,Checkpointer(conf,full))
    expected=read_database(result['candidate_database'])
    from checkpoints import LocalCheckpointStore
    snapshots=LocalCheckpointStore(tmp_path/'durable').committed_snapshots()
    selected=[p for p in snapshots if ('b000' in p and '-e001-q00000' in p)
              or ('b001' in p and '-e000-q00000' in p)
              or ('b001' in p and '-e001-q00000' in p)]
    assert len(selected)==3
    for index,snapshot in enumerate(selected):
        output=tmp_path/f'restored-{index}';output.mkdir()
        resume_conf={**conf,'resume_checkpoint':conf['checkpoint'],
                     'checkpoint':{'backend':'local','path':str(tmp_path/f'new-durable-{index}')}}
        (output/'frozen-config.json').write_text(json.dumps({**resume_conf,'output':str(output)}))
        cp=Checkpointer(resume_conf,output);restored=cp.restore(snapshot)
        actual_result=gsq_run(new_model(),records,resume_conf,output,cp,resume=restored)
        actual=read_database(actual_result['candidate_database'])
        for name in expected:
            a,b=Q1Candidate(expected[name]),Q1Candidate(actual[name])
            torch.testing.assert_close(a.rows(0,a.shape[0],'cpu',torch.float32),
                b.rows(0,b.shape[0],'cpu',torch.float32),atol=0,rtol=0)


def test_candidate_archive_canonical_across_file_permissions(tmp_path):
    from solver.run import archive_directory,digest
    source=tmp_path/'source';source.mkdir();payload=source/'weights';payload.write_bytes(b'weights')
    payload.chmod(0o600);first=archive_directory(source,tmp_path/'first.tar')
    payload.chmod(0o644);second=archive_directory(source,tmp_path/'second.tar')
    assert digest(first)==digest(second)


def test_smoke_persists_pre_update_restore_failure(tmp_path, monkeypatch):
    import pytest
    from solver import smoke as module
    model=new_model();conf=config(tmp_path)
    initial=tmp_path/'initial';initial.mkdir();initialize_candidates(model,initial)
    count=sum(m.weight.numel() for m in projection_modules(model).values())
    conf.update(candidate_database=str(initial/'initial-database.json'),
                candidate_archives={'initial_candidates':str(initial/'initial-candidates.tar')},
                target_tensor_bytes=int(count*4/8),fixed_tensor_bytes=0,smoke_component='rco')
    original=module.restore_state
    def corrupt(directory,trainer,optimizer,scheduler):
        original(directory,trainer,optimizer,scheduler)
        with torch.no_grad():trainer.alpha.add_(.1)
    monkeypatch.setattr(module,'restore_state',corrupt)
    output=tmp_path/'smoke';output.mkdir()
    with pytest.raises(AssertionError):
        module.smoke_run(model,[[1,3,7,9]],conf,output,Checkpointer(conf,output),allow_cpu_test=True)
    report=json.loads((output/'recovery-report.json').read_text())
    assert report['status']=='failed'
    assert report['checks']['publication_preserves_rng'] is True
    assert 'restored_solver_exact' not in report['checks']
    assert 'uninterrupted_update' in report and 'replayed_update' not in report
    assert report['cuda']['available'] is False
    assert not (output/'smoke-report.json').exists()
