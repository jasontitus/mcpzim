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


def test_block_objective_target_is_the_unquantized_block(tmp_path):
    # The fix that mattered most tonight: the target must be the *original* block applied
    # to the same input the student receives, not the output of a separate teacher stream.
    # Upstream's own form is `nn.MSELoss()(out_q, out_fp)` with both forwards on the same
    # batch (cuda_runtime/.context/sources/gsq/src/models/base.py:444-453). Two things are
    # pinned: the port's loss equals that expression computed independently here, and the
    # target does not move when the quantized weights do - the leak counterfactual an
    # adversarial review used to establish the same claim (it measured exactly 0.0).
    from solver.gsq import BlockTrainer
    from solver.qwen import block_kwargs
    model=new_model()
    trainer=BlockTrainer(model,0)
    student=torch.randn(1,4,128)
    block=model.model.layers[0]

    torch.manual_seed(7)
    loss=trainer.forward(student)

    torch.manual_seed(7)                       # same noise draw, same order, same args
    replacements={name+'.weight':quantizer(1.,1.).to(student.dtype)
                  for name,quantizer in zip(trainer.names,trainer.quantizers)}
    with torch.no_grad():
        target=block(student,**block_kwargs(model,student,0))
        output=torch.func.functional_call(block,replacements,(student,),
                                          block_kwargs(model,student,0),strict=False)
    expected=(output.float()-target.float()).square().mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    # The target is independent of the replacements: recomputing it with the quantized
    # weights installed changes the *output* side only.
    with torch.no_grad():
        leaked=torch.func.functional_call(block,replacements,(student,),
                                          block_kwargs(model,student,0),strict=False)
        assert not torch.equal(leaked,target)


def test_rco_step_averages_gradients_over_gumbel_samples(tmp_path):
    # Upstream runs n_gumbel_samples evaluations per update and *averages their
    # gradients* into one projected step (`rco/src/search/quant.py:669-728`), with
    # each sample drawing its own noise and its own budget-constrained assignment.
    # It is not four sequential updates and not an averaged loss, and the
    # distinction is invisible in the reported loss - so it is pinned numerically
    # here. An adversarial review measured this exact contract by hand and named it
    # the most-deserved test of the change.
    from solver.rco import RCOTrainer
    model=new_model();full=tmp_path/'alloc';full.mkdir();initialize_candidates(model,full)
    database=read_database(full/'initial-database.json')
    count=sum(m.weight.numel() for m in projection_modules(model).values())
    trainer=RCOTrainer(model,database,int(count*8/8))
    ids=torch.tensor([[1,3,7,9]])          # step() requires one 2-D invocation [1, T]
    optimizer=torch.optim.Adam(trainer.parameters(),lr=0.0)

    def logits_state():
        return trainer.alpha.detach().clone()

    # All four draws must be taken at the same alpha, which is what the
    # multi-sample call does before it updates anything, so each single-sample
    # replay is restored to that point first. lr=0 keeps Adam from moving alpha;
    # the retraction still shifts it along the cost vector, hence the restore.
    start=logits_state()
    torch.manual_seed(4321)
    trainer.step(ids,optimizer,temperature=.5,n_gumbel_samples=4)
    averaged=trainer.alpha.grad.detach().clone()

    torch.manual_seed(4321)
    singles=[]
    for _ in range(4):
        with torch.no_grad():
            trainer.alpha.copy_(start)
        trainer.step(ids,optimizer,temperature=.5,n_gumbel_samples=1)
        singles.append(trainer.alpha.grad.detach().clone())
    mean=torch.stack(singles).mean(0)
    # Measured on this machine: greatest absolute deviation 1.16e-10, i.e. float32
    # rounding between the accumulate path and the stack path. The contract being
    # pinned is the *semantic* one below, where the sum differs by a factor of four.
    torch.testing.assert_close(averaged,mean,rtol=1e-5,atol=1e-8)
    # And it is the mean, not the sum.
    assert not torch.allclose(averaged,torch.stack(singles).sum(0))


def test_rco_retraction_holds_the_budget_and_refuses_a_silent_uniform_answer(tmp_path):
    # Two properties RCO is defined on. The retraction must put the *expected*
    # cost on the budget (upstream: to machine precision), which is what makes the
    # constraint exact rather than a penalty approximation. And a uniform Q1 answer
    # above the feasibility floor must never be accepted silently: measured on this
    # port, target_bytes=4.0e9 produced 0 bf16 groups while 4.15e9 produced 96+,
    # and the failure was quiet - a model whose budget was never spent would have
    # shipped looking like a considered allocation. An *untrained* allocator
    # starting on Q1 is expected (the budget is spent as training moves alpha), so
    # the uniform case is simulated here rather than assumed.
    from solver.rco import RCOTrainer
    import pytest
    model=new_model();full=tmp_path/'alloc';full.mkdir();initialize_candidates(model,full)
    database=read_database(full/'initial-database.json')
    count=sum(m.weight.numel() for m in projection_modules(model).values())
    ample=int(count*8/8)                       # 8 bits/param: room for bf16 groups
    trainer=RCOTrainer(model,database,ample)
    probabilities=torch.softmax(trainer.alpha.detach(),dim=-1)
    expected=(trainer.weights*(probabilities*trainer.costs).sum(-1)).sum().item()
    assert abs(expected-trainer.target)<1e-5

    # The knapsack's 1/500-bit cost grid puts its all-Q1 floor *above* the true all-Q1
    # byte count (498 groups of ~0.002 weight each round up), so the exact all-Q1 budget
    # is infeasible for the solver and must be refused at construction: below that floor
    # the unwritten sentinel dp returns uniform Q1 for any logits, and `assignment` runs
    # once per Gumbel sample during training - so a stage would train against an
    # allocation the solver never chose.
    all_q1=sum(trainer.byte_costs[name][0] for name in trainer.names)
    with pytest.raises(ValueError,match='knapsack is infeasible'):
        RCOTrainer(model,database,all_q1)

    trainer.budget_constrained_argmax=lambda *_: torch.zeros(len(trainer.names),
        dtype=torch.long,device=trainer.weights.device)
    with pytest.raises(ValueError,match='uniform Q1 allocation'):
        trainer.hard_allocation(ample)


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


def test_per_block_sweep_matches_single_process(tmp_path):
    # A block-boundary anchor has to carry every completed block's candidate
    # archive, because a resumed process rebuilds `database` from the archives a
    # snapshot holds and the block loop re-exports only the block it trains. When
    # the superseded archives were dropped, a `gsq_max_blocks=1` sweep left every
    # earlier block at its untrained RTN initial candidate while still reporting
    # `completed`: measured before the fix on this model, 8 of layer 0's modules
    # differed from the single-process reference and only the last process's own
    # block matched.
    records=[[1,3,7,9],[2,5,8,11]]
    full=tmp_path/'full';full.mkdir();initialize_candidates(new_model(),full)
    reference=config(tmp_path)
    expected=read_database(gsq_run(new_model(),records,reference,full,Checkpointer(reference,full))['candidate_database'])
    previous_store=previous_snapshot=None
    for index in range(2):
        output=tmp_path/f'per-block-{index}';output.mkdir()
        store=tmp_path/f'per-block-store-{index}'
        conf={**config(tmp_path),'checkpoint':{'backend':'local','path':str(store)},'gsq_max_blocks':1}
        if previous_store is not None:
            conf['resume_checkpoint']={'backend':'local','path':str(previous_store)}
        (output/'frozen-config.json').write_text(json.dumps({**conf,'output':str(output)}))
        model=new_model()
        if previous_store is None:
            initialize_candidates(model,output)
        checkpointer=Checkpointer(conf,output)
        result=gsq_run(model,records,conf,output,checkpointer,
                       checkpointer.restore(previous_snapshot) if previous_snapshot else None)
        assert result['status']==('blocked_stop' if index==0 else 'completed')
        if index==0:
            assert result['next_block']==1
        previous_store=store
        previous_snapshot=json.loads((output/'latest-checkpoint.json').read_text())['snapshot']
    actual=read_database(result['candidate_database'])
    for name in expected:
        a,b=Q1Candidate(expected[name]),Q1Candidate(actual[name])
        torch.testing.assert_close(a.rows(0,a.shape[0],'cpu',torch.float32),
            b.rows(0,b.shape[0],'cpu',torch.float32),atol=0,rtol=0)


def test_resume_from_a_stop_on_a_blocks_final_update(tmp_path):
    # `max_steps` covering exactly one block's work stops after that block's LAST
    # update, so the snapshot carries epoch=0 with sequence==len(records): the
    # epoch loop cannot advance it and only the export remains. The zero-step guard
    # used to reject that state, and since every later resume of the same snapshot
    # fails identically, a sweep whose stop landed there was wedged.
    records=[[1,3,7,9],[2,5,8,11]]
    full=tmp_path/'full';full.mkdir();initialize_candidates(new_model(),full)
    reference=config(tmp_path)
    expected=read_database(gsq_run(new_model(),records,reference,full,Checkpointer(reference,full))['candidate_database'])
    partial=tmp_path/'partial';partial.mkdir();initialize_candidates(new_model(),partial)
    pconf={**config(tmp_path),'checkpoint':{'backend':'local','path':str(tmp_path/'partial-durable')}}
    stopped=gsq_run(new_model(),records,pconf,partial,Checkpointer(pconf,partial),max_steps=len(records))
    assert stopped['status']=='checkpointed_stop'
    assert stopped['progress']['epoch']==0 and stopped['progress']['sequence']==len(records)
    snapshot=json.loads((partial/'latest-checkpoint.json').read_text())['snapshot']
    output=tmp_path/'resumed';output.mkdir()
    resume_conf={**pconf,'resume_checkpoint':pconf['checkpoint'],
                 'checkpoint':{'backend':'local','path':str(tmp_path/'resume-durable')}}
    (output/'frozen-config.json').write_text(json.dumps({**resume_conf,'output':str(output)}))
    checkpointer=Checkpointer(resume_conf,output)
    resumed=gsq_run(new_model(),records,resume_conf,output,checkpointer,checkpointer.restore(snapshot))
    assert resumed['status']=='completed'
    actual=read_database(resumed['candidate_database'])
    for name in expected:
        a,b=Q1Candidate(expected[name]),Q1Candidate(actual[name])
        torch.testing.assert_close(a.rows(0,a.shape[0],'cpu',torch.float32),
            b.rows(0,b.shape[0],'cpu',torch.float32),atol=0,rtol=0)


def test_resume_refuses_a_snapshot_missing_a_completed_blocks_archive(tmp_path):
    # `initial_candidates` supplies a valid RTN candidate for every module NAME, so
    # a snapshot that has lost a completed block's archive still yields a
    # complete-looking database: without this check the block is written out as
    # calibrated while holding its untrained initial candidates and the run reports
    # `completed`. That is the shape of every anchor written before the retention
    # fix, which is what a per-block sweep would hand to the next process.
    import pytest
    records=[[1,3,7,9],[2,5,8,11]]
    conf={**config(tmp_path),'gsq_max_blocks':1}
    first=tmp_path/'first';first.mkdir();initialize_candidates(new_model(),first)
    stopped=gsq_run(new_model(),records,conf,first,Checkpointer(conf,first))
    assert stopped['status']=='blocked_stop'
    snapshot=json.loads((first/'latest-checkpoint.json').read_text())['snapshot']
    output=tmp_path/'stripped';output.mkdir()
    resume_conf={**conf,'resume_checkpoint':conf['checkpoint'],
                 'checkpoint':{'backend':'local','path':str(tmp_path/'stripped-store')}}
    (output/'frozen-config.json').write_text(json.dumps({**resume_conf,'output':str(output)}))
    checkpointer=Checkpointer(resume_conf,output)
    restored=checkpointer.restore(snapshot)
    (restored/'candidate_block_0').unlink()
    with pytest.raises(ValueError,match='carries no candidate archive'):
        gsq_run(new_model(),records,resume_conf,output,checkpointer,resume=restored)


def test_held_out_invocations_are_reserved_from_training(tmp_path):
    # The split this port's own documentation promised and the code did not have:
    # eight of the 87 corpus invocations are the text the PPL bar and every arm are
    # measured on, so training on them makes a trained arm's number optimistic
    # while the bar's is not. Reserving them is what makes the comparison mean
    # anything.
    from solver.run import held_out_invocations, is_held_out_invocation
    assert is_held_out_invocation('invocation-000000', 11)
    assert is_held_out_invocation('invocation-000011', 11)
    assert not is_held_out_invocation('invocation-000001', 11)
    assert not is_held_out_invocation('invocation-000010', 11)
    assert not is_held_out_invocation('invocation-000000', None)
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'manifest.json').write_text(json.dumps(
        {'sequences':[{'directory':f'invocation-{n:06d}'} for n in range(22)]}))
    assert held_out_invocations(corpus,11)==['invocation-000000','invocation-000011']


def test_block_loop_records_stream_drift_and_stops_when_it_grows(tmp_path):
    # `gsq_max_drift_growth` guards the failure this port actually has: the composed
    # student stream inflating block over block until the model is no better than
    # untrained RTN. Measured on the discarded run: 2.94x with cosine 0.934 after a
    # single quantized block, and an incoming loss that roughly doubled per block to
    # 60,800 by block 8. A threshold at machine epsilon must stop the chain at the
    # first block boundary that has a previous ratio to compare against, and the
    # record must be on disk for whoever reads the run next.
    # Four layers: the drift stop needs a block AFTER the one whose growth it
    # measures, and it deliberately does not fire on the final block, where there
    # is no next block to anchor a resume from.
    records=[[1,3,7,9],[2,5,8,11]]
    conf={**config(tmp_path),'gsq_max_blocks':3,'gsq_max_drift_growth':1e-12}
    output=tmp_path/'drift';output.mkdir();initialize_candidates(tiny_model(4),output)
    # The status must be `drift_stop`, not `blocked_stop`: the per-block driver
    # advances on `blocked_stop` and treats any other status as a loud stop, so a
    # guard trip reported as `blocked_stop` would be walked past. The threshold here
    # is machine epsilon, so this asserts the guard fires rather than any real growth.
    stopped=gsq_run(tiny_model(4),records,conf,output,Checkpointer(conf,output))
    assert stopped['status']=='drift_stop' and stopped['stop_reason']=='drift'
    history=json.loads((output/'drift.json').read_text())
    assert [record['block'] for record in history]==[0,1]
    first,second=history
    assert first['previous_norm_ratio'] is None and first['growth'] is None
    for record in history:
        assert record['samples']>=1
        assert record['norm_ratio']>0
        # A cosine, so it ranges over [-1, 1]: on a randomly initialised tiny model
        # it sits near zero, and only a trained model shows the strong agreement
        # (0.934) that the discarded run measured.
        assert -1<=record['cosine']<=1
    # The second block is the first one with a predecessor to compare against, and
    # its recorded growth is what tripped the stop.
    assert second['previous_norm_ratio']==first['norm_ratio']
    assert second['growth']>1e-12
    assert stopped['drift']['block']==1


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
