from pathlib import Path
import copy
import json
import weakref

import pytest
import torch

from solver.gsq_residency import GSQResidency, memory_mode, execution_device, BLOCK_CPU_OFFLOAD
from solver.qwen import tiny_model
from solver.run import gsq_run, initialize_candidates, cached_inputs, bind_identity, digest


def model():
    torch.manual_seed(73)
    return tiny_model()


def config(mode='full',device='cpu'):
    return {'stage':'gsq','gsq_memory_mode':mode,'gsq_execution_device':device}


class Recorder:
    def __init__(self,fail=False):self.states=[];self.references=[];self.fail=fail
    def save(self,trainer,optimizer,scheduler,progress,extras,force=False):
        self.references.append((weakref.ref(trainer),weakref.ref(optimizer)))
        if force:
            self.states.append({'solver':copy.deepcopy(trainer.state_dict()),
                                'optimizer':copy.deepcopy(optimizer.state_dict()),
                                'scheduler':copy.deepcopy(scheduler.state_dict()),'progress':copy.deepcopy(progress)})
        if self.fail and len(self.references)==int(self.fail):raise RuntimeError('Injected checkpoint failure')


def assert_same(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b


def test_full_and_block_residency_have_identical_cpu_math_and_outputs(tmp_path):
    torch.set_num_threads(1)
    initial=tmp_path/'initial';initial.mkdir();initialize_candidates(model(),initial)
    results=[];recorders=[]
    for mode in ('full',BLOCK_CPU_OFFLOAD):
        baseline=model();out=tmp_path/mode;out.mkdir();recorder=Recorder()
        conf={**config(mode),'candidate_database':str(initial/'initial-database.json'),
              'candidate_archives':{'initial_candidates':str(initial/'initial-candidates.tar')}}
        torch.manual_seed(901)
        result=gsq_run(baseline,[[1,3,7,9],[2,5,6]],conf,out,recorder)
        assert result['status']=='completed' and result['optimizer_updates']==4
        assert all(t.device.type=='cpu' for t in list(baseline.parameters())+list(baseline.buffers()))
        assert all(p.grad is None and not p.requires_grad for p in baseline.parameters())
        results.append(result);recorders.append(recorder)
    assert_same(recorders[0].states,recorders[1].states)
    left=json.loads(Path(results[0]['candidate_database']).read_text())
    right=json.loads(Path(results[1]['candidate_database']).read_text())
    assert left.keys()==right.keys()
    from safetensors import safe_open
    for name in left:
        with safe_open(left[name],framework='pt') as a, safe_open(right[name],framework='pt') as b:
            assert a.metadata()==b.metadata()
            assert_same({k:a.get_tensor(k) for k in a.keys()},{k:b.get_tensor(k) for k in b.keys()})
    for index in range(2):
        a=torch.load(Path(results[0]['final_cache'])/f'{index:06d}.pt',weights_only=True)
        b=torch.load(Path(results[1]['final_cache'])/f'{index:06d}.pt',weights_only=True)
        assert_same(a,b)
    assert all(ref() is None for recorder in recorders for pair in recorder.references for ref in pair)


@pytest.mark.parametrize('mode',['wrong','cpu_offload',None])
def test_invalid_modes_fail(mode):
    with pytest.raises(ValueError,match='Unknown GSQ memory'):GSQResidency(model(),config(mode))


def test_offload_requires_explicit_device_and_is_only_for_gsq():
    with pytest.raises(ValueError,match='explicit'):memory_mode({'gsq_memory_mode':BLOCK_CPU_OFFLOAD})
    with pytest.raises(ValueError,match='only for GSQ'):memory_mode({**config(BLOCK_CPU_OFFLOAD),'stage':'rco'})


@pytest.mark.parametrize('device',['mps','meta','cpu:1','not-a-device'])
def test_unsupported_devices_fail(device):
    with pytest.raises(ValueError):GSQResidency(model(),config(BLOCK_CPU_OFFLOAD,device))


def test_unavailable_cuda_rejected_without_moving_weights(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='unavailable'):execution_device(config(BLOCK_CPU_OFFLOAD,'cuda:0'),'cpu')


def test_unmaterialized_or_unfrozen_baseline_rejected():
    unfrozen=model();next(unfrozen.parameters()).requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):GSQResidency(unfrozen,config(BLOCK_CPU_OFFLOAD))
    baseline=model();next(baseline.parameters()).grad=torch.ones_like(next(baseline.parameters()))
    with pytest.raises(ValueError,match='no parameter gradients'):GSQResidency(baseline,config(BLOCK_CPU_OFFLOAD))
    baseline=model();baseline.model.layers[0].to('meta')
    with pytest.raises(ValueError,match='meta'):GSQResidency(baseline,config(BLOCK_CPU_OFFLOAD))


def test_active_scope_exclusive_and_unwinds_after_failure():
    baseline=model();resident=GSQResidency(baseline,config(BLOCK_CPU_OFFLOAD))
    with pytest.raises(RuntimeError,match='injected'):
        with resident.block(0):
            assert resident.active==0
            with pytest.raises(ValueError,match='one GSQ block'):
                with resident.block(1):pass
            raise RuntimeError('injected')
    assert resident.active is None
    resident.assert_layout()
    with resident.block(1):assert resident.active==1
    for index in (-1,2,True,'1'):
        with pytest.raises(ValueError,match='index'):
            with resident.block(index):pass


@pytest.mark.parametrize('failure_ordinal',[1,2])
def test_checkpoint_failure_clears_training_state_even_when_traceback_retained(tmp_path,failure_ordinal):
    from solver.gsq import BlockTrainer
    baseline=model();initial=tmp_path/'initial';initial.mkdir();initialize_candidates(baseline,initial)
    out=tmp_path/'out';out.mkdir();recorder=Recorder(fail=failure_ordinal)
    conf={**config(BLOCK_CPU_OFFLOAD),'candidate_database':str(initial/'initial-database.json'),
          'candidate_archives':{'initial_candidates':str(initial/'initial-candidates.tar')}}
    with pytest.raises(RuntimeError,match='Injected checkpoint') as failure:
        gsq_run(baseline,[[1,3,7,9]],conf,out,recorder)
    # Traceback can retain the trainer through failing recorder.save frame.
    for trainer_ref,optimizer_ref in recorder.references:
        trainer,optimizer=trainer_ref(),optimizer_ref()
        if trainer is not None:
            assert all(p.grad is None and p.device.type=='cpu' for p in trainer.parameters())
        if optimizer is not None:assert not optimizer.state
    assert failure.value.__traceback__ is not None
    assert all(p.device.type=='cpu' and p.grad is None for p in baseline.parameters())


def test_early_stop_releases_optimizer_and_keeps_active_checkpoint(tmp_path):
    baseline=model();initial=tmp_path/'initial';initial.mkdir();initialize_candidates(baseline,initial)
    out=tmp_path/'out';out.mkdir();recorder=Recorder()
    conf={**config(BLOCK_CPU_OFFLOAD),'candidate_database':str(initial/'initial-database.json'),
          'candidate_archives':{'initial_candidates':str(initial/'initial-candidates.tar')}}
    result=gsq_run(baseline,[[1,3,7,9],[2,5,6]],conf,out,recorder,max_steps=1)
    assert result['status']=='checkpointed_stop' and result['progress']['global_step']==1
    assert recorder.states[-1]['optimizer']['state']
    assert all(ref() is None for pair in recorder.references for ref in pair)
    assert all(p.grad is None and p.device.type=='cpu' for p in baseline.parameters())


def test_offload_activates_only_block_and_rotary_never_embeddings(monkeypatch):
    baseline=model();resident=GSQResidency(baseline,config(BLOCK_CPU_OFFLOAD));calls=[]
    for name,module in [('block0',baseline.model.layers[0]),('block1',baseline.model.layers[1]),
                        ('rotary',baseline.model.rotary_emb),('embedding',baseline.model.embed_tokens),
                        ('head',baseline.lm_head),('norm',baseline.model.norm)]:
        original=module.to
        def move(*args,_name=name,_original=original,**kwargs):calls.append(_name);return _original(*args,**kwargs)
        monkeypatch.setattr(module,'to',move)
    with resident.block(0):pass
    assert calls==['block0','rotary','block0','rotary']


def test_memory_mode_and_device_are_checkpoint_identity_fields(tmp_path):
    corpus=tmp_path/'corpus';corpus.mkdir();(corpus/'manifest.json').write_text('{}')
    base={'stage':'gsq','corpus':str(corpus),'runtime_sha256':'a'*64}
    manifest={'model':'Qwen/Qwen3.8-27B','revision':'b'*40}
    full={**base,**config()};offload={**base,**config(BLOCK_CPU_OFFLOAD)}
    first=bind_identity(full,manifest);second=bind_identity(offload,manifest)
    assert first['solver_config_sha256']!=second['solver_config_sha256']
    with pytest.raises(ValueError,match='differs from frozen'):
        bind_identity({**offload,'gsq_execution_device':'cuda:0'},manifest)
    with pytest.raises(ValueError,match='differs from frozen'):
        bind_identity({**full,'gsq_memory_mode':BLOCK_CPU_OFFLOAD},manifest)


def test_cache_requires_explicit_matching_embedding_device(tmp_path):
    with pytest.raises(ValueError,match='Embedding cache device'):
        cached_inputs(model(),[[1,2]],tmp_path/'cache',None,device='meta')


def test_cpu_offload_stop_restore_matches_uninterrupted_training(tmp_path):
    from solver.run import Checkpointer
    from safetensors import safe_open
    initial=tmp_path/'initial';initial.mkdir();initialize_candidates(model(),initial)
    corpus=tmp_path/'corpus';corpus.mkdir();(corpus/'manifest.json').write_text('{}')
    baseline_manifest={'model':'Qwen/Qwen3.8-27B','revision':'b'*40}
    base={**config(BLOCK_CPU_OFFLOAD),'runtime_sha256':'a'*64,'corpus':str(corpus),
          'candidate_database':str(initial/'initial-database.json'),
          'candidate_archives':{'initial_candidates':str(initial/'initial-candidates.tar')}}
    records=[[1,3,7,9],[2,5,6]]
    def setup(name):
        out=tmp_path/name;out.mkdir()
        conf={**base,'checkpoint':{'backend':'local','path':str(out/'store')}}
        bind_identity(conf,baseline_manifest)
        return out,conf,Checkpointer(conf,out)
    out,conf,cp=setup('full')
    baseline=model();torch.manual_seed(901)
    reference=gsq_run(baseline,records,conf,out,cp)
    partial,pconfig,pcp=setup('partial')
    baseline=model();torch.manual_seed(901)
    stopped=gsq_run(baseline,records,pconfig,partial,pcp,max_steps=1)
    assert stopped['progress']['global_step']==1
    snapshot=json.loads((partial/'latest-checkpoint.json').read_text())['snapshot']
    continued,cconfig,ccp=setup('continued')
    cconfig['resume_checkpoint']=pconfig['checkpoint'].copy()
    assert cconfig['identity']==pconfig['identity']
    restored=ccp.restore(snapshot)
    resumed=gsq_run(model(),records,cconfig,continued,ccp,resume=restored)
    assert resumed['optimizer_updates']==3
    left=json.loads(Path(reference['candidate_database']).read_text())
    right=json.loads(Path(resumed['candidate_database']).read_text())
    for name in left:
        with safe_open(left[name],framework='pt') as a, safe_open(right[name],framework='pt') as b:
            assert_same({k:a.get_tensor(k) for k in a.keys()},{k:b.get_tensor(k) for k in b.keys()})
    for index in range(len(records)):
        a=torch.load(Path(reference['final_cache'])/f'{index:06d}.pt',weights_only=True)
        b=torch.load(Path(resumed['final_cache'])/f'{index:06d}.pt',weights_only=True)
        assert_same(a,b)
