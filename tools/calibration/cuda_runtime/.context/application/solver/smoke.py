"""Actual-model smoke on the longest complete invocation; persistent useful state."""
import copy
import json
import sys
from pathlib import Path
import torch
from .performance import measure
from .gsq import BlockTrainer
from .qwen import run_block
from .rco import RCOTrainer,gather_candidate
from .candidates import Q1Candidate
from .qwen import block_kwargs
from .run import read_database,digest,project_scale_format,restore_state,atomic_json


def cpu_clone(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_clone(v) for v in value]
    return copy.deepcopy(value)


def assert_state_equal(actual,expected,exact=False):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(),expected,atol=0 if exact else 1e-6,rtol=0 if exact else 1e-5)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise ValueError('Restored optimizer state keys differ')
        for key,value in expected.items():assert_state_equal(actual[key],value,exact)
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise ValueError('Restored state length differs')
        for a,e in zip(actual,expected):assert_state_equal(a,e,exact)
    elif actual!=expected:raise ValueError('Restored state metadata differs')


def smoke_run(model,records,config,output,checkpointer,allow_cpu_test=False):
    output=Path(output);device=model.lm_head.weight.device
    if device.type!='cuda' and not allow_cpu_test:raise ValueError('Production smoke requires actual CUDA')
    ids=torch.tensor([max(records,key=len)],device=device)
    component=config.get('smoke_component','both')
    if not allow_cpu_test and component not in ('gsq','rco'):
        raise ValueError('Production GSQ/RCO smoke components require isolated processes')
    database=read_database(config['candidate_database'])
    costs=None
    if 'cost_manifest' in config:
        descriptor=config['cost_manifest']
        if digest(descriptor['path'])!=descriptor['sha256']:raise ValueError('Packing costs changed')
        costs=json.loads(Path(descriptor['path']).read_text())
        target=config['target_bytes'];fixed=costs['fixed_container_bytes']
    else:
        if not allow_cpu_test:raise ValueError('Actual smoke requires complete serialized byte costs')
        target=config['target_tensor_bytes'];fixed=config['fixed_tensor_bytes']
    extras={k:Path(v) for k,v in config['candidate_archives'].items()}
    details={};warmstarts={}
    # Original and RTN-prefix inputs are recomputed from complete token IDs.
    # No sampled activation matrix or truncated prompt substitutes for them.
    candidates={n:Q1Candidate(p) for n,p in database.items()} if component in ('gsq','both') else {}
    for kind in (('linear_attention','full_attention') if component in ('gsq','both') else ()):
        block=next(i for i,t in enumerate(model.config.layer_types) if t==kind)
        with torch.no_grad():
            teacher=model.model.embed_tokens(ids)
            student=gather_candidate(candidates['model.embed_tokens'],ids,model.model.embed_tokens.weight)
            for previous in range(block):
                teacher=run_block(model,previous,teacher)
                replacements={n.removeprefix(f'model.layers.{previous}.')+'.weight':q.rows(0,q.shape[0],device,student.dtype)
                              for n,q in candidates.items() if n.startswith(f'model.layers.{previous}.')}
                student=torch.func.functional_call(model.model.layers[previous],replacements,(student,),
                                block_kwargs(model,student,previous),strict=False)
        if block>0:del replacements
        trainer=BlockTrainer(model,block,config.get('upstream','/opt/upstream'))
        optimizer=torch.optim.Adam(trainer.parameters(),lr=config.get('gsq_lr',.001))
        with measure(output, 'smoke_gsq', 'update', device, block=block, kind=kind, tokens=ids.shape[1]):
            loss=trainer(student,teacher)
            loss.backward()
            finite=bool(torch.isfinite(loss)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainer.parameters())
            gradient=sum(p.grad.float().square().sum().item() for p in trainer.parameters())**.5 if finite else float('nan')
            if not finite or gradient<=0:raise FloatingPointError('Actual hybrid GSQ backward failed')
            optimizer.step();project_scale_format(trainer)
        state=output/f'warmstart-block-{block}.pt'
        torch.save({'solver':trainer.state_dict(),'optimizer':optimizer.state_dict(),
                    'block':block,'updates':1,'scope':'largest-invocation GSQ warm start'},state)
        extras[f'warmstart_block_{block}']=state;warmstarts[str(block)]=str(state)
        details[kind+'_gsq']={'passed':True,'block':block,'loss':loss.item(),'gradient_norm':gradient}
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1)
        progress={'stage':'smoke_gsq','block':block,'epoch':0,'sequence':1,'global_step':len(details)}
        checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=True)
        del trainer,optimizer,student,teacher,loss
        if device.type=='cuda':torch.cuda.empty_cache()
    if component=='gsq':
        result={'status':'completed','smoke':{'passed':device.type=='cuda','model':'Qwen/Qwen3.8-27B',
                    'revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','sequence_tokens':ids.shape[1],
                    'cuda':cuda_report(device),**details},'warmstart_states':warmstarts,
                'optimizer_updates':2,'durable_checkpoint':json.loads((output/'latest-checkpoint.json').read_text())}
        atomic_json(output/'smoke-report.json',result);return result
    for block,path in config.get('warmstart_states',{}).items():extras[f'warmstart_block_{block}']=Path(path)
    rco=RCOTrainer(model,database,target,fixed,config.get('upstream','/opt/upstream'),cost_manifest=costs)
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.Adam([rco.alpha],lr=config.get('rco_lr',.01))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,3)
    with measure(output, 'smoke_rco', 'first_update', device, tokens=ids.shape[1]):
        step=rco.step(ids,optimizer);scheduler.step()
    progress={'stage':'smoke_rco','epoch':0,'sequence':1,'global_step':1}
    from checkpoints import capture_rng_state
    before_solver=cpu_clone(rco.state_dict());before_optimizer=cpu_clone(optimizer.state_dict())
    before_scheduler=copy.deepcopy(scheduler.state_dict());before_rng=cpu_clone(capture_rng_state())
    diagnostics={'status':'running','checks':{},'first_update':step}
    diagnostic_path=output/'recovery-report.json'
    def check(name,actual,expected,exact=True):
        assert_state_equal(actual,expected,exact=exact)
        diagnostics['checks'][name]=True
    try:
        checkpointer.save(rco,optimizer,scheduler,progress,extras,force=True)
        receipt=json.loads((output/'latest-checkpoint.json').read_text())
        check('publication_preserves_rng',capture_rng_state(),before_rng)
        # Compare an uninterrupted next update with the same update restored
        # from durable storage. Pre-update equality separates restore corruption
        # from a CUDA replay/numerics discrepancy; do not relax the tolerance.
        with measure(output, 'smoke_rco', 'uninterrupted_update', device, tokens=ids.shape[1]):
            expected_report=rco.step(ids,optimizer);scheduler.step()
        diagnostics['uninterrupted_update']=expected_report
        expected_solver=cpu_clone(rco.state_dict());expected_optimizer=cpu_clone(optimizer.state_dict())
        expected_scheduler=copy.deepcopy(scheduler.state_dict())
        expected_rng=cpu_clone(capture_rng_state())
        generation=receipt['receipt'].get('commit',{}).get('generation')
        if not allow_cpu_test and generation is None:raise ValueError('Actual smoke requires a durable GCS generation')
        restored=checkpointer.restore(receipt['snapshot'],generation)
        restore_state(restored,rco,optimizer,scheduler)
        check('restored_solver_exact',rco.state_dict(),before_solver)
        check('restored_optimizer_exact',optimizer.state_dict(),before_optimizer)
        check('restored_scheduler_exact',scheduler.state_dict(),before_scheduler)
        check('restored_rng_exact',capture_rng_state(),before_rng)
        with measure(output, 'smoke_rco', 'replayed_update', device, tokens=ids.shape[1]):
            actual_report=rco.step(ids,optimizer);scheduler.step()
        diagnostics['replayed_update']=actual_report
        check('post_update_rng_exact',capture_rng_state(),expected_rng)
        check('post_update_solver',rco.state_dict(),expected_solver,exact=False)
        check('post_update_optimizer',optimizer.state_dict(),expected_optimizer,exact=False)
        check('post_update_scheduler',scheduler.state_dict(),expected_scheduler,exact=False)
        if abs(actual_report['loss']-expected_report['loss'])>1e-5*max(1.,abs(expected_report['loss'])):
            raise ValueError('Restored next loss differs')
        diagnostics['status']='passed'
    except Exception as error:
        diagnostics.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            diagnostics['cuda']=cuda_report(device)
            atomic_json(diagnostic_path,diagnostics)
        except Exception as diagnostic_error:
            if diagnostics['status']!='failed':raise
            try:
                print(f'recovery_diagnostic_write_failed: {diagnostic_error}',file=sys.stderr,flush=True)
            except Exception:
                pass
    progress.update(sequence=2,global_step=2)
    checkpointer.save(rco,optimizer,scheduler,progress,extras,force=True)
    durable=json.loads((output/'latest-checkpoint.json').read_text())
    cuda=cuda_report(device)
    result={'status':'completed','smoke':{'passed':not allow_cpu_test,'model':'Qwen/Qwen3.8-27B',
            'revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','sequence_tokens':ids.shape[1],
            'cuda':cuda,**details,'full_model_rco':{'passed':True,'full_vocabulary':True,
                                               'gradient_norm':step['raw_gradient_norm']},
            'checkpoint_resume':{'passed':True,'snapshot':receipt['snapshot'],
                                 'commit':receipt['receipt'].get('commit'),
                                 'next_update_matches':True}},
            'warmstart_states':warmstarts,'optimizer_updates':4 if component=='both' else 2,'durable_checkpoint':durable,
            'cpu_test_only':allow_cpu_test}
    atomic_json(output/'smoke-report.json',result)
    return result


def cuda_report(device):
    cuda={'available':device.type=='cuda'}
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(device)
        cuda.update(device=props.name,capability=list(torch.cuda.get_device_capability(device)),
                    total_memory_bytes=props.total_memory,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return cuda
