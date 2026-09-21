"""Runnable staged Qwen quantization job; stop/resume limits preserve full corpus.

Stages: initialize Q1 candidates, GSQ hybrid blocks, RCO full-model search.
Each invocation is replayed in full with fresh state. Tiny fixtures are tests,
never a substitute for the mandatory actual-27B CUDA smoke.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import signal

import torch
from safetensors.torch import save_file
from .performance import measure
from .candidates import Q1Candidate,pack_signs
from .qwen import load_original,projection_modules,run_block,tiny_model

STOP_REQUESTED=False
def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED=True

def should_stop(config,steps,max_steps):
    return STOP_REQUESTED or (max_steps is not None and steps>=max_steps) or time.time()>=config.get('deadline_unix',float('inf'))


def project_scale_format(module):
    """Constrain learned scales to the actual Q1 FP16 storage/BF16 compute grid."""
    with torch.no_grad():
        for name,parameter in module.named_parameters():
            if name.endswith('scales'):
                effective=parameter.half().bfloat16().float()
                if not torch.isfinite(effective).all() or not torch.isfinite(effective.half()).all():raise FloatingPointError('Q1 scale cannot fit FP16 storage')
                parameter.copy_(effective)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def corpus_inputs(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='completed' or manifest.get('model')!='Qwen/Qwen3.8-27B':
        raise ValueError('Require completed original-Qwen activation corpus')
    sequences=manifest['sequences']
    if len(sequences)!=manifest['source_invocation_count'] or not sequences:
        raise ValueError('Incomplete full invocation corpus')
    records=[]
    for item in sequences:
        directory=root/item['directory']
        if digest(directory/'sequence.json')!=item['manifest_sha256']:
            raise ValueError('Sequence manifest changed')
        seq=json.loads((directory/'sequence.json').read_text())
        if digest(directory/'input.json')!=seq['input_sha256'] or not seq.get('fresh_state'):
            raise ValueError('Invocation integrity/state mismatch')
        record=json.loads((directory/'input.json').read_text())
        tokens=record['tokenIDs']
        if len(tokens)!=item['tokens'] or len(tokens)<2 or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError('Invalid full token sequence; do not truncate')
        if hashlib.sha256(record['prompt'].encode()).hexdigest()!=record['promptSHA256']:
            raise ValueError('Captured prompt changed')
        records.append(tokens)
    return records


def validate_restored_inputs(config):
    root=Path(config['inputs']).resolve()
    receipt=json.loads((root/'restore-validation.json').read_text())
    corpus=root/'calibration';model=root/'model'
    if receipt.get('status')!='validated' or receipt.get('input_commit_sha256')!=config['input_commit_sha256']:
        raise ValueError('Trusted restored-input receipt mismatch')
    if receipt.get('manifest_sha256')!=digest(corpus/'manifest.json'):
        raise ValueError('Restored calibration changed after validation')
    manifest=json.loads((corpus/'manifest.json').read_text())
    if manifest.get('model')!='Qwen/Qwen3.8-27B' or manifest.get('revision')!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':
        raise ValueError('Wrong original baseline identity')
    files=manifest['model_artifacts']['files']
    if {p.name for p in model.iterdir()}!=set(files):raise ValueError('Restored model file coverage changed')
    for name,item in files.items():
        path=model/name
        if path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('Restored model file changed')
        # Large shards were hashed by CPU restore. The source disk must be
        # mounted read-only for the job. Recheck small configuration files here.
        if not name.endswith('.safetensors') and digest(path)!=item['sha256']:
            raise ValueError('Restored model configuration changed')
    config.update(model_dir=str(model),corpus=str(corpus))
    return manifest


def bind_identity(config,manifest):
    excluded={'identity','output','inputs','model_dir','corpus','checkpoint','deadline_unix',
              'candidate_database','candidate_archives','final_cache','boundary_reports','cost_manifest','warmstart_states','resume_checkpoint'}
    if config.get('warmstart_states'):
        hashes=config.setdefault('warmstart_sha256',{})
        for block,path in config['warmstart_states'].items():
            actual=digest(path) if Path(path).exists() else hashes.get(block)
            if actual is None or (block in hashes and hashes[block]!=actual):raise ValueError('Warmstart identity changed')
            hashes[block]=actual
    algorithm={k:v for k,v in config.items() if k not in excluded}
    algorithm['solver_files']={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    if 'cost_manifest' in config:algorithm['cost_manifest_sha256']=config['cost_manifest']['sha256']
    if 'boundary_reports' in config:algorithm['boundary_reports_sha256']=[x['sha256'] for x in config['boundary_reports']]
    encoded=json.dumps(algorithm,sort_keys=True,separators=(',',':')).encode()
    existing=config.get('identity',{})
    database={name:digest(path) for name,path in read_database(config['candidate_database']).items()} if config.get('candidate_database') and Path(config['candidate_database']).exists() else None
    candidate_hash=(hashlib.sha256(json.dumps(database,sort_keys=True).encode()).hexdigest()
                    if database is not None else existing.get('candidate_database_sha256',hashlib.sha256(b'no-candidates-yet').hexdigest()))
    identity={'baseline_repo':manifest['model'],'baseline_revision':manifest['revision'],
              'calibration_sha256':digest(Path(config['corpus'])/'manifest.json'),
              'solver_revision':'03fc16484c369e3127225615d5e03e8d3a6043e3',
              'solver_config_sha256':hashlib.sha256(encoded).hexdigest(),
              'candidate_database_sha256':candidate_hash,'runtime_sha256':config['runtime_sha256']}
    if existing and existing!=identity:raise ValueError('Job/code/data configuration differs from frozen checkpoint identity')
    config['identity']=identity
    return identity


def archive_directory(directory,output):
    directory,output=Path(directory),Path(output)
    if output.exists():
        expected={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()}
        observed={}
        with tarfile.open(output) as archive:
            for member in archive:
                if not member.isfile() or member.name in observed:raise ValueError('Invalid existing archive')
                h=hashlib.sha256()
                with archive.extractfile(member) as f:
                    for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
                observed[member.name]=h.hexdigest()
        if observed!=expected:raise ValueError('Existing archive differs from current immutable directory')
        return output
    pending=output.with_suffix('.pending')
    with tarfile.open(pending,'w') as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                info=archive.gettarinfo(str(path),arcname=str(path.relative_to(directory)))
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o600
                with path.open('rb') as f:
                    archive.addfile(info,f)
    os.replace(pending,output)
    return output


def extract_verified_archive(path,destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Restore requires new owned directory')
    destination.mkdir(parents=True)
    seen=set()
    with tarfile.open(path) as archive:
        for member in archive:
            name=Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name in seen:
                raise ValueError('Unsafe solver archive')
            seen.add(member.name)
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,target.open('xb') as out:
                shutil.copyfileobj(source,out,8*1024**2)


def initialize_candidates(model,output,row_chunk=512):
    output=Path(output);directory=output/'initial';directory.mkdir(parents=True,exist_ok=True)
    database={}
    for name,module in projection_modules(model).items():
        path=directory/(name+'.safetensors')
        if not path.exists():
            codes=[];scales=[]
            with torch.no_grad():
                for start in range(0,len(module.weight),row_chunk):
                    weight=module.weight[start:start+row_chunk]
                    scale=weight.float().reshape(len(weight),-1,128).abs().mean(-1).clamp_min(1e-8)
                    effective=scale.bfloat16().half().bfloat16()
                    if not torch.isfinite(effective).all():raise ValueError('RTN scale exceeds FP16 storage')
                    codes.append(pack_signs(weight).cpu());scales.append(effective.cpu())
                pending=path.with_suffix('.pending')
                save_file({'codes':torch.cat(codes),'scales':torch.cat(scales)},str(pending),
                    metadata={'format':'zimfo-q1-v1','group_size':'128','bit_order':'little',
                              'effective_scale_dtype':'BF16','zero_logit_sign':'negative'})
                os.replace(pending,path)
        candidate=Q1Candidate(path)
        if tuple(module.weight.shape)!=candidate.shape:
            raise ValueError('Existing candidate shape mismatch')
        database[name]=str(path)
    atomic_json(output/'initial-database.json',database)
    archive_directory(directory,output/'initial-candidates.tar')
    return database


def read_database(path):
    path=Path(path);data=json.loads(path.read_text())
    return {name:str((path.parent/value).resolve()) if not Path(value).is_absolute() else value
            for name,value in data.items()}


def restore_candidates(restored,output):
    database={};extras={};output=Path(output)
    for role in sorted(Path(restored).iterdir(),key=lambda p:(p.name!='initial_candidates',p.name)):
        if role.name=='initial_candidates':directory=output/'initial'
        elif role.name=='embedding_candidates':directory=output/'embedding'
        elif role.name=='head_candidates':directory=output/'head'
        elif role.name.startswith('candidate_block_'):directory=output/('block-'+role.name.removeprefix('candidate_block_'))
        elif role.name.startswith('cache_'):
            target=output/(role.name+'.tar');shutil.copyfile(role,target);extras[role.name]=target
            continue
        else:continue
        target=output/(role.name+'.tar');shutil.copyfile(role,target);extras[role.name]=target
        extract_verified_archive(target,directory)
        for path in directory.glob('*.safetensors'):
            name=path.stem
            if role.name.startswith('candidate_block_'):
                name=f"model.layers.{role.name.removeprefix('candidate_block_')}.{name}"
            database[name]=str(path)
    return database,extras


def make_store(config):
    from checkpoints import LocalCheckpointStore
    checkpoint=config['checkpoint']
    if checkpoint['backend']=='gcs':
        from gcs_checkpoints import GCSCheckpointStore
        return GCSCheckpointStore(checkpoint['bucket'],checkpoint['prefix'],project=checkpoint.get('project'),
                                  staging_dir=checkpoint.get('staging_dir'))
    if checkpoint['backend']=='local':
        return LocalCheckpointStore(Path(checkpoint['path']))
    raise ValueError('Unknown durable checkpoint backend')


class Checkpointer:
    def __init__(self,config,output):
        self.store=make_store(config);self.identity=config['identity'];self.output=Path(output);self.config=config
        self.last=time.monotonic();self.cadence=config.get('checkpoint_seconds',120);self.publish_ordinal=0
        if self.cadence<30:
            raise ValueError('No per-step/tiny durable checkpoint writes; minimum cadence30s')
    def save(self,module,optimizer,scheduler,progress,extras,force=False):
        from checkpoints import capture_rng_state
        if not force and time.monotonic()-self.last<self.cadence:
            return
        if torch.cuda.is_available():torch.cuda.synchronize()
        with measure(self.output, progress['stage'], 'checkpoint_total'):
            with tempfile.TemporaryDirectory(prefix='solver-state-',dir=self.output) as directory:
                directory=Path(directory)
                objects={'solver':module.state_dict(),'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict(),'rng':capture_rng_state(),'progress':progress}
                payloads={}
                for role,value in objects.items():
                    path=directory/role;torch.save(value,path);payloads[role]=path
                payloads.update(extras)
                for block,path in self.config.get('warmstart_states',{}).items():
                    if digest(path)!=self.config['warmstart_sha256'][block]:raise ValueError('Warmstart changed')
                    payloads['warmstart_block_'+block]=Path(path)
                for descriptor in self.config.get('boundary_reports',[]):
                    if digest(descriptor['path'])!=descriptor['sha256']:raise ValueError('Boundary report changed')
                    stage=json.loads(Path(descriptor['path']).read_text())['stage']
                    payloads['boundary_report_'+stage]=Path(descriptor['path'])
                if 'cost_manifest' in self.config:
                    descriptor=self.config['cost_manifest']
                    if digest(descriptor['path'])!=descriptor['sha256']:raise ValueError('Cost manifest changed')
                    payloads['packing_cost_manifest']=Path(descriptor['path'])
                if (self.output/'frozen-config.json').exists():
                    payloads['configuration']=self.output/'frozen-config.json'
                snapshot=(f"{progress['stage']}-b{progress.get('block',0):03d}-s{progress['global_step']:08d}"
                          f"-e{progress['epoch']:03d}-q{progress['sequence']:05d}-p{self.publish_ordinal:05d}")
                started=time.monotonic()
                with measure(self.output, progress['stage'], 'checkpoint_publish'):
                    receipt=self.store.publish(snapshot,self.identity,payloads)
                self.publish_ordinal+=1
                atomic_json(self.output/'latest-checkpoint.json',{'snapshot':snapshot,'receipt':receipt,
                    'publish_seconds':time.monotonic()-started,
                    'local_payload_bytes':sum(p.stat().st_size for p in payloads.values())})
        self.last=time.monotonic()
    def restore(self,snapshot,generation=None):
        directory=self.output/('restored-'+snapshot)
        kwargs={'commit_generation':generation} if generation is not None else {}
        source=make_store({**self.config,'checkpoint':self.config['resume_checkpoint']}) if self.config.get('resume_checkpoint') else self.store
        with measure(self.output, self.config.get('stage', 'unknown'), 'checkpoint_restore'):
            source.restore(snapshot,self.identity,directory,**kwargs)
        return directory


def remap_restored_auxiliary(config,directory):
    directory=Path(directory)
    for block in config.get('warmstart_states',{}):
        path=directory/('warmstart_block_'+block)
        if not path.exists() or digest(path)!=config['warmstart_sha256'][block]:raise ValueError('Missing/changed restored warmstart')
        config['warmstart_states'][block]=str(path)
    for descriptor in config.get('boundary_reports',[]):
        matches=[p for p in directory.glob('boundary_report_*') if digest(p)==descriptor['sha256']]
        if len(matches)!=1:raise ValueError('Missing/ambiguous restored boundary report')
        descriptor['path']=str(matches[0])
    if 'cost_manifest' in config:
        descriptor=config['cost_manifest'];path=directory/'packing_cost_manifest'
        if not path.exists() or digest(path)!=descriptor['sha256']:raise ValueError('Missing/changed restored costs')
        descriptor['path']=str(path)


def restore_state(directory,module,optimizer,scheduler):
    from checkpoints import restore_rng_state
    for role,object in [('solver',module),('optimizer',optimizer),('scheduler',scheduler)]:
        object.load_state_dict(torch.load(Path(directory)/role,map_location='cpu',weights_only=True))
    restore_rng_state(torch.load(Path(directory)/'rng',map_location='cpu',weights_only=True))
    return torch.load(Path(directory)/'progress',map_location='cpu',weights_only=True)


def cached_inputs(model,records,directory,embedding_candidate,device=None):
    from .rco import gather_candidate
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    device=torch.device(device) if device is not None else model.model.embed_tokens.weight.device
    if device!=model.model.embed_tokens.weight.device:
        raise ValueError('Embedding cache device must match baseline embedding residency')
    with torch.no_grad():
        for index,tokens in enumerate(records):
            path=directory/f'{index:06d}.pt'
            if path.exists():
                continue
            ids=torch.tensor([tokens],device=device)
            reference=model.model.embed_tokens(ids)
            student=gather_candidate(embedding_candidate,ids,model.model.embed_tokens.weight)
            torch.save({'teacher':reference.cpu(),'student':student.cpu()},path)


def gsq_run(model,records,config,output,checkpointer,resume=None,max_steps=None):
    from .gsq import BlockTrainer
    from .gsq_residency import GSQResidency
    residency=GSQResidency(model,config)
    output=Path(output);device=residency.device
    database={} if resume else read_database(config.get('candidate_database',output/'initial-database.json'))
    extras={k:Path(v) for k,v in config.get('candidate_archives',{'initial_candidates':output/'initial-candidates.tar'}).items()}
    progress={'stage':'gsq','block':0,'epoch':0,'sequence':0,'global_step':0}
    if resume:
        progress=torch.load(Path(resume)/'progress',weights_only=True)
        if progress['stage']!='gsq':
            raise ValueError('Cannot resume wrong stage')
        database,extras=restore_candidates(resume,output)
        extract_verified_archive(extras[f"cache_block_{progress['block']}"],output/f"cache-{progress['block']}")
    else:
        cached_inputs(model,records,output/'cache-0',Q1Candidate(database['model.embed_tokens']),device=residency.cache_device)
    steps_this_run=0
    for block_index in range(progress['block'],len(model.model.layers)):
        cache=output/f'cache-{block_index}'
        cache_role=f'cache_block_{block_index}'
        extras[cache_role]=archive_directory(cache,output/f'cache-{block_index}.tar')
        with residency.block(block_index):
            trainer=optimizer=scheduler=loss=tensors=teacher=student=state=None
            try:
                trainer=BlockTrainer(model,block_index,config.get('upstream','/opt/upstream'))
                optimizer=torch.optim.Adam(trainer.parameters(),lr=config.get('gsq_lr',.001))
                epochs=config.get('gsq_epochs',1)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs*len(records)))
                if resume:
                    progress=restore_state(resume,trainer,optimizer,scheduler);resume=None
                else:
                    warm=config.get('warmstart_states',{}).get(str(block_index))
                    if warm:
                        if digest(warm)!=config['warmstart_sha256'][str(block_index)]:raise ValueError('Warmstart changed')
                        state=torch.load(warm,map_location='cpu',weights_only=True)
                        if state['block']!=block_index or state['updates']!=1:raise ValueError('Invalid GSQ warmstart')
                        trainer.load_state_dict(state['solver']);optimizer.load_state_dict(state['optimizer'])
                    progress.update(block=block_index,epoch=0,sequence=0)
                # Commit phase-start state with its immutable cache before dropping old
                # cache archives. This bounds local scratch across transformer blocks.
                checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=True)
                for old in output.glob('cache-*.tar'):
                    if old != extras[cache_role]:old.unlink()
                for epoch in range(progress['epoch'],epochs):
                    first=progress['sequence'] if epoch==progress['epoch'] else 0
                    for index in range(first,len(records)):
                        with measure(output, 'gsq', 'update_with_input_load', device, block=block_index, kind=model.config.layer_types[block_index], tokens=len(records[index])):
                            tensors=torch.load(cache/f'{index:06d}.pt',weights_only=True)
                            optimizer.zero_grad(set_to_none=True)
                            fraction=(epoch*len(records)+index)/max(1,epochs*len(records)-1)
                            loss=trainer(tensors['student'].to(device),tensors['teacher'].to(device),
                                         temperature=1.-.9*fraction)
                            loss.backward()
                            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainer.parameters()):
                                raise FloatingPointError('Nonfinite/missing GSQ gradient')
                            optimizer.step();project_scale_format(trainer);scheduler.step()
                        progress.update(block=block_index,epoch=epoch,sequence=index+1,
                                        global_step=progress['global_step']+1,last_loss=loss.item())
                        steps_this_run+=1
                        print(json.dumps(progress),flush=True)
                        stopping=should_stop(config,steps_this_run,max_steps)
                        checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=stopping)
                        if stopping:
                            return {'status':'checkpointed_stop','progress':progress}
                    progress.update(epoch=epoch+1,sequence=0)
                block_output=output/f'block-{block_index}'
                database.update(trainer.export(block_output))
                extras[f'candidate_block_{block_index}']=archive_directory(block_output,output/f'block-{block_index}.tar')
                next_cache=output/f'cache-{block_index+1}';next_cache.mkdir(exist_ok=True)
                with measure(output, 'gsq', 'cache_propagation', device, block=block_index, kind=model.config.layer_types[block_index], tokens=sum(map(len,records)), tokens_squared=sum(len(r)**2 for r in records), sequences=len(records)):
                    with torch.no_grad():
                        for index in range(len(records)):
                            tensors=torch.load(cache/f'{index:06d}.pt',weights_only=True)
                            teacher=run_block(model,block_index,tensors['teacher'].to(device))
                            student=trainer.hard_forward(tensors['student'].to(device))
                            torch.save({'teacher':teacher.cpu(),'student':student.cpu()},next_cache/f'{index:06d}.pt')
                # Checkpoint final active state before deleting any resumable input cache.
                checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=True)
                atomic_json(output/'candidate-database.json',database)
            finally:
                loss=tensors=teacher=student=state=None
                residency.release_training_state(trainer,optimizer)
                trainer=optimizer=scheduler=None
        # Keep current cache archive for last committed checkpoint; discard only
        # live expanded cache. Completed archives are retained until run finalization.
        shutil.rmtree(cache)
        extras.pop(cache_role,None)
    return {'status':'completed','optimizer_updates':steps_this_run,'candidate_database':str(output/'candidate-database.json'),
            'candidate_archives':{k:str(v) for k,v in extras.items()},
            'final_cache':str(output/f'cache-{len(model.model.layers)}'),
            'embedding_head_strategy':'Embedding inherited from candidate database; head optimization remains a separate stage',
            'final_quality_recipe_ready':False,
            'blocking_gaps':['Learned embedding/head optimization','Complete serialized byte-cost accounting']}


def rco_run(model,records,config,output,checkpointer,resume=None,max_steps=None):
    from .rco import RCOTrainer
    output=Path(output)
    if resume:remap_restored_auxiliary(config,resume)
    learned={}
    for descriptor in config.get('boundary_reports',[]):
        if digest(descriptor['path'])!=descriptor['sha256']:raise ValueError('Boundary training report changed')
        report=json.loads(Path(descriptor['path']).read_text())
        if report.get('status')!='completed' or report.get('updates',0)<=0:
            raise ValueError('Boundary training did not complete')
        learned[report['stage']]=report
    if set(learned)!={'embedding','head'}:
        if max_steps is None or max_steps>2 or not config.get('allow_rtn_boundary_smoke',False):
            raise ValueError('RTN-only embedding/head are smoke seeds, not final recipe; bounded <=2-step smoke required')
    database={} if resume else read_database(config['candidate_database'])
    if resume:
        restored_progress=torch.load(Path(resume)/'progress',weights_only=True)
        if restored_progress['stage']!='rco':raise ValueError('Cannot resume wrong stage')
        database,restored_extras=restore_candidates(resume,output)
    for stage,name in [('embedding','model.embed_tokens'),('head','lm_head')]:
        if stage in learned and digest(database[name])!=learned[stage]['candidate_sha256']:
            raise ValueError('Candidate differs from learned boundary report')
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    costs=None
    if 'cost_manifest' in config:
        descriptor=config['cost_manifest']
        if digest(descriptor['path'])!=descriptor['sha256']:raise ValueError('Packing cost manifest changed')
        costs=json.loads(Path(descriptor['path']).read_text())
        target=config['target_bytes'];fixed=costs['fixed_container_bytes']
    else:
        if max_steps is None or max_steps>2:raise ValueError('Long RCO run requires exact serialized byte-cost manifest')
        target=config['target_tensor_bytes'];fixed=config['fixed_tensor_bytes']
    trainer=RCOTrainer(model,database,target,fixed,config.get('upstream','/opt/upstream'),cost_manifest=costs)
    optimizer=torch.optim.Adam([trainer.alpha],lr=config.get('rco_lr',.01))
    epochs=config.get('rco_epochs',1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs*len(records)))
    progress={'stage':'rco','epoch':0,'sequence':0,'global_step':0}
    if resume:progress=restore_state(resume,trainer,optimizer,scheduler)
    extras=restored_extras if resume else {name:Path(value) for name,value in config['candidate_archives'].items()}
    device=next(model.parameters()).device;steps_this_run=0
    for epoch in range(progress['epoch'],epochs):
        first=progress['sequence'] if epoch==progress['epoch'] else 0
        for index in range(first,len(records)):
            fraction=(epoch*len(records)+index)/max(1,epochs*len(records)-1)
            with measure(output, 'rco', 'update', device, tokens=len(records[index])):
                report=trainer.step(torch.tensor([records[index]],device=device),optimizer,
                                    temperature=max(.1,1.-.9*fraction))
                scheduler.step();steps_this_run+=1
            progress.update(epoch=epoch,sequence=index+1,global_step=progress['global_step']+1,**report)
            print(json.dumps(progress),flush=True)
            stopping=should_stop(config,steps_this_run,max_steps)
            checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=stopping)
            if stopping:return {'status':'checkpointed_stop','progress':progress}
        progress.update(epoch=epoch+1,sequence=0)
    checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=True)
    allocation=trainer.hard_allocation(target,fixed)
    atomic_json(output/'allocation.json',allocation)
    return {'status':'completed','allocation':allocation,'optimizer_updates':steps_this_run,'packaged_model_validated':False}


def boundary_run(stage,model,records,config,output,checkpointer,resume=None,max_steps=None):
    from .boundaries import EmbeddingTrainer,HeadTrainer
    output=Path(output);device=model.lm_head.weight.device
    if resume:
        database,extras=restore_candidates(resume,output)
    else:
        database=read_database(config['candidate_database'])
        extras={k:Path(v) for k,v in config['candidate_archives'].items()}
    cache=None
    if stage=='embedding':
        trainer=EmbeddingTrainer(model,records,config.get('upstream','/opt/upstream'))
    else:
        # Only head state is trained here; release the finished body from VRAM
        # before allocating ~20GB FP32 head logits, gradients and Adam moments.
        model.model.to('cpu');model.model.norm.to(device)
        if device.type=='cuda':torch.cuda.empty_cache()
        if resume:
            cache=output/'final-cache';extract_verified_archive(extras['cache_final'],cache)
        else:
            cache=Path(config['final_cache'])
            extras['cache_final']=archive_directory(cache,output/'cache-final.tar')
        trainer=HeadTrainer(model.lm_head.weight)
    optimizer=torch.optim.Adam(trainer.parameters(),lr=config.get(stage+'_lr',.001))
    epochs=config.get(stage+'_epochs',1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs*len(records)))
    progress={'stage':stage,'epoch':0,'sequence':0,'global_step':0}
    if resume:
        progress=restore_state(resume,trainer,optimizer,scheduler)
        if progress['stage']!=stage:raise ValueError('Wrong boundary resume stage')
    steps=0
    for epoch in range(progress['epoch'],epochs):
        first=progress['sequence'] if epoch==progress['epoch'] else 0
        for index in range(first,len(records)):
            with measure(output, stage, 'update_with_input_load', device, tokens=len(records[index])):
                optimizer.zero_grad(set_to_none=True)
                fraction=(epoch*len(records)+index)/max(1,epochs*len(records)-1)
                temperature=1.-.9*fraction
                if stage=='embedding':
                    loss=trainer(torch.tensor([records[index]],device=device),temperature)
                else:
                    pair=torch.load(cache/f'{index:06d}.pt',weights_only=True)
                    with torch.no_grad():
                        student=model.model.norm(pair['student'].to(device))[0,:-1]
                        teacher=model.model.norm(pair['teacher'].to(device))[0,:-1]
                    loss=trainer(student,teacher,temperature)
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainer.parameters()):
                    raise FloatingPointError('Nonfinite/missing learned boundary gradient')
                optimizer.step();project_scale_format(trainer);scheduler.step();steps+=1
            progress.update(epoch=epoch,sequence=index+1,global_step=progress['global_step']+1,last_loss=loss.item())
            print(json.dumps(progress),flush=True)
            stop=should_stop(config,steps,max_steps)
            checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=stop)
            if stop:return {'status':'checkpointed_stop','progress':progress}
        progress.update(epoch=epoch+1,sequence=0)
    directory=output/stage;directory.mkdir(exist_ok=True)
    name='model.embed_tokens' if stage=='embedding' else 'lm_head'
    candidate=directory/(name+'.safetensors')
    if not candidate.exists():
        if stage=='embedding':trainer.export(database[name],candidate)
        else:trainer.export(candidate)
    else:
        q=Q1Candidate(candidate)
        if stage=='embedding':
            from .rco import gather_candidate
            actual=gather_candidate(q,trainer.observed_ids,model.model.embed_tokens.weight)
            expected=trainer.quantizer.get_hard_weights()[0]
            torch.testing.assert_close(actual,expected.to(actual.dtype),rtol=0,atol=0)
        else:
            for row in range(0,q.shape[0],512):
                end=min(row+512,q.shape[0])
                expected=torch.where(trainer.sign_logits[row:end]>0,1.,-1.)*trainer.scales[row:end].bfloat16().float().repeat_interleave(128,1)
                torch.testing.assert_close(q.rows(row,end,device,torch.float32),expected,rtol=0,atol=0)
    database[name]=str(candidate)
    extras[stage+'_candidates']=archive_directory(directory,output/(stage+'-candidates.tar'))
    checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=True)
    atomic_json(output/'candidate-database.json',database)
    return {'status':'completed','stage':stage,'updates':progress['global_step'],
            'candidate_sha256':digest(candidate),
            'candidate_database':str(output/'candidate-database.json'),
            'candidate_archives':{k:str(v) for k,v in extras.items()},
            'objective':'first_hybrid_block_MSE_observed_token_rows' if stage=='embedding' else 'full_vocabulary_KL_original_teacher',
            'embedding_unobserved_rows':'RTN initialization' if stage=='embedding' else None,
            'scale_policy':'FP16_storage_BF16_compute_intersection_projected_after_each_update'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['initialize','smoke','embedding','gsq','head','rco'],required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--resume')
    parser.add_argument('--commit-generation',type=int)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    config=json.loads(args.config.read_text());config['stage']=args.stage;output=Path(config['output'])
    if not args.dry_run and output.exists() and any(output.iterdir()):
        raise FileExistsError('Every stage/resume needs a new owned output directory; candidates are external immutable inputs')
    if args.resume and (not config.get('resume_checkpoint') or config['resume_checkpoint']==config['checkpoint']):
        raise ValueError('Resume must read original checkpoint prefix and write a fresh attempt prefix')
    from .gsq_residency import memory_mode, execution_device, BLOCK_CPU_OFFLOAD
    mode=memory_mode(config)
    manifest=validate_restored_inputs(config)
    bind_identity(config,manifest)
    records=corpus_inputs(config['corpus'])
    if config.get('largest_first',False):records.sort(key=len,reverse=True)
    if args.max_steps is not None and args.max_steps<=0:raise ValueError('Positive step limit required')
    if args.dry_run:
        print(json.dumps({'status':'inputs_validated','invocations':len(records),
                          'total_tokens':sum(map(len,records)),'gpu_validated':False}));return
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'frozen-config.json',config)
    from .reproducibility import configure
    execution_policy=configure(config.get('seed',42))
    atomic_json(output/'execution-policy-report.json',execution_policy)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Actual Qwen27B job requires CUDA BF16; CPU is only for tiny tests')
    compute_device=(execution_device(config,torch.device('cuda')) if args.stage=='gsq' else torch.device('cuda'))
    if compute_device.type!='cuda':raise RuntimeError('Actual Qwen27B GSQ requires CUDA; CPU residency tests use tiny models directly')
    load_device=torch.device('cpu') if mode==BLOCK_CPU_OFFLOAD else compute_device
    with measure(output, args.stage, 'stage_total', compute_device):
        with measure(output, args.stage, 'model_load', compute_device):
            model=load_original(config['model_dir'],load_device)
        if args.stage=='initialize':
            database=initialize_candidates(model,output)
            result={'status':'completed','candidate_count':len(database),'scope':'RTN initialization, not GSQ optimization'}
        else:
            checkpointer=Checkpointer(config,output)
            restored=checkpointer.restore(args.resume,args.commit_generation) if args.resume else None
            if restored:remap_restored_auxiliary(config,restored)
            if args.stage=='smoke':
                if restored:raise ValueError('Smoke checkpoints are diagnostic; resume a production stage')
                from .smoke import smoke_run
                result=smoke_run(model,records,config,output,checkpointer)
            elif args.stage in ('embedding','head'):
                result=boundary_run(args.stage,model,records,config,output,checkpointer,restored,args.max_steps)
            else:
                result=(gsq_run if args.stage=='gsq' else rco_run)(model,records,config,output,checkpointer,restored,args.max_steps)
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['calibration_sha256']=config['identity']['calibration_sha256']
        result['runtime_sha256']=config['identity']['runtime_sha256']
        atomic_json(output/(args.stage+'-report.json'),result)


if __name__=='__main__':main()
