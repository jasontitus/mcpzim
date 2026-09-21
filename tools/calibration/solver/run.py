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
from . import device_policy
from .optim import quantizer_optimizer
from .device_policy import same_device
from .performance import measure
from .candidates import Q1Candidate,pack_signs
from .qwen import load_original,projection_modules,run_block,tiny_model

STOP_REQUESTED=False
#: How many invocations contribute to each per-block composed-stream drift record
#: (D3.1). Strided across the corpus, which is processed largest-first. Six is
#: enough: measured on the first block boundary available, the per-invocation ratio
#: ranged only 2.9447 to 2.9480.
DRIFT_SAMPLES = 6

#: Invocations reserved from training so a score is never measured on text the
#: model was calibrated on. The stride runs over the corpus's own invocation
#: numbering, matching the held-out text `solver/tools/quality_gate.py` measures
#: (`invocation-000000`, `-000011`, ... of the 87-invocation corpus).
HELD_OUT_STRIDE = 11


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED=True

def should_stop(config,steps,max_steps):
    return STOP_REQUESTED or (max_steps is not None and steps>=max_steps) or time.time()>=config.get('deadline_unix',float('inf'))


def should_stop_block(config,blocks_done):
    """Stop cleanly at a block boundary so the next block can start fresh.

    Block boundaries are exact resume points: the phase-start state and its
    input cache are committed before any update runs, so stopping here loses
    nothing. That matters because the port has a measured failure mode where a
    long-lived process produces a non-finite loss entering a block (block 3,
    reproducibly, after ~261 steps) while a FRESH process restoring the same
    checkpoint trains that block normally. Restarting per block turns that from
    a run-ending failure into a non-event.

    Off by default: `gsq_max_blocks` unset preserves a single-process run.
    """
    limit=config.get('gsq_max_blocks')
    return limit is not None and blocks_done>=int(limit)


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


def is_held_out_invocation(directory, stride):
    """True when an invocation directory is reserved for scoring, not training.

    An earlier revision of this port documented a held-out split that the code did
    not implement: the PPL bar and every arm were measured on eight invocations
    whose prompts were still in the training corpus, so a trained arm's number
    would have been optimistic while Bonsai's (trained elsewhere) was not.
    """
    if not stride:
        return False
    name=Path(directory).name
    if not name.startswith('invocation-'):
        return False
    try:
        return int(name.split('-',1)[1])%int(stride)==0
    except ValueError as error:
        raise ValueError(f'Unrecognized invocation directory: {name}') from error


def held_out_invocations(root, stride=HELD_OUT_STRIDE):
    """The reserved invocations' directories, for whoever scores the artifact."""
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    return [item['directory'] for item in manifest['sequences']
            if is_held_out_invocation(item['directory'],stride)]


def corpus_inputs(root, hold_out_stride=None):
    """Training records, optionally excluding the reserved invocation stride.

    Defaults to no exclusion so small synthetic corpora (and the CUDA canaries
    built on them) keep every sequence; `main` always passes
    ``config.get('held_out_stride', HELD_OUT_STRIDE)`` so every real stage trains
    without the text the artifact is scored on.
    """
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='completed' or manifest.get('model')!='Qwen/Qwen3.8-27B':
        raise ValueError('Require completed original-Qwen activation corpus')
    sequences=manifest['sequences']
    if len(sequences)!=manifest['source_invocation_count'] or not sequences:
        raise ValueError('Incomplete full invocation corpus')
    records=[]
    for item in sequences:
        directory=root/item['directory']
        if is_held_out_invocation(directory,hold_out_stride):
            continue
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
              'candidate_database','candidate_archives','final_cache','boundary_reports','cost_manifest','warmstart_states','resume_checkpoint',
              # Operational, not scientific: which directory this attempt writes
              # into cannot change what the calibration MEANS. Excluding it is
              # required for the attempt-scoped store to work at all, since a
              # resume always runs under a new attempt id.
              'attempt_id','_scope_attempt',
              # Operational, same class as `deadline_unix`: where a run stops and
              # how often it snapshots do not change the arithmetic. A per-block
              # sweep sets `gsq_max_blocks=1` while a single-process run leaves it
              # unset, and both must be able to resume the same chain. What stays
              # bound is the recipe: `gsq_epochs`, `seed`, `target_bytes`,
              # `stage`, and `gsq_memory_mode`/`gsq_execution_device`, because
              # those change the numbers a checkpoint means.
              'gsq_max_blocks','checkpoint_seconds'}
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
        root=Path(checkpoint['path'])
        # Snapshot names are built from stage + cursor + a publish ordinal that
        # restarts at 0 in every process, and the store refuses to overwrite an
        # existing name with different bytes -- correctly, since that is what
        # stops a resume reading a payload other than the one it was promised.
        # Together those mean a SECOND attempt at the same stage writes the same
        # snapshot names with different state and dies MID-RUN with "Immutable
        # object conflict", which reads like a numerical failure and is not one.
        # Measured: that collision masked a passing block-3 smoke result across
        # three attempts and cost most of a session to identify.
        #
        # Scoping the WRITE root per attempt makes the collision impossible
        # rather than procedural. Only writes are scoped: `restore` builds its
        # source store with an explicit `scoped=False` so a resume reads the
        # prefix it was told to read, at exactly the attempt directory named,
        # rather than a sibling attempt directory that happens to share the id.
        if config.get('attempt_id') is not None and config.get('_scope_attempt', True):
            root=root/str(config['attempt_id'])
        return LocalCheckpointStore(root)
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
        # Synchronize the device that produced the state before serialising it.
        # CUDA-only here would leave MPS checkpoint writes unserialized against
        # lazily-queued kernels. The device is taken from `module`, not from an
        # attribute on this class -- Checkpointer carries no `device`, so an
        # attribute lookup would silently resolve to None and no-op.
        from .device_policy import sync as _sync
        _params = getattr(module, 'parameters', None)
        _sync(next(_params()).device if _params is not None else None)
        with measure(self.output, progress['stage'], 'checkpoint_total'):
            with tempfile.TemporaryDirectory(prefix='solver-state-',dir=self.output) as directory:
                directory=Path(directory)
                objects={'solver':module.state_dict(),'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict(),'rng':capture_rng_state(),'progress':progress}
                payloads={}
                for role,value in objects.items():
                    path=directory/role;torch.save(value,path);payloads[role]=path
                payloads.update(extras)
                # Warmstart archives must be present in EVERY commit's payload:
                # restore reads exactly one manifest (LocalCheckpointStore.restore),
                # so a resume from a later commit would otherwise find no
                # warmstart and remap_restored_auxiliary would reject it. Their
                # cost is real (8.5 GB per commit here) and only the anchor
                # semantics justify it.
                for block,path in self.config.get('warmstart_states',{}).items():
                    role='warmstart_block_'+block
                    if role in payloads:continue
                    if digest(path)!=self.config['warmstart_sha256'][block]:raise ValueError('Warmstart changed')
                    payloads[role]=Path(path)
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
        source=make_store({**self.config,'checkpoint':self.config['resume_checkpoint'],'_scope_attempt':False}) if self.config.get('resume_checkpoint') else self.store
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
    if not same_device(device,model.model.embed_tokens.weight.device):
        raise ValueError('Embedding cache device must match baseline embedding residency')
    with torch.no_grad():
        for index,tokens in enumerate(records):
            path=directory/f'{index:06d}.pt'
            if path.exists():
                continue
            ids=torch.tensor([tokens],device=device)
            reference=model.model.embed_tokens(ids)
            student=gather_candidate(embedding_candidate,ids,model.model.embed_tokens.weight)
            # Write beside the target and rename into place. A bare torch.save to
            # `path` leaves a TRUNCATED file if the process dies mid-write, and
            # the `path.exists()` check above would then skip it forever --
            # silently training on a partially-written activation pair. Every
            # other durable artifact in this module (atomic_json,
            # archive_directory, save_candidate) already uses pending+replace;
            # this was the only non-atomic writer.
            pending=path.with_suffix('.pending')
            try:
                torch.save({'teacher':reference.cpu(),'student':student.cpu()},pending)
                os.replace(pending,path)
            finally:
                if pending.exists():
                    pending.unlink()


def gsq_run(model,records,config,output,checkpointer,resume=None,max_steps=None):
    from .gsq import BlockTrainer
    from .gsq_residency import GSQResidency
    residency=GSQResidency(model,config)
    output=Path(output);device=residency.device
    database={} if resume else read_database(config.get('candidate_database',output/'initial-database.json'))
    extras={k:Path(v) for k,v in config.get('candidate_archives',{'initial_candidates':output/'initial-candidates.tar'}).items()}
    progress={'stage':'gsq','block':0,'epoch':0,'sequence':0,'global_step':0}
    if config.get('gsq_epochs',1)<1:
        # `epochs` drives the block loop; zero would empty it and export every
        # block untrained. `checkpoint_plan.py` already treats >=1 as an invariant,
        # so refuse here rather than commit a block that nothing trained.
        raise ValueError(f"gsq_epochs must be at least 1, got {config.get('gsq_epochs')}")
    if resume:
        progress=torch.load(Path(resume)/'progress',weights_only=True)
        if progress['stage']!='gsq':
            raise ValueError('Cannot resume wrong stage')
        # A block below the resume point was trained by an EARLIER process and this
        # one will never train it again -- the loop below starts at
        # `progress['block']`, and the only record of those blocks is the candidate
        # archive each snapshot carries. `initial_candidates` supplies every module
        # name as well, so a snapshot missing one of those archives still yields a
        # complete-looking database and the run ends `completed` with that block
        # holding its untrained RTN candidates. Anchors written before the
        # retention fix are exactly this shape, so this is a live hand-off, not a
        # hypothetical: refuse it by name instead of reporting success.
        carried={path.name for path in Path(resume).iterdir()}
        missing=[block for block in range(progress['block'])
                 if f'candidate_block_{block}' not in carried]
        if missing:
            raise ValueError(
                f'Resume snapshot {Path(resume).name} carries no candidate archive for '
                f'block(s) {missing}; resuming it would record those blocks as calibrated '
                f'while holding their untrained initial candidates')
        database,extras=restore_candidates(resume,output)
        extract_verified_archive(extras[f"cache_block_{progress['block']}"],output/f"cache-{progress['block']}")
    else:
        cached_inputs(model,records,output/'cache-0',Q1Candidate(database['model.embed_tokens']),device=residency.cache_device)
    steps_this_run=0
    blocks_this_run=0
    for block_index in range(progress['block'],len(model.model.layers)):
        cache=output/f'cache-{block_index}'
        # Optimizer steps taken in THIS block. A calibrated block must have at
        # least one; see the export verification below.
        steps_in_block=0
        cache_role=f'cache_block_{block_index}'
        extras[cache_role]=archive_directory(cache,output/f'cache-{block_index}.tar')
        with residency.block(block_index):
            trainer=optimizer=scheduler=loss=tensors=teacher=student=state=None
            try:
                trainer=BlockTrainer(model,block_index,config.get('upstream','/opt/upstream'))
                # Upstream trains the quantizer with Lion on a two-group split - logits
                # at lr1=2e-4 with weight_decay 1.0, scales at lr2=1e-4 without - and the
                # port used Adam at lr=0.001 for both. Lion's update is a sign step with
                # the decay folded in, so the two are not interchangeable.
                optimizer=quantizer_optimizer(trainer.named_parameters(),
                                              lr1=config.get('gsq_lr1',2e-4),
                                              lr2=config.get('gsq_lr2',1e-4),
                                              weight_decay=config.get('gsq_weight_decay',1.0))
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
                # A snapshot taken at this block's own end -- its last sequence, or
                # an epoch counter that already reached `epochs` -- is a legitimate
                # resume point: the trainer state it carries is already trained, so
                # the epoch loop below does no work and only the export is left.
                # `steps_in_block` is 0 there and that must not be read as the
                # skipped-gradient corruption the export check exists to catch.
                already_complete=(progress['epoch']>=epochs
                    or (progress['epoch']==epochs-1 and progress['sequence']>=len(records)))
                for epoch in range(progress['epoch'],epochs):
                    first=progress['sequence'] if epoch==progress['epoch'] else 0
                    for index in range(first,len(records)):
                        with measure(output, 'gsq', 'update_with_input_load', device, block=block_index, kind=model.config.layer_types[block_index], tokens=len(records[index])):
                            tensors=torch.load(cache/f'{index:06d}.pt',weights_only=True)
                            optimizer.zero_grad(set_to_none=True)
                            fraction=(epoch*len(records)+index)/max(1,epochs*len(records)-1)
                            # Upstream's schedules, annealed linearly per optimizer step
                            # as its trainer does (src/config.py: temperature [2, 0.05],
                            # scale [100, 500]). The logit scale was previously never
                            # passed, so kappa stayed at 1.0 for every update.
                            # The input is the drifted composed stream that this
                            # block will actually see at inference; the target is
                            # the unquantized block applied to `teacher`, the
                            # original stream's input. That pairing is deliberate -
                            # see BlockTrainer.forward - and it is the only form
                            # whose gradient opposes the drift measured below.
                            loss=trainer(tensors['student'].to(device),
                                         tensors['teacher'].to(device),
                                         temperature=2.-1.95*fraction,scale=100.+400.*fraction)
                            loss.backward()
                            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainer.parameters()):
                                # Fail loudly. A skip-and-continue guard was tried here
                                # and REMOVED: it carried the run past block 3 while
                                # every step was skipped, so the block exported its
                                # warmstart back out unchanged -- byte-identical
                                # hard weights, i.e. an untrained block committed
                                # under a name that says it was calibrated. A crash
                                # that names the block is strictly better than a
                                # silently untrained one, which is the failure mode
                                # this port's identity checks exist to prevent.
                                #
                                # Measured, so the next reader does not repeat it:
                                # the loss here is non-finite on EVERY step of the
                                # affected block (loss=null in the event log), never
                                # intermittently, so there is nothing to retry.
                                raise FloatingPointError(
                                    f'Nonfinite/missing GSQ gradient at block {block_index} '
                                    f'sequence {index} (loss={float(loss) if torch.isfinite(loss) else "nonfinite"}); '
                                    f'no optimizer step was taken, so restart from the last '
                                    f'committed checkpoint rather than trusting this block')
                            optimizer.step();project_scale_format(trainer);scheduler.step()
                        progress.update(block=block_index,epoch=epoch,sequence=index+1,
                                        global_step=progress['global_step']+1,last_loss=loss.item())
                        steps_this_run+=1
                        steps_in_block+=1
                        print(json.dumps(progress),flush=True)
                        stopping=should_stop(config,steps_this_run,max_steps)
                        checkpointer.save(trainer,optimizer,scheduler,progress,extras,force=stopping)
                        if stopping:
                            return {'status':'checkpointed_stop','progress':progress}
                    progress.update(epoch=epoch+1,sequence=0)
                block_output=output/f'block-{block_index}'
                # A calibrated block must have been optimized, and its persisted
                # candidate must equal the state that was optimized.
                #
                # This is the check a skip-and-continue gradient guard defeated:
                # with every step skipped, `export` wrote the loaded warmstart
                # straight back out with byte-identical hard weights and the run
                # recorded the block as calibrated. Nothing downstream could tell,
                # because the export is a valid Q1 candidate of the right shape.
                # A block that took no optimizer step is not calibrated, whatever
                # its export looks like.
                #
                # The exception is a resume that lands on a snapshot taken at this
                # block's own end: that state is already trained, the epoch loop
                # correctly does no work, and refusing it made a legitimate resume
                # point unusable -- which is how this guard broke the next-phase
                # resume path in `test_final_block_and_next_phase_checkpoint_resume`.
                if steps_in_block==0 and not already_complete:
                    raise ValueError(
                        f'Block {block_index} completed with zero optimizer steps; '
                        f'refusing to record it as calibrated')
                database.update(trainer.export(block_output))
                for _name,_qz in zip(trainer.names,trainer.quantizers):
                    _path=block_output/(_name+'.safetensors')
                    if not _path.exists():
                        raise ValueError(f'Block {block_index} export missing {_name}')
                    _expected=_qz.get_hard_weights()[0]
                    # Compare on a single device. get_hard_weights() returns on
                    # the execution device while the candidate rows are read on
                    # CPU, and torch.equal refuses cross-device operands.
                    _actual=Q1Candidate(_path).rows(0,_expected.shape[0],torch.device('cpu'),torch.float32)
                    if not torch.equal(_actual,_expected.detach().to(device='cpu',dtype=torch.float32)):
                        raise ValueError(
                            f'Block {block_index} export disagrees with the trained '
                            f'quantizer at {_name}; the persisted candidate is not the '
                            f'state that was optimized')
                extras[f'candidate_block_{block_index}']=archive_directory(block_output,output/f'block-{block_index}.tar')
                # Every completed block's archive stays in `extras`, and so in
                # every later payload, because it is the only record of that
                # block's trained weights: `database` is rebuilt on resume by
                # `restore_candidates` from the archives a snapshot carries, and
                # the block loop re-exports only the block it is training. This
                # code previously dropped every superseded archive here, which on a
                # per-block restart silently discarded all of them -- measured on a
                # two-layer model, a blocked_stop sweep left layer 0 holding its
                # untrained RTN initial candidates while the run still reported
                # `completed`, and only the final run's own layer matched a
                # single-process reference. The store is content addressed, so
                # retention costs one stored copy per block (54 MB here) plus a
                # re-link per publish; dropping them costs the calibration.
                # The warmstarts are not stripped here: `Checkpointer.save` re-adds
                # every `warmstart_block_<block>` from config on every publish,
                # because a resume may land on any manifest. A pop here cannot take
                # effect, so it is gone rather than left to mislead.
                next_cache=output/f'cache-{block_index+1}';next_cache.mkdir(exist_ok=True)
                drift_ratios=[];drift_cosines=[]
                drift_stride=max(1,len(records)//DRIFT_SAMPLES)
                with measure(output, 'gsq', 'cache_propagation', device, block=block_index, kind=model.config.layer_types[block_index], tokens=sum(map(len,records)), tokens_squared=sum(len(r)**2 for r in records), sequences=len(records)):
                    with torch.no_grad():
                        for index in range(len(records)):
                            tensors=torch.load(cache/f'{index:06d}.pt',weights_only=True)
                            teacher=run_block(model,block_index,tensors['teacher'].to(device))
                            student=trainer.hard_forward(tensors['student'].to(device))
                            torch.save({'teacher':teacher.cpu(),'student':student.cpu()},next_cache/f'{index:06d}.pt')
                            # The composed stream this block hands to the next one,
                            # against the unquantized stream: the drift is what
                            # compounds across blocks, and both tensors are already
                            # in hand here, so measuring it costs two reductions.
                            # Sampling is strided rather than leading, because the
                            # corpus is processed largest-first.
                            if index%drift_stride==0:
                                flat_student=student.float().flatten();flat_teacher=teacher.float().flatten()
                                drift_ratios.append((flat_student.norm()/flat_teacher.norm().clamp_min(1e-12)).item())
                                drift_cosines.append(torch.nn.functional.cosine_similarity(
                                    flat_student,flat_teacher,dim=0).item())
                drift={'block':block_index,'samples':len(drift_ratios),
                       'norm_ratio':sum(drift_ratios)/len(drift_ratios),
                       'norm_ratio_max':max(drift_ratios),
                       'cosine':sum(drift_cosines)/len(drift_cosines),
                       'cosine_min':min(drift_cosines)}
                drift_path=output/'drift.json'
                drift_history=json.loads(drift_path.read_text()) if drift_path.exists() else []
                previous=(drift_history[-1]['norm_ratio'] if drift_history
                          else progress.get('drift_norm_ratio'))
                drift['previous_norm_ratio']=previous
                drift['growth']=(drift['norm_ratio']/previous) if previous else None
                drift_history.append(drift);atomic_json(drift_path,drift_history)
                # `drift.json` is attempt-scoped and `main` demands a fresh output
                # directory per attempt, so after any resume it is empty and the growth
                # bound would go unevaluated on that block - precisely the block where a
                # 2.94x jump sits below the absolute backstop. The baseline therefore
                # travels in `progress`, which every snapshot persists and every resume
                # restores (an adversarial review demonstrated the gap on the tiny model:
                # one sweep stops with `drift`, the same sweep split across two attempts
                # stops with `block_limit` and `growth: None`).
                progress['drift_norm_ratio']=drift['norm_ratio']
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
        blocks_this_run+=1
        # Stop here, but only after publishing an anchor the NEXT block can
        # resume from. The phase-start commit at the top of the next iteration is
        # what `--resume` reads, so stopping without one leaves the next process
        # with nothing to name (measured: a stop taken BEFORE the block committed
        # only b<N> and the driver's --resume pointed at a nonexistent b<N+1>).
        #
        # `gsq_max_drift_growth` stops the chain when the composed stream grows by
        # more than that factor from one block to the next. Unbounded growth is
        # the measured failure of this port: one quantized block already inflates
        # the stream 2.94x, the incoming loss roughly doubled every block through
        # block 8, and the result was no better than untrained RTN. Stopping at a
        # block boundary keeps every committed block valid; letting it run on
        # spends hours producing a chain that cannot be exported.
        #
        # The anchor is written now, from the state as it stands at the end of
        # this block: `progress` advanced to the next block, and the trainer state
        # that block would have started from. On resume, gsq_run enters the loop
        # at progress['block'] and re-runs this same phase-start save, so writing
        # it here is idempotent -- same state, same snapshot, immutability holds.
        drift_limit=config.get('gsq_max_drift_growth')
        ratio_limit=config.get('gsq_max_drift_ratio')
        # The two limits above bound magnitude only. The 2026-09-20 sweep held its
        # magnitude near 0.9 while its cosine fell from 0.893 to 0.700 across
        # blocks 11-21, so neither limit could fire on the failure this port
        # actually produced. Bound the direction too, or the guard watches a
        # composition collapse and reports nothing.
        cosine_limit=config.get('gsq_min_drift_cosine')
        drift_stop=((drift_limit is not None and drift.get('growth') is not None
                     and drift['growth']>float(drift_limit))
                    or (ratio_limit is not None and drift['norm_ratio']>float(ratio_limit))
                    or (cosine_limit is not None and drift['cosine']<float(cosine_limit)))
        if drift_stop:
            drift['stopped_by']=('cosine' if (cosine_limit is not None
                and drift['cosine']<float(cosine_limit))
                else 'ratio' if (ratio_limit is not None
                and drift['norm_ratio']>float(ratio_limit)) else 'growth')
            atomic_json(drift_path,drift_history)
        if (should_stop_block(config,blocks_this_run) or drift_stop) and block_index+1<len(model.model.layers):
            with residency.block(block_index+1):
                next_trainer=BlockTrainer(model,block_index+1,config.get('upstream','/opt/upstream'))
                next_optimizer=quantizer_optimizer(next_trainer.named_parameters(),
                                                   lr1=config.get('gsq_lr1',2e-4),
                                                   lr2=config.get('gsq_lr2',1e-4),
                                                   weight_decay=config.get('gsq_weight_decay',1.0))
                next_epochs=config.get('gsq_epochs',1)
                next_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                    next_optimizer,max(1,next_epochs*len(records)))
                warm=config.get('warmstart_states',{}).get(str(block_index+1))
                if warm:
                    if digest(warm)!=config['warmstart_sha256'][str(block_index+1)]:raise ValueError('Warmstart changed')
                    state=torch.load(warm,map_location='cpu',weights_only=True)
                    if state['block']!=block_index+1 or state['updates']!=1:raise ValueError('Invalid GSQ warmstart')
                    next_trainer.load_state_dict(state['solver'])
                    next_optimizer.load_state_dict(state['optimizer'])
                anchor={'stage':'gsq','block':block_index+1,'epoch':0,'sequence':0,
                        'global_step':progress['global_step'],'last_loss':progress.get('last_loss')}
                # `candidate_block_*` is deliberately NOT filtered out here: the
                # process that resumes from this anchor rebuilds `database` from
                # the archives the snapshot carries, so the completed blocks'
                # candidates have to be in it.
                next_extras={k:v for k,v in extras.items()
                             if not k.startswith(('cache_block_','warmstart_block_'))}
                next_cache=output/f'cache-{block_index+1}'
                next_extras[f'cache_block_{block_index+1}']=archive_directory(
                    next_cache,output/f'cache-{block_index+1}.tar')
                checkpointer.save(next_trainer,next_optimizer,next_scheduler,anchor,next_extras,force=True)
                residency.release_training_state(next_trainer,next_optimizer)
                next_trainer=None
            return {'status':'blocked_stop','blocks_completed':blocks_this_run,
                    'next_block':block_index+1,'progress':anchor,
                    'stop_reason':'drift' if drift_stop else 'block_limit','drift':drift}
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
    # 0.1 is upstream's default and what the pinned driver passes
    # (rco/rco_search_quant.py:63 `--lr type=float, default=0.1`;
    # rco/scripts/run_search_quant.sh:27 `--lr 0.1`). The port's 0.01 had no
    # recorded justification.
    optimizer=torch.optim.Adam([trainer.alpha],lr=config.get('rco_lr',.1))
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
                # Upstream anneals tau exponentially, not linearly:
                # `tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)`
                # over `progress = step / max(n_steps - 1, 1)`
                # (rco/src/search/quant.py:654-655 and :685), with tau_init 1.0
                # and tau_min 0.01 (rco_search_quant.py:64-65, pinned by
                # scripts/run_search_quant.sh:28) -- i.e. .01**fraction here.
                report=trainer.step(torch.tensor([records[index]],device=device),optimizer,
                                    temperature=max(.01,.01**fraction))
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
        from .device_policy import empty_cache as _empty_cache
        _empty_cache(device)
        if resume:
            cache=output/'final-cache';extract_verified_archive(extras['cache_final'],cache)
        else:
            cache=Path(config['final_cache'])
            extras['cache_final']=archive_directory(cache,output/'cache-final.tar')
        trainer=HeadTrainer(model.lm_head.weight)
    # The embedding stage trains a GSQ quantizer and so takes upstream's optimiser too;
    # the head stage is this port's own full-KL objective with no upstream analogue and
    # keeps Adam.
    if stage=='embedding':
        optimizer=quantizer_optimizer(trainer.named_parameters(),
                                      lr1=config.get('gsq_lr1',2e-4),
                                      lr2=config.get('gsq_lr2',1e-4),
                                      weight_decay=config.get('gsq_weight_decay',1.0))
    else:
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
                if stage=='embedding':
                    # A GSQ quantizer, so it takes upstream's schedule too.
                    loss=trainer(torch.tensor([records[index]],device=device),
                                 temperature=2.-1.95*fraction,scale=100.+400.*fraction)
                else:
                    # The head stage is this port's own full-KL objective rather
                    # than an upstream quantizer, so it keeps its own temperature
                    # range and has no logit scale.
                    temperature=1.-.9*fraction
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
    records=corpus_inputs(config['corpus'],hold_out_stride=config.get('held_out_stride',HELD_OUT_STRIDE))
    if config.get('largest_first',False):records.sort(key=len,reverse=True)
    if args.max_steps is not None and args.max_steps<=0:raise ValueError('Positive step limit required')
    if args.dry_run:
        print(json.dumps({'status':'inputs_validated','invocations':len(records),
                          'total_tokens':sum(map(len,records)),'gpu_validated':False}));return
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'frozen-config.json',config)
    # Resolve the compute device ONCE, before the determinism policy and the
    # BF16 gate. Previously the policy configured an internally auto-resolved
    # backend while the gate probed CUDA whenever CUDA was visible and the stage
    # then computed on the auto-resolved device -- three independent answers on a
    # host that has both MPS and CUDA. `gsq_residency.execution_device` remains
    # the authority for the gsq stage and honours an explicit
    # `gsq_execution_device`; every other stage takes the auto-resolved bump.
    from .device_policy import resolve_device as _resolve
    _default_accelerator=torch.device(_resolve('auto'))
    compute_device=(execution_device(config,_default_accelerator) if args.stage=='gsq' else _default_accelerator)
    from .reproducibility import configure
    execution_policy=configure(config.get('seed',42),compute_device)
    atomic_json(output/'execution-policy-report.json',execution_policy)
    # Device capability gate. Previously CUDA-only, with the rationale that a
    # production Qwen27B job needs a BF16 accelerator. That rationale still holds;
    # what changed is that MPS is now such an accelerator on this machine, so the
    # gate tests the *capability* rather than a vendor.
    #
    # It remains strict about BF16: the whole pipeline's numerics (BF16 rounding
    # boundaries, FP32 logits/scale reductions, the post-Adam scale grid) were
    # characterized under BF16, and silently running FP16 would change results.
    from .device_policy import (
        is_available as _available,
        supports_bf16 as _supports_bf16,
    )
    if not _available(compute_device):
        raise RuntimeError(
            'Qwen27B requires a BF16 accelerator (CUDA or MPS); CPU is only for '
            'tiny tests'
        )
    # BF16 is required on the device the run will ACTUALLY use, and it is
    # *probed* rather than assumed. Probing CUDA because CUDA happens to be
    # present validated a device this run would not touch.
    if compute_device.type != 'cpu' and not _supports_bf16(compute_device):
        raise RuntimeError(
            'The %s device this run selects does not support BF16 on this build; '
            'the pipeline requires BF16 (see device_policy.supports_bf16, which '
            'probes rather than version-sniffs)' % compute_device.type
        )
    if compute_device.type=='cpu':
        raise RuntimeError(
            'Actual Qwen27B GSQ requires an accelerator (CUDA or MPS); CPU '
            'residency tests use tiny models directly'
        )
    load_device=torch.device('cpu') if mode==BLOCK_CPU_OFFLOAD else compute_device
    with measure(output, args.stage, 'stage_total', compute_device):
        with measure(output, args.stage, 'model_load', compute_device):
            model=load_original(config['model_dir'],load_device)
        # MPS: install the determinism policy only now. It cannot make Metal
        # deterministic (see `MPS_CAVEATS`), but it does change which kernels the
        # dispatcher selects, and installing it before this load made every attempt's
        # first `full_attention` update non-finite -- block 3 fails at sequence 0 here
        # and trains all 87 steps with `configure` stubbed. See the bisection in
        # docs/PORT_VALIDATION.md and solver/tools/bisect_main.py.
        if compute_device.type=='mps' and execution_policy.get('deterministic_algorithms'):
            torch.use_deterministic_algorithms(True,warn_only=bool(execution_policy.get('warn_only')))
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
        # Memory accounting, reported for what it actually is.
        #
        # `cuda_peak_*` is a PEAK and only CUDA maintains peaks. On MPS there is
        # no peak counter at all, so writing MPS *current* allocation into a
        # field named `cuda_peak_*` would understate memory and mislabel the
        # device in one step. The honest shape is: always state the device, only
        # use the `peak` keys when a real peak exists, and record current
        # allocation under its own name otherwise.
        from .device_policy import memory_stats as _memory_stats, peak_memory_stats as _peak_stats
        current_alloc, current_res = _memory_stats(compute_device)
        result['memory_device']=str(compute_device)
        result['memory_current_allocated_bytes']=int(current_alloc * 1024**3)
        result['memory_current_reserved_bytes']=int(current_res * 1024**3)
        _peak = _peak_stats(compute_device)
        if _peak is not None:
            result['cuda_peak_allocated_bytes']=int(_peak[0] * 1024**3)
            result['cuda_peak_reserved_bytes']=int(_peak[1] * 1024**3)
            result['memory_peak_available']=True
        else:
            # No peak on this backend. Null says so; a number would not.
            result['memory_peak_available']=False
            result['memory_peak_absent_reason']=(
                'this backend exposes no allocator peak counter (only CUDA does); '
                'memory_current_* is a post-stage sample, not a high-water mark'
            )
        result['calibration_sha256']=config['identity']['calibration_sha256']
        result['runtime_sha256']=config['identity']['runtime_sha256']
        atomic_json(output/(args.stage+'-report.json'),result)


if __name__=='__main__':main()
