"""GPU validation from cached images/inputs; no installs, pulls, or downloads."""
import hashlib,json,math,os,re,shutil,stat,subprocess,time,urllib.parse,urllib.request
from datetime import datetime
from pathlib import Path
try:import gpu_common as common
except ImportError:from cloud_gpu import bootstrap as common
MOUNT=Path('/mnt/zimfo-inputs')
ROOT=Path('/opt/zimfo-gsq24')


def read_json(path,max_bytes=4*1024**2):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>max_bytes:raise ValueError('Missing or unsafe bounded report')
    return json.loads(path.read_text())


def validate_report(config,report):
    ready=config['ready'];runtime=ready['production_image'].split('@sha256:')[1]
    expected={'status':'passed','actual_24gb_validated':True,'numerical_parity_passed':True,
       'cold_checkpoint_replay_passed':True,'migration_applied':False,'production_progress_advanced':False,
       'production_gsq_loop_cuda_validated':True,'full_corpus_gsq_completed':False,
       'runtime_sha256':runtime,'inherited_gsq_updates':174,'inherited_embedding_updates':87,'complete_job_eta_seconds':None}
    if not isinstance(report,dict) or any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete scoped GSQ24 validation')
    workers=report.get('workers',[])
    if len(workers)!=6 or {(v.get('block'),v.get('phase')) for v in workers}!={(b,p) for b in (2,3) for p in ('reference','streamed','cold')}:
        raise ValueError('Need exactly all six longest-sequence worker results')
    for worker in workers:
        block=worker['block'];phase=worker['phase'];cuda=worker.get('cuda',{});total=cuda.get('total_memory_bytes')
        expected_worker={'status':'passed','runtime_sha256':runtime,'source_receipt_sha256':ready['source_receipt_sha256'],
            'kind':{2:'linear_attention',3:'full_attention'}[block],'sequence_tokens':4673,
            'diagnostic_prefix':block==3,'production_progress_advanced':False}
        if any(worker.get(k)!=v for k,v in expected_worker.items()):raise ValueError('Worker source/runtime/scope mismatch')
        if (cuda.get('available') is not True or 'L4' not in cuda.get('device','') or cuda.get('capability')!=[8,9]
                or type(total) is not int or not 20*1024**3<=total<=25*1024**3
                or any(type(cuda.get(k)) is not int or not 0<cuda[k]<=total for k in ('peak_allocated_bytes','peak_reserved_bytes'))):
            raise ValueError('Actual single-L4 memory/capability evidence missing')
        checks=worker.get('checks',[])
        if [v.get('step') for v in checks]!=([2] if phase=='cold' else [1,2]):raise ValueError('Worker update/replay missing')
        for check in checks:
            if check.get('all_gradients_and_states_compared') is not (phase!='reference'):raise ValueError('Numerical state/gradient comparison missing')
            for key in ('loss','update_seconds','propagation_seconds'):
                value=check.get(key)
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid measured worker result')
        projections=worker.get('projections',[]);names=[v.get('name') for v in projections]
        if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):raise ValueError('Missing projection inventory')
        candidates=worker.get('candidate_sha256',{})
        if set(candidates)!={f'model.layers.{block}.{name}' for name in names}:raise ValueError('Missing candidate export fingerprints')
        for candidate in candidates.values():
            if set(candidate.get('tensors',{}))!={'codes','scales'} or not candidate.get('metadata'):raise ValueError('Missing candidate tensor evidence')
            for tensor in candidate['tensors'].values():
                if not re.fullmatch('[a-f0-9]{64}',tensor.get('sha256','')) or not tensor.get('shape') or not tensor.get('dtype'):
                    raise ValueError('Invalid candidate tensor fingerprint')
    for block in (2,3):
        selected=[v for v in workers if v['block']==block];reference=next(v for v in selected if v['phase']=='reference')
        if any(v['candidate_sha256']!=reference['candidate_sha256'] or v['projections']!=reference['projections'] for v in selected):
            raise ValueError('Candidate/projection mismatch between reference,streamed,cold')
    canary=report.get('production_canary',{})
    if any(canary.get(key)!=value for key,value in {
        'status':'passed','new_diagnostic_updates':2,'checkpoint_verified':True,'diagnostic_only':True,
        'production_progress_advanced':False,'runtime_sha256':runtime,
        'source_receipt_sha256':ready['source_receipt_sha256']}.items()):
        raise ValueError('Actual production GSQ run-loop diagnostic proof missing')
    identity=canary.get('identity',{});source_identity=ready['source_validation']['source_identity']
    if (canary.get('source_commit')!=ready['source_validation']['source_commit']
            or identity==source_identity or identity.get('runtime_sha256')!=runtime
            or any(identity.get(key)!=source_identity[key] for key in ('baseline_repo','baseline_revision','calibration_sha256'))
            or any(canary.get('progress',{}).get(key)!=value for key,value in
                {'stage':'gsq','block':2,'epoch':0,'global_step':176,'sequence':2}.items())):
        raise ValueError('Canary source/cursor/performance evidence missing')
    groups=canary.get('performance',{}).get('groups',{})
    if not isinstance(groups,dict) or any(not isinstance(g,dict) for g in groups.values()):
        raise ValueError('Canary performance groups malformed')
    updates=[g for g in groups.values() if g.get('stage')=='gsq' and g.get('operation')=='update_with_input_load']
    if (not updates or any(g.get('status')!='completed' or type(g.get('count')) is not int or g['count']<=0
            or type(g.get('seconds')) not in (int,float) or not math.isfinite(g['seconds']) or g['seconds']<=0
            or g.get('dimensions',{}).get('block')!=2 for g in updates)
            or sum(g['count'] for g in updates)!=2):
        raise ValueError('Need two finite measured actual GSQ updates')
    cuda=canary.get('cuda',{});total=cuda.get('total_memory_bytes')
    if (cuda.get('available') is not True or 'L4' not in cuda.get('device','') or cuda.get('capability')!=[8,9]
            or type(total) is not int or not 20*1024**3<=total<=25*1024**3
            or any(type(cuda.get(k)) is not int or not 0<cuda[k]<=total for k in ('peak_allocated_bytes','peak_reserved_bytes'))):
        raise ValueError('Missing actual GSQ run-loop 24GB memory evidence')
    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
        if report.get(key)!=max(v['cuda'][key] for v in [*workers,canary]):raise ValueError('Aggregate peak memory differs from workers')
    return report


def validate_terminal(config,report):
    validate_report(config,report.get('gsq24_report'))
    probe=report.get('cuda_probe',{});hardware=report.get('hardware',[])
    if (report.get('runtime_proof')!=config['ready']['runtime_proof'] or probe.get('status')!='passed'
            or probe.get('capability')!=[8,9] or probe.get('bf16_backward') is not True
            or not str(probe.get('cuda','')).startswith('13.') or len(hardware)!=1
            or 'L4' not in hardware[0] or len(hardware[0].split(','))!=3
            or hardware[0].split(',')[2].strip()!=config['ready']['runtime_proof']['driver']):
        raise ValueError('Terminal driver/runtime/kernel probe missing')
    return report


def docker_command(config,harness_path,output):
    ready=config['ready']
    return ['docker','run','--name',config['run_id'],'--rm','--pull=never','--gpus=all','--network=none',
       '--cap-drop=ALL','--security-opt=no-new-privileges','--shm-size=8g',
       '--mount',f'type=bind,src={MOUNT}/prepared,dst=/inputs/prepared,readonly',
       '--mount',f'type=bind,src={MOUNT}/gsq24-source,dst=/inputs/gsq24-source,readonly',
       '--mount',f'type=bind,src={harness_path},dst=/config.json,readonly',
       '--mount',f'type=bind,src={output},dst=/output',ready['production_image'],
       'python','-m','solver.gsq24','--config','/config.json','--output','/output/results']


def emit_progress(root,summary):
    """Expose bounded diagnostic progress locally, without periodic GCS writes."""
    try:
        report=read_json(Path(root)/'results/gsq24-report.json')
        workers=report.get('workers',[]) if isinstance(report,dict) else None
        if isinstance(workers,list) and len(workers)<=6 and all(isinstance(w,dict) and w.get('status')=='passed' for w in workers):
            summary['completed_validation_workers']=len(workers)
            if report.get('status') in ('running','passed','failed'):summary['validation_status']=report['status']
    except (OSError,ValueError,TypeError):pass
    common.serial_summary(summary)


def execute(config):
    if common.metadata('instance/machine-type').decode().split('/')[-1]!='g2-standard-32':raise ValueError('Wrong validation machine')
    ready=config['ready'];common.verify_prepared_driver(ready['runtime_proof'])
    if MOUNT.is_symlink():raise ValueError('Prepared mount must be ordinary')
    if not MOUNT.is_mount():
        if common.command(['blkid','-s','LABEL','-o','value',common.DEVICE])!='zimfo-inputs':raise ValueError('Input disk label changed')
        MOUNT.mkdir(exist_ok=True);common.command(['mount','-o','noatime',common.DEVICE,str(MOUNT)])
    mounted=common.command(['findmnt','--noheadings','--output','SOURCE','--target',str(MOUNT)])
    attached=os.stat(common.DEVICE);observed=os.stat(mounted)
    if not stat.S_ISBLK(attached.st_mode) or not stat.S_ISBLK(observed.st_mode) or attached.st_rdev!=observed.st_rdev:
        raise ValueError('Mount does not refer to dedicated prepared input device')
    if read_json(MOUNT/'prepared/restore-validation.json')!=ready['restore_validation']:raise ValueError('Prepared input validation changed')
    if read_json(MOUNT/'gsq24-source/source-validation.json')!=ready['source_validation']:raise ValueError('Prepared source validation changed')
    if hashlib.sha256((MOUNT/'gsq24-source/source-receipt.json').read_bytes()).hexdigest()!=ready['source_receipt_sha256']:
        raise ValueError('Prepared source receipt changed')
    if shutil.disk_usage(MOUNT).free<110*1024**3:raise ValueError('Insufficient live validation output headroom')
    image=ready['production_image'];cached=json.loads(common.command(['docker','image','inspect',image]))
    if len(cached)!=1 or cached[0]['Architecture']!='amd64' or image not in cached[0]['RepoDigests']:raise ValueError('Expected image is not cached')
    hardware=common.command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits']).splitlines()
    if (len(hardware)!=1 or 'L4' not in hardware[0] or not 20000<=float(hardware[0].split(',')[1])<=26000
            or hardware[0].split(',')[2].strip()!=ready['runtime_proof']['driver']):
        raise ValueError('Expected exactly one 24GB L4')
    memory_kib=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemTotal:')))
    if memory_kib<96*1024**2:raise ValueError('Insufficient host memory for frozen BF16 baseline')
    # Real BF16 CUDA kernel/backward checks detect unsupported Ada binaries or
    # driver/runtime mismatch before a full model is loaded.
    probe_name=config['run_id']+'-probe'
    try:
        probe=common.command(['docker','run','--name',probe_name,'--rm','--pull=never','--gpus=all','--network=none',image,'python','-c',
      "import json,torch;assert torch.cuda.device_count()==1;assert torch.cuda.get_device_capability()==(8,9);assert torch.version.cuda.startswith('13.');assert torch.cuda.is_bf16_supported();x=torch.ones((256,256),device='cuda',dtype=torch.bfloat16,requires_grad=True);y=(x@x).float().mean();y.backward();torch.cuda.synchronize();assert torch.isfinite(y) and torch.isfinite(x.grad).all();print(json.dumps({'status':'passed','capability':[8,9],'cuda':torch.version.cuda,'bf16_backward':True}))"])
    finally:
        try:subprocess.run(['docker','stop','--time=5',probe_name],capture_output=True,timeout=15,check=False)
        except (OSError,subprocess.TimeoutExpired):pass
    root=MOUNT/'jobs'/config['run_id'];root.mkdir(parents=True,exist_ok=False)
    harness=dict(config['harness']);hard=datetime.fromisoformat(config['absolute_deadline']).timestamp()
    remaining=int(hard-time.time())
    if remaining<900:raise ValueError('Too little approved time remains')
    harness['deadline_unix']=hard-420
    harness_path=root/'harness-config.json';harness_path.write_text(json.dumps(harness,indent=2))
    try:
        result=common.run_solver(docker_command(config,harness_path,root),root,timeout=remaining-300,
                                 emit=lambda summary:emit_progress(root,summary))
    finally:
        try:subprocess.run(['docker','stop','--time=45',config['run_id']],capture_output=True,timeout=55,check=False)
        except (OSError,subprocess.TimeoutExpired):pass
    report=read_json(root/'results/gsq24-report.json')
    if result.returncode!=0:raise RuntimeError('GSQ24 diagnostic failed; retained detailed reports')
    validate_report(config,report)
    return {'status':'validated_gsq24','gsq24_report':report,'hardware':hardware,'cuda_probe':json.loads(probe),'runtime_proof':ready['runtime_proof']}


def main():
    config=read_json(ROOT/'config.json');body={k:v for k,v in config.items() if k!='config_sha256'}
    if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=config['config_sha256']:raise ValueError('Configuration changed')
    started=time.monotonic()
    try:report=execute(config)
    except BaseException as error:
        report={'status':'failed','error':type(error).__name__+': '+str(error)}
        path=MOUNT/'jobs'/config['run_id']/'results/gsq24-report.json'
        if path.is_file() and not path.is_symlink() and path.stat().st_size<=4*1024**2:report['gsq24_report']=read_json(path)
    report.update(run_id=config['run_id'],config_sha256=config['config_sha256'],elapsed_seconds=time.monotonic()-started)
    # Keep a bounded useful failure tail in the same single terminal object.
    log=MOUNT/'jobs'/config['run_id']/'solver.log'
    if log.is_file() and not log.is_symlink():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-128*1024));report['solver_log_tail']=stream.read(128*1024).decode(errors='replace')
    destination=(MOUNT if MOUNT.is_mount() else ROOT)/(config['run_id']+'-result.json')
    with destination.open('x') as stream:json.dump(report,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    common.command(['sync']);print(json.dumps({'status':report['status'],'run_id':config['run_id'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    common.publish(config,report)


if __name__=='__main__':main()
