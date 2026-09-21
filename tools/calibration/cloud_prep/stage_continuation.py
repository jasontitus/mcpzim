"""Offline bundle, CPU staging, and operator-verified continuation receipt.

No provisioning, IAM changes, model work, installs, or uploads. CPU staging writes
only a provisional receipt: the restricted worker cannot inspect Compute disk
IDs. Finalization uses the operator's existing gcloud login for read-only instance
and disk inspection, binding the same CPU instance ID and provisional-file hash.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request

PROJECT='tiltastech-zimfo'
ZONE='us-central1-b'
MOUNT=Path('/mnt/zimfo-inputs')
DEVICE='/dev/disk/by-id/google-zimfo-inputs'
MAX_FILE=2*1024**2
MAX_BUNDLE=32*1024**2
CPU_TYPES={'c4-standard-2','n4-standard-2','n4-standard-4'}


def sha(data): return hashlib.sha256(data).hexdigest()
def encoded(value): return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def descriptor(data): return {'sha256':sha(data),'bytes':len(data)}


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_FILE:
        raise ValueError('Missing, unsafe or oversized evidence')
    data=path.read_bytes()
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('Expected evidence object')
    return value,data


def relative(name):
    path=Path(name)
    if path.is_absolute() or not name or any(v in ('..','.') for v in name.split('/')):
        raise ValueError('Unsafe bundle evidence name')
    return path


def validate_config(config):
    body={k:v for k,v in config.items() if k!='config_sha256'}
    if config.get('config_sha256')!=sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode()):
        raise ValueError('Launch configuration changed')
    if config.get('mode')!='continuation' or config['ready'].get('project')!=PROJECT or config['ready'].get('zone')!=ZONE:
        raise ValueError('Expected reviewed continuation configuration')
    c=config['continuation'];ready=config['ready']
    if (not re.fullmatch(r'/mnt/zimfo-inputs/continuation-plans/[a-zA-Z0-9_-]+',c['plan_path'])
            or c['runtime_image']!=ready['production_image']
            or not re.fullmatch(r'us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/[a-zA-Z0-9_./-]+@sha256:[a-f0-9]{64}',c['runtime_image'])):
        raise ValueError('Changed staged destination/runtime')
    disk=ready['data_disk']
    if (not str(disk['id']).isdigit() or not re.fullmatch(r'zimfo-prep-[a-f0-9]{12}-inputs',disk['name'])
            or not disk['self_link'].endswith(f'/projects/{PROJECT}/zones/{ZONE}/disks/{disk["name"]}')):
        raise ValueError('Invalid retained disk binding')
    return c


def verify_bundle_files(files):
    config=json.loads(files['config.json']);c=validate_config(config)
    plan=json.loads(files['continuation-plan.json'])
    if sha(files['continuation-plan.json'])!=c['plan_sha256'] or plan.get('files')!=c['files']:
        raise ValueError('Bundle plan/evidence changed')
    if sha(files['continuation.py'])!=c['controller_sha256']:
        raise ValueError('Bundle controller changed')
    expected={'config.json','continuation-plan.json','continuation.py'}
    for name,item in c['files'].items():
        relative(name)
        if name=='continuation-plan.json':raise ValueError('Reserved plan evidence name')
        key='evidence/'+name;expected.add(key)
        if descriptor(files[key])!=item:raise ValueError('Bundle evidence hash/size changed')
    if c.get('checkpoint_mode')=='local-spool':
        expected.add('checkpoint_bridge.py')
        if sha(files['checkpoint_bridge.py'])!=c['publisher_sha256']:
            raise ValueError('Bundle publisher changed')
    if set(files)!=expected:raise ValueError('Unexpected or missing bundle file')
    source=json.loads(files['evidence/reconciled-status.json'])
    if (plan['runtime_image']!=c['runtime_image'] or plan['selected_checkpoint']!=c['source_commit']
            or source['smoke']['checkpoint_resume']['commit']!=c['smoke_commit']
            or source['input_commit_sha256']!=config['ready']['input_commit']['sha256']):
        raise ValueError('Bundle source ancestry changed')
    return config


def bundle(launch_dir, source_plan_dir, destination):
    """Create a deterministic small uploadable archive; no remote calls."""
    launch_dir=Path(launch_dir);source_plan_dir=Path(source_plan_dir).resolve()
    config,raw=read(launch_dir/'config.json');c=validate_config(config)
    files={'config.json':raw}
    for name in ('continuation-plan.json','continuation.py'):
        path=launch_dir/name
        if path.is_symlink() or path.stat().st_size>MAX_FILE:raise ValueError('Unsafe launch file')
        files[name]=path.read_bytes()
    for name in c['files']:
        path=source_plan_dir/relative(name)
        if path.is_symlink() or source_plan_dir not in path.resolve().parents:
            raise ValueError('Unsafe source plan evidence')
        _,files['evidence/'+name]=read(path)
    if c.get('checkpoint_mode')=='local-spool':
        path=launch_dir/'checkpoint_bridge.py'
        if path.is_symlink() or path.stat().st_size>MAX_FILE:raise ValueError('Unsafe publisher')
        files['checkpoint_bridge.py']=path.read_bytes()
    verify_bundle_files(files)
    if sum(map(len,files.values()))>MAX_BUNDLE:raise ValueError('Oversized continuation bundle')
    manifest=encoded({'schema':1,'files':{k:descriptor(v) for k,v in files.items()}})
    destination=Path(destination)
    with destination.open('xb') as output:
        with tarfile.open(fileobj=output,mode='w') as archive:
            for name,data in sorted({**files,'bundle-manifest.json':manifest}.items()):
                entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o600;entry.mtime=0
                archive.addfile(entry,io.BytesIO(data))
        output.flush();os.fsync(output.fileno())
    return {'status':'continuation_bundle_prepared','bundle':str(destination),
            'sha256':sha(destination.read_bytes()),'bytes':destination.stat().st_size,
            'config_sha256':config['config_sha256'],'staging_still_required':True}


def unpack(path, expected_sha256):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>MAX_BUNDLE+1024*1024 or sha(path.read_bytes())!=expected_sha256:
        raise ValueError('Bundle differs from reviewed hash/size')
    files={};total=0
    with tarfile.open(path,mode='r:') as archive:
        for member in archive:
            relative(member.name)
            if not member.isfile() or member.name in files or member.size>MAX_FILE:
                raise ValueError('Unsafe bundle member')
            total+=member.size
            if total>MAX_BUNDLE:raise ValueError('Oversized expanded bundle')
            files[member.name]=archive.extractfile(member).read(MAX_FILE+1)
    manifest=json.loads(files.pop('bundle-manifest.json'))
    if manifest!={'schema':1,'files':{k:descriptor(v) for k,v in files.items()}}:
        raise ValueError('Bundle manifest changed')
    return verify_bundle_files(files),files


def metadata(path):
    req=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/'+path,
                               headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(req,timeout=20) as response:return response.read(65536).decode().strip()


def command(args):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=60).stdout.strip()


def observe_cpu(config):
    """Observe real guest/device/cache state, without fabricating a disk ID."""
    project=metadata('project/project-id');zone=metadata('instance/zone').split('/')[-1]
    instance_id=metadata('instance/id');name=metadata('instance/name')
    machine=metadata('instance/machine-type').split('/')[-1]
    if project!=PROJECT or zone!=ZONE or not instance_id.isdigit() or machine not in CPU_TYPES:
        raise ValueError('Expected this project/zone on an approved CPU-only host')
    if MOUNT.is_symlink() or not MOUNT.is_mount():raise ValueError('Prepared volume must already be mounted')
    mounted=command(['findmnt','--noheadings','--output','SOURCE','--target',str(MOUNT)])
    observed=os.stat(mounted);device=os.stat(DEVICE)
    if not stat.S_ISBLK(device.st_mode) or not stat.S_ISBLK(observed.st_mode) or observed.st_rdev!=device.st_rdev:
        raise ValueError('Mounted volume is not the dedicated input device')
    if command(['blkid','-s','LABEL','-o','value',DEVICE])!='zimfo-inputs':raise ValueError('Prepared filesystem label mismatch')
    validation,_=read(MOUNT/'prepared/restore-validation.json')
    if validation!=config['ready']['restore_validation']:raise ValueError('Prepared input validation changed')
    image=config['ready']['production_image']
    cached=json.loads(command(['docker','image','inspect',image]))
    if len(cached)!=1 or image not in cached[0]['RepoDigests'] or cached[0]['Architecture']!='amd64':
        raise ValueError('Exact runtime is not already cached')
    return {'project':project,'zone':zone,'instance_name':name,'instance_id':instance_id,
            'machine_type':machine,'device_name':'zimfo-inputs','device_path':DEVICE,
            'mounted_source':mounted,'device_major':os.major(device.st_rdev),'device_minor':os.minor(device.st_rdev),
            'runtime_image':image,'prepared_validation_sha256':sha(encoded(validation))}


def atomic_json(path,value):
    path=Path(path)
    if path.exists() or path.is_symlink():raise FileExistsError('Receipt destination must be new')
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('xb') as out:out.write(encoded(value));out.flush();os.fsync(out.fileno())
    # link publishes without overwriting a concurrent writer; both files same fs.
    os.link(temporary,path);temporary.unlink()
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def stage(bundle_path,bundle_sha256,receipt_path,*,observer=observe_cpu):
    config,files=unpack(bundle_path,bundle_sha256);c=config['continuation']
    observation=observer(config)
    required=max(c['recovery_required_free_bytes'],c.get('spool_min_free_bytes',0))
    free=shutil.disk_usage(MOUNT).free
    if free<required+sum(map(len,files.values())):raise ValueError('Insufficient recovery/spool disk headroom')
    target=MOUNT/'continuation-plans'/Path(c['plan_path']).name
    target.parent.mkdir(exist_ok=True)
    if target.parent.is_symlink() or target.exists():raise ValueError('Staged destination must be new and ordinary')
    pending=Path(tempfile.mkdtemp(prefix='.staging-',dir=target.parent))
    try:
        to_stage={'continuation-plan.json':files['continuation-plan.json'],
                  **{name:files['evidence/'+name] for name in c['files']}}
        for name,data in to_stage.items():
            path=pending/relative(name);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
        # Rename preserves an all-or-nothing new plan directory; never edit an
        # existing staged plan to match a changed launcher.
        os.rename(pending,target)
        fd=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except BaseException:
        if pending.exists():shutil.rmtree(pending)
        raise
    free=shutil.disk_usage(MOUNT).free
    provisional={'status':'continuation_staged_pending_operator_disk_verification',
       'bundle_sha256':bundle_sha256,'launch_config_sha256':config['config_sha256'],
       'plan_path':c['plan_path'],'plan_sha256':c['plan_sha256'],'files':c['files'],'free_bytes':free,
       'expected_data_disk':config['ready']['data_disk'],'expected_disk_owner':config['ready']['run_id'],
       'observation':observation,'checkpoint_mode':c.get('checkpoint_mode','gcs'),
       'publisher_sha256':c.get('publisher_sha256'),'spool_min_free_bytes':c.get('spool_min_free_bytes'),
       'recovery_required_free_bytes':c['recovery_required_free_bytes']}
    atomic_json(receipt_path,provisional)
    return {'status':provisional['status'],'receipt':str(receipt_path),'sha256':sha(Path(receipt_path).read_bytes()),
            'gpu_launch_ready':False}


def inspect(args):return json.loads(command(['gcloud',*args,'--format=json']))


def finalize(provisional_path,expected_sha256,output,*,inspector=inspect):
    """Operator read-only attachment inspection; never trusts a supplied disk ID."""
    proof,data=read(provisional_path)
    if sha(data)!=expected_sha256 or proof.get('status')!='continuation_staged_pending_operator_disk_verification':
        raise ValueError('Provisional CPU receipt hash/status changed')
    obs=proof['observation'];disk=proof['expected_data_disk']
    if obs['project']!=PROJECT or obs['zone']!=ZONE or obs['machine_type'] not in CPU_TYPES:
        raise ValueError('Invalid observed CPU identity')
    instance=inspector(['compute','instances','describe',obs['instance_name'],f'--project={PROJECT}',f'--zone={ZONE}'])
    if (str(instance['id'])!=obs['instance_id'] or instance['name']!=obs['instance_name']
            or instance['machineType'].split('/')[-1]!=obs['machine_type']
            or instance.get('guestAccelerators') or instance.get('labels',{}).get('zimfo-purpose')!='cpu-preparation'):
        raise ValueError('CPU instance changed or is not a preparation host')
    matches=[item for item in instance['disks'] if item.get('deviceName')=='zimfo-inputs']
    if len(matches)!=1 or matches[0].get('source')!=disk['self_link'] or matches[0].get('autoDelete') is not False or matches[0].get('mode')!='READ_WRITE' or matches[0].get('boot'):
        raise ValueError('Observed device attachment does not identify expected retained disk')
    actual=inspector(['compute','disks','describe',disk['name'],f'--project={PROJECT}',f'--zone={ZONE}'])
    if (str(actual['id'])!=str(disk['id']) or actual['selfLink']!=disk['self_link']
            or actual.get('labels',{}).get('zimfo-run')!=proof['expected_disk_owner']
            or actual.get('users')!=[instance['selfLink']]):
        raise ValueError('Retained disk identity/ownership or attached instance changed')
    required=max(proof['recovery_required_free_bytes'],proof.get('spool_min_free_bytes') or 0)
    if proof['free_bytes']<required:raise ValueError('CPU staging lacks required headroom')
    result={key:proof[key] for key in ('plan_path','plan_sha256','files','free_bytes','checkpoint_mode','publisher_sha256','spool_min_free_bytes')}
    result.update(status='continuation_staged',data_disk_id=str(actual['id']),
                  provisional_sha256=sha(data),verified_instance_id=str(instance['id']),
                  bundle_sha256=proof['bundle_sha256'],launch_config_sha256=proof['launch_config_sha256'],
                  verification='operator_compute_readback_plus_same_cpu_metadata_device_observation')
    atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    pack=sub.add_parser('bundle')
    for name in ('launch-dir','source-plan-dir','output'):pack.add_argument('--'+name,required=True)
    cpu=sub.add_parser('stage')
    for name in ('bundle','bundle-sha256','receipt'):cpu.add_argument('--'+name,required=True)
    finish=sub.add_parser('finalize')
    for name in ('provisional','provisional-sha256','output'):finish.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.action=='bundle':result=bundle(args.launch_dir,args.source_plan_dir,args.output)
    elif args.action=='stage':result=stage(args.bundle,args.bundle_sha256,args.receipt)
    else:result=finalize(args.provisional,args.provisional_sha256,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
