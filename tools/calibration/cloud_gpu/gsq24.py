"""One bounded single-L4 validation from CPU-prepared compatible disks."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib,json,re,subprocess,uuid
from pathlib import Path
from cloud_prep.prepare import PROJECT,ZONE,SA,TAG,REGISTRY,run,ensure_ingress_denied
HERE=Path(__file__).resolve().parent
MACHINE='g2-standard-32'
SOURCE_RECEIPT_SHA256='8bebdd955cff647de882cc4ce03ab9f9179570a8de15a56d5ed13c76a7c44d19'
SOURCE_COMMIT={'object':'runs/zimfo-gpu-6583ec8c4659/gsq/commits/gsq-b002-s00000174-e000-q00000-p00004.json',
 'generation':1789869833752714,'bytes':3918,'sha256':'75ae90952b053fa2df0d155256fed4cbe2babc00b87b1c835fbad7ead0c89180'}
SOURCE_IDENTITY={'baseline_repo':'Qwen/Qwen3.8-27B','baseline_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
 'calibration_sha256':'b4a3584782917bccec8bfbdba91bbc2249290860431cdae8233f624cfa337ead',
 'solver_revision':'03fc16484c369e3127225615d5e03e8d3a6043e3',
 'solver_config_sha256':'532b3072e836f805ee32bb92b9bfb6af373890b819744b5e773063e7ab6e59fe',
 'candidate_database_sha256':'e3b3bd5c733f3d25a4439362a2a7c6f5082e6280bd1beff3f6e4ba27d6ad4362',
 'runtime_sha256':'f13a4e99a85f4f650a8e3e130757fb4ddee891eeb8e34e6cea973e99dc56c3cd'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def bind(config):
    config.pop('config_sha256',None)
    config['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return config


def validate_ready(ready):
    if (ready.get('status')!='gsq24_prepared_disks_detached' or ready.get('project')!=PROJECT or ready.get('zone')!=ZONE
            or ready.get('gpu_smoke_passed') is not False or ready.get('restore_validation',{}).get('status')!='validated'
            or ready.get('source_validation',{}).get('status')!='source_payloads_verified'
            or not re.fullmatch(re.escape(REGISTRY)+r'[a-z0-9_./-]+@sha256:[a-f0-9]{64}',ready.get('production_image',''))):
        raise ValueError('Complete CPU-prepared L4 disk/source/runtime proof required')
    source=ready['source_validation'];frozen=source.get('configuration',{});proof=ready.get('runtime_proof',{})
    if (ready['restore_validation'].get('input_commit_sha256')!=ready['input_commit']['sha256']
            or ready['restore_validation'].get('manifest_sha256')!=SOURCE_IDENTITY['calibration_sha256']
            or ready.get('source_receipt_sha256')!=SOURCE_RECEIPT_SHA256 or source.get('source_commit')!=SOURCE_COMMIT
            or source.get('source_manifest_sha256')!=SOURCE_COMMIT['sha256'] or source.get('source_identity')!=SOURCE_IDENTITY
            or frozen.get('identity')!=SOURCE_IDENTITY or frozen.get('runtime_sha256')!=SOURCE_IDENTITY['runtime_sha256']
            or frozen.get('stage')!='gsq' or source.get('migration_applied') is not False
            or not proof.get('driver','').startswith('580.') or not proof.get('kernel')
            or '/'+proof['kernel']+'/' not in proof.get('driver_module','')):
        raise ValueError('Prepared source/calibration/configuration/driver binding mismatch')
    for role in ('boot_disk','data_disk'):
        disk=ready[role]
        if (disk['type'] not in ('pd-balanced','pd-ssd') or not str(disk['id']).isdigit()
                or not re.fullmatch(r'zimfo-gsq24prep-[a-f0-9]{12}-(boot|inputs)',disk['name'])
                or not disk['self_link'].endswith(f"/projects/{PROJECT}/zones/{ZONE}/disks/{disk['name']}")):
            raise ValueError('G2-compatible dedicated persistent disk required')
    if ready['free_bytes']<110*1024**3:raise ValueError('Insufficient diagnostic checkpoint disk headroom')


def prepare(ready_path,harness_path,output,max_seconds=3600):
    ready=json.loads(Path(ready_path).read_text());validate_ready(ready)
    if type(max_seconds) is not int or not 1200<=max_seconds<=3600:raise ValueError('Validation cap must be1200..3600seconds')
    harness=json.loads(Path(harness_path).read_text())
    expected={'inputs':'/inputs/prepared','source_checkpoint':'/inputs/gsq24-source',
       'source_receipt_sha256':ready['source_receipt_sha256'],'input_commit_sha256':ready['input_commit']['sha256'],
       'runtime_sha256':ready['production_image'].split('@sha256:')[1],'expected_longest_tokens':4673}
    if any(harness.get(k)!=v for k,v in expected.items()):raise ValueError('Harness input/source/runtime identity mismatch')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    files={'gsq24_startup.sh':HERE/'gsq24_startup.sh','gsq24_guest.py':HERE/'gsq24_guest.py','gpu_common.py':HERE/'bootstrap.py'}
    for name,source in files.items():(output/name).write_bytes(source.read_bytes())
    config=bind({'run_id':'zimfo-gsq24-'+uuid.uuid4().hex[:12],'ready':ready,'harness':harness,'max_seconds':max_seconds,
       'files':{name:sha(output/name) for name in files},'machine_type':MACHINE,'gpu_count':1,'gpu_memory_gib':24,
       'production_progress_authorized':False})
    save(output/'config.json',config)
    plan={'status':'prepared_not_launched','config':config,'spot':True,'max_seconds':max_seconds,
          'automatic_retries':0,'retains_disks':True,'published_compute_usd_per_hour_estimate':1.040448,
          'price_source':'https://cloud.google.com/spot-vms/pricing','storage_network_extra':True}
    save(output/'plan.json',plan);return plan


def create_command(config,output):
    ready=config['ready']
    return ['gcloud','compute','instances','create',config['run_id'],f'--project={PROJECT}',f'--zone={ZONE}',f'--machine-type={MACHINE}',
       '--provisioning-model=SPOT','--maintenance-policy=TERMINATE','--instance-termination-action=DELETE','--no-restart-on-failure',
       f"--termination-time={config['absolute_deadline']}",
       f"--disk=name={ready['boot_disk']['name']},boot=yes,auto-delete=no,device-name=zimfo-boot",
       f"--disk=name={ready['data_disk']['name']},auto-delete=no,device-name=zimfo-inputs,mode=rw",
       f'--service-account={SA}','--scopes=https://www.googleapis.com/auth/cloud-platform',f'--tags={TAG}',
       f"--labels=zimfo-purpose=gsq24-validation,zimfo-run={config['run_id']}",
       '--metadata=enable-oslogin=true,block-project-ssh-keys=true,serial-port-enable=false',
       f'--metadata-from-file=startup-script={output}/gsq24_startup.sh,zimfo-gsq24-config={output}/config.json,zimfo-gsq24-guest={output}/gsq24_guest.py,zimfo-gsq24-common={output}/gpu_common.py','--format=json']


def execute(output):
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text());config=json.loads((output/'config.json').read_text())
    if plan['status']!='prepared_not_launched' or config!=plan['config'] or bind(dict(config))!=config:raise ValueError('Changed/already attempted plan')
    validate_ready(config['ready'])
    for name,digest in config['files'].items():
        if sha(output/name)!=digest:raise ValueError('Changed validation bootstrap')
    for role in ('boot_disk','data_disk'):
        expected=config['ready'][role]
        disk=run(['gcloud','compute','disks','describe',expected['name'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'])
        if (str(disk['id'])!=expected['id'] or disk.get('users') or disk['status']!='READY'
                or disk['type'].split('/')[-1]!=expected['type'] or disk.get('labels',{}).get('zimfo-run')!=config['ready']['run_id']):
            raise ValueError('Prepared compatible disks changed or still attached')
    vms=run(['gcloud','compute','instances','list',f'--project={PROJECT}','--format=json'])
    if any(v.get('labels',{}).get('zimfo-purpose') in ('cpu-preparation','quantization','gsq24-validation') for v in vms):
        raise ValueError('Prior work VM requires inspection')
    machine=run(['gcloud','compute','machine-types','describe',MACHINE,f'--project={PROJECT}',f'--zone={ZONE}','--format=json'])
    if machine['memoryMb']<96*1024 or machine.get('accelerators')!=[{'guestAcceleratorCount':1,'guestAcceleratorType':'nvidia-l4'}]:
        raise ValueError('Machine no longer matches one L4 and enough host RAM')
    ensure_ingress_denied()
    config['absolute_deadline']=(datetime.now(timezone.utc)+timedelta(seconds=config['max_seconds'])).isoformat(timespec='seconds')
    bind(config);save(output/'config.json',config)
    plan.update(status='launch_requested',config=config,create_command=create_command(config,output));save(output/'plan.json',plan)
    save(output/'instance.json',run(plan['create_command']));plan['status']='launched';save(output/'plan.json',plan);return plan


def finish(output):
    from gcs_checkpoints import make_client
    from cloud_gpu.gsq24_guest import validate_terminal
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text());config=plan['config']
    if plan['status'] not in ('launched','launch_requested'):raise ValueError('No launched validation')
    blob=make_client(PROJECT,use_gcloud=True).bucket(config['ready']['bucket']).get_blob(f"runs/{config['run_id']}/vm-result.json")
    if blob is None or blob.size>4*1024**2:raise ValueError('No bounded terminal validation report')
    report=json.loads(blob.download_as_bytes(if_generation_match=int(blob.generation)))
    if report.get('run_id')!=config['run_id'] or report.get('config_sha256')!=config['config_sha256']:raise ValueError('Terminal report ownership mismatch')
    passed=report.get('status')=='validated_gsq24'
    if passed:validate_terminal(config,report)
    save(output/'result.json',report)
    state=subprocess.run(['gcloud','compute','instances','describe',config['run_id'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'],capture_output=True,text=True)
    if state.returncode:
        if 'was not found' not in state.stderr and 'notFound' not in state.stderr:raise RuntimeError('Could not verify stopped VM')
    else:
        vm=json.loads(state.stdout)
        if vm.get('status')!='TERMINATED' or vm.get('labels',{}).get('zimfo-run')!=config['run_id'] or vm['machineType'].split('/')[-1]!=MACHINE:
            raise ValueError('Validation VM must be owned and stopped')
        for role in ('boot_disk','data_disk'):
            attached=[d for d in vm['disks'] if d['source']==config['ready'][role]['self_link']]
            if len(attached)!=1 or attached[0].get('autoDelete') is not False:raise ValueError('Disk preservation changed')
        run(['gcloud','compute','instances','delete',config['run_id'],f'--project={PROJECT}',f'--zone={ZONE}','--keep-disks=all','--quiet','--format=json'])
    for role in ('boot_disk','data_disk'):
        expected=config['ready'][role];disk=run(['gcloud','compute','disks','describe',expected['name'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'])
        if str(disk['id'])!=expected['id'] or disk.get('users'):raise ValueError('Retained disk identity/detachment mismatch')
    plan['status']='finished_passed' if passed else 'finished_failed';save(output/'plan.json',plan);return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--ready');p.add_argument('--harness-config')
    p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--execute',action='store_true');p.add_argument('--finish',action='store_true');args=p.parse_args()
    if args.execute and args.finish:p.error('Choose execute or finish')
    result=finish(args.output) if args.finish else execute(args.output) if args.execute else prepare(args.ready,args.harness_config,args.output,args.max_seconds)
    print(json.dumps(result,indent=2))
