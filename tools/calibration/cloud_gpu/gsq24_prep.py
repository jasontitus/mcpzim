"""Prepare compatible G2 disks on a bounded CPU host; never launch a GPU."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib,json,re,subprocess,uuid
from pathlib import Path
from cloud_prep.prepare import PROJECT,ZONE,SA,TAG,REGISTRY,run,ensure_ingress_denied
HERE=Path(__file__).resolve().parent
OS_IMAGE='ubuntu-2204-jammy-v20260918'
OS_IMAGE_ID='5599417423083116984'
CPU_TYPE='n2-standard-4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def bind(config):
    config.pop('config_sha256',None)
    config['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return config


def prepare(ready_path,output):
    ready=json.loads(Path(ready_path).read_text())
    if ready['restore_validation']['status']!='validated' or ready['input_commit']['sha256']!=ready['restore_validation']['input_commit_sha256']:
        raise ValueError('Verified existing GCS input commit required')
    if not re.fullmatch(re.escape(REGISTRY)+r'[a-z0-9_./-]+@sha256:[0-9a-f]{64}',ready['production_image']):
        raise ValueError('Pinned initial CPU restore runtime required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    runid='zimfo-gsq24prep-'+uuid.uuid4().hex[:12]
    files={'gsq24_prep_guest.py':HERE/'gsq24_prep_guest.py','gsq24_prep_startup.sh':HERE/'gsq24_prep_startup.sh',
           'cpu_common.py':HERE.parent/'cloud_prep/bootstrap.py'}
    for name,source in files.items():(output/name).write_bytes(source.read_bytes())
    config=bind({'run_id':runid,'machine_type':CPU_TYPE,'bucket':ready['bucket'],
      'input_commit':ready['input_commit'],'restore_image':ready['production_image'],
      'expected_restore_validation':ready['restore_validation'],'max_seconds':3600,
      'source_os_image':OS_IMAGE,'source_os_image_id':OS_IMAGE_ID,
      'files':{name:sha(output/name) for name in files}})
    save(output/'config.json',config)
    disk_commands=[['gcloud','compute','disks','create',runid+'-boot',f'--project={PROJECT}',f'--zone={ZONE}',
                   '--type=pd-balanced','--size=80GB',f'--image={OS_IMAGE}','--image-project=ubuntu-os-cloud',
                   f'--labels=zimfo-purpose=gsq24-preparation,zimfo-run={runid}','--format=json'],
                  ['gcloud','compute','disks','create',runid+'-inputs',f'--project={PROJECT}',f'--zone={ZONE}',
                   '--type=pd-ssd','--size=256GB',f'--labels=zimfo-purpose=gsq24-preparation,zimfo-run={runid}','--format=json']]
    plan={'status':'prepared_not_launched','config':config,'disk_commands':disk_commands,'cpu_machine':CPU_TYPE,
          'max_seconds':3600,'gpu_count':0,'retains_disks':True,'no_existing_disks_changed':True,
          'waiting_final_runtime_bound_seconds':2700,'gpu_launch_ready':False}
    save(output/'plan.json',plan);return plan


def create_command(config,output,deadline):
    runid=config['run_id']
    return ['gcloud','compute','instances','create',runid,f'--project={PROJECT}',f'--zone={ZONE}',f'--machine-type={CPU_TYPE}',
       '--provisioning-model=STANDARD','--maintenance-policy=TERMINATE','--instance-termination-action=DELETE',
       '--no-restart-on-failure',f'--termination-time={deadline}',
       f"--disk=name={config.get('disks',{}).get('boot_disk',{}).get('name',runid+'-boot')},boot=yes,auto-delete=no,device-name=zimfo-boot",
       f"--disk=name={config.get('disks',{}).get('data_disk',{}).get('name',runid+'-inputs')},auto-delete=no,device-name=zimfo-inputs,mode=rw",
       f'--service-account={SA}','--scopes=https://www.googleapis.com/auth/cloud-platform',
       f'--tags={TAG}',f'--labels=zimfo-purpose=cpu-preparation,zimfo-run={runid}',
       '--metadata=enable-oslogin=true,block-project-ssh-keys=true,serial-port-enable=false',
       f'--metadata-from-file=startup-script={output}/gsq24_prep_startup.sh,zimfo-gsq24-prep-config={output}/config.json,zimfo-gsq24-prep-guest={output}/gsq24_prep_guest.py,zimfo-gsq24-cpu-common={output}/cpu_common.py','--format=json']


def execute(output):
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text());config=json.loads((output/'config.json').read_text())
    if plan['status']!='prepared_not_launched' or config!=plan['config']:raise ValueError('Changed or attempted preparation')
    if bind(dict(config))!=config:raise ValueError('Changed preparation config')
    for name,digest in config['files'].items():
        if sha(output/name)!=digest:raise ValueError('Changed preparation script')
    image=run(['gcloud','compute','images','describe',OS_IMAGE,'--project=ubuntu-os-cloud','--format=json'])
    if str(image['id'])!=OS_IMAGE_ID or image['status']!='READY':raise ValueError('Pinned plain Ubuntu image changed')
    instances=run(['gcloud','compute','instances','list',f'--project={PROJECT}','--format=json'])
    if any(i.get('labels',{}).get('zimfo-purpose') in ('cpu-preparation','quantization','gsq24-validation') for i in instances):
        raise ValueError('Existing work VM must be inspected first')
    ensure_ingress_denied()
    plan['status']='disk_creation_requested';save(output/'plan.json',plan)
    disks=dict(config.get('disks',{})) if config.get('reuse_disks') else {}
    if config.get('reuse_disks'):
        for role,expected in disks.items():
            disk=run(['gcloud','compute','disks','describe',expected['name'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'])
            if str(disk['id'])!=expected['id'] or disk.get('users') or disk['type'].split('/')[-1]!=expected['type'] or disk.get('labels',{}).get('zimfo-run')!=config['disk_owner']:
                raise ValueError('Retained CPU disk changed or attached')
    for role,command in zip(('boot_disk','data_disk'),plan.get('disk_commands',[])):
        value=run(command);disk=value[0] if isinstance(value,list) else value
        disks[role]={'name':disk['name'],'id':str(disk['id']),'self_link':disk['selfLink'],'type':disk['type'].split('/')[-1]}
    plan['disks']=disks
    deadline=(datetime.now(timezone.utc)+timedelta(seconds=3600)).isoformat(timespec='seconds')
    config.update(disks=disks,absolute_deadline=deadline);bind(config);save(output/'config.json',config)
    plan.update(status='launch_requested',config=config,create_command=create_command(config,output,deadline))
    save(output/'plan.json',plan)
    save(output/'instance.json',run(plan['create_command']))
    plan['status']='launched';save(output/'plan.json',plan)
    return plan


def set_final(output,image,source_receipt):
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text())
    if plan['status']!='launched' or not re.fullmatch(re.escape(REGISTRY)+r'[a-z0-9_./-]+@sha256:[0-9a-f]{64}',image):
        raise ValueError('Live CPU preparation and immutable final image required')
    receipt_bytes=Path(source_receipt).read_bytes()
    if len(receipt_bytes)>1024*1024:raise ValueError('Oversized source receipt')
    receipt=json.loads(receipt_bytes)
    final={'runtime_image':image,'source_receipt':receipt,'source_receipt_sha256':hashlib.sha256(receipt_bytes).hexdigest(),
           'source_receipt_text':receipt_bytes.decode(),'cpu_run_id':plan['config']['run_id']}
    if (output/'final-runtime.json').exists():raise ValueError('Final runtime metadata is immutable per attempt')
    save(output/'final-runtime.json',final)
    run(['gcloud','compute','instances','add-metadata',plan['config']['run_id'],f'--project={PROJECT}',f'--zone={ZONE}',
         f'--metadata-from-file=zimfo-gsq24-final={output}/final-runtime.json','--format=json'])
    return final




def finish(output):
    """Collect verified CPU proof, delete only the stopped prep VM, keep disks."""
    from gcs_checkpoints import make_client
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text());config=plan['config']
    if plan['status'] not in ('launched','launch_requested'):raise ValueError('No active preparation to finish')
    bucket=make_client(PROJECT,use_gcloud=True).bucket(config['bucket'])
    blob=bucket.get_blob(f"preparation/{config['run_id']}/result.json")
    if blob is None or blob.size>2*1024**2:raise ValueError('Bounded CPU result not available')
    report=json.loads(blob.download_as_bytes(if_generation_match=int(blob.generation)))
    if report.get('run_id')!=config['run_id'] or report.get('config_sha256')!=config['config_sha256']:
        raise ValueError('CPU report identity mismatch')
    save(output/'result.json',report)
    inspected=subprocess.run(['gcloud','compute','instances','describe',config['run_id'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'],capture_output=True,text=True)
    if inspected.returncode:
        if 'was not found' not in inspected.stderr and 'notFound' not in inspected.stderr:raise RuntimeError('Cannot verify CPU state')
    else:
        vm=json.loads(inspected.stdout)
        if vm.get('status')!='TERMINATED' or vm.get('labels',{}).get('zimfo-run')!=config['run_id'] or vm['machineType'].split('/')[-1]!=CPU_TYPE:
            raise ValueError('CPU VM must be owned and stopped before detach')
        for role,disk in config['disks'].items():
            attached=[v for v in vm['disks'] if v['source']==disk['self_link']]
            if len(attached)!=1 or attached[0].get('autoDelete') is not False:raise ValueError('Disk preservation changed')
        run(['gcloud','compute','instances','delete',config['run_id'],f'--project={PROJECT}',f'--zone={ZONE}','--keep-disks=all','--quiet','--format=json'])
    for role,disk in config['disks'].items():
        actual=run(['gcloud','compute','disks','describe',disk['name'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'])
        if str(actual['id'])!=disk['id'] or actual.get('users') or actual['type'].split('/')[-1]!=disk['type'] or actual.get('labels',{}).get('zimfo-run')!=config.get('disk_owner',config['run_id']):
            raise ValueError('New PD disk identity/type/detachment mismatch')
    plan['status']='finished_failed';save(output/'plan.json',plan)
    if report.get('status')!='gsq24_cpu_prepared':return plan
    final=json.loads((output/'final-runtime.json').read_text())
    if (report.get('production_image')!=final['runtime_image'] or report.get('restore_validation')!=config['expected_restore_validation']
            or report.get('source_receipt_sha256')!=final['source_receipt_sha256']
            or report.get('source_validation',{}).get('status')!='source_payloads_verified'
            or report.get('source_validation',{}).get('source_commit')!=final['source_receipt']['receipt']['commit']):
        raise ValueError('CPU source/runtime preparation proof mismatch')
    probe=report.get('container_probe',{})
    if probe.get('gpu_available') is not False or not str(probe.get('cuda','')).startswith('13.') or not probe.get('torch'):
        raise ValueError('Prepared CPU runtime import/CUDA build proof missing')
    ready={'status':'gsq24_prepared_disks_detached','project':PROJECT,'zone':ZONE,'run_id':config.get('disk_owner',config['run_id']),'cpu_run_id':config['run_id'],
      'bucket':config['bucket'],'input_commit':config['input_commit'],**config['disks'],
      'production_image':report['production_image'],'runtime_proof':report['runtime_proof'],
      'source_validation':report['source_validation'],'source_receipt_sha256':report['source_receipt_sha256'],
      'restore_validation':report['restore_validation'],'gpu_smoke_passed':False,'source_os_image':OS_IMAGE,
      'source_os_image_id':OS_IMAGE_ID,'cpu_result_generation':int(blob.generation),'free_bytes':report['free_bytes'],
      'container_probe':probe}
    from cloud_gpu.gsq24 import validate_ready
    validate_ready(ready)
    save(output/'ready.json',ready);plan['status']='finished_prepared';save(output/'plan.json',plan);return plan


def prepare_resume(source,output):
    prior=json.loads((Path(source)/'plan.json').read_text())
    if prior['status'] not in ('finished_failed','finished_prepared'):raise ValueError('Finish/detach the prior CPU attempt first')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    config=dict(prior['config']);config.pop('absolute_deadline',None)
    config.update(run_id='zimfo-gsq24prep-'+uuid.uuid4().hex[:12],reuse_disks=True,
                  disk_owner=config.get('disk_owner',config['run_id']))
    for name,source in {'gsq24_prep_guest.py':HERE/'gsq24_prep_guest.py','gsq24_prep_startup.sh':HERE/'gsq24_prep_startup.sh',
                        'cpu_common.py':HERE.parent/'cloud_prep/bootstrap.py'}.items():
        (output/name).write_bytes(source.read_bytes())
    config['files']={name:sha(output/name) for name in config['files']};bind(config)
    save(output/'config.json',config)
    plan={'status':'prepared_not_launched','config':config,'disk_commands':[],
          'cpu_machine':CPU_TYPE,'max_seconds':3600,'gpu_count':0,'retains_disks':True,'gpu_launch_ready':False}
    save(output/'plan.json',plan);return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--resume-source');parser.add_argument('--ready');parser.add_argument('--execute',action='store_true');parser.add_argument('--finish',action='store_true')
    parser.add_argument('--final-image');parser.add_argument('--source-receipt');args=parser.parse_args()
    if args.finish:result=finish(args.output)
    elif args.final_image:result=set_final(args.output,args.final_image,args.source_receipt)
    elif args.execute:result=execute(args.output)
    elif args.resume_source:result=prepare_resume(args.resume_source,args.output)
    else:result=prepare(args.ready,args.output)
    print(json.dumps(result,indent=2))
