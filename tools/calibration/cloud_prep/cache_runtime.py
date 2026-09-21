"""CPU-only refresh of a prepared container cache and full-size GCS read test."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from cloud_prep.prepare import PROJECT, ZONE, REGISTRY, create_command, read, run, sha, verify_prepared_files
from cloud_prep.resume_cpu import prepare as prepare_continuation


def prepare(source, receipt, benchmark, image, output):
    ready=read(Path(receipt));entry=read(Path(benchmark))
    if not re.fullmatch(re.escape(REGISTRY)+r'[a-z0-9][a-z0-9_./-]*@sha256:[0-9a-f]{64}',image):
        raise ValueError('Immutable private runtime required')
    if ready['status']!='prepared_disks_detached' or ready['restore_validation']['status']!='validated':
        raise ValueError('Prepared input receipt required')
    if (not entry['object'].startswith('runs/') or entry['object'].rsplit('/',1)[-1]!=entry['sha256']
            or not re.fullmatch('[0-9a-f]{64}',entry['sha256']) or int(entry['generation'])<=0
            or not 1024**3<=entry['bytes']<=8*1024**3):
        raise ValueError('Pinned1–8GiB benchmark object required')
    plan=prepare_continuation(source,output);output=Path(output).resolve()
    config=plan['config'];config.pop('config_sha256')
    config.update(image=image,cache_only=True,benchmark_object=entry,expected_restore_validation=ready['restore_validation'])
    config['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    plan.update(status='cache_prepared',prepared_receipt=ready)
    plan['create_command']=create_command(plan,output)
    for name,value in [('plan.json',plan),('config.json',config)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    return plan


def execute(output):
    output=Path(output).resolve();plan=read(output/'plan.json');verify_prepared_files(plan,output)
    if plan['status']!='cache_prepared' or plan['machine_type']!='c4-standard-2' or not plan['config']['cache_only']:
        raise ValueError('Expected an unattempted CPU-only cache plan')
    instances=run(['gcloud','compute','instances','list',f'--project={PROJECT}','--format=json'])
    if any(i.get('labels',{}).get('zimfo-purpose') in ('cpu-preparation','quantization') for i in instances):
        raise ValueError('Existing preparation/GPU VM must be inspected first')
    for role in ('boot_disk','data_disk'):
        expected=plan['prepared_receipt'][role]
        disk=run(['gcloud','compute','disks','describe',expected['name'],f'--project={PROJECT}',f'--zone={ZONE}','--format=json'])
        if (str(disk['id'])!=expected['id'] or disk.get('users') or disk['status']!='READY'
                or disk.get('labels',{}).get('zimfo-run')!=plan['run_id']):
            raise ValueError('Prepared disk ownership/detachment changed')
    plan['status']='launch_requested';(output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=run(plan['create_command']);(output/'instance.json').write_text(json.dumps(result,indent=2)+'\n')
    plan['status']='launched';(output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','receipt','benchmark','image'):parser.add_argument('--'+name)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute and not all((args.source,args.receipt,args.benchmark,args.image)):parser.error('Missing preparation inputs')
    print(json.dumps(execute(args.output) if args.execute else prepare(args.source,args.receipt,args.benchmark,args.image,args.output),indent=2))
