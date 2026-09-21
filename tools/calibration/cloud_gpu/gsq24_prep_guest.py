"""Install/cache/restore on an explicitly bounded CPU host, never a GPU."""
import hashlib,json,os,re,shutil,subprocess,tempfile,time,urllib.error,urllib.request
from pathlib import Path
import cpu_common as common
ROOT=Path('/opt/zimfo-gsq24-prep')
MOUNT=common.MOUNT
REGISTRY='us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/'


def image_valid(image):return bool(re.fullmatch(re.escape(REGISTRY)+r'[a-z0-9_./-]+@sha256:[a-f0-9]{64}',image))
def download(url,path):
    with urllib.request.urlopen(url,timeout=60) as response:Path(path).write_bytes(response.read(4*1024*1024))


def configure_host():
    command=common.command;kernel=command(['uname','-r'])
    # apt's downloader/verifier runs as _apt, even though preparation is root.
    # Startup umask077 must not hide the public repository trust material.
    for path in ('/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg','/etc/apt/sources.list.d/nvidia-container-toolkit.list'):
        if Path(path).is_file():Path(path).chmod(0o644)
    command(['systemctl','mask','--now','apt-daily.service','apt-daily.timer','apt-daily-upgrade.service',
             'apt-daily-upgrade.timer','unattended-upgrades.service'])
    command(['apt-get','update','-qq'])
    command(['apt-get','install','-y','--no-install-recommends','docker.io','gnupg','ca-certificates',
             'linux-headers-'+kernel,'build-essential','dkms'])
    keyring=ROOT/'cuda-keyring.deb'
    download('https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb',keyring)
    command(['dpkg','-i',str(keyring)]);command(['apt-get','update','-qq'])
    command(['apt-get','install','-y','--no-install-recommends','nvidia-driver-pinning-580'])
    command(['apt-get','install','-y','--no-install-recommends','nvidia-open'])
    driver=command(['modinfo','-F','version','nvidia']);module=command(['modinfo','-F','filename','nvidia'])
    if not driver.startswith('580.') or '/'+kernel+'/' not in module:
        raise RuntimeError('NVIDIA580 module was not prepared for the running/future boot kernel')
    if command(['modinfo','-F','license','nvidia'])!='Dual MIT/GPL':raise ValueError('Expected open NVIDIA module')
    key=ROOT/'nvidia-toolkit.asc';download('https://nvidia.github.io/libnvidia-container/gpgkey',key)
    command(['gpg','--dearmor','--yes','--output','/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg',str(key)])
    Path('/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg').chmod(0o644)
    listing=ROOT/'nvidia-toolkit.list';download('https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list',listing)
    text=listing.read_text().replace('deb https://','deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://')
    Path('/etc/apt/sources.list.d/nvidia-container-toolkit.list').write_text(text)
    Path('/etc/apt/sources.list.d/nvidia-container-toolkit.list').chmod(0o644)
    command(['apt-get','update','-qq']);command(['apt-get','install','-y','--no-install-recommends','nvidia-container-toolkit'])
    command(['systemctl','enable','--now','docker']);command(['nvidia-ctk','runtime','configure','--runtime=docker'])
    command(['systemctl','restart','docker'])
    if 'nvidia' not in json.loads(command(['docker','info','--format','{{json .Runtimes}}'])):
        raise ValueError('NVIDIA Docker runtime missing')
    return {'kernel':kernel,'driver':driver,'driver_module':module,'toolkit':command(['nvidia-container-cli','--version'])}


def cache(image):
    if not image_valid(image):raise ValueError('Mutable/foreign runtime refused')
    with tempfile.TemporaryDirectory(prefix='zimfo-gsq24-auth-',dir='/run') as auth:
        common.command(['docker','--config',auth,'login','-u','oauth2accesstoken','--password-stdin','https://us-central1-docker.pkg.dev'],private_input=common.token())
        common.command(['docker','--config',auth,'pull',image])
    inspected=json.loads(common.command(['docker','image','inspect',image]))
    if len(inspected)!=1 or inspected[0]['Architecture']!='amd64' or image not in inspected[0]['RepoDigests']:
        raise ValueError('Cached image digest/architecture mismatch')


def container(image,args):
    return ['docker','run','--rm','--runtime=runc','--pull=never','--network=host','--cap-drop=ALL',
      '--security-opt=no-new-privileges','--mount',f'type=bind,src={MOUNT},dst=/data',image,*args]


def final_metadata(config,until):
    while time.monotonic()<until:
        try:value=json.loads(common.metadata('instance/attributes/zimfo-gsq24-final'))
        except urllib.error.HTTPError as error:
            if error.code!=404:raise
            value=None
        if value:
            if value.get('cpu_run_id')!=config['run_id'] or not image_valid(value.get('runtime_image','')):
                raise ValueError('Final runtime belongs to another attempt')
            source=value['source_receipt_text'].encode()
            if hashlib.sha256(source).hexdigest()!=value['source_receipt_sha256'] or json.loads(source)!=value['source_receipt']:
                raise ValueError('Final source receipt bytes changed')
            return value
        print(json.dumps({'event':'gsq24_cpu_preparation','phase':'waiting_for_reviewed_final_runtime'}),flush=True)
        time.sleep(min(30,max(0,until-time.monotonic())))
    raise TimeoutError('Final runtime did not arrive within CPU preparation budget; retain disks')


def execute(config):
    started=time.monotonic()
    if common.metadata('instance/machine-type').decode().split('/')[-1]!='n2-standard-4':
        raise ValueError('Must use reviewed CPU-only machine')
    if config.get('reuse_disks'):
        if (common.command(['blkid','-s','LABEL','-o','value',common.DEVICE])!='zimfo-inputs'
                or int(common.command(['blockdev','--getsize64',common.DEVICE]))!=256*1024**3):
            raise ValueError('Retained input disk identity/size changed; never reformat')
        MOUNT.mkdir(exist_ok=True)
        if not MOUNT.is_mount():common.command(['mount','-o','noatime',common.DEVICE,str(MOUNT)])
    else:common.mount_inputs()
    proof=configure_host();cache(config['restore_image'])
    manifest=MOUNT/'inputs-manifest.json'
    if manifest.exists():
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=config['input_commit']['sha256']:raise ValueError('Retained input manifest changed')
    else:common.download_descriptor(config['bucket'],config['input_commit'],manifest)
    if not (MOUNT/'prepared').exists():common.command(container(config['restore_image'],['python','-m','restore_inputs','--manifest','/data/inputs-manifest.json',
       '--expected-manifest-sha256',config['input_commit']['sha256'],'--destination','/data/prepared']))
    validation=json.loads((MOUNT/'prepared/restore-validation.json').read_text())
    if validation!=config['expected_restore_validation']:raise ValueError('Restored inputs differ from reviewed original')
    final=final_metadata(config,started+2700)
    image=final['runtime_image'];cache(image)
    source=MOUNT/'gsq24-source-receipt.json';source.write_text(final['source_receipt_text'])
    common.command(container(image,['python','-m','prepare_gsq24_source','--source-receipt','/data/gsq24-source-receipt.json',
       '--source-receipt-sha256',final['source_receipt_sha256'],'--output','/data/gsq24-source']))
    source_validation=json.loads((MOUNT/'gsq24-source/source-validation.json').read_text())
    probe=json.loads(common.command(container(image,['python','-c',
       "import json,torch,solver.gsq24;assert not torch.cuda.is_available();assert torch.version.cuda.startswith('13.');print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'compiled_arches':torch._C._cuda_getArchFlags(),'gpu_available':False}))"])))
    return {'status':'gsq24_cpu_prepared','runtime_proof':proof,'restore_validation':validation,
            'source_validation':source_validation,'source_receipt_sha256':final['source_receipt_sha256'],
            'production_image':image,'container_probe':probe,'free_bytes':shutil.disk_usage(MOUNT).free,
            'gpu_smoke_passed':False,'elapsed_seconds':time.monotonic()-started}


def main():
    config=json.loads((ROOT/'config.json').read_text());body={k:v for k,v in config.items() if k!='config_sha256'}
    if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=config['config_sha256']:
        raise ValueError('Preparation configuration changed')
    try:report=execute(config)
    except BaseException as error:report={'status':'failed','error':type(error).__name__+': '+str(error)}
    report.update(run_id=config['run_id'],config_sha256=config['config_sha256'])
    target=MOUNT if MOUNT.is_mount() else ROOT
    (target/'gsq24-preparation.json').write_text(json.dumps(report,indent=2))
    common.command(['sync']);print(json.dumps(report),flush=True);common.publish_status(config,report)


if __name__=='__main__':main()
