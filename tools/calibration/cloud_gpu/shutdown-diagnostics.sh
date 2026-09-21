#!/bin/bash
# One bounded terminal object; no per-step cloud writes.
timeout --signal=TERM --kill-after=2s 25s python3 - <<'PY'
import gzip, json, pathlib, re, urllib.request, urllib.parse
request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/attributes/zimfo-gpu-config',headers={'Metadata-Flavor':'Google'})
with urllib.request.urlopen(request,timeout=10) as response: config=json.loads(response.read(1024*1024))
run_id=config['run_id'];bucket=config['ready']['bucket']
if not re.fullmatch(r'zimfo-gpu-[a-f0-9]{12}',run_id) or bucket!='tiltastech-zimfo-quantization-us-central1':raise ValueError('Unexpected diagnostics owner')
root=pathlib.Path('/mnt/zimfo-inputs/jobs')/run_id
if root.is_symlink():raise ValueError('Unexpected diagnostics root')
result={};skipped=[]
def read_json(path,name):
    try:return json.loads(path.read_text())
    except (ValueError,OSError,UnicodeError):
        skipped.append(name);return None
for name in ('status.json','solver.log','controller.log'):
    path=root/name
    if path.is_symlink() or not path.is_file(): continue
    if name=='status.json':
        if path.stat().st_size > 1024*1024: continue
        value=read_json(path,name)
        if value is not None:result[name]=value
    else:
        with path.open('rb') as source:
            source.seek(max(0,path.stat().st_size-2*1024*1024));result[name]=source.read(2*1024*1024).decode('utf-8',errors='replace')
# Stage timing/size receipts are small and share this single terminal object.
total=0
# Continuation evidence: follow only the bounded, immutable published pointer.
# Never use producer-local checkpoints as a claim of GCS durability.
def add_json(path,name):
    global total
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:return None
    if total+path.stat().st_size>8*1024*1024:
        skipped.append(name);return None
    total+=path.stat().st_size
    value=read_json(path,name)
    if value is not None:result[name]=value
    return value
source=root/'source-checkpoint'
if source.is_dir() and not source.is_symlink():
    for name in ('latest-checkpoint.json','frozen-config.json'):
        add_json(source/name,'source-checkpoint/'+name)
cloud=root/'cloud'
if cloud.is_dir() and not cloud.is_symlink():
    for stage_name in ('embedding','gsq','head','rco'):
        stage=cloud/stage_name
        if stage.is_symlink() or not stage.is_dir():continue
        pointer=add_json(stage/'latest.json','cloud/'+stage_name+'/latest.json')
        snapshot=pointer.get('snapshot','') if isinstance(pointer,dict) else ''
        if not isinstance(snapshot,str) or not re.fullmatch(r'(embedding|gsq|head|rco)-b[0-9]{3,}-s[0-9]{8,}-e[0-9]{3,}-q[0-9]{5,}-p[0-9]{5,}',snapshot):continue
        snapshots=stage/'snapshots';target=snapshots/snapshot
        if snapshots.is_symlink() or target.is_symlink() or not target.is_dir():continue
        for name in ('latest-checkpoint.json','frozen-config.json','publication-audit.json'):
            add_json(target/name,'cloud/'+stage_name+'/snapshots/'+snapshot+'/'+name)
for stage in sorted(root.glob('[0-9][0-9]-*')):
    if stage.is_symlink() or not stage.is_dir():continue
    for path in sorted(stage.glob('*.json')):
        if path.name not in ('latest-checkpoint.json','frozen-config.json') and not path.name.endswith('-report.json'):continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:continue
        name=str(path.relative_to(root))
        if total+path.stat().st_size>8*1024*1024:
            skipped.append(name);continue
        total+=path.stat().st_size
        value=read_json(path,name)
        if value is not None:result[name]=value
supervision=pathlib.Path('/mnt/zimfo-inputs/supervision')/run_id
if supervision.is_dir() and not supervision.is_symlink():
    for name in ('supervisor.log',):
        path=supervision/name
        if path.is_symlink() or not path.is_file():continue
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-2*1024*1024))
            result['supervision/'+name]=stream.read(2*1024*1024).decode('utf-8',errors='replace')
    add_json(supervision/'active-container.json','supervision/active-container.json')
if not result: raise SystemExit(0)
if skipped:result['diagnostics_skipped_files']=skipped[:256]
request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',headers={'Metadata-Flavor':'Google'})
with urllib.request.urlopen(request,timeout=10) as response: token=json.loads(response.read(65536))['access_token']
# Bound the actual serialized payload as well as individual source files.
# Logs have lower recovery value than committed checkpoint provenance.
encoded=json.dumps(result).encode()
if len(encoded)>24*1024*1024:
    for name in ('solver.log','controller.log','supervision/supervisor.log'):
        if name in result:
            result.pop(name);skipped.append(name)
    result['diagnostics_skipped_files']=skipped[:256]
    encoded=json.dumps(result).encode()
if len(encoded)>24*1024*1024:raise ValueError('Bounded diagnostics exceeded serialized limit')
query=urllib.parse.urlencode({'uploadType' :'media','name':f'runs/{run_id}/terminal-diagnostics.json.gz','ifGenerationMatch':0})
request=urllib.request.Request('https://storage.googleapis.com/upload/storage/v1/b/tiltastech-zimfo-quantization-us-central1/o?'+query,data=gzip.compress(encoded),method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/gzip'})
with urllib.request.urlopen(request,timeout=15) as response: response.read(65536)
print('Zimfo terminal diagnostics uploaded as one bounded object.',flush=True)
PY
