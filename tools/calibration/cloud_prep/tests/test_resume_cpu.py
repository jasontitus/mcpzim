import copy
from pathlib import Path
import pytest
from cloud_prep.resume_cpu import check_instance
from cloud_prep.resume_guest import restore_command


def instance():
    return {'name':'zimfo-prep-test','labels':{'zimfo-purpose':'cpu-preparation','zimfo-run':'zimfo-prep-test'},
            'machineType':'zones/test/machineTypes/c4-standard-2',
            'scheduling':{'maxRunDuration':{'seconds':'3600'},'instanceTerminationAction':'DELETE'},
            'disks':[{'source':'disks/boot','autoDelete':False},{'source':'disks/input','autoDelete':False}]}


def test_only_preserved_owned_bounded_cpu_can_restart():
    plan={'run_id':'zimfo-prep-test','boot_disk':'boot','input_disk':'input'}
    good=instance();check_instance(good,plan)
    for edit in ('gpu','delete','owner','duration','disk'):
        bad=copy.deepcopy(good)
        if edit=='gpu': bad['guestAccelerators']=[{}]
        if edit=='delete': bad['disks'][0]['autoDelete']=True
        if edit=='owner': bad['labels']['zimfo-run']='foreign'
        if edit=='duration': bad['scheduling']['maxRunDuration']['seconds']='7200'
        if edit=='disk': bad['disks'][0]['source']='disks/foreign'
        with pytest.raises(ValueError): check_instance(bad,plan)


def test_restore_override_is_readonly_no_image_pull_and_exact_commit():
    config={'resume_container':'unique','image':'image@sha256:abc','input_commit':{'sha256':'a'*64}}
    command=restore_command(config,Path('/mnt/zimfo-inputs/.input-restore-known'))
    assert '--pull=never' in command and '--runtime=runc' in command
    assert '--gpus=all' not in command
    assert command[command.index('--resume-staging')+1]=='/data/.input-restore-known'
    assert command[command.index('--expected-manifest-sha256')+1]=='a'*64
    assert any('restore_inputs.py,readonly' in part for part in command)


@pytest.mark.parametrize('slow',[False,True])
def test_cache_refresh_checks_large_readback_without_restoring(tmp_path,monkeypatch,slow):
    import json,hashlib
    from types import SimpleNamespace
    from cloud_prep import resume_guest as guest
    root=tmp_path/'mount';(root/'prepared').mkdir(parents=True)
    (root/'inputs-manifest.json').write_bytes(b'pinned')
    validation={'status':'validated','input_commit_sha256':hashlib.sha256(b'pinned').hexdigest()}
    (root/'prepared/restore-validation.json').write_text(json.dumps(validation))
    config={'machine_type':'c4-standard-2','cache_only':True,'image':'private@sha256:a',
            'input_commit':{'sha256':validation['input_commit_sha256']},'expected_restore_validation':validation,
            'benchmark_object':{},'bucket':'private','run_id':'owned','config_sha256':'a'}
    monkeypatch.setattr(guest,'MOUNT',root)
    monkeypatch.setattr(Path,'is_mount',lambda self: self==root)
    monkeypatch.setattr(guest,'metadata',lambda key:b'zones/z/machineTypes/c4-standard-2' if key.endswith('machine-type') else b'{"access_token":"test-token"}')
    calls=[]
    def command(args):
        calls.append(args)
        if args[0]=='blockdev':return str(256*1024**3)
        if args[0]=='blkid':return 'zimfo-inputs'
        if args[0]=='uname':return 'kernel'
        if args[0]=='modinfo':return {'version':'580.1','filename':'/lib/modules/kernel/nvidia.ko','license':'Dual MIT/GPL'}[args[2]]
        if args[:3]==['docker','image','inspect']:return json.dumps([{'RepoDigests':[config['image']],'Architecture':'amd64'}])
        if args[:2]==['docker','run']:return json.dumps({'cuda':'13.0','gpu_available':False})
        return 'toolkit'
    monkeypatch.setattr(guest,'command',command)
    monkeypatch.setattr(guest.tempfile,'TemporaryDirectory',lambda **kw: __import__('contextlib').nullcontext(str(tmp_path)))
    def run(args,**kwargs):
        calls.append(args)
        if '--network=host' in args:return SimpleNamespace(stdout=json.dumps({'MiB_per_second':1 if slow else 90,'bytes':4634634240}))
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(guest.subprocess,'run',run)
    monkeypatch.setattr(guest,'restore_command',lambda *a:pytest.fail('Cache refresh must never restore/format'))
    if slow:
        with pytest.raises(ValueError,match='too slow'):guest.prepare(config)
    else:assert guest.prepare(config)['transport_benchmark']['MiB_per_second']==90
    assert all('mkfs.ext4' not in command for command in calls)
