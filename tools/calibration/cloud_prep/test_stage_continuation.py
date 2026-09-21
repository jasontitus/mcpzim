import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from cloud_prep import stage_continuation as stage
from cloud_gpu.test_continuation_host import prepared
from cloud_gpu.bootstrap import validate_staging_receipt


def setup_bundle(tmp_path):
    out,config,status=prepared(tmp_path,staged=False)
    archive=tmp_path/'bundle.tar'
    result=stage.bundle(out,tmp_path/'source',archive)
    return config,archive,result


def observation(config):
    return {'project':stage.PROJECT,'zone':stage.ZONE,'instance_name':'cpu-prep-test','instance_id':'123',
            'machine_type':'c4-standard-2','device_name':'zimfo-inputs','device_path':stage.DEVICE,
            'mounted_source':'/dev/test','device_major':1,'device_minor':2,
            'runtime_image':config['ready']['production_image'],'prepared_validation_sha256':'c'*64}


def mock_disk(tmp_path,monkeypatch,free=100*1024**3):
    mount=tmp_path/'mounted';mount.mkdir()
    monkeypatch.setattr(stage,'MOUNT',mount)
    monkeypatch.setattr(stage.shutil,'disk_usage',lambda _:SimpleNamespace(free=free))
    return mount


def operator(config,changes=None):
    selflink='https://www.googleapis.com/compute/v1/projects/tiltastech-zimfo/zones/us-central1-b/instances/cpu-prep-test'
    instance={'id':'123','name':'cpu-prep-test','machineType':'zones/us-central1-b/machineTypes/c4-standard-2',
              'labels':{'zimfo-purpose':'cpu-preparation'},'selfLink':selflink,
              'disks':[{'deviceName':'zimfo-inputs','source':config['ready']['data_disk']['self_link'],
                        'autoDelete':False,'boot':False,'mode':'READ_WRITE'}]}
    disk={'id':'2','selfLink':config['ready']['data_disk']['self_link'],'users':[selflink],
          'labels':{'zimfo-run':config['ready']['run_id']}}
    if changes:changes(instance,disk)
    def inspect(args):
        assert args[:2] in (['compute','instances'],['compute','disks'])
        assert 'create' not in args and 'delete' not in args
        return instance if args[1]=='instances' else disk
    return inspect


def test_deterministic_bundle_and_no_side_effects_until_cpu_staging(tmp_path):
    config,archive,result=setup_bundle(tmp_path)
    second=tmp_path/'second.tar';stage.bundle(tmp_path/'launch',tmp_path/'source',second)
    assert archive.read_bytes()==second.read_bytes()
    restored,files=stage.unpack(archive,result['sha256'])
    assert restored==config and files['continuation.py']
    with pytest.raises(ValueError,match='hash'):stage.unpack(archive,'0'*64)


def test_full_offline_bundle_provisional_and_operator_finalization(tmp_path,monkeypatch):
    config,archive,result=setup_bundle(tmp_path)
    mount=mock_disk(tmp_path,monkeypatch)
    provisional=tmp_path/'provisional.json'
    receipt=stage.stage(archive,result['sha256'],provisional,observer=observation)
    assert receipt['gpu_launch_ready'] is False
    assert (mount/'continuation-plans/third/reconciled-status.json').is_file()
    proof=json.loads(provisional.read_text())
    config['continuation']['staging_receipt']=proof
    with pytest.raises(ValueError,match='CPU-staged'):validate_staging_receipt(config)
    final=stage.finalize(provisional,receipt['sha256'],tmp_path/'final.json',inspector=operator(config))
    config['continuation']['staging_receipt']=final
    validate_staging_receipt(config)
    assert final['provisional_sha256']==receipt['sha256'] and final['verified_instance_id']=='123'
    assert final['data_disk_id']=='2'


@pytest.mark.parametrize('change',[
    lambda i,d:i.update(id='456'),
    lambda i,d:i['disks'][0].update(source='another disk'),
    lambda i,d:i['disks'][0].update(autoDelete=True),
    lambda i,d:d.update(id='999'),
    lambda i,d:d.update(users=['another instance']),
    lambda i,d:i.update(guestAccelerators=[{'acceleratorCount':1}]),
])
def test_operator_rejects_changed_vm_or_disk_chain(tmp_path,monkeypatch,change):
    config,archive,result=setup_bundle(tmp_path);mock_disk(tmp_path,monkeypatch)
    provisional=tmp_path/'provisional.json'
    receipt=stage.stage(archive,result['sha256'],provisional,observer=observation)
    with pytest.raises(ValueError):
        stage.finalize(provisional,receipt['sha256'],tmp_path/'final.json',inspector=operator(config,change))
    assert not (tmp_path/'final.json').exists()


def test_changed_receipt_or_insufficient_disk_rejected_without_staging(tmp_path,monkeypatch):
    config,archive,result=setup_bundle(tmp_path);mount=mock_disk(tmp_path,monkeypatch,free=1)
    with pytest.raises(ValueError,match='headroom'):
        stage.stage(archive,result['sha256'],tmp_path/'provisional.json',observer=observation)
    assert not (mount/'continuation-plans').exists()
    malformed=tmp_path/'bad.json';malformed.write_text('{"status":"fake"}')
    with pytest.raises(ValueError,match='hash/status'):
        stage.finalize(malformed,'0'*64,tmp_path/'final.json',inspector=lambda _:pytest.fail('remote before hash check'))


def test_modified_or_symlink_source_does_not_enter_bundle(tmp_path):
    out,config,_=prepared(tmp_path)
    source=tmp_path/'source/reconciled-status.json';saved=source.read_bytes()
    source.write_text('{}')
    with pytest.raises(ValueError,match='evidence'):stage.bundle(out,source.parent,tmp_path/'bundle.tar')
    source.unlink();outside=tmp_path/'outside';outside.write_bytes(saved);source.symlink_to(outside)
    with pytest.raises(ValueError,match='Unsafe'):stage.bundle(out,source.parent,tmp_path/'bundle2.tar')


def test_observer_reads_actual_cpu_metadata_mounted_device_and_cached_runtime(tmp_path,monkeypatch):
    config,_,_=setup_bundle(tmp_path)
    mount=mock_disk(tmp_path,monkeypatch)
    (mount/'prepared').mkdir();(mount/'prepared/restore-validation.json').write_text(json.dumps(config['ready']['restore_validation']))
    values={'project/project-id':stage.PROJECT,'instance/zone':'zones/'+stage.ZONE,'instance/id':'123',
            'instance/name':'cpu-prep-test','instance/machine-type':'types/c4-standard-2'}
    monkeypatch.setattr(stage,'metadata',lambda name:values[name])
    original_mount=Path.is_mount
    monkeypatch.setattr(Path,'is_mount',lambda self:True if self==mount else original_mount(self))
    original_stat=os.stat
    def observed_stat(path,*args,**kwargs):
        if str(path) in (stage.DEVICE,'/dev/test'):
            return SimpleNamespace(st_mode=0o060000,st_rdev=os.makedev(1,2))
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(stage.os,'stat',observed_stat)
    def command(args):
        if args[0]=='findmnt':return '/dev/test'
        if args[0]=='blkid':return 'zimfo-inputs'
        if args[:3]==['docker','image','inspect']:
            return json.dumps([{'RepoDigests':[config['ready']['production_image']],'Architecture':'amd64'}])
        pytest.fail(str(args))
    monkeypatch.setattr(stage,'command',command)
    actual=stage.observe_cpu(config)
    assert actual['instance_id']=='123' and actual['device_major']==1 and actual['device_minor']==2
    assert 'data_disk_id' not in actual  # only operator Compute readback certifies this
    values['instance/machine-type']='types/g4-standard-48'
    with pytest.raises(ValueError,match='CPU-only'):stage.observe_cpu(config)
