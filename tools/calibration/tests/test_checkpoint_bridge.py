import hashlib
import json
from pathlib import Path

import pytest

from checkpoint_bridge import CheckpointBridge, materialize_configuration, read_latest
from checkpoints import CheckpointError, LocalCheckpointStore, REQUIRED_ROLES, canonical_json
from gcs_checkpoints import GCSCheckpointStore
from test_gcs_checkpoints import MemoryBackend


@pytest.fixture
def case(tmp_path):
    identity={'baseline_repo':'Qwen/Qwen3.8-27B','baseline_revision':'a'*40,
              'solver_revision':'b'*40,'calibration_sha256':'1'*64,
              'solver_config_sha256':'2'*64,'candidate_database_sha256':'3'*64,'runtime_sha256':'4'*64}
    local=LocalCheckpointStore(tmp_path/'local')
    source=tmp_path/'source';source.mkdir()
    config={'checkpoint':{'backend':'local','path':str(local.root)},'identity':identity,'stage':'gsq',
            'runtime_sha256':'4'*64,'epochs':1,'learning_rate':0.0001,'checkpoint_seconds':120}
    payloads={}
    for role in REQUIRED_ROLES | {'configuration','candidate_block_0'}:
        payloads[role]=source/role
        payloads[role].write_bytes(canonical_json(config) if role=='configuration' else role.encode())
    backend=MemoryBackend(); target={'backend':'gcs','bucket':'test-bucket','prefix':'runs/attempt/production'}
    root=tmp_path/'receipts'
    bridge=CheckpointBridge(local.root,root,target,stage='gsq',backend=backend)
    def publish(step):
        payloads['progress'].write_bytes(str(step).encode())
        return local.publish(f'gsq-b000-s{step:08d}-e000-q{step:05d}-p{step:05d}',identity,payloads)
    return locals()


def test_roundtrip_cloud_config_and_original_audit(case,tmp_path):
    c=case;local_manifest=c['publish'](1)
    result=c['bridge'].publish_latest()
    assert result['status']=='published'
    latest=read_latest(c['root'])
    frozen=latest['configuration']
    assert frozen['checkpoint']==c['target']
    assert {k:v for k,v in frozen.items() if k!='checkpoint'}=={k:v for k,v in c['config'].items() if k!='checkpoint'}
    receipt=latest['checkpoint']['receipt']
    c['bridge'].store.restore(local_manifest['snapshot'],c['identity'],tmp_path/'restored',commit_generation=receipt['commit']['generation'])
    assert json.loads((tmp_path/'restored/configuration').read_bytes())==frozen
    assert (tmp_path/'restored/source_configuration').read_bytes()==c['payloads']['configuration'].read_bytes()
    for role in REQUIRED_ROLES:
        assert (tmp_path/'restored'/role).read_bytes()==c['payloads'][role].read_bytes()
    assert local_manifest==c['local'].verify(local_manifest['snapshot'],c['identity'])
    assert not (c['local'].root/'latest.json').exists()


def test_coalesces_multiple_local_commits(case):
    c=case
    for step in range(1,5):c['publish'](step)
    result=c['bridge'].publish_latest()
    assert result['checkpoint']['snapshot'].endswith('p00004')
    assert len(c['bridge'].store.committed_snapshots())==1
    assert len(c['local'].committed_snapshots())==4
    writes=len(c['backend'].objects)
    assert c['bridge'].publish_latest()['status']=='already_published'
    assert len(c['backend'].objects)==writes


def test_no_checkpoint_no_cloud_write(case):
    assert case['bridge'].publish_latest()=={'status':'no_checkpoint'}
    assert not case['backend'].objects


def test_after_remote_commit_crash_retries_idempotently(case):
    c=case;c['publish'](1)
    def crash(receipt):raise RuntimeError('simulated kill after commit')
    with pytest.raises(RuntimeError,match='simulated kill'):
        c['bridge'].publish_latest(after_remote_commit=crash)
    assert not (c['root']/'latest.json').exists()
    remote_keys=set(c['backend'].objects)
    result=c['bridge'].publish_latest()
    assert result['status']=='published'
    assert set(c['backend'].objects)==remote_keys


def test_partial_upload_has_no_commit_then_retry(case,monkeypatch):
    c=case;c['publish'](1);original=c['backend'].create
    def fail(key,stream,size):
        if '/commits/' in key:raise OSError('network interrupted')
        return original(key,stream,size)
    monkeypatch.setattr(c['backend'],'create',fail)
    with pytest.raises(OSError):c['bridge'].publish_latest()
    assert not c['bridge'].store.committed_snapshots()
    assert not (c['root']/'latest.json').exists()
    monkeypatch.setattr(c['backend'],'create',original)
    assert c['bridge'].publish_latest()['status']=='published'


def test_missing_object_no_remote_commit(case):
    c=case;manifest=c['publish'](1)
    (c['local'].objects/manifest['payloads']['optimizer']['sha256']).unlink()
    with pytest.raises(FileNotFoundError):c['bridge'].publish_latest()
    assert not c['backend'].objects


def test_corrupt_object_no_remote_commit(case):
    c=case;manifest=c['publish'](1)
    (c['local'].objects/manifest['payloads']['optimizer']['sha256']).write_bytes(b'wrong')
    with pytest.raises(CheckpointError,match='Corrupt payload'):c['bridge'].publish_latest()
    assert not c['backend'].objects


def test_source_modified_after_verify_before_copy_rejected_before_commit(case,monkeypatch):
    c=case;manifest=c['publish'](1);original=c['bridge'].store.publish
    def mutate(snapshot,identity,payloads):
        payloads['candidate_block_0'].write_bytes(b'changed between verify and copy')
        return original(snapshot,identity,payloads)
    monkeypatch.setattr(c['bridge'].store,'publish',mutate)
    with pytest.raises(CheckpointError,match='Uploaded bytes differ'):
        c['bridge'].publish_latest()
    assert not c['bridge'].store.committed_snapshots()
    assert not (c['root']/'latest.json').exists()


def test_source_modified_after_upload_rejected_before_commit(case,monkeypatch):
    c=case;manifest=c['publish'](1);original=c['backend'].create
    path=c['local'].objects/manifest['payloads']['candidate_block_0']['sha256']
    changed=False
    def mutate(key,stream,size):
        nonlocal changed
        result=original(key,stream,size)
        if '/objects/' in key and not changed:
            path.write_bytes(b'changed');changed=True
        return result
    monkeypatch.setattr(c['backend'],'create',mutate)
    with pytest.raises(CheckpointError):c['bridge'].publish_latest()
    assert not c['bridge'].store.committed_snapshots()


def test_source_commit_changed_after_publish_rejected(case):
    c=case;manifest=c['publish'](1);c['bridge'].publish_latest()
    path=c['local'].commits/(manifest['snapshot']+'.json')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(CheckpointError,match='source commit changed'):c['bridge'].publish_latest()


def test_changed_target_rejected_on_reuse(case):
    c=case;c['publish'](1);c['bridge'].publish_latest()
    target={**c['target'],'prefix':'runs/other'}
    bridge=CheckpointBridge(c['local'].root,c['root'],target,stage='gsq',backend=c['backend'])
    with pytest.raises(CheckpointError,match='target changed'):bridge.publish_latest()


def test_symlink_source_rejected(case,tmp_path):
    c=case;manifest=c['publish'](1)
    path=c['local'].objects/manifest['payloads']['optimizer']['sha256']
    elsewhere=tmp_path/'elsewhere';elsewhere.write_bytes(path.read_bytes());path.unlink();path.symlink_to(elsewhere)
    with pytest.raises(CheckpointError,match='regular file'):c['bridge'].publish_latest()
    assert not c['backend'].objects


def test_source_config_wrong_identity_rejected(case):
    c=case;c['config']['identity']={**c['identity'],'runtime_sha256':'9'*64}
    c['payloads']['configuration'].write_bytes(canonical_json(c['config']));c['publish'](1)
    with pytest.raises(CheckpointError,match='configuration identity'):c['bridge'].publish_latest()
    assert not c['backend'].objects


def test_local_and_receipt_directories_must_be_separate(case):
    c=case
    with pytest.raises(CheckpointError,match='must be separate'):
        CheckpointBridge(c['local'].root,c['local'].root/'receipts',c['target'],stage='gsq',backend=c['backend'])


def test_receipt_tamper_detected(case):
    c=case;c['publish'](1);r=c['bridge'].publish_latest()
    path=Path(r['directory'])/'frozen-config.json';config=json.loads(path.read_bytes());config['epochs']=7
    path.write_bytes(canonical_json(config))
    with pytest.raises(CheckpointError,match='not bound'):read_latest(c['root'])


def test_generation_pinned_read_rejects_missing_generation(case):
    c=case;c['publish'](1);r=c['bridge'].publish_latest()
    commit=r['checkpoint']['receipt']['commit'];generation,data=c['backend'].objects[commit['object']]
    c['backend'].objects[commit['object']]=(generation+1,data)
    with pytest.raises(FileNotFoundError):c['bridge'].publish_latest()


def test_subsequent_snapshot_reuses_unchanged_extra_objects(case,monkeypatch):
    c=case;c['publish'](1);first=c['bridge'].publish_latest();c['publish'](2)
    calls=[];original=c['backend'].create
    def record(key,*args):calls.append(key);return original(key,*args)
    monkeypatch.setattr(c['backend'],'create',record)
    c['bridge'].publish_latest()
    for role in ['candidate_block_0','source_configuration']:
        assert first['checkpoint']['receipt']['manifest']['payloads'][role]['object'] not in calls


def test_checkpoint_conversion_rejects_additional_transport_options(case):
    with pytest.raises(CheckpointError,match='Invalid target'):
        materialize_configuration(canonical_json(case['config']),{**case['target'],'epochs':20})


def test_watch_waits_for_producer_then_final_drains(case,tmp_path):
    from checkpoint_bridge import watch
    c=case;now=[0];done=tmp_path/'done';moved=tmp_path/'later'
    c['local'].root.rename(moved)
    events=[]
    def sleep(seconds):
        now[0]+=seconds
        moved.rename(c['local'].root);c['publish'](3);done.write_text('done')
    result=watch(c['local'].root,c['root'],c['target'],stage='gsq',watch_seconds=600,
                 interval_seconds=300,producer_done=done,backend=c['backend'],
                 clock=lambda:now[0],sleep=sleep,emit=events.append)
    assert events[0]['status']=='waiting_for_producer'
    assert result['checkpoint']['snapshot'].endswith('p00003')
    assert now[0]==1


def test_watch_final_drain_picks_checkpoint_created_during_previous_publish(case,tmp_path):
    from checkpoint_bridge import watch
    c=case;c['publish'](1);done=tmp_path/'done';now=[0]
    def emit(result):
        if not done.exists():c['publish'](2);done.write_text('done')
    result=watch(c['local'].root,c['root'],c['target'],stage='gsq',watch_seconds=600,
                 producer_done=done,backend=c['backend'],clock=lambda:now[0],emit=emit)
    assert result['checkpoint']['snapshot'].endswith('p00002')
    assert len(c['bridge'].store.committed_snapshots())==2


def test_watch_missing_store_deadline_returns_waiting(tmp_path):
    from checkpoint_bridge import watch
    now=[0]
    def sleep(seconds):now[0]+=seconds
    result=watch(tmp_path/'missing',tmp_path/'receipts',{},stage='gsq',watch_seconds=90,
                 interval_seconds=30,clock=lambda:now[0],sleep=sleep)
    assert now[0]==90
    assert result['status']=='waiting_for_producer'


def test_watch_done_without_checkpoint_explicitly_reports_none(tmp_path):
    from checkpoint_bridge import watch
    done=tmp_path/'done';done.write_text('done')
    assert watch(tmp_path/'missing',tmp_path/'receipts',{},stage='gsq',producer_done=done)=={'status':'no_checkpoint'}


def test_checkpoint_progress_cannot_regress(case):
    c=case;c['publish'](1);latest=c['publish'](2);c['bridge'].publish_latest()
    (c['local'].commits/(latest['snapshot']+'.json')).unlink()
    with pytest.raises(CheckpointError,match='regressed'):c['bridge'].publish_latest()


def test_new_snapshot_cannot_change_target(case):
    c=case;c['publish'](1);c['bridge'].publish_latest();c['publish'](2)
    bridge=CheckpointBridge(c['local'].root,c['root'],{**c['target'],'prefix':'runs/other'},stage='gsq',backend=c['backend'])
    with pytest.raises(CheckpointError,match='target changed'):bridge.publish_latest()


def test_mutated_configuration_after_local_verify_rejected(case,monkeypatch):
    c=case;manifest=c['publish'](1);original=LocalCheckpointStore.verify
    def mutate(self,snapshot,identity):
        result=original(self,snapshot,identity)
        path=self.objects/result['payloads']['configuration']['sha256']
        config=json.loads(path.read_bytes());config['epochs']=999;path.write_bytes(canonical_json(config))
        return result
    monkeypatch.setattr(LocalCheckpointStore,'verify',mutate)
    with pytest.raises(CheckpointError,match='Configuration changed'):c['bridge'].publish_latest()
    assert not c['backend'].objects


def test_two_publisher_owners_rejected(case):
    import fcntl
    c=case;c['publish'](1)
    with (c['root']/'.publisher.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(CheckpointError,match='Another publisher'):c['bridge'].publish_latest()
    assert not c['backend'].objects


def test_watch_does_not_poll_gcs_each_second(case,tmp_path,monkeypatch):
    from checkpoint_bridge import watch
    c=case;c['publish'](1);c['bridge'].publish_latest();now=[0];stats=[]
    original=c['backend'].stat
    def stat(*args,**kwargs):stats.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(c['backend'],'stat',stat)
    def sleep(seconds):now[0]+=seconds
    watch(c['local'].root,c['root'],c['target'],stage='gsq',watch_seconds=90,
          interval_seconds=300,backend=c['backend'],clock=lambda:now[0],sleep=sleep)
    assert len(stats)==1
