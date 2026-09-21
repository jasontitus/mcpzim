"""Separate-process, coalescing publisher for immutable local solver checkpoints.

This module does not modify the pinned solver or launch compute. The producer
must use LocalCheckpointStore on retained storage. Run this publisher in its own
process: never as a thread in the training interpreter. No source deletion/GC is
performed. Each stage needs its own bridge, so completion anchors survive later
stages. Local progress is not cloud-durable until a generation-pinned receipt is
published. Interrupted uploads are retried idempotently, including after commit.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from checkpoints import (CheckpointError, HASH, MAX_MANIFEST_BYTES, REQUIRED_ROLES,
                         LocalCheckpointStore, _identity, _name, _sync_directory,
                         canonical_json, digest_file)
from gcs_checkpoints import GCSCheckpointStore

SNAPSHOT = re.compile(r'(embedding|gsq|head|rco)-b(\d{3,})-s(\d{8,})-e(\d{3,})-q(\d{5,})-p(\d{5,})\Z')
RESERVED = {'source_configuration', 'publication_audit'}


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CheckpointError('Missing or symlinked bridge evidence')
    with path.open('rb') as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise CheckpointError('Bridge evidence exceeds size limit')
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CheckpointError('Malformed bridge evidence') from exc
    if not isinstance(value, dict):
        raise CheckpointError('Bridge evidence must be an object')
    return value, data


def _descriptor(data):
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _write(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _atomic(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fingerprint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CheckpointError('Local checkpoint object must be a regular file')
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def materialize_configuration(source_bytes, target):
    """Only checkpoint transport changes; all algorithm fields remain identical."""
    source = json.loads(source_bytes)
    if not isinstance(source, dict) or source.get('checkpoint', {}).get('backend') != 'local':
        raise CheckpointError('Expected a local-backend frozen solver configuration')
    if (not isinstance(target, dict) or target.get('backend') != 'gcs'
            or not {'backend', 'bucket', 'prefix'} <= set(target)
            or set(target) - {'backend', 'bucket', 'prefix', 'project', 'staging_dir'}):
        raise CheckpointError('Invalid target checkpoint transport')
    _identity(source.get('identity', {}))
    cloud = copy.deepcopy(source); cloud['checkpoint'] = copy.deepcopy(target)
    if {k:v for k,v in cloud.items() if k != 'checkpoint'} != {k:v for k,v in source.items() if k != 'checkpoint'}:
        raise CheckpointError('Cloud materialization changed solver semantics')
    return canonical_json(cloud)


class _CommitGuard:
    """Check source-bound digests before the remote commit can become visible."""
    def __init__(self, backend, expected, source_check, snapshot, prefix):
        self.backend, self.expected, self.source_check = backend, expected, source_check
        self.commit_key = f'{prefix}/commits/{snapshot}.json'

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def create(self, key, stream, size):
        if '/commits/' in key:
            if key != self.commit_key or size > MAX_MANIFEST_BYTES:
                raise CheckpointError('Unexpected bridge commit')
            stream.seek(0); data = stream.read(MAX_MANIFEST_BYTES + 1); stream.seek(0)
            manifest = json.loads(data)
            observed = {role: {k: descriptor[k] for k in ('sha256', 'bytes')}
                        for role, descriptor in manifest['payloads'].items()}
            if observed != self.expected:
                raise CheckpointError('Uploaded bytes differ from local committed snapshot')
            self.source_check()
        return self.backend.create(key, stream, size)


def read_latest(receipt_root):
    """Read a complete immutable receipt directory; never consult producer latest."""
    root = Path(receipt_root)
    pointer, _ = _read(root/'latest.json')
    snapshot = _name(pointer.get('snapshot'))
    directory = root/'snapshots'/snapshot
    if directory.is_symlink() or not directory.is_dir():
        raise CheckpointError('Invalid published receipt directory')
    checkpoint, data = _read(directory/'latest-checkpoint.json')
    if _descriptor(data) != pointer.get('receipt') or checkpoint.get('snapshot') != snapshot:
        raise CheckpointError('Published receipt pointer mismatch')
    config, config_bytes = _read(directory/'frozen-config.json')
    audit, audit_bytes = _read(directory/'publication-audit.json')
    manifest = checkpoint['receipt']['manifest']; commit = checkpoint['receipt']['commit']
    encoded = canonical_json(manifest)
    if (manifest.get('schema') != 3 or manifest.get('snapshot') != snapshot
            or manifest.get('identity') != config.get('identity')
            or config['checkpoint'].get('backend') != 'gcs'
            or manifest.get('bucket') != config['checkpoint'].get('bucket')
            or manifest.get('prefix') != config['checkpoint'].get('prefix')
            or commit.get('object') != manifest['prefix']+'/commits/'+snapshot+'.json'
            or type(commit.get('generation')) is not int or commit['generation'] <= 0
            or {k:commit.get(k) for k in ('sha256','bytes')} != _descriptor(encoded)):
        raise CheckpointError('Published receipt/configuration mismatch')
    for role, payload in [('configuration', config_bytes), ('publication_audit', audit_bytes)]:
        if {k:manifest['payloads'][role].get(k) for k in ('sha256','bytes')} != _descriptor(payload):
            raise CheckpointError('Published configuration/audit not bound by commit')
    if (audit.get('snapshot') != snapshot or audit.get('identity') != manifest['identity']
            or audit.get('changed_configuration_fields') != ['checkpoint']
            or audit.get('cloud_configuration') != _descriptor(config_bytes)
            or audit.get('source_configuration') != {k:manifest['payloads']['source_configuration'].get(k) for k in ('sha256','bytes')}):
        raise CheckpointError('Published provenance identity mismatch')
    return {'directory': str(directory), 'checkpoint': checkpoint, 'configuration': config, 'audit': audit}


class CheckpointBridge:
    def __init__(self, local_root, receipt_root, target, *, stage, backend=None):
        if stage not in ('embedding','gsq','head','rco'):
            raise CheckpointError('Unknown production stage')
        self.stage = stage
        raw = Path(local_root)
        if raw.is_symlink() or not raw.is_dir():
            raise CheckpointError('Local checkpoint root must already exist')
        self.local_root = raw.resolve()
        for name in ('objects','commits'):
            if (self.local_root/name).is_symlink() or not (self.local_root/name).is_dir():
                raise CheckpointError('Missing/unsafe local checkpoint directory')
        self.root = Path(receipt_root)
        if self.root.is_symlink():
            raise CheckpointError('Receipt root cannot be symlinked')
        self.root = self.root.resolve()
        if self.root == self.local_root or self.local_root in self.root.parents or self.root in self.local_root.parents:
            raise CheckpointError('Receipt and producer stores must be separate')
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root/'snapshots').is_symlink():
            raise CheckpointError('Receipt snapshots cannot be symlinked')
        (self.root/'snapshots').mkdir(exist_ok=True)
        self.target = copy.deepcopy(target)
        self.store = GCSCheckpointStore(target['bucket'],target['prefix'],project=target.get('project'),
                                        staging_dir=target.get('staging_dir'),backend=backend)

    def _latest_local(self):
        candidates=[]
        for path in (self.local_root/'commits').glob('*.json'):
            match=SNAPSHOT.fullmatch(path.stem)
            if not match or match[1] != self.stage:
                raise CheckpointError('Unexpected snapshot name/stage in local store')
            # Global steps are monotone across blocks; ordinal breaks boundary ties.
            block,step,epoch,sequence,ordinal=map(int,match.groups()[1:])
            candidates.append(((step,block,epoch,sequence,ordinal),path))
        return max(candidates)[1] if candidates else None

    def publish_latest(self, *, after_remote_commit=None):
        """One bounded snapshot operation, coalescing intermediate local commits.

        Repeated calls reuse the GCS store's verified immutable-extra cache.
        after_remote_commit is an injected crash hook for tests only.
        """
        lock = self.root/'.publisher.lock'
        if lock.is_symlink():
            raise CheckpointError('Publisher lock cannot be symlinked')
        with lock.open('a+b') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointError('Another publisher owns this receipt directory') from exc
            return self._publish_latest(after_remote_commit)

    def _publish_latest(self, after_remote_commit):
        source_path = self._latest_local()
        if source_path is None:
            return {'status':'no_checkpoint'}
        manifest, manifest_bytes = _read(source_path)
        snapshot = source_path.stem
        identity = _identity(manifest.get('identity', {}))
        if (self.root/'latest.json').exists():
            previous = read_latest(self.root)
            old = previous['checkpoint']
            prior_manifest = old['receipt']['manifest']
            if (prior_manifest['identity'] != identity
                    or prior_manifest['bucket'] != self.target['bucket']
                    or prior_manifest['prefix'] != self.target['prefix']):
                raise CheckpointError('Bridge identity or target changed')
            old_match, new_match = SNAPSHOT.fullmatch(old['snapshot']), SNAPSHOT.fullmatch(snapshot)
            def order(match):
                block, step, epoch, sequence, ordinal = map(int, match.groups()[1:])
                return step, block, epoch, sequence, ordinal
            if not old_match or old_match[1] != self.stage or order(new_match) < order(old_match):
                raise CheckpointError('Local checkpoint progress regressed')
            if old['snapshot'] == snapshot:
                if previous['audit'].get('source_manifest') != _descriptor(manifest_bytes):
                    raise CheckpointError('Previously published source commit changed')
                receipt=old['receipt']; commit=receipt['commit']
                if receipt['manifest']['bucket'] != self.target['bucket'] or receipt['manifest']['prefix'] != self.target['prefix']:
                    raise CheckpointError('Bridge target changed')
                observed=self.store.backend.stat(commit['object'],commit['generation'])
                if observed != {k:commit[k] for k in ('object','generation','bytes')}:
                    raise CheckpointError('Published generation no longer available')
                return {'status':'already_published', **previous}
        local = LocalCheckpointStore(self.local_root)
        verified = local.verify(snapshot, identity)
        if verified != manifest or canonical_json(verified) != manifest_bytes:
            raise CheckpointError('Local commit changed or is not canonical')
        if RESERVED & set(manifest['payloads']) or 'configuration' not in manifest['payloads']:
            raise CheckpointError('Missing configuration or reserved provenance role')
        sources = {role:self.local_root/'objects'/entry['sha256'] for role,entry in manifest['payloads'].items()}
        fingerprints = {path:_fingerprint(path) for path in sources.values()}
        config, source_config = _read(sources['configuration'])
        if _descriptor(source_config) != manifest['payloads']['configuration']:
            raise CheckpointError('Configuration changed after local verification')
        if config.get('identity') != identity or config.get('stage') != self.stage:
            raise CheckpointError('Local configuration identity/stage mismatch')
        cloud_config=materialize_configuration(source_config,self.target)
        audit={'schema':1,'snapshot':snapshot,'identity':identity,
               'source_manifest':_descriptor(manifest_bytes),'source_configuration':_descriptor(source_config),
               'cloud_configuration':_descriptor(cloud_config),'changed_configuration_fields':['checkpoint']}
        audit_bytes=canonical_json(audit)
        def source_check():
            if _read(source_path)[1] != manifest_bytes:
                raise CheckpointError('Local commit changed during publication')
            if any(_fingerprint(path) != fingerprint for path,fingerprint in fingerprints.items()):
                raise CheckpointError('Local checkpoint source changed during publication')
        with tempfile.TemporaryDirectory(prefix='.publication-',dir=self.root) as scratch:
            scratch=Path(scratch)
            _write(scratch/'configuration',cloud_config); _write(scratch/'publication_audit',audit_bytes)
            payloads={**sources,'source_configuration':sources['configuration'],
                      'configuration':scratch/'configuration','publication_audit':scratch/'publication_audit'}
            expected={**manifest['payloads'],'source_configuration':_descriptor(source_config),
                      'configuration':_descriptor(cloud_config),'publication_audit':_descriptor(audit_bytes)}
            backend=self.store.backend
            self.store.backend=_CommitGuard(backend,expected,source_check,snapshot,self.store.prefix)
            try:
                receipt=self.store.publish(snapshot,identity,payloads)
            finally:
                self.store.backend=backend
            if after_remote_commit is not None:
                after_remote_commit(receipt)
            source_check()
            checkpoint={'snapshot':snapshot,'receipt':receipt,'publication_source':'immutable_local_checkpoint'}
            checkpoint_bytes=canonical_json(checkpoint)
            destination=self.root/'snapshots'/snapshot
            files={'frozen-config.json':cloud_config,'latest-checkpoint.json':checkpoint_bytes,'publication-audit.json':audit_bytes}
            if destination.exists():
                if destination.is_symlink() or any(_read(destination/name)[1] != data for name,data in files.items()):
                    raise CheckpointError('Immutable receipt conflict')
            else:
                pending=scratch/'receipt';pending.mkdir()
                for name,data in files.items():_write(pending/name,data)
                _sync_directory(pending);os.rename(pending,destination);_sync_directory(destination.parent)
            _atomic(self.root/'latest.json',canonical_json({'schema':1,'snapshot':snapshot,'receipt':_descriptor(checkpoint_bytes)}))
        return {'status':'published',**read_latest(self.root)}


def watch(local_root, receipt_root, target, *, stage, watch_seconds=0,
          interval_seconds=300, producer_done=None, backend=None,
          clock=time.monotonic, sleep=time.sleep, emit=None):
    """Cooperative lifetime; supervisor must enforce the hard upload timeout.

    Can start before producer creates its store. The producer_done sentinel must
    be atomically installed only after producer exit. Final drain reselects the
    newest immutable commit after seeing that sentinel. No producer is stopped
    or mutated. A publication may run beyond the polling lifetime; the parent
    reserves a separate drain allowance and owns process cancellation.
    """
    if (type(watch_seconds) is not int or not 0 <= watch_seconds <= 86400
            or type(interval_seconds) is not int or not 30 <= interval_seconds <= 3600):
        raise CheckpointError('Require lifetime 0..86400 and interval 30..3600 seconds')
    deadline=clock()+watch_seconds
    root=Path(local_root);bridge=None
    def publish():
        nonlocal bridge
        if bridge is None:
            if not root.exists():return {'status':'waiting_for_producer'}
            if root.is_symlink():raise CheckpointError('Local checkpoint root cannot be symlinked')
            if not all((root/name).exists() for name in ('objects','commits')):
                return {'status':'waiting_for_producer'}
            bridge=CheckpointBridge(root,receipt_root,target,stage=stage,backend=backend)
        return bridge.publish_latest()
    next_publication=clock()
    result={'status':'waiting_for_producer'}
    while True:
        if clock() >= next_publication:
            result=publish()
            next_publication=clock()+interval_seconds
            if emit is not None:emit(result)
        if producer_done is not None and Path(producer_done).exists():
            if Path(producer_done).is_symlink() or not Path(producer_done).is_file():
                raise CheckpointError('Producer-done sentinel must be a regular file')
            result=publish()
            if result['status']=='waiting_for_producer':result={'status':'no_checkpoint'}
            if emit is not None:emit(result)
            return result
        remaining=deadline-clock()
        if remaining <= 0:return result
        # Local sentinel checks only; remote publication cadence is independent.
        sleep(min(1.0,remaining,max(0.001,next_publication-clock())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root',required=True);parser.add_argument('--receipt-root',required=True)
    parser.add_argument('--target-json',required=True);parser.add_argument('--stage',choices=['embedding','gsq','head','rco'],required=True)
    parser.add_argument('--watch-seconds',type=int,default=0,help='Cooperative polling lifetime; parent enforces hard timeout')
    parser.add_argument('--interval-seconds',type=int,default=300)
    parser.add_argument('--producer-done',help='Atomically create this file after producer exits for final drain')
    args=parser.parse_args()
    target,_=_read(args.target_json)
    def emit(result):
        print(json.dumps({'status':result['status'],'snapshot':result.get('checkpoint',{}).get('snapshot'),
                          'directory':result.get('directory')}),flush=True)
    watch(args.local_root,args.receipt_root,target,stage=args.stage,
          watch_seconds=args.watch_seconds,interval_seconds=args.interval_seconds,
          producer_done=args.producer_done,emit=emit)


if __name__=='__main__':main()
