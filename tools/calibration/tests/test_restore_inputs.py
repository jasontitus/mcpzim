import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from restore_inputs import download, unpack


def make_tar(path, members):
    with tarfile.open(path, 'w') as archive:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize('members', [
    [('../escape', b'x', tarfile.REGTYPE)],
    [('/escape', b'x', tarfile.REGTYPE)],
    [('link', b'', tarfile.SYMTYPE)],
    [('duplicate', b'x', tarfile.REGTYPE), ('duplicate', b'y', tarfile.REGTYPE)]])
def test_unsafe_extraction_fails(tmp_path, members):
    archive = tmp_path / 'unsafe.tar'
    make_tar(archive, members)
    root = tmp_path / 'restore'
    root.mkdir()
    with pytest.raises(ValueError):
        unpack(archive, root)
    assert not (tmp_path / 'escape').exists()


def test_safe_files_and_no_overwrite(tmp_path):
    archive = tmp_path / 'valid.tar'
    make_tar(archive, [('nested/tensor', b'verified', tarfile.REGTYPE)])
    root = tmp_path / 'restore'
    root.mkdir()
    unpack(archive, root)
    assert (root / 'nested/tensor').read_bytes() == b'verified'
    with pytest.raises(FileExistsError):
        unpack(archive, root)


class Bucket:
    def __init__(self, data):
        self.data = data

    def blob(self, name, generation):
        assert generation == 17
        return self

    def open(self, mode, **kwargs):
        assert kwargs['if_generation_match'] == 17
        return io.BytesIO(self.data)


def test_generation_pinned_download_verifies_before_publish(tmp_path):
    data = b'original'
    obj = {'object': 'immutable', 'generation': '17', 'bytes': len(data),
           'sha256': hashlib.sha256(data).hexdigest()}
    output = tmp_path / 'result'
    with pytest.raises(ValueError):
        download(Bucket(b'corrupt!'), obj, output)
    assert not output.exists() and not (tmp_path / 'result.partial').exists()
    download(Bucket(data), obj, output)
    assert output.read_bytes() == data


def test_does_not_delete_existing_partial_owned_by_another_writer(tmp_path):
    partial = tmp_path / 'result.partial'
    partial.write_bytes(b'keep')
    obj = {'object': 'immutable', 'generation': '17', 'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
    with pytest.raises(FileExistsError):
        download(Bucket(b''), obj, tmp_path / 'result')
    assert partial.read_bytes() == b'keep'


def resume_fixture(tmp_path, monkeypatch):
    import json
    import types
    import restore_inputs as restore
    monkeypatch.setattr(restore.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=10**12))
    staging = tmp_path / '.input-restore-test'
    for name in ('model', 'calibration', 'downloads'): (staging/name).mkdir(parents=True)
    checksum = lambda data: hashlib.sha256(data).hexdigest()
    objects, payloads, specs = [], {}, {}
    package = {'status':'completed', 'revision':'test', 'source_files':{}, 'sequences':[],
               'model_artifacts':{'files':{'weights':{'bytes':7,'sha256':checksum(b'weights')}}}}
    def add(name, payload):
        payloads[name] = payload
        objects.append({'object':name,'generation':'17','bytes':len(payload),'sha256':checksum(payload)})
    for index in range(2):
        directory = f'invocation-{index:06d}'
        data = b'tensor-' + str(index).encode(); input_data = b'input'
        sequence = {'input_sha256':checksum(input_data), 'files':{'tensor':{'file':directory+'/tensor', 'sha256':checksum(data)}}}
        seq_data = json.dumps(sequence).encode()
        package['sequences'].append({'directory':directory,'manifest_sha256':checksum(seq_data)})
        members = {directory+'/sequence.json':seq_data, directory+'/input.json':input_data,directory+'/tensor':data}
        archive = tmp_path/(directory+'.tar')
        make_tar(archive, [(name,value,tarfile.REGTYPE) for name,value in members.items()])
        add('inputs/'+directory+'.tar',archive.read_bytes()); specs[directory] = members
        for name,value in members.items():
            path = staging/'calibration'/name;path.parent.mkdir(exist_ok=True)
            path.write_bytes(value if index == 0 or name.endswith('sequence.json') else b'partial')
    package_data = json.dumps(package).encode()
    metadata = tmp_path/'metadata.tar';make_tar(metadata,[('manifest.json',package_data,tarfile.REGTYPE)])
    add('inputs/metadata.tar',metadata.read_bytes())
    archives = [obj['object'] for obj in objects]
    add('models/test/weights',b'weights')
    (staging/'model/weights').write_bytes(b'weights')
    manifest = {'objects':objects,'calibration_archives':archives,'revision':'test','package_manifest_sha256':checksum(package_data)}
    class MultiBucket:
        calls = []
        def blob(self,name,generation):
            self.calls.append(name)
            return Bucket(payloads[name])
    return staging, tmp_path/'prepared', manifest, MultiBucket(), specs


def test_resume_reuses_completed_hashes_repairs_only_known_partial(tmp_path, monkeypatch):
    import restore_inputs as restore
    staging, root, manifest, bucket, specs = resume_fixture(tmp_path, monkeypatch)
    restore.prepare_resume(staging,root,manifest,'a'*64,bucket,True)
    assert 'inputs/invocation-000000.tar' not in bucket.calls
    assert 'models/test/weights' not in bucket.calls
    assert 'inputs/invocation-000001.tar' in bucket.calls
    for members in specs.values():
        for name,data in members.items(): assert (staging/'calibration'/name).read_bytes() == data
    assert (staging/'.restore-owner.json').exists()


@pytest.mark.parametrize('case',['unknown','symlink','owner','legacy','model_corruption'])
def test_resume_rejects_foreign_or_untrusted_state(tmp_path,monkeypatch,case):
    import json
    import restore_inputs as restore
    staging,root,manifest,bucket,_ = resume_fixture(tmp_path,monkeypatch)
    adopt = True
    if case == 'unknown': (staging/'calibration/foreign').write_text('keep')
    if case == 'symlink': (staging/'model/link').symlink_to(tmp_path/'outside')
    if case == 'owner': (staging/'.restore-owner.json').write_text(json.dumps({'manifest_sha256':'wrong'}))
    if case == 'legacy': adopt = False
    if case == 'model_corruption': (staging/'model/weights').write_text('corrupt')
    with pytest.raises(ValueError): restore.prepare_resume(staging,root,manifest,'a'*64,bucket,adopt)
    if case == 'unknown': assert (staging/'calibration/foreign').read_text() == 'keep'
    if case == 'model_corruption': assert (staging/'model/weights').read_text() == 'corrupt'


def test_resume_rejects_mismatched_metadata_archive(tmp_path,monkeypatch):
    import restore_inputs as restore
    staging,root,manifest,bucket,_ = resume_fixture(tmp_path,monkeypatch)
    manifest['package_manifest_sha256'] = '0'*64
    with pytest.raises(ValueError,match='metadata hash'): restore.prepare_resume(staging,root,manifest,'a'*64,bucket,True)
    assert not (staging/'.restore-owner.json').exists()


def test_complete_restore_continuation_publishes_only_after_validation(tmp_path,monkeypatch):
    import json
    import types
    import restore_inputs as restore
    staging,root,manifest,bucket,_ = resume_fixture(tmp_path,monkeypatch)
    manifest.update(schema_version=1,status='staged',bucket='private')
    path=tmp_path/'inputs-manifest.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr(restore,'make_client',lambda *a,**k:types.SimpleNamespace(bucket=lambda name:bucket))
    monkeypatch.setattr(restore,'validate_package',lambda path:{'status':'validated','manifest_sha256':manifest['package_manifest_sha256']})
    args=types.SimpleNamespace(manifest=str(path),expected_manifest_sha256=restore.digest(path),
        destination=str(root),local_gcloud=False,resume_staging=str(staging),adopt_legacy_staging=True)
    restore.restore(args)
    assert root.is_dir() and not staging.exists()
    report=json.loads((root/'restore-validation.json').read_text())
    assert report['input_commit_sha256']==restore.digest(path)
    assert (root/'model/weights').read_bytes()==b'weights'
