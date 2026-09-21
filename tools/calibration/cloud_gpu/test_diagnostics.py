import gzip
import io
import json
from pathlib import Path
import urllib.request


def test_malformed_report_does_not_discard_valid_terminal_log(tmp_path,monkeypatch):
    config={'run_id':'zimfo-gpu-abcdef123456','ready':{'bucket':'tiltastech-zimfo-quantization-us-central1'}}
    (tmp_path/'status.json').write_text('unfinished json')
    (tmp_path/'solver.log').write_text('useful traceback')
    stage=tmp_path/'02-smoke_rco';stage.mkdir()
    (stage/'recovery-report.json').write_text('{"status":"failed"}')
    posted=[]
    def open_url(request,timeout):
        if request.get_method()=='POST':
            assert 'ifGenerationMatch=0' in request.full_url
            posted.append(json.loads(gzip.decompress(request.data)))
            return io.BytesIO(b'{}')
        return io.BytesIO(json.dumps(config if 'attributes' in request.full_url else {'access_token':'fake-test-token'}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',open_url)
    shell=Path(__file__).with_name('shutdown-diagnostics.sh').read_text()
    assert 'timeout --signal=TERM --kill-after=2s 25s python3' in shell
    source=shell.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    source=source.replace("root=pathlib.Path('/mnt/zimfo-inputs/jobs')/run_id",'root=pathlib.Path('+repr(str(tmp_path))+')')
    exec(compile(source,'shutdown-diagnostics','exec'),{})
    assert len(posted)==1
    assert posted[0]['solver.log']=='useful traceback'
    assert posted[0]['02-smoke_rco/recovery-report.json']=={'status':'failed'}
    assert posted[0]['diagnostics_skipped_files']==['status.json']
    assert 'fake-test-token' not in json.dumps(posted)


def test_continuation_bundle_preserves_published_pointer_source_and_supervision(tmp_path,monkeypatch):
    config={'run_id':'zimfo-gpu-abcdef123456','ready':{'bucket':'tiltastech-zimfo-quantization-us-central1'}}
    root=tmp_path/'job';root.mkdir()
    (root/'status.json').write_text('{"status":"running"}')
    (root/'controller.log').write_text('controller recovery details')
    source=root/'source-checkpoint';source.mkdir()
    (source/'latest-checkpoint.json').write_text('{"snapshot":"prior"}')
    (source/'frozen-config.json').write_text('{"source":"bound"}')
    cloud=root/'cloud/gsq';cloud.mkdir(parents=True)
    snapshot='gsq-b002-s00000174-e001-q00000-p00000'
    target=cloud/'snapshots'/snapshot;target.mkdir(parents=True)
    (cloud/'latest.json').write_text(json.dumps({'snapshot':snapshot,'receipt':{'sha256':'a'*64}}))
    for name in ('latest-checkpoint.json','frozen-config.json','publication-audit.json'):
        (target/name).write_text(json.dumps({'evidence':name}))
    # A rogue snapshot path is not followed or exported.
    rogue=root/'cloud/head';rogue.mkdir()
    (rogue/'latest.json').write_text('{"snapshot":"../../secret"}')
    supervision=tmp_path/'supervision';supervision.mkdir()
    (supervision/'supervisor.log').write_text('bounded supervisor detail')
    (supervision/'active-container.json').write_text('{"owner_run_id":"zimfo-gpu-abcdef123456"}')
    posted=[]
    def remote(request,timeout):
        if request.get_method()=='POST':
            posted.append(json.loads(gzip.decompress(request.data)));return io.BytesIO(b'{}')
        return io.BytesIO(json.dumps(config if 'attributes' in request.full_url else {'access_token':'never-store-me'}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',remote)
    shell=Path(__file__).with_name('shutdown-diagnostics.sh').read_text()
    code=shell.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    code=code.replace("root=pathlib.Path('/mnt/zimfo-inputs/jobs')/run_id",'root=pathlib.Path('+repr(str(root))+')')
    code=code.replace("supervision=pathlib.Path('/mnt/zimfo-inputs/supervision')/run_id",'supervision=pathlib.Path('+repr(str(supervision))+')')
    exec(compile(code,'shutdown-diagnostics','exec'),{})
    assert len(posted)==1
    result=posted[0]
    assert result['source-checkpoint/latest-checkpoint.json']['snapshot']=='prior'
    assert result['cloud/gsq/latest.json']['snapshot']==snapshot
    assert result[f'cloud/gsq/snapshots/{snapshot}/publication-audit.json']=={'evidence':'publication-audit.json'}
    assert result['supervision/supervisor.log']=='bounded supervisor detail'
    assert result['controller.log']=='controller recovery details'
    assert not any('secret' in name for name in result)
    assert 'never-store-me' not in json.dumps(result)


def test_recovery_receipts_get_budget_before_large_performance_reports(tmp_path,monkeypatch):
    config={'run_id':'zimfo-gpu-abcdef123456','ready':{'bucket':'tiltastech-zimfo-quantization-us-central1'}}
    stage=tmp_path/'04-gsq';stage.mkdir()
    for i in range(8):
        (stage/f'large-{i}-report.json').write_text(json.dumps({'data':'x'*(1024*1024-12)}))
    source=tmp_path/'source-checkpoint';source.mkdir()
    (source/'latest-checkpoint.json').write_text('{"snapshot":"safe-source"}')
    cloud=tmp_path/'cloud/gsq';cloud.mkdir(parents=True)
    snapshot='gsq-b002-s00000174-e001-q00000-p00000'
    target=cloud/'snapshots'/snapshot;target.mkdir(parents=True)
    (cloud/'latest.json').write_text(json.dumps({'snapshot':snapshot}))
    (target/'latest-checkpoint.json').write_text('{"snapshot":"safe-cloud"}')
    posted=[]
    def remote(request,timeout):
        if request.get_method()=='POST':
            posted.append(json.loads(gzip.decompress(request.data)));return io.BytesIO(b'{}')
        return io.BytesIO(json.dumps(config if 'attributes' in request.full_url else {'access_token':'test'}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',remote)
    shell=Path(__file__).with_name('shutdown-diagnostics.sh').read_text()
    code=shell.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    code=code.replace("root=pathlib.Path('/mnt/zimfo-inputs/jobs')/run_id",'root=pathlib.Path('+repr(str(tmp_path))+')')
    exec(compile(code,'shutdown-diagnostics','exec'),{})
    assert len(posted)==1
    assert posted[0]['source-checkpoint/latest-checkpoint.json']['snapshot']=='safe-source'
    assert posted[0][f'cloud/gsq/snapshots/{snapshot}/latest-checkpoint.json']['snapshot']=='safe-cloud'
    assert any('large-' in name for name in posted[0]['diagnostics_skipped_files'])
