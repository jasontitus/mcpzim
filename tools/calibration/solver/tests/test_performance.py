import json
import pytest
import torch
from solver import performance


def test_timing_synchronizes_cuda_and_separates_work(tmp_path, monkeypatch, capsys):
    clock=iter([10.,12.,20.,23.,30.,37.])
    monkeypatch.setattr(performance.time,'monotonic',lambda:next(clock))
    # Assert the routing, not torch.cuda's own guard: device_policy.sync is a
    # no-op when CUDA is absent (this Apple Silicon build), so the measure()
    # contract under test is "every timing with a device goes through
    # device_policy.sync with that device".
    sync=[]
    monkeypatch.setattr(performance,'_sync',lambda device:sync.append(str(device)))
    for tokens in (600,700):
        with performance.measure(tmp_path,'gsq','update','cuda',tokens=tokens,block=3):pass
    with performance.measure(tmp_path,'gsq','checkpoint_publish'):pass
    assert sync==['cuda']*4
    report=json.loads((tmp_path/'performance-report.json').read_text())
    groups=list(report['groups'].values())
    update=next(g for g in groups if g['operation']=='update')
    assert (update['count'],update['seconds'],update['tokens'],update['tokens_squared'])==(2,5.,1300,850000)
    assert update['dimensions']['token_bucket_upper']==1024
    assert next(g for g in groups if g['operation']=='checkpoint_publish')['seconds']==7.
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e.get('tokens') for e in events]==[600,700,None]


def test_timing_synchronizes_mps_through_device_policy(tmp_path, monkeypatch):
    """MPS executes lazily; an unsynchronized timing would exclude the work."""
    sync=[]
    monkeypatch.setattr(performance,'_sync',lambda device:sync.append(str(device)))
    with performance.measure(tmp_path,'gsq','update','mps',tokens=4):pass
    with performance.measure(tmp_path,'gsq','update',None,tokens=4):pass
    assert sync==['mps','mps']


def test_device_policy_sync_covers_mps_and_never_raises(monkeypatch):
    from solver import device_policy
    calls=[]
    monkeypatch.setattr(torch.mps,'synchronize',lambda:calls.append('mps'))
    monkeypatch.setattr(device_policy,'mps_available',lambda:True)
    device_policy.sync('mps');device_policy.sync(torch.device('cpu'))
    assert calls==['mps']


def test_failed_work_is_not_counted_as_successful_throughput(tmp_path):
    with performance.measure(tmp_path,'rco','update','cpu',tokens=4):pass
    with pytest.raises(RuntimeError,match='original failure'):
        with performance.measure(tmp_path,'rco','update','cpu',tokens=4):
            raise RuntimeError('original failure')
    groups=json.loads((tmp_path/'performance-report.json').read_text())['groups'].values()
    assert {g['status']:g['count'] for g in groups}=={'completed':1,'failed':1}


def test_telemetry_failure_preserves_original_exception_and_success(tmp_path,capsys):
    (tmp_path/'performance-report.json').write_text('invalid json')
    with pytest.raises(RuntimeError,match='original failure'):
        with performance.measure(tmp_path,'rco','update'):
            raise RuntimeError('original failure')
    with performance.measure(tmp_path,'rco','checkpoint_publish'):pass
    assert capsys.readouterr().err.count('solver_timing_record_failed')==2


def test_corpus_cache_uses_sum_of_squared_sequence_lengths(tmp_path):
    with performance.measure(tmp_path,'gsq','cache_propagation',tokens=13,tokens_squared=85,sequences=2):pass
    group=next(iter(json.loads((tmp_path/'performance-report.json').read_text())['groups'].values()))
    assert group['tokens_squared']==6**2+7**2
    assert 'token_bucket_upper' not in group['dimensions']
