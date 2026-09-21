"""Offline timing estimates must preserve missing coverage and nesting boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from performance_summary import summarize, read_json


def group(stage='gsq', operation='update_with_input_load', status='completed',
          lengths=(600, 700), elapsed=10., kind='linear_attention', block=0):
    dims = {}
    if lengths:
        buckets = {((n+511)//512)*512 for n in lengths}
        assert len(buckets) == 1
        dims['token_bucket_upper'] = buckets.pop()
    if kind is not None:
        dims['kind'] = kind
    if block is not None:
        dims['block'] = block
    return {'stage': stage, 'operation': operation, 'status': status,
            'dimensions': dims, 'count': len(lengths) or 1, 'seconds': elapsed,
            'tokens': sum(lengths), 'tokens_squared': sum(n*n for n in lengths)}


def report(*groups, path='04-gsq/performance-report.json'):
    return {path: {'schema': 1, 'groups': {str(i): g for i, g in enumerate(groups)}}}


def row(summary, stage, bucket, kind=None):
    return next(x for x in summary['production_update_coverage']
                if x['stage'] == stage and x['token_bucket_upper'] == bucket and x['kind'] == kind)


def test_only_observed_production_type_and_bucket_project_requested_workload():
    result = summarize(report(group()), [600, 700, 1300], {'gsq': 2})
    known = row(result, 'gsq', 1024, 'linear_attention')
    assert known['successful_production_samples'] == 2
    assert known['measured_blocks'] == [0]
    assert known['workload_updates'] == 2 * 48 * 2
    assert known['projected_update_seconds'] == 5 * 2 * 48 * 2
    assert row(result, 'gsq', 1536, 'linear_attention')['projected_update_seconds'] is None
    assert row(result, 'gsq', 1024, 'full_attention')['projected_update_seconds'] is None
    assert row(result, 'head', 1024)['projected_update_seconds'] is None
    projection = result['partial_workload_projection']
    assert projection['covered_updates'] == 192
    assert projection['unknown_updates'] == 3*64*2 + 3*3 - 192
    assert projection['complete_job_eta_seconds'] is None
    assert result['workload']['invocations'] == 3  # Never assume 87 when input differs.


def test_smoke_failed_zero_duration_and_wrong_operation_never_fill_coverage():
    smoke = group(stage='smoke_gsq', operation='update', elapsed=.01)
    failed = group(status='failed', elapsed=.01)
    zero = group(elapsed=0.)
    narrower_timer = group(operation='update', elapsed=.01)
    inputs = report(smoke, failed, zero, narrower_timer)
    # Even a misleading production stage label inside a smoke path stays smoke.
    inputs.update(report(group(elapsed=.01), path='02-smoke_rco/performance-report.json'))
    result = summarize(inputs, [600, 700])
    assert result['partial_workload_projection']['covered_update_seconds'] is None
    assert all(x['coverage'] == 'unknown' for x in result['production_update_coverage'])
    assert any(x['family'] == 'smoke' and x['mean_seconds'] == .005 for x in result['measured_groups'])
    assert any(x['status'] == 'failed' for x in result['measured_groups'])


def test_nested_totals_and_checkpoint_costs_are_observed_not_added_to_updates():
    update = group(stage='rco', operation='update', kind=None, block=None, elapsed=12.)
    total = group(stage='rco', operation='stage_total', kind=None, block=None, lengths=(), elapsed=100.)
    checkpoint = group(stage='rco', operation='checkpoint_total', kind=None, block=None, lengths=(), elapsed=30.)
    publish = group(stage='rco', operation='checkpoint_publish', kind=None, block=None, lengths=(), elapsed=20.)
    restore = group(stage='unknown', operation='checkpoint_restore', kind=None, block=None, lengths=(), elapsed=9.)
    result = summarize(report(update, total, checkpoint, publish, restore), [600, 700])
    assert result['partial_workload_projection']['covered_update_seconds'] == 12.
    assert result['observed_stage_totals'][0]['seconds'] == 100.
    assert [x['seconds'] for x in result['observed_checkpoint_costs']] == [30., 20., 9.]
    assert 'observed_total_seconds' not in result
    assert result['partial_workload_projection']['complete_job_eta_seconds'] is None


def test_all_update_bins_still_do_not_manufacture_complete_eta():
    data = report(group(), group(kind='full_attention', block=3))
    for stage in ('embedding', 'head', 'rco'):
        data.update(report(group(stage=stage, operation='update' if stage == 'rco' else 'update_with_input_load',
                                 kind=None, block=None), path=f'{stage}/performance-report.json'))
    result = summarize(data, [600]*87)
    p = result['partial_workload_projection']
    assert p['all_update_buckets_measured'] is True
    assert p['covered_updates'] == 87 * (64+3)
    assert p['unknown_updates'] == 0
    assert p['complete_job_eta_seconds'] is None
    assert any('Packing/export' in caveat for caveat in result['caveats'])


def test_group_means_are_sample_weighted_and_inputs_not_mutated():
    data = report(group(lengths=(600,), elapsed=2.), group(lengths=(600, 700, 800), elapsed=18., block=1))
    before = copy.deepcopy(data)
    result = summarize(data, [600])
    known = row(result, 'gsq', 1024, 'linear_attention')
    assert known['mean_update_seconds'] == 5.
    assert known['successful_production_samples'] == 4
    assert known['measured_blocks'] == [0, 1]
    assert data == before


def test_multi_sequence_cache_moments_do_not_use_square_of_total_tokens():
    cache = group(operation='cache_propagation', lengths=())
    cache.update(tokens=13, tokens_squared=85)
    result = summarize(report(cache), [6, 7])
    assert result['measured_groups'][0]['tokens_squared'] == 6**2+7**2
    assert result['partial_workload_projection']['covered_update_seconds'] is None


@pytest.mark.parametrize('mutation', [
    lambda g: g.update(seconds=float('nan')),
    lambda g: g.update(seconds=float('inf')),
    lambda g: g.update(seconds=-1),
    lambda g: g.update(count=True),
    lambda g: g.update(count=0),
    lambda g: g.update(tokens_squared=1),
    lambda g: g['dimensions'].update(token_bucket_upper=513),
    lambda g: g['dimensions'].update(token_bucket_upper=512),
    lambda g: g['dimensions'].update(block=64),
    lambda g: g['dimensions'].update(kind='unknown_attention'),
])
def test_invalid_observations_fail_closed(mutation):
    bad = group(); mutation(bad)
    with pytest.raises(ValueError):
        summarize(report(bad), [600])


def test_inconsistent_block_type_rejected():
    with pytest.raises(ValueError, match='inconsistent'):
        summarize(report(group(), group(kind='full_attention')), [600])


@pytest.mark.parametrize('lengths', [[], {}, [True], [0], [1], [2.5]])
def test_explicit_token_lengths_are_required(lengths):
    with pytest.raises(ValueError):
        summarize({}, lengths)


def test_no_timings_and_no_raw_log_disclosure():
    result = summarize({'solver.log': 'secret prompt', 'status.json': {'error': 'private error'}}, [20, 50])
    assert result['timing_reports_found'] == 0
    assert result['partial_workload_projection']['covered_updates'] == 0
    assert result['partial_workload_projection']['unknown_updates'] == 2 * 67
    assert 'secret' not in json.dumps(result) and 'private error' not in json.dumps(result)


def test_cli_reads_explicit_lengths_and_creates_offline_json(tmp_path):
    data = tmp_path/'terminal.json'; data.write_text(json.dumps(report(group())))
    lengths = tmp_path/'lengths.json'; lengths.write_text('[600,700]')
    output = tmp_path/'summary.json'
    script = Path(__file__).resolve().parents[1]/'performance_summary.py'
    args = [sys.executable, str(script), '--diagnostics', str(data), '--token-lengths', str(lengths),
            '--gsq-epochs', '3', '--output', str(output)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    result = read_json(output)
    assert result['workload']['epochs']['gsq'] == 3
    assert row(result, 'gsq', 1024, 'linear_attention')['workload_updates'] == 2*48*3
    assert subprocess.run(args, capture_output=True).returncode != 0  # Never clobber a retained summary.


def test_unknown_schema_bad_epochs_and_oversize_input_rejected(tmp_path, monkeypatch):
    import performance_summary
    with pytest.raises(ValueError, match='epoch'):
        summarize({}, [10], {'head': 0})
    with pytest.raises(ValueError, match='Unknown epoch'):
        summarize({}, [10], {'other': 1})
    bad = report(group()); next(iter(bad.values()))['schema'] = 2
    with pytest.raises(ValueError, match='Unsupported'):
        summarize(bad, [600])
    monkeypatch.setattr(performance_summary, 'MAX_INPUT_BYTES', 8)
    path = tmp_path/'large.json'; path.write_bytes(b' '*9)
    with pytest.raises(ValueError, match='limit'):
        read_json(path)
