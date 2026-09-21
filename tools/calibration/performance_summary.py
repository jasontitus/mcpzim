"""Summarize retained solver timings offline; never invent a complete-job ETA.

Inputs are terminal-diagnostics.json and a JSON list of calibration token counts.
Only successful production update groups in matching 512-token buckets support
workload projections. Smoke, failed work and nested overhead timers stay separate.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

MAX_INPUT_BYTES = 32 * 1024**2
MAX_GROUPS = 20000
BLOCK_COUNTS = {'linear_attention': 48, 'full_attention': 16}
UPDATE_OPERATIONS = {'embedding': 'update_with_input_load', 'gsq': 'update_with_input_load',
                     'head': 'update_with_input_load', 'rco': 'update'}


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def read_json(path):
    with Path(path).open('rb') as source:
        data = source.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError('Input exceeds the 32 MiB offline summary limit')
    return json.loads(data)


def is_smoke(stage, source):
    return stage.startswith('smoke') or any(
        part == 'smoke' or part.startswith('smoke_') or '-smoke' in part
        for part in Path(source).parts[:-1])


def parse_groups(diagnostics):
    if not isinstance(diagnostics, dict):
        raise ValueError('Terminal diagnostics must map filenames to report objects')
    groups = []
    for source, report in sorted(diagnostics.items()):
        if not isinstance(source, str):
            raise ValueError('Diagnostic paths must be strings')
        if Path(source).name != 'performance-report.json':
            continue
        if not isinstance(report, dict) or type(report.get('schema')) is not int or report['schema'] != 1 or not isinstance(report.get('groups'), dict):
            raise ValueError(f'Unsupported performance report: {source}')
        for raw in report['groups'].values():
            if len(groups) >= MAX_GROUPS:
                raise ValueError('Too many timing groups')
            if not isinstance(raw, dict):
                raise ValueError('Timing groups must be objects')
            stage, operation, status = (raw.get(k) for k in ('stage', 'operation', 'status'))
            if not isinstance(stage, str) or not stage or not isinstance(operation, str) or not operation or status not in ('completed', 'failed'):
                raise ValueError('Invalid timing stage, operation or status')
            dims = raw.get('dimensions')
            if not isinstance(dims, dict):
                raise ValueError('Timing dimensions must be an object')
            count = integer(raw.get('count'), 'count', 1)
            elapsed = seconds(raw.get('seconds'), 'seconds')
            tokens = integer(raw.get('tokens'), 'tokens')
            squares = integer(raw.get('tokens_squared'), 'tokens_squared')
            bucket = dims.get('token_bucket_upper')
            if bucket is not None:
                integer(bucket, 'token_bucket_upper', 512)
                if bucket % 512 or not count * (bucket-511) <= tokens <= count * bucket:
                    raise ValueError('Token totals disagree with bucket/count')
                if squares * count < tokens * tokens:
                    raise ValueError('Inconsistent token moments')
            block, kind = dims.get('block'), dims.get('kind')
            if block is not None:
                integer(block, 'block')
                if block >= 64:
                    raise ValueError('Block outside the 64-block Qwen workload')
            if kind is not None and kind not in BLOCK_COUNTS:
                raise ValueError('Unknown transformer block type')
            for key in ('min_seconds', 'max_seconds'):
                if key in raw:
                    seconds(raw[key], key)
            minimum, maximum = raw.get('min_seconds'), raw.get('max_seconds')
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError('Invalid timing extrema')
            smoke = is_smoke(stage, source)
            family = 'smoke' if smoke else 'production' if stage in UPDATE_OPERATIONS else 'other'
            groups.append({'source': source, 'stage': stage, 'family': family,
                           'operation': operation, 'status': status, 'block': block,
                           'kind': kind, 'token_bucket_upper': bucket, 'count': count,
                           'seconds': elapsed, 'tokens': tokens, 'tokens_squared': squares,
                           'mean_seconds': elapsed/count,
                           'min_seconds': minimum, 'max_seconds': maximum,
                           'accounting': 'enclosing_total' if operation.endswith('_total') else 'component'})
    return groups


def summarize(diagnostics, token_lengths, epochs=None):
    if not isinstance(token_lengths, list) or not token_lengths:
        raise ValueError('Provide a nonempty JSON list of calibration token lengths')
    for value in token_lengths:
        integer(value, 'token length', 2)
    epoch_counts = dict.fromkeys(UPDATE_OPERATIONS, 1)
    if epochs:
        if set(epochs) - set(epoch_counts):
            raise ValueError('Unknown epoch stage')
        epoch_counts.update(epochs)
    for stage, value in epoch_counts.items():
        integer(value, stage + ' epochs', 1)
    buckets = Counter(((n+511)//512)*512 for n in token_lengths)
    groups = parse_groups(diagnostics)
    observations = {}
    block_kinds = {}
    for group in groups:
        if group['family'] != 'production' or group['status'] != 'completed':
            continue
        stage = group['stage']
        if group['operation'] != UPDATE_OPERATIONS[stage] or group['token_bucket_upper'] is None or group['seconds'] == 0:
            continue
        kind = group['kind'] if stage == 'gsq' else None
        if stage == 'gsq':
            if kind is None or group['block'] is None:
                continue
            previous = block_kinds.setdefault(group['block'], kind)
            if previous != kind:
                raise ValueError('A measured block has inconsistent attention types')
        key = stage, kind, group['token_bucket_upper']
        cell = observations.setdefault(key, {'count': 0, 'seconds': 0., 'blocks': set(), 'sources': set()})
        cell['count'] += group['count']
        cell['seconds'] += group['seconds']
        cell['sources'].add(group['source'])
        if group['block'] is not None:
            cell['blocks'].add(group['block'])
    for kind, expected in BLOCK_COUNTS.items():
        if sum(v == kind for v in block_kinds.values()) > expected:
            raise ValueError('Measured attention block count exceeds model inventory')
    coverage = []
    for stage in UPDATE_OPERATIONS:
        families = BLOCK_COUNTS if stage == 'gsq' else {None: 1}
        for kind, blocks in families.items():
            for bucket, invocations in sorted(buckets.items()):
                expected = invocations * blocks * epoch_counts[stage]
                observed = observations.get((stage, kind, bucket))
                mean = observed['seconds']/observed['count'] if observed else None
                coverage.append({'stage': stage, 'kind': kind, 'token_bucket_upper': bucket,
                    'calibration_invocations': invocations, 'workload_updates': expected,
                    'successful_production_samples': observed['count'] if observed else 0,
                    'measured_blocks': sorted(observed['blocks']) if observed else [],
                    'sources': sorted(observed['sources']) if observed else [],
                    'mean_update_seconds': mean,
                    'projected_update_seconds': mean * expected if mean is not None else None,
                    'coverage': 'measured_bucket' if observed else 'unknown'})
    covered = [row for row in coverage if row['coverage'] == 'measured_bucket']
    unknown = [row for row in coverage if row['coverage'] == 'unknown']
    return {'schema_version': 1, 'status': 'offline_partial_timing_summary',
        'workload': {'model': 'Qwen/Qwen3.8-27B', 'blocks': 64, 'block_types': BLOCK_COUNTS.copy(),
                     'invocations': len(token_lengths), 'tokens': sum(token_lengths),
                     'tokens_squared': sum(n*n for n in token_lengths),
                     'epochs': epoch_counts, 'token_buckets': [
                         {'upper': key, 'invocations': value} for key, value in sorted(buckets.items())]},
        'timing_reports_found': len({g['source'] for g in groups}),
        'measured_groups': groups, 'production_update_coverage': coverage,
        'partial_workload_projection': {
            'scope': 'Whole configured workload update operations only; not remaining work or a complete-job ETA',
            'covered_updates': sum(row['workload_updates'] for row in covered),
            'unknown_updates': sum(row['workload_updates'] for row in unknown),
            'covered_update_seconds': sum(row['projected_update_seconds'] for row in covered) if covered else None,
            'all_update_buckets_measured': not unknown,
            'complete_job_eta_seconds': None},
        'observed_checkpoint_costs': [g for g in groups if g['operation'] in (
            'checkpoint_total', 'checkpoint_publish', 'checkpoint_restore')],
        'observed_stage_totals': [g for g in groups if g['operation'] == 'stage_total'],
        'caveats': [
            'Only completed positive-duration production update samples project workload; smoke and failed timings never fill coverage gaps.',
            'Means transfer only within the same stage, attention type and 512-token bucket. GSQ transfers observed blocks to the same-type block population; this assumption is not an accuracy guarantee.',
            'No extrapolation across missing buckets, stages, hardware or runtime versions. Timing groups alone do not independently verify hardware identity.',
            'Observed samples are not progress counters. Projected work is the whole requested epoch workload, without subtracting completed updates.',
            'stage_total encloses components; checkpoint_total encloses checkpoint_publish. Never add enclosing totals to their children or combine failed and completed throughput.',
            'Checkpoint observations are historical costs, not a predicted future checkpoint count or size. Head optimizer snapshots may be much larger than RCO snapshots.',
            'Packing/export, return transfer, evaluation and future recovery remain unmeasured. Cache propagation, startup and other overhead are not included in update projections.',
            'No confidence interval or complete-job ETA is supported by these observations.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', required=True, type=Path)
    parser.add_argument('--token-lengths', required=True, type=Path)
    for stage in UPDATE_OPERATIONS:
        parser.add_argument('--' + stage + '-epochs', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(read_json(args.diagnostics), read_json(args.token_lengths),
                       {stage: getattr(args, stage + '_epochs') for stage in UPDATE_OPERATIONS})
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as target:
            target.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
