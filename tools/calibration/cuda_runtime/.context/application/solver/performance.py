"""Synchronized wall timings, retained locally and in the ordinary solver log.

No cloud writes. Updates exclude checkpoints; preparation, publication, restore,
and cache propagation are separate operations. These are observed timings, not
an assertion that a smoke workload predicts the complete quantization recipe.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import sys
import torch


@contextmanager
def measure(output, stage, operation, device=None, **dimensions):
    def synchronize():
        if device is not None and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    synchronize()
    started = time.monotonic()
    status = 'failed'
    try:
        yield
        synchronize()
        status = 'completed'
    finally:
        event = dict(event='solver_timing', stage=stage, operation=operation,
                     status=status, seconds=time.monotonic()-started, **dimensions)
        try:
            print(json.dumps(event, sort_keys=True), flush=True)
            path = Path(output)/'performance-report.json'
            report = json.loads(path.read_text()) if path.exists() else {'schema': 1, 'groups': {}}
            # Fixed-width token buckets distinguish launch overhead from token work;
            # per-update exact token counts/times also remain in the solver log.
            grouping = {k:v for k,v in dimensions.items() if k not in ('tokens', 'tokens_squared', 'sequences', 'sequence', 'epoch', 'global_step')}
            if 'tokens' in dimensions and dimensions.get('sequences', 1)==1:
                grouping['token_bucket_upper'] = ((dimensions['tokens']+511)//512)*512
            key = json.dumps([stage, operation, status, grouping], sort_keys=True)
            group = report['groups'].setdefault(key, dict(stage=stage, operation=operation,
                status=status, dimensions=grouping, count=0, seconds=0., tokens=0, tokens_squared=0))
            group['count'] += 1
            group['seconds'] += event['seconds']
            group['tokens'] += dimensions.get('tokens', 0)
            group['tokens_squared'] += dimensions.get('tokens_squared', dimensions.get('tokens', 0)**2)
            group['min_seconds'] = min(group.get('min_seconds', event['seconds']), event['seconds'])
            group['max_seconds'] = max(group.get('max_seconds', event['seconds']), event['seconds'])
            report['last'] = event
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, sort_keys=True)+'\n')
            temporary.replace(path)
        except Exception as error:
            # Telemetry must not hide an optimizer error or make a committed
            # checkpoint appear unpublished. Missing evidence blocks an ETA;
            # it must not change the optimization/recovery state machine.
            try:
                print(f'solver_timing_record_failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            except Exception:
                pass
