"""Sequential actual-Qwen GPU job, bounded by a shared deadline.

Each stage gets a fresh process to isolate upstream GSQ/RCO Python namespaces
and release CUDA allocations. A completed smoke is required before the full
corpus recipe. This controller never installs packages or provisions resources.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
COSTS = Path(__file__).resolve().parents[1] / 'packing/evidence/qwen-cost-manifest.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def read(path):
    return json.loads(Path(path).read_text())


def commit_valid(checkpoint):
    commit = checkpoint.get('receipt', {}).get('commit', {})
    if (not checkpoint.get('snapshot') or not commit.get('object')
            or type(commit.get('generation')) is not int or commit['generation'] <= 0
            or type(commit.get('bytes')) is not int or commit['bytes'] <= 0
            or not re.fullmatch('[0-9a-f]{64}', commit.get('sha256', ''))):
        raise ValueError('Missing generation-pinned durable checkpoint')
    return checkpoint


def merge_smoke(gsq, rco, longest):
    if any(p.get('status') != 'completed' or p.get('cpu_test_only') is True for p in (gsq, rco)):
        raise ValueError('Actual CUDA smoke components did not complete')
    result = dict(gsq['smoke']); result.update(rco['smoke'])
    for name in ('linear_attention_gsq', 'full_attention_gsq'):
        result[name] = gsq['smoke'].get(name, {})
    if result.get('model') != MODEL or result.get('revision') != REVISION or result.get('sequence_tokens') != longest:
        raise ValueError('Smoke must use original baseline and longest complete invocation')
    for partial in (gsq['smoke'], rco['smoke']):
        cuda = partial.get('cuda', {})
        if cuda.get('available') is not True or not cuda.get('device') or not cuda.get('capability'):
            raise ValueError('Smoke lacks actual CUDA device evidence')
        for key in ('total_memory_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            if type(cuda.get(key)) not in (int, float) or not math.isfinite(cuda[key]) or cuda[key] <= 0:
                raise ValueError('Smoke lacks finite CUDA memory evidence')
        if max(cuda['peak_allocated_bytes'], cuda['peak_reserved_bytes']) > cuda['total_memory_bytes']:
            raise ValueError('Impossible smoke memory evidence')
        if partial.get('model') != MODEL or partial.get('revision') != REVISION or partial.get('sequence_tokens') != longest:
            raise ValueError('Smoke component identity mismatch')
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        result['cuda'][key] = max(gsq['smoke']['cuda'][key], rco['smoke']['cuda'][key])
    for name in ('linear_attention_gsq', 'full_attention_gsq', 'full_model_rco'):
        check = result.get(name, {}); norm = check.get('gradient_norm')
        if check.get('passed') is not True or type(norm) not in (int, float) or not math.isfinite(norm) or norm <= 0:
            raise ValueError('Actual backward/update not proven: ' + name)
        if name != 'full_model_rco' and (type(check.get('loss')) not in (int, float) or not math.isfinite(check['loss'])):
            raise ValueError('Invalid GSQ objective')
    if result['full_model_rco'].get('full_vocabulary') is not True:
        raise ValueError('Smoke requires full-vocabulary RCO objective')
    resume = result.get('checkpoint_resume', {})
    if resume.get('passed') is not True or resume.get('next_update_matches') is not True:
        raise ValueError('Actual CUDA durable resume not proven')
    commit_valid({'snapshot': resume.get('snapshot'), 'receipt': {'commit': resume.get('commit', {})}})
    commit_valid(rco.get('durable_checkpoint', {}))
    result['passed'] = True
    return result


def run_stage(stage, config_path, output, hard_deadline):
    """Forward termination, then enforce bounded cleanup of this child only."""
    command = [sys.executable, '-m', 'solver.run', '--config', str(config_path), '--stage', stage]
    child = subprocess.Popen(command, start_new_session=True)
    old = {}; interrupted = False
    def terminate(_signum=None, _frame=None):
        nonlocal interrupted
        interrupted = True
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM): old[sig] = signal.signal(sig, terminate)
        try:
            code = child.wait(timeout=max(1, hard_deadline - time.time()))
        except subprocess.TimeoutExpired:
            terminate()
            try: child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise TimeoutError('Stage exceeded job deadline; inspect any committed checkpoint')
        if code != 0: raise RuntimeError(f'{stage} failed with exit {code}')
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)
    result = read(Path(output) / (stage + '-report.json'))
    if interrupted:
        result = {**result, 'job_interrupted': True}
    return result


def execute(args, *, stage_runner=run_stage, clock=time.time):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'solver.log' for p in output.iterdir()):
        raise FileExistsError('Fresh job output required; retain previous job and resume its frozen stage explicitly')
    started = clock()
    status = {'schema_version': 1, 'status': 'running', 'started_unix': started,
              'runtime_sha256': args.runtime_sha256, 'input_commit_sha256': args.input_commit_sha256,
              'stages': [], 'progress': {'phase': 'preflight', 'optimizer_updates': 0,
                  'validation_optimizer_updates': 0, 'production_optimizer_updates': 0},
              'packaged_model_validated': False}
    save(output / 'status.json', status)
    try:
        for value in (args.runtime_sha256, args.input_commit_sha256):
            if not re.fullmatch('[0-9a-f]{64}', value): raise ValueError('Expected immutable SHA256 identity')
        if not re.fullmatch(r'runs/[A-Za-z0-9_-]+', args.prefix): raise ValueError('Dedicated run prefix required')
        if not args.bucket or args.deadline_seconds < 300 or args.deadline_seconds > 2700:
            raise ValueError('Job requires bucket and a deadline between300 and2700 seconds')
        if not 30 <= args.checkpoint_seconds <= 300: raise ValueError('Bounded checkpoint cadence required')
        costs = read(COSTS)
        if args.target_bytes < costs['all_q1_serialized_bytes']:
            raise ValueError('Requested budget below complete Q1 model size')
        inputs = Path(args.inputs).resolve()
        receipt = read(inputs / 'restore-validation.json')
        manifest = read(inputs / 'calibration/manifest.json')
        if (receipt.get('status') != 'validated' or receipt.get('input_commit_sha256') != args.input_commit_sha256
                or receipt.get('manifest_sha256') != sha(inputs / 'calibration/manifest.json')
                or manifest.get('status') != 'completed' or manifest.get('model') != MODEL or manifest.get('revision') != REVISION):
            raise ValueError('Prepared input identity mismatch')
        longest = max(item['tokens'] for item in manifest['sequences'])
        if len(manifest['sequences']) != manifest['source_invocation_count']: raise ValueError('Incomplete corpus')
        deadline = started + args.deadline_seconds
        # Leave time for a synchronous immutable checkpoint and wrapper shutdown.
        base = {'inputs': str(inputs), 'input_commit_sha256': args.input_commit_sha256,
                'runtime_sha256': args.runtime_sha256, 'largest_first': True,
                'checkpoint_seconds': args.checkpoint_seconds, 'deadline_unix': deadline - 180,
                'target_bytes': args.target_bytes, 'seed': 42,
                'cost_manifest': {'path': str(COSTS), 'sha256': sha(COSTS)}}
        candidate = {}; warmstarts = {}; boundaries = []; smoke_parts = {}
        durable = None
        for ordinal, label in enumerate(('initialize', 'smoke_gsq', 'smoke_rco', 'embedding', 'gsq', 'head', 'rco')):
            if clock() >= deadline - 180:
                if not status.get('smoke', {}).get('passed') or durable is None:
                    raise TimeoutError('Deadline reached before complete feasibility smoke')
                status['status'] = 'checkpointed'; break
            stage = 'smoke' if label.startswith('smoke_') else label
            directory = output / f'{ordinal:02d}-{label}'
            config = {**base, **candidate, 'output': str(directory),
                      'checkpoint': {'backend': 'gcs', 'bucket': args.bucket,
                         'prefix': args.prefix + '/' + label, 'project': 'tiltastech-zimfo',
                         'staging_dir': str(output)}}
            if label.startswith('smoke_'):
                config.update(smoke_component=label.removeprefix('smoke_'), allow_rtn_boundary_smoke=True)
            if warmstarts: config['warmstart_states'] = warmstarts
            if label == 'rco': config['boundary_reports'] = boundaries
            config_path = output / (label + '-config.json'); save(config_path, config)
            status['progress']['phase'] = label; save(output / 'status.json', status)
            result = stage_runner(stage, config_path, directory, deadline)
            report_path = directory / (stage + '-report.json')
            if result.get('runtime_sha256') != args.runtime_sha256 or result.get('calibration_sha256') != receipt['manifest_sha256']:
                raise ValueError('Stage result identity mismatch')
            if result.get('status') not in ('completed', 'checkpointed_stop'):
                raise ValueError('Stage did not complete or commit a controlled stop')
            stage_info = {'phase': label, 'report': str(report_path), 'sha256': sha(report_path), 'status': result['status']}
            status['stages'].append(stage_info)
            checkpoint_path = directory / 'latest-checkpoint.json'
            if checkpoint_path.exists():
                durable = commit_valid(read(checkpoint_path))
                status['progress']['durable_checkpoint'] = durable
                status['progress']['resume'] = {'kind': 'diagnostic_checkpoint' if stage == 'smoke' else 'production_stage',
                    'config': str(directory / 'frozen-config.json'),
                    'stage': stage, 'snapshot': durable['snapshot'],
                    'commit_generation': durable['receipt']['commit']['generation'],
                    'bucket': args.bucket, 'prefix': config['checkpoint']['prefix'],
                    'identity': durable['receipt'].get('manifest', {}).get('identity'),
                    'configuration_payload': durable['receipt'].get('manifest', {}).get('payloads', {}).get('configuration'),
                    'output_must_be_new': True}
            updates = result.get('updates', result.get('optimizer_updates', result.get('progress', {}).get('global_step', 0)))
            if type(updates) is not int or updates < 0: raise ValueError('Invalid optimizer update count')
            status['progress']['optimizer_updates'] += updates
            counter = 'validation_optimizer_updates' if stage == 'smoke' else 'production_optimizer_updates'
            status['progress'][counter] += updates
            if result['status'] == 'checkpointed_stop' or result.get('job_interrupted'):
                if not status.get('smoke', {}).get('passed') or not checkpoint_path.exists():
                    raise ValueError('Controlled stop lacks smoke or stage checkpoint')
                status['status'] = 'checkpointed'; break
            if label == 'initialize':
                candidate = {'candidate_database': str(directory / 'initial-database.json'),
                             'candidate_archives': {'initial_candidates': str(directory / 'initial-candidates.tar')}}
                if not all(Path(p).is_file() for p in [candidate['candidate_database'], *candidate['candidate_archives'].values()]):
                    raise ValueError('Initialization did not produce complete candidate artifacts')
            elif label.startswith('smoke_'):
                smoke_parts[label] = result
                warmstarts.update(result.get('warmstart_states', {}))
                if label == 'smoke_rco':
                    status['smoke'] = merge_smoke(smoke_parts['smoke_gsq'], result, longest)
                    durable = commit_valid(result['durable_checkpoint'])
                    status['progress']['durable_checkpoint'] = durable
            elif label in ('embedding', 'gsq', 'head'):
                candidate.update(candidate_database=result['candidate_database'], candidate_archives=result['candidate_archives'])
                if label == 'gsq': candidate['final_cache'] = result['final_cache']
                else:
                    if updates <= 0 or result.get('stage') != label: raise ValueError('Learned boundary optimization incomplete')
                    boundaries.append({'path': str(report_path), 'sha256': sha(report_path)})
            elif label == 'rco':
                status.update(status='ready_for_packaging', allocation=result['allocation'],
                              candidate_database=candidate['candidate_database'],
                              packaging_note='Explicit choice export and runtime validation are still required; this is not a finished model')
            save(output / 'status.json', status)
        if status['status'] == 'running': raise ValueError('Job ended without a committed outcome')
        if not status.get('smoke', {}).get('passed') or durable is None: raise ValueError('Missing feasibility evidence')
        status['elapsed_seconds'] = clock() - started
        save(output / 'status.json', status)
        return status
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=clock()-started)
        save(output / 'status.json', status)
        raise


def prepare_resume(status_path, inputs, output, runtime_sha256, *, deadline_seconds=2700,
                   store_factory=None, use_gcloud=False):
    """Recover a pinned production-stage snapshot onto an empty host directory.

    The first restore verifies every payload before producing a launch command.
    The stage CLI deliberately re-verifies/restores it on execution; this costs
    a second bounded download but never trusts an unverified local tensor cache.
    This resumes the interrupted stage, not the remaining orchestration phases.
    """
    status = read(status_path)
    if 'solver_status' in status: status = status['solver_status']
    progress = status['progress']; resume = progress['resume']
    if resume.get('kind') != 'production_stage' or resume.get('stage') not in ('embedding', 'gsq', 'head', 'rco'):
        raise ValueError('Diagnostic smoke checkpoint is not a resumable production phase')
    durable = commit_valid(progress['durable_checkpoint'])
    manifest = durable['receipt']['manifest']; identity = manifest['identity']
    if (runtime_sha256 != identity['runtime_sha256'] or status['runtime_sha256'] != runtime_sha256
            or identity['baseline_repo'] != MODEL or identity['baseline_revision'] != REVISION):
        raise ValueError('Recovery requires the identical original baseline and runtime image')
    if not 300 <= deadline_seconds <= 2700: raise ValueError('Invalid recovery deadline')
    if (manifest['snapshot'] != durable['snapshot'] or manifest['prefix'] != resume['prefix']
            or manifest['bucket'] != resume['bucket']
            or durable['receipt']['commit']['object'] != resume['prefix'] + '/commits/' + durable['snapshot'] + '.json'
            or durable['receipt']['commit']['generation'] != resume['commit_generation']):
        raise ValueError('Recovery receipt disagrees with pinned checkpoint')
    inputs = Path(inputs).resolve(); validation = read(inputs / 'restore-validation.json')
    if (validation.get('status') != 'validated' or validation.get('input_commit_sha256') != status['input_commit_sha256']
            or validation.get('manifest_sha256') != identity['calibration_sha256']
            or sha(inputs / 'calibration/manifest.json') != identity['calibration_sha256']):
        raise ValueError('Cold host inputs differ from calibrated checkpoint identity')
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Recovery requires an empty new destination')
    parent = output.parent
    while not parent.exists(): parent = parent.parent
    # Two verified downloads, temporary bundle, expanded uncompressed archives,
    # and subsequent optimizer serialization can coexist. Do not reclaim prior
    # evidence implicitly; reserve conservatively before starting any transfer.
    payload_bytes = sum(item['bytes'] for item in manifest['payloads'].values())
    required_free = 6 * payload_bytes + 10 * 1024**3
    if shutil.disk_usage(parent).free < required_free:
        raise ValueError(f'Insufficient recovery disk space: require {required_free} free bytes')
    output.mkdir(parents=True, exist_ok=False)
    if store_factory is None:
        from gcs_checkpoints import GCSCheckpointStore, make_client
        store = GCSCheckpointStore(resume['bucket'], resume['prefix'], project='tiltastech-zimfo',
            client=make_client('tiltastech-zimfo', use_gcloud=use_gcloud), staging_dir=output)
    else: store = store_factory(resume)
    restored = output / 'verified-checkpoint'
    actual = store.restore(durable['snapshot'], identity, restored,
                           commit_generation=resume['commit_generation'])
    if actual.get('commit') != durable['receipt']['commit']:
        raise ValueError('Restored commit differs from independently retained receipt')
    config = read(restored / 'configuration')
    if config.get('identity') != identity: raise ValueError('Frozen solver configuration identity mismatch')
    config.update(inputs=str(inputs), output=str(output/'stage'), deadline_unix=time.time()+deadline_seconds-180)
    config['resume_checkpoint'] = {**config['checkpoint'], 'backend': 'gcs',
        'bucket': resume['bucket'], 'prefix': resume['prefix'], 'project': 'tiltastech-zimfo', 'staging_dir': str(output)}
    config['checkpoint'].update(backend='gcs', bucket=resume['bucket'],
                                prefix=resume['prefix'] + '/resume-' + uuid.uuid4().hex[:12],
                                project='tiltastech-zimfo', staging_dir=str(output))
    # Stage restore rebuilds candidate paths from verified archives. An absent
    # placeholder retains the frozen starting database identity during binding.
    config['candidate_database'] = str(output/'rebuilt-by-stage.json')
    for descriptor in config.get('boundary_reports', []):
        matches = [path for path in restored.glob('boundary_report_*') if sha(path) == descriptor['sha256']]
        if len(matches) != 1: raise ValueError('Restored boundary report differs')
        path = matches[0]
        if read(path).get('stage') not in ('embedding', 'head'): raise ValueError('Wrong boundary report stage')
        descriptor['path'] = str(path)
    if 'cost_manifest' in config:
        path = restored/'packing_cost_manifest'
        if sha(path) != config['cost_manifest']['sha256']: raise ValueError('Restored packing costs differ')
        config['cost_manifest']['path'] = str(path)
    for block in config.get('warmstart_states', {}):
        path = restored / ('warmstart_block_' + block)
        if sha(path) != config['warmstart_sha256'][block]: raise ValueError('Restored warm start differs')
        config['warmstart_states'][block] = str(path)
    config_path = output/'resume-config.json'; save(config_path, config)
    command = [sys.executable, '-m', 'solver.run', '--config', str(config_path),
               '--stage', resume['stage'], '--resume', durable['snapshot'],
               '--commit-generation', str(resume['commit_generation'])]
    receipt = {'status': 'verified_stage_recovery_prepared', 'stage': resume['stage'],
               'source_commit': actual['commit'], 'identity': identity,
               'command': command, 'configuration': str(config_path),
               'automatic_remaining_phases': False,
               'payloads_downloaded_again_on_execution': True}
    save(output/'recovery-receipt.json', receipt)
    return receipt


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare-resume':
        parser = argparse.ArgumentParser(description=prepare_resume.__doc__)
        for name in ('status', 'inputs', 'output', 'runtime-sha256'):
            parser.add_argument('--' + name, required=True)
        parser.add_argument('--deadline-seconds', type=int, default=2700)
        parser.add_argument('--use-gcloud', action='store_true')
        parser.add_argument('--execute', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        receipt = prepare_resume(args.status, args.inputs, args.output, args.runtime_sha256,
                                 deadline_seconds=args.deadline_seconds, use_gcloud=args.use_gcloud)
        print(json.dumps(receipt), flush=True)
        if args.execute: subprocess.run(receipt['command'], check=True, timeout=args.deadline_seconds + 90)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'input-commit-sha256', 'output', 'bucket', 'prefix', 'runtime-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--target-bytes', type=int, required=True)
    parser.add_argument('--deadline-seconds', type=int, default=2700)
    parser.add_argument('--checkpoint-seconds', type=int, default=120)
    execute(parser.parse_args())


if __name__ == '__main__': main()
