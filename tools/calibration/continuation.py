"""Prepare or explicitly execute continuation outside the pinned solver runtime.

prepare() only inspects and copies small local evidence and emits a review plan.
continue_plan()/the explicit execute subcommand run existing pinned containers on
an already provisioned host. Neither path creates cloud resources or changes the
solver image. This controller is not integrated into the cloud launcher.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import os
import signal
import shutil
import subprocess
import time
import uuid

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
STAGES = ('embedding', 'gsq', 'head', 'rco')
DIRECTORIES = dict(zip(STAGES, ('03-embedding', '04-gsq', '05-head', '06-rco')))
LIMIT = 2 * 1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or symlinked evidence: {path.name}')
    with path.open('rb') as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError('Evidence exceeds small-file limit')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value, data


def is_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def read_cloud_publication(root):
    """Read bridge evidence on the stdlib-only VM host (no model/SDK imports)."""
    root=Path(root)
    if root.is_symlink():raise ValueError('Symlinked cloud receipt root')
    pointer,_=load(root/'latest.json');snapshot=pointer.get('snapshot','')
    if not isinstance(snapshot,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',snapshot):
        raise ValueError('Unsafe publication snapshot')
    directory=root/'snapshots'/snapshot
    if directory.is_symlink() or root.resolve() not in directory.resolve().parents:
        raise ValueError('Unsafe publication directory')
    checkpoint,data=load(directory/'latest-checkpoint.json')
    descriptor=lambda raw:{'sha256':digest(raw),'bytes':len(raw)}
    if descriptor(data)!=pointer.get('receipt') or checkpoint.get('snapshot')!=snapshot:
        raise ValueError('Publication pointer differs from immutable receipt')
    config,config_bytes=load(directory/'frozen-config.json')
    audit,audit_bytes=load(directory/'publication-audit.json')
    manifest=checkpoint['receipt']['manifest'];commit=checkpoint['receipt']['commit']
    encoded=json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if ({k:commit.get(k) for k in ('sha256','bytes')}!=descriptor(encoded)
            or type(commit.get('generation')) is not int or commit['generation']<=0
            or commit.get('object')!=manifest['prefix']+'/commits/'+snapshot+'.json'
            or manifest.get('snapshot')!=snapshot or manifest.get('schema')!=3
            or config.get('identity')!=manifest.get('identity')):
        raise ValueError('Publication manifest/identity mismatch')
    for role,raw in [('configuration',config_bytes),('publication_audit',audit_bytes)]:
        if {k:manifest['payloads'][role].get(k) for k in ('sha256','bytes')}!=descriptor(raw):
            raise ValueError('Publication configuration/audit changed')
    if (audit.get('snapshot')!=snapshot or audit.get('identity')!=manifest['identity']
            or audit.get('changed_configuration_fields')!=['checkpoint']
            or audit.get('cloud_configuration')!=descriptor(config_bytes)
            or audit.get('source_configuration')!={k:manifest['payloads']['source_configuration'].get(k) for k in ('sha256','bytes')}):
        raise ValueError('Publication provenance mismatch')
    return {'directory':str(directory),'checkpoint':checkpoint,'configuration':config,'audit':audit}


def validate_checkpoint(stage, checkpoint, config, config_bytes, runtime, status):
    receipt = checkpoint['receipt']; manifest = receipt['manifest']; commit = receipt['commit']
    identity = manifest['identity']; snapshot = checkpoint['snapshot']
    if (not isinstance(snapshot, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', snapshot)
            or manifest.get('schema') != 3 or snapshot != manifest['snapshot'] or not snapshot.startswith(stage + '-')):
        raise ValueError('Wrong checkpoint stage/schema')
    if (identity != config.get('identity') or identity.get('baseline_repo') != MODEL
            or identity.get('baseline_revision') != REVISION
            or identity.get('runtime_sha256') != runtime or config.get('runtime_sha256') != runtime
            or config.get('input_commit_sha256') != status.get('input_commit_sha256')
            or config.get('stage') != stage):
        raise ValueError('Checkpoint/frozen configuration identity mismatch')
    if not is_hash(identity.get('calibration_sha256')) or not is_hash(status.get('input_commit_sha256')):
        raise ValueError('Missing input identity')
    for key in ('solver_config_sha256', 'candidate_database_sha256'):
        if not is_hash(identity.get(key)):
            raise ValueError('Missing frozen algorithm/candidate identity')
    if not re.fullmatch('[0-9a-f]{40}', identity.get('solver_revision', '')):
        raise ValueError('Missing solver revision')
    prefix = config['checkpoint']['prefix']; bucket = config['checkpoint']['bucket']
    if (not isinstance(prefix, str) or not re.fullmatch(r'runs/[A-Za-z0-9_./-]+', prefix)
            or any(p in ('', '.', '..') for p in prefix.split('/'))):
        raise ValueError('Unsafe checkpoint prefix')
    if (config['checkpoint'].get('backend') != 'gcs' or manifest['prefix'] != prefix
            or manifest['bucket'] != bucket or not bucket
            or commit['object'] != prefix + '/commits/' + snapshot + '.json'
            or type(commit['generation']) is not int or commit['generation'] <= 0):
        raise ValueError('Checkpoint generation or namespace mismatch')
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if commit.get('sha256') != digest(encoded) or commit.get('bytes') != len(encoded):
        raise ValueError('Retained commit does not bind its manifest')
    payloads = manifest['payloads']
    if not {'solver', 'optimizer', 'scheduler', 'rng', 'progress', 'configuration'} <= set(payloads):
        raise ValueError('Incomplete committed state roles')
    descriptor = payloads['configuration']
    if descriptor.get('sha256') != digest(config_bytes) or descriptor.get('bytes') != len(config_bytes):
        raise ValueError('Frozen configuration differs from committed payload')
    return manifest


def prepare(job_dir, output, runtime_image, inputs, recovery_output, deadline_seconds=2700):
    """Prepare locally; remote receipt/payload verification remains mandatory."""
    if not re.fullmatch(r'[^\s@]+@sha256:[0-9a-f]{64}', runtime_image):
        raise ValueError('Use the identical digest-pinned runtime image')
    if type(deadline_seconds) is not int or not 300 <= deadline_seconds <= 2700:
        raise ValueError('Recovery deadline must be 300..2700 seconds')
    runtime = runtime_image.split('@sha256:')[1]
    root = Path(job_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Retained job must be a real directory')
    root = root.resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preparation requires a new output directory')
    if output == root or root in output.parents:
        raise ValueError('Preserve the retained job unchanged; prepare outside it')
    for path in (inputs, recovery_output):
        if not Path(path).is_absolute():
            raise ValueError('Planned host paths must be absolute')
    source_bytes = {}
    def evidence_file(path):
        value, data = load(path)
        source_bytes[Path(path)] = data
        return value, data
    status, status_bytes = evidence_file(root/'status.json')
    if status.get('runtime_sha256') != runtime:
        raise ValueError('Wrong runtime for retained job')
    if status.get('smoke', {}).get('passed') is not True:
        raise ValueError('Completed feasibility smoke is required before production continuation')
    candidates = []
    saved = {'original-status.json': status_bytes}
    for stage, name in DIRECTORIES.items():
        directory = root/name
        if directory.is_symlink():
            raise ValueError('Symlinked stage directory')
        latest = directory/'latest-checkpoint.json'
        if status.get('checkpoint_mode') == 'local-spool':
            publication = root/'cloud'/stage
            if not (publication/'latest.json').exists():continue
            directory = Path(read_cloud_publication(publication)['directory'])
            latest = directory/'latest-checkpoint.json'
        if not latest.exists():
            continue
        checkpoint, checkpoint_bytes = evidence_file(latest)
        config, config_bytes = evidence_file(directory/'frozen-config.json')
        manifest = validate_checkpoint(stage, checkpoint, config, config_bytes, runtime, status)
        candidates.append((stage, checkpoint, config, manifest))
        saved[name+'/latest-checkpoint.json'] = checkpoint_bytes
        saved[name+'/frozen-config.json'] = config_bytes
    if not candidates and status.get('source_checkpoint_stage') in STAGES:
        fallback=status['source_checkpoint_stage']; directory=root/'source-checkpoint'
        if directory.is_symlink():raise ValueError('Symlinked source checkpoint directory')
        checkpoint,checkpoint_bytes=evidence_file(directory/'latest-checkpoint.json')
        config,config_bytes=evidence_file(directory/'frozen-config.json')
        manifest=validate_checkpoint(fallback,checkpoint,config,config_bytes,runtime,status)
        candidates.append((fallback,checkpoint,config,manifest))
        saved[DIRECTORIES[fallback]+'/latest-checkpoint.json']=checkpoint_bytes
        saved[DIRECTORIES[fallback]+'/frozen-config.json']=config_bytes
    if not candidates:
        raise ValueError('No committed production checkpoint; diagnostic smoke cannot resume production')
    stage, checkpoint, config, manifest = candidates[-1]
    if len({item[3]['identity']['calibration_sha256'] for item in candidates}) != 1:
        raise ValueError('Stage checkpoints disagree on calibration identity')
    # Root status can lag a publication, so use the deepest validated stage's
    # actual latest receipt. Do not infer current progress from file timestamps.
    required = ['embedding'] if stage in ('gsq', 'head', 'rco') else []
    if stage == 'rco': required.append('head')
    evidence = []
    for boundary in ('embedding', 'head'):
        path = root/DIRECTORIES[boundary]/(boundary+'-report.json')
        entries = [v for v in status.get('stages', []) if v.get('phase') == boundary and v.get('status') == 'completed']
        if len(entries) > 1:
            raise ValueError('Ambiguous completed boundary report')
        if not path.exists():
            if boundary in required or entries:
                raise ValueError('Missing completed boundary report: ' + boundary)
            continue
        report, data = evidence_file(path)
        if report.get('status') != 'completed':
            if boundary in required or entries:
                raise ValueError('Required boundary report is incomplete: ' + boundary)
            continue
        actual = digest(data)
        bindings = [v.get('sha256') for v in entries]
        bindings += [v.get('sha256') for v in config.get('boundary_reports', [])
                     if Path(v.get('path', '')).name == boundary+'-report.json']
        descriptor = manifest['payloads'].get('boundary_report_'+boundary)
        if descriptor: bindings.append(descriptor.get('sha256'))
        if not bindings:
            # The child may have finished while its controller had not yet
            # recorded the report. Resume the terminal snapshot to re-emit it.
            if boundary in required:
                raise ValueError('Completed boundary report lacks retained hash binding: ' + boundary)
            continue
        if any(value != actual for value in bindings):
            raise ValueError('Changed completed boundary report: ' + boundary)
        if (report.get('stage') != boundary or report.get('runtime_sha256') != runtime
                or report.get('calibration_sha256') != manifest['identity']['calibration_sha256']
                or type(report.get('updates')) is not int or report['updates'] <= 0
                or not is_hash(report.get('candidate_sha256'))):
            raise ValueError('Boundary report identity or learned-update proof missing')
        relative = 'boundary-reports/'+boundary+'-report.json'
        saved[relative] = data
        evidence.append({'stage': boundary, 'path': relative, 'sha256': actual,
                         'candidate_sha256': report['candidate_sha256'], 'updates': report['updates']})
    reconciled = copy.deepcopy(status)
    reconciled['progress'] = {**reconciled.get('progress', {}), 'phase': stage,
        'durable_checkpoint': checkpoint,
        'resume': {'kind': 'production_stage', 'stage': stage,
                   'config': str(output/DIRECTORIES[stage]/'frozen-config.json'),
                   'snapshot': checkpoint['snapshot'], 'commit_generation': checkpoint['receipt']['commit']['generation'],
                   'bucket': manifest['bucket'], 'prefix': manifest['prefix'],
                   'identity': manifest['identity'], 'output_must_be_new': True}}
    saved['reconciled-status.json'] = (json.dumps(reconciled, indent=2, allow_nan=False)+'\n').encode()
    completed_entries = [v for v in status.get('stages', []) if v.get('phase') == stage and v.get('status') == 'completed']
    if len(completed_entries) > 1:
        raise ValueError('Ambiguous completed stage evidence')
    completed = bool(completed_entries)
    if completed:
        report, data = evidence_file(root/DIRECTORIES[stage]/(stage+'-report.json'))
        if (completed_entries[0].get('sha256') != digest(data) or report.get('status') != 'completed'
                or report.get('runtime_sha256') != runtime
                or report.get('calibration_sha256') != manifest['identity']['calibration_sha256']):
            raise ValueError('Completed stage report identity/hash mismatch')
        saved[DIRECTORIES[stage]+'/'+stage+'-report.json'] = data
    command = ['python', '-m', 'solver.job', 'prepare-resume', '--status', str(output/'reconciled-status.json'),
               '--inputs', str(inputs), '--output', str(recovery_output), '--runtime-sha256', runtime,
               '--deadline-seconds', str(deadline_seconds)]
    result = {'schema_version': 1, 'status': 'continuation_preparation_only',
        'runtime_image': runtime_image, 'runtime_sha256': runtime, 'selected_stage': stage,
        'selected_checkpoint': checkpoint['receipt']['commit'], 'identity': manifest['identity'],
        'recovery_required_free_bytes': 6*sum(v['bytes'] for v in manifest['payloads'].values())+10*1024**3,
        'disk_headroom_verified': False,
        'cloud_launcher_integration': False,
        'real_cuda_continuation_validated': False,
        'root_checkpoint_reconciled': status.get('progress', {}).get('durable_checkpoint') != checkpoint,
        'completed_stage_in_root': completed, 'resume_terminal_stage_even_if_no_new_updates': completed,
        'completed_boundary_reports': evidence,
        'remaining_stages_after_selected_completes': list(STAGES[STAGES.index(stage)+1:]),
        'preparation_command_inside_identical_runtime': command,
        'execution_controller': 'continuation.py execute; requires independent review before paid continuation',
        'command_preconditions': ['Run in the exact pinned image; mount the plan at its recorded absolute path and original inputs read-only.',
            'Recovery destination must be new with sufficient disk headroom; GCS credentials and remote verification are required.',
            'This command restores and verifies only. Review recovery-receipt.json before any separately authorized stage execution.'],
        'automatic_remaining_phases': False, 'remote_checkpoint_verified': False, 'commands_executed': False,
        'gaps': ['Preparation does not execute stages; the separate external execute controller requires independent review and an already provisioned host.',
            'A completed stage may need terminal resume to recreate portable candidates/cache/report with zero optimizer updates.',
            'Preserved reports must be explicitly carried into subsequent stage configuration and checked against candidate hashes.',
            'Do not change pinned solver source/image or frozen algorithm identity to accommodate continuation.',
            'Small local evidence is preserved here; remote generation-pinned payload verification is still pending.'],
        'files': {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in saved.items()}}
    # Recheck sources before exposing a prepared plan. The caller must retain a
    # quiescent job; concurrent mutation is rejected where observed, not locked.
    if any(load(path)[1] != data for path, data in source_bytes.items()):
        raise ValueError('Job changed during preparation; wait for a quiescent retained job')
    output.mkdir(parents=True, exist_ok=False)
    for name, data in saved.items():
        target = output/name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    with (output/'continuation-plan.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return result



def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside(path, root):
    path = Path(path).resolve()
    if root not in path.parents or path == root:
        raise ValueError('All mounted paths must be strictly inside the explicit workspace')
    return path


def docker_spec(image, workspace, inputs, command, name, *, gpu, log):
    argv = ['docker', 'run', '--name', name, '--rm', '--pull=never', '--network=host',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--shm-size=8g',
            '--mount', f'type=bind,src={workspace},dst={workspace}',
            '--mount', f'type=bind,src={inputs},dst={inputs},readonly']
    if gpu: argv += ['--gpus=all']
    return {'argv': argv + [image] + command, 'container_name': name, 'log': str(log)}


def run_container(spec, hard_deadline):
    """Execute only on an already provisioned host; terminate this owned container.

    Docker must already contain the exact image (--pull=never). No provisioning,
    package installs or credentials are included in this controller.
    """
    if spec.get('publisher'):
        return run_pipeline(spec,hard_deadline)
    def on_signal(*_):
        raise InterruptedError('Continuation interrupted; retain committed state')
    old = {}; completed = False
    ledger = Path(spec['owned_container_ledger']) if spec.get('owned_container_ledger') else None
    ownership = {'container_name':spec['container_name'], 'owner_run_id':spec.get('owner_run_id')}
    if ledger:
        if ledger.exists(): raise ValueError('Owned-container ledger must be reconciled before another container starts')
        save_json(ledger, ownership)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM): old[sig] = signal.signal(sig, on_signal)
        with Path(spec['log']).open('ab') as log:
            subprocess.run(spec['argv'], stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, hard_deadline-time.time()))
        completed = True
    except BaseException:
        # subprocess.run kills the CLI on timeout; stop its owned container too.
        # The pinned solver receives SIGTERM and can commit an update boundary.
        try:
            stopped = subprocess.run(['docker', 'stop', '--time', '90', spec['container_name']],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=100, check=False)
            completed = getattr(stopped, 'returncode', None) == 0
        except Exception:
            # Keep the original stage failure visible. Host runtime limits are
            # still required if Docker itself cannot stop the owned container.
            pass
        raise
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)
        if ledger and completed and ledger.exists():
            actual, _ = load(ledger)
            if actual == ownership: ledger.unlink()


def run_pipeline(spec, hard_deadline, *, process_factory=subprocess.Popen,
                 clock=time.time, sleep=time.sleep, stop=subprocess.run):
    """Supervise separate producer/uploader processes; never train in a thread.

    Complete local commits survive uploader failure on the retained disk. A
    stage succeeds only after the publisher has drained its final snapshot.
    """
    if clock() >= hard_deadline:raise TimeoutError('Checkpoint pipeline deadline already expired')
    publisher=spec['publisher']; processes=[]; handlers={}; cleanup_ok=True; completed=False
    ledger=Path(spec['owned_container_ledger']) if spec.get('owned_container_ledger') else None
    ownership={'container_name':spec['container_name'],'publisher_container_name':publisher['container_name'],
               'owner_run_id':spec.get('owner_run_id')}
    if ledger:
        if ledger.exists():raise ValueError('Owned-container ledger requires reconciliation')
        save_json(ledger,ownership)
    def interrupted(*_):raise InterruptedError('Checkpoint pipeline interrupted; retain local and cloud commits')
    def bounded_pause():
        if clock() >= hard_deadline:raise TimeoutError('Checkpoint pipeline exceeded its absolute deadline')
        sleep(max(0,min(1,hard_deadline-clock())))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
        with Path(publisher['log']).open('ab') as publisher_log, Path(spec['log']).open('ab') as producer_log:
            upload=process_factory(publisher['argv'],stdout=publisher_log,stderr=subprocess.STDOUT)
            processes.append((upload,publisher['container_name']))
            if clock() >= hard_deadline:raise TimeoutError('Checkpoint pipeline deadline expired before producer launch')
            producer=process_factory(spec['argv'],stdout=producer_log,stderr=subprocess.STDOUT)
            processes.append((producer,spec['container_name']))
            while producer.poll() is None:
                if upload.poll() is not None:raise RuntimeError('Checkpoint publisher exited before producer; retain local spool')
                bounded_pause()
            save_json(spec['producer_done'],{'producer_returncode':producer.returncode})
            while upload.poll() is None:bounded_pause()
            if producer.returncode:
                raise subprocess.CalledProcessError(producer.returncode,spec['argv'])
            if upload.returncode:
                raise RuntimeError('Final checkpoint publication failed; retain local spool')
            completed=True
    finally:
        # The host reserves120s beyond this deadline. Two owned-container stops
        # consume at most55s each; preserve the ledger if cleanup is uncertain.
        for process,name in reversed(processes):
            # A disconnected Docker CLI may exit while its daemon container is
            # still running. On every abnormal outcome stop both owned names.
            if not completed or process.poll() is None:
                try:
                    result=stop(['docker','stop','--time','45',name],stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,timeout=55,check=False)
                    cleanup_ok=cleanup_ok and result.returncode==0
                except Exception:cleanup_ok=False
                if process.poll() is None:process.kill()
                try:process.wait(timeout=5)
                except Exception:cleanup_ok=False
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if ledger and cleanup_ok and ledger.exists():
            try:
                if load(ledger)[0]==ownership:ledger.unlink()
            except (OSError,ValueError):pass


def continue_plan(plan_dir, output, workspace, inputs, *, deadline_seconds=2700,
                  checkpoint_reserve_seconds=900,
                  write_prefix=None, owned_container_ledger=None, owner_run_id=None,
                  checkpoint_mode='gcs', publisher_script=None, publisher_sha256=None,
                  spool_min_free_bytes=None, stop_after_stage=None,
                  runner=run_container, clock=time.time):
    """Run a prepared stage and remaining phases using the unchanged pinned image.

    Injectable runners are used in CPU tests. Calling this function authorizes
    stage execution on an existing host; prepare() by itself never executes it.
    """
    if type(deadline_seconds) is not int or not 1200 <= deadline_seconds <= 86400:
        raise ValueError('Continuation deadline must be 1200..86400 seconds')
    if (type(checkpoint_reserve_seconds) is not int or checkpoint_reserve_seconds < 900
            or checkpoint_reserve_seconds > deadline_seconds-300):
        raise ValueError('Checkpoint reserve must be at least 900 seconds and leave 300 seconds for work')
    if stop_after_stage not in (None,'gsq'):
        raise ValueError('Only the explicit GSQ stage handoff is supported')
    if checkpoint_mode not in ('gcs','local-spool'):raise ValueError('Unknown checkpoint mode')
    if checkpoint_mode == 'local-spool':
        if (not publisher_script or ',' in str(publisher_script) or Path(publisher_script).is_symlink() or not Path(publisher_script).is_file()
                or not is_hash(publisher_sha256) or digest(Path(publisher_script).read_bytes()) != publisher_sha256):
            raise ValueError('Local-spool mode requires the exact reviewed publisher script')
        if type(spool_min_free_bytes) is not int or spool_min_free_bytes <= 0:
            raise ValueError('Local-spool mode requires an explicit reviewed disk-space budget')
    workspace = Path(workspace).resolve()
    if not workspace.is_dir() or workspace == Path(workspace.anchor) or ',' in str(workspace):
        raise ValueError('Explicit non-root existing workspace required')
    if write_prefix is not None:
        if not re.fullmatch(r'runs/zimfo-gpu-[0-9a-f]{12}',write_prefix):
            raise ValueError('New cloud writes require the exact owned run namespace')
        if owner_run_id != write_prefix.removeprefix('runs/'):
            raise ValueError('Write namespace and owner run differ')
    if owned_container_ledger is not None:
        owned_container_ledger = inside(owned_container_ledger, workspace)
        if not isinstance(owner_run_id,str) or not re.fullmatch(r'zimfo-gpu-[0-9a-f]{12}',owner_run_id):
            raise ValueError('Container ledger requires explicit owned cloud run')
        if owned_container_ledger.exists():raise ValueError('Existing owned-container ledger requires cleanup')
    plan_dir = inside(plan_dir, workspace); output = inside(output, workspace); inputs = inside(inputs, workspace)
    if not inputs.is_dir() or ',' in str(inputs): raise ValueError('Existing prepared input directory required')
    if inputs == output or inputs in output.parents or inputs == plan_dir or inputs in plan_dir.parents:
        raise ValueError('Plans/output must remain outside read-only prepared inputs')
    if owned_container_ledger and (inputs in owned_container_ledger.parents or plan_dir in owned_container_ledger.parents
                                  or output in owned_container_ledger.parents):
        raise ValueError('Supervisor ledger must be outside inputs, plan and controller output')
    if output.exists(): raise FileExistsError('Every continuation needs a new output directory')
    if plan_dir in output.parents:raise ValueError('Preserve the prepared plan unchanged')
    plan, _ = load(plan_dir/'continuation-plan.json')
    if plan.get('status') != 'continuation_preparation_only': raise ValueError('Expected prepared continuation plan')
    for name, descriptor in plan['files'].items():
        path = inside(plan_dir/name, plan_dir)
        _, data = load(path)
        if digest(data) != descriptor['sha256'] or len(data) != descriptor['bytes']:
            raise ValueError('Prepared continuation evidence changed')
    image = plan['runtime_image']; runtime = plan['runtime_sha256']
    if not re.fullmatch(r'[^\s@]+@sha256:'+re.escape(runtime), image):
        raise ValueError('Plan lost immutable runtime binding')
    selected = plan['selected_stage']
    if selected not in STAGES: raise ValueError('Invalid selected production stage')
    if stop_after_stage is not None and STAGES.index(selected)>STAGES.index(stop_after_stage):
        raise ValueError('Requested stop stage precedes the selected checkpoint stage')
    source_status, _ = load(plan_dir/'reconciled-status.json')
    if source_status.get('runtime_sha256') != runtime or source_status.get('smoke', {}).get('passed') is not True:
        raise ValueError('Prepared status lost runtime/smoke evidence')
    checkpoint = source_status['progress']['durable_checkpoint']
    original_config, original_bytes = load(plan_dir/DIRECTORIES[selected]/'frozen-config.json')
    manifest = validate_checkpoint(selected, checkpoint, original_config, original_bytes, runtime, source_status)
    if manifest['identity'] != plan['identity'] or checkpoint['receipt']['commit'] != plan['selected_checkpoint']:
        raise ValueError('Plan selected identity differs from frozen evidence')
    required_free=6*sum(v['bytes'] for v in manifest['payloads'].values())+10*1024**3
    if checkpoint_mode == 'local-spool':required_free=max(required_free,spool_min_free_bytes)
    if shutil.disk_usage(workspace).free < required_free:
        raise ValueError(f'Insufficient recovery disk headroom: require {required_free} free bytes before continuation')
    started = clock(); deadline = started+deadline_seconds
    # Keep the owned-container cleanup (at most 100s) inside this controller's
    # absolute cap. Stop work earlier: the measured block-boundary path may
    # finish one checkpoint before observing the stop and then save again.
    container_deadline = deadline-120
    work_deadline = deadline-checkpoint_reserve_seconds
    attempt = 'continue-'+uuid.uuid4().hex[:12]
    base_prefix = manifest['prefix'].split('/')[0:2]
    prefix = (write_prefix or '/'.join(base_prefix))+'/'+attempt
    output.mkdir(parents=True, exist_ok=False)
    status = copy.deepcopy(source_status)
    # A new explicitly requested attempt owns its stop policy. Prior handoff
    # metadata must not masquerade as this attempt's outcome.
    status.pop('stop_reason',None);status.pop('stage_handoff',None)
    status.update(status='running', continuation_attempt=attempt, runtime_image=image,
                  started_unix=started, packaged_model_validated=False,
                  checkpoint_mode=checkpoint_mode, publisher_sha256=publisher_sha256,
                  requested_stop_after_stage=stop_after_stage,
                  runtime_budget={'seconds':deadline_seconds,'absolute_deadline':deadline,
                    'work_deadline':work_deadline,'container_deadline':container_deadline,
                    'checkpoint_reserve_seconds':checkpoint_reserve_seconds,
                    'automatic_extension':False})
    status['stages'] = [item for item in status.get('stages', [])
                        if item.get('phase') not in STAGES[STAGES.index(selected):]]
    status['progress'] = {'phase':'preparing_recovery', 'durable_checkpoint':checkpoint,
        'resume':copy.deepcopy(source_status['progress']['resume']), 'optimizer_update_totals_known':False}
    # A failed restore/model load must still leave the source checkpoint usable.
    source_dir = output/'source-checkpoint'; source_dir.mkdir()
    (source_dir/'frozen-config.json').write_bytes(original_bytes)
    save_json(source_dir/'latest-checkpoint.json', checkpoint)
    status['progress']['resume']['config'] = str(source_dir/'frozen-config.json')
    status['source_checkpoint_stage'] = selected
    boundaries = {}
    for item in plan['completed_boundary_reports']:
        if item['stage'] not in ('embedding','head') or item['path'] not in plan['files']:
            raise ValueError('Invalid prepared boundary report reference')
        path = inside(plan_dir/item['path'],plan_dir); _, data = load(path)
        if digest(data) != item['sha256']: raise ValueError('Boundary evidence changed')
        # Preserve earlier reports where prepare() expects them, while leaving
        # the selected solver stage's output empty until solver.run creates it.
        if STAGES.index(item['stage']) < STAGES.index(selected):
            target = output/DIRECTORIES[item['stage']]/(item['stage']+'-report.json')
        else: target = output/'prior-boundaries'/(item['stage']+'-report.json')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        boundaries[item['stage']] = {'path':str(target),'sha256':item['sha256']}
    save_json(output/'status.json',status)
    def reconcile(stage):
        directory = output/DIRECTORIES[stage]
        if checkpoint_mode == 'local-spool':
            local=directory/'latest-checkpoint.json'
            if local.exists():
                local_checkpoint,_=load(local)
                status['progress']['retained_local_checkpoint']={
                    'stage':stage,'snapshot':local_checkpoint.get('snapshot'),
                    'receipt':str(local),'store':str(output/'spool'/stage),
                    'cloud_durability_inferred':False}
            publication = output/'cloud'/stage
            if not (publication/'latest.json').exists():return False
            directory=Path(read_cloud_publication(publication)['directory'])
        latest = directory/'latest-checkpoint.json'
        if not latest.exists(): return False
        actual, _ = load(latest); frozen, data = load(directory/'frozen-config.json')
        bound = validate_checkpoint(stage, actual, frozen, data, runtime, status)
        if bound['identity']['calibration_sha256'] != manifest['identity']['calibration_sha256']:
            raise ValueError('Continued stage changed calibration identity')
        status['progress'].update(durable_checkpoint=actual, resume={
            'kind':'production_stage','stage':stage,'config':str(directory/'frozen-config.json'),
            'snapshot':actual['snapshot'],'commit_generation':actual['receipt']['commit']['generation'],
            'bucket':bound['bucket'],'prefix':bound['prefix'],'identity':bound['identity'],'output_must_be_new':True})
        return True
    active = None
    def execute_container(spec, hard_deadline):
        if owned_container_ledger:
            spec.update(owned_container_ledger=str(owned_container_ledger),owner_run_id=owner_run_id)
        return runner(spec,hard_deadline)
    try:
        recovery = output/'recovery'
        command = ['python','-m','solver.job','prepare-resume','--status',str(plan_dir/'reconciled-status.json'),
                   '--inputs',str(inputs),'--output',str(recovery),'--runtime-sha256',runtime,
                   '--deadline-seconds',str(min(deadline_seconds,2700))]
        spec = docker_spec(image,workspace,inputs,command,attempt+'-prepare',gpu=False,log=output/'controller.log')
        spec.update(action='prepare', output=str(recovery), stage=selected)
        execute_container(spec,min(container_deadline,started+2700))
        receipt, _ = load(recovery/'recovery-receipt.json')
        if (receipt.get('status') != 'verified_stage_recovery_prepared' or receipt.get('stage') != selected
                or receipt.get('source_commit') != checkpoint['receipt']['commit']
                or receipt.get('identity') != manifest['identity']):
            raise ValueError('Pinned runtime did not verify the selected remote checkpoint')
        recovery_config = inside(receipt['configuration'],recovery)
        config, _ = load(recovery_config)
        if config.get('identity') != manifest['identity'] or config.get('runtime_sha256') != runtime:
            raise ValueError('Recovered configuration changed frozen identity')
        if (config['resume_checkpoint']['prefix'] != manifest['prefix']
                or config['resume_checkpoint']['bucket'] != manifest['bucket']):
            raise ValueError('Recovery redirected source checkpoint')
        candidate = {}
        for stage in STAGES[STAGES.index(selected):]:
            if clock() >= work_deadline:
                status['status']='checkpointed'; break
            active = stage; directory = output/DIRECTORIES[stage]
            if stage == selected:
                stage_config = copy.deepcopy(config)
                stage_config['candidate_database'] = str(directory/'rebuilt-by-stage.json')
            else:
                stage_config = {k:copy.deepcopy(v) for k,v in config.items()
                                if k not in ('identity','resume_checkpoint','candidate_database','candidate_archives','final_cache','boundary_reports')}
                if stage in ('head','rco'):
                    # Residency settings are stage-specific scientific fields.
                    # Remove them only while forming a NEW stage identity;
                    # never rewrite a selected checkpoint's frozen identity.
                    stage_config.pop('gsq_memory_mode',None)
                    stage_config.pop('gsq_execution_device',None)
                stage_config.update(candidate)
                stage_config['boundary_reports'] = [boundaries[k] for k in ('embedding','head') if k in boundaries]
            stage_config.update(stage=stage, output=str(directory), inputs=str(inputs), deadline_unix=work_deadline)
            stage_config['checkpoint'] = {'backend':'gcs','bucket':manifest['bucket'],
                'prefix':prefix+'/'+stage,'project':'tiltastech-zimfo','staging_dir':str(output)}
            if checkpoint_mode == 'local-spool':
                cloud_target={**stage_config['checkpoint'],'prefix':prefix}
                stage_config['checkpoint']={'backend':'local','path':str(output/'spool'/stage)}
            config_path = output/(stage+'-config.json'); save_json(config_path,stage_config)
            status['progress']['phase']=stage; save_json(output/'status.json',status)
            command = ['python','-m','solver.run','--config',str(config_path),'--stage',stage]
            if stage == selected:
                command += ['--resume',checkpoint['snapshot'],'--commit-generation',str(checkpoint['receipt']['commit']['generation'])]
            spec=docker_spec(image,workspace,inputs,command,attempt+'-'+stage,gpu=True,log=output/'solver.log')
            spec.update(action='stage',stage=stage,config=str(config_path),output=str(directory))
            if checkpoint_mode == 'local-spool':
                if Path(publisher_script).is_symlink() or digest(Path(publisher_script).read_bytes())!=publisher_sha256:
                    raise ValueError('Reviewed publisher changed before stage launch')
                target_file=output/(stage+'-publication-target.json');save_json(target_file,cloud_target)
                done=output/(stage+'-producer-done.json')
                publisher_command=['python','/opt/zimfo-controller/checkpoint_bridge.py',
                    '--local-root',stage_config['checkpoint']['path'],'--receipt-root',str(output/'cloud'/stage),
                    '--target-json',str(target_file),'--stage',stage,'--producer-done',str(done),
                    '--watch-seconds',str(max(1,int(container_deadline-clock()))),'--interval-seconds','300']
                publisher=docker_spec(image,workspace,inputs,publisher_command,attempt+'-'+stage+'-publisher',
                                      gpu=False,log=output/(stage+'-publisher.log'))
                at=publisher['argv'].index(image)
                publisher['argv'][at:at]=['--mount',f'type=bind,src={Path(publisher_script).resolve()},dst=/opt/zimfo-controller/checkpoint_bridge.py,readonly']
                spec.update(publisher=publisher,producer_done=str(done))
            execute_container(spec,container_deadline)
            report_path = directory/(stage+'-report.json'); report, report_bytes=load(report_path)
            if (report.get('runtime_sha256') != runtime or report.get('calibration_sha256') != manifest['identity']['calibration_sha256']
                    or report.get('status') not in ('completed','checkpointed_stop')):
                raise ValueError('Continued stage failed identity/outcome checks')
            if checkpoint_mode == 'local-spool':
                local_checkpoint,_=load(directory/'latest-checkpoint.json')
                publication=read_cloud_publication(output/'cloud'/stage)
                published=publication['checkpoint']
                local_manifest=json.dumps(local_checkpoint['receipt'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                if (published['snapshot'] != local_checkpoint['snapshot']
                        or publication['audit']['source_manifest']!={'sha256':digest(local_manifest),'bytes':len(local_manifest)}
                        or published['receipt']['manifest']['identity']!=local_checkpoint['receipt']['identity']):
                    raise ValueError('Final local checkpoint has not drained to GCS')
            if not reconcile(stage): raise ValueError('Continued stage lacks committed checkpoint')
            status['stages'].append({'phase':stage,'report':str(report_path),'sha256':digest(report_bytes),'status':report['status']})
            if report['status']=='checkpointed_stop':
                status['status']='checkpointed'; break
            if stage in ('embedding','head'):
                if (report.get('stage')!=stage or type(report.get('updates')) is not int or report['updates']<=0
                        or not is_hash(report.get('candidate_sha256'))):
                    raise ValueError('Completed boundary lacks learned-update evidence')
                boundaries[stage]={'path':str(report_path),'sha256':digest(report_bytes)}
            if stage in ('embedding','gsq','head'):
                database = inside(report['candidate_database'],output)
                if not database.is_file(): raise ValueError('Missing continued candidate database')
                archives = {key:str(inside(value,output)) for key,value in report['candidate_archives'].items()}
                if not archives or not all(Path(value).is_file() for value in archives.values()):
                    raise ValueError('Missing continued candidate archives')
                candidate={'candidate_database':str(database),'candidate_archives':archives}
                if stage=='gsq':
                    cache=inside(report['final_cache'],output)
                    if not cache.is_dir(): raise ValueError('Missing propagated final cache')
                    candidate['final_cache']=str(cache)
            else:
                if not isinstance(report.get('allocation'),dict): raise ValueError('Missing final RCO allocation')
                status.update(status='quantization_stages_completed',allocation=report['allocation'],
                    candidate_artifacts_directory=str(directory),
                    packaging_note='Packing/export, candidate inventory reconstruction after RCO-only resume, and runtime/quality validation remain separate.')
            if stage==stop_after_stage:
                # Reaching this branch requires a completed report, verified
                # cloud publication and the complete candidate/cache handoff.
                # A later attempt terminal-resumes GSQ before starting head.
                status.update(status='checkpointed',stop_reason='requested_stage_boundary',
                    stage_handoff={'completed_stage':stage,'next_stage':'head',
                        'remaining_stages':['head','rco'],'terminal_recovery_required':True,
                        'report':str(report_path),'report_sha256':digest(report_bytes),
                        'snapshot':status['progress']['durable_checkpoint']['snapshot'],
                        'commit':status['progress']['durable_checkpoint']['receipt']['commit'],
                        **candidate})
                save_json(output/'status.json',status)
                break
            save_json(output/'status.json',status)
        if status['status']=='running': raise ValueError('Continuation ended without committed outcome')
        status['elapsed_seconds']=clock()-started
        save_json(output/'status.json',status)
        return status
    except BaseException as error:
        if active:
            try: reconcile(active)
            except Exception as secondary: status['checkpoint_reconciliation_error']=str(secondary)
        status.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=clock()-started)
        save_json(output/'status.json',status)
        raise

def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='execute':
        parser=argparse.ArgumentParser(description=continue_plan.__doc__)
        for name in ('plan-dir','output','workspace','inputs'):parser.add_argument('--'+name,required=True)
        parser.add_argument('--deadline-seconds',type=int,default=2700)
        parser.add_argument('--checkpoint-reserve-seconds',type=int,default=900)
        parser.add_argument('--write-prefix')
        parser.add_argument('--owned-container-ledger')
        parser.add_argument('--owner-run-id')
        parser.add_argument('--checkpoint-mode',choices=['gcs','local-spool'],default='gcs')
        parser.add_argument('--publisher-script');parser.add_argument('--publisher-sha256')
        parser.add_argument('--spool-min-free-bytes',type=int)
        parser.add_argument('--stop-after-stage',choices=['gsq'],help='Stop after committed GSQ completion for a separately sized head/RCO continuation')
        args=parser.parse_args(sys.argv[2:])
        print(json.dumps(continue_plan(args.plan_dir,args.output,args.workspace,args.inputs,deadline_seconds=args.deadline_seconds,
                                     checkpoint_reserve_seconds=args.checkpoint_reserve_seconds,write_prefix=args.write_prefix,
                                     owned_container_ledger=args.owned_container_ledger,owner_run_id=args.owner_run_id,
                                     checkpoint_mode=args.checkpoint_mode,publisher_script=args.publisher_script,
                                     publisher_sha256=args.publisher_sha256,spool_min_free_bytes=args.spool_min_free_bytes,
                                     stop_after_stage=args.stop_after_stage),indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job-dir', 'output', 'runtime-image', 'inputs', 'recovery-output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--deadline-seconds', type=int, default=2700)
    args = parser.parse_args()
    print(json.dumps(prepare(args.job_dir, args.output, args.runtime_image, args.inputs,
                             args.recovery_output, args.deadline_seconds), indent=2))


if __name__ == '__main__': main()
