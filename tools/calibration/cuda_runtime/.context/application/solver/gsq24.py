"""Actual 24 GB GSQ validation, with isolated reference/streamed/cold workers.

This is a diagnostic run, not a production continuation or a full-job ETA.
Original f13 checkpoints remain immutable. Both paths execute the unchanged
BlockTrainer math; only residency differs. A fresh worker tests next-update
recovery. Full-attention inputs are a diagnostic propagation of committed block2.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import torch
from checkpoints import capture_rng_state, restore_rng_state, LocalCheckpointStore
from .gsq import BlockTrainer
from .gsq_residency import GSQResidency
from .qwen import load_original, run_block
from .run import (atomic_json, corpus_inputs, digest, project_scale_format,
                  validate_restored_inputs, restore_state, bind_identity, gsq_run, Checkpointer)
from .smoke import assert_state_equal, cpu_clone, cuda_report
from .reproducibility import configure

# Filled from the immutable prior image, not a mutable Git working tree.
PRIOR_RUNTIME = 'f13a4e99a85f4f650a8e3e130757fb4ddee891eeb8e34e6cea973e99dc56c3cd'
PRIOR_GSQ_SHA256 = 'e13654823c070e5e71984b1c0cd80a6060855166bc8aff2a999c549afb990e30'
PRIOR_QWEN_SHA256 = 'aad58276128fcdd0a66d782c97dcae1748b57a9bbc0caf9738f01187988493b5'


def verify_reference_runtime():
    import accelerate
    import transformers
    if (digest(Path(__file__).with_name('gsq.py')) != PRIOR_GSQ_SHA256
            or digest(Path(__file__).with_name('qwen.py')) != PRIOR_QWEN_SHA256
            or torch.__version__ != '2.11.0+cu130' or transformers.__version__ != '5.7.0'
            or accelerate.__version__ != '1.13.0'):
        raise ValueError('Reference trainer, model layout or dependency versions changed')


def checked_hardware():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Actual CUDA BF16 hardware required')
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one GPU is required for this 24 GB validation')
    p = torch.cuda.get_device_properties(0)
    if not 20 * 1024**3 <= p.total_memory <= 25 * 1024**3:
        raise ValueError('This validation must run on an actual 24 GB class GPU')
    return torch.device('cuda:0')


def names_for(model, block):
    return [name for name, m in model.model.layers[block].named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.shape[1] % 128 == 0]


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def candidate_fingerprints(exported):
    from safetensors import safe_open
    result = {}
    for name, path in exported.items():
        with safe_open(path, framework='pt', device='cpu') as source:
            result[name] = {'metadata': source.metadata(), 'tensors': {}}
            for key in sorted(source.keys()):
                tensor = source.get_tensor(key).contiguous()
                result[name]['tensors'][key] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                    'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def validate_worker_report(report, config, block, phase):
    kind = {2: 'linear_attention', 3: 'full_attention'}[block]
    if any(report.get(key) != value for key, value in {
        'status': 'passed', 'block': block, 'phase': phase, 'kind': kind,
        'runtime_sha256': config['runtime_sha256'],
        'source_receipt_sha256': config['source_receipt_sha256'],
        'sequence_tokens': config['expected_longest_tokens'],
        'diagnostic_prefix': block == 3, 'production_progress_advanced': False}.items()):
        raise ValueError('Worker result identity/scope mismatch')
    if type(report.get('source_sequence_index')) is not int or report['source_sequence_index'] < 0:
        raise ValueError('Missing source sequence index')
    cuda = report.get('cuda', {})
    total = cuda.get('total_memory_bytes')
    if (cuda.get('available') is not True or not cuda.get('device') or not cuda.get('capability')
            or type(total) is not int or not 20*1024**3 <= total <= 25*1024**3
            or any(type(cuda.get(k)) is not int or not 0 < cuda[k] <= total
                   for k in ('peak_allocated_bytes', 'peak_reserved_bytes'))):
        raise ValueError('Missing actual 24 GB CUDA memory evidence')
    checks = report.get('checks', [])
    if [c.get('step') for c in checks] != ([2] if phase == 'cold' else [1, 2]):
        raise ValueError('Incomplete worker updates/replay')
    for check in checks:
        if check.get('all_gradients_and_states_compared') is not (phase != 'reference'):
            raise ValueError('Missing numerical comparison')
        for key in ('loss', 'update_seconds', 'propagation_seconds'):
            value = check.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid numerical/timing measurement')
    projections = report.get('projections', [])
    names = [p.get('name') for p in projections]
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Missing unique projection inventory')
    candidates = report.get('candidate_sha256', {})
    if set(candidates) != {f'model.layers.{block}.{n}' for n in names}:
        raise ValueError('Missing hard candidate export comparison')
    for item in candidates.values():
        if set(item.get('tensors', {})) != {'codes', 'scales'} or not item.get('metadata'):
            raise ValueError('Missing candidate tensor fingerprints')
        for tensor in item['tensors'].values():
            if len(tensor.get('sha256', '')) != 64 or not tensor.get('shape') or not tensor.get('dtype'):
                raise ValueError('Invalid candidate tensor fingerprint')
    return report


@contextlib.contextmanager
def reference_block(model, block, device):
    """Explicit old BlockTrainer placement, independent of GSQResidency."""
    model.model.layers[block].to(device)
    model.model.rotary_emb.to(device)
    try:
        yield
    finally:
        model.model.layers[block].to('cpu')
        model.model.rotary_emb.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()


def pair_from_source(payloads, source_config, records):
    progress = load_state(payloads / 'progress')
    if progress != {**progress, 'stage': 'gsq', 'block': 2, 'global_step': 174,
                     'epoch': 0, 'sequence': 0}:
        raise ValueError('Expected the reviewed block2-start source checkpoint')
    if source_config.get('largest_first'):
        records = sorted(records, key=len, reverse=True)
    index = max(range(len(records)), key=lambda n: len(records[n]))
    with tarfile.open(payloads / 'cache_block_2') as archive:
        member = archive.getmember(f'{index:06d}.pt')
        if not member.isfile() or member.size > 512 * 1024**2:
            raise ValueError('Unexpected longest-input cache member')
        with archive.extractfile(member) as stream:
            pair = torch.load(stream, map_location='cpu', weights_only=True)
    if set(pair) != {'teacher', 'student'}:
        raise ValueError('Invalid paired GSQ cache')
    for value in pair.values():
        if value.shape != (1, len(records[index]), 5120) or value.dtype != torch.bfloat16:
            raise ValueError('Source cache is not the full actual longest sequence')
    return pair, index, len(records[index])


def import_trainer(model, block, payloads, source_config, upstream):
    trainer = BlockTrainer(model, block, upstream)
    if trainer.names != names_for(model, block):
        raise ValueError('Quantizer projection ordering changed')
    optimizer = torch.optim.Adam(trainer.parameters(), lr=source_config.get('gsq_lr', .001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        max(1, source_config.get('gsq_epochs', 1) * source_config['_records']))
    if block == 2:
        state = load_state(payloads / 'solver')
        optimizer_state = load_state(payloads / 'optimizer')
        scheduler_state = load_state(payloads / 'scheduler')
    elif block == 3:
        warm = load_state(payloads / 'warmstart_block_3')
        if warm['block'] != 3 or warm['updates'] != 1:
            raise ValueError('Full-attention warmstart metadata changed')
        state, optimizer_state = warm['solver'], warm['optimizer']
        scheduler_state = None
    else:
        raise ValueError('Only reviewed source blocks2/3 supported')
    expected = trainer.state_dict()
    if expected.keys() != state.keys():
        raise ValueError('Source quantizer keys changed')
    for key in expected:
        if expected[key].shape != state[key].shape or expected[key].dtype != state[key].dtype:
            raise ValueError('Source quantizer shape/dtype changed')
    trainer.load_state_dict(state)
    optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    assert_state_equal(trainer.state_dict(), state, exact=True)
    assert_state_equal(optimizer.state_dict(), optimizer_state, exact=True)
    return trainer, optimizer, scheduler


def propagate_pair(model, trainer, pair, device):
    with torch.no_grad():
        teacher = run_block(model, trainer.block_index, pair['teacher'].to(device))
        student = trainer.hard_forward(pair['student'].to(device))
        return {'teacher': teacher.cpu(), 'student': student.cpu()}


def drop_training(trainer, optimizer):
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    trainer.to('cpu')


def update(model, trainer, optimizer, scheduler, pair, device, step, config, expected=None, save=None):
    torch.cuda.synchronize(); start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    # Same schedule as first two corpus updates; repeated longest input is a
    # deliberate stress/replay test, not an extra production training epoch.
    temperature = 1. - .9 * (step - 1) / max(1, config['_records'] * config.get('gsq_epochs', 1) - 1)
    loss = trainer(pair['student'].to(device), pair['teacher'].to(device), temperature)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainer.parameters()):
        raise FloatingPointError('Missing or nonfinite GSQ gradient')
    gradients = {n: p.grad.detach().cpu() for n, p in trainer.named_parameters()}
    if expected is not None:
        assert_state_equal(loss, expected['loss'], exact=False)
        assert_state_equal(gradients, expected['gradients'], exact=False)
    optimizer.step(); project_scale_format(trainer); scheduler.step()
    torch.cuda.synchronize(); update_seconds = time.monotonic() - start
    result = {'loss': loss.detach().cpu(), 'gradients': gradients,
              'solver': cpu_clone(trainer.state_dict()), 'optimizer': cpu_clone(optimizer.state_dict()),
              'scheduler': scheduler.state_dict(), 'rng': cpu_clone(capture_rng_state())}
    start = time.monotonic()
    result['propagated'] = propagate_pair(model, trainer, pair, device)
    torch.cuda.synchronize(); propagation_seconds = time.monotonic() - start
    if expected is not None:
        for role in ('solver', 'optimizer', 'propagated'):
            assert_state_equal(result[role], expected[role], exact=False)
        for role in ('scheduler', 'rng'):
            assert_state_equal(result[role], expected[role], exact=True)
    if save is not None:
        torch.save(result, save)
    return {'step': step, 'loss': loss.item(), 'update_seconds': update_seconds,
            'propagation_seconds': propagation_seconds,
            'all_gradients_and_states_compared': expected is not None}


def write_checkpoint(directory, trainer, optimizer, scheduler, source, identity):
    directory.mkdir()
    payloads = {}
    values = {'solver': trainer.state_dict(), 'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(), 'rng': capture_rng_state(),
              'progress': {'stage': 'gsq24_validation', 'step': 1, 'production_update': False}}
    for role, value in values.items():
        path = directory / role
        torch.save(value, path); payloads[role] = path
    store = LocalCheckpointStore(directory / 'store')
    before = cpu_clone(capture_rng_state())
    receipt = store.publish('after-first-update', identity, payloads)
    assert_state_equal(capture_rng_state(), before, exact=True)
    atomic_json(directory / 'receipt.json', receipt)
    return receipt


def production_canary(model, records, source_config, payloads, source_receipt, config, output, device):
    """Two real gsq_run updates; distinct diagnostic identity, never production progress."""
    work = output / 'production-canary'
    work.mkdir()
    canary = {k: v for k, v in source_config.items() if k not in ('identity', '_records', 'resume_checkpoint')}
    canary.update(stage='gsq', output=str(work), inputs=config['inputs'],
        model_dir=config['model_dir'], corpus=config['corpus'], runtime_sha256=config['runtime_sha256'],
        deadline_unix=config['deadline_unix'], gsq_memory_mode='block_cpu_offload', gsq_execution_device=str(device),
        diagnostic_only=True, diagnostic_source_commit=source_receipt['receipt']['commit'],
        checkpoint={'backend': 'local', 'path': str(work / 'store')})
    canary['warmstart_states'] = {str(b): str(payloads / f'warmstart_block_{b}') for b in (0, 3)}
    canary['cost_manifest'] = {**canary['cost_manifest'], 'path': str(payloads / 'packing_cost_manifest')}
    manifest = json.loads((Path(config['corpus']) / 'manifest.json').read_text())
    bind_identity(canary, manifest)
    atomic_json(work / 'frozen-config.json', canary)
    records = sorted(records, key=len, reverse=True)
    started = time.monotonic()
    result = gsq_run(model, records, canary, work, Checkpointer(canary, work), resume=payloads, max_steps=2)
    torch.cuda.synchronize()
    if result['status'] != 'checkpointed_stop' or result['progress']['global_step'] != 176 or result['progress']['sequence'] != 2:
        raise ValueError('Actual GSQ canary did not perform exactly two resumed updates')
    receipt = json.loads((work / 'latest-checkpoint.json').read_text())
    restored = work / 'verified-final-checkpoint'
    LocalCheckpointStore(work / 'store').restore(receipt['snapshot'], canary['identity'], restored)
    progress = load_state(restored / 'progress')
    if progress != result['progress']:
        raise ValueError('Actual GSQ canary checkpoint cursor differs')
    result = {'status': 'passed', 'scope': 'actual_gsq_run_two_updates_from_source_block2',
        'diagnostic_only': True, 'production_progress_advanced': False, 'new_diagnostic_updates': 2,
        'runtime_sha256': config['runtime_sha256'], 'source_receipt_sha256': config['source_receipt_sha256'],
        'input_tokens': [len(r) for r in records[:2]], 'source_commit': source_receipt['receipt']['commit'],
        'checkpoint_verified': True, 'identity': canary['identity'], 'progress': progress,
        'elapsed_seconds': time.monotonic() - started,
        'performance': json.loads((work / 'performance-report.json').read_text()), 'cuda': cuda_report(device)}
    atomic_json(output / 'production-canary-report.json', result)
    print(json.dumps(result), flush=True)
    return result


def prune_validation_scratch(output):
    """After all replay proofs pass, discard only reproducible diagnostic copies."""
    output = Path(output).resolve()
    removed = []; paths = []
    for block in (2, 3):
        work = output / f'block-{block}'
        if work.is_symlink() or not work.is_dir():
            raise ValueError('Unsafe diagnostic scratch directory')
        for phase in ('reference', 'streamed', 'cold'):
            if json.loads((work / f'{phase}-report.json').read_text()).get('status') != 'passed':
                raise ValueError('Cannot prune unverified diagnostic state')
        for name in ('reference-1.pt', 'reference-2.pt', 'checkpoint', 'cold-restored'):
            path = work / name
            if path.is_symlink() or not path.exists():
                raise ValueError('Unsafe diagnostic scratch artifact')
            paths.append(path)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path.relative_to(output)))
    if shutil.disk_usage(output).free < 100 * 1024**3:
        raise ValueError('Actual GSQ run-loop canary needs 100 GiB free scratch')
    return removed


def validate_canary_report(report, config, source_validation, expected_identity, input_tokens):
    expected = {'status': 'passed', 'scope': 'actual_gsq_run_two_updates_from_source_block2',
        'new_diagnostic_updates': 2, 'checkpoint_verified': True, 'diagnostic_only': True,
        'production_progress_advanced': False, 'runtime_sha256': config['runtime_sha256'],
        'source_receipt_sha256': config['source_receipt_sha256'],
        'source_commit': source_validation['source_commit'], 'identity': expected_identity,
        'input_tokens': input_tokens}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual GSQ canary provenance/scope differs')
    identity = report['identity']; old = source_validation['source_identity']
    if (identity == old or identity.get('runtime_sha256') != config['runtime_sha256']
            or any(identity.get(k) != old.get(k) for k in ('baseline_repo', 'baseline_revision', 'calibration_sha256'))):
        raise ValueError('Canary needs a new identity with unchanged source/data')
    progress = report.get('progress', {})
    if any(progress.get(k) != v for k, v in {'stage': 'gsq', 'block': 2, 'epoch': 0, 'sequence': 2, 'global_step': 176}.items()):
        raise ValueError('Actual GSQ canary cursor differs')
    cuda = report.get('cuda', {}); total = cuda.get('total_memory_bytes')
    if (cuda.get('available') is not True or not cuda.get('device') or not cuda.get('capability')
            or type(total) is not int or not 20*1024**3 <= total <= 25*1024**3
            or any(type(cuda.get(k)) is not int or not 0 < cuda[k] <= total
                   for k in ('peak_allocated_bytes', 'peak_reserved_bytes'))):
        raise ValueError('Missing actual canary CUDA memory evidence')
    groups = report.get('performance', {}).get('groups', {})
    updates = [g for g in groups.values() if g.get('stage') == 'gsq' and g.get('operation') == 'update_with_input_load']
    if (not updates or any(g.get('status') != 'completed' or type(g.get('count')) is not int
            or g['count'] <= 0 or type(g.get('seconds')) not in (int, float)
            or not math.isfinite(g['seconds']) or g['seconds'] <= 0 for g in updates)
            or sum(g['count'] for g in updates) != 2):
        raise ValueError('Missing two measured actual GSQ updates')
    return report


def worker(config, output, block, phase):
    configure(42)
    device = checked_hardware()
    torch.cuda.reset_peak_memory_stats(device)
    source = Path(config['source_checkpoint'])
    payloads = source / 'payloads'
    source_receipt = json.loads((source / 'source-receipt.json').read_text())
    source_config = json.loads((payloads / 'configuration').read_text())
    if source_config['runtime_sha256'] != PRIOR_RUNTIME:
        raise ValueError('Unexpected prior runtime')
    # Parent validated all source hashes; mounts must remain read-only throughout.
    if digest(source / 'source-receipt.json') != config['source_receipt_sha256']:
        raise ValueError('Source receipt changed')
    validate_restored_inputs(config)
    if (config['input_commit_sha256'] != source_config['input_commit_sha256']
            or digest(Path(config['corpus']) / 'manifest.json') != source_config['identity']['calibration_sha256']):
        raise ValueError('Source checkpoint and current calibration inputs differ')
    records = corpus_inputs(config['corpus'])
    source_config['_records'] = len(records)
    pair, index, tokens = pair_from_source(payloads, source_config, records)
    if tokens != config['expected_longest_tokens']:
        raise ValueError('Longest calibration prompt changed')
    started = time.monotonic()
    model = load_original(config['model_dir'], torch.device('cpu'))
    load_seconds = time.monotonic() - started
    if model.config.layer_types[2:4] != ['linear_attention', 'full_attention']:
        raise ValueError('Unexpected hybrid block layout')
    if phase == 'production':
        del pair
        return production_canary(model, records, source_config, payloads, source_receipt, config, output, device)
    residency = GSQResidency(model, {'stage': 'gsq', 'gsq_memory_mode': 'block_cpu_offload',
                                    'gsq_execution_device': str(device)})
    scope = (lambda n: reference_block(model, n, device)) if phase == 'reference' else residency.block
    if block == 3:
        with scope(2):
            prefix, opt, sch = import_trainer(model, 2, payloads, source_config, config.get('upstream', '/opt/upstream'))
            pair = propagate_pair(model, prefix, pair, device)
            drop_training(prefix, opt); del prefix, opt, sch
    work = output / f'block-{block}'
    work.mkdir(exist_ok=True)
    checks = []; checkpoint = work / 'checkpoint'
    with scope(block):
        trainer, optimizer, scheduler = import_trainer(model, block, payloads, source_config, config.get('upstream', '/opt/upstream'))
        projection_map = [{'name': n, 'shape': list(q.sign_logits.shape), 'dtype': str(q.sign_logits.dtype)}
                          for n, q in zip(trainer.names, trainer.quantizers)]
        identity = dict(source_config['identity'])
        identity.update(runtime_sha256=config['runtime_sha256'],
            solver_config_sha256=hashlib.sha256(json.dumps({'test': 'gsq24', 'block': block,
                'source': config['source_receipt_sha256'], 'runtime': config['runtime_sha256']}, sort_keys=True).encode()).hexdigest())
        if phase == 'cold':
            restored = work / 'cold-restored'
            LocalCheckpointStore(checkpoint / 'store').restore('after-first-update', identity, restored)
            restore_state(restored, trainer, optimizer, scheduler)
            first = load_state(work / 'reference-1.pt')
            for role, actual in [('solver', trainer.state_dict()), ('optimizer', optimizer.state_dict()),
                                 ('scheduler', scheduler.state_dict()), ('rng', capture_rng_state())]:
                assert_state_equal(actual, first[role], exact=role in ('scheduler', 'rng'))
            del first
            steps = (2,)
        else:
            restore_rng_state(load_state(payloads / 'rng'))
            steps = (1, 2)
        for step in steps:
            expected = None if phase == 'reference' else load_state(work / f'reference-{step}.pt')
            checks.append(update(model, trainer, optimizer, scheduler, pair, device, step, source_config,
                                 expected=expected, save=work / f'reference-{step}.pt' if phase == 'reference' else None))
            del expected
            if phase == 'streamed' and step == 1:
                write_checkpoint(checkpoint, trainer, optimizer, scheduler, source, identity)
        # Hard candidate serialization agrees across separately started workers.
        exported = trainer.export(work / f'{phase}-candidates')
        exported_hashes = candidate_fingerprints(exported)
        reference_path = work / 'reference-report.json'
        if phase != 'reference':
            reference = json.loads(reference_path.read_text())
            if exported_hashes != reference['candidate_sha256'] or projection_map != reference['projections']:
                raise ValueError('Hard candidate export or projection ordering differs')
        drop_training(trainer, optimizer); del trainer, optimizer, scheduler
    residency.assert_layout()
    result = {'status': 'passed', 'block': block, 'kind': model.config.layer_types[block], 'phase': phase,
              'sequence_tokens': tokens, 'source_sequence_index': index,
              'diagnostic_prefix': block == 3, 'production_progress_advanced': False,
              'load_seconds': load_seconds, 'checks': checks, 'projections': projection_map,
              'candidate_sha256': exported_hashes, 'cuda': cuda_report(device),
              'runtime_sha256': config['runtime_sha256'], 'source_receipt_sha256': config['source_receipt_sha256']}
    atomic_json(work / f'{phase}-report.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block', type=int, choices=(2, 3))
    parser.add_argument('--phase', choices=('reference', 'streamed', 'cold', 'production'))
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if (args.block is None) != (args.phase is None):
        parser.error('Worker block and phase must be specified together')
    if args.phase:
        worker(config, args.output, args.block, args.phase); return
    from .gsq_migration import validate_source_directory
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'production_progress_advanced': False,
              'complete_job_eta_seconds': None, 'runtime_sha256': config['runtime_sha256'], 'workers': []}
    try:
        source = Path(config['source_checkpoint'])
        receipt = json.loads((source / 'source-receipt.json').read_text())
        if digest(source / 'source-receipt.json') != config['source_receipt_sha256']:
            raise ValueError('Source receipt differs from prepared hash')
        validation = validate_source_directory(source / 'payloads', receipt)
        atomic_json(args.output / 'source-validation.json', validation)
        verify_reference_runtime()
        # Workers must not consume the original source or advance its cursor.
        for block in (2, 3):
            for phase in ('reference', 'streamed', 'cold'):
                print(json.dumps({'event': 'starting_worker', 'block': block, 'phase': phase}), flush=True)
                remaining = int(config['deadline_unix'] - time.time() - 90)
                if remaining <= 0:
                    raise TimeoutError('No time remains within approved validation cap')
                command = [sys.executable, '-m', 'solver.gsq24', '--config', str(args.config),
                           '--output', str(args.output), '--block', str(block), '--phase', phase]
                subprocess.run(command, check=True, timeout=remaining)
                report = json.loads((args.output / f'block-{block}' / f'{phase}-report.json').read_text())
                validate_worker_report(report, config, block, phase)
                result['workers'].append(report)
                atomic_json(args.output / 'gsq24-report.json', result)
        result['discarded_reproducible_diagnostic_scratch'] = prune_validation_scratch(args.output)
        remaining = int(config['deadline_unix'] - time.time() - 90)
        if remaining <= 0:
            raise TimeoutError('No time remains for actual GSQ run-loop canary')
        subprocess.run([sys.executable, '-m', 'solver.gsq24', '--config', str(args.config),
                        '--output', str(args.output), '--block', '2', '--phase', 'production'],
                       check=True, timeout=remaining)
        canary = json.loads((args.output / 'production-canary-report.json').read_text())
        frozen = json.loads((args.output / 'production-canary/frozen-config.json').read_text())
        manifest = json.loads((Path(frozen['corpus']) / 'manifest.json').read_text())
        if frozen.get('diagnostic_only') is not True or frozen.get('diagnostic_source_commit') != validation['source_commit']:
            raise ValueError('Actual GSQ canary configuration is not a bound diagnostic')
        identity = bind_identity(frozen, manifest)
        lengths = sorted([len(r) for r in corpus_inputs(frozen['corpus'])], reverse=True)[:2]
        validate_canary_report(canary, config, validation, identity, lengths)
        result['production_canary'] = canary
        result.update(status='passed', actual_24gb_validated=True, numerical_parity_passed=True,
                      cold_checkpoint_replay_passed=True, migration_applied=False,
                      production_gsq_loop_cuda_validated=True, full_corpus_gsq_completed=False,
                      inherited_gsq_updates=174, inherited_embedding_updates=87,
                      peak_allocated_bytes=max(r['cuda']['peak_allocated_bytes'] for r in [*result['workers'], canary]),
                      peak_reserved_bytes=max(r['cuda']['peak_reserved_bytes'] for r in [*result['workers'], canary]))
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(args.output / 'gsq24-report.json', result)


if __name__ == '__main__':
    main()
