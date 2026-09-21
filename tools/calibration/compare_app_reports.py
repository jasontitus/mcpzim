"""Strict paired accounting for ProbeDiscussCLI reports and exact input captures.

Binary app rubric scores are not a general model-quality judgment. The model-
involved subset is the UNION of cases that invoked either model, preserving
paired cases even when a model changes app routing. An optional third arm scores
a quantized candidate against BOTH reference and bonsai in bonsai's runtime. No
candidate is promoted.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile


# Roles share one runtime with their comparison partner; the quantized candidate
# must run where Bonsai runs so rubric rows compare the weights, not the backend.
ROLE_RUNTIMES = {'reference': 'mlx', 'bonsai': 'llamacpp', 'candidate': 'llamacpp'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(item, label):
    require(isinstance(item, dict), f'{label}: missing artifact')
    path = Path(item.get('path', ''))
    require(path.is_absolute() and path.is_file(), f'{label}: missing absolute file path')
    require(type(item.get('bytes')) is int and item['bytes'] >= 0, f'{label}: invalid byte count')
    require(path.stat().st_size == item['bytes'], f'{label}: size mismatch')
    require(isinstance(item.get('sha256'), str) and
            re.fullmatch('[0-9a-f]{64}', item['sha256']), f'{label}: invalid hash')
    require(sha256(path) == item['sha256'], f'{label}: checksum mismatch')


def validate_protocol(protocol):
    require(protocol.get('schema_version') == 1 and protocol.get('mode') == 'app-conversation',
            'Expected schema 1 app-conversation protocol')
    require(protocol.get('sampling') == {'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 42},
            'Expected frozen greedy comparison sampler')
    require(all(type(value) in (int, float) for value in protocol['sampling'].values()),
            'Invalid sampling scalar type')
    verify_artifact(protocol['merged_suite'], 'merged suite')
    require(load(protocol['merged_suite']['path']) == protocol['suite'], 'Embedded/frozen merged suite mismatch')
    for key in ('zim', 'streetzim'):
        verify_artifact(protocol['sources'][key], f'source {key}')
    require(bool(protocol.get('source_suites')), 'Missing frozen source suites')
    for item in protocol['source_suites']:
        verify_artifact(item, 'source suite')
    verify_artifact(protocol['evaluator'], 'evaluator')
    models = protocol['models']
    unknown = set(models) - set(ROLE_RUNTIMES)
    require(not unknown, f'Unknown comparison roles: {sorted(unknown)}')
    require('reference' in models and 'bonsai' in models, 'Missing reference/bonsai model')
    for role, runtime in ROLE_RUNTIMES.items():
        if role not in models:
            continue
        model = models[role]
        require(model.get('runtime') == runtime, f'{role}: wrong runtime')
        require(Path(model.get('artifact', '')).is_absolute(), f'{role}: invalid model path')
        require(bool(model.get('capture_model_id')), f'{role}: missing capture model identity')
        verify_artifact(model['manifest'], f'{role} model manifest')
        manifest = load(model['manifest']['path'])
        if role == 'reference':
            require(manifest.get('status') == 'validated' and manifest.get('precision') == 'BF16'
                    and manifest.get('model') == 'Qwen/Qwen3.8-27B',
                    'Reference manifest is not validated original BF16 Qwen3.8-27B')
        elif role == 'bonsai':
            require(manifest.get('matches_published_sha256') is True,
                    'Bonsai manifest is not hash-validated')
        else:
            # A locally quantized candidate has no published digest to match, so its
            # manifest must bind the exact GGUF bytes this protocol froze. Runtime
            # parity is declared here and proven per capture by verify_like_for_like().
            require(model['runtime'] == models['bonsai']['runtime']
                    and model.get('runtime_parity') == 'bonsai',
                    'Candidate does not declare bonsai runtime parity')
            require(manifest.get('status') == 'runtime-verified',
                    'Candidate manifest is not runtime-verified')
            require(manifest.get('artifact_sha256') == sha256(model['artifact']),
                    'Candidate manifest does not bind the frozen GGUF hash')
            require(manifest.get('artifact_bytes') == Path(model['artifact']).stat().st_size,
                    'Candidate manifest byte count mismatch')
    expected = {}
    conversations = protocol['suite']['conversations']
    require(isinstance(conversations, list) and conversations, 'Empty frozen suite')
    seen = set()
    for conversation in conversations:
        cid = conversation['id']
        require(isinstance(cid, str) and cid and cid not in seen, 'Duplicate/invalid conversation')
        seen.add(cid)
        require(bool(conversation['turns']), f'{cid}: empty conversation')
        for index, turn in enumerate(conversation['turns'], 1):
            require(isinstance(turn.get('user'), str) and turn['user'], f'{cid}: missing user prompt')
            expected[(cid, index)] = turn
    return expected


def validate_report(report, role, protocol, expected):
    model = protocol['models'][role]
    require(report.get('schemaVersion') == 2, f'{role}: unsupported report schema')
    for field, value in [('runtime', model['runtime']), ('modelArtifact', model['artifact']),
                         ('zim', protocol['sources']['zim']['path']),
                         ('streetzim', protocol['sources']['streetzim']['path']),
                         ('samplingTemperature', 0), ('samplingTopP', 1),
                         ('samplingTopK', 0), ('samplingSeed', 42)]:
        require(field in report and report[field] == value, f'{role}: mismatched {field}')
    for field in ('samplingTemperature', 'samplingTopP', 'samplingTopK', 'samplingSeed'):
        require(type(report[field]) in (int, float), f'{role}: invalid {field} type')
    cases = {}
    require(isinstance(report.get('turns'), list), f'{role}: missing turns')
    for turn in report['turns']:
        require(type(turn.get('turn')) is int and turn['turn'] >= 1, f'{role}: invalid report turn')
        key = (turn.get('conversationID'), turn['turn'])
        require(key in expected, f'{role}: unexpected turn {key}')
        require(key not in cases, f'{role}: duplicate turn {key}')
        require(turn.get('user') == expected[key]['user'], f'{role}: mismatched user prompt {key}')
        require(type(turn.get('passed')) is bool, f'{role}: unscored turn {key}')
        failures = turn.get('failures')
        require(isinstance(failures, list) and all(isinstance(v, str) and v for v in failures),
                f'{role}: invalid failures {key}')
        require(turn['passed'] == (len(failures) == 0), f'{role}: inconsistent rubric result {key}')
        require(isinstance(turn.get('answer'), str), f'{role}: missing answer {key}')
        elapsed = turn.get('elapsedSeconds')
        require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed >= 0,
                f'{role}: invalid latency {key}')
        cases[key] = turn
    require(set(cases) == set(expected), f'{role}: missing paired turns')
    passed = sum(turn['passed'] for turn in cases.values())
    require(type(report.get('passedTurns')) is int and type(report.get('failedTurns')) is int
            and report['passedTurns'] == passed and report['failedTurns'] == len(cases) - passed,
            f'{role}: report totals mismatch')
    return cases


def validate_capture(directory, role, protocol, expected):
    directory = Path(directory)
    require(directory.is_dir(), f'{role}: missing capture directory')
    run = load(directory / 'capture-run.json')
    require(run.get('status') == 'completed', f'{role}: capture is incomplete')
    count = run.get('invocationCount')
    require(isinstance(count, str) and re.fullmatch(r'0|[1-9][0-9]*', count),
            f'{role}: invalid invocation count')
    count = int(count)
    model = protocol['models'][role]
    binding = {'modelArtifact': model['artifact'], 'runtime': model['runtime'],
               'zimPath': protocol['sources']['zim']['path'],
               'streetzimPath': protocol['sources']['streetzim']['path'],
               'suitePath': protocol['merged_suite']['path']}
    for field, value in binding.items():
        require(run.get(field) == value, f'{role}: mismatched capture {field}')
    filenames = {f'invocation-{i:06d}.json' for i in range(count)} | {'capture-run.json'}
    require({p.name for p in directory.iterdir()} == filenames, f'{role}: missing/extra capture files')
    invocations = {}
    warnings = []
    policies = {}
    for ordinal in range(count):
        record = load(directory / f'invocation-{ordinal:06d}.json')
        require(type(record.get('schemaVersion')) is int and record['schemaVersion'] == 1
                and type(record.get('ordinal')) is int and record['ordinal'] == ordinal,
                f'{role}: duplicate/mismatched invocation ordinal')
        require(type(record.get('turn')) is int and record['turn'] >= 0,
                f'{role}: invalid zero-based capture turn')
        key = (record.get('conversationID'), record['turn'] + 1)
        require(key in expected, f'{role}: unexpected capture case {key}')
        require(record.get('modelID') == model['capture_model_id'] and
                record.get('runtime') == model['runtime'], f'{role}: invocation model/runtime mismatch')
        for field, value in binding.items():
            require(record.get('metadata', {}).get(field) == value,
                    f'{role}: invocation metadata mismatch {field}')
        prompt = record.get('prompt')
        require(isinstance(prompt, str) and prompt and
                hashlib.sha256(prompt.encode()).hexdigest() == record.get('promptSHA256'),
                f'{role}: corrupted captured prompt')
        tokens = record.get('tokenIDs')
        require(isinstance(tokens, list) and tokens and all(type(v) is int and 0 <= v <= 2147483647 for v in tokens),
                f'{role}: invalid token IDs')
        sampler = record.get('sampler', {})
        for field, value in [('temperature', 0), ('topP', 1)]:
            require(type(sampler.get(field)) in (int, float) and sampler[field] == value,
                    f'{role}: invocation sampler mismatch {field}')
        top_k = sampler.get('topK')
        require(type(top_k) in (int, float) and math.isfinite(top_k)
                and top_k >= 0 and int(top_k) == top_k,
                f'{role}: invalid invocation topK')
        if top_k != 0:
            # LlamaCppProvider's temperature <= 0 branch installs ONLY the
            # greedy sampler. Task-default topK=40 is recorded but never used.
            # Keep this backend-specific; do not relax temperature validation.
            require(model['runtime'] == 'llamacpp', f'{role}: invocation sampler mismatch topK')
            warnings.append(f'{ordinal}: recorded topK={top_k} is inactive in llama.cpp temperature-zero greedy branch')
        if 'seed' in sampler:
            require(sampler['seed'] == 42, f'{role}: invocation sampler mismatch seed')
        elif model['runtime'] == 'mlx':
            warnings.append(f'{ordinal}: MLX capture omits seed; report seed=42 and greedy sampler verified')
        else:
            raise ValueError(f'{role}: missing invocation seed')
        invocations.setdefault(key, []).append(ordinal)
        policy = policies.setdefault(key, {'template': set(), 'effective_sampler': set()})
        policy['template'].add(record.get('metadata', {}).get('template'))
        # topK is recorded but inactive in a temperature-zero llama.cpp capture, so it
        # is excluded from the cross-role sampler policy that must agree exactly.
        policy['effective_sampler'].add(tuple(sorted(
            (name, value) for name, value in sampler.items() if name != 'topK')))
    return invocations, warnings, policies


def summarize(cases, keys):
    conversations = {}
    for key in keys:
        conversations.setdefault(key[0], []).append(float(cases[key]['passed']))
    scores = {cid: statistics.mean(values) for cid, values in conversations.items()}
    return {'conversation_count': len(scores), 'turn_count': len(keys),
            'passed_turns': sum(cases[key]['passed'] for key in keys),
            'conversation_weighted_pass_rate': statistics.mean(scores.values()) if scores else None,
            'per_conversation_pass_rate': scores}


def verify_like_for_like(reports, policies, warnings):
    """Prove the candidate arm shares Bonsai's runtime policy, not just its runtime.

    ProbeE2ECLI.swift:737-745 picks the chat template and model identity from the
    lowercased GGUF path, and :761-766 pins context tokens/KV cache type from the
    runtime, so a candidate captured outside those settings is not comparable even
    when both captures say runtime=llamacpp. Returns the paired-invocation count.
    """
    candidate, bonsai = policies['candidate'], policies['bonsai']
    shared = sorted(set(candidate) & set(bonsai))
    for key in shared:
        require(candidate[key]['template'] == bonsai[key]['template'],
                f'candidate: chat template differs from bonsai on {key}')
        require(candidate[key]['effective_sampler'] == bonsai[key]['effective_sampler'],
                f'candidate: effective sampler differs from bonsai on {key}')
    if not shared:
        warnings['candidate'].append('no turn invoked by both candidate and bonsai; '
                                     'template and sampler parity unverified')
    for field in ('contextTokens', 'kvCacheType'):
        require(field in reports['candidate'] and reports['candidate'][field] == reports['bonsai'].get(field),
                f'candidate: report {field} differs from bonsai; context/KV policy is not like-for-like')
    return len(shared)


def compare(reference, bonsai, reference_capture, bonsai_capture, protocol,
            candidate=None, candidate_capture=None):
    require((candidate is None) == (candidate_capture is None),
            'Candidate report and capture must be supplied together')
    require(('candidate' in protocol['models']) == (candidate is not None),
            'Candidate must be declared in the protocol and supplied together')
    expected = validate_protocol(protocol)
    reports = {'reference': reference, 'bonsai': bonsai}
    directories = {'reference': reference_capture, 'bonsai': bonsai_capture}
    if candidate is not None:
        reports['candidate'], directories['candidate'] = candidate, candidate_capture
    indexed, captures, warnings, policies = {}, {}, {}, {}
    for role, directory in directories.items():
        indexed[role] = validate_report(reports[role], role, protocol, expected)
        captures[role], warnings[role], policies[role] = validate_capture(directory, role, protocol, expected)
    all_keys = list(expected)
    # The *pair* row must not move when a third arm routes differently. Unioning over
    # every supplied role lets a candidate-only turn enter the denominator of the
    # documented reference-minus-bonsai gap - measured: three candidate-only invocations
    # moved model_involved_app_turns bonsai 0.6458 -> 0.6667 and reference_minus_bonsai
    # +0.1042 -> +0.0833 while all_app_turns stayed bit-identical. The candidate keeps
    # its own row over the any-role union, which is also the flag per case below.
    pair_union = [key for key in all_keys
                  if any(key in captures[role] for role in ('reference', 'bonsai'))]
    union = [key for key in all_keys if any(key in captures[role] for role in captures)]
    shared = verify_like_for_like(reports, policies, warnings) if candidate is not None else None
    summaries = {}
    rows = [('all_app_turns', all_keys), ('model_involved_app_turns', pair_union)]
    if candidate is not None:
        rows.append(('model_involved_app_turns_with_candidate', union))
    for label, keys in rows:
        row = {role: summarize(indexed[role], keys) for role in indexed}
        deltas = {'reference_minus_bonsai': ('reference', 'bonsai')}
        if candidate is not None:
            deltas['candidate_minus_reference'] = ('candidate', 'reference')
            deltas['candidate_minus_bonsai'] = ('candidate', 'bonsai')
        for name, (left, right) in deltas.items():
            pair = (row[left]['conversation_weighted_pass_rate'], row[right]['conversation_weighted_pass_rate'])
            row[name] = pair[0] - pair[1] if all(value is not None for value in pair) else None
        summaries[label] = row
    cases = []
    for key in all_keys:
        row = {'conversation_id': key[0], 'turn': key[1], 'user': expected[key]['user'],
               'rubric': expected[key], 'model_involved_union': key in union,
               'critical_judgment': 'missing_manual_review_required'}
        for role in indexed:
            turn = indexed[role][key]
            row[role] = {'model_invoked': key in captures[role],
                         'invocation_ordinals': captures[role].get(key, []),
                         'raw_turn_report': turn}
        if candidate is not None:
            for role in ('reference', 'bonsai'):
                row[f'candidate_vs_{role}'] = {
                    'rubric_regression': indexed[role][key]['passed'] and not indexed['candidate'][key]['passed'],
                    'rubric_gain': indexed['candidate'][key]['passed'] and not indexed[role][key]['passed']}
        row['reference_rubric_regression'] = (indexed['bonsai'][key]['passed'] and
                                              not indexed['reference'][key]['passed'])
        cases.append(row)
    result = {'schema_version': 1, 'status': 'compared_rubric_only', 'mode': 'app-conversation',
              'promotion_eligible': False, 'critical_review_status': 'missing',
              'critical_judgments_missing': len(cases) * len(indexed),
              'interpretation': 'Binary app rubric only. Deterministic turns are not model quality. '
                                'Model-involved subset is a paired union, not a model-only benchmark. '
                                'Closed-loop retrieval/context may differ. Manual critical review remains required.',
              'summaries': summaries, 'capture_warnings': warnings, 'cases': cases}
    if candidate is not None:
        result['like_for_like'] = {
            'candidate_shared_runtime_with': protocol['models']['candidate']['runtime_parity'],
            'runtime': protocol['models']['candidate']['runtime'],
            'paired_turns_compared': shared,
            'context_tokens': reports['candidate']['contextTokens'],
            'kv_cache_type': reports['candidate']['kvCacheType'],
            'caveat': 'Same runtime, chat template, effective sampler and context/KV policy as bonsai; '
                      'the GGUF artifact and its weights differ. Binary rubric only, no promotion.'}
    return result



def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'bonsai', 'reference-capture', 'bonsai-capture', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('candidate', 'candidate-capture'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    try:
        result = compare(load(args.reference), load(args.bonsai), args.reference_capture,
                         args.bonsai_capture, load(args.protocol),
                         load(args.candidate) if args.candidate else None, args.candidate_capture)
        result['protocol_sha256'] = sha256(args.protocol)
        result['report_sha256'] = {'reference': sha256(args.reference), 'bonsai': sha256(args.bonsai)}
        if args.candidate:
            result['report_sha256']['candidate'] = sha256(args.candidate)
        write_atomic(args.output, result)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        write_atomic(args.output, {'schema_version': 1, 'status': 'failed',
                                  'promotion_eligible': False, 'error': str(exc)})
        parser.exit(2, f'Comparison rejected: {exc}\n')


if __name__ == '__main__':
    main()
