import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_app_reports import compare, sha256, write_atomic


def save(path, content):
    path.write_text(json.dumps(content))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


@pytest.fixture
def sample(tmp_path):
    suite = {'schema_version': 1, 'name': 'heldout', 'conversations': [
        {'id': 'a', 'turns': [{'user': 'one', 'anchorGroups': [['yes']]}, {'user': 'two'}]},
        {'id': 'b', 'turns': [{'user': 'three'}]}]}
    source_suite = save(tmp_path / 'source-suite.json', suite)
    protocol = {'schema_version': 1, 'mode': 'app-conversation', 'suite': suite,
                'merged_suite': save(tmp_path / 'merged-suite.json', suite),
                'sources': {name: save(tmp_path / name, {'offline': name})
                            for name in ('zim', 'streetzim')},
                'source_suites': [source_suite],
                'evaluator': save(tmp_path / 'evaluator', {'binary': 'frozen'}),
                'sampling': {'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 42},
                'models': {}}
    reports, captures = {}, {}
    for role, runtime in [('reference', 'mlx'), ('bonsai', 'llamacpp')]:
        manifest = ({'status': 'validated', 'precision': 'BF16', 'model': 'Qwen/Qwen3.8-27B'}
                    if role == 'reference' else {'matches_published_sha256': True})
        model = {'artifact': str(tmp_path / ('model-' + role)), 'runtime': runtime,
                 'capture_model_id': 'model-id-' + role,
                 'manifest': save(tmp_path / (role + '-manifest.json'), manifest)}
        protocol['models'][role] = model
        turns = []
        for cid, turn, user, passed in [('a', 1, 'one', role == 'reference'),
                                         ('a', 2, 'two', False), ('b', 1, 'three', True)]:
            turns.append({'conversationID': cid, 'turn': turn, 'user': user,
                          'passed': passed, 'failures': [] if passed else ['missing anchor'],
                          'answer': 'some answer', 'elapsedSeconds': .5})
        reports[role] = {'schemaVersion': 2, 'runtime': runtime, 'modelArtifact': model['artifact'],
                         'zim': protocol['sources']['zim']['path'],
                         'streetzim': protocol['sources']['streetzim']['path'],
                         'contextTokens': 16384 if runtime == 'llamacpp' else 0,
                         'kvCacheType': 'Q4_0' if runtime == 'llamacpp' else 'mlx-unquantized',
                         'samplingTemperature': 0, 'samplingTopP': 1, 'samplingTopK': 0,
                         'samplingSeed': 42, 'passedTurns': sum(t['passed'] for t in turns),
                         'failedTurns': sum(not t['passed'] for t in turns), 'turns': turns}
        directory = tmp_path / (role + '-capture')
        directory.mkdir()
        metadata = {'runtime': runtime, 'modelArtifact': model['artifact'],
                    'template': 'QwenChatMLTemplate',
                    'zimPath': protocol['sources']['zim']['path'],
                    'streetzimPath': protocol['sources']['streetzim']['path'],
                    'suitePath': protocol['merged_suite']['path']}
        save(directory / 'capture-run.json', dict(metadata, status='completed', invocationCount='1'))
        prompt = 'model-specific rendered prompt ' + role
        record = {'schemaVersion': 1, 'ordinal': 0, 'conversationID': 'a',
                  'turn': 0 if role == 'reference' else 1, 'modelID': model['capture_model_id'],
                  'runtime': runtime, 'metadata': metadata, 'prompt': prompt,
                  'promptSHA256': hashlib.sha256(prompt.encode()).hexdigest(), 'tokenIDs': [1, 2, 3],
                  'sampler': {'temperature': 0, 'topP': 1, 'topK': 0, 'seed': 42}}
        save(directory / 'invocation-000000.json', record)
        captures[role] = directory
    return reports, captures, protocol


def run(sample):
    reports, captures, protocol = sample
    return compare(reports['reference'], reports['bonsai'], captures['reference'],
                   captures['bonsai'], protocol)


@pytest.fixture
def candidate_sample(sample, tmp_path):
    """Same two arms plus a third arm for a quantized candidate in bonsai's runtime."""
    reports, captures, protocol = sample
    artifact = tmp_path / 'candidate-qwen-q1.gguf'
    artifact.write_bytes(b'quantized candidate artifact')
    model = {'artifact': str(artifact), 'runtime': 'llamacpp', 'runtime_parity': 'bonsai',
             'capture_model_id': 'local-discuss-model',
             'manifest': save(tmp_path / 'candidate-manifest.json', {
                 'model': 'Qwen/Qwen3.8-27B', 'status': 'runtime-verified',
                 'artifact_bytes': artifact.stat().st_size, 'artifact_sha256': sha256(artifact)})}
    protocol['models']['candidate'] = model
    turns = []
    for cid, turn, user, passed in [('a', 1, 'one', True), ('a', 2, 'two', True), ('b', 1, 'three', False)]:
        turns.append({'conversationID': cid, 'turn': turn, 'user': user, 'passed': passed,
                      'failures': [] if passed else ['missing anchor'],
                      'answer': 'candidate answer', 'elapsedSeconds': .25})
    reports['candidate'] = {'schemaVersion': 2, 'runtime': 'llamacpp', 'modelArtifact': model['artifact'],
                            'zim': protocol['sources']['zim']['path'],
                            'streetzim': protocol['sources']['streetzim']['path'],
                            'contextTokens': 16384, 'kvCacheType': 'Q4_0',
                            'samplingTemperature': 0, 'samplingTopP': 1, 'samplingTopK': 0,
                            'samplingSeed': 42, 'passedTurns': sum(t['passed'] for t in turns),
                            'failedTurns': sum(not t['passed'] for t in turns), 'turns': turns}
    directory = tmp_path / 'candidate-capture'
    directory.mkdir()
    metadata = {'runtime': 'llamacpp', 'modelArtifact': model['artifact'],
                'template': 'QwenChatMLTemplate',
                'zimPath': protocol['sources']['zim']['path'],
                'streetzimPath': protocol['sources']['streetzim']['path'],
                'suitePath': protocol['merged_suite']['path']}
    save(directory / 'capture-run.json', dict(metadata, status='completed', invocationCount='1'))
    prompt = 'candidate rendered prompt'
    save(directory / 'invocation-000000.json',
         {'schemaVersion': 1, 'ordinal': 0, 'conversationID': 'a', 'turn': 1,
          'modelID': model['capture_model_id'], 'runtime': 'llamacpp', 'metadata': metadata,
          'prompt': prompt, 'promptSHA256': hashlib.sha256(prompt.encode()).hexdigest(),
          'tokenIDs': [4, 5, 6],
          # Bonsai's recorded task default: topK is inactive in the greedy branch.
          'sampler': {'temperature': 0, 'topP': 1, 'topK': 40, 'seed': 42}})
    captures['candidate'] = directory
    return reports, captures, protocol


def run_three(sample):
    reports, captures, protocol = sample
    return compare(reports['reference'], reports['bonsai'], captures['reference'],
                   captures['bonsai'], protocol, reports['candidate'], captures['candidate'])


def mutate_capture(sample, role, filename, fn):
    path = sample[1][role] / filename
    value = json.loads(path.read_text())
    fn(value)
    path.write_text(json.dumps(value))


def test_two_role_result_has_no_candidate_artifacts(sample):
    result = run(sample)
    assert list(result['summaries']['all_app_turns']) == ['reference', 'bonsai', 'reference_minus_bonsai']
    assert list(result['capture_warnings']) == ['reference', 'bonsai']
    assert 'like_for_like' not in result
    assert all(not any(name.startswith('candidate') for name in case) for case in result['cases'])


def test_candidate_is_scored_against_both_roles(candidate_sample):
    result = run_three(candidate_sample)
    all_turns = result['summaries']['all_app_turns']
    assert list(all_turns) == ['reference', 'bonsai', 'candidate',
                               'reference_minus_bonsai', 'candidate_minus_reference', 'candidate_minus_bonsai']
    assert all_turns['candidate']['conversation_weighted_pass_rate'] == .5
    assert all_turns['candidate_minus_reference'] == -.25
    assert all_turns['candidate_minus_bonsai'] == 0
    involved = result['summaries']['model_involved_app_turns']
    assert involved['candidate']['turn_count'] == 2
    assert involved['candidate']['conversation_weighted_pass_rate'] == 1
    assert involved['candidate_minus_bonsai'] == 1
    turn_one, turn_two, turn_three = result['cases']
    assert turn_one['candidate_vs_reference'] == {'rubric_regression': False, 'rubric_gain': False}
    assert turn_one['candidate_vs_bonsai'] == {'rubric_regression': False, 'rubric_gain': True}
    assert turn_two['candidate_vs_reference'] == {'rubric_regression': False, 'rubric_gain': True}
    assert turn_three['candidate_vs_bonsai'] == {'rubric_regression': True, 'rubric_gain': False}
    assert turn_two['candidate']['invocation_ordinals'] == [0]
    assert result['critical_judgments_missing'] == 9
    assert result['promotion_eligible'] is False
    assert result['interpretation'].endswith('Manual critical review remains required.')
    assert result['like_for_like'] == {
        'candidate_shared_runtime_with': 'bonsai', 'runtime': 'llamacpp', 'paired_turns_compared': 1,
        'context_tokens': 16384, 'kv_cache_type': 'Q4_0',
        'caveat': 'Same runtime, chat template, effective sampler and context/KV policy as bonsai; '
                  'the GGUF artifact and its weights differ. Binary rubric only, no promotion.'}
    assert 'topK=40 is inactive' in result['capture_warnings']['candidate'][0]


def test_candidate_only_invocation_stays_out_of_the_pair_union(candidate_sample):
    # The pair row is the documented reference-minus-bonsai gap, so a third arm routing a
    # turn neither other arm invoked must not enter its denominator. Measured before this
    # fix: three candidate-only invocations moved the pair's bonsai rate 0.6458 -> 0.6667
    # and its delta +0.1042 -> +0.0833, while all_app_turns stayed bit-identical.
    before = run_three(candidate_sample)['summaries']
    directory = candidate_sample[1]['candidate']
    record = json.loads((directory / 'invocation-000000.json').read_text())
    record.update(conversationID='b', turn=0, ordinal=1)
    save(directory / 'invocation-000001.json', record)
    mutate_capture(candidate_sample, 'candidate', 'capture-run.json',
                   lambda x: x.update(invocationCount='2'))
    after = run_three(candidate_sample)['summaries']
    assert after['model_involved_app_turns'] == before['model_involved_app_turns']
    assert after['all_app_turns'] == before['all_app_turns']
    with_candidate = after['model_involved_app_turns_with_candidate']
    assert with_candidate['candidate']['turn_count'] == 3
    # The candidate-only turn is still visible per case, via the any-role flag.
    assert after['model_involved_app_turns_with_candidate']['candidate']['turn_count'] == 3
    assert any(row['model_involved_union'] and row['candidate']['model_invoked']
               for row in run_three(candidate_sample)['cases'])


def test_candidate_role_must_be_declared_and_supplied_together(sample):
    reports, captures, protocol = sample
    with pytest.raises(ValueError, match='supplied together'):
        compare(reports['reference'], reports['bonsai'], captures['reference'], captures['bonsai'],
                protocol, reports['reference'])
    with pytest.raises(ValueError, match='declared in the protocol'):
        compare(reports['reference'], reports['bonsai'], captures['reference'], captures['bonsai'],
                protocol, reports['reference'], captures['reference'])


def test_candidate_capture_must_share_bonsai_runtime_policy(candidate_sample):
    mutate_capture(candidate_sample, 'candidate', 'invocation-000000.json',
                   lambda x: x['metadata'].update(template='Gemma3Template'))
    with pytest.raises(ValueError, match='chat template'):
        run_three(candidate_sample)


def test_candidate_effective_sampler_must_match_bonsai(candidate_sample):
    mutate_capture(candidate_sample, 'candidate', 'invocation-000000.json',
                   lambda x: x['sampler'].update(maxTokens=512))
    with pytest.raises(ValueError, match='effective sampler'):
        run_three(candidate_sample)


@pytest.mark.parametrize('field,value', [('contextTokens', 8192), ('kvCacheType', 'F16')])
def test_candidate_context_and_kv_policy_must_match_bonsai(candidate_sample, field, value):
    candidate_sample[0]['candidate'][field] = value
    with pytest.raises(ValueError, match=field):
        run_three(candidate_sample)


@pytest.mark.parametrize('mutation', ['runtime', 'parity', 'hash', 'status'])
def test_candidate_protocol_parity_is_enforced(candidate_sample, mutation):
    model = candidate_sample[2]['models']['candidate']
    if mutation == 'runtime':
        model['runtime'] = 'mlx'
    elif mutation == 'parity':
        model.pop('runtime_parity')
    elif mutation == 'hash':
        model['manifest'] = save(Path(model['manifest']['path']), {
            'status': 'runtime-verified', 'artifact_bytes': Path(model['artifact']).stat().st_size,
            'artifact_sha256': '0' * 64})
    else:
        model['manifest'] = save(Path(model['manifest']['path']), {
            'status': 'exported', 'artifact_bytes': Path(model['artifact']).stat().st_size,
            'artifact_sha256': sha256(model['artifact'])})
    with pytest.raises(ValueError, match='[Cc]andidate'):
        run_three(candidate_sample)


def test_equal_conversation_weighting_and_paired_union(sample):
    result = run(sample)
    all_turns = result['summaries']['all_app_turns']
    assert all_turns['reference']['conversation_weighted_pass_rate'] == .75
    assert all_turns['bonsai']['conversation_weighted_pass_rate'] == .5
    involved = result['summaries']['model_involved_app_turns']
    assert involved['reference']['turn_count'] == involved['bonsai']['turn_count'] == 2
    assert involved['reference']['conversation_weighted_pass_rate'] == .5
    assert involved['bonsai']['conversation_weighted_pass_rate'] == 0
    assert result['cases'][0]['reference']['model_invoked']
    assert not result['cases'][0]['bonsai']['model_invoked']
    assert not result['promotion_eligible'] and result['critical_judgments_missing'] == 6
    assert result['cases'][1]['reference']['raw_turn_report']['failures'] == ['missing anchor']


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'user', 'unscored', 'sampler', 'runtime'])
def test_report_drift_rejected(sample, mutation):
    report = sample[0]['reference']
    if mutation == 'missing':
        report['turns'].pop()
    elif mutation == 'extra':
        report['turns'].append(dict(report['turns'][0], turn=3))
    elif mutation == 'duplicate':
        report['turns'].append(copy.deepcopy(report['turns'][0]))
    elif mutation == 'user':
        report['turns'][0]['user'] = 'changed'
    elif mutation == 'unscored':
        report['turns'][0]['passed'] = None
    elif mutation == 'sampler':
        report['samplingTemperature'] = .5
    elif mutation == 'runtime':
        report['runtime'] = 'llamacpp'
    with pytest.raises(ValueError):
        run(sample)


@pytest.mark.parametrize('mutation', ['incomplete', 'count', 'ordinal', 'turn', 'hash', 'sampler', 'model', 'extra'])
def test_capture_drift_rejected(sample, mutation):
    if mutation in ('incomplete', 'count'):
        mutate_capture(sample, 'reference', 'capture-run.json',
                       lambda x: x.update(**({'status': 'running'} if mutation == 'incomplete'
                                            else {'invocationCount': '2'})))
    elif mutation == 'extra':
        (sample[1]['reference'] / 'unexpected.json').write_text('{}')
    else:
        fields = {'ordinal': {'ordinal': 1}, 'turn': {'turn': 99}, 'hash': {'prompt': 'corrupt'},
                  'sampler': {'sampler': {'temperature': .1}}, 'model': {'modelID': 'other'}}
        mutate_capture(sample, 'reference', 'invocation-000000.json', lambda x: x.update(fields[mutation]))
    with pytest.raises(ValueError):
        run(sample)


@pytest.mark.parametrize('artifact', ['source', 'manifest', 'evaluator', 'suite'])
def test_artifact_checksum_drift_rejected(sample, artifact):
    protocol = sample[2]
    path = {'source': protocol['sources']['zim']['path'],
            'manifest': protocol['models']['reference']['manifest']['path'],
            'evaluator': protocol['evaluator']['path'],
            'suite': protocol['source_suites'][0]['path']}[artifact]
    original = Path(path).read_bytes()
    Path(path).write_bytes(b'X' + original[1:])
    with pytest.raises(ValueError, match='checksum'):
        run(sample)


def test_mlx_missing_seed_explicit_warning_and_gguf_missing_seed_rejected(sample):
    mutate_capture(sample, 'reference', 'invocation-000000.json', lambda x: x['sampler'].pop('seed'))
    assert run(sample)['capture_warnings']['reference']
    mutate_capture(sample, 'bonsai', 'invocation-000000.json', lambda x: x['sampler'].pop('seed'))
    with pytest.raises(ValueError, match='seed'):
        run(sample)


def test_llama_greedy_ignores_recorded_task_default_top_k(sample):
    mutate_capture(sample, 'bonsai', 'invocation-000000.json', lambda x: x['sampler'].update(topK=40))
    assert 'inactive' in run(sample)['capture_warnings']['bonsai'][0]
    mutate_capture(sample, 'bonsai', 'invocation-000000.json', lambda x: x['sampler'].update(temperature=.1))
    with pytest.raises(ValueError, match='temperature'):
        run(sample)


def test_mlx_top_k_mismatch_still_rejected(sample):
    mutate_capture(sample, 'reference', 'invocation-000000.json', lambda x: x['sampler'].update(topK=40))
    with pytest.raises(ValueError, match='topK'):
        run(sample)


def test_zero_invocation_runs_do_not_claim_model_quality(sample):
    for role in ('reference', 'bonsai'):
        (sample[1][role] / 'invocation-000000.json').unlink()
        mutate_capture(sample, role, 'capture-run.json', lambda x: x.update(invocationCount='0'))
    result = run(sample)
    assert result['summaries']['model_involved_app_turns']['reference']['conversation_weighted_pass_rate'] is None
    assert result['summaries']['all_app_turns']['reference']['turn_count'] == 3


def test_repeated_invocations_within_one_turn_are_valid(sample):
    directory = sample[1]['reference']
    record = json.loads((directory / 'invocation-000000.json').read_text())
    record['ordinal'] = 1
    save(directory / 'invocation-000001.json', record)
    mutate_capture(sample, 'reference', 'capture-run.json', lambda x: x.update(invocationCount='2'))
    assert run(sample)['cases'][0]['reference']['invocation_ordinals'] == [0, 1]


def test_atomic_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        write_atomic(tmp_path / 'result.json', {'invalid': float('nan')})
    assert not (tmp_path / 'result.json').exists()
    assert list(tmp_path.iterdir()) == []


def test_embedded_suite_drift_rejected(sample):
    sample[2]['suite']['conversations'][0]['turns'][0]['user'] = 'changed'
    with pytest.raises(ValueError, match='merged suite mismatch'):
        run(sample)


def test_capture_wrong_suite_path_rejected(sample):
    mutate_capture(sample, 'reference', 'capture-run.json', lambda x: x.update(suitePath='/wrong.json'))
    with pytest.raises(ValueError, match='suitePath'):
        run(sample)


def test_bool_sampler_rejected(sample):
    sample[0]['reference']['samplingTemperature'] = False
    with pytest.raises(ValueError, match='type'):
        run(sample)


def test_inconsistent_pass_and_failure_rejected(sample):
    sample[0]['reference']['turns'][0]['failures'] = ['critical factual error']
    with pytest.raises(ValueError, match='inconsistent'):
        run(sample)


def test_inconsistent_aggregate_rejected(sample):
    sample[0]['reference']['passedTurns'] += 1
    with pytest.raises(ValueError, match='totals'):
        run(sample)


def test_reference_manifest_must_be_validated_bf16(sample):
    protocol = sample[2]
    item = protocol['models']['reference']['manifest']
    protocol['models']['reference']['manifest'] = save(Path(item['path']), {
        'status': 'validated', 'precision': 'Q4', 'model': 'Qwen/Qwen3.8-27B'})
    with pytest.raises(ValueError, match='original BF16'):
        run(sample)
