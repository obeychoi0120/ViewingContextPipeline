"""Cross-run cache contracts; generation/BGE fakes, only tiny CPU recommendation data."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import shutil

import numpy as np
import pytest

from pipeline_runtime import RunContext, read_json, write_json
from validation.representation_provenance import read_state
from validation.selection import load_validation_cohort
from validation.shared_cache import SharedCache
from validation.steps import embed_representations


def next_run(context, name):
    other = RunContext.load(name, root=context.root)
    other.config.update(deepcopy(context.config))
    return other


def forbid(*a, **k):
    pytest.fail('cache hit must not initialize encoder/GPU or train')


def test_all_arm_reuse_and_independent_invalidation(ready_context, fake_models, monkeypatch):
    from test_pipeline import generate_all
    first = ready_context
    generate_all(first)
    embed_representations(first, summary_source='qwen')
    old = {a: read_state(first, a)['input_hash'] for a in first.config['protocol']['arms']}
    second = next_run(first, 'second')
    # Explicit fixture transfer only; production never copies generation outputs.
    shutil.copytree(first.run_root / 'extraction', second.run_root / 'extraction')
    with monkeypatch.context() as patch:
        patch.setattr('validation.features.BGETextEncoder', forbid)
        result = embed_representations(second, summary_source='qwen')
        assert set(result['reuse']['shared']) == set(old)
        assert not result['generated_arms']
    path = next(second.graph_summary_dir('gemini').glob('*.json'))
    doc = read_json(path)
    doc.update(status='failed', text='', word_count=0, violations=['failed'])
    write_json(path, doc)
    result = embed_representations(second, summary_source='qwen')
    assert result['generated_arms'] == ['graph_gemini']
    for arm in old.keys() - {'graph_gemini'}:
        assert read_state(second, arm)['input_hash'] == old[arm]
    cohort = load_validation_cohort(second)
    assert len(cohort['catalog']) == 4 and len(cohort['events']) == 72
    with np.load(second.representations_dir / 'graph_gemini_embeddings.npz') as arrays:
        index = next(i for i, row in enumerate(cohort['catalog']) if row['content_id'] == path.stem)
        assert not arrays['values'][index].any()
    # Restoring the expression also restores its prior shared embedding.
    write_json(path, read_json(first.graph_summary_dir('gemini') / path.name))
    with monkeypatch.context() as patch:
        patch.setattr('validation.features.BGETextEncoder', forbid)
        assert embed_representations(second, summary_source='qwen')['reuse']['shared'] == ['graph_gemini']


@pytest.mark.parametrize('field', ['text', 'summary_prompt', 'scene_prompt', 'model', 'settings'])
def test_visual_keys_bind_actual_input_and_generation(ready_context, fake_models, field):
    from test_pipeline import generate_all
    ctx = ready_context
    generate_all(ctx)
    embed_representations(ctx, summary_source='qwen')
    path = next(ctx.graph_summary_dir('gemini').glob('*.json'))
    doc = read_json(path)
    if field == 'text':
        doc['text'] += ' Changed.'
        doc['word_count'] += 1
    elif field == 'summary_prompt':
        doc['provenance']['prompt_hash'] = 'changed body, same filename'
    elif field == 'scene_prompt':
        doc['provenance']['scene_provenance'][0]['prompt_hash'] = 'changed body'
    elif field == 'model':
        doc['provenance']['model']['files_signature'] = 'new weights'
    else:
        doc['provenance']['settings']['max_new_tokens'] += 1
    write_json(path, doc)
    result = embed_representations(ctx, summary_source='qwen')
    assert result['generated_arms'] == ['graph_gemini']


def test_unknown_visual_provenance_stays_local_and_force_does_not_overwrite(ready_context, fake_models):
    from test_pipeline import generate_all
    ctx = ready_context
    generate_all(ctx)
    for path in ctx.graph_summary_dir('gemini').glob('*.json'):
        doc = read_json(path)
        doc['provenance'].pop('scene_provenance')
        write_json(path, doc)
    embed_representations(ctx, summary_source='qwen')
    state = read_state(ctx, 'graph_gemini')
    assert not state['shareable']
    assert not SharedCache(ctx, 'embeddings', state['input_hash']).path.exists()
    cache = SharedCache(ctx, 'embeddings', read_state(ctx, 'metadata')['input_hash'])
    before = {p.name: p.read_bytes() for p in cache.path.iterdir()}
    assert embed_representations(ctx, summary_source='qwen', target=['metadata'], force=True)['generated_arms'] == ['metadata']
    assert before == {p.name: p.read_bytes() for p in cache.path.iterdir()}


def test_atomic_concurrent_publication_and_corruption(ready_context, tmp_path):
    cache = SharedCache(ready_context, 'embeddings', 'test-key')
    sources = []
    for i in range(6):
        source = tmp_path / str(i)
        source.mkdir()
        (source / 'a').write_text(str(i))
        (source / 'b').write_text(str(i))
        sources.append(source)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda p: cache.publish(p, ['a', 'b'], origin={'run_id': p.name}), sources))
    restored = tmp_path / 'restored'
    assert cache.restore(restored)
    assert (restored / 'a').read_bytes() == (restored / 'b').read_bytes()
    (cache.path / 'a').write_text('corrupt')
    assert cache.restore(restored) is None
    cache.publish(sources[0], ['a', 'b'], origin={'run_id': '0'})
    assert cache.valid()
    (cache.path / 'cache.json').unlink()
    assert cache.restore(restored) is None


@pytest.mark.torch
def test_metadata_recommendation_reuses_checkpoint_and_metrics(ready_context, fake_models, monkeypatch):
    import torch
    from validation.rolling_recommendation import run_rolling
    from validation.rolling_diagnosis import diagnose
    first = ready_context
    first.config['protocol']['arms'] = ['metadata']
    embed_representations(first, summary_source='qwen')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_rolling(first)['completed'] == 21
    finally:
        torch.set_num_threads(threads)
    second = next_run(first, 'reused')
    with monkeypatch.context() as patch:
        patch.setattr('validation.features.BGETextEncoder', forbid)
        patch.setattr('validation.rolling_recommendation.worker_devices', forbid)
        patch.setattr('validation.rolling_recommendation.run_combination', forbid)
        embed_representations(second, summary_source='qwen')
        assert run_rolling(second)['skipped'] == 21
    assert read_json(second.recommendations_dir / 'reuse.json')['shared'] == 21
    assert diagnose(second, target=['metadata'])['status'] == 'pass'
    for path in first.recommendations_dir.rglob('sasrec.pt'):
        copied = second.recommendations_dir / path.relative_to(first.recommendations_dir)
        original = torch.load(path, weights_only=True)
        reused = torch.load(copied, weights_only=True)
        assert reused['metadata']['run_id'] == second.run_id
        assert reused['metadata']['reused_from']['source_run_id'] == first.run_id
        assert all(torch.equal(v, reused['state_dict'][k]) for k, v in original['state_dict'].items())
        assert path.stat().st_ino != copied.stat().st_ino
    # Both a corrupt local result and its corrupt shared source must be recomputed.
    complete = next(second.recommendations_dir.rglob('complete.json'))
    identity = read_json(complete)['identity']
    from validation.recommendation_cache import cache_for
    cache = cache_for(second, identity)
    (cache.path / 'sasrec.pt').write_bytes(b'corrupt')
    (complete.parent / 'sasrec.pt').write_bytes(b'corrupt')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_rolling(second)['completed'] == 1
    finally:
        torch.set_num_threads(threads)
    entries = list((first.run_root.parent.parent / 'shared_cache/v2/recommendations').glob('*/cache.json'))
    before = {p: p.read_bytes() for p in entries}
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr('validation.rolling_recommendation.worker_devices', lambda _: ['cpu'])
        patch.setattr('validation.rolling_recommendation.run_combination',
                      lambda *args: calls.append(args[4]))
        assert run_rolling(second, force=True, target=['metadata'])['completed'] == 21
    assert len(calls) == 21
    assert before == {p: p.read_bytes() for p in entries}


@pytest.mark.parametrize('representation', ['description', 'graph'])
def test_summary_uses_only_successful_scenes_and_recovers_empty(ready_context, fake_models, representation):
    import extraction.steps as steps
    from extraction.failures import FailureLog
    from extraction.scene_storage import read_scene_records, write_scene_records
    ctx = ready_context
    extract = getattr(steps, f'extract_{representation}_scenes')
    summarize = getattr(steps, f'summarize_{representation}')
    extract(ctx, model='qwen', schema=f'prompts/scene_{representation}_v{3 if representation == "graph" else 2}.md')
    directory = ctx.extraction_dir(representation, 'qwen', 'scenes')
    paths = sorted(directory.glob('*.jsonl'))
    good = read_scene_records(paths[0])[0]
    # A legacy failed scene must never enter the summarization request.
    raw = ({'schema_version': 'graph-scene-raw/v1', 'status': 'raw_fallback',
            'scene_idx': 1, 'keyframes': [], 'raw_response': 'FORBIDDEN FAILURE TEXT'}
           if representation == 'graph' else
           {**good, 'scene_idx': 1, 'description': 'FORBIDDEN FAILURE TEXT'})
    write_scene_records(paths[0], [good, raw])
    failures = FailureLog(directory, scenes=True)
    failures.record(paths[0].stem, 1, 'failed', 'FORBIDDEN FAILURE TEXT')
    failures.record(paths[1].stem, 0, 'failed', '', provenance=good['provenance'])
    second_good = read_scene_records(paths[1])
    paths[1].unlink()
    fake_models.clear()
    options = dict(model='qwen', source='qwen', schema=f'prompts/summary_{representation}_v4.md')
    result = summarize(ctx, **options)
    assert result['failure_count'] == 1
    tasks = [task for batch in fake_models for task in batch]
    assert len(tasks) == 3
    assert all('FORBIDDEN FAILURE TEXT' not in task.prompt for task in tasks)
    output = ctx.summary_dir(representation, 'qwen', 'qwen')
    partial = read_json(output / f'{paths[0].stem}.json')
    empty = read_json(output / f'{paths[1].stem}.json')
    assert partial['scene_count'] == 1
    assert empty['status'] == 'failed' and empty['text'] == '' and empty['scene_count'] == 0
    # Retry after scene recovery even though the previous failure had no penalty.
    failures.remove(paths[1].stem, 0)
    write_scene_records(paths[1], second_good)
    fake_models.clear()
    assert summarize(ctx, **options)['failure_count'] == 0
    assert [task.task_id for batch in fake_models for task in batch] == [paths[1].stem]


def test_keys_separate_data_model_and_arm_conditions(ready_context, fake_models):
    from arm_registry import select_arms
    from validation.representation_inputs import documents_for_arm, representation_signature
    from validation.selection import training_signature, data_identity
    from validation.steps import validation_config
    ctx = ready_context
    embed_representations(ctx, summary_source='qwen', target=['metadata'])
    cohort = load_validation_cohort(ctx)
    arm = select_arms(ctx.config)['metadata']
    docs = documents_for_arm(ctx, cohort, arm)
    embedding = representation_signature(ctx, cohort['catalog'], arm, docs)
    common = deepcopy(data_identity(cohort))
    training = training_signature(ctx, cohort, validation_config(ctx))
    ctx.config['models']['gemini']['model_id'] = 'different visual model'
    assert representation_signature(ctx, cohort['catalog'], arm, docs) == embedding
    assert training_signature(ctx, cohort, validation_config(ctx)) == training
    ctx.config['validation']['model']['learning_rate'] *= 2
    assert training_signature(ctx, cohort, validation_config(ctx)) != training
    assert representation_signature(ctx, cohort['catalog'], arm, docs) == embedding
    cohort['events'][0]['timestamp'] += 1
    assert data_identity(cohort) != common
    # Data changes affect recommendation identity, not encoding of the same ordered texts.
    assert training_signature(ctx, cohort, validation_config(ctx)) != training
    assert representation_signature(ctx, cohort['catalog'], arm, docs) == embedding
    ctx.config['validation']['encoder']['batch_size'] += 1
    assert representation_signature(ctx, cohort['catalog'], arm, docs) != embedding
