"""Nine-arm public API, explicit migration and title-isolated cache integration."""
from copy import deepcopy
from dataclasses import replace
import shutil

import numpy as np
import pytest

from arm_registry import registry, ARM_CONTRACT
from pipeline_runtime import read_json, write_json
from extraction import steps
from extraction.arm_migration import migrate_arm_layout
from validation.steps import embed_representations
from validation.representation_provenance import read_state
from validation.selection import load_validation_cohort

NAMES = ['meta', 'graph_qwen', 'graph_gemini', 'desc_qwen', 'desc_gemini',
         'graph_meta_qwen', 'graph_meta_gemini', 'desc_meta_qwen', 'desc_meta_gemini']


@pytest.fixture
def nine(ready_context, fake_models):
    ctx = ready_context
    ctx.config['schema_version'] = 'viewing-context-config/v6'
    ctx.config['protocol']['arms'] = list(NAMES)
    for kind in ('graph', 'description'):
        template = (ctx.root / f'prompts/summary_{kind}_v4.md').read_text()
        (ctx.root / f'prompts/summary_{kind}_v4_meta.md').write_text(template.replace("Scene observations:", "English Title: {english_title}\n\nScene observations:"))
    return ctx


def scenes(ctx, name):
    arm = registry(ctx.config)[name]
    return getattr(steps, f'extract_{arm.representation}_scenes')(
        ctx, arm=name, model=arm.model,
        schema=f'prompts/scene_{arm.representation}_v{3 if arm.representation == "graph" else 2}.md')


def summary(ctx, name, model='qwen', **kwargs):
    arm = registry(ctx.config)[name]
    return getattr(steps, f'summarize_{arm.representation}')(
        ctx, arm=name, model=model,
        schema=f'prompts/summary_{arm.representation}_v4{"_meta" if arm.uses_title else ""}.md', **kwargs)


def generate(ctx):
    for name, arm in registry(ctx.config).items():
        if arm.model and name == arm.scene_arm:
            scenes(ctx, name)
    for name, arm in registry(ctx.config).items():
        if arm.model:
            summary(ctx, name)


def test_nine_arms_share_four_scene_sources(nine, fake_models):
    assert list(registry(nine.config)) == NAMES
    generate(nine)
    assert len(fake_models) == 12  # Four extraction passes, eight Summary passes.
    assert len(list((nine.run_root / 'extraction/scenes').glob('*/*.jsonl'))) == 16
    assert len(list((nine.run_root / 'extraction/summaries').glob('*/*.json'))) == 32
    result = embed_representations(nine)
    assert result['generated_arms'] == NAMES
    assert load_validation_cohort(nine)['manifest']['arm_contract'] == ARM_CONTRACT
    assert len(load_validation_cohort(nine)['events']) == 72
    for arm in registry(nine.config).values():
        state = read_state(nine, arm.name)
        assert state['shareable']
        if arm.model:
            for doc in state['sources']:
                prov = doc['source_provenance']
                assert 'arm' not in prov
                assert all('arm' not in scene for scene in prov.get('scene_provenance', []))
                assert prov['scene_arm'] == arm.scene_arm
                assert ('english_title' in prov) == arm.uses_title
    titles = [r['title'] for r in nine.require_ready_cohort()['metadata_titles'] if r['title']]
    # First four batches contain Scene tasks; remaining batches follow registry Summary order.
    for name, batch in zip(NAMES[1:], fake_models[4:], strict=True):
        if not registry(nine.config)[name].uses_title:
            assert all('English Title:' not in task.prompt for task in batch)
            assert all(title not in task.prompt for task in batch for title in titles)


@pytest.mark.parametrize('name', ['graph_qwen', 'desc_meta_gemini', 'meta'])
def test_partial_targets_require_no_other_outputs(nine, name):
    embed_representations(nine, target=[name])
    assert {p.stem for p in nine.representations_dir.glob('*.npz')} == {f'{name}_embeddings'}
    assert len(load_validation_cohort(nine)['catalog']) == 4
    if name != 'meta':
        with np.load(nine.representations_dir / f'{name}_embeddings.npz') as data:
            assert not data['values'].any()


def test_schema_contract_and_model_change_isolation(nine, fake_models):
    with pytest.raises(ValueError, match='Scene arm'):
        scenes(nine, 'graph_meta_qwen')
    with pytest.raises(ValueError, match='Scene arm'):
        steps.extract_graph_scenes(nine, arm='graph_qwen', model='gemini', schema='prompts/scene_graph_v3.md')
    with pytest.raises(ValueError, match='english_title'):
        steps.summarize_graph(nine, arm='graph_qwen', model='qwen', schema='prompts/summary_graph_v4_meta.md')
    scenes(nine, 'graph_qwen')
    summary(nine, 'graph_qwen')
    summary(nine, 'graph_meta_qwen')
    saved = {p: p.read_bytes() for p in nine.summary_arm_dir('graph_meta_qwen').glob('*.json')}
    with pytest.raises(RuntimeError, match='model mismatch'):
        summary(nine, 'graph_qwen', model='gemini')
    summary(nine, 'graph_qwen', model='gemini', force=True)
    assert saved == {p: p.read_bytes() for p in saved}
    assert len(fake_models) == 4


def test_title_change_does_not_invalidate_title_free_arm(nine, monkeypatch):
    scenes(nine, 'graph_qwen')
    summary(nine, 'graph_qwen')
    summary(nine, 'graph_meta_qwen')
    embed_representations(nine, target=['meta', 'graph_qwen', 'graph_meta_qwen'])
    before = read_state(nine, 'graph_qwen')['input_hash']
    source = deepcopy(nine.require_ready_cohort())
    for row in source['metadata_titles']:
        row['title'] += ' UPDATED TITLE'
    monkeypatch.setattr(type(nine), 'require_ready_cohort', lambda _: source)
    summary(nine, 'graph_qwen', force=True)
    summary(nine, 'graph_meta_qwen', force=True)
    result = embed_representations(nine, target=['meta', 'graph_qwen', 'graph_meta_qwen'])
    assert result['generated_arms'] == ['meta', 'graph_meta_qwen']
    assert read_state(nine, 'graph_qwen')['input_hash'] == before


def test_new_run_reuses_all_embeddings_without_encoder(nine, monkeypatch):
    generate(nine)
    embed_representations(nine)
    other = replace(nine, run_id='second', run_root=nine.run_root.parent / 'second')
    shutil.copytree(nine.run_root / 'extraction', other.run_root / 'extraction')
    def fail(*a, **k):
        pytest.fail('shared embedding must not initialize encoder')
    monkeypatch.setattr('validation.features.BGETextEncoder', fail)
    assert embed_representations(other)['reuse']['shared'] == NAMES
    assert (nine.run_root.parent.parent / 'shared_cache/v2/embeddings').is_dir()


def legacy_generation(ctx):
    settings = deepcopy(ctx.config)
    settings['schema_version'] = 'viewing-context-config/v5'
    settings['protocol']['arms'] = ['metadata', 'graph_qwen', 'graph_gemini', 'desc_qwen', 'desc_gemini']
    old = replace(ctx, config=settings)
    for representation in ('graph', 'description'):
        for model in ('qwen', 'gemini'):
            getattr(steps, f'extract_{representation}_scenes')(
                old, model=model, schema=f'prompts/scene_{representation}_v{3 if representation == "graph" else 2}.md')
            getattr(steps, f'summarize_{representation}')(
                old, source=model, model='qwen', schema=f'prompts/summary_{representation}_v4_meta.md')
    return old


def test_explicit_migration_preserves_originals_and_normalizes_raw(nine):
    old = legacy_generation(nine)
    raw = next(old.graph_summary_dir('qwen').glob('*.json'))
    doc = read_json(raw)
    doc.update(status='raw_fallback', violations=['max_tokens'])
    write_json(raw, doc)
    before = {p: p.read_bytes() for p in (nine.run_root / 'extraction').rglob('*') if p.is_file()}
    result = migrate_arm_layout(nine, summary_model='qwen')
    assert result['file_count'] == 33 and not result['skipped']
    assert before == {p: p.read_bytes() for p in before}
    migrated = read_json(nine.summary_arm_dir('graph_meta_qwen') / raw.name)
    assert migrated['text'] == '' and migrated['status'] == 'failed'
    assert migrated['provenance'] == doc['provenance']
    assert not nine.summary_arm_dir('graph_qwen').exists()
    assert migrate_arm_layout(nine, summary_model='qwen') == result
    embed_representations(nine)
    assert read_state(nine, 'graph_qwen')['zero_vector_count'] == 4
    with pytest.raises(ValueError, match='different Summary model'):
        migrate_arm_layout(nine, summary_model='gemini')
    target = nine.summary_arm_dir('graph_meta_qwen') / raw.name
    changed = read_json(target)
    changed['violations'] = ['changed']
    write_json(target, changed)
    with pytest.raises(ValueError, match='conflict'):
        migrate_arm_layout(nine, summary_model='qwen')


def test_migration_skips_unverified_title_provenance(nine):
    old = legacy_generation(nine)
    path = next(old.description_summary_dir('gemini').glob('*.json'))
    doc = read_json(path)
    del doc['provenance']['english_title']
    write_json(path, doc)
    result = migrate_arm_layout(nine, summary_model='qwen')
    assert len(result['skipped']) == 1
    assert not (nine.summary_arm_dir('desc_meta_gemini') / path.name).exists()


def test_comparison_families_and_partial_dimensions(nine):
    from validation.diagnosis_statistics import comparison_families, multiple_comparison_policy
    from validation.rolling_diagnosis import comparisons
    from validation.steps import validation_config
    assert {k: len(v) for k, v in comparison_families(nine.config).items()} == {
        'metadata_baseline': 8, 'representation': 4, 'model': 4, 'title_input': 4}
    config = validation_config(nine)
    values = comparisons(np.ones(9), np.ones((100, 9)), config.evaluation, config=nine.config)
    assert len(values) == 22
    partial = ['graph_qwen', 'graph_meta_qwen']
    values = comparisons(np.ones(2), np.ones((100, 2)), config.evaluation, config=nine.config, arms=partial)
    assert list(values) == ['graph_meta_qwen-graph_qwen']
    policy = multiple_comparison_policy(config.evaluation.model_dump(), arms=partial, config=nine.config)
    assert policy['families']['title_input']['comparison_count'] == 4
    assert len(policy['families']['metadata_baseline']['skipped']) == 8


def test_cli_uses_explicit_arms(nine, monkeypatch):
    from extraction.cli import main
    from validation.cli import main as validate
    monkeypatch.setattr('extraction.cli.RunContext.load', lambda _: nine)
    monkeypatch.setattr('validation.cli.RunContext.load', lambda _: nine)
    assert main(['extract-graph-scenes', '--run-id', 'run', '--model', 'qwen', '--arm', 'graph_qwen',
                 '--schema', 'prompts/scene_graph_v3.md']) == 0
    assert main(['summarize-graph', '--run-id', 'run', '--model', 'gemini', '--arm', 'graph_meta_qwen',
                 '--schema', 'prompts/summary_graph_v4_meta.md']) == 0
    assert validate(['embed-representations', '--run-id', 'run', '--target', 'meta', 'graph_meta_qwen']) == 0
    with pytest.raises(SystemExit):
        main(['summarize-graph', '--run-id', 'run', '--source', 'qwen'])
    with pytest.raises(SystemExit):
        validate(['embed-representations', '--run-id', 'run', '--summary-source', 'qwen'])


@pytest.mark.torch
def test_nine_arm_training_reuse_partial_diagnosis_and_run_comparison(nine, monkeypatch):
    import torch
    import yaml
    from validation.model import SASRec
    from validation.rolling_recommendation import run_rolling
    from validation.rolling_diagnosis import diagnose
    from validation.run_comparison import compare_graph_runs
    def tiny(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm='graph', item_features=features).to(device)
    monkeypatch.setattr('validation.rolling_recommendation._new_model', tiny)
    generate(nine)
    embed_representations(nine)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_rolling(nine, target=['graph_qwen'])['completed'] == 21
        assert diagnose(nine, target=['graph_qwen'])['status'] == 'pass'
        assert run_rolling(nine)['completed'] == 168
        assert diagnose(nine)['status'] == 'pass'
        assert len(read_json(nine.diagnosis_path)['statistics']['comparisons']) == 22
        other = replace(nine, run_id='second', run_root=nine.run_root.parent / 'second')
        shutil.copytree(nine.run_root / 'extraction', other.run_root / 'extraction')
        (nine.root / 'config.yaml').write_text(yaml.safe_dump(nine.config))
        def fail(*a, **k):
            pytest.fail('cache hit must not encode or train')
        monkeypatch.setattr('validation.features.BGETextEncoder', fail)
        monkeypatch.setattr('validation.rolling_recommendation.worker_devices', fail)
        monkeypatch.setattr('validation.rolling_recommendation.run_combination', fail)
        assert embed_representations(other)['reuse']['shared'] == NAMES
        assert run_rolling(other)['skipped'] == 189
        assert diagnose(other)['status'] == 'pass'
        comparison = compare_graph_runs(other, nine.run_id)
        assert len(comparison['comparisons']) == comparison['family_size'] == 4
        assert all(value['difference'] == 0 for value in comparison['comparisons'].values())
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('provenance', [
    {'representation': 'description'}, {'summary_model': 'gemini'}, {'uses_title': True},
    {'scene_arm': 'graph_gemini'}, {'english_title': 'unexpected title'},
])
def test_failure_only_wrong_provenance_is_not_treated_as_missing(nine, provenance):
    from pipeline_runtime import write_jsonl
    directory = nine.summary_arm_dir('graph_qwen')
    write_jsonl(directory / 'failures.jsonl', [{
        'content_id': str(nine.require_ready_cohort()['catalog'][0]['content_id']),
        'summary_model': 'qwen', 'error': 'failed', 'raw_output': '', 'provenance': provenance,
    }])
    with pytest.raises(ValueError, match='provenance|policy|title'):
        embed_representations(nine, target=['graph_qwen'])


def test_migration_rejects_conflicting_destination_failure(nine):
    from pipeline_runtime import write_jsonl
    legacy_generation(nine)
    write_jsonl(nine.summary_arm_dir('graph_meta_qwen') / 'failures.jsonl', [{
        'content_id': 'existing', 'summary_model': 'qwen', 'error': 'different result', 'raw_output': '',
    }])
    with pytest.raises(ValueError, match='conflict'):
        migrate_arm_layout(nine, summary_model='qwen')
    assert not list((nine.run_root / 'extraction/scenes').glob('*/*.jsonl'))


@pytest.mark.torch
def test_new_config_diagnoses_historical_metadata_without_reinterpreting(nine, monkeypatch):
    from validation.model import SASRec
    from validation.rolling_recommendation import run_rolling
    from validation.rolling_diagnosis import diagnose
    from validation.selection import diagnosis_context
    old_config = deepcopy(nine.config)
    old_config['schema_version'] = 'viewing-context-config/v5'
    old_config['protocol']['arms'] = ['metadata', 'desc_qwen', 'desc_gemini', 'graph_qwen', 'graph_gemini']
    old = replace(nine, config=old_config)
    monkeypatch.setattr('validation.rolling_recommendation._new_model',
                        lambda config, *, item_count, branch, features, device:
                        SASRec(item_count, 10, 8, 1, 2, 0, arm='graph', item_features=features).to(device))
    embed_representations(old, target=['metadata'])
    run_rolling(old, target=['metadata'])
    adapted = diagnosis_context(nine)
    assert registry(adapted.config)['graph_qwen'].uses_title
    assert diagnose(nine, target=['metadata'])['status'] == 'pass'


def test_unstructured_graph_is_successful_summary_input(nine, monkeypatch):
    from contextlib import contextmanager
    from extraction.scene_storage import read_scene_records
    text = '[Entities]\nperson1: person\n[Relations]\nperson1 -> waving ->'
    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                kwargs['runtime'].current_result = {'finish_reason': 'stop'}
                callback(task.task_id, text)
            return {}
        yield generate
    monkeypatch.setattr(steps, 'qwen_generator', generator)
    assert scenes(nine, 'graph_qwen')['failure_count'] == 0
    assert not list((nine.scene_arm_dir('graph_qwen') / 'failures').glob('*.jsonl'))
    for path in nine.scene_arm_dir('graph_qwen').glob('*.jsonl'):
        assert all(row['graph'] == text and row['parse_mode'] == 'text' for row in read_scene_records(path))
    for name in ('graph_qwen', 'graph_meta_qwen'):
        assert summary(nine, name, model='gemini')['failure_count'] == 0
        assert all(read_json(path)['scene_count'] > 0 for path in nine.summary_arm_dir(name).glob('*.json'))
    embed_representations(nine, target=['graph_qwen', 'graph_meta_qwen'])
    assert read_state(nine, 'graph_qwen')['zero_vector_count'] == 0


def test_relocated_summary_normalization_preserves_generation_evidence(nine):
    from extraction.arm_migration import normalize_summary_arm
    from validation.cache_identity import generation_identity
    from validation.representation_inputs import documents_for_arm
    old = legacy_generation(nine)
    source = next(old.graph_summary_dir('qwen').glob('*.json'))
    original = read_json(source)
    original['provenance']['arm'] = 'graph_qwen'
    original['provenance'].pop('uses_title', None)
    original['provenance'].pop('scene_arm', None)
    original['provenance']['scene_provenance'][0]['arm'] = 'graph_qwen'
    arm = registry(nine.config)['graph_meta_qwen']
    result = normalize_summary_arm(original, arm, source_path=str(source), source_hash='original-hash')
    assert result['arm'] == arm.name
    assert 'arm' not in result['provenance']
    assert 'arm' not in result['provenance']['scene_provenance'][0]
    assert result['text'] == original['text']
    assert result['provenance']['model'] == original['provenance']['model']
    assert result['provenance']['prompt_hash'] == original['provenance']['prompt_hash']
    assert 'uses_title' not in result['provenance']
    assert generation_identity(result['provenance']) == generation_identity(original['provenance'])
    assert normalize_summary_arm(result, arm, source_path=str(source), source_hash='new-hash') == result
    write_json(nine.summary_arm_dir(arm.name) / source.name, result)
    docs = documents_for_arm(nine, nine.require_ready_cohort(), arm)
    assert next(d for d in docs if d['content_id'] == source.stem)['text'] == original['text']


def test_normalization_command_backs_up_and_is_idempotent(nine, monkeypatch):
    import tarfile
    from extraction.normalize_summary_arms import main
    old = legacy_generation(nine)
    dest = nine.summary_arm_dir('graph_meta_qwen')
    shutil.copytree(old.graph_summary_dir('qwen'), dest)
    before = {p.name: p.read_bytes() for p in dest.glob('*.json')}
    monkeypatch.setattr('extraction.normalize_summary_arms.RunContext.load', lambda _: nine)
    assert main(['--run-id', nine.run_id]) == 0
    backup = next((nine.run_root.parents[1] / 'backups' / nine.run_id).glob('summary-arm-normalization-*'))
    assert read_json(backup / 'manifest.json')['state'] == 'complete'
    with tarfile.open(backup / 'originals.tar.gz') as archive:
        for name, data in before.items():
            assert archive.extractfile(f'extraction/summaries/graph_meta_qwen/{name}').read() == data
    after = {p.name: p.read_bytes() for p in dest.glob('*.json')}
    assert all(read_json(dest / name)['arm'] == 'graph_meta_qwen' for name in after)
    assert main(['--run-id', nine.run_id]) == 0
    assert {p.name: p.read_bytes() for p in dest.glob('*.json')} == after
