"""Six-arm input, generation, cache and diagnostic contracts; no real models."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import shutil

import numpy as np
import pytest

from arm_registry import registry, generation_registry, select_arms, active_arms, resolve_generation_arm
from extraction import steps
from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl
from validation.representation_inputs import documents_for_arm, representation_signature
from validation.representation_provenance import read_state
from validation.steps import embed_representations

NAMES = ['meta', 'graph_qwen', 'desc_qwen', 'graph_qwen_meta', 'desc_qwen_meta', 'graph_gemini_meta']


@pytest.fixture
def six(ready_context):
    ctx = ready_context
    ctx.config.pop('schema_version', None)
    ctx.config['experiment_config_version'] = 'v4'
    ctx.config['protocol']['arms'] = list(NAMES)
    ctx.config['protocol']['description_extractors'] = ['qwen']
    return ctx


def generate(ctx, summary_model='gemini'):
    for name, source in generation_registry(ctx.config).items():
        kind = source.representation
        getattr(steps, f'extract_{kind}_scenes')(
            ctx, arm=name, model=source.model,
            schema=f'prompts/scene_{kind}_v{4 if kind == "graph" else 2}.md')
        getattr(steps, f'summarize_{kind}')(
            ctx, arm=name, model=summary_model, schema=f'prompts/summary_{kind}_v5.md')


def test_registry_and_config(six):
    from pipeline_runtime import ConfigError, _validate_config
    assert list(registry(six.config)) == NAMES
    _validate_config(six.config)
    invalid = deepcopy(six.config)
    invalid['experiment_config_version'] = 'v5'
    with pytest.raises(ConfigError, match='experiment_config_version must be v4'):
        _validate_config(invalid)
    obsolete = deepcopy(six.config)
    obsolete.pop('experiment_config_version')
    obsolete['schema_version'] = 'viewing-context-config/v7'
    with pytest.raises(ConfigError, match='accepted only for historical v5/v6'):
        _validate_config(obsolete)
    assert list(generation_registry(six.config)) == ['graph_qwen', 'desc_qwen', 'graph_gemini']
    for name in ['graph_gemini', 'desc_gemini', 'graph_meta_qwen', 'desc_meta_qwen']:
        with pytest.raises(ValueError):
            select_arms(six.config, [name])
        config = deepcopy(six.config)
        config['protocol']['arms'] = [name]
        with pytest.raises(ValueError):
            active_arms(config)
    with pytest.raises(ValueError, match='use source --arm graph_qwen'):
        resolve_generation_arm(six.config, 'graph_qwen_meta', 'graph', 'gemini', summary=True)
    assert resolve_generation_arm(six.config, 'graph_gemini', 'graph', 'qwen', summary=True).model == 'gemini'


@pytest.mark.parametrize('model', ['qwen', 'gemini'])
def test_shared_title_free_generation(six, fake_models, model):
    generate(six, model)
    assert len(fake_models) == 6
    assert {p.name for p in (six.run_root / 'extraction/summaries').iterdir()} == set(generation_registry(six.config))
    for tasks in fake_models[1::2]:
        assert all(t.max_new_tokens == 2048 for t in tasks)
        assert all('First title' not in t.prompt and 'English Title:' not in t.prompt for t in tasks)
    for source in generation_registry(six.config):
        doc = read_json(next(six.summary_arm_dir(source).glob('*.json')))
        assert doc['provenance']['uses_title'] is False
        assert 'english_title' not in doc['provenance']


def test_composition_cases_and_zero_vectors(six, fake_models):
    generate(six)
    cohort = six.require_ready_cohort()
    titles = cohort['metadata_titles']
    titles[0]['title'] = '  Title\ninside  '
    titles[1]['title'] = '  '
    titles[3]['title'] = ''
    write_jsonl(six.cohort_dir / 'metadata_titles.jsonl', titles)
    for index in (2, 3):
        (six.summary_arm_dir('graph_qwen') / f"{cohort['catalog'][index]['content_id']}.json").unlink()
    arm = registry(six.config)['graph_qwen_meta']
    docs = documents_for_arm(six, cohort, arm)
    assert [d['components'] for d in docs] == ['both', 'summary_only', 'title_only', 'neither']
    assert docs[0]['text'].startswith('Title\ninside\n\n')
    assert docs[2]['text'] == 'Third title' and docs[3]['text'] == ''
    plain = documents_for_arm(six, cohort, registry(six.config)['graph_qwen'])
    assert docs[0]['text'] == 'Title\ninside\n\n' + plain[0]['text']
    assert plain[2]['text'] == plain[3]['text'] == ''
    result = embed_representations(six)
    assert result['generated_arms'] == NAMES
    assert read_state(six, arm.name)['shareable']
    with np.load(six.representations_dir / f'{arm.name}_embeddings.npz') as data:
        assert data['values'][2].any() and not data['values'][3].any()
    from validation.diagnosis_representations import representation_report
    report, _ = representation_report(six, [arm.name])
    summary = report[arm.name]['sources_summary']
    assert summary['component_counts'] == dict(both=1, summary_only=1, title_only=1, neither=1)
    assert summary['visual_summary_coverage'] == .5
    assert summary['title_fallback_count'] == 1


def test_failed_summary_fallback_and_corruption(six, fake_models):
    generate(six)
    cohort = six.require_ready_cohort()
    arm = registry(six.config)['graph_qwen_meta']
    path = six.summary_arm_dir('graph_qwen') / f"{cohort['catalog'][0]['content_id']}.json"
    doc = read_json(path)
    doc.update(status='failed', text='', word_count=0, violations=['max_tokens'])
    write_json(path, doc)
    row = documents_for_arm(six, cohort, arm)[0]
    assert row['text'] == 'First title' and row['summary_status'] == 'failed'
    path.write_text('{broken')
    with pytest.raises(RuntimeError, match='invalid summary'):
        documents_for_arm(six, cohort, arm)


def test_reject_title_conditioned_summary(six, fake_models):
    generate(six)
    path = next(six.summary_arm_dir('graph_qwen').glob('*.json'))
    doc = read_json(path)
    doc['provenance']['english_title'] = 'Old title'
    write_json(path, doc)
    with pytest.raises(ValueError, match='title present'):
        documents_for_arm(six, six.require_ready_cohort(), registry(six.config)['graph_qwen_meta'])


def test_cache_invalidation_and_run_rename(six, fake_models, monkeypatch):
    generate(six)
    embed_representations(six)
    other = replace(six, run_id='renamed', run_root=six.run_root.parent / 'renamed')
    shutil.copytree(six.run_root / 'extraction', other.run_root / 'extraction')
    def forbidden(*args, **kwargs):
        pytest.fail('shared cache must avoid BGE initialization')
    with monkeypatch.context() as patch:
        patch.setattr('validation.features.BGETextEncoder', forbidden)
        result = embed_representations(other)
    assert result['reuse']['shared'] == NAMES
    titles = read_jsonl(six.cohort_dir / 'metadata_titles.jsonl')
    titles[0]['title'] = 'Changed title'
    write_jsonl(six.cohort_dir / 'metadata_titles.jsonl', titles)
    result = embed_representations(other)
    assert result['generated_arms'] == ['meta', 'graph_qwen_meta', 'desc_qwen_meta', 'graph_gemini_meta']
    path = next(other.summary_arm_dir('graph_qwen').glob('*.json'))
    doc = read_json(path)
    doc['text'] += ' Additional grounded detail.'
    doc['word_count'] = len(doc['text'].split())
    write_json(path, doc)
    result = embed_representations(other)
    assert result['generated_arms'] == ['graph_qwen', 'graph_qwen_meta']


def test_missing_summary_cache_then_available(six, fake_models, monkeypatch):
    result = embed_representations(six, target=['graph_qwen_meta'])
    assert result['generated_arms'] == ['graph_qwen_meta']
    assert read_state(six, 'graph_qwen_meta')['shareable']
    other = replace(six, run_id='missing-copy', run_root=six.run_root.parent / 'missing-copy')
    result = embed_representations(other, target=['graph_qwen_meta'])
    assert result['reuse']['shared'] == ['graph_qwen_meta']
    generate(other)
    assert embed_representations(other, target=['graph_qwen_meta'])['generated_arms'] == ['graph_qwen_meta']


def test_new_signature_separates_old_policy(six, fake_models):
    generate(six)
    cohort = six.require_ready_cohort()
    arm = registry(six.config)['graph_qwen_meta']
    docs = documents_for_arm(six, cohort, arm)
    new = representation_signature(six, cohort['catalog'], arm, docs)
    old_config = {**six.config, 'schema_version': 'viewing-context-config/v6'}
    old_config.pop('experiment_config_version')
    old_ctx = replace(six, config=old_config)
    old = representation_signature(old_ctx, cohort['catalog'], registry(old_ctx.config)['graph_meta_qwen'], docs)
    assert new != old


def test_comparisons_and_partial_targets(six):
    from validation.diagnosis_statistics import comparison_families
    from validation.rolling_diagnosis import comparisons
    families = comparison_families(six.config)
    assert {k: len(v) for k, v in families.items()} == {'metadata_baseline': 5, 'representation': 2, 'title_input': 2}
    settings = SimpleNamespace(familywise_alpha=.05)
    result = comparisons(np.arange(6.), np.tile(np.arange(6.), (100, 1)), settings, arms=NAMES, config=six.config)
    assert result['graph_qwen_meta-meta']['primary']
    assert result['graph_gemini_meta-graph_qwen_meta']['family'] == 'teacher_reference'
    assert not any('interaction' in key for key in result)
    partial = comparisons(np.array([0., 1.]), np.tile([0., 1.], (100, 1)), settings,
                          arms=['meta', 'graph_qwen_meta'], config=six.config)
    assert list(partial) == ['graph_qwen_meta-meta']
    assert partial['graph_qwen_meta-meta']['family_size'] == 5


def test_historical_diagnosis_context(six):
    from arm_registry import ARM_CONTRACT
    from validation.selection import diagnosis_context, cohort_directory
    directory = cohort_directory(six)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'manifest.json', {'arm_contract': ARM_CONTRACT})
    old = diagnosis_context(six)
    assert old.config['schema_version'] == 'viewing-context-config/v6'
    assert 'graph_meta_qwen' in registry(old.config)
    assert six.config['experiment_config_version'] == 'v4'


@pytest.mark.torch
def test_six_arm_training_diagnosis_and_shared_recommendations(six, fake_models, monkeypatch):
    torch = pytest.importorskip('torch')
    from validation.model import SASRec
    from validation.rolling_recommendation import run_rolling
    from validation.rolling_diagnosis import diagnose
    from validation.selection import load_validation_cohort
    generate(six)
    embed_representations(six)
    def tiny(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm=branch, item_features=features).to(device)
    monkeypatch.setattr('validation.rolling_recommendation._new_model', tiny)
    monkeypatch.setattr('validation.rolling_recommendation.worker_devices', lambda *args: ['cpu'])
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_rolling(six)['completed'] == 126
        assert diagnose(six)['status'] == 'pass'
        doc = read_json(six.diagnosis_path)
        assert doc['statistics']['comparisons']['graph_qwen_meta-meta']['primary']
        assert len(doc['statistics']['comparisons']) == 10
        assert len(load_validation_cohort(six)['catalog']) == 4
        assert diagnose(six, target=['meta', 'graph_qwen_meta'])['status'] == 'pass'
        other = replace(six, run_id='shared-recommendations', run_root=six.run_root.parent / 'shared-recommendations')
        shutil.copytree(six.run_root / 'extraction', other.run_root / 'extraction')
        def forbidden(*args, **kwargs):
            pytest.fail('shared reuse must not encode or train')
        monkeypatch.setattr('validation.features.BGETextEncoder', forbidden)
        monkeypatch.setattr('validation.rolling_recommendation.worker_devices', forbidden)
        monkeypatch.setattr('validation.rolling_recommendation.run_combination', forbidden)
        assert embed_representations(other)['reuse']['shared'] == NAMES
        assert run_rolling(other)['skipped'] == 126
        assert read_json(other.recommendations_dir / 'reuse.json')['shared'] == 126
        assert diagnose(other)['status'] == 'pass'
    finally:
        torch.set_num_threads(previous)


def test_cli_source_and_target_contract(six, fake_models, monkeypatch):
    from extraction.cli import main as extract
    from validation.cli import main as validate
    monkeypatch.setattr('extraction.cli.RunContext.load', lambda _: six)
    monkeypatch.setattr('validation.cli.RunContext.load', lambda _: six)
    assert extract(['extract-graph-scenes', '--run-id', 'new', '--model', 'gemini',
                    '--arm', 'graph_gemini', '--schema', 'prompts/scene_graph_v4.md']) == 0
    assert extract(['summarize-graph', '--run-id', 'new', '--model', 'gemini',
                    '--arm', 'graph_gemini', '--schema', 'prompts/summary_graph_v5.md']) == 0
    assert extract(['summarize-graph', '--run-id', 'new', '--model', 'gemini',
                    '--arm', 'graph_gemini_meta', '--schema', 'prompts/summary_graph_v5.md']) == 1
    assert validate(['embed-representations', '--run-id', 'new', '--target', 'graph_gemini']) == 1
    assert validate(['embed-representations', '--run-id', 'new', '--target', 'graph_gemini_meta']) == 0


def test_v7_rejects_title_prompt_and_historical_migration(six, fake_models):
    from extraction.arm_migration import migrate_arm_layout
    with pytest.raises(ValueError, match='title policy'):
        steps.summarize_graph(six, arm='graph_qwen', model='gemini',
                              schema='prompts/summary_graph_v4_meta.md')
    with pytest.raises(ValueError, match='v6'):
        migrate_arm_layout(six, summary_model='gemini')
    assert not fake_models


def test_scene_schema_migration_uses_three_sources(six, fake_models):
    from extraction.scene_storage import migrate_scene_schema
    generate(six)
    result = migrate_scene_schema(six)
    assert result == {'converted': 0, 'unchanged': 12}
