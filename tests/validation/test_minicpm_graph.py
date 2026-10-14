"""Graph-only MiniCPM execution includes Metadata and separate GNN diagnoses."""

import numpy as np
import pytest

from artifact_io import read_json
from extraction.steps import extract_graph_scenes
from validation.graph_context import graph_context
from validation.steps import embed_representations, run_recommendation, run_diagnosis
from validation.leave_one_out_diagnosis import comparisons


@pytest.mark.parametrize('size,count', [(1, 0), (2, 1), (3, 3), (8, 28)])
def test_selected_pairs_and_direction(current_context, size, count):
    from arm_registry import registry

    names = list(registry(current_context.config))[:size]
    settings = type('Evaluation', (), {'familywise_alpha': 0.05})()
    values = np.arange(size, dtype=float) + 1
    draws = np.random.default_rng(42).normal(values, 0.1, (1000, size))
    result = comparisons(values, draws, settings, arms=names, config=current_context.config)
    reordered = comparisons(values[::-1], draws[:, ::-1], settings, arms=names[::-1], config=current_context.config)
    assert result == reordered and len(result) == count
    for key, row in result.items():
        left, right = key.split('-')
        assert names.index(left) > names.index(right)
        delta = draws[:, names.index(left)] - draws[:, names.index(right)]
        alpha = settings.familywise_alpha / count
        lo, hi = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
        assert row['ci_low'] == lo and row['ci_high'] == hi
        assert row['family_size'] == count
        assert row['confidence_level'] == pytest.approx(1 - alpha)
        assert 'primary' not in row and row['role'] == 'confirmatory'


@pytest.mark.torch
def test_graph_only_extraction_training_diagnosis(ready_context, fake_models, monkeypatch):
    torch = pytest.importorskip('torch')
    from validation import leave_one_out_recommendation as runner

    context = ready_context
    context.config['extraction']['minicpm']['vision_downsample'] = 16
    monkeypatch.setattr(runner, 'worker_devices', lambda *a: ['cpu'])
    monkeypatch.setattr('validation.graph_inputs.encode_texts',
                        lambda settings, texts: np.ones((len(texts), settings.embedding_dim), dtype=np.float32))
    targets = ['meta', 'graph_minicpm', 'graph_minicpm_meta']
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        extract_graph_scenes(context, model='minicpm', arm='graph_minicpm', schema='prompts/scene_graph_v8.md')
        assert not context.summary_arm_dir('graph_minicpm').exists()
        embed_representations(context, target=targets, representation_mode='graph')
        for pooling in ('gnn', 'gnn_attn'):
            assert run_recommendation(context, target=targets, representation_mode='graph', scene_aggregation=pooling)['completed'] == 9
            assert run_diagnosis(context, target=targets, representation_mode='graph', scene_aggregation=pooling)['status'] == 'pass'
            path = graph_context(context, pooling).diagnosis_path
            diagnosis = read_json(path)
            assert diagnosis['schema_version'] == 'leave-one-out-diagnosis/v4'
            assert diagnosis['scene_aggregation'] == pooling
            assert diagnosis['statistics']['targets'] == targets
            assert diagnosis['statistics']['comparison_count'] == 3
            assert len(diagnosis['statistics']['comparisons']) == 3
            assert all({'NDCG@10', 'HR@10'} <= row.keys() for row in diagnosis['recommendations']['means'].values())
            assert run_recommendation(context, target=targets, representation_mode='graph', scene_aggregation=pooling)['skipped'] == 9
        assert graph_context(context, 'gnn').diagnosis_path != graph_context(context, 'gnn_attn').diagnosis_path
        assert not context.summary_arm_dir('graph_minicpm').exists()
        # A generation change also invalidates already prepared Graph/training inputs.
        context.config['extraction']['minicpm']['vision_downsample'] = 4
        with pytest.raises(RuntimeError, match='--force'):
            run_recommendation(context, target=targets, representation_mode='graph', scene_aggregation='gnn')
        extract_graph_scenes(context, model='minicpm', arm='graph_minicpm', schema='prompts/scene_graph_v8.md', force=True)
        prepared = embed_representations(context, target=targets, representation_mode='graph')
        assert prepared['generated_arms'] == targets[1:]
        assert run_recommendation(context, target=targets, representation_mode='graph', scene_aggregation='gnn')['completed'] == 6
    finally:
        torch.set_num_threads(previous)
