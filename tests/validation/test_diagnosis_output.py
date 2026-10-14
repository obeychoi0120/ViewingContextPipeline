"""Final-score output stays bounded while full evidence and failure checks remain available."""
import json

import pytest

from artifact_io import read_json
from validation.diagnosis_support import write_diagnosis
from validation.recommendation_contracts import EVALUATION_CONTRACT


@pytest.mark.parametrize('mode,aggregation', [('text', None), ('graph', 'mean'), ('graph', 'gnn')])
def test_output_size_is_independent_of_cohort_and_provenance(tmp_path, mode, aggregation):
    scores = {'meta': {'HR@10': 0.5, 'NDCG@10': 0.123456789012345}}
    comparisons = {'graph_qwen3vl_meta-meta': {'difference': 0.02, 'ci_low': 0.01, 'ci_high': 0.03}}
    bulk = [{'event_ids': list(range(13)), 'source_provenance': 'full evidence ' * 200}
            for _ in range(1000)]
    document = {
        'run_id': 'test', 'representation_mode': mode,
        'runtime_decision': {'status': 'pass', 'errors': []},
        'recommendations': {'means': scores, 'per_seed': bulk, 'reuse': bulk},
        'statistics': {'status': 'computed', 'comparisons': comparisons, 'policy': bulk},
        'cohort': bulk, 'selection': bulk, 'representations': bulk,
        'run_comparison': {'reference_run_id': 'reference', 'direction': 'run minus reference',
                           'metric': 'NDCG@10', 'comparisons': comparisons, 'sources': bulk},
    }
    if aggregation:
        document['scene_aggregation'] = aggregation
    path = tmp_path / 'diagnosis.json'
    write_diagnosis(path, document)
    result = read_json(path)
    assert result['schema_version'] == 'leave-one-out-diagnosis/v4'
    assert result['evaluation_contract'] == EVALUATION_CONTRACT
    assert result['recommendations'] == {'means': scores}
    assert result['statistics']['comparisons'] == comparisons
    assert result['statistics']['metric'] == 'NDCG@10'
    assert result['run_comparison'] == {k: document['run_comparison'][k]
                                        for k in ('reference_run_id', 'direction', 'metric', 'comparisons')}
    assert 'cohort' not in result and 'selection' not in result and 'representations' not in result
    assert ('scene_aggregation' in result) == bool(aggregation)
    assert len(json.dumps(document)) > 1_000_000
    before = path.read_bytes()
    assert len(before) < 8192
    document['cohort'] *= 2
    write_diagnosis(path, document)
    assert path.read_bytes() == before


@pytest.mark.parametrize('mode,aggregation', [('text', None), ('graph', 'mean'), ('graph', 'gnn')])
def test_invalid_evidence_still_fails_and_records_reason(
    current_context, monkeypatch, capsys, mode, aggregation,
):
    from validation.steps import run_diagnosis
    from validation.graph_context import graph_context

    def invalid(*args, **kwargs):
        raise ValueError('invalid cohort fingerprint: validation/cohort/manifest.json')

    monkeypatch.setattr('validation.leave_one_out_diagnosis.load_validation_cohort', invalid)
    monkeypatch.setattr('validation.selection.load_validation_cohort', invalid)
    context = current_context if mode == 'text' else graph_context(current_context, aggregation)
    with pytest.raises(RuntimeError, match='diagnosis failed'):
        run_diagnosis(current_context, target=['meta'], representation_mode=mode,
                      scene_aggregation=aggregation)
    output = capsys.readouterr().out
    assert output.count('[STEP] run-diagnosis') == 1
    assert f'  output_dir: {context.diagnosis_path.parent}' in output
    assert '  selected_arms: ["meta"]' in output
    assert f'  representation_mode: {mode}' in output
    assert f'  scene_aggregation: {aggregation or "null"}' in output
    result = read_json(context.diagnosis_path)
    assert result['runtime_decision']['status'] == 'fail'
    assert 'invalid cohort fingerprint' in result['runtime_decision']['errors'][0]['message']
    assert result['statistics'] == {'status': 'not_computed', 'comparison_policy': 'selected-target-all-pairs/v1', 'targets': ['meta'], 'comparison_count': 0, 'correction': 'bonferroni'}
    assert 'recommendations' not in result
