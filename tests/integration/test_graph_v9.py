"""Exercise v9 orchestration with fake backends; no real model quality claims."""
import json
from pathlib import Path
import re

import pytest

from arm_registry import registry
from extraction import steps
from extraction.scene_storage import read_scene_records
from validation.graph_context import graph_context
from validation.graph_inputs import build_input
from validation.selection import prepare_validation_cohort


@pytest.fixture
def graph_response():
    prompt = Path('prompts/scene_graph_v9.md').read_text()
    return re.findall(r'\[Context\]\n.*?(?=\n\s*\n|\Z)', prompt.split('[Examples]')[1], re.S)[0]


@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5', 'gemini'])
def test_v9_generation_storage_summary_and_direct_graph(ready_context, fake_models, model):
    arm = f'graph_{model}'
    steps.extract_graph_scenes(ready_context, model=model, arm=arm, schema='prompts/scene_graph_v9.md')
    path = next(ready_context.scene_arm_dir(arm).glob('*.jsonl'))
    graph = read_scene_records(path)[0]['graph']
    assert graph['actions'][0]['target'] == 'e3'
    assert graph['actions'][0]['place'] == 'e5'
    steps.summarize(ready_context, arm=arm, model='gemini', schema='prompts/summary_graph_v9.md')
    summary = json.loads(next(ready_context.summary_arm_dir(arm).glob('*.json')).read_text())
    assert summary['status'] == 'complete'
    assert '"place": "e5"' in fake_models[-1][0].prompt
    context = graph_context(ready_context, 'mean')
    cohort = prepare_validation_cohort(context)
    texts, arrays, stats = build_input(context, cohort, registry(context.config)[arm + '_meta'])
    assert stats['counts']['valid_scenes'] > 0 and stats['counts']['raw'] == 0
    assert any('Name: sea urchin' in text for text in texts)
    assert [2, 5, 1] in arrays['edges'].tolist()  # sea urchin -> cutting (target)
    assert [4, 5, 4] in arrays['edges'].tolist()  # kitchen -> cutting (place)
    assert arrays['missing'][0].tolist() == [5, -1, -1, -1, 0, -1]
