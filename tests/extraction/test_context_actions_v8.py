"""Receiver-free extraction, compatibility, storage and Summary delivery."""
import copy
import json
import re
from pathlib import Path

import pytest

from extraction.scene_executor import graph_scene_result
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import graph_summary_prompt, parse_or_repair_graph
from extraction.structured_output import validate_graph_structure

PREFIX = '''[Context]
medium: live_action
format: demonstration
topics: food preparation
[Entities]
p-1: person
c1: carrot
k1: knife
l1: kitchen
[Actions]
'''


@pytest.mark.parametrize('line,location', [
    ('p-1 - cross-cutting - c1; k1; l1', 'l1'),
    ('p-1 -> cross-cutting -> c1; k1; unknown', 'unknown'),
    ('p-1 - handing - c1; none; none', None),
    ('p-1 - handing - c1; none', 'unknown'),
    ('p-1 - handing - c1; none; undeclared receiver; l1', 'l1'),
])
def test_roles_storage_and_summary(tmp_path, line, location):
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, PREFIX + line)
    assert failure is None and record['semantic_warnings'] == []
    action = record['graph']['actions'][0]
    assert 'receiver' not in action
    assert action['location'] == location
    assert action['target'] == 'c1'
    path = tmp_path / 'video.jsonl'
    write_scene_records(path, [record])
    observations = json.loads(graph_summary_prompt('{scenes}', read_scene_records(path)))
    assert observations[0]['observation']['actions'][0] == action


@pytest.mark.parametrize('receiver', ['missing', '', ['p-1'], {'id': 'p-1'}, 12, None])
def test_legacy_receiver_validation_disabled_without_mutation(receiver):
    graph = parse_or_repair_graph(PREFIX + 'p-1 - handing - c1; none; unknown').graph
    graph['actions'][0]['receiver'] = receiver
    before = copy.deepcopy(graph)
    validate_graph_structure(graph)
    assert graph == before
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, json.dumps(graph))
    assert failure is None and record['semantic_warnings'] == []
    assert 'receiver' not in record['graph']['actions'][0]
    # Previously saved structured records also omit receiver from Summary inputs.
    observation = json.loads(graph_summary_prompt('{scenes}', [{'scene_idx': 0, 'graph': graph}]))
    assert 'receiver' not in observation[0]['observation']['actions'][0]
    assert graph == before


@pytest.mark.parametrize('field', ['actor', 'target', 'tool', 'location'])
def test_remaining_reference_checks_still_active(field):
    graph = parse_or_repair_graph(PREFIX + 'p-1 - cutting - c1; k1; l1').graph
    graph['actions'][0][field] = 'undeclared'
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, json.dumps(graph))
    assert failure is None and isinstance(record['graph'], str)
    assert 'INVALID_REFERENCE' in record['semantic_warnings']


def test_prompt_examples_and_truncation():
    root = Path(__file__).resolve().parents[2]
    prompt = (root / 'prompts/scene_graph_v8.md').read_text()
    assert 'receiver' not in prompt and '[End]' not in prompt
    examples = re.findall(r'\[Context\]\n.*?(?=\n\s*\n|\Z)', prompt.split('[Examples]', 1)[1], re.S)
    assert len(examples) == 3
    graphs = []
    for example in examples:
        parsed = parse_or_repair_graph(example)
        assert parsed.parse_mode == 'native'
        validate_graph_structure(parsed.graph)
        graphs.append(parsed.graph)
    assert graphs[1]['actions'][0]['target'] == 'f1'
    assert graphs[2]['actions'] == []
    template = (root / 'prompts/summary_graph_v8.md').read_text()
    summary = graph_summary_prompt(template, [{'scene_idx': 0, 'graph': graphs[0]}])
    assert '{scenes}' not in summary and 'Ignore any legacy receiver field' in summary
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, examples[0],
                                         error='graph: output truncated at token limit')
    assert record is None and failure['raw_response'] == examples[0]
