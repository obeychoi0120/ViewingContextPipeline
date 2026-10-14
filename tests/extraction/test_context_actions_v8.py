"""Receiver-free extraction, compatibility, storage and Summary delivery."""
import json
from copy import deepcopy
import re
from pathlib import Path

import pytest

from extraction.scene_executor import graph_scene_result
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import graph_summary_prompt, parse_or_repair_graph
from extraction.structured_output import normalize_graph_references, validate_graph_structure

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


@pytest.mark.parametrize('line,place', [
    ('p-1 - cross-cutting - c1; k1; l1', 'l1'),
    ('p-1 -> cross-cutting -> c1; k1; unknown', 'unknown'),
    ('p-1 - handing - c1; none; none', None),
])
def test_roles_storage_and_summary(tmp_path, line, place):
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, PREFIX + line)
    assert failure is None and record['semantic_warnings'] == []
    action = record['graph']['actions'][0]
    assert 'receiver' not in action and 'location' not in action
    assert action['place'] == place
    assert action['target'] == 'c1'
    path = tmp_path / 'video.jsonl'
    write_scene_records(path, [record])
    observations = json.loads(graph_summary_prompt('{scenes}', read_scene_records(path)))
    assert observations[0]['observation']['actions'][0] == action


@pytest.mark.parametrize('field', ['actor', 'target', 'tool', 'place'])
def test_undeclared_references_become_unknown(field):
    graph = parse_or_repair_graph(PREFIX + 'p-1 - cutting - c1; k1; l1').graph
    graph['actions'][0][field] = 'undeclared'
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, json.dumps(graph))
    assert failure is None and record['parse_mode'] == 'repaired'
    assert record['graph']['actions'][0][field] == 'unknown'
    assert record['semantic_warnings'] == []
    validate_graph_structure(record['graph'])
    assert graph['actions'][0][field] == 'undeclared'


@pytest.mark.parametrize('text', [
    PREFIX + 'absent - cutting - c1; missing; missing-place',
    (PREFIX + 'absent - cutting - c1; missing; missing-place').replace(' - ', ' -> '),
])
def test_text_reference_repair_preserves_declared_target_without_warning(text):
    parsed = parse_or_repair_graph(text)
    assert parsed.parse_mode == 'repaired' and parsed.warning == ()
    assert parsed.graph['actions'][0] == {
        'actor': 'unknown', 'action': 'cutting', 'target': 'c1',
        'tool': 'unknown', 'place': 'unknown',
    }


@pytest.mark.parametrize('case,tag', [
    ('missing-field', 'MISSING_REQUIRED'),
    ('type', 'PARSE_FAILED'),
])
def test_reference_repair_does_not_bypass_other_constraints(case, tag):
    graph = parse_or_repair_graph(PREFIX + 'p-1 - cutting - c1; k1; l1').graph
    graph['actions'][0]['tool'] = 'absent'
    if case == 'missing-field':
        graph['actions'][0].pop('place')
    else:
        graph['actions'][0]['actor'] = 1
    text = json.dumps(graph)
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, text)
    assert failure is None and record['graph'] == text
    assert tag in record['semantic_warnings']


def test_reference_repair_does_not_change_source_or_valid_absent_roles():
    graph = parse_or_repair_graph(PREFIX + 'p-1 - cutting - c1; none; unknown').graph
    graph['actions'][0]['actor'] = 'absent'
    original = deepcopy(graph)
    repaired, changed = normalize_graph_references(graph)
    assert changed and graph == original
    assert repaired['actions'][0]['actor'] == 'unknown'
    assert repaired['actions'][0]['tool'] is None
    assert repaired['actions'][0]['place'] == 'unknown'


@pytest.mark.parametrize('case', ['duplicates', 'unknown-endpoints', 'none-endpoints', 'action-text'])
def test_readable_semantically_invalid_graphs_have_no_warning(case):
    graph = parse_or_repair_graph(PREFIX + 'p-1 - cutting - c1; k1; l1').graph
    if case == 'duplicates':
        graph['entities'].append({**graph['entities'][0], 'name': 'different person'})
    elif case == 'unknown-endpoints':
        graph['actions'][0].update(actor='absent', target='absent')
    elif case == 'none-endpoints':
        graph['actions'][0].update(actor=None, target=None)
    else:
        graph['actions'][0]['action'] = 'cutting - slicing; moving -> placing'
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, json.dumps(graph))
    assert failure is None and isinstance(record['graph'], dict)
    assert record['semantic_warnings'] == []
    assert len(record['graph']['entities']) == len(graph['entities'])
    assert len(record['graph']['actions']) == 1


def test_duplicate_entity_text_is_kept_without_warning():
    text = PREFIX.replace('p-1: person', 'p-1: person; red shirt\np-1: person; blue shirt')
    parsed = parse_or_repair_graph(text + 'p-1 - cutting - c1; k1; l1')
    assert parsed.graph is not None and parsed.warning == ()
    assert [e['attributes'] for e in parsed.graph['entities'] if e['id'] == 'p-1'] == [
        ['red shirt'], ['blue shirt'],
    ]


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
    template = (root / 'prompts/summary_graph_v9.md').read_text()
    summary = graph_summary_prompt(template, [{'scene_idx': 0, 'graph': graphs[0]}])
    assert '{scenes}' not in summary
    observations = summary.split('Scene observations:\n', 1)[1].split('\n\nOutput only', 1)[0]
    assert json.loads(observations)[0]['observation'] == graphs[0]
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, examples[0],
                                         error='graph: output truncated at token limit')
    assert record is None and failure['raw_response'] == examples[0]


@pytest.mark.parametrize("line", ["p-1 - handing - c1; none", "p-1 - handing - c1; none; p-1; l1"])
def test_unsupported_generated_roles_preserve_raw_response(tmp_path, line):
    text = PREFIX + line
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert failure is None and record["graph"] == text
    assert record["semantic_warnings"] == ["PARSE_FAILED"]
    path = tmp_path / "video.jsonl"
    write_scene_records(path, [record])
    assert read_scene_records(path)[0]["graph"] == text
