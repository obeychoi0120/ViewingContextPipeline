"""V9 names resolve exactly before storage, independently of legacy IDs."""
import json
from pathlib import Path
import re

import pytest

from extraction.scene_executor import graph_scene_result
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import graph_summary_prompt, parse_or_repair_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = re.findall(
    r'\[Context\]\n.*?(?=\n\s*\n|\Z)',
    (ROOT / 'prompts/scene_graph_v9.md').read_text().split('[Examples]', 1)[1], re.S,
)


@pytest.mark.parametrize('index', range(3))
def test_v9_examples_roundtrip_to_summary(tmp_path, index):
    parsed = parse_or_repair_graph(EXAMPLES[index])
    assert parsed.parse_mode == 'native' and parsed.warning == ()
    graph = parsed.graph
    assert [e['id'] for e in graph['entities']] == [f'e{i}' for i in range(1, len(graph['entities']) + 1)]
    if index == 0:
        assert graph['entities'][0] == {'id': 'e1', 'name': 'person1', 'attributes': ['wearing apron', 'wearing gloves']}
        assert graph['actions'] == [
            {'actor': 'e1', 'action': 'cutting', 'target': 'e3', 'tool': 'e4', 'place': 'e5'},
            {'actor': 'e2', 'action': 'watching', 'target': 'e1', 'tool': None, 'place': 'e5'},
        ]
    elif index == 1:
        assert graph['actions'][0]['place'] == 'unknown'
    else:
        assert graph['actions'] == []
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, EXAMPLES[index])
    assert failure is None
    path = tmp_path / 'video.jsonl'
    write_scene_records(path, [record])
    template = (ROOT / 'prompts/summary_graph_v9.md').read_text()
    summary = graph_summary_prompt(template, read_scene_records(path))
    observations = summary.split('Scene observations:\n')[1].split('\n\nOutput only')[0]
    assert json.loads(observations)[0]['observation'] == graph
    assert parse_or_repair_graph(json.dumps(graph)).graph == graph
    raw_path = tmp_path / 'raw.jsonl'
    write_scene_records(raw_path, [{'scene_idx': 0, 'graph': EXAMPLES[index]}])
    original = raw_path.read_bytes()
    assert read_scene_records(raw_path)[0]['graph'] == graph
    assert raw_path.read_bytes() == original


@pytest.mark.parametrize('name', ['e3', 'Sea Urchin'])
def test_names_are_exact_and_cannot_alias_generated_ids(name):
    text = EXAMPLES[0].replace(' - sea urchin;', f' - {name};')
    parsed = parse_or_repair_graph(text)
    assert parsed.parse_mode == 'repaired' and parsed.warning == ()
    assert parsed.graph['actions'][0]['target'] == 'unknown'
    assert parsed.graph['actions'][0]['actor'] == 'e1'


@pytest.mark.parametrize('transform', [
    lambda s: s.replace(';', '；'),
    lambda s: s.replace(' - ', ' → '),
])
def test_v9_conservative_delimiter_repairs(transform):
    parsed = parse_or_repair_graph(transform(EXAMPLES[0]))
    assert parsed.parse_mode == 'repaired'
    assert parsed.graph == parse_or_repair_graph(EXAMPLES[0]).graph


@pytest.mark.parametrize('transform', [
    lambda s: s.replace('person1; wearing apron', 'p1: person; wearing apron'),
    lambda s: s.replace('person1; wearing apron', 'unknown; wearing apron'),
    lambda s: s.replace('; knife; kitchen', '; ; kitchen'),
    lambda s: s.replace('; knife; kitchen', '; knife'),
    lambda s: s.replace(' - cutting - ', ' - cutting - slicing - '),
])
def test_ambiguous_or_incomplete_v9_keeps_raw(transform):
    text = transform(EXAMPLES[0])
    record, failure = graph_scene_result({'scene_idx': 0, 'keyframes': []}, text)
    assert failure is None and record['graph'] == text and record['semantic_warnings']


def test_duplicate_names_keep_all_entities_and_last_reference_policy():
    text = EXAMPLES[0].replace('person2', 'person1')
    graph = parse_or_repair_graph(text).graph
    assert len(graph['entities']) == 5
    assert graph['actions'][0]['actor'] == 'e2'


def test_empty_entities_and_actions():
    prefix = EXAMPLES[0].split('[Entities]')[0]
    graph = parse_or_repair_graph(prefix + '[Entities]\nnone\n[Actions]\nnone').graph
    assert graph['entities'] == [] and graph['actions'] == []
