import json

import pytest

from extraction.scene_storage import read_scene_records
from extraction.step_support import write_scene_results
from artifact_io import read_jsonl
GRAPH = {"medium": "live_action", "format": "other", "topics": [], "entities": [], "actions": []}


def scene(kind, index=0):
    row = {
        "scene_idx": index,
        "keyframes": [index * 30 + 5],
        "provenance": {"arm": f"{kind}_qwen3vl"},
        "generation": {"attempt_count": 2, "input_key": "key", "repair_mode": "native"},
    }
    if kind == "description":
        row.update(
            schema_version="scene-description/v2",
            content_id="video",
            description="A person walks outdoors.",
        )
    elif kind == "graph":
        row.update(
            graph=GRAPH,
            parse_mode="native",
            semantic_warnings=[],
        )
    else:
        row.update(
            graph='{"entities": [',
            parse_mode="text",
            semantic_warnings=["PARSE_FAILED"],
        )
    return row


@pytest.mark.parametrize("kind", ["description", "graph", "raw"])
def test_scenes_have_explicit_indices_without_metadata(tmp_path, kind):
    path = tmp_path / "scenes/video.jsonl"
    records = [scene(kind, 3), scene(kind, 0)]
    write_scene_results(path, records)
    payload = read_jsonl(path)
    field = "description" if kind == "description" else "scene_graph"
    expected = {"content_id", "scene_idx", "tokens", field}
    if kind != "description":
        expected.add("warning")
    assert all(set(row) == expected for row in payload)
    assert [row["scene_idx"] for row in payload] == [0, 3]
    assert not (path.parent / ".metadata").exists()
    restored = read_scene_records(path)
    assert [row["scene_idx"] for row in restored] == [0, 3]
    body = "graph" if kind == "raw" else kind
    assert restored[0][body] == records[0][body]
    assert all("provenance" not in row
               and "generation" not in row for row in restored)


def test_interrupted_publication_preserves_readable_previous_results(tmp_path, monkeypatch):
    import extraction.scene_storage as storage

    path = tmp_path / "scenes/video.jsonl"
    write_scene_results(path, [scene("graph", 3)])
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_write_jsonl", fail)
        with pytest.raises(OSError, match="interrupted"):
            write_scene_results(path, [scene("graph", 3), scene("raw", 1)])
    assert path.read_bytes() == before
    assert read_scene_records(path)[0]["scene_idx"] == 3
    assert not (path.parent / ".metadata").exists()


@pytest.mark.parametrize('raw', [False, True])
def test_existing_reference_warning_is_recovered_in_memory_and_delivered_to_summary(tmp_path, raw):
    from extraction.semantic_graph import graph_summary_prompt

    graph = {
        **GRAPH,
        'entities': [{'id': 'p', 'name': 'person', 'attributes': []}],
        'actions': [{'actor': 'p', 'action': 'holding', 'target': 'missing',
                     'tool': None, 'place': 'unknown'}],
    }
    path = tmp_path / 'video.jsonl'
    path.write_text(json.dumps({
        'content_id': 'video', 'scene_idx': 0, 'tokens': 12,
        'warning': ['INVALID_REFERENCE'],
        'scene_graph': json.dumps(graph) if raw else graph,
    }) + '\n')
    original = path.read_bytes()
    records = read_scene_records(path)
    restored = records[0]
    assert restored['parse_mode'] == 'repaired'
    assert restored['tokens'] == 12 and restored['semantic_warnings'] == []
    assert restored['graph']['actions'][0]['target'] == 'unknown'
    assert json.loads(graph_summary_prompt('{scenes}', records))[0]['observation'] == restored['graph']
    assert path.read_bytes() == original


@pytest.mark.parametrize('legacy,current', [
    (['PARSE_ERROR'], ['PARSE_FAILED']),
    (['INVALID_REFERENCE', 'DUPLICATE_ENTITY_ID', 'INVALID_ACTION_SYNTAX'], []),
    (['MISSING_REQUIRED', 'PARSE_ERROR', 'PARSE_FAILED'], ['MISSING_REQUIRED', 'PARSE_FAILED']),
])
def test_only_two_warning_tags_remain_and_legacy_tags_are_normalized(legacy, current):
    from extraction.graph_warnings import WARNING_TAGS, warning_tags

    assert WARNING_TAGS == ('MISSING_REQUIRED', 'PARSE_FAILED')
    assert warning_tags(legacy) == current
    with pytest.raises(ValueError):
        warning_tags(['UNRECOGNIZED'])


@pytest.mark.parametrize('interruption', ['publish', 'cleanup'])
def test_legacy_failures_migrate_atomically_with_precedence_and_resume(
    tmp_path, monkeypatch, interruption,
):
    from pathlib import Path
    import extraction.scene_storage as storage
    from artifact_io import write_jsonl

    path = tmp_path / 'video.jsonl'
    write_scene_results(path, [scene('graph', 0), scene('graph', 3)])
    before = path.read_bytes()
    good = read_jsonl(path)[1]
    legacy = tmp_path / 'failures/video.jsonl'
    write_jsonl(legacy, [
        {'content_id': 'video', 'scene_idx': 0, 'error': 'old failure', 'raw_output': 'old'},
        {'content_id': 'video', 'scene_idx': 0, 'error': 'latest failure', 'raw_output': 'last',
         'tokens': 17, 'provenance': {'old': 'generation'}},
        {'content_id': 'video', 'scene_idx': 1, 'error': 'empty', 'raw_output': ''},
    ])
    original_unlink = Path.unlink

    def fault(*args, **kwargs):
        raise OSError('interrupted migration')

    def fail_cleanup(target, *args, **kwargs):
        if target == legacy:
            fault()
        return original_unlink(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        if interruption == 'publish':
            patch.setattr(storage, 'atomic_write_jsonl', fault)
        else:
            patch.setattr(Path, 'unlink', fail_cleanup)
        with pytest.raises(OSError, match='migration'):
            storage.migrate_scene_failures(tmp_path)
    assert legacy.exists()
    if interruption == 'publish':
        assert path.read_bytes() == before
    else:
        assert read_jsonl(path)[0]['error'] == 'latest failure'
    storage.migrate_scene_failures(tmp_path)
    storage.migrate_scene_failures(tmp_path)  # A completed migration is a no-op.
    assert not legacy.parent.exists()
    assert read_jsonl(path) == [
        {'content_id': 'video', 'scene_idx': 0, 'tokens': 17,
         'error': 'latest failure', 'raw_output': 'last'},
        {'content_id': 'video', 'scene_idx': 1, 'tokens': None,
         'error': 'empty', 'raw_output': ''},
        good,
    ]
    assert [row['scene_idx'] for row in read_scene_records(path)] == [3]
    assert len(read_scene_records(path, include_failures=True)) == 3


def test_failure_rows_validate_even_when_excluded_from_observations(tmp_path):
    from artifact_io import write_jsonl

    path = tmp_path / 'video.jsonl'
    write_jsonl(path, [{'content_id': 'video', 'scene_idx': 0, 'tokens': -1,
                       'error': 'failure', 'raw_output': ''}])
    with pytest.raises(ValueError, match='tokens'):
        read_scene_records(path)


def test_persisted_warning_is_retried_even_when_current_parser_repairs_it(tmp_path):
    from artifact_io import write_jsonl
    from extraction.steps import prepare_scene_batch
    from extraction.step_support import minimal_graph_records

    path = tmp_path / 'video.jsonl'
    write_jsonl(path, [{
        'content_id': 'video', 'scene_idx': 0, 'tokens': 12,
        'warning': ['PARSE_FAILED'], 'scene_graph': json.dumps(GRAPH),
    }])
    before = path.read_bytes()
    assert read_scene_records(path)[0]['semantic_warnings'] == []
    records, pending = prepare_scene_batch(
        [{'content_id': 'video'}], tmp_path,
        lambda _: [{'scene_idx': 0, 'keyframes': []}], minimal_graph_records, force=False,
    )
    assert records['video'][0]['semantic_warnings'] == ['PARSE_FAILED']
    assert records['video'][0]['graph'] == json.dumps(GRAPH)
    assert pending[0][1][0]['scene_idx'] == 0
    assert path.read_bytes() == before
