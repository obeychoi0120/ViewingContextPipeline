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
            semantic_warnings=["PARSE_ERROR"],
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
