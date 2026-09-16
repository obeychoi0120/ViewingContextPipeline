from copy import deepcopy

import pytest

from extraction.recovery import fingerprint
from extraction.scene_storage import metadata_path, migrate_scene_schema, read_scene_records
from extraction.step_support import write_scene_checkpoint
from pipeline_runtime import read_json, read_jsonl, write_jsonl


def scene(kind, index=0):
    row = {"scene_idx": index, "keyframes": [index * 30 + 5],
           "provenance": {"arm": f"{kind}_qwen"},
           "generation": {"attempt_count": 2, "input_key": "key", "repair_mode": "native"}}
    if kind == "description":
        row.update(schema_version="scene-description/v2", content_id="video",
                   description="A person walks outdoors.")
    elif kind == "graph":
        row.update(graph={"entities": [], "relations": [], "context": []},
                   parse_mode="native", semantic_warnings=[])
    else:
        row.update(schema_version="graph-scene-raw/v1", status="raw_fallback",
                   raw_response='{"entities": [')
    return row


@pytest.mark.parametrize("kind", ["description", "graph", "raw"])
def test_public_scenes_have_only_content_id_and_body_with_lossless_metadata(tmp_path, kind):
    path = tmp_path / "scenes/video.jsonl"
    records = [scene(kind, 0), scene(kind, 3)]
    original = deepcopy(records)
    write_scene_checkpoint(path, path.parent / "failures/video.jsonl", records, [])
    payload = read_jsonl(path)
    field = "description" if kind == "description" else "scene_graph"
    assert all(set(row) == {"content_id", field} and row["content_id"] == "video"
               for row in payload)
    if kind == "raw":
        assert payload[0][field] == records[0]["raw_response"]
    assert read_scene_records(path) == original
    assert fingerprint(read_scene_records(path)) == fingerprint(original)
    saved = read_json(metadata_path(path))
    assert saved["payload_hash"] == fingerprint(payload)
    assert all(not ({"description", "graph", "raw_response"} & set(row["record"]))
               for row in saved["rows"])


def test_scene_checkpoint_recovers_interrupted_payload_and_metadata_write(tmp_path, monkeypatch):
    import extraction.scene_storage as storage

    path = tmp_path / "scenes/video.jsonl"
    failure = path.parent / "failures/video.jsonl"
    write_scene_checkpoint(path, failure, [scene("graph")], [])
    before = path.read_bytes()
    records = [scene("graph"), scene("raw", 1)]

    def fail(*args, **kwargs):
        raise OSError("interrupted after metadata write")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_write_jsonl", fail)
        with pytest.raises(OSError, match="interrupted"):
            write_scene_checkpoint(path, failure, records, [])
    assert path.read_bytes() == before
    assert read_scene_records(path) == records
    assert not (path.parent / ".checkpoints").exists()


@pytest.mark.parametrize("damage", ["missing_metadata", "changed_payload"])
def test_compact_scene_cannot_silently_use_missing_or_stale_metadata(tmp_path, damage):
    path = tmp_path / "scenes/video.jsonl"
    write_scene_checkpoint(path, path.parent / "failures/video.jsonl", [scene("description")], [])
    if damage == "missing_metadata":
        metadata_path(path).unlink()
    else:
        payload = read_jsonl(path)
        payload[0]["description"] = "Different observation."
        write_jsonl(path, payload)
    with pytest.raises(ValueError, match="metadata"):
        read_scene_records(path)


def test_migration_preserves_existing_summary_and_embedding_reuse(ready_context, fake_models):
    from benchmarks.qwen_benchmark import export_requests
    from extraction.steps import (
        extract_graph_scenes, extract_description_scenes, summarize_graph, summarize_description,
    )
    from validation.steps import embed_representations

    context = ready_context
    records_by_path = {}
    for model in ("qwen", "gemini"):
        extract_graph_scenes(context, model=model, schema="prompts/graph_scene_v3.md")
        extract_description_scenes(context, model=model, schema="prompts/description_scene_v2.md")
        for representation in ("graph", "description"):
            for path in context.extraction_dir(representation, model, "scenes").glob("*.jsonl"):
                records = read_scene_records(path)
                records_by_path[path] = records
                # Simulate outputs created before the storage format changed.
                write_jsonl(path, records)
                metadata_path(path).unlink()
        summarize_graph(context, source=model, schema="prompts/graph_summary_v4.md")
        summarize_description(context, source=model, schema="prompts/description_summary_v4.md")
    arms = ["graph_qwen", "graph_gemini", "desc_qwen", "desc_gemini"]
    embed_representations(context, target=arms)
    summaries = {path: path.read_bytes() for path in context.run_root.glob(
        "extraction/*/*/summaries/*.json")}
    benchmark_schemas = {
        "description-summary": "prompts/description_summary_v4.md",
        "graph-summary-qwen": "prompts/graph_summary_v4.md",
        "graph-summary-gemini": "prompts/graph_summary_v4.md",
    }
    requests = {stage: export_requests(context, stage, 2, schema)
                for stage, schema in benchmark_schemas.items()}
    calls = len(fake_models)
    assert migrate_scene_schema(context) == {"converted": 16, "unchanged": 0}
    assert migrate_scene_schema(context) == {"converted": 0, "unchanged": 16}
    for path, records in records_by_path.items():
        assert read_scene_records(path) == records
        assert len(read_jsonl(path)[0]) == 2
    for model in ("qwen", "gemini"):
        extract_graph_scenes(context, model=model, schema="prompts/graph_scene_v3.md")
        extract_description_scenes(context, model=model, schema="prompts/description_scene_v2.md")
        summarize_graph(context, source=model, schema="prompts/graph_summary_v4.md")
        summarize_description(context, source=model, schema="prompts/description_summary_v4.md")
    assert len(fake_models) == calls
    assert all(path.read_bytes() == saved for path, saved in summaries.items())
    assert embed_representations(context, target=arms)["generated_arms"] == []
    assert all(export_requests(context, stage, 2, benchmark_schemas[stage]) == saved
               for stage, saved in requests.items())
