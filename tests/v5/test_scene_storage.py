import pytest

from extraction.recovery import fingerprint
from extraction.scene_storage import metadata_path, migrate_scene_schema, read_scene_records
from extraction.step_support import write_scene_results
from pipeline_runtime import read_jsonl, write_jsonl


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
def test_scenes_have_explicit_indices_without_metadata(tmp_path, kind):
    path = tmp_path / "scenes/video.jsonl"
    records = [scene(kind, 3), scene(kind, 0)]
    write_scene_results(path, records)
    payload = read_jsonl(path)
    field = "description" if kind == "description" else "scene_graph"
    assert all(set(row) == {"content_id", "scene_idx", field, "provenance"} for row in payload)
    assert [row["scene_idx"] for row in payload] == [0, 3]
    assert not metadata_path(path).parent.exists()
    restored = read_scene_records(path)
    assert [row["scene_idx"] for row in restored] == [0, 3]
    body = "raw_response" if kind == "raw" else kind
    assert restored[0][body] == records[0][body]
    assert all("provenance" in row and "generation" not in row for row in restored)


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
    assert not metadata_path(path).parent.exists()


def legacy_file(path):
    records = [scene("graph", 2), scene("graph", 7)]
    payload = [{"content_id": "video", "scene_graph": row["graph"]} for row in records]
    write_jsonl(path, payload)
    from artifact_io import atomic_write_json
    atomic_write_json(metadata_path(path), {
        "schema_version": "scene-metadata/v1", "content_id": "video",
        "payload_hash": fingerprint(payload),
        "rows": [{"body_field": "graph", "record": {k: v for k, v in row.items() if k != "graph"}}
                 for row in records],
    }, durable=True)


@pytest.mark.parametrize("damage", ["missing_metadata", "changed_payload"])
def test_legacy_missing_identity_is_not_silently_guessed(tmp_path, damage):
    path = tmp_path / "scenes/video.jsonl"
    legacy_file(path)
    if damage == "missing_metadata":
        metadata_path(path).unlink()
    else:
        payload = read_jsonl(path)
        payload[0]["scene_graph"]["context"] = ["changed"]
        write_jsonl(path, payload)
    with pytest.raises(ValueError, match="metadata"):
        read_scene_records(path)


def test_legacy_migration_deletes_metadata_only_after_successful_write(tmp_path, monkeypatch):
    import extraction.scene_storage as storage
    path = tmp_path / "scenes/video.jsonl"
    legacy_file(path)
    records = read_scene_records(path)
    assert [row["scene_idx"] for row in records] == [2, 7]
    assert all(row['provenance'] == scene('graph')['provenance'] for row in records)
    before = path.read_bytes()
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("disk full")
        patch.setattr(storage, "atomic_write_jsonl", fail)
        with pytest.raises(OSError):
            storage.migrate_scene_file(path, records)
    assert path.read_bytes() == before and metadata_path(path).exists()
    assert storage.migrate_scene_file(path, records)
    assert not metadata_path(path).parent.exists()
    assert read_scene_records(path) == records


def test_migration_preserves_existing_summary_and_embedding_reuse(ready_context, fake_models):
    from benchmarks.qwen_benchmark import export_requests
    from extraction.steps import (
        extract_graph_scenes, extract_description_scenes, summarize_graph, summarize_description,
    )
    from validation.steps import embed_representations

    context = ready_context
    records_by_path = {}
    for model in ("qwen", "gemini"):
        extract_graph_scenes(context, model=model, schema="prompts/scene_graph_v3.md")
        extract_description_scenes(context, model=model, schema="prompts/scene_description_v2.md")
        for representation in ("graph", "description"):
            for path in context.extraction_dir(representation, model, "scenes").glob("*.jsonl"):
                records = read_scene_records(path)
                records_by_path[path] = records
                # Simulate outputs created before the storage format changed.
                write_jsonl(path, records)
                metadata_path(path).unlink(missing_ok=True)
        summarize_graph(context, model="qwen", source=model, schema="prompts/summary_graph_v4.md")
        summarize_description(context, model="qwen", source=model, schema="prompts/summary_description_v4.md")
    arms = ["graph_qwen", "graph_gemini", "desc_qwen", "desc_gemini"]
    embed_representations(context, summary_source="qwen", target=arms)
    summaries = {path: path.read_bytes() for path in context.run_root.glob(
        "extraction/*/*/summaries/*.json")}
    benchmark_schemas = {
        "description-summary": "prompts/summary_description_v4.md",
        "graph-summary-qwen": "prompts/summary_graph_v4.md",
        "graph-summary-gemini": "prompts/summary_graph_v4.md",
    }
    requests = {stage: export_requests(context, stage, 2, schema)
                for stage, schema in benchmark_schemas.items()}
    calls = len(fake_models)
    assert migrate_scene_schema(context) == {"converted": 16, "unchanged": 0}
    assert migrate_scene_schema(context) == {"converted": 0, "unchanged": 16}
    for path, records in records_by_path.items():
        assert read_scene_records(path) == records
        assert len(read_jsonl(path)[0]) == 4
    for model in ("qwen", "gemini"):
        extract_graph_scenes(context, model=model, schema="prompts/scene_graph_v3.md")
        extract_description_scenes(context, model=model, schema="prompts/scene_description_v2.md")
        summarize_graph(context, model="qwen", source=model, schema="prompts/summary_graph_v4.md")
        summarize_description(context, model="qwen", source=model, schema="prompts/summary_description_v4.md")
    assert len(fake_models) == calls
    assert all(path.read_bytes() == saved for path, saved in summaries.items())
    assert embed_representations(context, summary_source="qwen", target=arms)["generated_arms"] == []
    assert all(export_requests(context, stage, 2, benchmark_schemas[stage]) == saved
               for stage, saved in requests.items())


@pytest.mark.parametrize("model", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["graph", "description"])
def test_changed_settings_retry_only_explicit_scene_and_summary_failures(
    ready_context, fake_models, model, representation,
):
    import extraction.steps as steps
    from extraction.failures import FailureLog
    from validation.steps import embed_representations

    context = ready_context
    extract = getattr(steps, f"extract_{representation}_scenes")
    summarize = getattr(steps, f"summarize_{representation}")
    scene_schema = f"prompts/scene_{representation}_v{'3' if representation == 'graph' else '2'}.md"
    summary_schema = f"prompts/summary_{representation}_v4.md"
    extract(context, model=model, schema=scene_schema)
    summarize(context, model="qwen", source=model, schema=summary_schema)
    directory = context.extraction_dir(representation, model, "scenes")
    summary_dir = context.summary_dir(representation, model, "qwen")
    scene_paths = sorted(directory.glob("*.jsonl"))
    selected = scene_paths[0].stem
    before = {path: path.read_bytes() for path in scene_paths[1:]}
    summaries = {path: path.read_bytes() for path in summary_dir.glob("*.json")}
    failures = FailureLog(directory, scenes=True)
    failures.record(selected, 0, "previous failure", "")
    context.config["extraction"][representation]["scene_max_new_tokens"] = 768
    context.config["extraction"][representation]["summary_max_new_tokens"] = 768
    for schema in (scene_schema, summary_schema):
        path = context.prompt_path(schema)
        path.write_text(path.read_text() + "\nChanged prompt instructions.\n")
    count = len(fake_models)
    assert extract(context, model=model, schema=scene_schema)["failure_count"] == 0
    assert len(fake_models) == count + 1
    assert [task.task_id for task in fake_models[-1]] == [f"{selected}:0"]
    assert all(path.read_bytes() == saved for path, saved in before.items())
    assert not (directory / ".metadata").exists()
    assert not failures.path_for(selected).exists()
    # Even newly generated scene inputs do not replace a successful summary.
    summarize(context, model="qwen", source=model, schema=summary_schema)
    assert len(fake_models) == count + 1
    assert all(path.read_bytes() == saved for path, saved in summaries.items())
    summary_failures = FailureLog(summary_dir)
    summary_failures.record(selected, None, "previous failure", "", repetition_penalty=1.0)
    summarize(context, model="qwen", source=model, schema=summary_schema)
    assert len(fake_models) == count + 2
    assert [task.task_id for task in fake_models[-1]] == [selected]
    assert not summary_failures.path.exists()
    assert all(path.read_bytes() == saved for path, saved in summaries.items() if path.stem != selected)
    arm = f"{'desc' if representation == 'description' else 'graph'}_{model}"
    context.config["protocol"]["arms"] = [arm]
    assert embed_representations(context, summary_source="qwen", target=[arm])["generated_arms"] == [arm]
    extract(context, model=model, schema=scene_schema, force=True)
    assert len(fake_models[-1]) == len(scene_paths)
