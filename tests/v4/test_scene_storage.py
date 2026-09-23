import pytest

from extraction.scene_storage import metadata_path, read_scene_records
from extraction.step_support import write_scene_results
from pipeline_runtime import read_jsonl


def scene(kind, index=0):
    row = {
        "scene_idx": index,
        "keyframes": [index * 30 + 5],
        "provenance": {"arm": f"{kind}_qwen"},
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
            graph={"entities": [], "relations": [], "context": []},
            parse_mode="native",
            semantic_warnings=[],
        )
    else:
        row.update(
            schema_version="graph-scene-raw/v1",
            status="raw_fallback",
            raw_response='{"entities": [',
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
    assert not metadata_path(path).parent.exists()
    restored = read_scene_records(path)
    assert [row["scene_idx"] for row in restored] == [0, 3]
    body = "raw_response" if kind == "raw" else kind
    assert restored[0][body] == records[0][body]
    assert all("provenance" not in row
               and "generation" not in row for row in restored)


def test_compact_text_graph_remains_summary_input_without_format_marker(tmp_path):
    path = tmp_path / "video.jsonl"
    record = scene("graph")
    record.update(graph="unparsed text", parse_mode="text")
    write_scene_results(path, [record])
    assert read_jsonl(path) == [{"content_id": "video", "warning": ["PARSE_ERROR"], "scene_idx": 0, "tokens": None,
                               "scene_graph": "unparsed text"}]
    restored = read_scene_records(path)[0]
    assert restored["graph"] == "unparsed text" and restored["parse_mode"] == "text"
    assert "status" not in restored


@pytest.mark.parametrize("kind", ["structured", "text", "failed_text"])
def test_legacy_graph_read_and_explicit_migration(tmp_path, kind):
    from extraction.scene_storage import migrate_scene_file
    from pipeline_runtime import write_jsonl

    path = tmp_path / "video.jsonl"
    value = {"entities": [], "relations": []} if kind == "structured" else "old raw text"
    row = {"content_id": "video", "scene_idx": 7, "scene_graph": value,
           "provenance": {"prompt_hash": "legacy"}}
    if kind == "text":
        row["graph_format"] = "text"
    write_jsonl(path, [row])
    restored = read_scene_records(path)
    assert restored[0]["provenance"] == row["provenance"]
    assert migrate_scene_file(path, restored)
    assert read_jsonl(path) == [{"content_id": "video", "warning": [] if kind == "structured" else ["PARSE_ERROR"],
                                "scene_idx": 7, "tokens": None, "scene_graph": value}]
    reread = read_scene_records(path)[0]
    if kind == "failed_text":
        assert reread["status"] == "raw_fallback"
        assert reread["raw_response"] == value
    else:
        assert reread["graph"] == value
        assert "status" not in reread
    assert not migrate_scene_file(path, [reread])


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


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_changed_settings_retry_only_explicit_scene_and_summary_failures(
    ready_context,
    fake_models,
    representation,
):
    model = "qwen"
    import extraction.steps as steps
    from extraction.failures import FailureLog
    from validation.steps import embed_representations

    context = ready_context
    extract = getattr(steps, f"extract_{representation}_scenes")
    summarize = getattr(steps, f"summarize_{representation}")
    scene_schema = f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md"
    summary_schema = f"prompts/summary_{representation}_v5.md"
    extract(
        context,
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{model}",
        model=model,
        schema=scene_schema,
    )
    summarize(
        context,
        model="qwen",
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{model}",
        schema=summary_schema,
    )
    directory = context.extraction_dir(representation, model, "scenes")
    summary_dir = context.summary_arm_dir(
        f"{'graph' if representation == 'graph' else 'desc'}_{model}"
    )
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
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{model}",
            model=model,
            schema=scene_schema,
        )["failure_count"]
        == 0
    )
    assert len(fake_models) == count + 1
    assert [task.task_id for task in fake_models[-1]] == [f"{selected}:0"]
    assert all(path.read_bytes() == saved for path, saved in before.items())
    assert not (directory / ".metadata").exists()
    assert not failures.path_for(selected).exists()
    # Even newly generated scene inputs do not replace a successful summary.
    summarize(
        context,
        model="qwen",
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{model}",
        schema=summary_schema,
    )
    assert len(fake_models) == count + 1
    assert all(path.read_bytes() == saved for path, saved in summaries.items())
    summary_failures = FailureLog(summary_dir)
    summary_failures.record(selected, None, "previous failure", "", repetition_penalty=1.0)
    summarize(
        context,
        model="qwen",
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{model}",
        schema=summary_schema,
    )
    assert len(fake_models) == count + 2
    assert [task.task_id for task in fake_models[-1]] == [selected]
    assert not summary_failures.path.exists()
    assert all(
        path.read_bytes() == saved for path, saved in summaries.items() if path.stem != selected
    )
    arm = f"{'desc' if representation == 'description' else 'graph'}_{model}"
    context.config["protocol"]["arms"] = [arm]
    assert embed_representations(context, target=[arm])["generated_arms"] == [arm]
    extract(
        context,
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{model}",
        model=model,
        schema=scene_schema,
        force=True,
    )
    assert len(fake_models[-1]) == len(scene_paths)
