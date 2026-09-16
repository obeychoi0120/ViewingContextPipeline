from contextlib import contextmanager

import pytest

from extraction.steps import extract_description_scenes, extract_graph_scenes
from pipeline_runtime import read_json, read_jsonl, write_jsonl


@pytest.mark.parametrize("source", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["graph", "description"])
def test_summary_single_pass_records_first_failure_without_scene_or_correction(
    ready_context, fake_models, monkeypatch, source, representation,
):
    import extraction.steps as steps
    import extraction.summary_executor as executor
    context = ready_context
    extract = extract_graph_scenes if representation == "graph" else extract_description_scenes
    extract(context, model=source,
            schema=f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md")
    context.config["extraction"]["summary_repetition_penalty"] = [1.0, 1.05, 1.1]
    summarize = getattr(steps, f"summarize_{representation}")
    options = {"source": source, "schema": f"prompts/{representation}_summary_v4.md"}
    directory = context.extraction_dir(representation, source, "summaries")
    ids = [item["content_id"] for item in context.require_ready_cohort()["catalog"]]
    calls, progress_instances = [], []
    interrupt, succeed = True, False
    original = executor.InferenceProgress

    def progress_factory(**kwargs):
        progress = original(**kwargs)
        progress_instances.append(progress)
        return progress
    monkeypatch.setattr(executor, "InferenceProgress", progress_factory)

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append(task.task_id)
                assert task.repetition_penalty == 1.0 and "Draft to correct:" not in task.prompt
                index = ids.index(task.task_id)
                text = "A person walks outdoors." if succeed or index == 3 else "" if index == 0 else "word " * 201 if index == 1 else "A truncated sentence"
                kwargs["runtime"].current_result = {"finish_reason": "length" if index == 2 and not succeed else "stop"}
                callback(task.task_id, text)
                if interrupt and index == 1:
                    rows = read_jsonl(directory / "failures.jsonl")
                    assert len(rows) == 2 and all(set(row) == {"content_id", "error", "raw_output"} for row in rows)
                    assert progress_instances[-1].failed == 2
                    for name in (".recovery", ".pending", "failures"):
                        assert not (directory / name).exists()
                    raise KeyboardInterrupt
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    with pytest.raises(KeyboardInterrupt):
        summarize(context, **options)
    interrupt = False
    assert summarize(context, **options)["failure_count"] == 3
    assert calls == ids
    rows = read_jsonl(directory / "failures.jsonl")
    assert rows == [
        {"content_id": ids[0], "error": "empty", "raw_output": ""},
        {"content_id": ids[1], "error": "over_200_words", "raw_output": "word " * 201},
        {"content_id": ids[2], "error": "max_tokens", "raw_output": "A truncated sentence"},
    ]
    for cid in ids[1:3]:
        doc = read_json(directory / f"{cid}.json")
        assert doc["status"] == "raw_fallback" and doc["correction_count"] == 0
    assert not (directory / f"{ids[0]}.json").exists()
    assert summarize(context, **options)["failure_count"] == 3
    assert calls == ids
    succeed = True
    assert summarize(context, **options, force=True)["failure_count"] == 0
    assert calls == ids * 2
    assert not (directory / "failures.jsonl").exists()


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_summary_publish_error_requires_regeneration_without_journal(
    ready_context, fake_models, monkeypatch, representation,
):
    import extraction.steps as steps
    import extraction.summary_executor as executor
    context = ready_context
    extract = getattr(steps, f"extract_{representation}_scenes")
    extract(context, model="qwen", schema=f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md")
    calls = []
    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append(task.task_id)
                callback(task.task_id, "A person walks outdoors.")
            return {}
        yield generate
    monkeypatch.setattr(steps, "qwen_generator", generator)
    original = executor.atomic_write_json
    def fault(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(executor, "atomic_write_json", fault)
    summarize = getattr(steps, f"summarize_{representation}")
    options = {"source": "qwen", "schema": f"prompts/{representation}_summary_v4.md"}
    with pytest.raises(OSError, match="disk full"):
        summarize(context, **options)
    directory = context.extraction_dir(representation, "qwen", "summaries")
    assert not list(directory.rglob("*.json"))
    monkeypatch.setattr(executor, "atomic_write_json", original)
    assert summarize(context, **options)["failure_count"] == 0
    assert calls[0] == calls[1] and len(calls) == 5


@pytest.mark.parametrize("scenes", [False, True])
def test_legacy_failures_migrate_with_raw_output_and_no_temporary_state(tmp_path, scenes):
    from extraction.failures import FailureLog
    write_jsonl(tmp_path / "failures/a.jsonl", [{
        "scene_idx": 2, "error": "invalid JSON", "raw_response": "  raw text\n",
        "attempt_count": 5, "status": "retry_pending", "keyframes": [5],
    }])
    write_jsonl(tmp_path / "failure.jsonl", [{
        "content_id": "b", "scene_idx": 0, "error": "empty",
    }])
    write_jsonl(tmp_path / "failures.jsonl", [{
        "content_id": "c", "scene_idx": 1, "error": "invalid", "raw_output": "truncated",
    }])
    for name in (".recovery", ".pending", ".checkpoints"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "old.json").write_text("unfinished")
    (tmp_path / ".pending-contents.json").write_text("old cursor")
    log = FailureLog(tmp_path, scenes=scenes)
    expected = [
        {"content_id": "b", "error": "empty", "raw_output": "", **({"scene_idx": 0} if scenes else {})},
        {"content_id": "c", "error": "invalid", "raw_output": "truncated", **({"scene_idx": 1} if scenes else {})},
        {"content_id": "a", "error": "invalid JSON", "raw_output": "  raw text\n", **({"scene_idx": 2} if scenes else {})},
    ]
    rows = [row for cid in ("b", "c", "a") for row in read_jsonl(log.path_for(cid))] if scenes else read_jsonl(log.path)
    assert sorted(rows, key=lambda row: row["content_id"]) == sorted(expected, key=lambda row: row["content_id"])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["failures" if scenes else "failures.jsonl"]
    log = FailureLog(tmp_path, scenes=scenes)  # Reopening preserves the new format and raw bytes.
    log.record("b", 0 if scenes else None, "empty", "")
    log.record("d", 3 if scenes else None, "invalid", "new response")
    log.record("d", 3 if scenes else None, "invalid", "replacement response\n")
    assert log.count(["a", "b", "c", "d"]) == 4
    row = next(r for r in read_jsonl(log.path_for("d")) if r["content_id"] == "d")
    assert row["raw_output"] == "replacement response\n"
    log.clear_contents(["a", "b", "c"])
    assert log.count(["a", "b", "c", "d"]) == 1
    if scenes:
        assert sorted(path.name for path in log.path.iterdir()) == ["d.jsonl"]
    else:
        assert len(read_jsonl(log.path)) == 1


@pytest.mark.parametrize("scenes", [False, True])
def test_current_failure_format_wins_over_old_records_after_partial_migration(tmp_path, scenes):
    from extraction.failures import FailureLog
    old = {"content_id": "a", "scene_idx": 2, "error": "old", "raw_output": "old output"}
    current = {"content_id": "a", "error": "current", "raw_output": "current output"}
    if scenes:
        current["scene_idx"] = 2
    write_jsonl(tmp_path / "failure.jsonl", [old])
    write_jsonl(tmp_path / "failures.jsonl", [old if scenes else current])
    write_jsonl(tmp_path / "failures/a.jsonl", [current if scenes else old])
    log = FailureLog(tmp_path, scenes=scenes)
    assert read_jsonl(log.path_for("a")) == [current]
