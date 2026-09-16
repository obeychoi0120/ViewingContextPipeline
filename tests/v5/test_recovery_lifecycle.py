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
                    rows = read_jsonl(directory / "failure.jsonl")
                    assert len(rows) == 2 and all(set(row) == {"content_id", "error"} for row in rows)
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
    rows = read_jsonl(directory / "failure.jsonl")
    assert rows == [
        {"content_id": ids[0], "error": "empty"},
        {"content_id": ids[1], "error": "over_200_words"},
        {"content_id": ids[2], "error": "max_tokens"},
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
    assert not (directory / "failure.jsonl").exists()


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


def test_legacy_failure_records_are_compacted_and_temporary_state_removed(tmp_path):
    from extraction.failures import FailureLog
    write_jsonl(tmp_path / "failures/a.jsonl", [{
        "scene_idx": 2, "error": "invalid JSON", "raw_response": "discard me",
        "attempt_count": 5, "status": "retry_pending", "keyframes": [5],
    }])
    for name in (".recovery", ".pending", ".checkpoints"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "old.json").write_text("unfinished")
    (tmp_path / ".pending-contents.json").write_text("old cursor")
    log = FailureLog(tmp_path)
    assert read_jsonl(log.path) == [{"content_id": "a", "scene_idx": 2, "error": "invalid JSON"}]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["failure.jsonl"]
    log.record("b", 0, "empty")
    log.record("b", 0, "empty")
    assert len(read_jsonl(log.path)) == 2
    log.clear_contents(["a"])
    assert read_jsonl(log.path) == [{"content_id": "b", "scene_idx": 0, "error": "empty"}]
