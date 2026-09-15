from contextlib import contextmanager
from pathlib import Path

import pytest

from extraction.steps import extract_description_scenes, extract_graph_scenes, summarize_graph


@pytest.mark.parametrize("force", [False, True])
def test_completed_scene_journals_removed_and_interrupted_force_resumes(
    ready_context, fake_models, monkeypatch, force
):
    context = ready_context
    schema = "prompts/description_scene_v2.md"
    if force:
        extract_description_scenes(context, model="qwen", schema=schema)
    first = []

    @contextmanager
    def interrupted(**kwargs):
        def generate(tasks, callback):
            iterator = iter(tasks)
            task = next(iterator)
            first.append(task.task_id)
            callback(task.task_id, "A person waves.")
            raise KeyboardInterrupt

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", interrupted)
    with pytest.raises(KeyboardInterrupt):
        extract_description_scenes(context, model="qwen", schema=schema, force=force)
    assert not list((context.description_scene_dir("qwen") / ".recovery").glob("*.json"))
    resumed = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                resumed.append(task.task_id)
                callback(task.task_id, "A person waves.")
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    extract_description_scenes(context, model="qwen", schema=schema)
    assert first[0] not in resumed and len(resumed) == 3
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(resumed) == 3
    assert not (context.description_scene_dir("qwen") / ".recovery").exists()


def test_summary_correction_resume_replays_response_after_publish_failure(
    ready_context, fake_models, monkeypatch
):
    import extraction.summary_executor as executor

    context = ready_context
    extract_graph_scenes(context, model="qwen", schema="prompts/graph_scene_v3.md")
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append([t.task_id for t in tasks])
            for task in tasks:
                callback(
                    task.task_id, "word " * (201 if "Draft to correct:" not in task.prompt else 100)
                )
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    original = executor.atomic_write_json

    def write(path, doc, **kw):
        if Path(path).parent == context.graph_summary_dir("qwen"):
            raise OSError("disk full")
        return original(path, doc, **kw)

    monkeypatch.setattr(executor, "atomic_write_json", write)
    with pytest.raises(OSError, match="disk full"):
        summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md", force=True)
    first = calls[-1][0]
    monkeypatch.setattr(executor, "atomic_write_json", original)
    calls.clear()
    summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md")
    assert all(first not in batch for batch in calls)
    assert len(calls) == 1  # Remaining corrections only; no draft is regenerated.
    assert not (context.graph_summary_dir("qwen") / ".pending").exists()
    assert not (context.graph_summary_dir("qwen") / ".recovery").exists()


def test_all_empty_summary_is_failure_and_gemini_missing_can_fall_back(
    ready_context, fake_models, monkeypatch
):
    context = ready_context
    for model in ("qwen", "gemini"):
        extract_graph_scenes(context, model=model, schema="prompts/graph_scene_v3.md")

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                callback(task.task_id, "")
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    with pytest.raises(RuntimeError, match="summary failed"):
        summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md")
    assert not list(context.graph_summary_dir("qwen").glob("*.json"))
    assert len(list(context.graph_summary_failure_dir("qwen").glob("*.jsonl"))) == 4
    assert (
        summarize_graph(context, source="gemini", schema="prompts/graph_summary_v4.md")[
            "failure_count"
        ]
        == 4
    )


def test_compact_generation_history_does_not_duplicate_response_text(tmp_path):
    from extraction.backends.qwen_workers import QwenGenerationTask
    from extraction.qwen_runtime import QwenRuntime
    from extraction.recovery import generate_with_recovery

    runtime = QwenRuntime()
    runtime.engine_ready(
        {"worker_index": 0, "gpu_id": "0", "backend": "fixture", "gpu_name": "test"}
    )
    runtime.current_result = {
        "worker_index": 0,
        "task_id": "x",
        "text": "private raw response",
        "output_tokens": 8,
        "finish_reason": "stop",
    }
    outputs = []
    generate_with_recovery(
        lambda tasks, callback: {"x": "A person waves."},
        [QwenGenerationTask("x", (), "prompt", 32)],
        penalties=[1],
        directory=tmp_path,
        identity={"model": "test"},
        validate=lambda cid, text: ({"text": text}, "native"),
        complete=lambda cid, doc: outputs.append(doc),
        failed=lambda *args: pytest.fail("unexpected failure"),
        runtime=runtime,
    )
    assert "private raw response" not in str(outputs)
    assert outputs[0]["generation"]["output_tokens"] == 8
    assert outputs[0]["generation"]["origin"]["backend"] == "fixture"
    assert not list(tmp_path.glob("*.json"))
