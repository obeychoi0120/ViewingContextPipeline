from contextlib import contextmanager
from pathlib import Path

import pytest

from extraction.steps import extract_description_scenes, extract_graph_scenes, summarize_graph


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("representation", ["graph", "description"])
def test_completed_scene_journals_removed_and_interrupted_force_resumes(
    ready_context, fake_models, monkeypatch, force, representation
):
    context = ready_context
    schema = f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md"
    extract = extract_graph_scenes if representation == "graph" else extract_description_scenes
    scene_dir = context.extraction_dir(representation, "qwen", "scenes")
    if force:
        extract(context, model="qwen", schema=schema)
    first = []

    @contextmanager
    def interrupted(**kwargs):
        def generate(tasks, callback):
            iterator = iter(tasks)
            task = next(iterator)
            first.append(task.task_id)
            callback(task.task_id, '{"entities": [], "relations": [], "context": []}')
            raise KeyboardInterrupt

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", interrupted)
    with pytest.raises(KeyboardInterrupt):
        extract(context, model="qwen", schema=schema, force=force)
    assert not list((scene_dir / ".recovery").glob("*.json"))
    resumed = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                resumed.append(task.task_id)
                callback(task.task_id, '{"entities": [], "relations": [], "context": []}')
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    extract(context, model="qwen", schema=schema)
    assert first[0] not in resumed and len(resumed) == 3
    extract(context, model="qwen", schema=schema)
    assert len(resumed) == 3
    assert not (scene_dir / ".recovery").exists()


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


def test_repeated_gemini_interrupt_keeps_unvisited_cached_contents(
    ready_context, fake_models, monkeypatch,
):
    from extraction.backends import GeminiGenerationOutcome

    context = ready_context
    schema = "prompts/graph_scene_v3.md"
    extract_graph_scenes(context, model="gemini", schema=schema)
    paths = sorted(context.graph_scene_dir("gemini").glob("*.jsonl"))
    missing = paths[1]
    missing.unlink()
    preserved = {path: path.read_bytes() for path in paths if path != missing}
    calls = []
    interrupt = True

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                calls.append(task.task_id)
                callback(GeminiGenerationOutcome(
                    task.task_id, '{"entities": [], "relations": [], "context": []}',
                ))
                if interrupt:
                    raise KeyboardInterrupt

    monkeypatch.setattr("extraction.steps.GeminiWorkerPool", Pool)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            extract_graph_scenes(context, model="gemini", schema=schema)
    interrupt = False
    extract_graph_scenes(context, model="gemini", schema=schema)
    assert calls == [f"{missing.stem}:0"] * 3
    assert all(path.read_bytes() == contents for path, contents in preserved.items())
