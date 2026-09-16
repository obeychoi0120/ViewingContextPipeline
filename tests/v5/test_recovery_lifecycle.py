from contextlib import contextmanager
from pathlib import Path

import pytest

from extraction.steps import extract_description_scenes, extract_graph_scenes, summarize_graph


@pytest.mark.parametrize("source", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["graph", "description"])
def test_summary_retries_after_full_pass_then_corrects_once(
    v5_context, source, representation,
):
    from arm_registry import select_arms
    from extraction.descriptions import SCENE_SCHEMA_VERSION
    from extraction.summary_executor import run_summary_stage
    from pipeline_runtime import read_json, write_jsonl

    context = v5_context
    name = f"{'graph' if representation == 'graph' else 'desc'}_{source}"
    arm = select_arms(context.config)[name]
    context.config["extraction"]["summary_repetition_penalty"] = [1.0, 1.05, 1.1]
    # Cross the old 256-item retry boundary for every summary source and representation.
    ids = [f"content_{i:03d}" for i in range(259)]
    correction_ids = set(ids[1:-1])
    scene_dir = context.extraction_dir(representation, source, "scenes")
    for cid in ids:
        record = {"scene_idx": 0, "keyframes": [5],
                  "provenance": {"arm": name, "representation": representation}}
        if representation == "graph":
            record.update(graph={"entities": [], "relations": [], "context": []},
                          parse_mode="native", semantic_warnings=[])
        else:
            record.update(schema_version=SCENE_SCHEMA_VERSION, content_id=cid,
                          description="A person walks outdoors.")
        write_jsonl(scene_dir / f"{cid}.jsonl", [record])
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            penalties = {task.repetition_penalty for task in tasks}
            assert len(penalties) == 1
            calls.append((penalties.pop(), {task.task_id for task in tasks}))
            for task in tasks:
                correction = "Draft to correct:" in task.prompt
                if correction:
                    assert task.task_id in correction_ids
                    assert len(calls) == 4  # All draft penalty passes have finished.
                threshold = {ids[0]: 1.1, ids[-1]: 1.05}.get(task.task_id, 1.0)
                text = ("A person walks outdoors." if correction else
                        "" if task.repetition_penalty < threshold else
                        "word " * 201 if task.task_id in correction_ids else "A person walks outdoors.")
                callback(task.task_id, text)
            return {}
        yield generate

    assert run_summary_stage(
        context, arm=arm, schema=context.prompt_path(f"prompts/{representation}_summary_v4.md"),
        catalog=[{"content_id": cid} for cid in ids], provenance={"fixture": name},
        generation={"repetition_penalty": 1.0}, generator_factory=generator,
    )["failure_count"] == 0
    assert calls == [(1.0, set(ids)), (1.05, {ids[0], ids[-1]}),
                     (1.1, {ids[0]}), (1.0, correction_ids)]
    output_dir = context.extraction_dir(representation, source, "summaries")
    assert read_json(output_dir / f"{ids[0]}.json")["generation"]["attempt_count"] == 3
    assert read_json(output_dir / f"{ids[1]}.json")["correction_count"] == 1


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
                if interrupt:
                    raise KeyboardInterrupt
                callback(GeminiGenerationOutcome(
                    task.task_id, '{"entities": [], "relations": [], "context": []}',
                ))

    monkeypatch.setattr("extraction.steps.GeminiWorkerPool", Pool)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            extract_graph_scenes(context, model="gemini", schema=schema)
    interrupt = False
    extract_graph_scenes(context, model="gemini", schema=schema)
    assert calls == [f"{missing.stem}:0"] * 3
    assert all(path.read_bytes() == contents for path, contents in preserved.items())
