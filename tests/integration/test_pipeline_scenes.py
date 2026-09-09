from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

import extraction.steps as extraction_steps
from extraction.backends.qwen_workers import QwenGenerationTask
from pipeline_runtime import RunContext, read_jsonl, write_jsonl


from pipeline_fixtures import context as context


def test_graph_scene_failure_is_recorded_and_stage_continues(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", (), "prompt", 10),
            "scene_idx": index,
            "scene_start_seconds": index * 30,
            "scene_end_seconds": (index + 1) * 30,
            "keyframes": [index * 30 + 5],
            "image_paths": ["unused.png"],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )
    graph = {
        "setting_context": "indoor",
        "entities": [],
        "events": [],
        "semantic_topics": [],
        "affect": {"valence": "neutral", "arousal": "medium"},
    }

    @contextmanager
    def fake_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {tasks[0].task_id: json.dumps(graph), tasks[1].task_id: ""}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", fake_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 1
    scenes = read_jsonl(context.graph_scene_dir("qwen") / "c1.jsonl")
    assert len(scenes) == 1
    assert set(scenes[0]) == {
        "scene_idx",
        "keyframes",
        "graph",
        "parse_mode",
        "semantic_warnings",
    }
    assert scenes[0]["parse_mode"] == "native"
    failures = read_jsonl(context.graph_failure_dir("qwen") / "c1.jsonl")
    assert failures[0]["scene_idx"] == 1
    assert failures[0]["failure_kind"] == "json_repair"

    @contextmanager
    def successful_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {task.task_id: json.dumps(graph) for task in tasks}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", successful_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen", force=True)

    assert result["failure_count"] == 0
    failure_path = context.graph_failure_dir("qwen") / "c1.jsonl"
    assert not failure_path.exists()

    write_jsonl(failure_path, [])

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("completed scenes must be reused")
        yield

    monkeypatch.setattr(extraction_steps, "qwen_generator", unexpected_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 0
    assert not failure_path.exists()


@pytest.fixture()
def gemini_retry_case(context, monkeypatch):
    from extraction.backends import GeminiGenerationOutcome

    visuals = [{"content_id": name} for name in ("c1", "c2", "c3")]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _: visuals)
    monkeypatch.setattr(extraction_steps, "_video_name_map", lambda _: {})

    def scene_rows(visual, **_kwargs):
        content_id = visual["content_id"]
        return [
            {
                "task": QwenGenerationTask(f"{content_id}:{index}", (), "prompt", 32),
                "scene_idx": index,
                "keyframes": [index * 30 + 5],
            }
            for index in range(4 if content_id == "c1" else 1)
        ]

    monkeypatch.setattr(extraction_steps, "_scene_generation_rows", scene_rows)
    graph = {
        "setting_context": "indoor",
        "entities": [],
        "events": [],
        "semantic_topics": [],
        "affect": {"valence": "neutral", "arousal": "medium"},
    }
    successful = {
        "scene_idx": 0,
        "keyframes": [5],
        "graph": graph,
        "parse_mode": "native",
        "semantic_warnings": [],
    }
    for name in ("c1", "c2"):
        write_jsonl(context.graph_scene_dir("gemini") / f"{name}.jsonl", [successful])
    failures = [
        {
            "scene_idx": index,
            "keyframes": [index * 30 + 5],
            "failure_kind": "generation",
            "error": "old empty response",
            "raw_response": "",
        }
        for index in (1, 2)
    ]
    failure_path = context.graph_failure_dir("gemini") / "c1.jsonl"
    write_jsonl(failure_path, failures)
    diagnostics = {"candidates": [{"finish_reason": "SAFETY", "finish_message": "blocked"}]}
    outcomes = {
        "c1:1": GeminiGenerationOutcome("c1:1", json.dumps(graph)),
        "c1:2": GeminiGenerationOutcome("c1:2", "", "empty: SAFETY", diagnostics),
        "c1:3": GeminiGenerationOutcome("c1:3", json.dumps(graph)),
        "c3:0": GeminiGenerationOutcome("c3:0", json.dumps(graph)),
    }
    calls = []

    class Pool:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, tasks, callback, *, on_progress=None):
            calls.append([task.task_id for task in tasks])
            for task in reversed(tasks):
                callback(outcomes[task.task_id])

    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", Pool)
    return successful, failure_path, outcomes, calls, diagnostics


def test_gemini_default_resumes_failed_and_missing_scenes_preserving_successes(
    context,
    gemini_retry_case,
    capsys,
):
    from extraction.backends import GeminiGenerationOutcome

    successful, failure_path, outcomes, calls, diagnostics = gemini_retry_case
    c2 = context.graph_scene_dir("gemini") / "c2.jsonl"
    c2_bytes = c2.read_bytes()
    result = extraction_steps.extract_graph_scenes(context, model="gemini")
    assert result["failure_count"] == 1
    assert calls == [["c1:1", "c1:2", "c1:3", "c3:0"]]
    rows = read_jsonl(context.graph_scene_dir("gemini") / "c1.jsonl")
    assert [row["scene_idx"] for row in rows] == [0, 1, 3]
    assert rows[0] == successful
    assert c2.read_bytes() == c2_bytes
    assert (context.graph_scene_dir("gemini") / "c3.jsonl").exists()
    remaining = read_jsonl(failure_path)
    assert len(remaining) == 1 and remaining[0]["scene_idx"] == 2
    assert remaining[0]["response_diagnostics"] == diagnostics
    assert "SAFETY" in capsys.readouterr().err
    # Normal cache normalization must retain the new diagnostic fields.
    assert extraction_steps._minimal_graph_failures(remaining) == remaining

    outcomes["c1:2"] = GeminiGenerationOutcome("c1:2", json.dumps(successful["graph"]))
    result = extraction_steps.extract_graph_scenes(context, model="gemini")
    assert result["failure_count"] == 0
    assert calls[-1] == ["c1:2"]
    assert not failure_path.exists()
    rows = read_jsonl(context.graph_scene_dir("gemini") / "c1.jsonl")
    assert [row["scene_idx"] for row in rows] == [0, 1, 2, 3]
    extraction_steps.extract_graph_scenes(context, model="gemini")
    assert len(calls) == 2  # No failures left: no additional API calls.


@pytest.mark.parametrize("mismatch", ["keyframes", "unknown_index", "overlap"])
def test_gemini_retry_rejects_incompatible_cache_before_calls(
    context,
    gemini_retry_case,
    mismatch,
):
    _, failure_path, _, calls, _ = gemini_retry_case
    failures = read_jsonl(failure_path)
    if mismatch == "keyframes":
        failures[0]["keyframes"] = [999]
    elif mismatch == "unknown_index":
        failures[0]["scene_idx"] = 99
    else:
        failures[0]["scene_idx"] = 0
        failures[0]["keyframes"] = [5]
    write_jsonl(failure_path, failures)
    before = failure_path.read_bytes()
    with pytest.raises(RuntimeError, match="incompatible cached"):
        extraction_steps.extract_graph_scenes(context, model="gemini")
    assert calls == []
    assert failure_path.read_bytes() == before


def test_gemini_force_regenerates_all_and_default_retries_persistent_failure(
    context,
    gemini_retry_case,
):
    from extraction.backends import GeminiGenerationOutcome

    successful, failure_path, outcomes, calls, diagnostics = gemini_retry_case
    for task_id in ("c1:0", "c1:3", "c2:0", "c3:0"):
        outcomes[task_id] = GeminiGenerationOutcome(task_id, json.dumps(successful["graph"]))
    result = extraction_steps.extract_graph_scenes(context, model="gemini", force=True)
    assert calls == [["c1:0", "c1:1", "c1:2", "c1:3", "c2:0", "c3:0"]]
    assert result["failure_count"] == 1
    assert read_jsonl(failure_path)[0]["response_diagnostics"] == diagnostics
    before = failure_path.read_bytes()
    result = extraction_steps.extract_graph_scenes(context, model="gemini")
    assert result["failure_count"] == 1
    assert failure_path.read_bytes() == before
    assert len(calls) == 2
    assert calls[-1] == ["c1:2"]


def test_gemini_retry_interrupt_preserves_existing_checkpoint(
    context, gemini_retry_case, monkeypatch
):
    _, failure_path, _, _, _ = gemini_retry_case
    scene_path = context.graph_scene_dir("gemini") / "c1.jsonl"
    before = (scene_path.read_bytes(), failure_path.read_bytes())

    class InterruptedPool:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", InterruptedPool)
    with pytest.raises(KeyboardInterrupt):
        extraction_steps.extract_graph_scenes(context, model="gemini")
    assert (scene_path.read_bytes(), failure_path.read_bytes()) == before


def test_description_failure_force_retry_and_cache_reuse(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", (), "prompt", 10),
            "scene_idx": index,
            "scene_start_seconds": index * 30,
            "scene_end_seconds": (index + 1) * 30,
            "keyframes": [index * 30 + 5],
            "image_paths": ["unused.png"],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )

    @contextmanager
    def failing_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {tasks[0].task_id: "visible action", tasks[1].task_id: ""}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", failing_generator)
    result = extraction_steps.extract_description_scenes(context)
    assert result["failure_count"] == 1
    failure_path = context.description_failure_dir / "c1.jsonl"
    assert read_jsonl(failure_path)[0]["failure_kind"] == "empty_response"

    @contextmanager
    def successful_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {task.task_id: f"visible action {task.task_id}" for task in tasks}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", successful_generator)
    result = extraction_steps.extract_description_scenes(context, force=True)
    assert result["failure_count"] == 0
    assert not failure_path.exists()
    assert len(read_jsonl(context.description_scene_dir / "c1.jsonl")) == 2

    write_jsonl(failure_path, [])

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("completed description scenes must be reused")
        yield

    monkeypatch.setattr(extraction_steps, "qwen_generator", unexpected_generator)
    result = extraction_steps.extract_description_scenes(context)
    assert result["failure_count"] == 0
    assert not failure_path.exists()


def test_qwen_graph_scene_is_checkpointed_before_content_finishes(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", (), "prompt", 10),
            "scene_idx": index,
            "keyframes": [index * 30 + 5],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )
    graph = {
        "setting_context": "indoor",
        "entities": [],
        "events": [],
        "semantic_topics": [],
        "affect": {"valence": "neutral", "arousal": "medium"},
    }
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"

    @contextmanager
    def streaming_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[1].task_id, json.dumps(graph))
            assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [1]
            callback(tasks[0].task_id, json.dumps(graph))
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", streaming_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 0
    assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [0, 1]
    assert "each completed scene is checkpointed immediately" in capsys.readouterr().err


def test_qwen_description_scene_is_checkpointed_before_content_finishes(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", (), "prompt", 10),
            "scene_idx": index,
            "keyframes": [index * 30 + 5],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )
    scene_path = context.description_scene_dir / "c1.jsonl"
    failure_path = context.description_failure_dir / "c1.jsonl"

    @contextmanager
    def streaming_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[1].task_id, "visible action")
            assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [1]
            callback(tasks[0].task_id, "")
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", streaming_generator)
    result = extraction_steps.extract_description_scenes(context)

    assert result["failure_count"] == 1
    assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [1]
    assert [row["scene_idx"] for row in read_jsonl(failure_path)] == [0]
    assert "each completed scene is checkpointed immediately" in capsys.readouterr().err


def test_gemini_scene_stage_aggregates_out_of_order_errors(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", (), "prompt", 10),
            "scene_idx": index,
            "scene_start_seconds": index * 30,
            "scene_end_seconds": (index + 1) * 30,
            "keyframes": [index * 30 + 5],
            "image_paths": ["unused.png"],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )
    graph = {
        "setting_context": "indoor",
        "entities": [],
        "events": [],
        "semantic_topics": [],
        "affect": {"valence": "neutral", "arousal": "medium"},
    }

    class Pool:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, tasks, callback, *, on_progress=None):
            outcomes = {
                tasks[1].task_id: extraction_steps.GeminiGenerationOutcome(
                    tasks[1].task_id, "", "RuntimeError: quota"
                ),
                tasks[0].task_id: extraction_steps.GeminiGenerationOutcome(
                    tasks[0].task_id, json.dumps(graph), None
                ),
            }
            for task_id in (tasks[1].task_id, tasks[0].task_id):
                callback(outcomes[task_id])
            return outcomes

    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", Pool)
    result = extraction_steps.extract_graph_scenes(context, model="gemini")
    assert result["failure_count"] == 1
    assert len(read_jsonl(context.graph_scene_dir("gemini") / "c1.jsonl")) == 1
    failure = read_jsonl(context.graph_failure_dir("gemini") / "c1.jsonl")[0]
    assert failure["scene_idx"] == 1
    assert failure["failure_kind"] == "generation"
