from contextlib import contextmanager
import json

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.scene_storage import read_scene_records


def test_qwen_resume_reuses_successes_and_retries_failures_from_first_penalty(
    context,
    monkeypatch,
):
    representation = "graph"
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [
            {
                "task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                "scene_idx": i,
                "keyframes": [i * 30 + 5],
            }
            for i in (
                range(2)
                if kwargs.get("scenes") is None
                else (scene["scene_idx"] for scene in kwargs["scenes"])
            )
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    graph = json.dumps({"medium": "live_action", "format": "other", "topics": [], "entities": [], "actions": []})
    interrupt = True
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                event = (task.repetition_penalty, task.task_id)
                calls.append(event)
                kwargs["runtime"].current_result = {
                    "finish_reason": "length" if interrupt and task.task_id == "a:0" else "stop"
                }
                callback(task.task_id, "" if interrupt and task.task_id == "a:0" else graph)
                if interrupt and event == (1.05, "a:1"):
                    raise KeyboardInterrupt
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    schema = f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md"
    extract = getattr(steps, f"extract_{representation}_scenes")
    with pytest.raises(KeyboardInterrupt):
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_qwen3vl",
            model="qwen3vl",
            schema=schema,
        )
    interrupt = False
    calls.clear()
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_qwen3vl",
            model="qwen3vl",
            schema=schema,
        )["failure_count"]
        == 0
    )
    assert calls == [
        (1.05, "a:0"),
        (1.05, "b:0"),
        (1.05, "b:1"),
    ]
    scene_dir = context.extraction_dir(representation, "qwen3vl", "scenes")
    assert not (scene_dir / ".recovery").exists()
    for cid in ("a", "b"):
        records = read_scene_records(scene_dir / f"{cid}.jsonl")
        assert [r["scene_idx"] for r in records] == [0, 1]


def test_gemini_resume_preserves_out_of_order_completions_across_interrupts(
    context,
    monkeypatch,
):
    scene_dir = context.graph_scene_dir("gemini")
    visuals = [{"content_id": cid} for cid in ("a", "b", "c")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [
            {
                "task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                "scene_idx": i,
                "keyframes": [i * 30 + 5],
            }
            for i in (
                range(2)
                if kwargs.get("scenes") is None
                else (row["scene_idx"] for row in kwargs["scenes"])
            )
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    run = 0
    submitted = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            if run == 1:
                # Interrupt again before discovery reaches the already saved b.
                next(iter(tasks))
                raise KeyboardInterrupt
            tasks = list(tasks)
            submitted.append([task.task_id for task in tasks])
            if run == 0:
                for task_id in ("b:1", "b:0", "a:0"):
                    callback(
                        GeminiGenerationOutcome(
                            task_id, '{"medium": "live_action", "format": "other", "topics": [], "entities": [], "actions": []}'
                        )
                    )
                raise KeyboardInterrupt
            for task in tasks:
                callback(
                    GeminiGenerationOutcome(
                        task.task_id, '{"medium": "live_action", "format": "other", "topics": [], "entities": [], "actions": []}'
                    )
                )

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    schema = "prompts/scene_graph_v4.md"
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, arm="graph_gemini", schema=schema, model="gemini")
    saved_b = (scene_dir / "b.jsonl").read_bytes()
    run = 1
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, arm="graph_gemini", schema=schema, model="gemini")
    run = 2
    assert (
        steps.extract_graph_scenes(context, arm="graph_gemini", schema=schema, model="gemini")[
            "failure_count"
        ]
        == 0
    )
    assert submitted[-1] == ["a:1", "c:0", "c:1"]
    assert (scene_dir / "b.jsonl").read_bytes() == saved_b
    assert not (scene_dir / ".pending-contents.json").exists()
