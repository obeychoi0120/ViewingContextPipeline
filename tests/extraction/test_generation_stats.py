"""Final Graph counts include reused scenes and reflect retries and interruptions."""

from contextlib import contextmanager

import pytest

from artifact_io import read_json, read_jsonl
from extraction import steps
from extraction.backends import GeminiGenerationOutcome
from extraction.backends.qwen_workers import QwenGenerationTask


@pytest.mark.parametrize("model", ["qwen3vl", "qwen3.5", "gemini"])
def test_graph_generation_stats_cover_resume_recovery_and_interruption(
    ready_context, monkeypatch, model,
):
    visual = steps.visual_rows(ready_context)[0]
    cid = visual["content_id"]
    directory = ready_context.graph_scene_dir(model)
    stats_path = directory / "generation_stats.json"
    valid = (
        "[Context]\nmedium: live_action\nformat: other\ntopics: none\n"
        "[Entities]\nnone\n[Actions]\nnone"
    )
    responses = [valid, valid, valid.split("[Actions]")[0], "{broken JSON", ""]
    monkeypatch.setattr(steps, "visual_rows", lambda _: [visual])

    def rows(visual, *, scenes=None, **kwargs):
        indices = range(len(responses)) if scenes is None else [s["scene_idx"] for s in scenes]
        return [
            {"scene_idx": i, "keyframes": [],
             "task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32)}
            for i in indices
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    calls = []
    interrupt = False

    def generate(tasks, callback):
        assert not stats_path.exists()
        for task in tasks:
            calls.append(task.task_id)
            callback(task.task_id, responses[int(task.task_id.split(":")[1])])
            if interrupt:
                raise KeyboardInterrupt
        return {}

    @contextmanager
    def generator(**kwargs):
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            generate(tasks, lambda key, text: callback(GeminiGenerationOutcome(key, text)))

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    options = {"arm": f"graph_{model}", "model": model, "schema": "prompts/scene_graph_v9.md"}
    result = steps.extract_graph_scenes(ready_context, **options)
    assert result["failure_count"] == 1
    assert read_json(stats_path) == {
        "total": 5, "success": 2, "warning": 2, "failed": 1,
        "warning_counts": {"MISSING_REQUIRED": 1, "PARSE_FAILED": 1},
    }
    assert calls == [f"{cid}:{i}" for i in range(5)]

    # A warning can become a failure; other pending rows must survive an interruption.
    scene_path = directory / f"{cid}.jsonl"
    before = read_jsonl(scene_path)
    responses[2] = ""
    interrupt = True
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(ready_context, **options)
    saved = read_jsonl(scene_path)
    assert saved[0:2] == before[0:2] and saved[3:] == before[3:]
    assert saved[2]["error"] == "model produced an empty graph"
    assert not stats_path.exists()
    interrupt = False
    responses[2] = valid.split("[Actions]")[0]
    responses[4] = valid
    result = steps.extract_graph_scenes(ready_context, **options)
    expected = {
        "total": 5, "success": 3, "warning": 2, "failed": 0,
        "warning_counts": {"MISSING_REQUIRED": 1, "PARSE_FAILED": 1},
    }
    assert result["failure_count"] == 0
    assert read_json(stats_path) == expected
    assert calls == [f"{cid}:{i}" for i in range(5)] + [f"{cid}:{i}" for i in (2, 2, 3, 4)]

    responses[2] = responses[3] = valid
    steps.extract_graph_scenes(ready_context, **options)
    expected.update(success=5, warning=0, warning_counts={"MISSING_REQUIRED": 0, "PARSE_FAILED": 0})
    assert read_json(stats_path) == expected
    assert calls[-2:] == [f"{cid}:2", f"{cid}:3"]
    stats_path.unlink()
    steps.extract_graph_scenes(ready_context, **options)
    assert read_json(stats_path) == expected
    assert len(calls) == 11  # All five persisted scenes were reused.

    interrupt = True
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(ready_context, force=True, **options)
    assert not stats_path.exists()
