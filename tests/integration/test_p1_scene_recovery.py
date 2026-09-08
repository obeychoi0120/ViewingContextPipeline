import json
from contextlib import contextmanager

import pytest

import extraction.steps as steps
from extraction.backends.qwen_workers import QwenGenerationTask
from pipeline_runtime import read_jsonl, write_jsonl
from pipeline_fixtures import context as context


@pytest.mark.parametrize("arm", ["graph", "description"])
@pytest.mark.parametrize("initial_success", [False, True])
def test_qwen_failed_and_missing_scenes_resume_without_losing_checkpoints(
    context, monkeypatch, arm, initial_success,
):
    visual = {"content_id": "c1"}
    monkeypatch.setattr(steps, "_visual_rows", lambda _: [visual])
    monkeypatch.setattr(steps, "_video_name_map", lambda _: {})
    rows = [
        {
            "scene_idx": index,
            "keyframes": [5 + index * 30],
            "task": QwenGenerationTask(f"c1:{index}", (), "prompt", 32),
        }
        for index in range(3)
    ]
    monkeypatch.setattr(steps, "_scene_generation_rows", lambda *args, **kwargs: rows)
    valid = json.dumps({"setting_context": "room"}) if arm == "graph" else "A person walks."
    invalid = "not json" if arm == "graph" else ""
    if arm == "graph":
        scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"
        failure_path = context.graph_failure_dir("qwen") / "c1.jsonl"

        def run(**kwargs):
            return steps.extract_graph_scenes(context, model="qwen", **kwargs)
    else:
        scene_path = context.description_scene_dir / "c1.jsonl"
        failure_path = context.description_failure_dir / "c1.jsonl"

        def run(**kwargs):
            return steps.extract_description_scenes(context, **kwargs)

    calls = []
    phase = "initial"

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            calls.append([task.task_id for task in tasks])
            for task in tasks:
                text = valid if phase != "initial" or initial_success and task.task_id == "c1:0" else invalid
                callback(task.task_id, text)
                if phase == "interrupt":
                    raise KeyboardInterrupt
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    assert run()["failure_count"] == 3 - int(initial_success)
    saved_success = read_jsonl(scene_path)
    if initial_success:
        # Scene 1 has no outcome; scene 2 still has a recorded failure.
        write_jsonl(failure_path, [r for r in read_jsonl(failure_path) if r["scene_idx"] == 2])
    phase = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        run()
    expected_pending = [f"c1:{i}" for i in range(int(initial_success), 3)]
    assert calls[-1] == expected_pending
    checkpoint = read_jsonl(scene_path)
    assert checkpoint[:len(saved_success)] == saved_success
    assert len(checkpoint) == int(initial_success) + 1
    remaining = [r["scene_idx"] for r in read_jsonl(failure_path)]
    assert remaining == list(range(int(initial_success) + 1, 3))
    phase = "recover"
    assert run()["failure_count"] == 0
    assert calls[-1] == [f"c1:{i}" for i in remaining]
    assert not failure_path.exists()
    assert [r["scene_idx"] for r in read_jsonl(scene_path)] == [0, 1, 2]
    before = scene_path.read_bytes()
    calls_before = len(calls)
    assert run()["failure_count"] == 0
    assert len(calls) == calls_before
    assert scene_path.read_bytes() == before
    assert run(force=True)["failure_count"] == 0
    assert calls[-1] == ["c1:0", "c1:1", "c1:2"]
