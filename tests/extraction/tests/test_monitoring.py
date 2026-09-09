from __future__ import annotations

import json

import pytest

import extraction.summary_executor as summary_executor
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.monitoring import (
    graph_skip_message,
    scene_messages,
    video_names,
)


def test_video_names_use_source_basename_and_fallback() -> None:
    assert video_names([
        {"content_id": "a", "source_video_path": "/videos/12.mp4"},
        {"content_id": "b", "source_video_path": None},
        {"content_id": "c", "source_video_path": r"C:\videos\34.mp4"},
    ]) == {"a": "12.mp4", "b": "b.mp4", "c": "34.mp4"}


def test_graph_scene_messages_are_sorted_and_use_normalized_json() -> None:
    records = [
        {"scene_idx": 2, "graph": {"setting_context": "outdoor_urban"}},
        {"scene_idx": 0, "graph": {"setting_context": "indoor"}},
    ]
    messages = scene_messages("1.mp4", records, arm="graph")

    assert messages[0].startswith("[Graph] 1.mp4 | scene #000\n")
    assert messages[1].startswith("[Graph] 1.mp4 | scene #002\n")
    assert json.loads(messages[0].split("\n", 1)[1]) == {
        "setting_context": "indoor"
    }


def test_description_scene_message_format() -> None:
    assert scene_messages(
        "2.mp4",
        [{"scene_idx": 3, "description": " visible action "}],
        arm="description",
    ) == ["[Desc] 2.mp4 | scene #003\nvisible action"]


def test_graph_skip_message_flattens_error_lines() -> None:
    assert graph_skip_message(
        "1.mp4",
        {"scene_idx": 2, "error": "JSON repair failed\nplain prose"},
    ) == "[Graph_skip] 1.mp4 | scene #002\nJSON repair failed plain prose"
    assert graph_skip_message(
        "1.mp4",
        {"scene_idx": 2, "error": "API failed"},
        source="gemini",
    ) == "[Graph_skip_gemini] 1.mp4 | scene #002\nAPI failed"


def test_graph_monitoring_can_identify_source() -> None:
    assert scene_messages(
        "1.mp4",
        [{"scene_idx": 0, "graph": {"entities": []}}],
        arm="graph",
        source="qwen",
    )[0].startswith("[Graph_qwen] 1.mp4 | scene #000\n")


def test_generator_defaults_to_one_gpu_and_passes_completed_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from extraction.qwen_runtime import QwenRuntimeLog

    runtime = QwenRuntimeLog(tmp_path, "summary")

    class Pool:
        def __init__(self, count, model_path, **kwargs):
            assert count == 1
            kwargs["on_runtime"]({"worker_index": 0, "gpu_id": "0"})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate(self, tasks, callback):
            for task in reversed(tasks):
                assert task.repetition_penalty == {"first": 1.05, "second": 1.2}[task.prompt]
                self.last_result = {"task_id": task.task_id, "worker_index": 0}
                callback(task.task_id, f"result:{task.prompt}")
                assert runtime.current_result is None
            return {}

    monkeypatch.setattr(summary_executor, "QwenWorkerPool", Pool)
    tasks = [
        QwenGenerationTask("a", (), "first", 10, repetition_penalty=1.05),
        QwenGenerationTask("b", (), "second", 10, repetition_penalty=1.2),
    ]
    completed = []

    def complete(task_id, text):
        assert runtime.current_result["task_id"] == task_id
        completed.append((task_id, text))

    with summary_executor.qwen_generator(model_path=tmp_path, gpus=None, runtime=runtime) as generate:
        results = generate(tasks, complete)

    assert completed == [("b", "result:second"), ("a", "result:first")]
    assert results == {}
