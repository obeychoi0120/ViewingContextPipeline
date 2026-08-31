from __future__ import annotations

import json

import pytest

import extraction.step_support as step_support
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


def test_serial_generator_calls_completion_callback_per_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class Backend:
        @classmethod
        def from_pretrained(cls, model_path, *, use_fc_patch):
            return cls()

        def generate(self, images, prompt, max_new_tokens, **_generation):
            return f"result:{prompt}"

    monkeypatch.setattr(summary_executor, "QwenBackend", Backend)
    monkeypatch.setattr(summary_executor, "load_images", lambda paths: paths)
    tasks = [
        QwenGenerationTask("a", (), "first", 10),
        QwenGenerationTask("b", (), "second", 10),
    ]
    completed = []

    with summary_executor.qwen_generator(model_path=tmp_path, gpus=None) as generate:
        results = generate(tasks, lambda task_id, text: completed.append((task_id, text)))

    assert completed == [("a", "result:first"), ("b", "result:second")]
    assert results == {}


def test_complete_content_progress_updates_then_writes_blank_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class Progress:
        def update(self, amount):
            events.append(("update", amount))

    monkeypatch.setattr(
        step_support,
        "write_progress",
        lambda _progress, message: events.append(("write", message)),
    )

    step_support.complete_content_progress(Progress())

    assert events == [("update", 1), ("write", "")]
