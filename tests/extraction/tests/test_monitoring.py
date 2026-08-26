from __future__ import annotations

import json

import pytest

import extraction.steps as steps_module
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.monitoring import scene_messages, summary_message, video_names


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


def test_description_and_summary_message_formats() -> None:
    assert scene_messages(
        "2.mp4",
        [{"scene_idx": 3, "description": " visible action "}],
        arm="description",
    ) == ["[Desc] 2.mp4 | scene #003\nvisible action"]
    assert summary_message(
        "2.mp4",
        arm="graph",
        scene_count=4,
        text=" graph summary ",
    ) == "[Summary_graph] 2.mp4 | 4 scenes\ngraph summary"
    assert summary_message(
        "2.mp4",
        arm="description",
        scene_count=4,
        text="description summary",
    ) == "[Summary_desc] 2.mp4 | 4 scenes\ndescription summary"


def test_serial_generator_calls_completion_callback_per_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class Backend:
        @classmethod
        def from_pretrained(cls, model_path, *, use_fc_patch):
            return cls()

        def generate(self, images, prompt, max_new_tokens):
            return f"result:{prompt}"

    monkeypatch.setattr(steps_module, "QwenBackend", Backend)
    monkeypatch.setattr(steps_module, "load_images", lambda paths: paths)
    tasks = [
        QwenGenerationTask("a", (), "first", 10),
        QwenGenerationTask("b", (), "second", 10),
    ]
    completed = []

    with steps_module._qwen_generator(model_path=tmp_path, gpus=None) as generate:
        results = generate(tasks, lambda task_id, text: completed.append((task_id, text)))

    assert completed == [("a", "result:first"), ("b", "result:second")]
    assert results == {"a": "result:first", "b": "result:second"}
