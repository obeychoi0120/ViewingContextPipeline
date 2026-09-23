from contextlib import contextmanager
import json

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.scene_storage import read_scene_records
from pipeline_runtime import read_jsonl


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_graph_is_a_generation_failure(text):
    from extraction.scene_executor import graph_scene_result

    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert record is None
    assert failure["failure_kind"] == "generation"
    assert failure["error"] == "model produced an empty graph"


@pytest.mark.parametrize("recover", [False, True])
def test_qwen_empty_graph_retries_without_publishing_raw(ready_context, monkeypatch, recover):
    context = ready_context
    context.config["extraction"]["graph_repetition_penalty"] = [1.0, 1.05]
    visuals = steps.visual_rows(context)[:1]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    cid = visuals[0]["content_id"]
    directory = context.graph_scene_dir("qwen")
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append(task.repetition_penalty)
                kwargs["runtime"].current_result = {"finish_reason": "stop"}
                text = ('[Entities]\nnone\n[Relations]\nnone' if recover and
                        task.repetition_penalty == 1.05 else ' \n\t ')
                callback(task.task_id, text)
                if task.repetition_penalty == 1.0:
                    assert not (directory / f"{cid}.jsonl").exists()
                    assert read_jsonl(directory / "failures" / f"{cid}.jsonl")[0]["error"] == (
                        "model produced an empty graph")
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    result = steps.extract_graph_scenes(context, model="qwen", arm="graph_qwen",
                                       schema="prompts/scene_graph_v4.md")
    assert calls == [1.0, 1.05]
    assert result["failure_count"] == (0 if recover else 1)
    path = directory / f"{cid}.jsonl"
    if recover:
        assert isinstance(read_jsonl(path)[0]["scene_graph"], dict)
        assert not (directory / "failures" / f"{cid}.jsonl").exists()
    else:
        assert not path.exists()


@pytest.mark.parametrize("truncated", [False, True])
def test_description_cutoff_is_saved_only_as_failure(ready_context, monkeypatch, truncated):
    model = "qwen"
    context = ready_context
    context.config["extraction"]["graph_repetition_penalty"] = [1.0]
    context.config["extraction"]["description_repetition_penalty"] = [1.0]
    visuals = steps.visual_rows(context)[:1]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    cid = visuals[0]["content_id"]
    text = "  A person walks across the room.\n"

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                kwargs["runtime"].current_result = {
                    "finish_reason": "length" if truncated else "stop"
                }
                callback(task.task_id, text)
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    result = steps.extract_description_scenes(
        context,
        arm=f"desc_{model}",
        model=model,
        schema="prompts/scene_description_v2.md",
    )
    assert result["failure_count"] == int(truncated)
    directory = context.extraction_dir("description", model, "scenes")
    failure_path = directory / "failures" / f"{cid}.jsonl"
    if truncated:
        assert not (directory / f"{cid}.jsonl").exists()
        assert read_jsonl(failure_path) == [
            {
                "content_id": cid,
                "scene_idx": 0,
                "error": "description: output truncated at token limit",
                "raw_output": "",
                "provenance": read_jsonl(failure_path)[0]["provenance"],
            }
        ]
    else:
        assert not failure_path.exists()
        records = read_scene_records(directory / f"{cid}.jsonl")
        assert records[0]["description"] == text.strip()


@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_graph_text_repair_and_cutoff_share_storage_and_failure_policy(
    ready_context,
    monkeypatch,
    model,
):
    from enum import Enum

    from extraction.backends.qwen_workers import QwenGenerationTask
    from extraction.semantic_graph.schema import graph_summary_prompt

    class FinishReason(Enum):
        MAX_TOKENS = "MAX_TOKENS"

    context = ready_context
    context.config["extraction"]["graph_repetition_penalty"] = [1.0]
    context.config["extraction"]["description_repetition_penalty"] = [1.0]
    visual = steps.visual_rows(context)[0]
    cid = visual["content_id"]
    monkeypatch.setattr(steps, "visual_rows", lambda _: [visual])
    text = """[Entities]
person1: person; long-haired
cup1: cup; blue-green
[Relations]
person1 -> holding -> cup1
[Context]
An indoor gathering.
[End]"""
    responses = [
        text,
        text.replace("[Entities]", "Entities:").replace(" -> ", " - "),
        text.removesuffix("[End]"),
        text.replace("person1 ->", "person1 <-"),
        text,  # Even a complete-looking response fails when the backend reports truncation.
        text.removesuffix("[End]"),  # Missing End does not hide a backend token cutoff.
        json.dumps({"entities": [], "relations": [], "context": []}),
    ]

    def rows(visual, **kwargs):
        indices = (
            range(len(responses))
            if kwargs.get("scenes") is None
            else [row["scene_idx"] for row in kwargs["scenes"]]
        )
        return [
            {
                "task": QwenGenerationTask(f"{cid}:{i}", (), kwargs["prompt"], 1024),
                "scene_idx": i,
                "keyframes": [i * 30 + 5],
            }
            for i in indices
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    instances = []
    original_progress = steps.InferenceProgress

    def progress_factory(**kwargs):
        progress = original_progress(**kwargs)
        instances.append(progress)
        return progress

    monkeypatch.setattr(steps, "InferenceProgress", progress_factory)
    calls = []

    def receive_tasks(tasks, receive):
        for task in tasks:
            assert task.structured_output is None
            assert "[Entities]" in task.prompt
            calls.append(task.task_id)
            index = int(task.task_id.split(":")[1])
            receive(task.task_id, index, responses[index])

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            def receive(key, index, output):
                kwargs["runtime"].current_result = {
                    "finish_reason": "length" if index in (4, 5) else "stop",
                }
                callback(key, output)

            receive_tasks(tasks, receive)
            return {}

        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            def receive(key, index, output):
                reason = (
                    FinishReason.MAX_TOKENS
                    if index == 4
                    else "MAX_TOKENS"
                    if index == 5
                    else "STOP"
                )
                callback(
                    GeminiGenerationOutcome(
                        key,
                        output,
                        response_diagnostics={
                            "candidates": [{"finish_reason": reason}],
                        },
                    )
                )

            receive_tasks(tasks, receive)

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    options = {"model": model, "schema": "prompts/scene_graph_v4.md"}
    assert (
        steps.extract_graph_scenes(context, arm=f"graph_{options['model']}", **options)[
            "failure_count"
        ]
        == 2
    )
    assert calls == [f"{cid}:{i}" for i in range(len(responses))]
    assert (instances[-1].success, instances[-1].failed, instances[-1].raw) == (5, 2, 1)
    directory = context.graph_scene_dir(model)
    records = read_scene_records(directory / f"{cid}.jsonl")
    by_scene = {row["scene_idx"]: row for row in records}
    assert by_scene[0]["graph"] == by_scene[1]["graph"] == by_scene[2]["graph"]
    assert by_scene[3]["graph"] == responses[3]
    assert by_scene[3]["parse_mode"] == "text"
    assert by_scene[0]["parse_mode"] == "unknown"
    assert by_scene[1]["parse_mode"] == "unknown"
    assert "provenance" not in by_scene[0]
    assert all(list(row) == ["content_id", "warning", "scene_idx", "scene_graph"]
               for row in read_jsonl(directory / f"{cid}.jsonl"))
    assert by_scene[0]["graph"]["entities"][0]["attributes"] == ["long-haired"]
    assert by_scene[6]["graph"] == {"entities": [], "relations": [], "context": []}
    failures = read_jsonl(directory / "failures" / f"{cid}.jsonl")
    assert {row["scene_idx"] for row in failures} == {4, 5}
    for failure in failures:
        index = failure["scene_idx"]
        assert failure["raw_output"] == responses[index]
        assert index not in by_scene
        if index in (4, 5):
            assert failure["error"] == "graph: output truncated at token limit"
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert observations[0]["observation"] == by_scene[0]["graph"]
    assert len(observations) == 5
    assert observations[4]["observation"] == by_scene[6]["graph"]
    assert (
        steps.extract_graph_scenes(context, arm=f"graph_{options['model']}", **options)[
            "failure_count"
        ]
        == 2
    )
    expected = [f"{cid}:{i}" for i in range(len(responses))]
    expected.extend(f"{cid}:{i}" for i in (4, 5))
    assert calls == expected
