from contextlib import contextmanager
import json

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.scene_storage import read_scene_records
from pipeline_runtime import read_jsonl


@pytest.mark.parametrize("model", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["description", "graph"])
def test_scene_failures_are_retried_on_resume(
    ready_context, monkeypatch, model, representation, capsys,
):
    context = ready_context
    context.config["extraction"]["graph_repetition_penalty"] = [1.0]
    context.config["extraction"]["description_repetition_penalty"] = [1.0]
    visuals = steps.visual_rows(context)[:2]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    context.config["extraction"][f"{representation}_repetition_penalty"] = [1.0]
    scene_dir = context.extraction_dir(representation, model, "scenes")
    failure_path = scene_dir / "failures" / f"{visuals[0]['content_id']}.jsonl"
    first_id, other_id = [f"{v['content_id']}:0" for v in visuals]
    instances = []
    original_progress = steps.InferenceProgress

    def progress_factory(**kwargs):
        progress = original_progress(**kwargs)
        instances.append(progress)
        return progress

    monkeypatch.setattr(steps, "InferenceProgress", progress_factory)
    interrupt, succeed = True, False
    calls = []
    good = json.dumps({"entities": [], "relations": [], "context": []})

    runtime = None

    def generate(tasks, callback):
        for task in tasks:
            calls.append((task.task_id, task.repetition_penalty))
            text = good if succeed or task.task_id == other_id else " \n\t"
            if runtime is not None:
                runtime.current_result = {"finish_reason": "length" if not succeed and task.task_id == first_id else "stop"}
            callback(task.task_id, text)
            if interrupt:
                assert read_jsonl(failure_path) == [{
                    "content_id": visuals[0]["content_id"], "scene_idx": 0,
                    "error": read_jsonl(failure_path)[0]["error"], "raw_output": "", "provenance": read_jsonl(failure_path)[0]["provenance"],
                }]
                progress = instances[-1]
                assert (progress.success, progress.failed) == (0, 1)
                for name in (".recovery", ".pending", ".checkpoints", ".pending-contents.json"):
                    assert not (scene_dir / name).exists()
                raise KeyboardInterrupt
        return {}

    @contextmanager
    def generator(**kwargs):
        nonlocal runtime
        runtime = kwargs["runtime"]
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, tasks, callback, **kwargs):
            return generate(tasks, lambda key, text: callback(GeminiGenerationOutcome(key, text, response_diagnostics={"candidates": [{"finish_reason": "MAX_TOKENS" if not succeed and key == first_id else "STOP"}]})))

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    extract = getattr(steps, f"extract_{representation}_scenes")
    options = {"model": model, "schema": f"prompts/scene_{representation}_v{'3' if representation == 'graph' else '2'}.md"}
    with pytest.raises(KeyboardInterrupt):
        extract(context, **options)
    interrupt = False
    assert extract(context, **options)["failure_count"] == 1
    expected = [(first_id, 1.0)] * 2 + [(other_id, 1.0)]
    assert calls == expected
    output = capsys.readouterr()
    assert "[Graph_skip" not in output.err and "[SKIPPED]" not in output.err
    assert read_jsonl(failure_path)[0]["error"] not in output.err
    assert extract(context, **options)["failure_count"] == 1
    expected.append((first_id, 1.0))
    assert calls == expected
    from validation.diagnosis_scenes import _scene_arm_contract, scene_arms
    errors = []
    name = f"{'desc' if representation == 'description' else 'graph'}_{model}"
    coverage, _, valid = _scene_arm_contract(
        name, context.run_root, [v["content_id"] for v in visuals],
        {(v["content_id"], 0) for v in visuals}, errors, scene_arms(context.config),
    )
    assert valid and not errors
    assert coverage["failure_scene_count"] == 1 and coverage["outcome_coverage"] == 1
    succeed = True
    assert extract(context, **options, force=True)["failure_count"] == 0
    assert not failure_path.exists()
    assert len(calls) == len(expected) + 2


@pytest.mark.parametrize("model", ["qwen", "gemini"])
@pytest.mark.parametrize("truncated", [False, True])
def test_description_cutoff_is_saved_only_as_failure(ready_context, monkeypatch, model, truncated):
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
                kwargs["runtime"].current_result = {"finish_reason": "length" if truncated else "stop"}
                callback(task.task_id, text)
            return {}
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                callback(GeminiGenerationOutcome(task.task_id, text, response_diagnostics={
                    "candidates": [{"finish_reason": "MAX_TOKENS" if truncated else "STOP"}],
                }))

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    result = steps.extract_description_scenes(
        context, model=model, schema="prompts/scene_description_v2.md",
    )
    assert result["failure_count"] == int(truncated)
    directory = context.extraction_dir("description", model, "scenes")
    failure_path = directory / "failures" / f"{cid}.jsonl"
    if truncated:
        assert not (directory / f"{cid}.jsonl").exists()
        assert read_jsonl(failure_path) == [{
            "content_id": cid, "scene_idx": 0,
            "error": "description: output truncated at token limit", "raw_output": "", "provenance": read_jsonl(failure_path)[0]["provenance"],
        }]
    else:
        assert not failure_path.exists()
        records = read_scene_records(directory / f"{cid}.jsonl")
        assert records[0]["description"] == text.strip()


def test_graph_length_cutoff_logs_identity_reason_and_raw_output(ready_context, monkeypatch):
    context = ready_context
    context.config["extraction"]["graph_repetition_penalty"] = [1.0]
    context.config["extraction"]["description_repetition_penalty"] = [1.0]
    visuals = steps.visual_rows(context)[:1]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    scene_dir = context.graph_scene_dir("qwen")
    text = json.dumps({"entities": [], "relations": [], "context": []})
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append(task.task_id)
                kwargs["runtime"].current_result = {"finish_reason": "length", "output_tokens": 1024}
                callback(task.task_id, text)
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    options = {"model": "qwen", "schema": "prompts/scene_graph_v3.md"}
    assert steps.extract_graph_scenes(context, **options)["failure_count"] == 1
    cid = visuals[0]["content_id"]
    assert read_jsonl(scene_dir / "failures" / f"{visuals[0]['content_id']}.jsonl") == [{
        "content_id": cid, "scene_idx": 0, "error": "graph: output truncated at token limit", "raw_output": "", "provenance": read_jsonl(scene_dir / "failures" / f"{cid}.jsonl")[0]["provenance"],
    }]
    assert not (scene_dir / f"{cid}.jsonl").exists()
    assert steps.extract_graph_scenes(context, **options)["failure_count"] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_graph_text_repair_and_cutoff_share_storage_and_failure_policy(
    ready_context, monkeypatch, model,
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
        indices = (range(len(responses)) if kwargs.get("scenes") is None
                   else [row["scene_idx"] for row in kwargs["scenes"]])
        return [{"task": QwenGenerationTask(f"{cid}:{i}", (), kwargs["prompt"], 1024),
                 "scene_idx": i, "keyframes": [i * 30 + 5]} for i in indices]

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
                reason = (FinishReason.MAX_TOKENS if index == 4 else
                          "MAX_TOKENS" if index == 5 else "STOP")
                callback(GeminiGenerationOutcome(key, output, response_diagnostics={
                    "candidates": [{"finish_reason": reason}],
                }))
            receive_tasks(tasks, receive)

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    options = {"model": model, "schema": "prompts/scene_graph_v3.md"}
    assert steps.extract_graph_scenes(context, **options)["failure_count"] == 2
    assert calls == [f"{cid}:{i}" for i in range(len(responses))]
    assert (instances[-1].success, instances[-1].failed, instances[-1].raw) == (5, 2, 0)
    directory = context.graph_scene_dir(model)
    records = read_scene_records(directory / f"{cid}.jsonl")
    by_scene = {row["scene_idx"]: row for row in records}
    assert by_scene[0]["graph"] == by_scene[1]["graph"] == by_scene[2]["graph"]
    assert by_scene[3]["graph"] == responses[3]
    assert by_scene[3]["parse_mode"] == "text"
    assert by_scene[0]["parse_mode"] == "unknown"
    assert by_scene[1]["parse_mode"] == "unknown"
    assert by_scene[0]["provenance"]["prompt_hash"]
    assert by_scene[0]["graph"]["entities"][0]["attributes"] == ["long-haired"]
    assert by_scene[6]["graph"] == {"entities": [], "relations": [], "context": []}
    failures = read_jsonl(directory / "failures" / f"{cid}.jsonl")
    assert {row["scene_idx"] for row in failures} == {4, 5}
    for failure in failures:
        index = failure["scene_idx"]
        assert failure["raw_output"] == ""
        assert index not in by_scene
        if index in (4, 5):
            assert failure["error"] == "graph: output truncated at token limit"
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert observations[0]["observation"] == by_scene[0]["graph"]
    assert len(observations) == 5
    assert observations[4]["observation"] == by_scene[6]["graph"]
    assert steps.extract_graph_scenes(context, **options)["failure_count"] == 2
    expected = [f"{cid}:{i}" for i in range(len(responses))]
    expected.extend(f"{cid}:{i}" for i in (4, 5))
    assert calls == expected
