from contextlib import contextmanager
import json

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.scene_storage import read_scene_records
from pipeline_runtime import read_jsonl


@pytest.mark.parametrize("model", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["description", "graph"])
def test_first_scene_failure_is_terminal_minimal_and_skipped_on_resume(
    ready_context, monkeypatch, model, representation,
):
    context = ready_context
    visuals = steps.visual_rows(context)[:2]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    context.config["extraction"][f"{representation}_repetition_penalty"] = [1.0, 1.05, 1.1]
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

    def generate(tasks, callback):
        for task in tasks:
            calls.append((task.task_id, task.repetition_penalty))
            text = good if succeed or task.task_id == other_id else " \n\t"
            callback(task.task_id, text)
            if interrupt:
                assert read_jsonl(failure_path) == [{
                    "content_id": visuals[0]["content_id"], "scene_idx": 0,
                    "error": read_jsonl(failure_path)[0]["error"], "raw_output": text,
                }]
                progress = instances[-1]
                assert (progress.success, progress.failed) == (0, 1)
                for name in (".recovery", ".pending", ".checkpoints", ".pending-contents.json"):
                    assert not (scene_dir / name).exists()
                raise KeyboardInterrupt
        return {}

    @contextmanager
    def generator(**kwargs):
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, tasks, callback, **kwargs):
            return generate(tasks, lambda key, text: callback(GeminiGenerationOutcome(key, text)))

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    extract = getattr(steps, f"extract_{representation}_scenes")
    options = {"model": model, "schema": f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md"}
    with pytest.raises(KeyboardInterrupt):
        extract(context, **options)
    interrupt = False
    assert extract(context, **options)["failure_count"] == 1
    assert calls == [(first_id, 1.0), (other_id, 1.0)]
    assert extract(context, **options)["failure_count"] == 1
    assert len(calls) == 2
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
    assert len(calls) == 4


def test_graph_length_cutoff_logs_identity_reason_and_raw_output(ready_context, monkeypatch):
    context = ready_context
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
    options = {"model": "qwen", "schema": "prompts/graph_scene_v3.md"}
    assert steps.extract_graph_scenes(context, **options)["failure_count"] == 1
    cid = visuals[0]["content_id"]
    assert read_jsonl(scene_dir / "failures" / f"{visuals[0]['content_id']}.jsonl") == [{
        "content_id": cid, "scene_idx": 0, "error": "graph: output truncated at token limit", "raw_output": text,
    }]
    raw = read_scene_records(scene_dir / f"{cid}.jsonl")[0]
    assert raw["raw_response"] == text and raw["status"] == "raw_fallback"
    assert steps.extract_graph_scenes(context, **options)["failure_count"] == 1
    assert len(calls) == 1
