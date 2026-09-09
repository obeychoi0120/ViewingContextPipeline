from __future__ import annotations

from contextlib import contextmanager
import json

import pytest

import extraction.scene_executor as scene_executor
import extraction.steps as steps
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.qwen_runtime import QwenRuntimeLog
from pipeline_runtime import read_json, read_jsonl, write_jsonl
from pipeline_fixtures import context as context


@pytest.mark.parametrize("arm", ["graph", "description"])
def test_scene_progress_advances_before_video_finishes(context, monkeypatch, arm):
    context.initialize()
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "_visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "_video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        return [{"task": QwenGenerationTask(f"{visual['content_id']}:{i}", (), "prompt", 32),
                 "scene_idx": i, "keyframes": [i * 30 + 5]}
                for i in range(2 if visual["content_id"] == "a" else 1)]

    monkeypatch.setattr(steps, "_scene_generation_rows", rows)
    bars = []
    factory = steps.InferenceProgress

    def progress(**kwargs):
        value = factory(**kwargs)
        bars.append(value)
        return value

    monkeypatch.setattr(steps, "InferenceProgress", progress)

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            assert bars[-1].bar.total == 3
            for index, task_id in enumerate(("a:0", "b:0", "a:1")):
                good = json.dumps({"setting_context": "indoor", "entities": [], "events": [], "semantic_topics": [], "affect": {"valence": "neutral", "arousal": "medium"}}) if arm == "graph" else "A person indoors."
                bad = ""
                callback(task_id, bad if index == 0 else good)
                assert bars[-1].bar.n == index + 1
                assert bars[-1].failed == 1
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    result = (steps.extract_graph_scenes(context, model="qwen") if arm == "graph"
              else steps.extract_description_scenes(context))
    assert result["failure_count"] == 1
    assert bars[-1].success == 2 and bars[-1].failed == 1


@pytest.mark.parametrize("arm", ["graph", "description"])
@pytest.mark.parametrize("failure", ["oom", "save"])
def test_cross_video_completion_checkpoint_and_resume(context, monkeypatch, arm, failure):
    context.initialize()
    bars = []
    progress_factory = steps.InferenceProgress

    def progress(**kwargs):
        value = progress_factory(**kwargs)
        bars.append(value)
        return value

    monkeypatch.setattr(steps, "InferenceProgress", progress)
    visuals = [{"content_id": content_id} for content_id in ("a", "b")]
    monkeypatch.setattr(steps, "_visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "_video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        return [{"task": QwenGenerationTask(f"{visual['content_id']}:{i}", (), "prompt", 32),
                 "scene_idx": i + 10, "keyframes": [i * 30 + 5]}
                for i in range(2 if visual["content_id"] == "a" else 1)]

    monkeypatch.setattr(steps, "_scene_generation_rows", rows)
    graph = {"setting_context": "indoor", "entities": [], "events": [],
             "semantic_topics": [], "affect": {"valence": "neutral", "arousal": "medium"}}
    text = json.dumps(graph) if arm == "graph" else "a person indoors"
    first_row = rows(visuals[0])[0]
    cached, _ = (scene_executor.graph_scene_result(first_row, text) if arm == "graph"
                 else scene_executor.description_scene_result(first_row, text, content_id="a"))
    scene_dir = context.graph_scene_dir("qwen") if arm == "graph" else context.description_scene_dir
    failure_dir = context.graph_failure_dir("qwen") if arm == "graph" else context.description_failure_dir
    stage = "extract-graph-scenes-qwen" if arm == "graph" else "extract-description-scenes"
    write_jsonl(scene_dir / "a.jsonl", [cached])
    write_jsonl(failure_dir / "a.jsonl", [{"scene_idx": 11, "keyframes": [35],
                                        "failure_kind": "generation", "error": "old failure", "raw_response": ""}])
    original = (scene_dir / "a.jsonl").read_bytes()
    checkpoint = scene_executor.write_scene_checkpoint
    attempt = 0

    def maybe_fail(path, *args):
        if attempt == 0 and failure == "save" and path.name == "a.jsonl":
            raise OSError("disk full")
        return checkpoint(path, *args)

    monkeypatch.setattr(scene_executor, "write_scene_checkpoint", maybe_fail)

    @contextmanager
    def generator(**kwargs):
        runtime = kwargs["runtime"]
        runtime.engine_ready({"worker_index": 0, "gpu_id": "0", "backend": "vllm"})

        def generate(tasks, callback):
            assert [task.task_id for task in tasks] == (["a:1", "b:0"] if attempt == 0 else ["a:1"])
            for task in reversed(tasks):
                if attempt == 0 and task.task_id == "a:1" and failure == "oom":
                    raise RuntimeError("CUDA out of memory")
                runtime.current_result = {"task_id": task.task_id, "worker_index": 0, "gpu_id": "0",
                                          "prompt_tokens": 10, "output_tokens": 3}
                callback(task.task_id, text)
                assert bars[-1].bar.n == bars[-1].success + bars[-1].failed
                runtime.current_result = None
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)

    def run():
        if arm == "graph":
            return steps.extract_graph_scenes(context, model="qwen")
        return steps.extract_description_scenes(context)

    with pytest.raises((RuntimeError, OSError), match="out of memory|disk full"):
        run()
    assert bars[-1].total == 2 and bars[-1].reused == 1 and bars[-1].bar.n == 1
    assert (scene_dir / "a.jsonl").read_bytes() == original
    completed_b = read_jsonl(scene_dir / "b.jsonl")[0]
    runtime = QwenRuntimeLog(context.run_root, stage)
    assert runtime.classification("b:10", completed_b) == "vllm"
    assert runtime.classification("a:10", cached) == "legacy_unknown"
    attempt = 1
    assert run()["failure_count"] == 0
    assert bars[-1].total == bars[-1].bar.n == 1 and bars[-1].reused == 2
    assert not (failure_dir / "a.jsonl").exists()
    records = read_jsonl(scene_dir / "a.jsonl")
    assert records[0] == cached
    assert [row["scene_idx"] for row in records] == [10, 11]
    assert QwenRuntimeLog(context.run_root, stage).classification("a:11", records[1]) == "vllm"
    # A completely cached run must not construct any engine.
    monkeypatch.setattr(steps, "qwen_generator", lambda **kwargs: pytest.fail("unexpected inference"))
    assert run()["failure_count"] == 0


@pytest.mark.parametrize("source", ["qwen", "gemini", "description"])
def test_summary_provenance_and_resume_after_interruption(context, monkeypatch, source):
    from extraction.summary_validation import SUMMARY_SECTIONS

    context.initialize()
    description = source == "description"
    scene_dir = context.description_scene_dir if description else context.graph_scene_dir(source)
    summary_dir = context.description_summary_dir if description else context.graph_summary_dir(source)
    stage = "summarize-description" if description else f"summarize-graph-{source}"
    monkeypatch.setattr(steps, "_visual_rows", lambda _: [{"content_id": cid} for cid in ("a", "b")])
    monkeypatch.setattr(steps, "_video_name_map", lambda _: {})
    for cid in ("a", "b"):
        row = {"scene_idx": 0, "keyframes": [5]}
        if description:
            row.update(schema_version="scene-description/v1", content_id=cid, description="A room.")
        else:
            row.update(graph={"setting_context": "indoor"}, parse_mode="native", semantic_warnings=[])
        write_jsonl(scene_dir / f"{cid}.jsonl", [row])
    submissions = []
    text = "\n".join(f"{section}: " + ("A room." if index == 0 else "")
                     for index, section in enumerate(SUMMARY_SECTIONS))

    @contextmanager
    def generator(**kwargs):
        runtime = kwargs["runtime"]
        runtime.engine_ready({"worker_index": 0, "gpu_id": "0", "backend": "vllm"})

        def generate(tasks, callback):
            submissions.append([task.task_id for task in tasks])
            for task in reversed(tasks):
                if len(submissions) == 1 and task.task_id == "a":
                    raise KeyboardInterrupt
                runtime.current_result = {"task_id": task.task_id, "worker_index": 0,
                                          "gpu_id": "0", "prompt_tokens": 10, "output_tokens": 3}
                callback(task.task_id, text)
                runtime.current_result = None
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)

    def run():
        return steps.summarize_description(context) if description else steps.summarize_graph(context, source=source)

    with pytest.raises(KeyboardInterrupt):
        run()
    saved = (summary_dir / "b.json").read_bytes()
    assert not (summary_dir / "a.json").exists()
    assert QwenRuntimeLog(context.run_root, stage).classification("b", read_json(summary_dir / "b.json")) == "vllm"
    assert run()["content_count"] == 2
    assert submissions == [["a", "b"], ["a"]]
    assert (summary_dir / "b.json").read_bytes() == saved
    monkeypatch.setattr(steps, "qwen_generator", lambda **kwargs: pytest.fail("unexpected inference"))
    assert run()["content_count"] == 2
