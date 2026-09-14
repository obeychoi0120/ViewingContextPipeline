from __future__ import annotations

from contextlib import contextmanager
import json

import pytest

import extraction.scene_executor as scene_executor
import extraction.steps as steps
from extraction.backends.qwen_workers import QwenGenerationTask
from pipeline_runtime import read_jsonl, write_jsonl
from pipeline_fixtures import context as context


def test_gemini_graph_summary_streams_past_512_before_retrying(context, monkeypatch, capsys):
    from extraction.summary_validation import SUMMARY_SECTIONS

    context.config["schema_version"] = "viewing-context-config/v4"
    context.config["extraction"]["summary_repetition_penalty"] = [1, 1.05, 1.1]
    context.config["extraction"]["qwen"]["max_num_seqs"] = 128
    content_ids = [f"c{i}" for i in range(1025)]
    monkeypatch.setattr(steps, "_visual_rows", lambda _: [{"content_id": cid} for cid in content_ids])
    monkeypatch.setattr(steps, "_video_name_map", lambda _: {})
    scene_dir = context.graph_scene_dir("gemini")
    summary_dir = context.graph_summary_dir("gemini")
    for cid in content_ids:
        write_jsonl(scene_dir / f"{cid}.jsonl", [{
            "scene_idx": 0, "keyframes": [5], "graph": {"setting_context": "indoor"},
            "parse_mode": "native", "semantic_warnings": [],
        }])
    text = "\n".join(f"{name}: A room." for name in SUMMARY_SECTIONS)
    submissions = []
    pools = []

    @contextmanager
    def generator(**kwargs):
        pools.append(True)

        def generate(tasks, callback):
            iterator = iter(tasks)
            first = next(iterator)
            submissions.append([(first.task_id, first.repetition_penalty)])
            # Hold the first request while others complete, crossing the old 512 boundary.
            for task in iterator:
                submissions[-1].append((task.task_id, task.repetition_penalty))
                callback(task.task_id, text)
                assert (summary_dir / f"{task.task_id}.json").is_file()
                assert not (summary_dir / "c0.json").exists()
            if first.repetition_penalty > 1:
                assert len(submissions[0]) == len(content_ids)
            callback(first.task_id, text if first.repetition_penalty == 1.1 else "")
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    result = steps.summarize_graph(context, source="gemini", gpus=1)
    assert result["content_count"] == len(content_ids) and result["failure_count"] == 0
    assert submissions == [[(cid, 1) for cid in content_ids], [("c0", 1.05)], [("c0", 1.1)]]
    assert pools == [True]
    assert not (summary_dir / ".recovery").exists()
    stderr = capsys.readouterr().err
    assert "preparing tasks" not in stderr and "[RETRY]" not in stderr
    assert "[RECOVERY] pass=2/3 repetition_penalty=1.05 pending=1" in stderr
    assert stderr.rstrip().endswith("1025/1025 Done")


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
            tasks = list(tasks)
            assert bars[-1].bar.total == 3
            task_ids = ([task.task_id for task in tasks] if arm == "graph"
                        else ["a:0", "b:0", "a:1"])
            for task_id in task_ids:
                previous = bars[-1].success + bars[-1].failed
                good = json.dumps({"setting_context": "indoor", "entities": [], "events": [], "semantic_topics": [], "affect": {"valence": "neutral", "arousal": "medium"}}) if arm == "graph" else "A person indoors."
                bad = ""
                callback(task_id, bad if task_id == "a:0" else good)
                assert bars[-1].success + bars[-1].failed == previous + 1
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
    order = ("b", "a") if arm == "graph" else ("a", "b")
    visuals = [{"content_id": content_id} for content_id in order]
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
    first_row = rows({"content_id": "a"})[0]
    cached, _ = (scene_executor.graph_scene_result(first_row, text) if arm == "graph"
                 else scene_executor.description_scene_result(first_row, text, content_id="a"))
    scene_dir = context.graph_scene_dir("qwen") if arm == "graph" else context.description_scene_dir
    failure_dir = context.graph_failure_dir("qwen") if arm == "graph" else context.description_failure_dir
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
            tasks = list(tasks)
            if arm == "graph":
                assert len(tasks) == 1
                assert tasks[0].task_id in {"a:1", "b:0"}
                if tasks[0].task_id == "a:1":
                    assert (scene_dir / "b.jsonl").is_file()
            else:
                assert [task.task_id for task in tasks] == (
                    ["a:1", "b:0"] if attempt == 0 else ["a:1"])
            for task in reversed(tasks):
                if attempt == 0 and task.task_id == "a:1" and failure == "oom":
                    raise RuntimeError("CUDA out of memory")
                runtime.current_result = {"task_id": task.task_id, "worker_index": 0, "gpu_id": "0",
                                          "prompt_tokens": 10, "output_tokens": 3}
                callback(task.task_id, text)
                assert bars[-1].bar.n <= bars[-1].success + bars[-1].failed
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
    assert read_jsonl(scene_dir / "b.jsonl")[0]["scene_idx"] == 10
    assert not (context.run_root / "extraction" / "qwen_runtime.jsonl").exists()
    attempt = 1
    assert run()["failure_count"] == 0
    assert bars[-1].total == bars[-1].bar.n == 1 and bars[-1].reused == 2
    assert not (failure_dir / "a.jsonl").exists()
    records = read_jsonl(scene_dir / "a.jsonl")
    assert records[0] == cached
    assert [row["scene_idx"] for row in records] == [10, 11]
    assert not (context.run_root / "extraction" / "qwen_runtime.jsonl").exists()
    # A completely cached run must not construct any engine.
    monkeypatch.setattr(steps, "qwen_generator", lambda **kwargs: pytest.fail("unexpected inference"))
    assert run()["failure_count"] == 0


@pytest.mark.parametrize("source", ["qwen", "gemini", "description"])
@pytest.mark.parametrize("existing_log", [False, True])
def test_summary_resume_after_interruption_without_runtime_log(
    context, monkeypatch, source, existing_log,
):
    from extraction.summary_validation import SUMMARY_SECTIONS

    context.initialize()
    runtime_path = context.run_root / "extraction" / "qwen_runtime.jsonl"
    if existing_log:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_bytes(b"corrupt legacy runtime log\n")
    description = source == "description"
    scene_dir = context.description_scene_dir if description else context.graph_scene_dir(source)
    summary_dir = context.description_summary_dir if description else context.graph_summary_dir(source)
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
            tasks = list(tasks)
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
    assert run()["content_count"] == 2
    assert submissions == [["a", "b"], ["a"]]
    assert (summary_dir / "b.json").read_bytes() == saved
    monkeypatch.setattr(steps, "qwen_generator", lambda **kwargs: pytest.fail("unexpected inference"))
    assert run()["content_count"] == 2
    if existing_log:
        assert runtime_path.read_bytes() == b"corrupt legacy runtime log\n"
    else:
        assert not runtime_path.exists()
