from contextlib import contextmanager
import json
import threading

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.backends.qwen_workers import QwenGenerationTask
from pipeline_runtime import read_jsonl, write_jsonl
from pipeline_fixtures import context as context


@pytest.mark.parametrize(("model", "threads", "limit"), [
    ("qwen", 16, 8), ("gemini", 1, 1), ("gemini", 8, 8), ("gemini", 16, 16),
])
def test_graph_scenes_finish_content_before_next(
    context, monkeypatch, capsys, model, threads, limit,
):
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})
    context.config["extraction"]["graph_repetition_penalty"] = [1.0, 1.05]

    context.config["extraction"]["gemini"]["threads"] = threads

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [{"task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                 "scene_idx": i, "keyframes": [i * 30 + 5]}
                for i in range(limit + 2 if cid == "a" else limit + 1)]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    graph = json.dumps({
        "entities": [], "relations": [], "context": [],
    })
    submissions = []

    def generate(tasks, callback):
        tasks = list(tasks)
        submissions.append([task.task_id for task in tasks])
        if model == "qwen":
            assert 1 <= len(tasks) <= limit
        assert len({task.task_id.split(":")[0] for task in tasks}) == 1
        if tasks[0].task_id.startswith("b:"):
            assert len(read_jsonl(context.graph_scene_dir(model) / "a.jsonl")) == limit + 2
        for task in reversed(tasks):
            text = "" if task.task_id == f"b:{limit}" or (
                model == "qwen" and task.task_id == "a:0" and task.repetition_penalty == 1.0
            ) else graph
            callback(task.task_id, text)
        return {}

    @contextmanager
    def generator(**kwargs):
        yield generate

    class Pool:
        def __init__(self, concurrency, **kwargs):
            assert concurrency == threads

        def generate(self, tasks, callback, **kwargs):
            return generate(tasks, lambda task_id, text: callback(
                GeminiGenerationOutcome(task_id, text)))

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    assert steps.extract_graph_scenes(context, schema="prompts/graph_scene_v3.md", model=model)["failure_count"] == 1
    expected = [[f"a:{i}" for i in range(limit)]]
    if model == "qwen":
        expected.append(["a:0"])
    expected += [[f"a:{i}" for i in range(start, min(start + limit, limit + 2))]
                 for start in range(limit, limit + 2, limit)]
    expected += [[f"b:{i}" for i in range(limit)], [f"b:{limit}"]]
    if model == "qwen":
        expected.append([f"b:{limit}"])
    else:
        expected = [[f"a:{i}" for i in range(limit + 2)],
                    [f"b:{i}" for i in range(limit + 1)]]
    assert submissions == expected
    assert len(read_jsonl(context.graph_scene_dir(model) / "b.jsonl")) == limit
    failures = read_jsonl(context.graph_failure_dir(model) / "b.jsonl")
    assert [row["scene_idx"] for row in failures] == [limit]
    output = capsys.readouterr()
    assert f"[Graph_skip_{model}] b.mp4 | scene #{limit:03d}" in output.err
    assert f"[Graph_{model}]" not in output.err + output.out
    assert "setting_context" not in output.err + output.out


@pytest.mark.parametrize("threads", [2, 8, 16])
def test_gemini_refills_workers_and_saves_only_finished_contents(
    context, monkeypatch, threads,
):
    from extraction.backends.gemini_workers import GeminiWorkerPool
    from extraction import scene_executor

    scene_dir = context.graph_scene_dir("gemini")
    context.config["extraction"]["gemini"]["threads"] = threads
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})
    counts = {"a": threads + 3, "b": 2}

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [{"task": QwenGenerationTask(f"{cid}:{i}", (), f"{cid}:{i}", 32),
                 "scene_idx": i, "keyframes": [i * 30 + 5]} for i in range(counts[cid])]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    initial_workers = threading.Barrier(threads, timeout=5)
    next_scene_started = threading.Event()
    lock = threading.Lock()
    active = peak = 0

    class Backend:
        def generate(self, images, prompt, max_new_tokens):
            nonlocal active, peak
            cid, index = prompt.split(":")
            index = int(index)
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if cid == "a":
                    assert not (scene_dir / "a.jsonl").exists()
                    assert not (scene_dir / ".recovery").exists()
                    if index < threads:
                        initial_workers.wait()
                    if index == 0:
                        # A request beyond the old batch boundary must start
                        # before this slow first request is allowed to finish.
                        assert next_scene_started.wait(timeout=5)
                    elif index == threads:
                        next_scene_started.set()
                    elif index == 1:
                        raise RuntimeError("one failed scene must not stall its worker")
                else:
                    assert len(read_jsonl(scene_dir / "a.jsonl")) == counts["a"] - 1
                return json.dumps({"setting_context": "indoor"})
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(steps, "GeminiWorkerPool", lambda *args, **kwargs: GeminiWorkerPool(
        *args, **kwargs, backend_factory=Backend))
    saves = []
    checkpoint = scene_executor.write_scene_checkpoint

    def save(scene_path, failure_path, records, failures):
        assert active == 0
        assert len(records) + len(failures) == counts[scene_path.stem]
        saves.append(scene_path.stem)
        checkpoint(scene_path, failure_path, records, failures)

    monkeypatch.setattr(scene_executor, "write_scene_checkpoint", save)
    assert steps.extract_graph_scenes(context, schema="prompts/graph_scene_v3.md", model="gemini")["failure_count"] == 1
    assert peak == threads
    assert next_scene_started.is_set()
    assert saves == ["a", "b"]
    assert not (scene_dir / ".recovery").exists()
    assert not (scene_dir / ".pending-contents.json").exists()


@pytest.mark.parametrize("force", [False, True])
def test_gemini_interrupt_discards_memory_and_restarts_unfinished_contents(
    context, monkeypatch, force,
):
    from extraction.raw_output import raw_graph_record

    scene_dir = context.graph_scene_dir("gemini")
    visuals = [{"content_id": cid} for cid in ("a", "b", "c")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [{"task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                 "scene_idx": i, "keyframes": [i * 30 + 5]} for i in range(3)]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    if force:
        for visual in visuals:
            write_jsonl(scene_dir / f"{visual['content_id']}.jsonl",
                        [raw_graph_record(row, "old") for row in rows(visual)])
    before_b = (scene_dir / "b.jsonl").read_bytes() if force else None
    interrupted = True
    calls = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            calls.append([task.task_id for task in tasks])
            for task in tasks:
                callback(GeminiGenerationOutcome(task.task_id, "new observation"))
                if interrupted and task.task_id == "b:0":
                    raise KeyboardInterrupt

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, schema="prompts/graph_scene_v3.md", model="gemini", force=force)
    saved_a = (scene_dir / "a.jsonl").read_bytes()
    assert ((scene_dir / "b.jsonl").read_bytes() if force else None) == before_b
    if not force:
        assert not (scene_dir / "b.jsonl").exists()
    assert not (scene_dir / ".recovery").exists()
    interrupted = False
    assert steps.extract_graph_scenes(context, schema="prompts/graph_scene_v3.md", model="gemini")["failure_count"] == 0
    assert calls == [[f"{cid}:{i}" for i in range(3)] for cid in ("a", "b", "b", "c")]
    assert (scene_dir / "a.jsonl").read_bytes() == saved_a
    assert not (scene_dir / ".pending-contents.json").exists()
    for cid in ("b", "c"):
        assert "old" not in str(read_jsonl(scene_dir / f"{cid}.jsonl"))
