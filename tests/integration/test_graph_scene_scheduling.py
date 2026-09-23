from contextlib import contextmanager
import json
import threading

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.scene_storage import read_scene_records


def test_qwen_resume_reuses_successes_and_retries_failures_from_first_penalty(
    context,
    monkeypatch,
):
    representation = "graph"
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})
    context.config["extraction"][f"{representation}_repetition_penalty"] = [1.0, 1.05, 1.1]

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [
            {
                "task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                "scene_idx": i,
                "keyframes": [i * 30 + 5],
            }
            for i in (
                range(2)
                if kwargs.get("scenes") is None
                else (scene["scene_idx"] for scene in kwargs["scenes"])
            )
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    graph = json.dumps({"entities": [], "relations": [], "context": []})
    interrupt = True
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                event = (task.repetition_penalty, task.task_id)
                calls.append(event)
                threshold = {"a:0": 1.1, "b:0": 1.05}.get(task.task_id, 1.0)
                kwargs["runtime"].current_result = {
                    "finish_reason": "stop" if task.repetition_penalty >= threshold else "length"
                }
                callback(task.task_id, graph if task.repetition_penalty >= threshold else "")
                if interrupt and event == (1.0, "a:1"):
                    raise KeyboardInterrupt
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    schema = f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md"
    extract = getattr(steps, f"extract_{representation}_scenes")
    with pytest.raises(KeyboardInterrupt):
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_qwen",
            model="qwen",
            schema=schema,
        )
    interrupt = False
    calls.clear()
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_qwen",
            model="qwen",
            schema=schema,
        )["failure_count"]
        == 0
    )
    assert calls == [
        (1.0, "a:0"),
        (1.0, "b:0"),
        (1.0, "b:1"),
        (1.05, "a:0"),
        (1.05, "b:0"),
        (1.1, "a:0"),
    ]
    scene_dir = context.extraction_dir(representation, "qwen", "scenes")
    assert not (scene_dir / ".recovery").exists()
    for cid in ("a", "b"):
        records = read_scene_records(scene_dir / f"{cid}.jsonl")
        assert [r["scene_idx"] for r in records] == [0, 1]


def test_gemini_refills_across_contents_and_saves_finished_contents(
    context,
    monkeypatch,
):
    threads = 2
    representation = "graph"
    from extraction.backends.gemini_workers import GeminiWorkerPool
    from extraction import scene_executor

    scene_dir = context.extraction_dir(representation, "gemini", "scenes")
    context.config["extraction"]["gemini"]["threads"] = threads
    visuals = [{"content_id": str(i)} for i in range(threads + 3)]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [
            {"task": QwenGenerationTask(f"{cid}:0", (), cid, 32), "scene_idx": 0, "keyframes": [5]}
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    initial_workers = threading.Barrier(threads, timeout=5)
    next_content_started = threading.Event()
    lock = threading.Lock()
    active = peak = 0

    class Backend:
        def generate(self, images, prompt, max_new_tokens):
            nonlocal active, peak
            index = int(prompt)
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if index < threads:
                    initial_workers.wait()
                if index == 0:
                    # Later videos must run while the first video is still blocked.
                    assert next_content_started.wait(timeout=5)
                elif index == threads:
                    next_content_started.set()
                elif index == 1:
                    raise RuntimeError("one failed scene must not stall its worker")
                return json.dumps({"entities": [], "relations": [], "context": []})
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(
        steps,
        "GeminiWorkerPool",
        lambda *args, **kwargs: GeminiWorkerPool(*args, **kwargs, backend_factory=Backend),
    )
    saves = []
    save_results = scene_executor.write_scene_results

    def save(scene_path, records):
        assert len(records) <= 1
        saves.append(scene_path.stem)
        save_results(scene_path, records)

    monkeypatch.setattr(scene_executor, "write_scene_results", save)
    extract = getattr(steps, f"extract_{representation}_scenes")
    schema = f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md"
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{'gemini'}",
            schema=schema,
            model="gemini",
        )["failure_count"]
        == 1
    )
    assert peak == threads
    assert next_content_started.is_set()
    assert sorted(saves, key=int) == [str(i) for i in range(threads + 3)]
    assert not (scene_dir / ".pending-contents.json").exists()


def test_gemini_resume_preserves_out_of_order_completions_across_interrupts(
    context,
    monkeypatch,
):
    scene_dir = context.graph_scene_dir("gemini")
    visuals = [{"content_id": cid} for cid in ("a", "b", "c")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [
            {
                "task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                "scene_idx": i,
                "keyframes": [i * 30 + 5],
            }
            for i in (
                range(2)
                if kwargs.get("scenes") is None
                else (row["scene_idx"] for row in kwargs["scenes"])
            )
        ]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    run = 0
    submitted = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            if run == 1:
                # Interrupt again before discovery reaches the already saved b.
                next(iter(tasks))
                raise KeyboardInterrupt
            tasks = list(tasks)
            submitted.append([task.task_id for task in tasks])
            if run == 0:
                for task_id in ("b:1", "b:0", "a:0"):
                    callback(
                        GeminiGenerationOutcome(
                            task_id, '{"entities": [], "relations": [], "context": []}'
                        )
                    )
                raise KeyboardInterrupt
            for task in tasks:
                callback(
                    GeminiGenerationOutcome(
                        task.task_id, '{"entities": [], "relations": [], "context": []}'
                    )
                )

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    schema = "prompts/scene_graph_v4.md"
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, arm="graph_gemini", schema=schema, model="gemini")
    saved_b = (scene_dir / "b.jsonl").read_bytes()
    run = 1
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, arm="graph_gemini", schema=schema, model="gemini")
    run = 2
    assert (
        steps.extract_graph_scenes(context, arm="graph_gemini", schema=schema, model="gemini")[
            "failure_count"
        ]
        == 0
    )
    assert submitted[-1] == ["a:1", "c:0", "c:1"]
    assert (scene_dir / "b.jsonl").read_bytes() == saved_b
    assert not (scene_dir / ".pending-contents.json").exists()
