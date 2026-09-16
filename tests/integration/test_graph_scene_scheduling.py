from contextlib import contextmanager
import json
import threading

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.backends.qwen_workers import QwenGenerationTask
from pipeline_runtime import read_jsonl, write_jsonl
from extraction.scene_storage import read_scene_records
from pipeline_fixtures import context as context


def test_qwen_graph_generates_once_even_with_legacy_penalty_list(
    context, monkeypatch, capsys,
):
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})
    penalties = [1.0, 1.05, 1.1, 1.15, 1.2]
    context.config["extraction"]["graph_repetition_penalty"] = penalties

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [{"task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                 "scene_idx": i, "keyframes": [i * 30 + 5]}
                for i in (range(10 if cid == "a" else 9) if kwargs.get("scenes") is None
                          else (scene["scene_idx"] for scene in kwargs["scenes"]))]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    graph = json.dumps({"entities": [], "relations": [], "context": []})
    submissions = []
    opened = []

    def generate(tasks, callback):
        tasks = list(tasks)
        penalty = penalties[len(submissions)]
        assert {task.repetition_penalty for task in tasks} == {penalty}
        submissions.append({task.task_id for task in tasks})
        for task in reversed(tasks):
            assert task.structured_output is not None
            text = graph
            if task.task_id == "b:8":
                text = ""  # Exhaust the budget without any raw output.
            elif task.task_id == "a:1":
                text = '{"context": []}'  # Exhaust the budget and retain raw output.
            elif task.task_id == "a:0" and penalty < 1.05:
                text = ""
            elif task.task_id == "b:0" and penalty < 1.1:
                text = ""
            callback(task.task_id, text)
        return {}

    @contextmanager
    def generator(**kwargs):
        opened.append(True)
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    assert steps.extract_graph_scenes(
        context, schema="prompts/graph_scene_v3.md", model="qwen",
    )["failure_count"] == 4  # Every first failure is terminal.
    assert submissions == [{f"{cid}:{i}" for cid, count in (("a", 10), ("b", 9))
                            for i in range(count)}]
    assert opened == [True]
    records = {f"{cid}:{row['scene_idx']}": row for cid in ("a", "b")
               for row in read_scene_records(context.graph_scene_dir("qwen") / f"{cid}.jsonl")}
    assert len(records) == 16
    assert records["a:1"]["status"] == "raw_fallback"
    failures = [row for cid in ("a", "b")
                for row in read_jsonl(context.graph_failure_path("qwen", cid))]
    assert {(row["content_id"], row["scene_idx"]) for row in failures} == {
        ("a", 0), ("a", 1), ("b", 0), ("b", 8),
    }
    assert all(set(row) == {"content_id", "scene_idx", "error", "raw_output"} for row in failures)
    assert next(row for row in failures if row["content_id"] == "a" and row["scene_idx"] == 1)["raw_output"] == '{"context": []}'
    assert all(row["raw_output"] == "" for row in failures if (row["content_id"], row["scene_idx"]) != ("a", 1))
    output = capsys.readouterr()
    assert "[Graph_skip_qwen] b.mp4 | scene #008" in output.err
    assert "[RECOVERY]" not in output.err


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_qwen_resume_skips_successful_and_failed_scenes(
    context, monkeypatch, representation,
):
    visuals = [{"content_id": cid} for cid in ("a", "b")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})
    context.config["extraction"][f"{representation}_repetition_penalty"] = [1.0, 1.05, 1.1]

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [{"task": QwenGenerationTask(f"{cid}:{i}", (), "prompt", 32),
                 "scene_idx": i, "keyframes": [i * 30 + 5]}
                for i in (range(2) if kwargs.get("scenes") is None
                          else (scene["scene_idx"] for scene in kwargs["scenes"]))]

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
                callback(task.task_id, graph if task.repetition_penalty >= threshold else "")
                if interrupt and event == (1.0, "a:1"):
                    raise KeyboardInterrupt
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    schema = f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md"
    extract = getattr(steps, f"extract_{representation}_scenes")
    with pytest.raises(KeyboardInterrupt):
        extract(context, model="qwen", schema=schema)
    interrupt = False
    calls.clear()
    assert extract(context, model="qwen", schema=schema)["failure_count"] == 2
    assert calls == [(1.0, "b:0"), (1.0, "b:1")]
    scene_dir = context.extraction_dir(representation, "qwen", "scenes")
    assert not (scene_dir / ".recovery").exists()
    for cid in ("a", "b"):
        records = read_scene_records(scene_dir / f"{cid}.jsonl")
        assert [r["scene_idx"] for r in records] == [1]


@pytest.mark.parametrize("threads", [2, 8, 16])
@pytest.mark.parametrize("representation", ["graph", "description"])
def test_gemini_refills_across_contents_and_saves_finished_contents(
    context, monkeypatch, threads, representation,
):
    from extraction.backends.gemini_workers import GeminiWorkerPool
    from extraction import scene_executor

    scene_dir = context.extraction_dir(representation, "gemini", "scenes")
    context.config["extraction"]["gemini"]["threads"] = threads
    visuals = [{"content_id": str(i)} for i in range(threads + 3)]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        return [{"task": QwenGenerationTask(f"{cid}:0", (), cid, 32),
                 "scene_idx": 0, "keyframes": [5]}]

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

    monkeypatch.setattr(steps, "GeminiWorkerPool", lambda *args, **kwargs: GeminiWorkerPool(
        *args, **kwargs, backend_factory=Backend))
    saves = []
    save_results = scene_executor.write_scene_results

    def save(scene_path, records):
        assert len(records) <= 1
        saves.append(scene_path.stem)
        save_results(scene_path, records)

    monkeypatch.setattr(scene_executor, "write_scene_results", save)
    extract = getattr(steps, f"extract_{representation}_scenes")
    schema = f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md"
    assert extract(context, schema=schema, model="gemini")["failure_count"] == 1
    assert peak == threads
    assert next_content_started.is_set()
    assert sorted(saves, key=int) == [str(i) for i in range(threads + 3)]
    assert not (scene_dir / ".pending-contents.json").exists()


@pytest.mark.parametrize("force", [False, True])
def test_gemini_interrupt_preserves_each_completed_scene(
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
                 "scene_idx": i, "keyframes": [i * 30 + 5]}
                for i in (range(3) if kwargs.get("scenes") is None
                          else (row["scene_idx"] for row in kwargs["scenes"]))]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    if force:
        for visual in visuals:
            write_jsonl(scene_dir / f"{visual['content_id']}.jsonl",
                        [raw_graph_record(row, "old") for row in rows(visual)])
    interrupted = True
    calls = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            tasks = list(tasks)
            calls.append([task.task_id for task in tasks])
            for task in tasks:
                callback(GeminiGenerationOutcome(task.task_id, "new observation"))
                if interrupted and task.task_id == "b:0":
                    raise KeyboardInterrupt

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, schema="prompts/graph_scene_v3.md", model="gemini", force=force)
    saved_a = (scene_dir / "a.jsonl").read_bytes()
    assert [row["scene_idx"] for row in read_scene_records(scene_dir / "b.jsonl")] == [0]
    assert not (scene_dir / ".recovery").exists()
    interrupted = False
    assert steps.extract_graph_scenes(context, schema="prompts/graph_scene_v3.md", model="gemini")["failure_count"] == 9
    assert calls == [[f"{cid}:{i}" for cid in ("a", "b", "c") for i in range(3)],
                     ["b:1", "b:2", "c:0", "c:1", "c:2"]]
    assert (scene_dir / "a.jsonl").read_bytes() == saved_a
    assert not (scene_dir / ".pending-contents.json").exists()
    for cid in ("b", "c"):
        assert "old" not in str(read_jsonl(scene_dir / f"{cid}.jsonl"))


@pytest.mark.parametrize("model", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["graph", "description"])
def test_scene_total_is_known_before_streaming_inference(
    context, monkeypatch, capsys, model, representation,
):
    visuals = [{"content_id": cid} for cid in ("a", "b", "c")]
    monkeypatch.setattr(steps, "visual_rows", lambda _: visuals)
    monkeypatch.setattr(steps, "video_name_map", lambda _: {})
    completed = []
    scene_dir = context.extraction_dir(representation, model, "scenes")
    opened = []
    prepared = []
    progress_instances = []
    now = [0.0]
    original_progress = steps.InferenceProgress

    def progress_factory(**kwargs):
        assert kwargs["total"] == 3
        progress = original_progress(**kwargs, clock=lambda: now[0])
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(steps, "InferenceProgress", progress_factory)

    def rows(visual, **kwargs):
        cid = visual["content_id"]
        if len(prepared) < len(visuals):
            assert not opened
            prepared.append(cid)
        elif cid != "a":
            previous = chr(ord(cid) - 1)
            assert previous in completed
            assert (scene_dir / f"{previous}.jsonl").is_file()
        return [{"task": QwenGenerationTask(f"{cid}:0", (), "prompt", 32),
                 "scene_idx": 0, "keyframes": [5]}]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    response = json.dumps({"entities": [], "relations": [], "context": []})

    def generate(tasks, callback):
        assert prepared == ["a", "b", "c"]
        progress = progress_instances[0]
        assert progress.total == progress.bar.total == 3
        for task in tasks:
            now[0] += 1
            callback(task.task_id, response)
            completed.append(task.task_id.split(":")[0])
            progress._render(refresh=True)
            assert progress.bar.n == len(completed)
            assert "ETA=estimating" not in progress.bar.postfix
            assert "/?" not in str(progress.bar)
        return {}

    @contextmanager
    def generator(**kwargs):
        opened.append(True)
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            opened.append(True)

        def generate(self, tasks, callback, **kwargs):
            return generate(tasks, lambda task_id, text: callback(
                GeminiGenerationOutcome(task_id, text)))

    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    extract = getattr(steps, f"extract_{representation}_scenes")
    schema = f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md"
    assert extract(context, model=model, schema=schema)["failure_count"] == 0
    assert completed == ["a", "b", "c"]
    assert opened == [True]
    output = capsys.readouterr()
    assert "Prepare scene tasks" not in output.out + output.err


@pytest.mark.parametrize("force", [False, True])
def test_gemini_resume_preserves_out_of_order_completions_across_interrupts(
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
                 "scene_idx": i, "keyframes": [i * 30 + 5]}
                for i in (range(2) if kwargs.get("scenes") is None
                          else (row["scene_idx"] for row in kwargs["scenes"]))]

    monkeypatch.setattr(steps, "scene_generation_rows", rows)
    if force:
        for visual in visuals:
            write_jsonl(scene_dir / f"{visual['content_id']}.jsonl",
                        [raw_graph_record(row, "old") for row in rows(visual)])
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
                    callback(GeminiGenerationOutcome(task_id, "new observation"))
                raise KeyboardInterrupt
            for task in tasks:
                callback(GeminiGenerationOutcome(task.task_id, "new observation"))

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    schema = "prompts/graph_scene_v3.md"
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, schema=schema, model="gemini", force=force)
    saved_b = (scene_dir / "b.jsonl").read_bytes()
    run = 1
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(context, schema=schema, model="gemini")
    run = 2
    assert steps.extract_graph_scenes(context, schema=schema, model="gemini")["failure_count"] == 6
    assert submitted[-1] == ["a:1", "c:0", "c:1"]
    assert (scene_dir / "b.jsonl").read_bytes() == saved_b
    assert not (scene_dir / ".pending-contents.json").exists()
