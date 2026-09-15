from dataclasses import replace
import json

import pytest

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.recovery import generate_with_recovery, penalty_schedule
from extraction.structured_output import GRAPH_JSON_SCHEMA, OutputValidationError



def graph():
    return {"entities": [], "relations": [], "context": []}


@pytest.mark.parametrize("value", [[], [True], float("nan"), [float("inf")], [0.99], [2.01], "1"])
def test_penalty_rejects_invalid_schedule(value):
    with pytest.raises(ValueError):
        penalty_schedule(value)


def test_penalty_preserves_order_and_scalar_compatibility():
    assert penalty_schedule(1.05) == [1.05]
    assert penalty_schedule([1.1, 1, 1.2]) == [1.1, 1, 1.2]


def task(name):
    return QwenGenerationTask(name, (), "prompt", 32,
                              structured_output={"json": GRAPH_JSON_SCHEMA})


def validate(task_id, text):
    if not text.startswith("valid"):
        raise OutputValidationError("invalid response")
    return {"status": "complete", "text": text}, "native"


def run(tmp_path, generate, tasks=None, **kwargs):
    outputs, failures = {}, {}
    options = dict(penalties=[1, 1.05, 1.1, 1.15, 1.2], directory=tmp_path,
                   identity={"model": "fixture"}, validate=validate,
                   complete=lambda key, value: outputs.update({key: value}),
                   failed=lambda key, value: failures.update({key: value}),
                   raw_fallback=lambda key, raw: {"status": "raw_fallback", "text": raw})
    options.update(kwargs)
    generate_with_recovery(generate, tasks or [task("a")], **options)
    return outputs, failures


def cycles(tmp_path, name="a"):
    return next(json.loads(path.read_text())["cycles"] for path in tmp_path.glob("*.json")
                if json.loads(path.read_text())["task_id"] == name)


def test_only_failed_tasks_retry_constraints_stay_enabled_and_raw_is_last_nonempty(tmp_path):
    batches = []

    def generate(tasks, callback):
        batches.append([(t.task_id, t.repetition_penalty) for t in tasks])
        for t in tasks:
            assert t.structured_output == {"json": GRAPH_JSON_SCHEMA}
            text = ("valid first" if t.task_id == "b" else
                    "" if t.repetition_penalty == 1.2 else f"bad {t.repetition_penalty}")
            callback(t.task_id, text)
            callback(t.task_id, text)  # Duplicate callbacks do not consume another attempt.
        return {}

    outputs, failures = run(tmp_path, generate, [task("a"), task("b")])
    assert not failures
    assert outputs["a"]["status"] == "raw_fallback" and outputs["a"]["text"] == "bad 1.15"
    assert outputs["b"]["status"] == "complete"
    assert len(batches) == 5 and all([name for name, _ in b] == ["a"] for b in batches[1:])
    assert outputs["a"]["generation"]["attempt_count"] == 5
    assert not list(tmp_path.glob("*.json"))


def test_interrupt_resumes_next_penalty_and_failed_cycle_restarts(tmp_path):
    def interrupted(tasks, callback):
        callback("a", "")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, interrupted)
    seen = []

    def empty(tasks, callback):
        seen.append(tasks[0].repetition_penalty)
        return {"a": ""}

    outputs, failures = run(tmp_path, empty)
    assert not outputs and len(failures["a"]) == 5
    assert seen == [1.05, 1.1, 1.15, 1.2]
    outputs, _ = run(tmp_path, lambda tasks, _: {"a": "valid recovered"})
    assert outputs["a"]["status"] == "complete"
    assert outputs["a"]["generation"]["attempt_count"] == 1
    assert not list(tmp_path.glob("*.json"))


def test_publish_failure_replays_durable_response_and_force_starts_fresh(tmp_path):
    def fault(*_):
        raise OSError("disk unavailable")

    with pytest.raises(OSError):
        run(tmp_path, lambda *_: {"a": "valid saved"}, complete=fault)
    assert cycles(tmp_path)[0]["status"] == "prepared"
    outputs, _ = run(tmp_path, lambda *_: pytest.fail("must replay response"))
    assert outputs["a"]["text"] == "valid saved"
    run(tmp_path, lambda *_: {"a": "valid forced"}, force=True)
    assert not list(tmp_path.glob("*.json"))


def test_changed_input_does_not_use_previous_raw(tmp_path):
    run(tmp_path, lambda *_: {"a": "old raw"}, penalties=1)
    outputs, failures = run(tmp_path, lambda *_: {"a": ""},
                            [replace(task("a"), prompt="changed")], penalties=1)
    assert not outputs and failures["a"][0]["raw_response"] == ""


def test_engine_error_does_not_become_raw_or_consume_budget(tmp_path):
    def crash(*_):
        raise RuntimeError("structured grammar compilation failed")

    with pytest.raises(RuntimeError, match="compilation"):
        run(tmp_path, crash)
    assert not list(tmp_path.glob("*.json"))


def test_middle_penalty_succeeds_and_stops_retrying(tmp_path):
    seen = []
    def generate(tasks, _):
        seen.append(tasks[0].repetition_penalty)
        return {"a": "valid recovery" if len(seen) == 3 else "invalid"}
    outputs, failures = run(tmp_path, generate)
    assert seen == [1, 1.05, 1.1] and not failures
    assert outputs["a"]["status"] == "complete"


def test_scene_pass_finishes_all_batches_before_increasing_penalty(tmp_path):
    seen = []
    def generate(batch, _):
        batch = list(batch)
        seen.append([(t.task_id, t.repetition_penalty) for t in batch])
        return {t.task_id: "valid" if t.task_id == "b" or t.repetition_penalty > 1 else "bad"
                for t in batch}
    outputs, failures = run(tmp_path, generate, [task(n) for n in "abcd"],
                            batch_size=2, rounds_across_batches=True)
    assert not failures and len(outputs) == 4
    assert seen[0] == [("a", 1), ("b", 1), ("c", 1), ("d", 1)]
    assert len(seen) == 2
    assert set(seen[1]) == {("a", 1.05), ("c", 1.05), ("d", 1.05)}


def test_scene_pass_resume_finishes_unstarted_scenes_before_retry(tmp_path):
    def interrupted(batch, callback):
        next(iter(batch))
        callback("a", "bad")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, interrupted, [task("a"), task("b")],
            batch_size=1, rounds_across_batches=True)
    seen = []
    def generate(batch, _):
        batch = list(batch)
        seen.extend((t.task_id, t.repetition_penalty) for t in batch)
        return {t.task_id: "valid" for t in batch}
    run(tmp_path, generate, [task("a"), task("b")],
        batch_size=1, rounds_across_batches=True)
    assert seen == [("b", 1), ("a", 1.05)]


def test_scene_stream_prepares_only_admitted_tasks_and_refills_before_completion(tmp_path, monkeypatch):
    import extraction.recovery as recovery
    hashed = []
    monkeypatch.setattr(recovery, "file_fingerprint", lambda path: (hashed.append(path) or "fixture"))
    tasks = [replace(task(str(i)), image_paths=(str(i),)) for i in range(1000)]
    calls = []
    def generate(stream, callback):
        calls.append(True)
        assert hashed == ["0"]
        assert not list(tmp_path.glob("*.json"))
        iterator = iter(stream)
        first, second, third = next(iterator), next(iterator), next(iterator)
        assert hashed == ["0", "1", "2"]
        callback(second.task_id, "valid")
        fourth = next(iterator)  # The first scene is still running.
        for item in (first, third, fourth):
            callback(item.task_id, "valid")
        for item in iterator:
            callback(item.task_id, "valid")
        return {}
    outputs, failures = run(tmp_path, generate, tasks, rounds_across_batches=True)
    assert len(calls) == 1 and len(outputs) == 1000 and not failures


def test_graph_repair_never_invents_required_fields():
    from extraction.scene_executor import graph_scene_result
    row = {"scene_idx": 0, "keyframes": [5]}
    repaired, failure = graph_scene_result(row, repr(graph()), strict=True)
    assert failure is None and repaired["parse_mode"] == "repaired"
    assert repaired["graph"] == graph()
    record, failure = graph_scene_result(row, '{"setting_context": "indoor",}', strict=True)
    assert record is None and failure


def test_checkpoint_write_failure_preserves_last_durable_attempt(tmp_path, monkeypatch):
    import extraction.recovery as recovery
    original = recovery._save
    def fail_second_response(path, document):
        if len(document["cycles"][-1]["attempts"]) == 2:
            raise OSError("disk unavailable")
        original(path, document)
    monkeypatch.setattr(recovery, "_save", fail_second_response)
    with pytest.raises(OSError):
        run(tmp_path, lambda *_: {"a": "invalid"})
    assert len(cycles(tmp_path)[0]["attempts"]) == 1
    monkeypatch.setattr(recovery, "_save", original)
    seen = []
    run(tmp_path, lambda tasks, _: (seen.append(tasks[0].repetition_penalty) or {"a": "valid replay"}))
    assert seen == [1.05]


def test_first_inference_does_not_wait_for_all_input_hashes_or_checkpoints(tmp_path, monkeypatch):
    import extraction.recovery as recovery
    hashed = []
    monkeypatch.setattr(recovery, "file_fingerprint", lambda path: (hashed.append(path) or "fixture"))
    tasks = [replace(task(str(i)), image_paths=(f"image-{i}.png",)) for i in range(1000)]
    def generate(batch, callback):
        assert [t.task_id for t in batch] == ["0", "1"]
        assert hashed == ["image-0.png", "image-1.png"]
        assert not list(tmp_path.glob("*.json"))
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, generate, tasks, batch_size=2)
