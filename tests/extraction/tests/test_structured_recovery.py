from copy import deepcopy
from dataclasses import replace
import json

import pytest

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.recovery import generate_with_recovery, penalty_schedule
from extraction.structured_output import (
    GRAPH_JSON_SCHEMA, SUMMARY_GRAMMAR, OutputValidationError, validate_graph_structure,
)
from extraction.summary_validation import (
    SUMMARY_SECTIONS, SummaryContractError, parse_or_repair_summary,
)


def graph():
    return {"setting_context": "indoor", "entities": [], "events": [],
            "semantic_topics": [], "affect": {"valence": "neutral", "arousal": "medium"}}


@pytest.mark.parametrize("value", [[], [True], float("nan"), [float("inf")], [0.99], [2.01], "1"])
def test_penalty_rejects_invalid_schedule(value):
    with pytest.raises(ValueError):
        penalty_schedule(value)


def test_penalty_preserves_order_and_scalar_compatibility():
    assert penalty_schedule(1.05) == [1.05]
    assert penalty_schedule([1.1, 1, 1.2]) == [1.1, 1, 1.2]


def test_graph_structure_checks_shape_but_not_cardinality_or_references():
    value = graph()
    value["entities"] = [{"local_id": "e1", "name": "person", "function": None,
                          "count": "many", "salience": "primary"}] * 8
    value["events"] = [{"local_id": "ev1", "actor_id": "unresolved", "action": "walk",
                        "target_id": None, "instrument_id": None, "location_id": None}]
    validate_graph_structure(value)
    from extraction.semantic_graph import graph_semantic_warnings
    assert graph_semantic_warnings(value)
    for mutate in (lambda g: g.update(extra=True), lambda g: g.pop("entities"),
                   lambda g: g.update(setting_context="invalid"),
                   lambda g: g["affect"].update(valence=2),
                   lambda g: g["entities"][0].update(function=1)):
        invalid = deepcopy(value)
        mutate(invalid)
        with pytest.raises(OutputValidationError):
            validate_graph_structure(invalid)
    assert "maxItems" not in json.dumps(GRAPH_JSON_SCHEMA)


def test_summary_repair_is_content_preserving_and_idempotent():
    lines = [f"{name}: Value {index}." for index, name in enumerate(SUMMARY_SECTIONS)]
    canonical = "\n".join(lines)
    repaired, mode = parse_or_repair_summary("```text\n" + "\n".join(reversed(lines)) + "\n```")
    assert mode == "repaired"
    assert repaired == parse_or_repair_summary(canonical)[0]
    assert parse_or_repair_summary(canonical)[1] == "native"
    assert parse_or_repair_summary("```markdown\n" + canonical + "\n```")[0] == repaired
    display = canonical.replace("setting_and_environments", "Setting and environments")
    assert parse_or_repair_summary(display)[0] == repaired
    for invalid in (canonical + "\n" + lines[0], "\n".join(lines[:-1]),
                    "\n".join(f"{key}:" for key in SUMMARY_SECTIONS),
                    "Comment\n" + canonical):
        with pytest.raises(SummaryContractError):
            parse_or_repair_summary(invalid)
    assert all(name in SUMMARY_GRAMMAR for name in SUMMARY_SECTIONS)


def task(name):
    return QwenGenerationTask(name, (), "prompt", 32,
                              structured_output={"grammar": SUMMARY_GRAMMAR})


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
            assert t.structured_output == {"grammar": SUMMARY_GRAMMAR}
            text = ("valid first" if t.task_id == "b" else
                    "" if t.repetition_penalty == 1.2 else f"bad {t.repetition_penalty}")
            callback(t.task_id, text)
            callback(t.task_id, text)  # Duplicate callbacks do not consume another attempt.
        return {}

    outputs, failures = run(tmp_path, generate, [task("a"), task("b")])
    assert not failures
    assert outputs["a"] == {"status": "raw_fallback", "text": "bad 1.15"}
    assert outputs["b"]["status"] == "complete"
    assert len(batches) == 5 and all([name for name, _ in b] == ["a"] for b in batches[1:])
    assert len(cycles(tmp_path)[0]["attempts"]) == 5
    run(tmp_path, lambda *_: pytest.fail("committed results must be replayed"),
        [task("a"), task("b")])


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
    assert len(cycles(tmp_path)) == 2
    assert len(cycles(tmp_path)[-1]["attempts"]) == 1


def test_publish_failure_replays_durable_response_and_force_starts_fresh(tmp_path):
    def fault(*_):
        raise OSError("disk unavailable")

    with pytest.raises(OSError):
        run(tmp_path, lambda *_: {"a": "valid saved"}, complete=fault)
    assert cycles(tmp_path)[0]["status"] == "prepared"
    outputs, _ = run(tmp_path, lambda *_: pytest.fail("must replay response"))
    assert outputs["a"]["text"] == "valid saved"
    run(tmp_path, lambda *_: {"a": "valid forced"}, force=True)
    assert len(cycles(tmp_path)) == 2


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


def test_force_resume_includes_later_unprepared_batches_with_old_successes(tmp_path):
    from extraction.recovery import active_force_run, has_pending_recovery
    tasks = [task(name) for name in ("a", "b", "c")]
    run(tmp_path, lambda batch, _: {t.task_id: "valid old" for t in batch}, tasks)
    def interrupted(batch, _):
        if batch[0].task_id == "b":
            raise KeyboardInterrupt
        return {batch[0].task_id: "valid forced"}
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, interrupted, tasks, batch_size=1, force=True)
    force_id = active_force_run(tmp_path)
    assert force_id and not has_pending_recovery(tmp_path, "a", force_id)
    assert has_pending_recovery(tmp_path, "b", force_id)
    assert has_pending_recovery(tmp_path, "c", force_id)
    assert len(cycles(tmp_path, "c")) == 1  # Its old success must not hide unstarted forced work.
    calls = []
    def resume(batch, _):
        calls.extend(t.task_id for t in batch)
        return {t.task_id: "valid resumed" for t in batch}
    run(tmp_path, resume, tasks[1:], batch_size=1)
    assert calls == ["b", "c"] and active_force_run(tmp_path) is None
    assert all(len(cycles(tmp_path, name)) == 2 for name in ("a", "b", "c"))
