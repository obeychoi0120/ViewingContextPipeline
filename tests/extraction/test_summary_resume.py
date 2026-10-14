"""Summary retries resume from persisted attempts, including terminal publication."""

from contextlib import contextmanager

import pytest

import extraction.steps as steps
import extraction.summary_executor as executor
from artifact_io import read_json, read_jsonl, write_jsonl
from arm_registry import registry
from validation.representation_inputs import documents_for_arm


@pytest.fixture(params=[("graph", "qwen3vl"), ("description", "qwen3vl")])
def summary_case(request, ready_context, fake_models):
    representation, source = request.param
    context = ready_context
    getattr(steps, f"extract_{representation}_scenes")(
        context,
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{source}",
        model=source,
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md",
    )
    cohort = context.require_ready_cohort()
    ids = [str(row["content_id"]) for row in cohort["catalog"]]
    directory = context.summary_arm_dir(
        f"{'graph' if representation == 'graph' else 'desc'}_{source}"
    )
    arm_name = f"{'desc' if representation == 'description' else 'graph'}_{source}"

    def run(**kwargs):
        return steps.summarize(
            context,
            model="qwen3vl",
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{source}",
            schema=f"prompts/summary_{representation}_v5.md",
            **kwargs,
        )

    def documents():
        return documents_for_arm(context, cohort, registry(context.config)[arm_name if arm_name == "graph_qwen3vl" else f"{arm_name}_meta"])

    return context, ids, directory, run, documents


def test_resume_after_out_of_order_results(summary_case, monkeypatch):
    interrupt_after = 2
    _, ids, directory, run, documents = summary_case
    calls = []
    thresholds = dict(zip(ids, [1.0, 2.0, 2.0, 2.0]))

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            # Completion order deliberately differs from catalog order.
            for task in reversed(list(tasks)):
                calls.append((task.task_id, task.repetition_penalty))
                failed = task.repetition_penalty < thresholds[task.task_id]
                kwargs["runtime"].current_result = {"finish_reason": "length" if failed else "stop"}
                callback(
                    task.task_id, "  unfinished\n\ntext  " if failed else "- First\n\n- Second"
                )
                if len(calls) == interrupt_after:
                    raise KeyboardInterrupt
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert run()["content_count"] == len(ids)
    expected = [(cid, 1.0) for cid in ids] + [(cid, 1.0) for cid in ids[2:]]
    assert sorted(calls) == sorted(expected)
    rows = read_jsonl(directory / "failures.jsonl")
    assert len(rows) == 3 and all(row["repetition_penalty"] == 1.0 for row in rows)
    assert len(documents()) == len(ids)
    assert run()["failure_count"] == 3
    assert len(calls) == len(expected) + 3



def test_terminal_write_failure_retries_with_same_policy(summary_case, monkeypatch):
    context, ids, directory, run, documents = summary_case
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append(task.task_id)
                kwargs["runtime"].current_result = {"finish_reason": "length"}
                callback(task.task_id, "  unfinished\n\nraw  ")
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    original = executor.atomic_write_json

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(executor, "atomic_write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        run()
    (row,) = read_jsonl(directory / "failures.jsonl")
    assert row["repetition_penalty"] == 1.0
    assert not (directory / f"{ids[0]}.json").exists()
    monkeypatch.setattr(executor, "atomic_write_json", original)
    assert run()["content_count"] == len(ids)
    assert calls == [ids[0], *ids]
    assert len(documents()) == len(ids)


@pytest.mark.parametrize("scene_state", ["missing", "empty", "invalid"])
def test_invalid_scene_input_reports_error(summary_case, scene_state):
    context, ids, directory, run, _ = summary_case
    from extraction.errors import ExtractionStepError

    scene_path = context.scene_arm_dir(directory.name) / f"{ids[0]}.jsonl"
    if scene_state == "missing":
        scene_path.unlink()
    else:
        write_jsonl(scene_path, [] if scene_state == "empty" else [{"bad": "record"}])
    if scene_state == "invalid":
        with pytest.raises(ExtractionStepError, match="scene"):
            run()
        assert not (directory / f"{ids[0]}.json").exists()
    else:
        run()
        doc = read_json(directory / f"{ids[0]}.json")
        assert doc["status"] == "failed" and doc["text"] == ""
        assert doc["scene_count"] == doc["word_count"] == 0
