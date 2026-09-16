import pytest

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.generation import generate_once
from extraction.structured_output import GRAPH_JSON_SCHEMA


def task(name):
    return QwenGenerationTask(name, (), "prompt", 32,
                              structured_output={"json": GRAPH_JSON_SCHEMA})


def test_streaming_once_refills_before_completion_and_never_retries():
    seen, outputs, calls = [], {}, []

    def tasks():
        for i in range(1000):
            seen.append(str(i))
            yield task(str(i))

    def generate(stream, callback):
        calls.append(True)
        assert seen == []
        iterator = iter(stream)
        first, second = next(iterator), next(iterator)
        assert seen == ["0", "1"]
        callback(second.task_id, "")
        third = next(iterator)  # Refill while the first task is still running.
        for item in (first, third):
            assert item.structured_output == {"json": GRAPH_JSON_SCHEMA}
            callback(item.task_id, "invalid output")
            callback(item.task_id, "duplicate ignored")
        for item in iterator:
            callback(item.task_id, "valid output")
        return {}

    generate_once(generate, tasks(), lambda key, value: outputs.update({key: value}))
    assert calls == [True] and len(outputs) == 1000
    assert outputs["0"] == "invalid output" and outputs["1"] == ""


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("disk unavailable"), RuntimeError("OOM")])
def test_single_generation_propagates_execution_and_publication_errors(error, tmp_path):
    def generate(tasks, callback):
        for item in tasks:
            callback(item.task_id, "response")
        return {}

    def publish(*_):
        raise error

    with pytest.raises(type(error)):
        generate_once(generate, [task("a")], publish)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case", ["missing", "unexpected", "duplicate"])
def test_single_generation_rejects_invalid_backend_results(case):
    def generate(tasks, callback):
        list(tasks)
        if case == "unexpected":
            callback("other", "text")
        return {}

    tasks = [task("a"), task("a")] if case == "duplicate" else [task("a")]
    with pytest.raises((ValueError, RuntimeError), match=case):
        generate_once(generate, tasks, lambda *_: None)


def test_returned_results_are_published_once():
    outputs = {}
    def generate(tasks, callback):
        return {item.task_id: "text" for item in tasks}
    generate_once(generate, [task("a")], lambda key, value: outputs.update({key: value}))
    assert outputs == {"a": "text"}


def test_graph_repair_never_invents_required_fields():
    from extraction.scene_executor import graph_scene_result
    graph = {"entities": [], "relations": [], "context": []}
    row = {"scene_idx": 0, "keyframes": [5]}
    repaired, failure = graph_scene_result(row, repr(graph), strict=True)
    assert failure is None and repaired["parse_mode"] == "repaired"
    assert repaired["graph"] == graph
    record, failure = graph_scene_result(row, '{"setting_context": "indoor",}', strict=True)
    assert record is None and failure
