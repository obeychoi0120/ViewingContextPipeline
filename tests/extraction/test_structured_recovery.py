import pytest

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.generation import generate_once
from extraction.structured_output import GRAPH_JSON_SCHEMA


def task(name):
    return QwenGenerationTask(name, (), "prompt", 32,
                              structured_output={"json": GRAPH_JSON_SCHEMA})


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
