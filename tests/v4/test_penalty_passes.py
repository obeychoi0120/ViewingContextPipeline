from contextlib import contextmanager

import pytest

import extraction.steps as steps
from pipeline_runtime import read_json, read_jsonl


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_scene_passes_retry_only_failures_and_preserve_graph_cutoff_text(
    ready_context, monkeypatch, representation
):
    context = ready_context
    context.config["extraction"][f"{representation}_repetition_penalty"] = [1.0, 1.05, 1.1]
    ids = [v["content_id"] for v in steps.visual_rows(context)]
    calls, engines = [], []
    succeed = False
    good = (
        '{"entities": [], "relations": [], "context": []}'
        if representation == "graph"
        else "A person walks."
    )

    @contextmanager
    def generator(**kwargs):
        engines.append(True)

        def generate(tasks, callback):
            for task in tasks:
                index = ids.index(task.task_id.rsplit(":", 1)[0])
                penalty = task.repetition_penalty
                calls.append((index, penalty))
                failed = not succeed and penalty < [1.0, 1.05, 1.1, 2.0][index]
                text = f"  unfinished output {penalty}\n" if failed else good
                kwargs["runtime"].current_result = {"finish_reason": "length" if failed else "stop"}
                callback(task.task_id, text)
                if failed and penalty < 1.1:
                    directory = context.extraction_dir(representation, "qwen", "scenes")
                    assert not (directory / f"{ids[index]}.jsonl").exists()
                    saved = read_jsonl(directory / "failures" / f"{ids[index]}.jsonl")[0]
                    assert saved["raw_output"] == (text if representation == "graph" else "")
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    extract = getattr(steps, f"extract_{representation}_scenes")
    options = dict(
        model="qwen",
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md",
    )
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )["failure_count"]
        == 1
    )
    assert calls == (
        [(i, 1.0) for i in range(4)] + [(i, 1.05) for i in (1, 2, 3)] + [(i, 1.1) for i in (2, 3)]
    )
    assert len(engines) == 1
    directory = context.extraction_dir(representation, "qwen", "scenes")
    assert not (directory / f"{ids[3]}.jsonl").exists()
    assert read_jsonl(directory / "failures" / f"{ids[3]}.jsonl")[0]["raw_output"] == (
        "  unfinished output 1.1\n" if representation == "graph" else ""
    )
    for cid in ids[:3]:
        assert not (directory / "failures" / f"{cid}.jsonl").exists()
    from validation.diagnosis_scenes import _scene_arm_contract, scene_arms

    errors = []
    _, _, valid = _scene_arm_contract(
        "graph_qwen" if representation == "graph" else "desc_qwen_meta",
        context.run_root,
        ids,
        {(cid, 0) for cid in ids},
        errors,
        scene_arms(context.config),
    )
    assert valid and not errors
    succeed = True
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )["failure_count"]
        == 0
    )
    assert calls[-1] == (3, 1.0) and len(calls) == 10
    assert not (directory / "failures" / f"{ids[3]}.jsonl").exists()
    for name in (".pending", ".recovery"):
        assert not (directory / name).exists()


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_summary_passes_retry_only_failures_and_keep_empty_failure(
    ready_context,
    fake_models,
    monkeypatch,
    representation,
):
    source = "qwen"
    context = ready_context
    context.config["extraction"]["summary_repetition_penalty"] = [1.0, 1.05, 1.1]
    extract = getattr(steps, f"extract_{representation}_scenes")
    extract(
        context,
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{source}",
        model=source,
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md",
    )
    ids = [v["content_id"] for v in steps.visual_rows(context)]
    calls, engines = [], []

    @contextmanager
    def generator(**kwargs):
        engines.append(True)

        def generate(tasks, callback):
            for task in tasks:
                index = ids.index(task.task_id)
                penalty = task.repetition_penalty
                calls.append((index, penalty))
                failed = penalty < [1.0, 1.05, 1.1, 2.0][index]
                kwargs["runtime"].current_result = {"finish_reason": "length" if failed else "stop"}
                callback(task.task_id, f"  - failed {penalty}\n" if failed else "A person walks.")
                if failed and penalty < 1.1:
                    directory = context.summary_arm_dir(
                        f"{'graph' if representation == 'graph' else 'desc'}_{source}"
                    )
                    assert not (directory / f"{ids[index]}.json").exists()
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    summarize = getattr(steps, f"summarize_{representation}")
    assert (
        summarize(
            context,
            model="qwen",
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{source}",
            schema=f"prompts/summary_{representation}_v5.md",
        )["failure_count"]
        == 1
    )
    assert calls == (
        [(i, 1.0) for i in range(4)] + [(i, 1.05) for i in (1, 2, 3)] + [(i, 1.1) for i in (2, 3)]
    )
    assert len(engines) == 1
    directory = context.summary_arm_dir(
        f"{'graph' if representation == 'graph' else 'desc'}_{source}"
    )
    raw = read_json(directory / f"{ids[3]}.json")
    assert raw["status"] == "failed" and raw["text"] == ""
    assert read_jsonl(directory / "failures.jsonl") == [
        {
            "content_id": ids[3],
            "error": "max_tokens",
            "raw_output": raw["text"],
            "repetition_penalty": 1.1,
            "summary_model": "qwen",
            "provenance": raw["provenance"],
        }
    ]
