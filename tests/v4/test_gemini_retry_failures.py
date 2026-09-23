import json
from pathlib import Path

import pytest

import extraction.steps as steps
from extraction.backends import GeminiGenerationOutcome
from extraction.scene_storage import read_scene_records
from pipeline_runtime import read_jsonl


def test_gemini_scene_retry_preserves_other_failures_and_removes_file_after_last_success(
    ready_context,
    monkeypatch,
):
    representation = "graph"
    context = ready_context
    context.config["extraction"]["summary_repetition_penalty"] = [1.0]
    visual = steps.visual_rows(context)[0]
    cid = visual["content_id"]
    timestamp = Path(visual["timestamp_json"])
    original = json.loads(timestamp.read_text())[0]
    timestamp.write_text(json.dumps([{**original, "scene_idx": i} for i in range(3)]))
    monkeypatch.setattr(steps, "visual_rows", lambda _: [visual])
    directory = context.extraction_dir(representation, "gemini", "scenes")
    failure_path = directory / "failures" / f"{cid}.jsonl"
    extract = getattr(steps, f"extract_{representation}_scenes")
    options = dict(
        model="gemini",
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md",
    )
    good = (
        '{"entities": [], "relations": [], "context": []}'
        if representation == "graph"
        else "A person walks."
    )
    calls = []
    phase = 0

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                idx = int(task.task_id.rsplit(":", 1)[1])
                calls.append((phase, idx))
                if phase == 1:
                    assert len(read_jsonl(failure_path)) == 2
                    raise KeyboardInterrupt
                error = None
                text = good
                if phase == 0 and idx == 0:
                    error, text = "429 RESOURCE_EXHAUSTED", ""
                if phase < 3 and idx == 1:
                    text = f"failed output {phase}"
                    error = "429 RESOURCE_EXHAUSTED"
                callback(GeminiGenerationOutcome(task.task_id, text, error=error))
                if phase == 2 and idx == 0:
                    assert [row["scene_idx"] for row in read_jsonl(failure_path)] == [1]
            return {}

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )["failure_count"]
        == 2
    )
    saved_good = next(
        row for row in read_scene_records(directory / f"{cid}.jsonl") if row["scene_idx"] == 2
    )
    before = failure_path.read_bytes()
    phase = 1
    with pytest.raises(KeyboardInterrupt):
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )
    assert failure_path.read_bytes() == before
    phase = 2
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )["failure_count"]
        == 1
    )
    rows = read_jsonl(failure_path)
    assert len(rows) == 1 and rows[0]["raw_output"] == ""
    phase = 3
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )["failure_count"]
        == 0
    )
    assert not failure_path.exists()
    assert calls == [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (3, 1)]
    saved = read_scene_records(directory / f"{cid}.jsonl")
    assert [row["scene_idx"] for row in saved] == [0, 1, 2]
    assert saved[2] == saved_good
    assert (
        extract(
            context,
            arm=f"{'graph' if representation == 'graph' else 'desc'}_{options['model']}",
            **options,
        )["failure_count"]
        == 0
    )
    assert len(calls) == 7
