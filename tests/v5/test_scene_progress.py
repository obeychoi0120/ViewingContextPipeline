import json
from functools import partial
from io import StringIO
from pathlib import Path

import pytest
from tqdm import tqdm

import extraction.steps as steps
from extraction.progress import InferenceProgress
from extraction.scene_storage import read_scene_records
from extraction.step_support import write_scene_results


@pytest.mark.parametrize("unit", ["scene", "summary"])
def test_progress_displays_step_rate_for_each_unit(unit):
    now = [0.0]
    stream = StringIO()
    with InferenceProgress(
        total=4, desc="Generation", unit=unit, reused=10,
        clock=lambda: now[0], progress_factory=partial(tqdm, file=stream),
    ) as progress:
        assert f"{unit}/s=--" in progress.bar.postfix
        now[0] = 5.0
        progress._render(refresh=True)
        assert f"{unit}/s=0.00" in progress.bar.postfix

        progress.complete(task_id="a")
        progress.complete(task_id="b", failed=True, raw=True)
        now[0] = 10.0
        progress._render(refresh=True)
        assert f"{unit}/s=0.20" in progress.bar.postfix
        assert "ETA=00:10" in progress.bar.postfix
        assert "success=1 failed=1 raw=1" in progress.bar.postfix

        # A backend reset must not remove or reset the step-wide rate.
        progress.update_stats({"phase": "initializing", "requests_per_second": 0})
        progress._render(refresh=True)
        assert f"{unit}/s=0.20" in progress.bar.postfix
        progress.complete(task_id="c")
        progress.complete(task_id="d")
        now[0] = 20.0

    assert f"{unit}/s=0.20" in progress.bar.postfix
    assert "ETA=00:00" in progress.bar.postfix
    assert f"{unit}/s=0.20" in stream.getvalue()


@pytest.mark.parametrize("model", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["description", "graph"])
def test_progress_counts_only_pending_scenes(
    ready_context, fake_models, monkeypatch, model, representation,
):
    context = ready_context
    timestamp = Path(steps.visual_rows(context)[0]["timestamp_json"])
    scenes = json.loads(timestamp.read_text())
    scenes.append({**scenes[0], "scene_idx": 1})
    timestamp.write_text(json.dumps(scenes))
    extract = getattr(steps, f"extract_{representation}_scenes")
    schema = f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md"
    expected_total = 5
    expected_reused = 0
    instances = []
    original_progress = steps.InferenceProgress

    def progress_factory(**kwargs):
        # Check before a worker can consume (and exhaust) the task iterator.
        assert kwargs["total"] == expected_total
        assert kwargs["reused"] == expected_reused
        progress = original_progress(**kwargs)
        instances.append(progress)
        return progress

    monkeypatch.setattr(steps, "InferenceProgress", progress_factory)

    def run(**kwargs):
        assert extract(context, model=model, schema=schema, **kwargs)["failure_count"] == 0
        progress = instances[-1]
        assert progress.total == progress.bar.total == expected_total
        assert progress.success == progress.bar.n == expected_total
        assert progress.failed == 0
        assert "ETA=00:00" in progress.bar.postfix
        assert "scene/s=" in progress.bar.postfix

    run()
    expected_total, expected_reused = 0, 5
    run()

    paths = sorted(context.extraction_dir(representation, model, "scenes").glob("*.jsonl"))
    # Reuse one scene within a partially completed content.
    rows = read_scene_records(paths[0])
    assert len(rows) == 2
    write_scene_results(paths[0], rows[:1])
    expected_total, expected_reused = 1, 4
    run()
    assert [row["scene_idx"] for row in read_scene_records(paths[0])] == [0, 1]

    paths[1].unlink()
    run()

    # Successful output is reused even if its generation provenance is stale.
    rows = read_scene_records(paths[1])
    rows[0]["provenance"] = {}
    rows[0]["generation"] = {"input_key": "stale"}
    expected_total, expected_reused = 0, 5
    write_scene_results(paths[1], rows)
    run()

    expected_total, expected_reused = 5, 0
    run(force=True)
