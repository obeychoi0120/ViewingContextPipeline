from __future__ import annotations

import json
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest

import extraction.steps as extraction_steps
import extraction.summary_executor as summary_executor
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.summary_validation import SUMMARY_SECTIONS, parse_summary_sections
from pipeline_runtime import RunContext, read_jsonl, write_json, write_jsonl


from pipeline_fixtures import context as context, _summary_lines


def test_summary_batch_saves_valid_callbacks_and_reports_failure_once() -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "visual_atmosphere": "A calm indoor atmosphere",
        "visible_affect": "Neutral visible affect",
        "semantic_topics": "Indoor activity",
    }
    structured = _summary_lines(sections)
    completed = []
    failures = []
    submissions = []

    def generate(tasks, callback=None):
        assert callback is not None
        submissions.append([task.task_id for task in tasks])
        callback("c2", structured)
        assert completed == ["c2"]
        callback("c1", "not labeled text")
        return {}

    def complete(task_id, text):
        parse_summary_sections(text)
        completed.append(task_id)

    with pytest.raises(
        extraction_steps.ExtractionStepError,
        match=r"task_ids=\['c1'\]",
    ):
        summary_executor.generate_summaries_once(
            generate,
            [
                QwenGenerationTask("c1", (), "prompt", 512),
                QwenGenerationTask("c2", (), "prompt", 512),
            ],
            complete,
            lambda task_id, attempt, seed, raw_response, error: failures.append(
                (task_id, attempt, seed, raw_response, str(error))
            ),
        )

    assert submissions == [["c1", "c2"]]
    assert completed == ["c2"]
    assert len(failures) == 1
    assert failures[0][:4] == ("c1", 1, None, "not labeled text")


def test_reused_summary_rejects_noncanonical_text(
    tmp_path: Path,
) -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "visual_atmosphere": "A calm indoor atmosphere",
        "visible_affect": "Neutral visible affect",
        "semantic_topics": "Indoor activity",
    }
    path = tmp_path / "summary.json"
    write_json(
        path,
        {
            "schema_version": "graph-video-summary/v3",
            "content_id": "c1",
            "arm": "graph_qwen",
            "status": "complete",
            "sections": sections,
            "text": "legacy single-line text",
            "scene_count": 1,
        },
    )

    with pytest.raises(
        summary_executor.ExtractionStepError,
        match="mismatched fields: text",
    ):
        summary_executor.reuse_summary_document(
            path,
            schema_version="graph-video-summary/v3",
            content_id="c1",
            arm="graph_qwen",
            scene_count=1,
        )


def test_reused_summary_rejects_v2_with_field_diagnostics(tmp_path: Path) -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "visual_atmosphere": "A calm indoor atmosphere",
        "visible_affect": "Neutral visible affect",
        "semantic_topics": "Indoor activity",
    }
    path = tmp_path / "summary.json"
    write_json(
        path,
        {
            "schema_version": "graph-video-summary/v2",
            "content_id": "c1",
            "arm": "graph_qwen",
            "status": "complete",
            "sections": sections,
            "text": "legacy",
            "scene_count": 1,
        },
    )

    with pytest.raises(
        summary_executor.ExtractionStepError,
        match="mismatched fields: schema_version, text",
    ):
        summary_executor.reuse_summary_document(
            path,
            schema_version="graph-video-summary/v3",
            content_id="c1",
            arm="graph_qwen",
            scene_count=1,
        )


@pytest.mark.parametrize("source", ["qwen", "gemini", None])
@pytest.mark.parametrize(
    "problem", ["text", "scene_count", "arm", "schema_version", "sections", "invalid_json"]
)
def test_summary_resume_retries_only_unusable_or_missing_outputs(
    context,
    monkeypatch,
    capsys,
    source,
    problem,
):
    content_ids = ["c1", "c2", "c3", "c4", "c5"]
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": cid, "source_video_path": f"{cid}.mp4"} for cid in content_ids],
    )
    monkeypatch.setattr(
        extraction_steps, "_visual_rows", lambda _: [{"content_id": cid} for cid in content_ids]
    )
    if source:
        scene_dir = context.graph_scene_dir(source)
        summary_dir = context.graph_summary_dir(source)
        schema = "graph-video-summary/v3"
        arm = f"graph_{source}"

        def run(**kwargs):
            return extraction_steps.summarize_graph(context, source=source, **kwargs)
    else:
        scene_dir = context.description_scene_dir
        summary_dir = context.description_summary_dir
        schema = "description-video-summary/v3"
        arm = "description"

        def run(**kwargs):
            return extraction_steps.summarize_description(context, **kwargs)

    sections = dict.fromkeys(SUMMARY_SECTIONS, "")
    sections["setting_and_environments"] = "An indoor room."
    for cid in content_ids:
        record = {"scene_idx": 0, "keyframes": [5]}
        if source:
            record.update(
                graph={"setting_context": "indoor"}, parse_mode="native", semantic_warnings=[]
            )
        else:
            record.update(
                schema_version="scene-description/v1", content_id=cid, description="An indoor room."
            )
        write_jsonl(scene_dir / f"{cid}.jsonl", [] if cid == "c5" else [record])
    for cid in ("c1", "c2"):
        document = dict(
            schema_version=schema,
            content_id=cid,
            arm=arm,
            status="complete",
            sections=sections,
            text=summary_executor.serialize_summary_sections(sections),
            scene_count=1,
        )
        if cid == "c2" and problem != "invalid_json":
            document[problem] = {"scene_count": 2, "sections": None}.get(problem, "legacy")
        write_json(summary_dir / f"{cid}.json", document)
    invalid_path = summary_dir / "c2.json"
    if problem == "invalid_json":
        invalid_path.write_text('{"truncated":', encoding="utf-8")
    original_invalid = invalid_path.read_bytes()
    successful_path = summary_dir / "c1.json"
    original_success = successful_path.read_bytes()
    write_jsonl(summary_dir / "failures/c3.jsonl", [{"error": "previous failure"}])
    submitted = []
    bars = []
    real_tqdm = extraction_steps.tqdm

    def tracked_tqdm(*args, **kwargs):
        bar = real_tqdm(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(extraction_steps, "tqdm", tracked_tqdm)

    @contextmanager
    def generator(**_kwargs):
        def generate(tasks, callback):
            submitted.append([task.task_id for task in tasks])
            for task in tasks:
                assert (
                    task.repetition_penalty
                    == context.config["extraction"]["summary_repetition_penalty"]
                )
                fail = len(submitted) == 1 and task.task_id == "c2"
                callback(task.task_id, "invalid response" if fail else _summary_lines(sections))
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", generator)
    with pytest.raises(extraction_steps.ExtractionStepError, match="structured summary failed"):
        run()
    assert submitted == [["c2", "c3", "c4"]]
    assert bars[-1].initial == 0 and bars[-1].total == bars[-1].n == 3
    assert "success=2 failed=1" in bars[-1].postfix
    assert invalid_path.read_bytes() == original_invalid
    assert successful_path.read_bytes() == original_success
    assert not (summary_dir / "failures/c3.jsonl").exists()
    stderr = capsys.readouterr().err
    assert "reused=1 pending=3 incompatible=1" in stderr
    assert "[SUMMARY RETRY]" in stderr and str(invalid_path) in stderr

    context.config["extraction"]["summary_repetition_penalty"] = 1.3
    assert run()["content_count"] == 4
    assert submitted[-1] == ["c2"]
    assert bars[-1].initial == 0 and bars[-1].total == bars[-1].n == 1
    assert "success=1 failed=0" in bars[-1].postfix
    assert successful_path.read_bytes() == original_success
    assert not (summary_dir / "failures/c2.jsonl").exists()
    assert run()["content_count"] == 4
    assert len(submitted) == 2  # Fully cached: no model is loaded.
    assert run(force=True)["content_count"] == 4
    assert submitted[-1] == ["c1", "c2", "c3", "c4"]


@pytest.fixture(params=["graph", "description"])
def summary_failure_case(context, monkeypatch, request):
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _: [{"content_id": "c1"}])
    scene = {"scene_idx": 0, "keyframes": [5, 15, 25]}
    if request.param == "graph":
        scene.update(graph={"setting_context": "indoor"}, parse_mode="native", semantic_warnings=[])
        scene_dir = context.graph_scene_dir("qwen")
        summary_dir = context.graph_summary_dir("qwen")
        failure_dir = context.graph_summary_failure_dir("qwen")
        run = partial(extraction_steps.summarize_graph, context, source="qwen", gpus=1)
        arm, schema, invalid = "graph_qwen", "graph-video-summary/v3", "not labeled text"
    else:
        scene.update(
            schema_version="scene-description/v1",
            content_id="c1",
            description="A person walks indoors.",
        )
        scene_dir = context.description_scene_dir
        summary_dir = context.description_summary_dir
        failure_dir = context.description_summary_failure_dir
        run = partial(extraction_steps.summarize_description, context)
        arm, schema, invalid = "description", "description-video-summary/v3", "not json"
    write_jsonl(scene_dir / "c1.jsonl", [scene])
    sections = {name: "" for name in SUMMARY_SECTIONS}
    sections.update(
        setting_and_environments="An indoor setting.", main_characters_and_objects="A person."
    )
    return run, summary_dir / "c1.json", failure_dir / "c1.jsonl", arm, schema, invalid, sections


def test_summary_failure_is_saved_once_and_manual_resume_removes_it(
    summary_failure_case,
    monkeypatch,
    capsys,
):
    run, output_path, failure_path, arm, schema, invalid, sections = summary_failure_case

    @contextmanager
    def invalid_generator(**_kwargs):
        def generate(tasks, callback):
            assert tasks[0].do_sample is False
            assert tasks[0].repetition_penalty == 1.05
            assert tasks[0].max_new_tokens == 512
            callback(tasks[0].task_id, invalid)
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", invalid_generator)
    with pytest.raises(
        extraction_steps.ExtractionStepError, match=r"structured summary failed: task_ids=\['c1'\]"
    ):
        run()
    failures = read_jsonl(failure_path)
    assert len(failures) == 1
    assert failures[0]["schema_version"] == "summary-generation-failure/v1"
    assert failures[0]["attempt"] == 1
    assert failures[0]["seed"] is None
    assert failures[0]["failure_kind"] == "schema_validation"
    assert failures[0]["raw_response"] == invalid
    stderr = capsys.readouterr().err
    assert f"[Qwen_summary_{arm}_fail]" in stderr
    assert f"Raw output:\n{invalid}" in stderr
    assert "generation started" not in stderr

    @contextmanager
    def valid_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[0].task_id, _summary_lines(sections))
            assert output_path.is_file()
            assert not failure_path.exists()
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", valid_generator)
    result = run()
    assert result["content_count"] == 1
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["sections"] == sections
    assert document["schema_version"] == schema
    assert document["arm"] == arm
    assert not failure_path.exists()


@pytest.mark.parametrize("fault", ["interrupt", "save_error"])
def test_summary_callback_fault_preserves_committed_state(summary_failure_case, monkeypatch, fault):
    import artifact_io

    run, output_path, failure_path, arm, schema, invalid, sections = summary_failure_case
    previous = {"previous": "incompatible output must survive a failed write"}
    previous_failure = [{"error": "previous failure"}]
    write_json(output_path, previous)
    write_jsonl(failure_path, previous_failure)
    closed = []
    replace = artifact_io.os.replace

    def fail_output_replace(source, destination):
        if Path(destination) == output_path:
            raise OSError("simulated summary save failure")
        return replace(source, destination)

    @contextmanager
    def generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[0].task_id, _summary_lines(sections))
            raise KeyboardInterrupt

        try:
            yield generate
        finally:
            closed.append(True)

    monkeypatch.setattr(extraction_steps, "qwen_generator", generator)
    if fault == "save_error":
        monkeypatch.setattr(artifact_io.os, "replace", fail_output_replace)
    with pytest.raises(OSError if fault == "save_error" else KeyboardInterrupt):
        run()
    assert closed == [True]
    if fault == "save_error":
        assert json.loads(output_path.read_text(encoding="utf-8")) == previous
        assert read_jsonl(failure_path) == previous_failure
    else:
        assert json.loads(output_path.read_text(encoding="utf-8"))["sections"] == sections
        assert not failure_path.exists()


def test_graph_summary_rejects_legacy_scene_shape(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    monkeypatch.setattr(
        extraction_steps,
        "_visual_rows",
        lambda _context: [{"content_id": "c1"}],
    )
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"
    write_jsonl(
        scene_path,
        [
            {
                "schema_version": "legacy",
                "content_id": "c1",
                "scene_idx": 0,
                "keyframes": [5, 15, 25],
                "image_paths": ["a.png", "b.png", "c.png"],
                "graph_source": "gemini",
                "input_fingerprint": "legacy",
                "graph": {"entities": []},
            }
        ],
    )

    with pytest.raises(
        extraction_steps.ExtractionStepError,
        match="use --force or a new run_id",
    ):
        extraction_steps.summarize_graph(context, source="qwen")
