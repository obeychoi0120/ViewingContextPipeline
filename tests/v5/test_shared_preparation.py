"""Preparation is shared across runs; extraction and validation remain isolated."""
import pytest

from pipeline_runtime import RunContext, read_json, write_json
from preparation.steps import prepare_cohort_step
from extraction.steps import extract_description_scenes, summarize_description
from validation.steps import embed_representations


def test_new_run_consumes_preparation_without_rebuilding(ready_context, fake_models):
    first = ready_context
    second = RunContext.load("second_run", root=first.root)
    shared = first.root / "artifacts" / "preperation"
    assert first.preparation_dir == second.preparation_dir == shared
    assert first.cohort_dir == second.cohort_dir == shared / "cohort"
    assert first.keyframes_dir == second.keyframes_dir == shared / "resized_keyframes"
    assert first.source_assets_dir == second.source_assets_dir == shared / "source_assets"
    assert "run_id" not in read_json(first.cohort_dir / "cohort_plan.json")
    assert "run_id" not in read_json(first.cohort_dir / "eligibility.json")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in shared.rglob("*") if p.is_file()}
    assert first.require_ready_cohort() == second.require_ready_cohort()
    for context in (first, second):
        context.config["protocol"]["arms"] = ["desc_qwen", "metadata"]
        extract_description_scenes(context, model="qwen", schema="prompts/description_scene_v2.md")
        summarize_description(context, source="qwen", model="gemini",
                              schema="prompts/description_summary_v4.md")
        assert embed_representations(context, target=["desc_qwen", "metadata"], summary_source="gemini")["generated_arms"] == ["desc_qwen", "metadata"]
        assert not (context.run_root / "cohort").exists()
        docs = list(context.description_summary_dir("qwen", "gemini").glob("*.json"))
        assert docs and context.representations_dir.is_dir()
    assert first.description_summary_dir("qwen", "gemini") != second.description_summary_dir("qwen", "gemini")
    assert first.representations_dir != second.representations_dir
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in shared.rglob("*") if p.is_file()}


def test_shared_cohort_accepts_legacy_run_identity(ready_context):
    path = ready_context.cohort_dir / "eligibility.json"
    write_json(path, {**read_json(path), "run_id": "old_run"})
    assert RunContext.load("new_run", root=ready_context.root).require_ready_cohort()


def test_plan_only_invalidates_previous_shared_ready_state(ready_context):
    other = RunContext.load("other", root=ready_context.root)
    prepare_cohort_step(other, plan_only=True)
    with pytest.raises(RuntimeError, match="not ready"):
        ready_context.require_ready_cohort()
    prepare_cohort_step(other)
    assert ready_context.require_ready_cohort()


def test_failed_shared_cohort_refresh_is_not_readable(ready_context):
    from pathlib import Path
    other = RunContext.load("other", root=ready_context.root)
    Path(other.config["data"]["videos_dir"]).joinpath("1.mp4").unlink()
    with pytest.raises(RuntimeError, match="unresolved assets"):
        prepare_cohort_step(other)
    with pytest.raises(RuntimeError, match="not ready"):
        ready_context.require_ready_cohort()
