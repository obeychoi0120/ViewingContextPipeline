from __future__ import annotations

import json

import pytest

import validation.steps as validation_steps
from pipeline_runtime import RunContext, write_json


from pipeline_fixtures import context as context


def test_failure_directories_are_nested_under_their_artifact_stage(
    context: RunContext,
) -> None:
    assert context.graph_failure_dir("qwen") == (context.graph_scene_dir("qwen") / "failures")
    assert context.graph_summary_failure_dir("gemini") == (
        context.graph_summary_dir("gemini") / "failures"
    )
    assert context.description_failure_dir == (context.description_scene_dir / "failures")
    assert context.description_summary_failure_dir == (context.description_summary_dir / "failures")


def test_diagnosis_recomputes_runtime_data_and_overwrites_stale_pass(
    context: RunContext,
) -> None:
    context.initialize()
    write_json(
        context.diagnosis_path,
        {
            "schema_version": "diagnosis/v3",
            "runtime_decision": {"status": "pass", "checks": {}, "errors": []},
        },
    )

    with pytest.raises(validation_steps.ValidationStepError, match="runtime diagnosis failed"):
        validation_steps.run_diagnosis(context)

    diagnosis = json.loads(context.diagnosis_path.read_text(encoding="utf-8"))
    assert diagnosis["schema_version"] == "diagnosis/v4"
    assert diagnosis["runtime_decision"]["status"] == "fail"
    assert "report_ready" not in diagnosis
