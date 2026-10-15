"""Inline failures are accounted for but never become Summary or Graph observations."""

import pytest

from arm_registry import registry
from artifact_io import read_json
from extraction import steps
from extraction.errors import ExtractionStepError
from extraction.recovery_report import recovery_report
from extraction.scene_storage import read_scene_records, write_scene_records
from validation.diagnosis_scenes import _scene_arm_contract, scene_arms
from validation.graph_context import graph_context
from validation.graph_inputs import build_input, source_identity
from validation.selection import prepare_validation_cohort


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_inline_failures_are_excluded_from_inputs_and_counted_by_diagnosis(
    ready_context, fake_models, representation,
):
    context = ready_context
    source = f"{'graph' if representation == 'graph' else 'desc'}_qwen3vl"
    target = source if representation == "graph" else source + "_meta"
    getattr(steps, f"extract_{representation}_scenes")(
        context, model="qwen3vl", arm=source,
        schema=f"prompts/scene_{representation}_v{9 if representation == 'graph' else 3}.md",
    )
    ids = [row["content_id"] for row in context.require_ready_cohort()["catalog"]]
    path = context.scene_arm_dir(source) / f"{ids[0]}.jsonl"
    records = read_scene_records(path)
    observation = (
        {"graph": "unparseable warning observation", "parse_mode": "text",
         "semantic_warnings": ["PARSE_FAILED"]}
        if representation == "graph" else {"description": "A second observation."}
    )
    failure = {"content_id": ids[0], "scene_idx": 2, "tokens": 17,
               "error": "generation failed", "raw_output": "NEVER_SUMMARIZE_THIS"}
    write_scene_records(path, [
        *records, {**records[0], **observation, "scene_idx": 1}, failure,
    ])
    second_path = path.with_name(f"{ids[1]}.jsonl")
    write_scene_records(second_path, [{**failure, "content_id": ids[1], "scene_idx": 0}])
    with pytest.raises(ExtractionStepError, match="1 Gemini summaries failed"):
        steps.summarize(
            context, model="gemini", arm=source,
            schema=f"prompts/summary_{representation}_v{9 if representation == 'graph' else 5}.md",
        )
    assert len(fake_models[-1]) == len(ids) - 1
    assert all("NEVER_SUMMARIZE_THIS" not in task.prompt for task in fake_models[-1])
    if representation == "graph":
        assert "unparseable warning observation" in fake_models[-1][0].prompt
    failed_summary = read_json(context.summary_arm_dir(source) / f"{ids[1]}.json")
    assert failed_summary["status"] == "failed" and failed_summary["scene_count"] == 0

    errors = []
    expected = {(cid, 0) for cid in ids} | {(ids[0], 1), (ids[0], 2)}
    report, _, valid = _scene_arm_contract(
        target, context.run_root, ids, expected, errors, scene_arms(context.config),
    )
    assert valid and not errors
    assert report["accounted_scene_count"] == len(expected)
    assert report["failure_scene_count"] == (3 if representation == "graph" else 2)
    assert report["missing_scene_count"] == report["overlap_scene_count"] == 0
    recovery = recovery_report(context, branches=[target])[target]
    assert recovery["scene_modes"]["failed"] == 2

    if representation == "graph":
        graph = graph_context(context, "mean")
        cohort = prepare_validation_cohort(graph)
        arm = registry(context.config)[target]
        assert source_identity(graph, cohort, arm)
        texts, _, stats = build_input(graph, cohort, arm)
        assert stats["counts"]["failed"] == 2
        assert stats["counts"]["valid_scenes"] == len(ids) - 1
        assert stats["counts"]["raw"] == 1
        assert all("NEVER_SUMMARIZE_THIS" not in text for text in texts)
        assert all("unparseable warning observation" not in text for text in texts)
