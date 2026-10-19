from types import SimpleNamespace

import numpy as np
import pytest

from artifact_io import read_json, read_jsonl, write_jsonl
from helpers import (
    TARGETS,
    assert_run_grid,
    extract_scenes,
    prepare_text,
    refresh_checksums,
    result_identity,
    selected_table,
    train_text,
)
from validation.diagnosis_statistics import comparison_contract, comparison_families
from validation.graph_context import graph_context
from validation.leave_one_out_diagnosis import cluster_bootstrap, collect_metrics, comparisons
from validation.leave_one_out_recommendation import combination_complete
from validation.steps import (
    embed_representations,
    run_diagnosis,
    run_recommendation,
    validation_config,
)


def test_bootstrap_pairs_users_across_arms():
    values = np.array([[0.1, 0.35], [0.6, 0.85], [0.3, 0.55]])
    observed, draws, metadata = cluster_bootstrap(values, np.ones(3), samples=32, seed=42)
    np.testing.assert_allclose(observed, [1 / 3, 1 / 3 + 0.25])
    np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 0.25, rtol=0, atol=1e-12)
    assert metadata["unit"] == "user"
    assert metadata["aggregation"] == "seed mean then equal user mean"
    other = cluster_bootstrap(values, np.ones(3), samples=32, seed=42)
    np.testing.assert_array_equal(draws, other[1])
    assert np.unique(draws[:, 0]).size > 1
    with pytest.raises(ValueError, match="one test event"):
        cluster_bootstrap(values, np.array([1, 2, 1]), samples=32)


def test_selected_pairs_use_bonferroni(context):
    targets = ["graph_qwen3vl_meta", "meta", "graph_qwen3vl"]
    families = comparison_families(context.config, targets)
    assert families == {
        "selected_targets": [
            ("graph_qwen3vl", "meta"),
            ("graph_qwen3vl_meta", "meta"),
            ("graph_qwen3vl_meta", "graph_qwen3vl"),
        ]
    }
    assert comparison_contract(context.config, targets)["comparison_count"] == 3
    observed = np.array([0.2, 0.1, 0.4])  # same order as the requested targets
    draws = observed + np.arange(32)[:, None] * np.array([0.001, 0.002, -0.001])
    report = comparisons(
        observed, draws, SimpleNamespace(familywise_alpha=0.05), arms=targets, config=context.config
    )
    assert len(report) == 3
    for name, item in report.items():
        left, right = name.split("-")
        expected = observed[targets.index(left)] - observed[targets.index(right)]
        assert item["family_size"] == 3
        assert item["difference"] == pytest.approx(expected)
        assert item["confidence_level"] == pytest.approx(1 - 0.05 / 3)
        delta = draws[:, targets.index(left)] - draws[:, targets.index(right)]
        low, high = np.quantile(delta, [0.05 / 6, 1 - 0.05 / 6])
        assert item["ci_low"] == pytest.approx(low)
        assert item["ci_high"] == pytest.approx(high)


@pytest.mark.torch
def test_diagnosis_rejects_metric_rank_mismatch(context, monkeypatch):
    train_text(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    config = validation_config(context)
    directory = context.recommendations_dir / "seed_42" / "meta"
    path = directory / "per_event_metrics.jsonl"
    rows = read_jsonl(path)
    rows[0]["NDCG@10"] += 0.125
    write_jsonl(path, rows)
    refresh_checksums(directory)
    # Structural completion still passes; the diagnosis must inspect metric semantics.
    assert combination_complete(directory, result_identity(context, cohort), 3)
    with pytest.raises(ValueError, match="metric/rank mismatch"):
        collect_metrics(context, config, cohort, arms={name: name for name in TARGETS})
    with pytest.raises(RuntimeError, match="diagnosis failed"):
        run_diagnosis(context, target=TARGETS)
    report = read_json(context.diagnosis_path)
    assert report["runtime_decision"]["status"] == "fail"
    assert report["statistics"]["status"] == "not_computed"
    assert "metric/rank mismatch" in report["runtime_decision"]["errors"][0]["message"]


@pytest.mark.torch
def test_text_pipeline_cpu_roundtrip(context, monkeypatch):
    scene_generator, summary_generator = prepare_text(context, monkeypatch)
    assert len(scene_generator.tasks) == 48
    assert len(summary_generator.tasks) == 24
    assert all("TITLE_SECRET" not in t.prompt for t in summary_generator.tasks)
    result = run_recommendation(context, target=TARGETS)
    assert result["completed"] == 6 and result["skipped"] == 0
    assert_run_grid(context)
    assert run_diagnosis(context, target=TARGETS)["status"] == "pass"
    report = read_json(context.diagnosis_path)
    assert report["runtime_decision"] == {"status": "pass", "errors": []}
    cohort, _, _ = selected_table(context)
    _, _, collected = collect_metrics(
        context, validation_config(context), cohort, arms={name: name for name in TARGETS}
    )
    assert collected["actual_event_count"] == collected["expected_event_count"] == 18
    assert collected["combination_count"] == 6
    assert report["recommendations"] == {"means": collected["means"]}
    assert report["statistics"]["comparison_count"] == 1
    before = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes())
        for p in context.recommendations_dir.rglob("complete.json")
    }

    def forbidden_training(*args, **kwargs):
        raise AssertionError("a completed combination was retrained")

    monkeypatch.setattr(
        "validation.leave_one_out_recommendation.run_combination", forbidden_training
    )
    reused = run_recommendation(context, target=TARGETS)
    assert reused["completed"] == 0 and reused["skipped"] == 6
    assert before == {
        str(p): (p.stat().st_mtime_ns, p.read_bytes())
        for p in context.recommendations_dir.rglob("complete.json")
    }
    assert run_diagnosis(context, target=TARGETS)["status"] == "pass"


@pytest.mark.torch
def test_graph_pipeline_cpu_roundtrip(context, monkeypatch):
    extract_scenes(context, monkeypatch)
    assert not (context.run_root / "extraction" / "summaries").exists()
    embedded = embed_representations(context, target=TARGETS, representation_mode="graph")
    assert set(embedded["generated_arms"]) == set(TARGETS)
    result = run_recommendation(
        context, target=TARGETS, representation_mode="graph", scene_aggregation="mean"
    )
    assert result["completed"] == 6
    graph = graph_context(context, "mean")
    assert_run_grid(graph, graph_mode=True)
    assert (
        run_diagnosis(
            context, target=TARGETS, representation_mode="graph", scene_aggregation="mean"
        )["status"]
        == "pass"
    )
    report = read_json(graph.diagnosis_path)
    assert report["representation_mode"] == "graph"
    assert report["scene_aggregation"] == "mean"
    cohort, _, _ = selected_table(graph)
    _, _, collected = collect_metrics(
        graph, validation_config(graph), cohort, arms={name: name for name in TARGETS}
    )
    assert collected["actual_event_count"] == collected["expected_event_count"] == 18
    assert collected["combination_count"] == 6
    assert report["recommendations"] == {"means": collected["means"]}
    assert report["statistics"]["comparison_count"] == 1
    assert report["runtime_decision"] == {"status": "pass", "errors": []}
    assert not (context.run_root / "extraction" / "summaries").exists()
