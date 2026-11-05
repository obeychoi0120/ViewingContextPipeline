from types import SimpleNamespace

import numpy as np
import pytest

from artifact_io import read_json, read_jsonl, write_jsonl
from helpers import (
    TARGETS,
    assert_run_grid,
    prepare_inputs,
    refresh_checksums,
    result_identity,
    selected_table,
    train_inputs,
)
from validation.diagnosis_statistics import comparison_contract, comparison_families
from validation.leave_one_out_diagnosis import cluster_bootstrap, collect_metrics, comparisons
from validation.leave_one_out_recommendation import combination_complete
from validation.steps import (
    run_diagnosis,
    run_recommendation,
    validation_config,
)


def test_paired_bootstrap_and_bonferroni_comparisons(context):
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

    targets = ["description", "base", "graph-activity"]
    families = comparison_families(context.config, targets)
    assert families == {
        "selected_targets": [
            ("description", "base"),
            ("graph-activity", "base"),
            ("graph-activity", "description"),
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
        left, right = item["left"], item["right"]
        expected = observed[targets.index(left)] - observed[targets.index(right)]
        assert item["family_size"] == 3
        assert item["difference"] == pytest.approx(expected)
        assert item["confidence_level"] == pytest.approx(1 - 0.05 / 3)
        delta = draws[:, targets.index(left)] - draws[:, targets.index(right)]
        low, high = np.quantile(delta, [0.05 / 6, 1 - 0.05 / 6])
        assert item["ci_low"] == pytest.approx(low)
        assert item["ci_high"] == pytest.approx(high)


@pytest.mark.torch
def test_diagnosis_rejects_metric_rank_mismatch(context, monkeypatch, capsys):
    train_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    config = validation_config(context)
    directory = context.recommendations_dir / "seed_42" / "base"
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
    report = read_json(context.validation_dir / "diagnosis" / "base_graph-activity" / "diagnosis.json")
    assert report["runtime_decision"]["status"] == "fail"
    assert report["statistics"]["status"] == "not_computed"
    assert "metric/rank mismatch" in report["runtime_decision"]["errors"][0]["message"]
    assert "[Diagnosis] Arm별 최종 test 지표" not in capsys.readouterr().out


@pytest.mark.torch
def test_five_arm_cpu_roundtrip(context, monkeypatch, capsys):
    from arm_registry import RECOMMENDATION_ARM_CONTRACT
    from pipeline_runtime import RunContext
    from validation.cli import main
    from validation.graph_inputs import GraphStore
    from validation.representation_provenance import recommendation_identity

    graph_generator, description_generator = prepare_inputs(context, monkeypatch)
    assert len(graph_generator.tasks) == len(description_generator.tasks) == 48
    arms = ["base", "description", "graph", "graph-activity", "graph-role-activity"]
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)

    def command(step, target):
        return main([step, "--run-id", context.run_id, "--dataset", context.dataset])

    assert command("embed-representations", arms) == 0
    assert command("prepare-module", ["graph-role-activity"]) == 0
    graph_arms = ["graph", "graph-activity", "graph-role-activity"]
    assert len({recommendation_identity(context, name)["embedding_hash"] for name in graph_arms}) == 1
    assert len(GraphStore(context.representations_dir / "desc_qwen3vl")["contexts"]) == 48
    d_identity = recommendation_identity(context, "description")
    assert d_identity["item_model"]["description_projection_dim"] == 384
    assert "scene_segments" not in d_identity
    source_bytes = {p: p.read_bytes() for p in context.representations_dir.rglob("*") if p.is_file()}

    def forbidden_preparation(*args, **kwargs):
        raise AssertionError("training attempted feature preparation or extraction")

    monkeypatch.setattr("validation.graph_inputs.prepare", forbidden_preparation)
    monkeypatch.setattr("validation.features.BGETextEncoder", forbidden_preparation)
    assert command("run-recommendation", arms) == 0
    assert len(list(context.recommendations_dir.rglob("complete.json"))) == 15
    capsys.readouterr()
    assert command("run-diagnosis", arms) == 0
    console = capsys.readouterr().out.strip().splitlines()
    target_set = "_".join(arms)
    document = read_json(context.diagnosis_path(arms))
    assert document["targets"] == arms
    assert document["statistics"]["comparison_count"] == len(document["statistics"]["comparisons"]) == 10
    assert all(value["family_size"] == 10 for value in document["statistics"]["comparisons"].values())
    assert document["recommendations"]["actual_event_count"] == 45
    assert document["arm_contract"] == RECOMMENDATION_ARM_CONTRACT
    assert console[-6].split() == ["Arm", "NDCG@10", "HR@10"]
    for line, arm, label in zip(console[-5:], arms, arms, strict=True):
        means = document["recommendations"]["means"][arm]
        std = document["recommendations"]["standard_deviations"][arm]
        assert line.split() == [label, f"{means['NDCG@10']:.6f}", "±", f"{std['NDCG@10']:.6f}",
                                f"{means['HR@10']:.6f}", "±", f"{std['HR@10']:.6f}"]
    text = (context.run_root / "reports" / target_set / "report.md").read_text()
    for comparison in ("graph − description", "graph-activity − graph", "graph-role-activity − graph-activity"):
        assert comparison in text
    assert "장면 확보율" in text and "제목 fallback" in text
    # A remaining Markdown report must not prevent rebuilding missing diagnosis evidence.
    context.diagnosis_path(arms).unlink()
    assert command("run-diagnosis", arms) == 0
    rebuilt = read_json(context.diagnosis_path(arms))
    assert rebuilt == document
    assert (context.run_root / "reports" / target_set / "report.md").read_text() == text
    assert_run_grid(context, graph_mode=True)
    assert run_recommendation(context, target=arms)["skipped"] == 15
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    assert not (context.run_root / "run_contract.json").exists()
