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
    targets = ["g1", "base", "g0"]
    families = comparison_families(context.config, targets)
    assert families == {
        "selected_targets": [
            ("g0", "base"),
            ("g1", "base"),
            ("g1", "g0"),
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
    report = read_json(context.validation_dir / "diagnosis" / "base_g0" / "diagnosis.json")
    assert report["runtime_decision"]["status"] == "fail"
    assert report["statistics"]["status"] == "not_computed"
    assert "metric/rank mismatch" in report["runtime_decision"]["errors"][0]["message"]
    assert "[Diagnosis] Arm별 최종 test 지표" not in capsys.readouterr().out


@pytest.mark.torch
def test_basic_then_ablation_cpu_roundtrip(context, monkeypatch, capsys):
    from arm_registry import RECOMMENDATION_ARM_CONTRACT
    from pipeline_runtime import RunContext
    from validation.cli import main
    from validation.graph_inputs import GraphStore
    from validation.representation_provenance import recommendation_identity

    graph_generator, description_generator = prepare_inputs(context, monkeypatch)
    assert len(graph_generator.tasks) == len(description_generator.tasks) == 48
    assert not (context.run_root / "extraction" / "summaries").exists()
    basic = ["base", "d0", "d1", "g0", "g1", "g2", "g3", "g4"]
    ablation = ["g1a", "g1b", "g1c", "g3a"]
    all_arms = basic + ablation
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)

    def command(step, target):
        return main([step, "--run-id", context.run_id, "--dataset", context.dataset,
                     "--target", *target])

    assert command("embed-representations", basic) == 0
    assert command("prepare-module", ["g3"]) == 0
    assert recommendation_identity(context, "d0")["embedding_hash"] == (
        recommendation_identity(context, "d1")["embedding_hash"]
    )
    graph_arms = basic[3:] + ablation[:3]
    assert len({recommendation_identity(context, name)["embedding_hash"] for name in graph_arms}) == 1
    assert len(GraphStore(context.representations_dir / "desc_qwen3vl")["contexts"]) == 48
    source_bytes = {p: p.read_bytes() for p in context.representations_dir.rglob("*") if p.is_file()}

    def forbidden_preparation(*args, **kwargs):
        raise AssertionError("training attempted feature preparation or extraction")

    monkeypatch.setattr("validation.graph_inputs.prepare", forbidden_preparation)
    monkeypatch.setattr("validation.features.BGETextEncoder", forbidden_preparation)
    assert command("run-recommendation", basic) == 0
    assert len(list(context.recommendations_dir.rglob("complete.json"))) == 24
    assert command("run-diagnosis", basic) == 0
    basic_set = "_".join(basic)
    basic_path = context.validation_dir / "diagnosis" / basic_set / "diagnosis.json"
    report = read_json(basic_path)
    assert report["targets"] == basic
    assert report["statistics"]["comparison_count"] == 28
    assert report["recommendations"]["actual_event_count"] == 72
    assert report["arm_contract"] == RECOMMENDATION_ARM_CONTRACT
    before = {p: p.read_bytes() for p in context.recommendations_dir.rglob("complete.json")}
    assert command("prepare-module", ["g3a"]) == 0
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    assert recommendation_identity(context, "g3a")["embedding_hash"] == (
        recommendation_identity(context, "g3")["embedding_hash"]
    )
    assert command("run-recommendation", ablation) == 0
    assert len(list(context.recommendations_dir.rglob("complete.json"))) == 36
    assert all(path.read_bytes() == content for path, content in before.items())
    capsys.readouterr()
    assert command("run-diagnosis", ablation) == 0
    console = capsys.readouterr().out.strip().splitlines()
    ablation_set = "_".join(ablation)
    document = read_json(context.validation_dir / "diagnosis" / ablation_set / "diagnosis.json")
    assert document["targets"] == ablation
    assert console[-5].split() == ["Arm", "NDCG@10", "HR@10"]
    for line, arm, label in zip(console[-4:], ablation, ["G1a", "G1b", "G1c", "G3a"], strict=True):
        means = document["recommendations"]["means"][arm]
        assert line.split() == [label, f"{means['NDCG@10']:.6f}", f"{means['HR@10']:.6f}"]
    assert document["statistics"]["comparison_count"] == len(document["statistics"]["comparisons"]) == 6
    recommendations = document["recommendations"]
    assert recommendations["actual_event_count"] == 36
    assert recommendations["base_relative_changes"] is None
    assert "standard_deviations" in recommendations
    assert recommendations["model_reports"]["g1a"]["parameters"]["removed_role_fusion"] == 83072
    parameters = recommendations["model_reports"]["g1c"]["parameters"]
    assert parameters["active_trainable"] < parameters["trainable"]
    assert recommendations["model_reports"]["g1b"]["role_corruption"]["changed_ratio"] == 1
    assert read_json(basic_path) == report
    assert (context.run_root / "reports" / basic_set / "report.md").is_file()
    assert (context.run_root / "reports" / ablation_set / "report.md").is_file()
    assert_run_grid(context, graph_mode=True)
    assert run_recommendation(context, target=all_arms)["skipped"] == 36
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    assert not (context.run_root / "run_contract.json").exists()


@pytest.mark.torch
def test_ablation_can_run_first(context, monkeypatch):
    prepare_inputs(context, monkeypatch)
    context.config["validation"]["model"].update(max_epochs=1)
    assert run_recommendation(context, target=["g1c"])["completed"] == 3
    assert run_diagnosis(context, target=["g1c"])["status"] == "pass"
    report = read_json(context.validation_dir / "diagnosis" / "g1c" / "diagnosis.json")
    assert report["recommendations"]["base_relative_changes"] is None
    assert report["statistics"]["comparison_count"] == 0
    assert report["statistics"]["comparisons"] == {}
