from pathlib import Path
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
    from validation.graph_inputs import GraphStore
    from validation.representation_provenance import recommendation_identity

    graph_generator, description_generator = prepare_inputs(context, monkeypatch)
    assert len(graph_generator.tasks) == len(description_generator.tasks) == 48
    assert not (context.run_root / "extraction" / "summaries").exists()
    basic = ["base", "d0", "d1", "g0", "g1"]
    all_arms = basic + ["a1", "a2", "a3"]
    assert embed_representations(context, target=basic)["generated_sources"] == ["desc_qwen3vl"]
    assert recommendation_identity(context, "d0")["embedding_hash"] == recommendation_identity(context, "d1")["embedding_hash"]
    assert len({recommendation_identity(context, name)["embedding_hash"] for name in all_arms[3:]}) == 1
    assert len(GraphStore(context.representations_dir / "desc_qwen3vl")["contexts"]) == 48
    run_contract = (context.run_root / "run_contract.json").read_bytes()
    source_bytes = {str(p): p.read_bytes() for p in context.representations_dir.rglob("*.npy")}

    def forbidden_preparation(*args, **kwargs):
        raise AssertionError("training attempted feature preparation or extraction")

    monkeypatch.setattr("validation.graph_inputs.prepare", forbidden_preparation)
    monkeypatch.setattr("validation.features.BGETextEncoder", forbidden_preparation)
    result = run_recommendation(context, target=basic)
    assert result["completed"] == 15 and result["skipped"] == 0
    assert run_diagnosis(context, target=basic)["status"] == "pass"
    basic_path = context.validation_dir / "diagnosis" / "base_d0_d1_g0_g1" / "diagnosis.json"
    report = read_json(basic_path)
    assert report["statistics"]["comparison_count"] == 10
    assert report["recommendations"]["actual_event_count"] == 45
    assert report["arm_contract"] == RECOMMENDATION_ARM_CONTRACT
    before = {str(p): p.read_bytes() for p in context.recommendations_dir.rglob("complete.json")}
    assert run_recommendation(context, target=["a1", "a2", "a3"])["completed"] == 9
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    capsys.readouterr()
    assert run_diagnosis(context, target=list(reversed(all_arms)))["status"] == "pass"
    console = capsys.readouterr().out.strip().splitlines()
    full_path = context.validation_dir / "diagnosis" / "base_d0_d1_g0_g1_a1_a2_a3" / "diagnosis.json"
    full = read_json(full_path)
    assert console[-9].split() == ["Arm", "NDCG@10", "HR@10"]
    for line, arm in zip(console[-8:], all_arms, strict=True):
        label = "Base" if arm == "base" else arm.upper()
        means = full["recommendations"]["means"][arm]
        assert line.split() == [label, f"{means['NDCG@10']:.6f}", f"{means['HR@10']:.6f}"]
    assert full["statistics"]["comparison_count"] == len(full["statistics"]["comparisons"]) == 28
    assert full["recommendations"]["actual_event_count"] == 72
    assert full["recommendations"]["base_relative_changes"]["base"]["NDCG@10"] in (0.0, None)
    assert "standard_deviations" in full["recommendations"]
    assert full["recommendations"]["model_reports"]["a1"]["parameters"]["removed_role_fusion"] == 83072
    a3 = full["recommendations"]["model_reports"]["a3"]["parameters"]
    assert a3["active_trainable"] < a3["trainable"]
    assert full["recommendations"]["model_reports"]["a2"]["role_corruption"]["changed_ratio"] == 1
    assert read_json(basic_path) == report
    assert (context.run_root / "reports" / "base_d0_d1_g0_g1_a1_a2_a3" / "report.md").is_file()
    assert_run_grid(context, graph_mode=True)
    assert run_recommendation(context, target=all_arms)["skipped"] == 24
    assert source_bytes == {str(p): p.read_bytes() for p in context.representations_dir.rglob("*.npy")}
    assert (context.run_root / "run_contract.json").read_bytes() == run_contract


@pytest.mark.torch
def test_ablation_can_run_first(context, monkeypatch):
    prepare_inputs(context, monkeypatch)
    context.config["validation"]["model"].update(max_epochs=1)
    assert run_recommendation(context, target=["a3"])["completed"] == 3
    assert run_diagnosis(context, target=["a3"])["status"] == "pass"
    report = read_json(context.validation_dir / "diagnosis" / "a3" / "diagnosis.json")
    assert report["recommendations"]["base_relative_changes"] is None
    assert report["statistics"]["comparison_count"] == 0
    assert report["statistics"]["comparisons"] == {}
