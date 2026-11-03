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
    targets = ["d", "base", "g1"]
    families = comparison_families(context.config, targets)
    assert families == {
        "selected_targets": [
            ("d", "base"),
            ("g1", "base"),
            ("g1", "d"),
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
    report = read_json(context.validation_dir / "diagnosis" / "base_g1" / "diagnosis.json")
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
    arms = ["base", "d", "g0", "g1", "g2"]
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)

    def command(step, target):
        return main([step, "--run-id", context.run_id, "--dataset", context.dataset,
                     "--target", *target])

    assert command("embed-representations", arms) == 0
    assert command("prepare-module", ["g2"]) == 0
    graph_arms = ["g0", "g1", "g2"]
    assert len({recommendation_identity(context, name)["embedding_hash"] for name in graph_arms}) == 1
    assert len(GraphStore(context.representations_dir / "desc_qwen3vl")["contexts"]) == 48
    d_identity = recommendation_identity(context, "d")
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
    for line, arm, label in zip(console[-5:], arms, ["Base", "D", "G0", "G1", "G2"], strict=True):
        means = document["recommendations"]["means"][arm]
        std = document["recommendations"]["standard_deviations"][arm]
        assert line.split() == [label, f"{means['NDCG@10']:.6f}", "±", f"{std['NDCG@10']:.6f}",
                                f"{means['HR@10']:.6f}", "±", f"{std['HR@10']:.6f}"]
    text = (context.run_root / "reports" / target_set / "report.md").read_text()
    for comparison in ("G0 − D", "G1 − G0", "G2 − G1"):
        assert comparison in text
    assert "장면 확보율" in text and "제목 fallback" in text
    assert_run_grid(context, graph_mode=True)
    assert run_recommendation(context, target=arms)["skipped"] == 15
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    assert not (context.run_root / "run_contract.json").exists()


@pytest.mark.torch
def test_d_runs_without_graph_and_reuses_completed_results(context, monkeypatch):
    from extraction.scene_storage import write_scene_records
    from helpers import FakeEncoder, failure_row, ready_cohort
    from validation.graph_model import new_graph_model
    from validation.representation_provenance import recommendation_identity
    from validation.steps import embed_representations
    import torch

    path = context.path("data", "titles_csv")
    path.write_text(path.read_text().replace("2,TITLE_SECRET_2\n", "2,\n"))
    ready_cohort(context)
    catalog = context.require_ready_cohort()["catalog"]
    for index, item in enumerate(catalog):
        cid = item["content_id"]
        rows = [{"content_id": cid, "scene_idx": 0, "tokens": 4,
                 "warning": [], "description": "A person cooks."}, failure_row(cid, 1)]
        if index in (1, 2):
            rows = [failure_row(cid)]
        write_scene_records(context.scene_arm_dir("desc_qwen3vl") / f"{cid}.jsonl", rows)
    monkeypatch.setattr("validation.features.BGETextEncoder", FakeEncoder)
    embed_representations(context, target=["d"])
    assert not context.scene_arm_dir("graph_qwen3vl").exists()
    assert not (context.representations_dir / "graph_qwen3vl").exists()
    assert not list(context.representations_dir.rglob("scene_segments*"))
    identity = recommendation_identity(context, "d")
    assert identity["representation_mode"] == "description"
    assert identity["experiment_arm"]["name"] == "d"
    assert identity["scene_aggregation"] == "mean"
    assert run_recommendation(context, target=["d"])["completed"] == 3
    assert run_diagnosis(context, target=["d"])["status"] == "pass"
    document = read_json(context.diagnosis_path(["d"]))
    counts = document["representations"]["d"]["statistics"]["counts"]
    assert counts["failed"] > 0 and counts["empty_video"] == 1 and counts["zero_scene_video"] == 2
    assert document["recommendations"]["base_relative_changes"] is None
    assert document["statistics"]["comparison_count"] == 0
    assert document["statistics"]["comparisons"] == {}
    restored = new_graph_model(context, validation_config(context), "d", "cpu").eval()
    directory = context.recommendations_dir / "seed_42" / "d"
    checkpoint = torch.load(directory / "sasrec.pt", map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        np.testing.assert_allclose(restored.catalog_vectors().numpy(), np.load(directory / "catalog_vectors.npy"),
                                   rtol=1e-5, atol=1e-6)
    before = {p: p.read_bytes() for p in context.recommendations_dir.rglob("complete.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("completed combinations must skip model initialization")

    monkeypatch.setattr("validation.graph_model.new_graph_model", forbidden)
    assert run_recommendation(context, target=["d"])["skipped"] == 3
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.torch
def test_arm_registration_changes_preserve_existing_cohort_and_results(context, monkeypatch):
    from artifact_io import write_json
    from validation.selection import cohort_directory, load_validation_cohort

    train_inputs(context, monkeypatch)
    path = cohort_directory(context) / "manifest.json"
    old = read_json(path)
    old["arms"] = ["base", "d0", "d1", "g0", "g1", "g2", "g3", "g4", "g1a", "g1b", "g1c", "g3a"]
    write_json(path, old)
    assert load_validation_cohort(context)["manifest"] == old
    assert run_recommendation(context, target=TARGETS)["skipped"] == 6
    assert read_json(path) == old
    # Changes to actual cohort inputs are still rejected.
    old["statistics"]["removed_event_count"] += 1
    write_json(path, old)
    with pytest.raises(RuntimeError, match="selection inputs changed"):
        load_validation_cohort(context)


def test_old_diagnosis_and_report_are_preserved(context):
    from artifact_io import write_json

    destination = context.diagnosis_path(["base", "g1"])
    write_json(destination, {"recommendations": {"architecture_version": "sasrec-scene-content/v1"}})
    report = context.run_root / "reports" / "base_g1" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("historical result")
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="new Run ID"):
        run_diagnosis(context, target=["base", "g1"], force=True)
    assert destination.read_bytes() == before and report.read_text() == "historical result"
