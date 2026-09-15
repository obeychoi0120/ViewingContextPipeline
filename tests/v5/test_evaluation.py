from types import SimpleNamespace

import numpy as np
import pytest

from arm_registry import registry
from pipeline_runtime import read_json
from validation.rolling_diagnosis import comparisons
from validation.diagnosis_statistics import multiple_comparison_policy
from validation.steps import embed_representations
from test_pipeline import generate_all


def test_fixed_families_partial_targets_zero_control_and_effects(v5_context):
    names = list(registry(v5_context.config))
    observed = np.array([0.3, 0.2, 0.4, 0.3, 0])
    draws = np.tile(observed, (100, 1))
    settings = SimpleNamespace(familywise_alpha=0.05)
    all_results = comparisons(observed, draws, settings, arms=names, config=v5_context.config)
    assert len(all_results) == 9
    assert all_results["graph_gemini-desc_gemini"]["difference"] == pytest.approx(0.1)
    assert all_results["desc_gemini-metadata"]["relative_difference"] is None
    assert all_results["desc_gemini-metadata"]["superior"] is True
    for name, row in all_results.items():
        if row["role"] == "confirmatory":
            assert row["confidence_level"] == 1 - 0.05 / (
                4 if row["family"] == "metadata_baseline" else 2
            )
    subset = ["graph_gemini", "graph_qwen"]
    indices = [names.index(n) for n in subset]
    results = comparisons(
        observed[indices], draws[:, indices], settings, arms=subset, config=v5_context.config
    )
    assert list(results) == ["graph_gemini-graph_qwen"]
    assert results[next(iter(results))] == all_results[next(iter(results))]
    policy = multiple_comparison_policy(
        {"familywise_alpha": 0.05}, arms=subset, config=v5_context.config
    )
    assert [f["comparison_count"] for f in policy["families"].values()] == [4, 2, 2]
    assert len(policy["families"]["metadata_baseline"]["skipped"]) == 4


def test_bootstrap_paired_clusters_equal_date_and_seeds():
    from validation.rolling_diagnosis import cluster_bootstrap

    counts = np.array([[1, 3], [1, 1]], dtype=float)
    sums = np.array([[[1, 0.5], [0, 0]], [[0, 0], [1, 0.5]]])
    observed, draws, stats = cluster_bootstrap(sums, counts, samples=10000)
    np.testing.assert_allclose(observed, [0.375, 0.1875])
    np.testing.assert_allclose(draws[:, 0], draws[:, 1] * 2)
    assert stats["aggregation"] == "seed mean then equal date mean"
    assert stats["working_bytes_upper_bound"] <= stats["memory_limit_bytes"]


@pytest.mark.torch
def test_105_combinations_real_small_training_resume_and_diagnosis(
    ready_context, fake_models, monkeypatch
):
    torch = pytest.importorskip("torch")
    from validation.model import SASRec
    from validation.rolling_recommendation import run_rolling
    from validation.rolling_diagnosis import diagnose

    torch.set_num_threads(1)
    context = ready_context
    generate_all(context)
    embed_representations(context)

    def tiny(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm="metadata", item_features=features).to(device)

    monkeypatch.setattr("validation.rolling_recommendation._new_model", tiny)
    assert run_rolling(context)["completed"] == 105
    assert run_rolling(context)["skipped"] == 105
    assert len(list(context.recommendations_dir.rglob("sasrec.pt"))) == 105
    for path in context.recommendations_dir.rglob("training.json"):
        assert "refit_item_frequency" not in read_json(path)
    assert diagnose(context)["status"] == "pass"
    document = read_json(context.diagnosis_path)
    assert document["recommendations"]["combination_count"] == 105
    assert (
        len(document["statistics"]["comparisons"]) == 9
    )  # 4 baseline + 2 repr + 2 model + 1 interaction
    assert document["metadata_missing"]["missing_count"] == 1
    assert all(name.islower() for name in document["selected_arms"])
    assert set(document["scene_coverage"]["arms"]) == {
        "desc_qwen",
        "desc_gemini",
        "graph_qwen",
        "graph_gemini",
    }
    assert diagnose(context, target=["metadata"])["status"] == "pass"
    subset = read_json(context.diagnosis_path)
    assert (
        subset["recommendations"]["combination_count"] == 21
        and subset["statistics"]["comparisons"] == {}
    )
