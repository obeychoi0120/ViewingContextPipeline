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
    # A structurally incompatible checkpoint must be retrained, not resumed.
    from pipeline_runtime import write_json

    old_training_path = next(context.recommendations_dir.rglob("training.json"))
    old_training = read_json(old_training_path)
    old_training["architecture_version"] = "sasrec-content-v2"
    write_json(old_training_path, old_training)
    result = run_rolling(context)
    assert result["completed"] == 1 and result["skipped"] == 104
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
