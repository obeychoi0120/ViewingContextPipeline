from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from pipeline_runtime import read_json, read_jsonl, write_json
from test_four_arm_diagnosis import DECISION_CONFIG, _valid_run
from test_rolling import full_context as full_context
from test_rolling_parallel import prepare_embeddings
from validation.diagnosis import diagnose_recommendations
from validation.diagnosis_scenes import SCENE_ARMS
from validation.recommendation_contracts import (
    RECOMMENDATION_ARMS, TARGET_SOURCES, resolve_target_arms,
)
from validation.rolling_diagnosis import comparisons, diagnose
from validation.rolling_recommendation import combination_dir, run_rolling
from validation.steps import validation_config


SUBSETS = [list(group) for size in range(1, 5) for group in combinations(TARGET_SOURCES, size)]


@pytest.mark.parametrize("target", SUBSETS)
def test_legacy_diagnosis_accepts_every_nonempty_subset(tmp_path, target):
    config, runtime, _, _ = _valid_run(tmp_path)
    config.evaluation.bootstrap_samples = 20
    full = diagnose_recommendations(config, runtime, DECISION_CONFIG)
    arms = resolve_target_arms(target)
    root = Path(runtime["run_root"])
    for arm, branch in RECOMMENDATION_ARMS.items():
        if arm in arms:
            continue
        (root / "validation/representations" / f"{branch}_embeddings.npz").unlink()
        for seed in config.model.seeds:
            (root / "validation/recommendations/checkpoints" / f"seed_{seed}"
             / arm.lower() / "sasrec.pt").unlink()
        if branch in SCENE_ARMS:
            for path in (root / SCENE_ARMS[branch][0]).glob("*.jsonl"):
                path.unlink()
    if target == ["METADATA"]:
        for path in (root / "data/cohort/source_assets").rglob("timestamp_fixed_30s.json"):
            path.unlink()
    # Shared metric/training files still contain other arms. They must be ignored.
    result = diagnose_recommendations(config, runtime, DECISION_CONFIG, arms=arms)
    assert result["runtime_decision"]["status"] == "pass", result["runtime_decision"]
    assert result["statistical_analysis"]["status"] == "computed"
    assert set(result["metrics"]) == set(arms)
    expected = {key: value for key, value in full["comparisons"].items()
                if all(arm in arms for arm in key.split("-"))}
    assert result["comparisons"] == expected
    assert result["statistical_analysis"]["expected_comparison_count"] == len(expected)
    assert result["artifact_integrity"]["recommendation_grid"]["expected_cell_count"] == 6 * len(arms)
    assert set(result["scene_coverage"]["arms"]) == set(arms.values()) - {"metadata"}
    assert set(result["excluded_arms"]) == set(RECOMMENDATION_ARMS) - set(arms)


@pytest.mark.parametrize("target", SUBSETS)
def test_rolling_dispatch_checks_and_schedules_only_selected_sources(full_context, monkeypatch, target):
    context = full_context
    prepare_embeddings(context)
    arms = resolve_target_arms(target)
    for arm, branch in RECOMMENDATION_ARMS.items():
        if arm not in arms:
            (context.representations_dir / f"{branch}_embeddings.npz").unlink()
            write_json(context.representations_dir / ".inputs" / f"{branch}.pending", {})
    monkeypatch.setattr("validation.rolling_recommendation.worker_devices", lambda *_: ["cpu", "cpu"])
    jobs_seen = []

    def capture(context, jobs, devices, progress):
        jobs_seen.extend(jobs)
        assert progress.total == 21 * len(arms)
        return len(jobs)

    monkeypatch.setattr("validation.rolling_workers.run_parallel", capture)
    result = run_rolling(context, target=target)
    assert result["completed"] == 21 * len(arms)
    assert {identity["arm"] for _, identity, _ in jobs_seen} == set(arms)
    assert {branch for _, _, branch in jobs_seen} == set(arms.values())
    # Arbitrary input order/duplicates never create duplicate training jobs.
    assert resolve_target_arms(target[::-1] + target) == arms


@pytest.mark.parametrize("target", SUBSETS)
def test_rolling_subset_comparisons_keep_full_family_correction(target):
    from types import SimpleNamespace

    settings = SimpleNamespace(familywise_alpha=0.05, non_inferiority_margin=0.05)
    observed = np.array([0.2, 0.3, 0.4, 0.25])
    draws = observed + np.random.default_rng(42).normal(0, 0.01, (200, 4))
    full = comparisons(observed, draws, settings)
    arms = resolve_target_arms(target)
    indices = [list(RECOMMENDATION_ARMS).index(arm) for arm in arms]
    actual = comparisons(observed[indices], draws[:, indices], settings, arms=arms)
    assert actual == {key: value for key, value in full.items()
                      if all(arm in arms for arm in key.split("-"))}


@pytest.mark.torch
def test_metadata_only_real_cpu_training_resume_and_diagnosis(full_context, monkeypatch):
    import torch
    from validation.model import SASRec
    from validation.steps import run_diagnosis, run_recommendation

    context = full_context
    prepare_embeddings(context)
    for branch in ("graph_qwen", "graph_gemini", "desc"):
        (context.representations_dir / f"{branch}_embeddings.npz").unlink()
    (context.representations_dir / "graph_gemini_fallbacks.json").unlink()
    # Excluded corrupt recovery files must not be inspected by diagnosis.
    recovery = context.graph_scene_dir("gemini") / ".recovery" / "bad.json"
    recovery.parent.mkdir(parents=True)
    recovery.write_text("invalid json")
    from validation.steps import _embedding_documents, _embedding_work
    from validation.representation_provenance import finish_write, input_hash
    catalog = context.require_ready_cohort()["catalog"]
    config = validation_config(context)
    sources, _, _ = _embedding_work(context, catalog, config, False, branches={"metadata"})
    docs = _embedding_documents(context, catalog, sources, ["metadata"], set())["metadata"]
    signature = input_hash(docs, catalog, {
        **config.encoder.model_dump(mode="json"), "model": str(context.path("models", "bge")),
    })
    finish_write(context, "metadata", signature, None)

    def tiny_model(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm="metadata", item_features=features).to(device)

    monkeypatch.setattr("validation.rolling_recommendation._new_model", tiny_model)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_recommendation(context, target=["METADATA"])["completed"] == 21
        protected = {path: path.read_bytes() for path in context.recommendations_dir.rglob("complete.json")}
        assert run_recommendation(context, target=["METADATA"])["skipped"] == 21
        assert all(path.read_bytes() == data for path, data in protected.items())
        excluded = combination_dir(context, "2022-09-05", 42, "SASRec_GRAPH_QWEN") / "complete.json"
        excluded.parent.mkdir(parents=True, exist_ok=True)
        excluded.write_bytes(b"excluded result")
        assert run_recommendation(context, target=["METADATA"], force=True)["completed"] == 21
        assert excluded.read_bytes() == b"excluded result"
        assert run_diagnosis(context, target=["METADATA"])["status"] == "pass"
        report = read_json(context.diagnosis_path)
        assert report["selected_arms"] == ["SASRec_METADATA"]
        assert report["scene_coverage"]["status"] == "not_applicable"
        assert report["generation_recovery"] == {}
        assert report["statistics"]["comparisons"] == {}
        assert report["recommendations"]["combination_count"] == 21
        assert "gemini_summary_fallbacks" not in report
        with pytest.raises(RuntimeError, match="diagnosis failed"):
            diagnose(context)  # Omission still requires all sources.
    finally:
        torch.set_num_threads(threads)


@pytest.mark.torch
def test_legacy_subset_training_preserves_other_arm_rows_and_checkpoints(tmp_path, monkeypatch):
    import torch
    from validation.model import SASRec
    from validation.recommendation import train_recommendation_arms

    config, runtime, _, _ = _valid_run(tmp_path)
    config.model.max_epochs = 1
    root = Path(runtime["run_root"])
    representations = root / "validation/representations"
    runtime["paths"]["representations_dir"] = str(representations)
    output = Path(runtime["paths"]["recommendations_dir"])
    protected = {path: path.read_bytes() for path in output.rglob("sasrec.pt")
                 if path.parent.name != "sasrec_desc"}
    for branch in ("metadata", "graph_qwen", "graph_gemini"):
        (representations / f"{branch}_embeddings.npz").unlink()
    before_rows = {name: [row for row in read_jsonl(output / name) if row["arm"] != "SASRec_DESC"]
                   for name in ("per_user_metrics.jsonl", "training_runs.jsonl")}

    def tiny_model(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm="desc", item_features=features).to(device)

    monkeypatch.setattr("validation.recommendation._new_model", tiny_model)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_recommendation_arms(config, runtime, branches={"desc"})
    finally:
        torch.set_num_threads(threads)
    for name, rows in before_rows.items():
        assert [row for row in read_jsonl(output / name) if row["arm"] != "SASRec_DESC"] == rows
    assert all(path.read_bytes() == data for path, data in protected.items())


def test_legacy_target_resume_expansion_and_force_scope(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import validation.steps as steps

    config, runtime, _, _ = _valid_run(tmp_path)
    root = Path(runtime["run_root"])
    context = SimpleNamespace(
        run_id="test", config={"schema_version": "viewing-context-config/v3"},
        initialize=lambda: None, require_ready_cohort=lambda: {},
        recommendations_dir=Path(runtime["paths"]["recommendations_dir"]),
        representations_dir=root / "validation/representations",
    )
    monkeypatch.setattr(steps, "validation_config", lambda _: config)
    monkeypatch.setattr(steps, "_runtime", lambda _: runtime)
    calls = []
    monkeypatch.setattr("validation.recommendation.train_recommendation_arms",
                        lambda *args, **kwargs: calls.append(kwargs["branches"]))
    # No tracked input dependencies; a healthy selected arm is reusable by itself.
    steps.run_recommendation(context, target=["METADATA"])
    assert calls == []
    missing = steps._checkpoint_paths(context, {"SASRec_DESC"})[0]
    missing.unlink()
    steps.run_recommendation(context, target=["METADATA", "DESC_QWEN"])
    assert calls == [{"desc"}]
    steps.run_recommendation(context, target=["METADATA"], force=True)
    assert calls[-1] == {"metadata"}
