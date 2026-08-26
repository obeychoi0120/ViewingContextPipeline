from __future__ import annotations

from validation.config import ValidationConfig
from validation.diagnosis import diagnose_recommendations
from validation.recommendation import RECOMMENDATION_ARMS
from viewing_context_pipeline.runtime import write_jsonl

from conftest import config_data


def test_four_arm_contract_and_paired_comparisons(tmp_path) -> None:
    assert RECOMMENDATION_ARMS == {
        "SASRec_ID": None,
        "SASRec_GRAPH_QWEN": "graph_qwen",
        "SASRec_GRAPH_GEMINI": "graph_gemini",
        "SASRec_DESC": "desc",
    }
    config = ValidationConfig.model_validate(config_data(tmp_path, users=2))
    run_root = tmp_path / "run"
    cohort_dir = run_root / "data" / "cohort"
    write_jsonl(
        cohort_dir / "catalog.jsonl",
        [
            {"content_id": "c1", "item_id": "i1"},
            {"content_id": "c2", "item_id": "i2"},
        ],
    )
    scores = {
        "SASRec_ID": 0.30,
        "SASRec_GRAPH_QWEN": 0.40,
        "SASRec_GRAPH_GEMINI": 0.45,
        "SASRec_DESC": 0.42,
    }
    rows = []
    for seed in config.model.seeds:
        for user_id in ("u1", "u2"):
            for arm, score in scores.items():
                rows.append({
                    "seed": seed,
                    "user_id": user_id,
                    "arm": arm,
                    "target_frequency_bucket": "warm",
                    "top_item_ids": ["i1", "i2"],
                    **{
                        f"{metric}@{cutoff}": score
                        for cutoff in config.evaluation.cutoffs
                        for metric in ("HR", "NDCG")
                    },
                })
    recommendations_dir = run_root / "validation" / "recommendations"
    metrics_path = recommendations_dir / "per_user_metrics.jsonl"
    write_jsonl(metrics_path, rows)
    for seed in config.model.seeds:
        for arm in scores:
            checkpoint = recommendations_dir / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
    runtime = {
        "run_id": "test",
        "modality": "visual_only",
        "paths": {
            "recommendations_dir": str(recommendations_dir),
            "cohort_dir": str(cohort_dir),
        },
    }

    result = diagnose_recommendations(config, runtime)

    assert result["report_ready"] is True
    assert result["arm_count"] == 4
    assert set(result["paired_bootstrap"]) == {
        "SASRec_GRAPH_QWEN-SASRec_ID",
        "SASRec_GRAPH_GEMINI-SASRec_ID",
        "SASRec_DESC-SASRec_ID",
        "SASRec_GRAPH_QWEN-SASRec_DESC",
        "SASRec_GRAPH_GEMINI-SASRec_DESC",
        "SASRec_GRAPH_GEMINI-SASRec_GRAPH_QWEN",
    }
