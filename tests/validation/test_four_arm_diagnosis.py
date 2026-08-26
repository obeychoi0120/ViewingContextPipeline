from __future__ import annotations

import json

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
    metrics_path = run_root / "validation" / "recommendations" / "metrics.jsonl"
    write_jsonl(metrics_path, rows)
    runs = []
    for seed in config.model.seeds:
        for arm in scores:
            checkpoint = run_root / "checkpoints" / str(seed) / arm / "sasrec.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            runs.append({"seed": seed, "arm": arm, "checkpoint": str(checkpoint)})
    recommendation_path = run_root / "validation" / "recommendations" / "manifest.json"
    recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation_path.write_text(
        json.dumps({
            "schema_version": "recommendations/v1",
            "modality": "visual_only",
            "arms": list(scores),
            "per_user_metrics": str(metrics_path),
            "runs": runs,
            "fingerprint": "recommendations-fp",
            "complete": True,
        }),
        encoding="utf-8",
    )
    runtime = {
        "run_id": "test",
        "modality": "visual_only",
        "paths": {
            "recommendations_manifest": str(recommendation_path),
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
