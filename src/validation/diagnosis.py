from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import ValidationConfig
from .io import read_jsonl
from .metrics import paired_bootstrap_ci, paired_relative_bootstrap_ci
from .recommendation import RECOMMENDATION_ARMS


def _mean_by_user(rows: list[dict[str, Any]], users: list[str], seeds: list[int], arm: str, metric: str) -> np.ndarray:
    by_key = {(row["seed"], row["user_id"], row["arm"]): row for row in rows}
    return np.asarray([np.mean([by_key[(seed, user, arm)][metric] for seed in seeds]) for user in users], dtype=np.float64)


def diagnose_recommendations(config: ValidationConfig, runtime: dict[str, Any]) -> dict[str, Any]:
    recommendations_dir = Path(runtime["paths"]["recommendations_dir"])
    metrics_path = recommendations_dir / "per_user_metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError(f"missing per-user metrics: {metrics_path}")
    rows = read_jsonl(metrics_path)
    arms = list(RECOMMENDATION_ARMS)
    users = sorted({row["user_id"] for row in rows})
    seeds = config.model.seeds
    metrics = [f"{name}@{cutoff}" for cutoff in config.evaluation.cutoffs for name in ("HR", "NDCG")]
    summary = {arm: {metric: float(np.mean([row[metric] for row in rows if row["arm"] == arm])) for metric in metrics} for arm in arms}
    catalog_size = len(read_jsonl(Path(runtime["paths"]["cohort_dir"]) / "catalog.jsonl"))
    diagnostics: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        recommended = [item for row in arm_rows for item in row["top_item_ids"]]
        top_one = Counter(row["top_item_ids"][0] for row in arm_rows)
        buckets = sorted({row["target_frequency_bucket"] for row in arm_rows})
        diagnostics[arm] = {
            "top20_coverage": len(set(recommended)) / catalog_size,
            "top1_concentration": max(top_one.values()) / len(arm_rows),
            "frequency_bucket": {
                bucket: {metric: float(np.mean([row[metric] for row in arm_rows if row["target_frequency_bucket"] == bucket])) for metric in metrics}
                for bucket in buckets
            },
        }
    comparisons: dict[str, Any] = {}
    baseline = _mean_by_user(rows, users, seeds, "SASRec_ID", "NDCG@10")
    for arm in arms:
        if arm == "SASRec_ID":
            continue
        treatment = _mean_by_user(rows, users, seeds, arm, "NDCG@10")
        comparisons[f"{arm}-SASRec_ID"] = paired_bootstrap_ci(treatment - baseline, samples=config.evaluation.bootstrap_samples)
    desc = "SASRec_DESC"
    graph_arms = ("SASRec_GRAPH_QWEN", "SASRec_GRAPH_GEMINI")
    for graph in graph_arms:
        if graph not in arms or desc not in arms:
            continue
        result = paired_relative_bootstrap_ci(
            _mean_by_user(rows, users, seeds, graph, "NDCG@10"),
            _mean_by_user(rows, users, seeds, desc, "NDCG@10"),
            samples=config.evaluation.bootstrap_samples,
        )
        result["non_inferiority_margin"] = -config.evaluation.non_inferiority_margin
        result["non_inferior"] = result["ci_low"] > -config.evaluation.non_inferiority_margin
        comparisons[f"{graph}-{desc}"] = result
    qwen_graph, gemini_graph = graph_arms
    if qwen_graph in arms and gemini_graph in arms:
        comparisons[f"{gemini_graph}-{qwen_graph}"] = paired_relative_bootstrap_ci(
            _mean_by_user(rows, users, seeds, gemini_graph, "NDCG@10"),
            _mean_by_user(rows, users, seeds, qwen_graph, "NDCG@10"),
            samples=config.evaluation.bootstrap_samples,
        )
    expected_rows = len(users) * len(seeds) * len(arms)
    checkpoints_complete = all(
        (
            recommendations_dir
            / "checkpoints"
            / f"seed_{seed}"
            / arm.lower()
            / "sasrec.pt"
        ).is_file()
        for seed in seeds
        for arm in arms
    )
    report_ready = len(users) == config.cohort.user_count and len(rows) == expected_rows and checkpoints_complete
    document = {
        "schema_version": "diagnosis/v1", "run_id": runtime["run_id"], "modality": runtime["modality"],
        "metrics": summary, "diagnostics": diagnostics, "paired_bootstrap": comparisons,
        "report_ready": report_ready,
        "user_count": len(users), "seed_count": len(seeds), "arm_count": len(arms), "checkpoints_complete": checkpoints_complete,
    }
    if not report_ready:
        raise RuntimeError("diagnosis artifacts are incomplete; report_ready=false")
    return document
