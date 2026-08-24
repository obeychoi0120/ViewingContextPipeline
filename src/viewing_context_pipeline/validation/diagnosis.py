from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ValidationConfig
from .io import fingerprint, read_jsonl
from .metrics import paired_bootstrap_ci, paired_relative_bootstrap_ci


def _mean_by_user(rows: list[dict[str, Any]], users: list[str], seeds: list[int], arm: str, metric: str) -> np.ndarray:
    by_key = {(row["seed"], row["user_id"], row["arm"]): row for row in rows}
    return np.asarray([np.mean([by_key[(seed, user, arm)][metric] for seed in seeds]) for user in users], dtype=np.float64)


def diagnose_recommendations(config: ValidationConfig, runtime: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(runtime["paths"]["recommendations_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("diagnosis requires recommendations/v1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "recommendations/v1" or not manifest.get("complete"):
        raise RuntimeError("diagnosis requires complete recommendations/v1")
    if manifest.get("modality") != runtime["modality"]:
        raise RuntimeError("recommendation modality mismatch")
    rows = read_jsonl(Path(manifest["per_user_metrics"]))
    arms = manifest["arms"]
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
    for backend in ("ONDEVICE", "GEMINI"):
        graph = f"SASRec_{backend}_GRAPH"
        desc = f"SASRec_{backend}_DESC"
        if graph in arms and desc in arms:
            result = paired_relative_bootstrap_ci(
                _mean_by_user(rows, users, seeds, graph, "NDCG@10"),
                _mean_by_user(rows, users, seeds, desc, "NDCG@10"),
                samples=config.evaluation.bootstrap_samples,
            )
            result["non_inferiority_margin"] = -config.evaluation.non_inferiority_margin
            result["non_inferior"] = result["ci_low"] > -config.evaluation.non_inferiority_margin
            comparisons[f"{graph}-{desc}"] = result
    expected_rows = len(users) * len(seeds) * len(arms)
    checkpoints_complete = len(manifest["runs"]) == len(seeds) * len(arms) and all(Path(run["checkpoint"]).is_file() for run in manifest["runs"])
    report_ready = len(users) == config.cohort.user_count and len(rows) == expected_rows and checkpoints_complete
    document = {
        "schema_version": "diagnosis/v1", "run_id": runtime["run_id"], "modality": runtime["modality"],
        "metrics": summary, "diagnostics": diagnostics, "paired_bootstrap": comparisons,
        "recommendations_fingerprint": manifest["fingerprint"], "report_ready": report_ready,
        "user_count": len(users), "seed_count": len(seeds), "arm_count": len(arms), "checkpoints_complete": checkpoints_complete,
    }
    document["fingerprint"] = fingerprint(document)
    if not report_ready:
        raise RuntimeError("diagnosis artifacts are incomplete; report_ready=false")
    return document
