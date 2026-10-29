"""Compare Arm results from two runs with matched users, events, partitions and seeds."""

from __future__ import annotations
import numpy as np
from artifact_io import fingerprint
from pipeline_runtime import RunContext
from validation.selection import load_validation_cohort
from validation.representation_provenance import read_state


def compare_runs(context, reference_run_id, *, target=None):
    from validation.leave_one_out_diagnosis import cluster_bootstrap, collect_metrics
    from validation.steps import validation_config

    if reference_run_id == context.run_id:
        raise ValueError("comparison requires two different run IDs")
    reference = RunContext.load(reference_run_id, dataset=context.dataset, root=context.root)
    from arm_registry import select_arms

    names = list(select_arms(context.config, target))
    if not names:
        raise ValueError("run comparison requires at least one target")
    arms = {name: name for name in names}
    cohorts = [load_validation_cohort(ctx) for ctx in (context, reference)]
    for key in ("protocol", "preprocessing", "source_data_hash"):
        if cohorts[0]["plan"][key] != cohorts[1]["plan"][key]:
            raise ValueError(f"run comparison requires identical {key}")
    for filename in ("events", "catalog"):
        if fingerprint(cohorts[0][filename]) != fingerprint(cohorts[1][filename]):
            raise ValueError(f"run comparison requires identical {filename}")
    if cohorts[0]["plan"]["splits"] != cohorts[1]["plan"]["splits"]:
        raise ValueError("run comparison requires identical evaluation partitions")
    configs = [validation_config(ctx) for ctx in (context, reference)]
    if sorted(configs[0].model.seeds) != sorted(configs[1].model.seeds):
        raise ValueError("run comparison requires identical model seeds")
    aggregates = [
        collect_metrics(ctx, config, cohort, arms=arms)
        for ctx, config, cohort in zip((context, reference), configs, cohorts, strict=True)
    ]
    left, counts, left_report = aggregates[0]
    right, other_counts, right_report = aggregates[1]
    from validation.recommendation_contracts import EVALUATION_CONTRACT

    if any(report.get("evaluation_contract") != EVALUATION_CONTRACT
           for report in (left_report, right_report)):
        raise ValueError("run comparison requires identical current evaluation contracts")
    grids = [
        {(r["split_hash"], r["seed"], r["arm"]) for r in report["per_seed"]}
        for report in (left_report, right_report)
    ]
    if grids[0] != grids[1] or not np.array_equal(counts, other_counts):
        raise ValueError("run comparison requires identical seeds and eligible events")
    observed, draws, bootstrap = cluster_bootstrap(
        left - right, counts, samples=configs[0].evaluation.bootstrap_samples
    )
    alpha = configs[0].evaluation.familywise_alpha / len(names)
    intervals = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    results = {
        name: {
            "difference": float(observed[i]),
            "ci_low": float(intervals[0, i]),
            "ci_high": float(intervals[1, i]),
            "confidence_level": 1 - alpha,
        }
        for i, name in enumerate(names)
    }
    return {
        "dataset": context.dataset,
        "run_id": context.run_id,
        "reference_run_id": reference.run_id,
        "direction": "run minus reference",
        "metric": "NDCG@10",
        "bootstrap": bootstrap,
        "correction": "bonferroni",
        "family_size": len(names),
        "targets": names,
        "comparisons": results,
        "interpretation": "동일 Arm·분할·seed 비교이며 추출·학습 조건 차이는 provenance를 확인해야 합니다.",
        "sources": {
            ctx.run_id: {name: read_state(ctx, name).get("sources", []) for name in names}
            for ctx in (context, reference)
        },
    }
