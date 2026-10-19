"""Compare graph results from two runs with matched users, events, dates and seeds."""

from __future__ import annotations

import numpy as np

from extraction.recovery import fingerprint
from pipeline_runtime import RunContext, read_jsonl
from validation.representation_provenance import read_state


def compare_graph_runs(context, reference_run_id, *, target=None):
    from validation.rolling_diagnosis import cluster_bootstrap, collect_metrics
    from validation.steps import validation_config

    if reference_run_id == context.run_id:
        raise ValueError("comparison requires two different run IDs")
    reference = RunContext.load(reference_run_id, root=context.root)
    names = [name for name in ("graph_gemini", "graph_qwen") if target is None or name in target]
    if not names:
        raise ValueError("run comparison requires at least one Graph target")
    arms = {name: name for name in names}
    cohorts = [ctx.require_ready_cohort() for ctx in (context, reference)]
    for filename in ("events.jsonl", "catalog.jsonl"):
        if fingerprint(read_jsonl(context.cohort_dir / filename)) != fingerprint(
            read_jsonl(reference.cohort_dir / filename)
        ):
            raise ValueError(f"run comparison requires identical {filename}")
    if cohorts[0]["plan"]["splits"] != cohorts[1]["plan"]["splits"]:
        raise ValueError("run comparison requires identical evaluation dates")
    configs = [validation_config(ctx) for ctx in (context, reference)]
    aggregates = [
        collect_metrics(ctx, config, cohort, arms=arms)
        for ctx, config, cohort in zip((context, reference), configs, cohorts, strict=True)
    ]
    left, counts, left_report = aggregates[0]
    right, other_counts, right_report = aggregates[1]
    grids = [
        [(r["evaluation_date"], r["seed"], r["arm"]) for r in report["daily"]]
        for report in (left_report, right_report)
    ]
    if grids[0] != grids[1] or not np.array_equal(counts, other_counts):
        raise ValueError("run comparison requires identical seeds and eligible events")
    observed, draws, bootstrap = cluster_bootstrap(
        left - right, counts, samples=configs[0].evaluation.bootstrap_samples
    )
    # Two prespecified model comparisons, even when only one model is selected.
    alpha = configs[0].evaluation.familywise_alpha / 2
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
    interaction = None
    if len(names) == 2:
        lo, hi = np.quantile(draws[:, 0] - draws[:, 1], [0.025, 0.975])
        interaction = {
            "role": "exploratory",
            "difference": float(observed[0] - observed[1]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "confidence_level": 0.95,
        }
    return {
        "run_id": context.run_id,
        "reference_run_id": reference.run_id,
        "direction": "run minus reference",
        "metric": "NDCG@10",
        "bootstrap": bootstrap,
        "correction": "bonferroni",
        "family_size": 2,
        "comparisons": results,
        "model_interaction": interaction,
        "interpretation": "Includes differences in extraction, summary prompts, models and fallback; inspect provenance.",
        "sources": {
            ctx.run_id: {name: read_state(ctx, name).get("sources", []) for name in names}
            for ctx in (context, reference)
        },
    }
