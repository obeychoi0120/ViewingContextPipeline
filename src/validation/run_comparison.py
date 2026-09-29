"""Compare graph results from two runs with matched users, events, dates and seeds."""

from __future__ import annotations
import numpy as np
from artifact_io import fingerprint
from pipeline_runtime import RunContext
from validation.selection import load_validation_cohort
from validation.representation_provenance import read_state


def compare_graph_runs(context, reference_run_id, *, target=None):
    from validation.rolling_diagnosis import cluster_bootstrap, collect_metrics
    from validation.steps import validation_config

    if reference_run_id == context.run_id:
        raise ValueError("comparison requires two different run IDs")
    reference = RunContext.load(reference_run_id, root=context.root)
    from validation.graph_context import is_graph, graph_context

    if is_graph(context):
        reference = graph_context(reference, context.scene_aggregation)
    from arm_registry import registry

    names = [
        name
        for name, arm in registry(context.config).items()
        if arm.representation == "graph" and (target is None or name in target)
    ]
    if not names:
        raise ValueError("run comparison requires at least one Graph target")
    arms = {name: name for name in names}
    cohorts = [load_validation_cohort(ctx) for ctx in (context, reference)]
    for filename in ("events", "catalog"):
        if fingerprint(cohorts[0][filename]) != fingerprint(cohorts[1][filename]):
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
    alpha = configs[0].evaluation.familywise_alpha / len(
        [a for a in registry(context.config).values() if a.representation == "graph"]
    )
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
    interactions = {}
    interaction = None
    return {
        "run_id": context.run_id,
        "reference_run_id": reference.run_id,
        "direction": "run minus reference",
        "metric": "NDCG@10",
        "bootstrap": bootstrap,
        "correction": "bonferroni",
        "family_size": len(
            [a for a in registry(context.config).values() if a.representation == "graph"]
        ),
        "comparisons": results,
        "model_interaction": interaction,
        **{"model_interactions": interactions},
        "interpretation": "Includes extraction and direct-graph training differences; identical pooling mode required."
        if is_graph(context)
        else "Includes differences in extraction, summary prompts, models and fallback; inspect provenance.",
        **(
            {"representation_mode": "graph", "scene_aggregation": context.scene_aggregation}
            if is_graph(context)
            else {}
        ),
        "sources": {
            ctx.run_id: {name: read_state(ctx, name).get("sources", []) for name in names}
            for ctx in (context, reference)
        },
    }
