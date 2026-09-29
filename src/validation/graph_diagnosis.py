"""Diagnostics restricted to one direct-graph representation and pooling mode."""

from artifact_io import read_json, write_json
from validation.recommendation_contracts import resolve_target_arms, target_scope
from validation.rolling_diagnosis import collect_metrics, cluster_bootstrap, comparisons
from validation.diagnosis_statistics import multiple_comparison_policy


def diagnose_graph(context, *, target, compare_run_id=None):
    from validation.selection import load_validation_cohort
    from validation.steps import validation_config

    config = validation_config(context)
    arms = resolve_target_arms(target, config=context.config)
    document = {
        "schema_version": "graph-diagnosis/v1",
        "run_id": context.run_id,
        "representation_mode": "graph",
        "scene_aggregation": context.scene_aggregation,
        **target_scope(arms, config=context.config),
        "statistics": {"status": "not_computed"},
    }
    errors = []
    try:
        cohort = load_validation_cohort(context)
        document["cohort"] = cohort["plan"]
        document["graph_inputs"] = {
            arm: read_json(context.representations_dir / f"{arm}_embeddings" / "statistics.json")
            for arm in arms
        }
        sums, counts, report = collect_metrics(context, config, cohort, arms=arms)
        document["recommendations"] = report
        observed, draws, bootstrap = cluster_bootstrap(
            sums, counts, samples=config.evaluation.bootstrap_samples
        )
        document["statistics"] = {
            "status": "computed",
            "bootstrap": bootstrap,
            "comparisons": comparisons(
                observed, draws, config.evaluation, arms=arms, config=context.config
            ),
            "policy": multiple_comparison_policy(
                config.evaluation.model_dump(), True, "NDCG@10", arms=arms, config=context.config
            ),
        }
        if compare_run_id is not None:
            from validation.run_comparison import compare_graph_runs

            document["run_comparison"] = compare_graph_runs(context, compare_run_id, target=arms)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        errors.append({"code": "invalid_graph_evidence", "message": str(exc)})
    document["runtime_decision"] = {"status": "fail" if errors else "pass", "errors": errors}
    write_json(context.diagnosis_path, document)
    if errors:
        raise RuntimeError(f"graph diagnosis failed; see {context.diagnosis_path}")
    return {"stage": "run-diagnosis", "status": "pass"}
