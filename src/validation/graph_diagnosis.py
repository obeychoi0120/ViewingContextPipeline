"""Diagnostics restricted to one direct-graph representation and pooling mode."""

from validation.diagnosis_support import write_diagnosis
from validation.recommendation_contracts import resolve_target_arms
from validation.leave_one_out_diagnosis import collect_metrics, cluster_bootstrap, comparisons


def diagnose_graph(context, *, target, compare_run_id=None):
    from validation.selection import load_validation_cohort
    from validation.steps import validation_config

    config = validation_config(context)
    arms = resolve_target_arms(target, config=context.config)
    from validation.diagnosis_statistics import comparison_contract

    contract = comparison_contract(context.config, arms)
    document = {
        "run_id": context.run_id,
        "representation_mode": "graph",
        "scene_aggregation": context.scene_aggregation,
        "statistics": {"status": "not_computed", **contract},
    }
    errors = []
    try:
        cohort = load_validation_cohort(context)
        sums, counts, report = collect_metrics(context, config, cohort, arms=arms)
        document["recommendations"] = report
        observed, draws, _ = cluster_bootstrap(
            sums, counts, samples=config.evaluation.bootstrap_samples
        )
        document["statistics"] = {
            "status": "computed",
            **contract,
            "comparisons": comparisons(
                observed, draws, config.evaluation, arms=arms, config=context.config
            ),
        }
        if compare_run_id is not None:
            from validation.run_comparison import compare_graph_runs

            document["run_comparison"] = compare_graph_runs(context, compare_run_id, target=arms)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        errors.append({"code": "invalid_graph_evidence", "message": str(exc)})
    document["runtime_decision"] = {"status": "fail" if errors else "pass", "errors": errors}
    write_diagnosis(context.diagnosis_path, document)
    if errors:
        raise RuntimeError(f"graph diagnosis failed; see {context.diagnosis_path}")
    return {"stage": "run-diagnosis", "status": "pass"}
