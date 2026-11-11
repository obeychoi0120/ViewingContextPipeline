from arm_registry import RECOMMENDATION_ARM_CONTRACT, registry
from rec_experiments import experiment_definition
from validation.experiment_report import write_report
from validation.recommendation_contracts import ARCHITECTURE_VERSION


def test_report_uses_recorded_definitions_and_all_contrasts(context):
    arms = list(registry(context.config))
    means = {arm: {"NDCG@10": 0.2, "HR@10": 0.4} for arm in arms}
    deviations = {arm: {"NDCG@10": 0.01, "HR@10": 0.02} for arm in arms}
    models = {arm: {"parameters": {"trainable": 100, "active_trainable": 100}} for arm in arms}
    from itertools import combinations

    document = {
        "run_id": context.run_id, "arm_contract": RECOMMENDATION_ARM_CONTRACT,
        "targets": arms,
        "experiments": [experiment_definition(arm) for arm in registry(context.config).values()],
        "recommendations": {
            "architecture_version": ARCHITECTURE_VERSION,
            "means": means, "standard_deviations": deviations,
            "baseline": None, "base_relative_changes": None, "model_reports": models,
            "per_seed": [{"arm": arm, "seed": 42, **means[arm]} for arm in arms],
        },
        "statistics": {
            "comparison_count": 10,
            "comparisons": {f"{later}|{earlier}": {"left": later, "right": earlier,
                "difference": -0.1, "ci_low": -0.15, "ci_high": -0.05,
                "confidence_level": 1 - 0.05 / 10} for earlier, later in combinations(arms, 2)},
        },
        "representations": {
            arm: {"sources": [{"source": "desc_qwen3vl_2b" if arm == "description" else "graph_qwen3vl_2b"}],
                  "statistics": {"counts": {"valid_scenes": 5, "zero_scene_video": 3, "empty_video": 1},
                                 "scene_coverage": 0.5}}
            for arm in arms
        },
    }
    # Rendering must use the saved definitions even after the active configuration changes.
    context.config.pop("_rec_experiments")
    selected = "_".join(arms)
    write_report(context, selected, document)
    text = (context.run_root / "reports" / selected / "report.md").read_text()
    assert "| graph − description |" in text
    assert "-0.100000 | [-0.150000, -0.050000]" in text
    assert "열위 근거 확보" in text and "50.00%" in text
    assert "| base |" in text
    assert "| description | desc_qwen3vl_2b | 5 | 50.00% | 2 | 1 | 100 | 100 |" in text
    assert "activity_hypergraph=segments-v1" in text
    assert "role_fusion=role-fusion/v1" in text
    assert ARCHITECTURE_VERSION in text
    assert text.count("| -0.100000 |") == 10
