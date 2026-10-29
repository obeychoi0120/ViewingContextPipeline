from arm_registry import RECOMMENDATION_ARM_CONTRACT
from validation.experiment_report import write_report


def test_report_reverses_contrast_interval_and_preserves_tables(context):
    arms = ["base", "g1", "g1a", "g1b", "g1c"]
    means = {arm: {"NDCG@10": 0.2, "HR@10": 0.4} for arm in arms}
    deviations = {arm: {"NDCG@10": 0.01, "HR@10": 0.02} for arm in arms}
    models = {arm: {"parameters": {"trainable": 100, "active_trainable": 90},
                    "role_corruption": None} for arm in arms}
    models["g1b"]["role_corruption"] = {
        "total_links": 10, "changed_links": 8, "changed_ratio": 0.8, "impossible_links": 2,
    }
    document = {
        "run_id": context.run_id, "arm_contract": RECOMMENDATION_ARM_CONTRACT,
        "targets": arms,
        "recommendations": {
            "means": means, "standard_deviations": deviations,
            "base_relative_changes": None, "model_reports": models,
            "per_seed": [{"arm": arm, "seed": 42, **means[arm]} for arm in arms],
        },
        "statistics": {
            "comparison_count": 10,
            "comparisons": {"g1a-g1": {"difference": -0.1, "ci_low": -0.15,
                                       "ci_high": -0.05, "confidence_level": 0.995}},
        },
        "representations": {
            arm: {"statistics": {"counts": {"valid_scenes": 5, "zero_scene_video": 1},
                                 "scene_coverage": 0.5}}
            for arm in arms
        },
    }
    selected = "_".join(arms)
    write_report(context, selected, document)
    text = (context.run_root / "reports" / selected / "report.md").read_text()
    assert "| G1 − G1a |" in text
    assert "+0.100000 | [+0.050000, +0.150000]" in text
    assert "우위 근거 확보" in text and "50.00%" in text
    assert "| Base |" in text and "| BASE |" not in text
    assert text.index("| G1c | 5 |") < text.index("G1b: 연결 10개 중 8개 변경")
