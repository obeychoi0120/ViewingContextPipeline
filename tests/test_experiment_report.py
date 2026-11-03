from arm_registry import RECOMMENDATION_ARM_CONTRACT
from validation.experiment_report import write_report


def test_report_reverses_contrast_interval_and_preserves_tables(context):
    arms = ["base", "g1", "g2", "d2", "g3", "g4"]
    means = {arm: {"NDCG@10": 0.2, "HR@10": 0.4} for arm in arms}
    deviations = {arm: {"NDCG@10": 0.01, "HR@10": 0.02} for arm in arms}
    models = {arm: {"parameters": {"trainable": 100, "active_trainable": 100}} for arm in arms}
    document = {
        "run_id": context.run_id, "arm_contract": RECOMMENDATION_ARM_CONTRACT,
        "targets": arms,
        "recommendations": {
            "means": means, "standard_deviations": deviations,
            "base_relative_changes": None, "model_reports": models,
            "per_seed": [{"arm": arm, "seed": 42, **means[arm]} for arm in arms],
        },
        "statistics": {
            "comparison_count": 15,
            "comparisons": {"d2-g2": {"difference": -0.1, "ci_low": -0.15,
                                       "ci_high": -0.05, "confidence_level": 1 - 0.05 / 15}},
        },
        "representations": {
            arm: {"sources": [{"source": "desc_qwen3vl" if arm == "d2" else "graph_qwen3vl"}],
                  "statistics": {"counts": {"valid_scenes": 5, "zero_scene_video": 3, "empty_video": 1},
                                 "scene_coverage": 0.5}}
            for arm in arms
        },
    }
    selected = "_".join(arms)
    write_report(context, selected, document)
    text = (context.run_root / "reports" / selected / "report.md").read_text()
    assert "| G2 − D2 |" in text
    assert "+0.100000 | [+0.050000, +0.150000]" in text
    assert "우위 근거 확보" in text and "50.00%" in text
    assert "| Base |" in text and "| BASE |" not in text
    assert "| D2 | desc_qwen3vl | 5 | 50.00% | 2 | 1 | 100 | 100 |" in text
    assert "G1a" not in text and "교란" not in text
    assert "입력 인코더·파라미터 수" in text
