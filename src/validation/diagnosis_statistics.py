"""Prespecified families keep their multiplicity even for partial target runs."""

from arm_registry import registry
from validation.recommendation_contracts import DEFAULT_PROTOCOL


def comparison_families(config=None):
    arms = registry(config or DEFAULT_PROTOCOL)
    return {
        "metadata_baseline": [(name, "meta") for name in arms if name != "meta"],
        "representation": [
            ("graph_qwen_meta", "desc_qwen_meta"),
            ("graph_gemini_meta", "desc_gemini_meta"),
        ],
        "title_input": [("graph_qwen_meta", "graph_qwen")],
    }


def multiple_comparison_policy(
    settings, decision_valid=True, metric="NDCG@10", *, arms=None, config=None
):
    names = set(registry(config or DEFAULT_PROTOCOL) if arms is None else arms)
    alpha = settings["familywise_alpha"]
    return {
        "metric": metric,
        "correction": "bonferroni",
        "sides": "two-sided",
        "families": {
            family: {
                "familywise_alpha": alpha,
                "comparison_count": len(pairs),
                "per_comparison_alpha": alpha / len(pairs),
                "computed": [f"{a}-{b}" for a, b in pairs if {a, b} <= names],
                "skipped": [
                    {"comparison": f"{a}-{b}", "reason": "required arm not selected"}
                    for a, b in pairs
                    if not {a, b} <= names
                ],
            }
            for family, pairs in comparison_families(config).items()
        },
        "primary_comparison": "graph_qwen_meta-meta",
        "exploratory": "Teacher reference gap: graph_gemini_meta-graph_qwen_meta; unadjusted 95% interval",
    }
