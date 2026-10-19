"""Prespecified families keep their multiplicity even for partial target runs."""

from arm_registry import registry
from validation.recommendation_contracts import DEFAULT_PROTOCOL


def comparison_families(config=None):
    arms = registry(config or DEFAULT_PROTOCOL)
    by_kind = {(a.representation, a.model): a.name for a in arms.values()}
    metadata = [(name, "metadata") for name in arms if name != "metadata"]
    representations = []
    for model in ("gemini", "qwen"):
        for left, right in (
            ("graph", "description"),
        ):
            representations.append((by_kind[left, model], by_kind[right, model]))
    models = [
        (by_kind[kind, "gemini"], by_kind[kind, "qwen"])
        for kind in ("description", "graph")
    ]
    return {"metadata_baseline": metadata, "representation": representations, "model": models}


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
        "exploratory": "Graph improvement differences between models; unadjusted 95% intervals",
    }
