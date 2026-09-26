"""Prespecified families keep their multiplicity even for partial target runs."""

from arm_registry import registry, concat_layout
from validation.recommendation_contracts import DEFAULT_PROTOCOL


def comparison_families(config=None):
    arms = registry(config or DEFAULT_PROTOCOL)
    if concat_layout(config or DEFAULT_PROTOCOL):
        return {
            "metadata_baseline": [(name, "meta") for name in arms if name != "meta"],
            "representation": ([("graph_qwen", "desc_qwen"), ("graph_qwen_meta", "desc_qwen_meta")]
                               if "desc_qwen" in arms else
                               [("graph_qwen_meta", "desc_qwen_meta"), ("graph_gemini_meta", "desc_gemini_meta")]),
            "title_input": ([("graph_qwen_meta", "graph_qwen"), ("desc_qwen_meta", "desc_qwen")]
                            if "desc_qwen" in arms else [("graph_qwen_meta", "graph_qwen")]),
        }
    from arm_registry import legacy_layout
    old = legacy_layout(config or DEFAULT_PROTOCOL)
    baseline = "metadata" if old else "meta"
    by_kind = {(a.representation, a.model, a.uses_title): a.name for a in arms.values()}
    titles = (True,) if old else (False, True)
    metadata = [(name, baseline) for name in arms if name != baseline]
    representations = [(by_kind["graph", model, title], by_kind["description", model, title])
                       for title in titles for model in ("gemini", "qwen")]
    models = [(by_kind[kind, "gemini", title], by_kind[kind, "qwen", title])
              for title in titles for kind in ("description", "graph")]
    result = {"metadata_baseline": metadata, "representation": representations, "model": models}
    if not old:
        result["title_input"] = [(by_kind[kind, model, True], by_kind[kind, model, False])
                                 for kind in ("graph", "description") for model in ("qwen", "gemini")]
    return result


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
        **({"primary_comparison": "graph_qwen_meta-meta"} if concat_layout(config or DEFAULT_PROTOCOL) else {}),
        "exploratory": ("Teacher reference gap: graph_gemini_meta-graph_qwen_meta; unadjusted 95% interval"
                        if concat_layout(config or DEFAULT_PROTOCOL) else
                        "Graph improvement differences between models; unadjusted 95% intervals"),
    }
