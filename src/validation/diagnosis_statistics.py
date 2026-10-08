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
