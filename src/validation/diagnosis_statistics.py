"""One Bonferroni family containing all pairs of the selected targets."""

from itertools import combinations

from arm_registry import select_arms
from validation.recommendation_contracts import DEFAULT_PROTOCOL

COMPARISON_POLICY = "selected-target-all-pairs/v1"


def comparison_families(config=None, target=None):
    names = list(select_arms(config or DEFAULT_PROTOCOL, target))
    return {"selected_targets": [(later, earlier) for earlier, later in combinations(names, 2)]}


def comparison_contract(config, target):
    names = list(select_arms(config, target))
    return {
        "comparison_policy": COMPARISON_POLICY,
        "targets": names,
        "comparison_count": len(names) * (len(names) - 1) // 2,
        "correction": "bonferroni",
    }
