from __future__ import annotations


TRAINING_RUNS_FILENAME = "training_runs.jsonl"
TRAINING_RUN_SCHEMA_VERSION = "sasrec-training-run/v2"
ARCHITECTURE_VERSION = "sasrec-content-v2"

RECOMMENDATION_ARMS: dict[str, str] = {
    "SASRec_METADATA": "metadata",
    "SASRec_GRAPH_QWEN": "graph_qwen",
    "SASRec_GRAPH_GEMINI": "graph_gemini",
    "SASRec_DESC": "desc",
}

TARGET_SOURCES = {
    "METADATA": "metadata",
    "GRAPH_QWEN": "graph_qwen",
    "GRAPH_GEMINI": "graph_gemini",
    "DESC_QWEN": "desc",
}


def resolve_target_arms(target: list[str] | None = None) -> dict[str, str]:
    if target is None:
        return dict(RECOMMENDATION_ARMS)
    if not target:
        raise ValueError("--target requires at least one source")
    names = {name.upper() for name in target}
    unknown = names - TARGET_SOURCES.keys()
    if unknown:
        raise ValueError(f"unknown target source(s): {', '.join(sorted(unknown))}")
    branches = {TARGET_SOURCES[name] for name in names}
    return {arm: branch for arm, branch in RECOMMENDATION_ARMS.items() if branch in branches}


def target_scope(arms: dict[str, str]) -> dict:
    return {
        "target_sources": [name for name, branch in TARGET_SOURCES.items() if branch in arms.values()],
        "selected_arms": list(arms),
        "excluded_arms": [arm for arm in RECOMMENDATION_ARMS if arm not in arms],
    }
