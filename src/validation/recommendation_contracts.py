from __future__ import annotations


TRAINING_RUNS_FILENAME = "training_runs.jsonl"
TRAINING_RUN_SCHEMA_VERSION = "sasrec-training-run/v1"

RECOMMENDATION_ARMS: dict[str, str | None] = {
    "SASRec_ID": None,
    "SASRec_GRAPH_QWEN": "graph_qwen",
    "SASRec_GRAPH_GEMINI": "graph_gemini",
    "SASRec_DESC": "desc",
}
