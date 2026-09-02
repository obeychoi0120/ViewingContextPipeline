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
