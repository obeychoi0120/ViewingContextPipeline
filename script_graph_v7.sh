#!/usr/bin/env bash
# Full Module 2/3 experiment only; no extraction, summary generation, or API calls.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID=${RUN_ID:-260928_v7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
ARMS=(meta graph_qwen graph_qwen_meta graph_gemini_meta)
python -m validation embed-representations --run-id "$RUN_ID" --representation-mode graph --target "${ARMS[@]}"
for POOL in mean attention; do
    python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation "$POOL" --workers-per-gpu 1 --target "${ARMS[@]}"
    python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation "$POOL" --target "${ARMS[@]}"
done
