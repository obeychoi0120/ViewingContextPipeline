#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID=pilot_1k_260826

# python -m validation prepare-cohort --run-id $RUN_ID
# python -m extraction prepare-input-data --run-id $RUN_ID
CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-graph-scenes-gemini --run-id $RUN_ID
# CUDA_VISIBLE_DEVICES=$GPU python -m extraction summarize-graph --run-id $RUN_ID --source qwen --gpus 2
# CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-description-scenes --run-id $RUN_ID --gpus 2
# CUDA_VISIBLE_DEVICES=$GPU python -m extraction summarize-description --run-id $RUN_ID --gpus 2
# python -m validation embed-representations --run-id $RUN_ID
# python -m validation run-recommendation --run-id $RUN_ID
# python -m validation run-diagnosis --run-id $RUN_ID