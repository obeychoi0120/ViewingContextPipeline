#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

ANNOTATIONS=/home_nvme/shared/data/microlens_100k/Annotations
RUN_ID=pilot_v2_user1k_260904
GPU=0,1,2,3

python -m validation prepare-cohort --run-id $RUN_ID --plan-only

python -m validation.complete_titles \
  --primary "$ANNOTATIONS/MicroLens-100k_title_en.csv" \
  --supplement "$ANNOTATIONS/MicroLens-50k_titles.csv" \
  --required-items "artifacts/$RUN_ID/data/cohort/required_items.jsonl" \
  --output "$ANNOTATIONS/MicroLens-100k_title_en_completed.csv"

python -m validation prepare-cohort --run-id $RUN_ID
python -m extraction prepare-input-data --run-id $RUN_ID

CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-graph-scenes --model qwen --run-id $RUN_ID --gpus 4
CUDA_VISIBLE_DEVICES=$GPU python -m extraction summarize-graph --run-id $RUN_ID --source qwen --gpus 4

# python -m extraction extract-graph-scenes --model gemini --run-id pilot_v2_user1k_260904
CUDA_VISIBLE_DEVICES=$GPU python -m extraction summarize-graph --run-id $RUN_ID --source gemini --gpus 4

CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-description-scenes --run-id $RUN_ID --gpus 4
CUDA_VISIBLE_DEVICES=$GPU python -m extraction summarize-description --run-id $RUN_ID --gpus 4

python -m validation embed-representations --run-id $RUN_ID
python -m validation run-recommendation --run-id $RUN_ID
python -m validation run-diagnosis --run-id $RUN_ID