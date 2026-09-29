#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=""
export VLLM_LOGGING_LEVEL="ERROR"
export RUN_ID=${RUN_ID:-260928_v7}
GPU=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ### Preparation (Device)
# python -m preparation prepare-cohort --run-id $RUN_ID
# python -m preparation prepare-input-data --run-id $RUN_ID

## Graph, Description Extraction
## Device
# CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/scene_description_v3.md --model qwen --arm desc_qwen
# CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v7.md --model qwen --arm graph_qwen

## Cloud API
# python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/scene_description_v3.md --model gemini --arm desc_gemini
# python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v7.md --model gemini --arm graph_gemini

# ## Summary (Cloud API)
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v7.md --model gemini --arm graph_gemini
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v7.md --model gemini --arm graph_qwen
# python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_description_v5.md --model gemini --arm desc_gemini
# python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_description_v5.md --model gemini --arm desc_qwen

# ### Recommendation (Device)
# CUDA_VISIBLE_DEVICES=$GPU python -m validation embed-representations --run-id "$RUN_ID" --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta --representation-mode text
# CUDA_VISIBLE_DEVICES=$GPU python -m validation run-recommendation --run-id "$RUN_ID" --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta --workers-per-gpu 8 --representation-mode text
# CUDA_VISIBLE_DEVICES=$GPU python -m validation run-diagnosis --run-id "$RUN_ID" --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta --representation-mode text
