#!/usr/bin/env bash

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=""
export VLLM_LOGGING_LEVEL="ERROR"
export RUN_ID="261006_v9"
GPU=0,1,2,3

# ### Preparation (Device)
python -m preparation prepare-cohort --run-id $RUN_ID
python -m preparation prepare-input-data --run-id $RUN_ID

## Graph, Description Extraction
## Device
CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v8.md --model qwen --arm graph_qwen

## Cloud API
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v8.md --model gemini --arm graph_gemini

# ## Summary (Cloud API)
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v8.md --model gemini --arm graph_qwen
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v8.md --model gemini --arm graph_gemini

# ### Recommendation (Device)
## Text
ARMS=(meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta)
CUDA_VISIBLE_DEVICES=$GPU python -m validation embed-representations --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode text
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-recommendation --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode text --workers-per-gpu 2
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-diagnosis --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode text

## Graph
ARMS=(graph_qwen graph_qwen_meta graph_gemini_meta)
CUDA_VISIBLE_DEVICES=$GPU python -m validation embed-representations --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-recommendation --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph --scene-aggregation mean --workers-per-gpu 2
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-diagnosis --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph --scene-aggregation mean

ARMS=(graph_qwen graph_qwen_meta graph_gemini_meta)
CUDA_VISIBLE_DEVICES=$GPU python -m validation embed-representations --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-recommendation --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph --scene-aggregation gnn --workers-per-gpu 2
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-diagnosis --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph --scene-aggregation gnn