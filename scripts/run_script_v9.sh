#!/usr/bin/env bash

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=""
export VLLM_LOGGING_LEVEL="ERROR"
export RUN_ID="261006_v9"
GPU=0,1,2,3

# Extraction·Summary는 완료된 결과를 사용합니다.
# runs/$RUN_ID/extraction -> 저장소 artifacts/extraction (symlink)
# 재실행하면 Qwen 실패 장면 재추출과 Gemini 호출이 시도되므로 주석 처리합니다.

# ### Preparation
# leave-one-out cohort(preparation/cohort/leave_one_out)를 만듭니다. 완료 후에는 재사용됩니다.
python -m preparation prepare-cohort --run-id $RUN_ID
# 키프레임은 기존 결과를 사용하므로 생략합니다.
# python -m preparation prepare-input-data --run-id $RUN_ID

# ## Graph, Description Extraction
# ## Device
# CUDA_VISIBLE_DEVICES=$GPU python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v8.md --model qwen --arm graph_qwen

# ## Cloud API
# python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v8.md --model gemini --arm graph_gemini

# ## Summary (Cloud API)
# python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v8.md --model gemini --arm graph_qwen
# python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v8.md --model gemini --arm graph_gemini

# ### Recommendation (Device): 사용자별 leave-one-out
## embed-representations가 Run별 분할·평가 catalog를 준비하고 reports/leave_one_out_cohort.{json,md}를 기록합니다.
## Text: 제목 + Graph Summary / Description(caption) Summary
ARMS=(meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta)
CUDA_VISIBLE_DEVICES=$GPU python -m validation embed-representations --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode text
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-recommendation --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode text --workers-per-gpu 2
CUDA_VISIBLE_DEVICES=$GPU python -m validation run-diagnosis --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode text

## Graph: Scene Graph 직접 학습
## mean, gnn: 기준선
## Scene attention: attn_pool (advisor table B), gnn_attn (advisor table D)
## Scene Transformer: C0 (transformer_local), C (transformer), D (transformer_pos)
ARMS=(graph_qwen graph_qwen_meta graph_gemini_meta)
CUDA_VISIBLE_DEVICES=$GPU python -m validation embed-representations --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph
for AGG in mean gnn attn_pool gnn_attn transformer_local transformer transformer_pos; do
  CUDA_VISIBLE_DEVICES=$GPU python -m validation run-recommendation --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph --scene-aggregation "$AGG" --workers-per-gpu 2
  CUDA_VISIBLE_DEVICES=$GPU python -m validation run-diagnosis --run-id "$RUN_ID" --target "${ARMS[@]}" --representation-mode graph --scene-aggregation "$AGG"
done
