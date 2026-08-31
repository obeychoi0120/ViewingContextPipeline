# ViewingContextPipeline

## Pipeline Overview

MicroLens-100K local MP4에서 동일한 visual evidence를 사용해 Graph와 Description 표현을 만들고, 같은 cohort·split·seed의 독립 SASRec arm으로 비교하는 파이프라인입니다. 전체 구조는 [Pipeline Overview PPTX](ViewingContextPipeline_Overview.pptx)에 정리되어 있습니다.

```text
MicroLens-100K
  + visual_only
  + fixed_30s (5s, 15s, 25s keyframes)
  + Graph Extractor: Qwen3-VL-2B | Gemini 3.7 Flash
  + Graph/Description Summarizer: Qwen3-VL-2B
  + Recommendation Arms:
      SASRec_ID
      SASRec_GRAPH_QWEN
      SASRec_GRAPH_GEMINI
      SASRec_DESC
```

이 파이프라인은 지정된 next-item ranking protocol에서 representation arm 간 차이를 측정합니다. 결과를 CTR, 시청시간, 만족도 또는 인과효과로 해석하지 않습니다.

구성은 다음 세 부분으로 나뉩니다.

- `src/extraction/`: fixed-30s keyframe, Graph/Description scene, 영상 단위 summary
- `src/validation/`: cohort, BGE embedding, 4-arm SASRec, runtime diagnosis
- `src/pipeline_runtime.py`: 공통 config 검증, run 경로, JSON I/O

`extraction`과 `validation`은 각각 독립 CLI를 제공합니다. 전체 실행 스크립트에서는 아래 11개 명령을 같은 `run_id`로 순서대로 호출해야 합니다.

## 환경 설정

```bash
conda activate llmjg
python -m pip install -e ".[qwen,gemini,train]"
gcloud auth application-default login
```

`config/pipeline.yaml`에서 다음 값만 실행 환경에 맞게 설정합니다.

- `data.videos_dir`, `data.pairs_tsv`
- `models.qwen`, `models.bge`
- `models.gemini.project_id`, `location`, `model_id`, `temperature`, `max_output_tokens`, `thinking_level`

Gemini는 Vertex AI Application Default Credentials를 사용합니다. editable install을 하지 않은 checkout에서는 저장소 루트에서 `PYTHONPATH`에 `src`를 추가합니다.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 실험 계약

Qwen Graph, Gemini Graph, Description은 같은 fixed-30s 장면과 시간순 keyframe을 입력으로 사용합니다. 두 Graph source의 scene graph는 각각 보존하고, 영상 단위 summary는 모두 Qwen으로 생성합니다. Description도 직접 보이는 사실만 기술하는 strict visible-only prompt를 사용하며, Graph와 마찬가지로 story, identity, demographics, 직업, 관계, 목적, audience, private trait, 실제 mental state 또는 화면 text를 추론하지 않습니다.

Graph JSON은 native object를 우선 사용하고 실패 시 deterministic repair를 한 번 적용합니다. 복구할 수 없는 Graph scene, Gemini API 최종 실패, 빈 Description scene만 해당 scene의 failure artifact로 남기고 나머지는 계속 처리합니다. 실제 success/failure scene 수와 arm별 coverage는 `run-diagnosis`가 다시 집계합니다.

### 영상 Summary 계약

Qwen3-VL-2B는 Graph와 Description의 영상 단위 summary를 자유 문단으로 직접 생성하지 않습니다. 두 branch 모두 다음과 같은 동일한 5개 필드의 JSON object를 생성합니다.

```json
{
  "setting_and_environments": "...",
  "main_characters_and_objects": "...",
  "chronological_events": "...",
  "relations": "...",
  "affect_or_topic": "..."
}
```

필드명과 분류는 Qwen 출력 계약에 포함됩니다. 후처리는 자유 문장을 필드로 재분류하지 않습니다. 먼저 native JSON object를 읽고 실패하면 `json_repair.py`의 deterministic syntax-only repair를 한 번 적용한 뒤, exact field set·문자열 타입·비어 있지 않은 전체 evidence를 검증합니다. 근거가 없는 개별 필드는 빈 문자열로 출력합니다. 유효한 필드는 위 순서대로 `Setting and environments: ...` 형태로 결정적으로 직렬화하고 각 section 끝에 줄바꿈을 넣습니다. summary artifact에는 원래 `sections`와 BGE 입력용 `text`를 함께 저장합니다. BGE에는 raw JSON이나 Markdown이 아니라 줄 단위 section으로 구성된 `text`만 입력합니다.

Graph와 Description은 같은 필드·repair·검증·직렬화기를 사용하므로 출력 형식 차이가 representation arm 비교의 교란변수가 되지 않도록 합니다. 최종 summary JSON에는 `validation_warnings`를 저장하지 않습니다. 이 구조화 summary 계약을 변경하는 실험은 새 `run_id`를 사용해야 합니다.

최초 Summary 생성은 기존과 같이 greedy decoding(`do_sample=false`)을 사용합니다. JSON repair 또는 5필드 계약 검증이 실패한 content만 같은 Qwen model worker에서 다시 생성하며 모델을 재로딩하지 않습니다. retry는 순서대로 seed `42`, `43`, `44`를 사용하고 각 시도에 `do_sample=true`, `temperature=0.1`, `top_p=0.8`, `top_k=20`을 적용합니다. 세 번 모두 실패하면 malformed summary를 저장하지 않고 해당 summary Step을 실패 처리합니다. 이 retry 정책은 영상 Summary에만 적용하며 Graph scene의 1회 생성·실패 artifact 계약은 변경하지 않습니다.

4-arm 평가는 `validation.evaluation`의 family-wise alpha와 Bonferroni 정책을 사용합니다. confirmatory family는 세 Viewing Context arm과 `SASRec_ID`의 superiority 비교 3개, 두 Graph arm과 `SASRec_DESC`의 non-inferiority 비교 2개입니다. Qwen–Gemini 직접 비교는 exploratory입니다.

`min_scene_coverage`와 `max_arm_coverage_gap`은 runtime 통과 기준입니다. `[직접 결정 필요]` 현재 `0.95`와 `0.05`는 provisional PoC gate이므로 실제 pilot 결과를 해석하기 전에 확정해야 합니다.

## 독립 Step 실행

모든 명령은 `--run-id`를 요구하며 선행 Step을 자동 실행하지 않습니다. `run-diagnosis`를 제외한 Step은 필요한 실제 출력이 이미 있으면 재사용하고, `--force`는 선택한 Step의 출력만 다시 생성합니다.

아래 경로는 모두 `artifacts/{run_id}/` 기준입니다. `failures/*.jsonl`과 `data/cohort/{failures,preparation_failures}.jsonl`은 실패가 있을 때만 생성됩니다.

| Step | Ingest artifact | Output artifact |
| --- | --- | --- |
| `prepare-cohort` | `data.pairs_tsv`, `data.videos_dir`의 MP4 | `data/cohort/{item_inventory,catalog,sequences}.jsonl`, `eligibility_summary.json`, 선택적 `failures.jsonl` |
| `prepare-input-data` | `data/cohort/catalog.jsonl`, catalog의 `source_video_path`, `duration_seconds` | `data/cohort/source_assets/{content_id}/assets/timestamp_fixed_30s.json`, `data/fixed_30s/resized_keyframes/{content_id}/*.png`, 선택적 `preparation_failures.jsonl` |
| `extract-graph-scenes --model qwen` | cohort catalog, fixed-30s timestamp/keyframe, `models.qwen` | `extraction/graph/qwen/scenes/{content_id}.jsonl`, 선택적 `failures/{content_id}.jsonl` |
| `summarize-graph --source qwen` | Qwen Graph scene JSONL, Graph summary prompt, `models.qwen` | `extraction/graph/qwen/summaries/{content_id}.json` |
| `extract-graph-scenes --model gemini` | cohort catalog, fixed-30s timestamp/keyframe, Vertex ADC와 `models.gemini` | `extraction/graph/gemini/scenes/{content_id}.jsonl`, 선택적 `failures/{content_id}.jsonl` |
| `summarize-graph --source gemini` | Gemini Graph scene JSONL, Graph summary prompt, `models.qwen` | `extraction/graph/gemini/summaries/{content_id}.json` |
| `extract-description-scenes` | cohort catalog, fixed-30s timestamp/keyframe, Description scene prompt, `models.qwen` | `extraction/description/scenes/{content_id}.jsonl`, 선택적 `failures/{content_id}.jsonl` |
| `summarize-description` | Description scene JSONL, Description summary prompt, `models.qwen` | `extraction/description/summaries/{content_id}.json` |
| `embed-representations` | cohort catalog, Qwen/Gemini Graph summary, Description summary, `models.bge` | `validation/representations/{graph_qwen_embeddings,graph_gemini_embeddings,desc_embeddings}.npz`, `item_index.json` |
| `run-recommendation` | cohort `catalog.jsonl`, `sequences.jsonl`, 세 branch embedding과 `item_index.json` | `validation/recommendations/per_user_metrics.jsonl`, `training_runs.jsonl`, `checkpoints/` |
| `run-diagnosis` | cohort, scene success/failure, representation, metric, training run, checkpoint의 실제 파일 | `validation/diagnosis/diagnosis.json` |

```bash
python -m validation prepare-cohort --run-id 1k_pilot_260827
python -m extraction prepare-input-data --run-id 1k_pilot_260827
python -m extraction extract-graph-scenes --run-id 1k_pilot_260827 --model qwen
python -m extraction summarize-graph --run-id 1k_pilot_260827 --source qwen
python -m extraction extract-graph-scenes --run-id 1k_pilot_260827 --model gemini
python -m extraction summarize-graph --run-id 1k_pilot_260827 --source gemini
python -m extraction extract-description-scenes --run-id 1k_pilot_260827
python -m extraction summarize-description --run-id 1k_pilot_260827
python -m validation embed-representations --run-id 1k_pilot_260827
python -m validation run-recommendation --run-id 1k_pilot_260827
python -m validation run-diagnosis --run-id 1k_pilot_260827
```

Qwen을 사용하는 Step은 `--gpus N`을 지원합니다. 멀티-GPU 실행의 Qwen worker는 부모 process가 수명주기를 관리합니다. Linux local에서 `Ctrl+C`가 들어오면 대기 중인 작업을 버리고 모든 GPU worker에 즉시 종료를 요청하며, 짧은 유예 뒤에도 남은 worker는 강제 종료해 CUDA memory를 해제합니다. CLI는 이 경우 exit code `130`을 반환합니다. Gemini scene extraction은 `--gpus` 대신 `extraction.graph.gemini_concurrency`만 사용합니다. `embed-representations`는 유효한 branch embedding을 재사용하고, 인코딩이 필요한 branch들은 한 번 로드한 BGE runtime을 공유합니다.

## Runtime 판정과 저장 정책

- config, prompt, model, summary schema 또는 protocol이 달라지는 새 실험은 새 `run_id`를 사용합니다.
- manifest와 fingerprint는 생성하지 않으며 고정 경로의 실제 파일을 직접 읽습니다.
- content별 title/tag metadata와 빈 failure JSONL은 생성하지 않습니다.
- Description scene에는 `schema_version`, `content_id`, `scene_idx`, `keyframes`, `description`만 저장합니다.
- `training_runs.jsonl`에는 seed×arm별 epoch history, best validation, checkpoint와 실행 정보를 저장합니다.
- `diagnosis.json`은 `report_ready`를 사용하지 않습니다. `run-diagnosis`가 매번 실제 artifact를 집계해 `runtime_decision.status`, `checks`, `errors`를 기록하고 기준 미달이면 파일을 기록한 뒤 실패합니다.
- runtime artifact 통과와 통계적 superiority/non-inferiority 결론은 별개입니다.

주요 계약은 `viewing-context-config/v1`, `scene-description/v1`, `graph-video-summary/v2`, `description-video-summary/v2`, `diagnosis/v2`입니다.

## 검증

```bash
conda activate llmjg
ruff check src tests
python -m pytest -q tests
python -m compileall -q src tests
git diff --check
```

synthetic test와 contract test 통과는 구현 검증입니다. 실제 MicroLens·GPU pilot이 실행되기 전에는 추천 품질 결과로 보고하지 않습니다.
