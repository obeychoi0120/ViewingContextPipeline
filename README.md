# ViewingContextPipeline

영상에서 추출한 시각 정보가 추천에 얼마나 유용한지 평가하는 연구용 PoC입니다. MicroLens-100K의 사용자 행동 이력을 바탕으로, **Title·Graph·Description 기반 item representation**을 SASRec에서 비교합니다.

## 파이프라인

![Viewing Context Validation PoC Pipeline](docs/design/ViewingContextPipeline_260907.png)

| Arm                  | BGE 입력                       | Scene extractor  |
| -------------------- | ------------------------------ | ---------------- |
| Metadata · Baseline | Creator-written English title  | 없음             |
| Graph Qwen           | Graph 기반 video summary       | Qwen3-VL-2B      |
| Graph Gemini         | Graph 기반 video summary       | Gemini 3.7 Flash |
| Description          | Description 기반 video summary | Qwen3-VL-2B      |

영상은 기본 30초 구간마다 최대 6장의 keyframe으로 처리하고, Graph/Description을 Qwen으로 요약합니다. 모든 Arm은 frozen BGE embedding을 사용하며, 같은 사용자·item ID sequence·split·catalog와 동일한 SASRec 구조로 각각 학습합니다. **비교 대상은 각 ID에 연결되는 item representation**이며, 별도 ID embedding을 더하지 않습니다.

기본 규모는 **사용자 1,000명**입니다. 각 사용자의 마지막 두 interaction을 validation/test로 사용하는 leave-two-out 방식이며, full-catalog ranking의 후보는 선정 사용자들의 보존된 train·valid·test item 합집합입니다.

## 실행 준비

저장소를 내려받은 뒤 루트 디렉터리에서 실행합니다. Python 3.11+, `ffmpeg`·`ffprobe`, 로컬 Qwen/BGE checkpoint가 필요합니다. Qwen·BGE·SASRec은 Ubuntu/CUDA에서, Gemini API 호출은 Vertex ADC와 project 권한을 설정한 Windows VM에서 실행합니다.

```bash
conda create -n llmjg python=3.11 -y  # 새 환경이 필요한 경우
conda activate llmjg
python -m pip install -e ".[qwen,gemini,train,dev]"
```

데이터는 [공식 MicroLens portal](https://recsys.westlake.edu.cn/)에서 준비합니다.

- 사용자 이력: `MicroLens-100k_pairs.tsv` (`user_id<TAB>space-separated item IDs`)
- 영상: item ID를 파일명으로 사용하는 `{item_id}.mp4`
- Title: `MicroLens-100k_title_en.csv`, 결측 보완용 `MicroLens-50k_titles.csv`

[config/pipeline.yaml](config/pipeline.yaml)의 다음 항목을 실행 환경에 맞게 수정합니다. 데이터와 모델은 자동 다운로드되지 않습니다.

| 설정                                    | 수정할 내용                                                   |
| --------------------------------------- | ------------------------------------------------------------- |
| `data.pairs_tsv`, `data.videos_dir` | 원본 pairs와 영상 경로                                        |
| `data.titles_csv`                     | 아래에서 생성할`MicroLens-100k_title_en_completed.csv` 경로 |
| `models.qwen`, `models.bge`         | 로컬 checkpoint 디렉터리                                      |
| `models.gemini`                       | Vertex project ID, location, model ID                         |
| `extraction.visual_evidence`          | `scene_duration`: Scene 길이(초), `num_keyframes`: 완전한 Scene의 장수, `image_resolution`: 이미지 크기 |
| `validation.cohort`                   | 사용자 수와 sampling seed. 기본값은 1,000명, seed 42          |

`protocol.sampling: fixed_windows`는 `scene_duration / num_keyframes` 길이의 구간별 중앙점을 추출합니다. 기본값 `scene_duration: 30`, `num_keyframes: 6`은 Scene 시작 기준 `[2.5, 7.5, 12.5, 17.5, 22.5, 27.5]`초입니다. 마지막 짧은 Scene은 같은 구간 간격을 유지하고 마지막 구간만 실제 영상 끝에서 잘라 중앙점을 구합니다(방식 A). 예를 들어 12초가 남으면 `[2.5, 7.5, 11]`초를 사용합니다. 영상 길이를 정수 초로 올리지 않습니다.

추출 시각은 영상 전체 기준이며 소수점 둘째 자리부터 버립니다(`1.666… → 1.6`). JSON과 추출 요청에 같은 시각을 사용하고, 이미지는 `data/fixed_{scene_duration}s/resized_keyframes/{content_id}/0002_5.png`, 정수 초는 `0005.png` 형식으로 저장합니다. Scene 정보는 `data/cohort/source_assets/{content_id}/assets/timestamp_fixed_{scene_duration}s.json`에 저장합니다. 두 설정은 양의 정수이며, 0.1초 정밀도에서 구분할 수 있도록 `num_keyframes <= scene_duration × 10`이어야 합니다. 마지막 짧은 구간에서 버림 후 같은 시각이 생기면 한 번만 추출합니다. Graph Qwen·Graph Gemini·Description은 같은 이미지 목록을 사용합니다.

## 실행 방법

아래 Bash 명령은 Ubuntu 기준입니다. Windows에서 실행하는 Gemini 단계는 별도로 표시했습니다. 두 환경에서 **같은 run ID와 해당 run의 artifact·keyframe에 접근**할 수 있도록 경로를 준비합니다.

**1. 사용자 선정과 데이터 준비**

먼저 pairs만으로 사용자를 선정하고 필요한 item을 확인합니다. 이 단계에서는 영상 처리나 모델 추론을 실행하지 않습니다.

```bash
export RUN_ID=pilot_user1k_YYYYMMDD  # 새 실험을 구분할 고유 이름
python -m validation prepare-cohort --run-id "$RUN_ID" --plan-only
```

`artifacts/$RUN_ID/data/cohort/`의 `cohort_plan.json`과 `required_items.jsonl`을 확인합니다. 아래 도구는 원본 title을 보존하고, 필요한 item의 빈 title만 공식 50K 파일에서 보완합니다. `--output`은 앞서 설정한 `data.titles_csv`와 같아야 합니다.

```bash
ANNOTATIONS=/path/to/Annotations  # 다운로드한 title 파일 위치
python -m validation.complete_titles \
  --primary "$ANNOTATIONS/MicroLens-100k_title_en.csv" \
  --supplement "$ANNOTATIONS/MicroLens-50k_titles.csv" \
  --required-items "artifacts/$RUN_ID/data/cohort/required_items.jsonl" \
  --output "$ANNOTATIONS/MicroLens-100k_title_en_completed.csv"

python -m validation prepare-cohort --run-id "$RUN_ID"
python -m extraction prepare-input-data --run-id "$RUN_ID"
```

필수 영상/title이 누락되면 준비가 중단됩니다. 자산을 보완한 뒤 같은 명령으로 재개합니다. 필요한 영상·scene 규모를 확인하고 **GPU/Vertex 실행 범위와 예산을 승인한 뒤** 다음 단계로 진행합니다.

**2. Graph·Description 추출과 요약**

Ubuntu/CUDA에서 Qwen branch를 실행합니다. `--gpus 1`은 사용할 visible GPU 개수입니다.

```bash
python -m extraction extract-graph-scenes --model qwen --run-id "$RUN_ID" --gpus 1
python -m extraction summarize-graph --source qwen --run-id "$RUN_ID" --gpus 1
python -m extraction extract-description-scenes --run-id "$RUN_ID" --gpus 1
python -m extraction summarize-description --run-id "$RUN_ID" --gpus 1
```

Gemini Graph 추출은 Windows/PowerShell에서 같은 run ID로 실행합니다.

```powershell
conda activate llmjg
$RUN_ID = "pilot_user1k_YYMMDD"  # Ubuntu에서 사용한 값과 동일하게 지정
python -m extraction extract-graph-scenes --model gemini --run-id $RUN_ID
```

Gemini Graph 산출물을 Ubuntu의 같은 run에서 사용할 수 있는지 확인한 뒤, Qwen으로 요약합니다.

```bash
python -m extraction summarize-graph --source gemini --run-id "$RUN_ID" --gpus 1
```

**3. Embedding·추천·평가**

세 branch의 summary가 준비되면 Ubuntu/CUDA에서 실행합니다. 추천은 4개 Arm × 3개 seed로 학습하며, validation으로 학습 epoch를 선택한 뒤 train+valid로 refit하여 test를 평가합니다.

```bash
python -m validation embed-representations --run-id "$RUN_ID"
python -m validation run-recommendation --run-id "$RUN_ID"
python -m validation run-diagnosis --run-id "$RUN_ID"
```

`embed-representations`는 Gemini Graph summary 파일이 없는 항목에 같은 항목의 Qwen Graph summary를 사용합니다. BGE 로딩 전에 대체 항목의 `item_id`, `content_id`, summary 경로를 콘솔에 출력하고, 사용한 목록은 `representations/graph_gemini_fallbacks.json`에 저장합니다. 원본 summary 파일은 변경하지 않습니다. Gemini summary가 존재하지만 잘못됐거나 대체할 Qwen summary도 없으면 오류로 처리합니다. 대체 목록이 바뀌면 Gemini 임베딩을 재생성합니다. 이때 Gemini branch는 Qwen 대체가 포함된 결과이므로 모델별 비교 시 대체 목록을 함께 확인해야 합니다. 이미 추천을 실행했다면 `run-recommendation --force` 후 diagnosis를 다시 실행합니다.

## 결과 확인 및 유의사항

결과는 `artifacts/{run_id}/`에 저장됩니다.

| 파일                                                  | 확인할 내용                               |
| ----------------------------------------------------- | ----------------------------------------- |
| `validation/diagnosis/diagnosis.json`               | 실행 완전성, Arm별 지표와 비교, 통계 경고 |
| `validation/recommendations/per_user_metrics.jsonl` | 사용자·Arm·seed별 ranking 지표          |
| `validation/recommendations/training_runs.jsonl`    | 학습 이력과 epoch 선택·refit 기록        |
| `validation/recommendations/checkpoints/`           | Arm·seed별 학습 모델                     |

먼저 `diagnosis.json`의 `runtime_decision.status`가 `pass`인지 확인하고, `statistical_analysis`의 상태·경고를 읽습니다. NDCG@K는 정답의 순위, HR@K는 상위 K개 내 정답 포함 여부를 평가합니다. Top-20 catalog coverage와 top-1 concentration은 추천의 분산·쏠림을 보여줍니다. `computed_with_warnings`는 경고 내용을 확인한 뒤 비교별 해석 가능 여부를 판단해야 합니다.

- **중단 후 재개:** 같은 조건이면 같은 명령을 재실행합니다. 완료된 artifact는 재사용됩니다.
- **Gemini 실패 장면 재시도:** 현재 추출 프로세스가 끝난 뒤 같은 명령을 다시 실행합니다. 별도 옵션 없이 성공 장면은 재사용하고, 실패 장면과 아직 결과가 없는 장면만 한 번씩 호출합니다. 재시도 성공 시 해당 실패 기록을 제거하며, 실패가 남으면 진단 정보를 갱신합니다. 실행 중 같은 실패를 반복 재시도하지는 않습니다. 모델·prompt·입력·생성 설정은 기존 run과 같아야 합니다. 기존 요약이 있다면 요약 이후 단계도 재생성해야 합니다.
- **Gemini 빈 응답 진단:** 콘솔과 `extraction/graph/gemini/scenes/failures/{content_id}.jsonl`의 `response_diagnostics`에 `candidates[].finish_reason`, `finish_message`, `prompt_feedback`, `usage_metadata`를 기록합니다. 과거 실패에는 이 정보가 없으며, 다음 빈 응답부터 기록됩니다. `MAX_TOKENS`는 출력 한도 도달, `SAFETY`나 `prompt_feedback.block_reason`은 차단 원인을 확인하는 단서입니다. 단순히 텍스트가 비었다는 이유만으로 차단으로 분류하지 않습니다.
- **강제 재생성:** `--force`는 해당 단계에만 적용됩니다. 추출을 다시 했다면 요약 → embedding → 추천도 각각 재생성하고 diagnosis를 다시 실행합니다.
- **조건 변경:** 데이터·사용자 수·seed·모델·prompt·protocol·`scene_duration`·`num_keyframes`를 바꾸면 새 run ID를 사용합니다. 이미지 재사용은 현재 sampling 시각·Scene 구간·크기를 검증하지만, 기존 추출·요약·embedding·추천 cache는 입력 변경을 자동 검증하지 않습니다. 이전 3장 조건의 결과와 새 6장 조건의 결과를 같은 run에 섞지 않습니다.
- **검증 범위:** 테스트 통과와 실제 MicroLens 전체 실험 완료는 구분합니다. 모델의 일반적 우열이나 온라인 추천 효과는 이 PoC 결과만으로 주장할 수 없습니다.

> **[직접 결정 필요]** Pilot 결과 해석 전 scene coverage 기준(0.95), Arm 간 coverage gap(0.05), partial summary 허용 여부, primary NDCG@10, non-inferiority margin(0.05), comparison family·다중비교 정책을 승인해야 합니다. 현재 값은 provisional 설정입니다.

상세 설정은 [pipeline config](config/pipeline.yaml), [추출·요약 prompt](config/prompts/), [평가·비교 구현](src/validation/diagnosis_statistics.py)을 참고하세요.
