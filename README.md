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

영상은 30초 구간마다 최대 3장의 keyframe으로 처리하고, Graph/Description을 Qwen으로 요약합니다. 모든 Arm은 frozen BGE embedding을 사용하며, 같은 사용자·item ID sequence·split·catalog와 동일한 SASRec 구조로 각각 학습합니다. **비교 대상은 각 ID에 연결되는 item representation**이며, 별도 ID embedding을 더하지 않습니다.

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
| `validation.cohort`                   | 사용자 수와 sampling seed. 기본값은 1,000명, seed 42          |

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
- **강제 재생성:** `--force`는 해당 단계에만 적용됩니다. 추출을 다시 했다면 요약 → embedding → 추천도 각각 재생성하고 diagnosis를 다시 실행합니다.
- **조건 변경:** 데이터·사용자 수·seed·모델·prompt·protocol을 바꾸면 새 run ID를 사용합니다. 기존 cache는 입력 변경을 자동 검증하지 않습니다.
- **검증 범위:** 테스트 통과와 실제 MicroLens 전체 실험 완료는 구분합니다. 모델의 일반적 우열이나 온라인 추천 효과는 이 PoC 결과만으로 주장할 수 없습니다.

> **[직접 결정 필요]** Pilot 결과 해석 전 scene coverage 기준(0.95), Arm 간 coverage gap(0.05), partial summary 허용 여부, primary NDCG@10, non-inferiority margin(0.05), comparison family·다중비교 정책을 승인해야 합니다. 현재 값은 provisional 설정입니다.

상세 설정은 [pipeline config](config/pipeline.yaml), [추출·요약 prompt](config/prompts/), [평가·비교 구현](src/validation/diagnosis_statistics.py)을 참고하세요.
