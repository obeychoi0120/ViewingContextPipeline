# ViewingContextPipeline

MicroLens-100K interaction cohort의 local MP4에서 Viewing Context를 추출하고, 같은 cohort·split·seed의 독립 SASRec arm으로 검증하는 파이프라인입니다. 모든 공개 contract는 현재 기준 `v1`입니다.

이 파이프라인은 지정된 next-item ranking protocol의 차이를 측정합니다. 결과를 CTR, 시청시간, 만족도 또는 인과효과로 해석하지 않습니다.

## 구현 구성

메인 구현은 `src/viewing_context_pipeline/` 아래의 두 그룹으로 나뉩니다.

```text
src/viewing_context_pipeline/
├─ extraction/    # MicroLens MP4 준비, Qwen/Gemini Context 추출
└─ validation/    # BGE representation, SASRec 추천, diagnosis
```

공개 stage의 DAG, resume 및 부분 재실행은 `pipeline.py`가 관리하고, 각 stage의 실제 실행은 `stages.py`가 위 두 구현 그룹을 호출합니다. 별도 `scripts/` wrapper 계층은 없습니다.

## 전체 흐름

```text
(Extraction: src/viewing_context_pipeline/extraction)

prepare_data
├─ extract_ondevice_graph_context
├─ extract_ondevice_desc_context
├─ extract_gemini_graph_context
└─ extract_gemini_desc_context          # optional, default off

(Validation: src/viewing_context_pipeline/validation)

embed_representations
        ↓
run_recommendation
        ↓
run_diagnosis
```

`modality`는 run 전체에서 `visual_only | multimodal` 중 하나만 사용하며 기본값은 `visual_only`입니다. 활성 Extraction branch가 서로 다른 modality를 섞어 쓰는 것은 허용하지 않습니다. sampling은 `fixed_30s`, on-device VLM은 Qwen, embedding은 BGE, baseline은 `SASRec_ID`로 고정됩니다.

## Extraction 단계

Extraction 구현은 `src/viewing_context_pipeline/extraction/`에 있습니다.

### 1. `prepare_data`

MicroLens cohort와 local MP4를 다음 Extraction branch가 사용할 canonical evidence로 준비합니다.

- MicroLens interaction pairs와 실제 MP4를 검사하여 cohort, sequence split 및 catalog를 확정합니다.
- cohort catalog에 포함된 MP4만 처리합니다.
- 각 영상을 `fixed_30s` scene과 10초 reference 구간으로 나누고 resized keyframe을 생성합니다.
- `visual_manifest.jsonl`과 evidence fingerprint를 생성합니다.
- `multimodal` run에서만 faster-whisper ASR과 PaddleOCR을 실행하고 `multimodal_ref`를 생성합니다.
- `visual_only` run은 `multimodal_ref`를 만들거나 읽지 않습니다.

주요 출력:

```text
data/cohort/catalog.jsonl
data/cohort/sequences.jsonl
data/cohort/extraction_manifest.csv
data/fixed_30s/resized_keyframes/{content_id}/
data/fixed_30s/visual_manifest.jsonl
data/fixed_30s/multimodal_ref/{content_id}_multimodal_ref.jsonl  # multimodal only
```

`multimodal_ref`의 각 timeline record는 이미지와 같은 timestamp를 가지며 `raw_asr`, `raw_ocr`를 string으로 보존합니다. Extraction 전에 image/reference의 개수와 timestamp 정렬을 1:1로 검사합니다.

### 2-A. `extract_ondevice_graph_context`

Qwen으로 각 scene의 visual graph를 추출하고 성공한 scene graph를 영상 단위 `VC_graph`로 집계합니다.

- `visual_only`: resized keyframe만 Qwen payload에 포함합니다.
- `multimodal`: 각 이미지 직후에 동일 timestamp의 `shot_reference(asr_text, ocr_text)`를 추가합니다.
- scene extraction, graph aggregation 및 embedding용 deterministic text serialization을 한 stage에서 완료합니다.
- 불완전한 scene이나 fingerprint가 맞지 않는 결과는 complete Context로 전달하지 않습니다.

### 2-B. `extract_ondevice_desc_context`

같은 keyframe을 Qwen으로 서술하고 scene description을 영상 단위 `VC_desc`로 요약합니다.

- Graph branch와 동일한 visual 또는 multimodal evidence loader를 사용합니다.
- visual-only prompt에는 ASR, OCR, 제목, 장르 또는 기타 metadata를 넣지 않습니다.
- multimodal에서는 image/reference 정렬을 다시 검증합니다.

### 2-C. `extract_gemini_graph_context`

Gemini로 scene graph를 추출하고 영상 단위 `VC_graph`를 생성합니다.

- evidence와 modality 계약은 on-device Graph branch와 같습니다.
- cloud backend의 내부 표기와 output은 `gemini`로 통일됩니다.
- project, location, model 및 thinking level은 root `config/local.yaml`에서 전달됩니다.

### 2-D. `extract_gemini_desc_context`

Gemini로 descriptive scene context와 영상 단위 `VC_desc`를 생성하는 optional branch입니다. 기본 pipeline config에서는 비활성화되어 있으며, 활성화해도 다른 branch와 동일한 `video-context/v1` handoff를 생성합니다.

네 Extraction branch의 root 출력은 모두 다음 공통 필드를 가집니다.

```text
schema_version: video-context/v1
content_id
context_type: graph | description
backend: ondevice | gemini
branch
modality: visual_only | multimodal
status: complete
text
evidence_fingerprint
model_fingerprint
source
```

## Validation 단계

Validation 구현은 `src/viewing_context_pipeline/validation/`에 있습니다. 동일한 MicroLens cohort와 leave-two-out split에서 ID baseline 및 활성 Context arm을 독립적으로 학습·비교합니다.

### 3. `embed_representations`

활성 Extraction branch를 동적으로 탐색하여 local BGE embedding을 생성합니다.

- catalog의 모든 `content_id`에 complete Context가 있는지 검사합니다.
- branch 간 modality 혼합을 거부합니다.
- 같은 콘텐츠의 evidence fingerprint가 branch마다 일치하는지 검사합니다.
- 각 Context의 `text`를 BGE 1024D L2-normalized vector로 변환합니다.
- encoder model file manifest와 source fingerprint를 함께 기록합니다.

### 4. `run_recommendation`

`SASRec_ID` baseline과 활성 Context arm을 각각 독립 학습합니다.

- arm별로 별도 SASRec checkpoint를 생성합니다.
- 동일한 cohort, split, training protocol 및 seeds를 사용합니다.
- 실제 `item_id`와 `content_id`를 포함한 Top-K recommendation을 저장합니다.
- Validation 입력은 ordered user-item interaction sequence이며 likes, dwell time 또는 completion event를 가정하지 않습니다.

### 5. `run_diagnosis`

recommendation artifact만 읽어 평가와 readiness를 생성합니다. 모델을 다시 학습하지 않으므로 recommendation 완료 후 diagnosis만 독립 재실행할 수 있습니다.

- HR/NDCG
- catalog coverage
- Top-1 concentration
- item frequency bucket별 결과
- paired bootstrap
- `report_ready`

## 설정

로컬 경로와 credential은 root `config/local.yaml`에서만 관리합니다.

```powershell
conda activate llmjg
python -m pip install -e .
Copy-Item config/local.example.yaml config/local.yaml
```

`config/local.yaml`에 다음 값을 지정합니다.

- MicroLens videos, titles, tags, interaction pairs 경로
- Qwen, BGE, faster-whisper local model 경로
- Gemini project, location, model, thinking level

`data.pairs_tsv`에는 item catalog가 1,000개로 제한된 `MicroLens-100k_pairs_1k.tsv`를 지정합니다. pilot cohort는 이 파일과 실제 MP4의 교집합에서 `min_sequence_length >= 5`를 만족하는 적격 사용자 전원을 사용하며, 현재 dataset snapshot에서는 59명입니다. 요청한 `cohort.user_count`보다 적격 사용자가 적으면 `prepare_data`가 실패하고 `data/cohort/eligibility_summary.json`에 원인을 기록합니다.

pipeline protocol과 활성 branch는 `config/pipelines/`에, component의 모델 처리 설정은 `config/extraction/`, SASRec protocol은 `config/validation/`에 있습니다. run 도중 원본 설정은 수정하지 않으며 `artifacts/{run_id}/runtime/components/`에 실행용 복사본을 만듭니다.

## 실행

전체 pipeline 실행:

```powershell
python -m viewing_context_pipeline run `
  --config config/pipelines/microlens_graph_vs_desc_pilot.yaml `
  --local-config config/local.yaml `
  --stage all
```

Linux checkout에서는 root `run.sh`을 사용할 수 있습니다. script가 `<repo>/src`를 `PYTHONPATH`에 추가하므로 editable install 없이도 stage subprocess가 같은 package를 import합니다.

```bash
# 기본값: prepare_data
bash run.sh

# 전체 pipeline
STAGE=run bash run.sh --stage all

# 기존 run의 독립 downstream stage
STAGE=extract_ondevice_graph_context RUN_ID=260824_1313 bash run.sh
```

`PIPELINE`, `LOCAL_CONFIG`, `STAGE`, `RUN_ID`는 환경변수로 덮어쓸 수 있습니다. `STAGE=run`에서는 추가 인자 `--stage all`을 전달합니다.

새 run의 ID는 Asia/Seoul 기준 `YYMMDD_HHmm`으로 한 번 생성됩니다. 명시적인 `--run-id`가 우선하며 같은 ID의 non-empty artifact directory는 덮어쓰지 않습니다. resume과 독립 downstream 실행에는 `--run-id`가 필요합니다.

```powershell
$PIPELINE = "config/pipelines/microlens_graph_vs_desc_pilot.yaml"
$RUN_ID = "260824_0938"

python -m viewing_context_pipeline run --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID --resume
python -m viewing_context_pipeline run --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID --stage all --force-stage extract_ondevice_graph_context
```

`--force-stage`는 지정 branch와 실제 downstream만 무효화하며 독립 Extraction branch는 보존합니다. `--dry-run`은 artifact를 만들지 않고 preflight와 실행 명령을 표시합니다.

단계별 독립 실행:

```powershell
python -m viewing_context_pipeline prepare_data --config $PIPELINE --local-config config/local.yaml
python -m viewing_context_pipeline extract_ondevice_graph_context --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
python -m viewing_context_pipeline extract_ondevice_desc_context --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
python -m viewing_context_pipeline extract_gemini_graph_context --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
python -m viewing_context_pipeline extract_gemini_desc_context --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
python -m viewing_context_pipeline embed_representations --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
python -m viewing_context_pipeline run_recommendation --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
python -m viewing_context_pipeline run_diagnosis --config $PIPELINE --local-config config/local.yaml --run-id $RUN_ID
```

각 단계는 선행 단계를 자동 실행하지 않습니다.

## Artifact 구조와 계약

```text
artifacts/{run_id}/
├─ data/
│  ├─ cohort/
│  └─ fixed_30s/
│     ├─ resized_keyframes/{content_id}/
│     ├─ visual_manifest.jsonl
│     └─ multimodal_ref/{content_id}_multimodal_ref.jsonl
├─ extraction/contexts/{modality}/{branch}/
└─ validation/
   ├─ representations/
   ├─ recommendations/
   └─ diagnosis/
```

Canonical contract:

- `pipeline/v1`
- `prepared-data/v1`
- `visual-manifest/v1`
- `multimodal-reference/v1`
- `video-context/v1`
- `representations/v1`
- `recommendations/v1`
- `diagnosis/v1`
- `validation-config/v1`

새 run은 `artifacts/{run_id}` 밖의 legacy output을 읽거나 수정하지 않으며 compatibility fallback을 제공하지 않습니다.

## 검증

```powershell
conda activate llmjg
ruff check src tests
python -m pytest -q tests
python -m compileall -q src tests
git diff --check
```

synthetic E2E와 contract test 통과는 구현 검증입니다. 실제 MicroLens·GPU·Gemini smoke가 실행되기 전에는 실제 pipeline 실행 검증으로 보고하지 않습니다.
