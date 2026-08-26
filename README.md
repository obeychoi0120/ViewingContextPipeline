# ViewingContextPipeline

## Pipeline Overview

![ViewingContextPipeline overview](src/ViewingContextPipeline_260825.png)

MicroLens-100K local MP4에서 visual-only Viewing Context를 추출하고, 같은 cohort·split·seed의 독립 SASRec arm으로 비교하는 파이프라인입니다.

고정 pilot protocol은 다음과 같습니다.

```text
MicroLens-100K
  + visual_only
  + fixed_30s (5s, 15s, 25s keyframes)
  + Graph Extractor: Qwen3-VL-2B vs Vertex Gemini
  + Graph/Description Summarizer: Qwen3-VL-2B
  + SASRec_ID vs SASRec_GRAPH_QWEN vs SASRec_GRAPH_GEMINI vs SASRec_DESC
```

이 코드는 지정된 next-item ranking protocol의 차이를 측정합니다. 결과를 CTR, 시청시간, 만족도 또는 인과효과로 해석하지 않습니다.

## Package 구조

```text
src/
├─ extraction/               # visual evidence, scene Graph/Description, video summaries
├─ validation/               # cohort, BGE, SASRec, diagnosis
└─ viewing_context_pipeline/ # fixed config context와 전체 DAG runner
```

`extraction`과 `validation`은 독립 CLI를 제공합니다. 전체 runner도 같은 step handler를 호출하므로 별도 구현 경로가 없습니다.

## Step DAG

```text
prepare-cohort
  → prepare-input-data
    ├→ extract-graph-scenes-qwen   → summarize-graph-qwen
    ├→ extract-graph-scenes-gemini → summarize-graph-gemini
    └→ extract-description-scenes  → summarize-description
      → embed-representations → run-recommendation → run-diagnosis
```

두 Graph extractor는 같은 fixed-30s 장면, 시간순 keyframe 1–3장, prompt와 taxonomy를 사용합니다. 생성된 `minimal-semantic-scene/v1` graph는 source와 관계없이 동일한 Qwen 모델로 영상 단위 요약합니다.

Description arm은 같은 keyframe에서 장면별 factual description을 생성한 뒤 별도 Qwen 호출로 영상 단위 요약을 생성합니다. Qwen 생성은 deterministic 설정을 사용합니다.

## 설정

실행 시 `config/pipeline.yaml` 하나를 고정으로 읽습니다. 이 파일에는 protocol, 로컬 data/model 경로, generation, cohort, BGE, SASRec, evaluation 설정이 모두 포함됩니다.

```powershell
conda activate llmjg
python -m pip install -e ".[qwen,gemini,train]"
gcloud auth application-default login
```

```bash
conda activate llmjg
python -m pip install -e ".[qwen,gemini,train]"
gcloud auth application-default login
```

`config/pipeline.yaml`의 data/model 경로와 `models.gemini.project_id`, `location`, `model_id`를 실행 환경에 맞게 수정합니다. Gemini는 API key가 아닌 Vertex AI Application Default Credentials를 사용합니다.

Graph scene prompt와 minimal taxonomy는 `extraction.semantic_graph`가 코드로 소유합니다. 별도의 ontology JSON은 사용하지 않습니다.

Qwen scene analyzer는 category/type, `IS_A`, `INTERACTS_WITH`, relation family, scene function, media/style, confidence/score를 출력하지 않습니다. Facet Projector, PPR, TVTI 축·점수·UI는 이 pilot DAG 범위 밖입니다.

현재 Graph scene 출력은 semantic field, enum, ID reference를 검증하지 않습니다. JSON object 추출에 실패하면 로컬 deterministic repair를 적용하며, 복구할 수 없는 scene과 Gemini API 최종 실패는 raw 응답과 오류를 source별 `extraction/graph/{source}/failures/`에 저장합니다. 해당 scene만 제외하고 stage는 끝까지 진행하므로 arm별 실제 사용 scene 수가 다를 수 있습니다. Description의 빈 scene 응답도 `extraction/description/failures/`에 기록하고 같은 방식으로 계속 진행합니다.

entity 6개와 summary 150단어는 강제 제약이 아니라 prompt guidance입니다. 초과 결과는 절단하거나 실패시키지 않습니다. Graph scene JSONL은 `scene_idx`, `keyframes`, `graph`만 저장하며 summary 길이 초과만 summary의 `validation_warnings`에 기록합니다. 모델 재생성 retry는 하지 않습니다.

## 독립 step 실행

모든 명령은 명시적인 `--run-id`를 요구합니다. 선행 단계를 자동 실행하지 않으며, 해당 step의 실제 출력 파일이 이미 있으면 기본적으로 건너뜁니다. 선행 산출물이 없으면 `required stage is incomplete` 같은 manifest 오류 대신 실제 누락된 파일 또는 디렉터리 경로를 표시합니다.
`prepare-input-data`는 cohort item을 4개 worker thread로 병렬 처리하고 content 단위 progress bar를 표시합니다.

editable install을 하지 않은 Linux checkout에서는 저장소 루트에서 먼저 source root를 등록합니다. `src`는 module 이름이 아니므로 `python -m src.validation`은 사용하지 않습니다.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m validation prepare-cohort --run-id 1k_pilot_260825
```

```powershell
python -m validation prepare-cohort --run-id 1k_pilot_260825
python -m extraction prepare-input-data --run-id 1k_pilot_260825
python -m extraction extract-graph-scenes --run-id 1k_pilot_260825 --model qwen
python -m extraction summarize-graph --run-id 1k_pilot_260825 --source qwen
python -m extraction extract-graph-scenes --run-id 1k_pilot_260825 --model gemini
python -m extraction summarize-graph --run-id 1k_pilot_260825 --source gemini
python -m extraction extract-description-scenes --run-id 1k_pilot_260825
python -m extraction summarize-description --run-id 1k_pilot_260825
python -m validation embed-representations --run-id 1k_pilot_260825
python -m validation run-recommendation --run-id 1k_pilot_260825
python -m validation run-diagnosis --run-id 1k_pilot_260825
```

Graph 명령의 `--model`과 `--source`는 필수입니다. 각 CLI의 `--force`는 선택한 step의 실제 출력만 다시 생성합니다. downstream을 자동 삭제하거나 재실행하지 않습니다.

Qwen을 사용하는 stage는 `--gpus N`으로 GPU 개수를 지정할 수 있습니다. 예를 들어 `--gpus 2`는 CUDA 장치 `0`, `1`에 worker process를 하나씩 만들고 각 process에서 Qwen을 한 번 로드합니다. Gemini extraction에는 `--gpus`를 사용할 수 없으며 `extraction.graph.gemini_concurrency`의 기본 4개 thread로 Vertex API를 호출합니다.

```powershell
python -m extraction extract-graph-scenes --run-id 1k_pilot_260825 --model qwen --gpus 2
python -m extraction extract-graph-scenes --run-id 1k_pilot_260825 --model gemini
python -m extraction summarize-graph --run-id 1k_pilot_260825 --source qwen --gpus 2
python -m extraction summarize-graph --run-id 1k_pilot_260825 --source gemini --gpus 2
python -m extraction extract-description-scenes --run-id 1k_pilot_260825 --gpus 2
python -m extraction summarize-description --run-id 1k_pilot_260825 --gpus 2
```

Extraction stage는 content 단위 progress bar를 표시합니다. Graph 로그는 `[Graph_qwen]`, `[Graph_gemini]`, `[Summary_graph_qwen]`, `[Summary_graph_gemini]`로 source를 구분합니다. 이미 실제 출력이 있는 content는 별도 로그 없이 progress bar의 초기 완료 수에 포함됩니다.

## 전체 실행

```powershell
python -m viewing_context_pipeline run --run-id 1k_pilot_260824 --gpus 2
```

```bash
bash run.sh 1k_pilot_260824 --gpus 2
```

지원 옵션:

- `--dry-run`: preflight와 stage 순서만 표시하고 artifact를 쓰지 않음
- `--gpus N`: Qwen extraction/summarization에서 사용할 CUDA 장치 개수

root runner는 고정 순서로 모든 step을 호출하고, 각 step은 실제 출력 존재 여부로 resume합니다. config, prompt, model 또는 protocol을 바꿔 새 실험을 시작할 때는 새 `run-id`를 사용합니다. `--resume`, `--force-stage`, `--config`, `--local-config` 및 이전 underscore stage 이름은 지원하지 않습니다.

## Artifact 구조

```text
artifacts/{run_id}/
├─ data/
│  ├─ cohort/{catalog.jsonl,sequences.jsonl,item_inventory.jsonl,eligibility_summary.json}
│  └─ fixed_30s/resized_keyframes/{content_id}/
├─ extraction/
│  ├─ graph/qwen/{scenes,failures,summaries}/
│  ├─ graph/gemini/{scenes,failures,summaries}/
│  └─ description/{scenes,failures,summaries}/
└─ validation/
   ├─ representations/{graph_qwen_embeddings.npz,graph_gemini_embeddings.npz,desc_embeddings.npz,item_index.json}
   ├─ recommendations/{per_user_metrics.jsonl,checkpoints/}
   └─ diagnosis/diagnosis.json
```

`pipeline_manifest.json`, step manifest, `visual_manifest.jsonl`, `cohort_manifest.json`, representations/recommendations manifest, checkpoint manifest는 생성하지 않습니다. 실제 데이터 파일을 직접 읽고 고정 경로로 연결합니다. legacy flat-triple output은 읽지 않습니다.

주요 계약:

- `viewing-context-config/v1`
- `scene-description/v1`
- `description-video-summary/v1`
- `diagnosis/v1`

## Optional Multimodal API

ASR/OCR multimodal 전처리는 canonical DAG에 포함되지 않는 Python API입니다.

```powershell
python -m pip install -e ".[multimodal]"
```

```python
from extraction import prepare_multimodal_evidence

result = prepare_multimodal_evidence(
    "video.mp4",
    "timestamp_fixed_30s.json",
    "optional_multimodal_output",
    asr_model="small",
    ocr_model_root="/models/PaddleOCR",
)
```

Optional API도 별도 manifest 없이 `multimodal_timeline.jsonl`, ASR, OCR 실제 파일만 생성합니다. 이 결과는 고정 pilot 결과로 해석하지 않습니다.

## 검증

```powershell
conda activate llmjg
ruff check src tests
python -m pytest -q tests
python -m compileall -q src tests
git diff --check
```

synthetic test와 contract test 통과는 구현 검증입니다. 실제 MicroLens·GPU pilot이 실행되기 전에는 실제 추천 품질 결과로 보고하지 않습니다.
