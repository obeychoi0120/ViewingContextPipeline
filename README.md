# ViewingContextPipeline

## Pipeline Overview

![ViewingContextPipeline overview](src/ViewingContextPipeline_260825.png)

MicroLens-100K local MP4에서 visual-only Viewing Context를 추출하고, 같은 cohort·split·seed의 독립 SASRec arm으로 비교하는 파이프라인입니다.

고정 pilot protocol은 다음과 같습니다.

```text
MicroLens-100K
  + visual_only
  + fixed_30s (5s, 15s, 25s keyframes)
  + Qwen3-VL-2B
  + SASRec_ID vs SASRec_GRAPH vs SASRec_DESC
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
  → prepare-visual-evidence
    ├→ extract-graph-scenes → summarize-graph
    └→ extract-description-scenes → summarize-description
      → embed-representations → run-recommendation → run-diagnosis
```

Graph arm은 장면별 `subject-relation-object` triple을 만들고 Qwen으로 영상 단위 요약을 생성합니다.

Description arm은 같은 keyframe에서 장면별 factual description을 생성한 뒤 별도 Qwen 호출로 영상 단위 요약을 생성합니다. 두 arm은 같은 visual evidence fingerprint와 deterministic generation 설정을 사용합니다.

## 설정

사용자가 선택할 pipeline config는 없습니다. 다음 두 파일을 고정으로 읽습니다.

- `config/pipelines/microlens_graph_vs_desc_pilot.yaml`: protocol, generation, cohort, BGE, SASRec, evaluation
- `config/local.yaml`: MicroLens와 Qwen/BGE의 machine-local 경로

```powershell
conda activate llmjg
Copy-Item config/local.example.yaml config/local.yaml
python -m pip install -e ".[qwen,train]"
```

Graph ontology와 prompt는 고정 pilot YAML에서 명시적으로 참조합니다.

- `relational-graph-ontology/v1`은 현재 `provisional`입니다.
- 최종 팀 taxonomy/prompt가 같은 asset interface로 교체되면 fingerprint가 바뀌어 Graph scene과 downstream만 다시 실행됩니다.
- `scene_type`, `people_density`, `graphic_density` 및 미사용 필드는 provisional Graph 출력에 포함하지 않습니다.

## 독립 step 실행

모든 명령은 명시적인 `--run-id`를 요구합니다. 선행 단계를 자동 실행하지 않습니다.

editable install을 하지 않은 Linux checkout에서는 저장소 루트에서 먼저 source root를 등록합니다. `src`는 module 이름이 아니므로 `python -m src.validation`은 사용하지 않습니다.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m validation prepare-cohort --run-id 1k_pilot_260825
```

```powershell
python -m validation prepare-cohort --run-id 1k_pilot_260824
python -m extraction prepare-visual-evidence --run-id 1k_pilot_260824
python -m extraction extract-graph-scenes --run-id 1k_pilot_260824
python -m extraction summarize-graph --run-id 1k_pilot_260824
python -m extraction extract-description-scenes --run-id 1k_pilot_260824
python -m extraction summarize-description --run-id 1k_pilot_260824
python -m validation embed-representations --run-id 1k_pilot_260824
python -m validation run-recommendation --run-id 1k_pilot_260824
python -m validation run-diagnosis --run-id 1k_pilot_260824
```

각 CLI의 `--force`는 요청한 step만 다시 수행하고 실제 downstream manifest만 stale 처리합니다. 반대쪽 Extraction arm의 산출물은 보존합니다.

## 전체 실행

```powershell
python -m viewing_context_pipeline run --run-id 1k_pilot_260824
```

```bash
bash run.sh 1k_pilot_260824
```

지원 옵션:

- `--resume`: 같은 v1 config snapshot의 기존 run을 이어서 실행
- `--force-stage <step>`: 해당 step과 실제 downstream만 재실행
- `--dry-run`: preflight와 stage 순서만 표시하고 artifact를 쓰지 않음

`--config`, `--local-config` 및 이전 underscore stage 이름은 지원하지 않습니다.

## Artifact 구조

```text
artifacts/{run_id}/
├─ runtime/config_snapshot.json
├─ pipeline_manifest.json
├─ manifests/{step}.json
├─ data/
│  ├─ cohort/
│  └─ fixed_30s/visual_manifest.jsonl
├─ extraction/
│  ├─ graph/scenes/{content_id}.jsonl
│  ├─ graph/summaries/{content_id}.json
│  ├─ description/scenes/{content_id}.jsonl
│  └─ description/summaries/{content_id}.json
└─ validation/
   ├─ representations/
   ├─ recommendations/
   └─ diagnosis/
```

새 run은 기존 v1 artifact나 legacy output을 읽지 않습니다. 각 step manifest는 upstream, ontology, prompt, model 및 output fingerprint를 기록합니다.

주요 계약:

- `relational-graph-ontology/v1`
- `scene-relational-graph/v1`
- `graph-video-summary/v1`
- `scene-description/v1`
- `description-video-summary/v1`
- `visual-manifest/v1`
- `step-manifest/v1`
- `representations/v1`
- `recommendations/v1`
- `diagnosis/v1`
- `pipeline-run/v1`

## Optional Extraction API

Gemini와 ASR/OCR multimodal 전처리는 canonical DAG에 포함되지 않는 Python API입니다. 필요한 extra만 별도로 설치합니다.

```powershell
python -m pip install -e ".[gemini]"
python -m pip install -e ".[multimodal]"
```

Graph와 Description 코어는 `VLMBackend.generate(images, prompt, max_new_tokens, references=())` 계약을 사용합니다. `QwenBackend`와 `GeminiBackend`가 이 계약을 구현합니다.

```python
from extraction import GeminiBackend, prepare_multimodal_evidence

backend = GeminiBackend.vertex(
    project_id="my-project",
    model_id="gemini-model-id",
)

manifest = prepare_multimodal_evidence(
    "video.mp4",
    "timestamp_fixed_30s.json",
    "optional_multimodal_output",
    asr_model="small",
    ocr_model_root="/models/PaddleOCR",
)
```

Optional API의 결과는 고정 pilot 결과나 canonical stage manifest로 해석하지 않습니다.

## 검증

```powershell
conda activate llmjg
ruff check src tests
python -m pytest -q tests
python -m compileall -q src tests
git diff --check
```

synthetic test와 contract test 통과는 구현 검증입니다. 실제 MicroLens·GPU pilot이 실행되기 전에는 실제 추천 품질 결과로 보고하지 않습니다.
