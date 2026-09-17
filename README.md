# ViewingContextPipeline v5

![ViewingContextPipeline 구성도](docs/design/Diagram_preview.png)

MicroLens-100K 영상의 Description, Scene Graph, 영문 title 표현을 BGE로 임베딩하고 동일한 SASRec 구조로 추천 성능을 비교하는 PoC입니다. 시각 표현이 title을 대체했을 때의 효과를 측정합니다.

## 비교 Arm

각 Run은 선택한 프롬프트로 하나의 Graph 표현을 평가합니다. ASIS·TOBE 전용 설정이나 버전별 Arm은 없으며, 프롬프트 경로·내용 해시와 생성 provenance로 실험을 구분합니다.

| 입력 표현 | Arm / CLI target |
| --- | --- |
| Gemini Description → Qwen Summary | `desc_gemini` |
| Qwen Description → Qwen Summary | `desc_qwen` |
| Gemini Graph → Qwen Summary | `graph_gemini` |
| Qwen Graph → Qwen Summary | `graph_qwen` |
| 영문 title | `metadata` |

```yaml
protocol:
  arms: [desc_gemini, desc_qwen, graph_gemini, graph_qwen, metadata]
```

`protocol.arms`는 실행할 Arm 목록입니다. embedding·추천·진단에서 `--target`을 생략하면 이 목록을 사용합니다. CLI, 저장 문서의 `arm`, embedding 파일명, 추천 디렉터리와 진단은 위의 고정 이름을 사용합니다. 프롬프트 파일명의 버전을 바꿔도 Arm 이름은 바뀌지 않습니다.

Run 내부에서는 표현 방식(Description–Graph), 추출 모델(Gemini–Qwen), 컨텍스트 제공(시각 표현–Metadata)을 비교합니다. Graph 프롬프트 간 비교는 Run을 나누어 수행합니다. 모델 비교는 현재 프롬프트·fallback을 포함한 파이프라인 비교로 해석합니다.

## 실행 준비

Ubuntu/Bash, Python 3.11+와 저장소 루트를 기준으로 합니다. 입력 준비에는 ffmpeg·ffprobe가 필요합니다. 모델과 데이터는 자동 다운로드하지 않습니다.

```bash
python -m pip install -e ".[qwen,train,gemini,dev]"
python -m pip check
```

Gemini 전용 환경은 `.[gemini,dev]`로 설치할 수 있습니다. Vertex ADC와 설정한 프로젝트·모델 호출 권한, 준비된 cohort·timestamp·공유 keyframe이 필요합니다. Qwen은 로컬 checkpoint와 vLLM 0.28.0을 사용합니다. [Qwen 실행 가이드](docs/qwen_vllm.md)를 참고하세요.

[config.yaml](config.yaml)에서 다음 경로와 모델 설정을 맞춥니다.

| 설정 | 입력 |
| --- | --- |
| `data.pairs_csv` | `user,item,timestamp` CSV, 정수 밀리초 |
| `data.pairs_tsv` | 존재하면 CSV와 사용자별 interaction multiset을 검증 |
| `data.videos_dir` | `{item_id}.mp4` 파일들 |
| `data.titles_csv` | 원본 `MicroLens-100k_title_en.csv` (영문 title) |
| `data.titles_supplement_csv` | 보완용 `MicroLens-50k_titles.csv`. 지정하면 빈 제목·누락 행을 자동 보완 |
| `models.qwen`, `models.bge` | 로컬 모델 디렉터리 |
| `models.gemini` | Vertex 프로젝트·위치·모델·생성 설정 |
| `artifacts_root` | 산출물 상위 경로. 기본 `artifacts` |

## 전체 실행

```bash
RUN_ID=experiment_v5

python -m preparation prepare-cohort --run-id "$RUN_ID"
python -m preparation prepare-input-data --run-id "$RUN_ID"

python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/description_scene_v2.md --model qwen
python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/description_scene_v2.md --model gemini
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/graph_scene_v3.md --model qwen
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/graph_scene_v3.md --model gemini

python -m extraction summarize-description --run-id "$RUN_ID" --schema prompts/description_summary_v4.md --source qwen
python -m extraction summarize-description --run-id "$RUN_ID" --schema prompts/description_summary_v4.md --source gemini
python -m extraction summarize-graph --run-id "$RUN_ID" --schema prompts/graph_summary_v4.md --source qwen
python -m extraction summarize-graph --run-id "$RUN_ID" --schema prompts/graph_summary_v4.md --source gemini

python -m validation embed-representations --run-id "$RUN_ID"
python -m validation run-recommendation --run-id "$RUN_ID"
python -m validation run-diagnosis --run-id "$RUN_ID"
```

`prepare-cohort` 한 번으로 required items 생성 → 원본·보완 CSV의 제목 병합 → 영상·제목 검증을 완료합니다. 원본의 비어 있지 않은 제목을 우선하고, 필요한 아이템의 빈 제목·누락 행만 보완합니다. 끝내 찾지 못한 제목은 빈 값으로 저장해 Metadata embedding에서 영벡터로 처리합니다. 보완 CSV 자체가 없거나 손상된 경우에는 실패합니다.

병합 결과는 `artifacts/runs/$RUN_ID/cohort/metadata_titles.jsonl`, 출처·보완 통계는 같은 폴더의 `cohort_plan.json`에 저장합니다. 원본 CSV를 변경하거나 별도 completed CSV·report 파일을 만들지 않습니다. `--plan-only`나 별도 `validation.complete_titles` 실행은 필요하지 않습니다. 재실행하면 최신 CSV를 다시 읽습니다. 이미 완성된 제목 CSV를 사용할 때는 `data.titles_csv`에 지정하고 `data.titles_supplement_csv`를 생략하면 됩니다(이 경우 누락 행은 오류).

기존 `validation.complete_titles` 독립 명령도 유지합니다. 수동 실행 시 `--required-items`의 현재 경로는 `artifacts/runs/$RUN_ID/cohort/required_items.jsonl`이며 이전 `data/cohort/` 경로를 사용하지 않습니다. `--plan-only`는 목록만 미리 확인할 때 선택적으로 사용할 수 있습니다.

`--schema`는 **실제 존재하는 Markdown 프롬프트 파일 하나**입니다. 저장소 루트 기준 상대 경로와 절대 경로를 허용합니다. 와일드카드 문자열은 허용하지 않습니다. 네 생성 명령에서 필수이며, 추출은 `--model`, 요약은 `--source`도 필수입니다. 요약 모델은 항상 Qwen이고 `--source`는 입력 장면을 만든 모델입니다. Graph 명령은 항상 `graph`에 쓰며 선택 프롬프트와 관계없이 `entities / relations / context` 출력 계약으로 검증합니다.

Qwen 추출·요약은 `CUDA_VISIBLE_DEVICES`에 지정된 GPU를 모두 사용하며 `--gpus` 인자를 받지 않습니다. 예를 들어 `CUDA_VISIBLE_DEVICES=0,2 python -m extraction summarize-graph --source gemini --run-id "$RUN_ID" --schema prompts/graph_summary_v4.md`는 두 GPU를 사용합니다. 환경 변수를 설정하지 않으면 CUDA에서 보이는 모든 GPU를 사용하고, 보이는 GPU가 없으면 오류를 냅니다. 추천도 보이는 GPU를 모두 사용하며 `--gpus`를 받지 않습니다. 기본 GPU당 한 작업을 실행하고 `--workers-per-gpu N`으로 GPU당 동시 작업 수를 조절합니다. 추천은 GPU가 없으면 기본 설정에서 CPU로 실행합니다. Gemini는 영상 구분 없이 scene 큐를 공유하며, 전체 장면 동시 실행 수는 `extraction.gemini.threads`입니다.

일부 Arm만 실행하는 예:

```bash
python -m validation embed-representations --run-id "$RUN_ID" --target desc_qwen graph_qwen metadata
python -m validation run-recommendation --run-id "$RUN_ID" --target desc_qwen graph_qwen metadata
python -m validation run-diagnosis --run-id "$RUN_ID" --target desc_qwen graph_qwen metadata
```

## Graph 프롬프트 비교: Run 분리

비교할 프롬프트마다 별도의 Run을 사용하고 동일한 cohort·catalog·평가 날짜·학습 설정·seed를 유지합니다. 각 Run에서 장면 추출 → 요약 → embedding → 추천을 완료한 뒤 비교합니다.

```bash
python -m validation run-diagnosis --run-id "$RUN_ID" --compare-run-id reference_run --target graph_gemini graph_qwen
```

현재 Run에서 reference Run을 뺀 NDCG@10 차이를 `validation/diagnosis/diagnosis.json`의 `run_comparison`에 기록합니다. 원본 사건·catalog·평가 날짜·seed와 사건 수가 일치해야 하며 두 Run의 추천 캐시도 검증합니다. 모델별 두 비교는 Bonferroni 보정, 개선 폭의 모델 간 차이는 탐색적 95% 구간을 사용합니다. 프롬프트·모델·요약 정책·fallback의 차이가 함께 포함될 수 있으므로 출처를 확인합니다.

`--schema`는 프롬프트 선택이며 Python 출력 검증 계약을 바꾸지 않습니다. 다른 본문 구조의 과거 Graph나 Summary를 그대로 입력하는 ASIS 전용 경로·본문 자동 변환은 제공하지 않습니다. 기존 artifact를 수동 재사용하려면 현재 계약과 provenance를 충족해야 합니다.

## Artifact 구조

```text
artifacts/
├── resized_keyframes/{content_id}/{timestamp}.png
├── source_assets/{content_id}/
│   ├── video_duration.json
│   └── timestamp_fixed_30s.json
└── runs/{RUN_ID}/
    ├── cohort/
    ├── extraction/
    │   ├── description/{gemini|qwen}/{scenes|summaries}/
    │   └── graph/{gemini|qwen}/{scenes|summaries}/
    └── validation/
        ├── representations/
        ├── recommendations/{date}/seed_{seed}/{arm}/
        └── diagnosis/diagnosis.json
```

`scenes/{content_id}.jsonl`은 장면당 한 줄이며, Description·Graph 모두 `content_id`, `scene_idx`, 본문 세 필드만 저장합니다. Qwen·Gemini에 같은 형식을 적용합니다.

```json
{"content_id":"123","scene_idx":0,"description":"A person walks outdoors."}
{"content_id":"123","scene_idx":0,"scene_graph":{"entities":[],"relations":[],"context":[]}}
```

장면 번호 순서로 저장하며 `.metadata`는 만들지 않습니다. Graph의 Raw Output은 `scene_graph` 문자열로 저장하고, Desc의 Raw Output은 `description`에 저장하며 같은 장면의 실패 기록으로 식별합니다. 중단·비동기 완료로 장면이 빠져 있어도 각 행의 `scene_idx`로 정확히 재개합니다.

기존 전체 필드 JSONL과 두 필드 JSONL도 읽을 수 있습니다. 두 필드 파일은 원래 `.metadata`에서 장면 번호를 읽어야 하므로 먼저 삭제하지 마세요. 추출 재실행 또는 아래 명령으로 세 필드 형식으로 변환하며, 저장이 성공한 파일의 기존 메타데이터만 삭제합니다. 메타데이터가 없거나 본문과 맞지 않는 이전 파일은 장면 번호를 추측하거나 전체 재생성하지 않고 오류를 보고합니다. 별도 변환 명령은 해당 Run의 장면 추출을 중지한 상태에서 실행합니다.

```bash
python -m extraction migrate-scene-schema --run-id "$RUN_ID"
```

공유 프레임과 영상 길이·장면 timestamp는 `artifacts_root` 바로 아래의 `resized_keyframes`, `source_assets`에 저장하며 Run 간 재사용합니다. 최초 준비나 필요한 파일이 없는 경우 `prepare-input-data`를 실행합니다. 공유 timestamp와 프레임이 준비되어 있으면 새 Run에서도 `prepare-cohort` 후 바로 추출할 수 있습니다. 장면 추출은 `prepare-input-data`가 정상 완료됐다고 가정합니다. 이미지·asset 존재 검사, 폴더 스캔, duration·샘플링 재검증 및 이미지 내용 해시 계산을 하지 않습니다. timestamp JSON으로 scene 수와 재사용 여부를 확인하고, 추론 입력을 공급할 때 필요한 경로를 준비 단계의 PNG 파일명 규칙으로 구성합니다. 이미지는 실제 추론 시 읽으며, 필요한 파일이 없으면 해당 입력을 읽는 시점에 실패합니다. 자동 준비 호출은 하지 않습니다.

장면 추출은 추론 전에 timestamp와 저장 결과를 확인해 처리할 전체 scene 수를 계산합니다. 추론용 task와 이미지는 요청을 공급하면서 읽습니다. Gemini는 제한된 수의 scene을 미리 공급하고 빈 worker가 영상 경계 없이 다음 scene을 바로 처리합니다. 처리 속도와 ETA는 추론 단계의 완료 건수와 경과 시간을 기준으로 표시합니다.

기본 6개 keyframe 정책은 `timestamp_fixed_{scene_duration}s.json`, 다른 개수는 `timestamp_fixed_{scene_duration}s_{num_keyframes}kf.json`을 사용하므로 서로 다른 정책이 공유 timestamp를 덮어쓰지 않습니다. 콘텐츠별 잠금과 원자적 저장으로 동시 준비를 보호합니다. `--force`도 정상 이미지를 덮어쓰지 않습니다. `prepare-input-data`를 명시적으로 실행할 때는 공유 duration의 원본 식별 정보가 달라지면 재사용을 거부하므로 변경된 영상은 별도 `artifacts_root`로 분리합니다.

`prepare-input-data`의 `Prepare visual evidence` 진행률은 영상별 준비·검증 건수입니다. `reused_frames`는 재사용 이미지 수, `new_frames`는 실제 신규 추출 이미지 수입니다. 누락된 이미지를 추출할 때만 `[KEYFRAMES] extracting ...`과 누락 timestamp를 출력합니다. 공유 duration과 timestamp가 유효하면 영상 길이를 재확인하거나 timestamp를 다시 만들지 않습니다. 준비 실패 보고서는 각 Run의 `cohort/preparation_failures.jsonl`에 저장합니다.

기존 Run은 `artifacts/runs/{RUN_ID}/`로, 기존 준비 정보는 `artifacts/source_assets/`로 직접 배치해야 합니다. 자동 Run 이동·삭제는 수행하지 않습니다.

Qwen Desc·Graph와 모든 Summary는 설정된 repetition penalty별로 전체 pass를 완료하고 실패 항목만 다음 pass에서 재생성합니다. 기본 순서는 `1.00 → 1.05 → 1.10 → 1.15 → 1.20`이며 성공 항목은 제외합니다. Vertex Gemini의 `429 RESOURCE_EXHAUSTED`만 30초 뒤 한 번 재시도하며, Summary 교정 및 `.recovery`, `.pending`, `.checkpoints`, 콘텐츠 진행 cursor를 저장하지 않습니다. 장면 번호는 본문 파일의 `scene_idx`에, 요약의 생성 당시 provenance는 요약 문서에 남깁니다.

장면 실패는 `scenes/failures/{content_id}.jsonl`에 `content_id`, `scene_idx`, `error`, `raw_output`을 저장합니다. Summary 생성 실패는 `summaries/failures.jsonl`에 `content_id`, `error`, `raw_output`, `repetition_penalty`를 저장합니다. `raw_output`은 공백·줄바꿈까지 보존하며 응답이 없으면 빈 문자열입니다. 장면 생성은 재실행 시 첫 penalty부터 실패 항목을 처리합니다. Summary는 아직 처리하지 않은 콘텐츠를 먼저 초기 penalty로 처리한 뒤, 실패별 마지막 penalty보다 큰 다음 설정값부터 이어갑니다. 마지막 penalty까지 완료했거나 penalty 필드가 없는 기존 실패는 재생성 없이 콘텐츠별 `.json`의 `text`로 복구합니다. 원문이 비어 있으면 Scene 관측 내용을 대신 저장하고 provenance에 `summary_fallback=scene_observations`를 기록합니다. 최종 실패 기록은 유지하며 `--force`로 전체 재생성을 요청할 수 있습니다. 재시도 성공 결과를 저장한 뒤 해당 실패 행을 제거하고, 남은 실패가 없으면 파일도 삭제합니다. 실행 시작 시 이전 실패 파일을 현재 형식으로 옮기며 기존 penalty와 원문을 보존합니다.

```json
{"content_id":"123","scene_idx":2,"error":"model produced an empty description","raw_output":""}
{"content_id":"123","error":"max_tokens","raw_output":"Truncated output...","repetition_penalty":1.2}
```

진행률의 `success`와 `failed`는 현재 패스에서 완료된 성공·실패 수이고, `raw`는 그중 fallback 결과를 보존한 수입니다. Qwen 재시도는 실패 항목만 다음 패스로 넘기며, 매 패스 시작 시 분모를 실제 처리 대상 수로 바꾸고 완료 건수·경과 시간을 초기화합니다. `pass=N/M`은 설정된 penalty 순서를 나타내며, `scene/s`와 `summary/s` 및 ETA는 현재 패스 기준입니다. 재사용 결과는 집계에서 제외하고 전체 최종 결과는 단계 종료 로그에 표시합니다. Qwen·Gemini Graph와 Qwen Summary의 토큰 한도 종료는 실패로 기록합니다.

각 추천 조합의 `training.json`, `per_event_metrics.jsonl`, `complete.json`, **최종 `sasrec.pt`**를 보존합니다. 기존 run이나 수동 보관한 archive는 자동 삭제하지 않습니다.

## 생성 정책과 재사용

- 기본 장면 상한은 1,024 tokens, 요약 상한은 512 tokens입니다.
- 화면 텍스트는 의미 해석의 단서로만 사용하며 문구 전사·인용·번역 출력은 금지하도록 지시합니다. 근거 있는 장르·목적·배경지식 해석을 허용하고 불확실성을 보존합니다.
- Qwen Graph·Description·Summary는 repetition penalty 목록 순서로 실패 항목만 재생성합니다. 마지막까지 실패하면 최종 원문을 Raw로 사용하며, 빈 장면 원문은 실패 기록만 남기며, 빈 Summary 원문은 Scene 관측 내용으로 대체합니다. 성공한 항목은 재생성하지 않습니다.
- Graph는 `entities`, `relations`, `context`입니다. `name`은 자유 어휘 **개체 종류**이고 실명이나 고유 신원이 아닙니다. 중복 `name`은 허용하며 고유한 장면 내부 `id`와 외형·상태·활동 `attributes`로 구분합니다. 현재 E2E 실행에서는 ID 중복·관계 참조 일치 여부를 검증하지 않으며, 생성된 ID와 관계를 그대로 저장합니다. 필수 필드·타입·빈 문자열 등 JSON 구조 검증은 유지합니다. 객체 추적기는 없으며 장면 사이 ID를 연결하지 않습니다.
- Graph 생성에는 JSON Schema를 강제하지 않습니다. `graph_scene_v3.md`의 줄 단위 출력을 두 모델의 공통 파서가 기존 JSON 구조로 변환합니다. 화살표·하이픈 구분자, 헤더, bullet 등 명확한 형식 변형은 Repair하고, 누락·모호한 행·토큰 잘림은 원문과 실패 기록을 보존합니다. 완전한 JSON 응답도 지원하며, 잘린 내용을 추측해서 채우지 않습니다. 상세 규칙은 [Qwen 실행 가이드](docs/qwen_vllm.md)에 있습니다.
- 신규 Summary는 문단·목록·마크업·구조화 필드를 허용하며 줄바꿈을 보존합니다. 단어 수는 기록만 하고 실패 조건으로 사용하지 않습니다. 빈 출력과 `finish_reason=length`만 실패로 기록하고 다음 penalty에서 재생성합니다. 마지막 실패 원문은 `raw_fallback`으로 저장하며, 원문이 비어 있으면 프롬프트 지시문을 제외한 Scene 관측 내용을 저장합니다. 유효한 Scene 입력이 없으면 명시적인 입력 오류로 보고합니다.
- Qwen·Gemini Graph·Desc 및 모든 Summary는 설정·프롬프트·경로가 달라도 기존 성공 결과를 그대로 재사용합니다. 일반 실행은 결과가 없는 항목과 재시도가 남은 실패만 생성합니다. 장면을 갱신해도 이미 성공한 Summary는 보존하며, 성공 결과까지 다시 만들려면 해당 단계에 `--force`를 지정합니다. 실패 기록은 재생성 성공 후 제거합니다. 다른 run ID의 결과를 자동으로 가져오지는 않습니다.
- Gemini Summary **파일이 없을 때만** 같은 Run·표현의 Qwen Summary로 fallback합니다. Raw Gemini가 있으면 우선 사용합니다. 손상된 일반 파일은 오류이며 Qwen 요약 누락을 영벡터로 대체하지 않습니다. 실제 사용 경로를 기록하고 Gemini 결과가 추가되면 embedding·추천 캐시를 갱신합니다.
- BGE 입력 상한은 512 tokens이고 실제 truncation 건수를 기록합니다. 빈 Metadata title만 영벡터를 사용합니다.

## 평가와 검증

기본 데이터는 사용자 100,000명·interaction 719,405건·아이템 19,738개입니다. 같은 사건·후보 catalog에서 날짜별 독립 selection → refit → test를 실행합니다. 5개 Arm은 **7일 × 3 seeds × 5 = 105개 조합**입니다.

NDCG@10을 주 지표로 사용자 단위 paired bootstrap 10,000회, seed 평균 후 날짜 균등 평균을 적용합니다. Metadata 대비 4개, 표현 방식 2개, 모델 2개 비교군에 각각 α=0.05와 Bonferroni 양측 신뢰구간을 적용합니다. 부분 target에서도 보정 분모 4·2·2은 유지합니다. Graph 개선 폭의 모델 간 차이는 탐색적으로 보고합니다. Description·Graph의 기존 coverage 기준을 유지합니다. 자세한 규칙은 [전체 실험 가이드](docs/full_rolling.md)에 있습니다.

```bash
python -m pytest -q
ruff check --config pyproject.toml src tests benchmarks
```

v5 mock 전체 흐름, 작은 CPU 학습, backend·분할·통계·실패 처리 회귀 테스트를 사용합니다. 전체 MicroLens 추론·학습과 실제 GPU/API 출력 품질 평가는 별도 실험 실행 단계입니다.
