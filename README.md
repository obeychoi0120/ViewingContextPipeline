# ViewingContextPipeline v5

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

python -m validation prepare-cohort --run-id "$RUN_ID"
python -m extraction prepare-input-data --run-id "$RUN_ID"

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

병합 결과는 `artifacts/$RUN_ID/cohort/metadata_titles.jsonl`, 출처·보완 통계는 같은 폴더의 `cohort_plan.json`에 저장합니다. 원본 CSV를 변경하거나 별도 completed CSV·report 파일을 만들지 않습니다. `--plan-only`나 별도 `validation.complete_titles` 실행은 필요하지 않습니다. 재실행하면 최신 CSV를 다시 읽습니다. 이미 완성된 제목 CSV를 사용할 때는 `data.titles_csv`에 지정하고 `data.titles_supplement_csv`를 생략하면 됩니다(이 경우 누락 행은 오류).

기존 `validation.complete_titles` 독립 명령도 유지합니다. 수동 실행 시 `--required-items`의 현재 경로는 `artifacts/$RUN_ID/cohort/required_items.jsonl`이며 이전 `data/cohort/` 경로를 사용하지 않습니다. `--plan-only`는 목록만 미리 확인할 때 선택적으로 사용할 수 있습니다.

`--schema`는 **실제 존재하는 Markdown 프롬프트 파일 하나**입니다. 저장소 루트 기준 상대 경로와 절대 경로를 허용합니다. 와일드카드 문자열은 허용하지 않습니다. 네 생성 명령에서 필수이며, 추출은 `--model`, 요약은 `--source`도 필수입니다. 요약 모델은 항상 Qwen이고 `--source`는 입력 장면을 만든 모델입니다. Graph 명령은 항상 `graph`에 쓰며 선택 프롬프트와 관계없이 `entities / relations / context` 출력 계약으로 검증합니다.

Qwen 명령에는 `--gpus N`, 추천에는 `--gpus N --workers-per-gpu N`을 사용할 수 있습니다. Gemini 추출에는 `--gpus`를 사용하지 않습니다. Gemini의 콘텐츠 내부 장면 동시 실행 수는 `extraction.gemini.threads`입니다.

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

`--schema`는 프롬프트 선택이며 Python 출력 검증 계약을 바꾸지 않습니다. 다른 구조의 과거 Graph나 Summary를 그대로 입력하는 ASIS 전용 경로·자동 변환은 제공하지 않습니다. 기존 artifact를 수동 재사용하려면 현재 계약과 provenance를 충족해야 합니다.

## Artifact 구조

```text
artifacts/
├── resized_keyframes/{content_id}/{timestamp}.png
└── RUN_ID/
    ├── cohort/
    │   └── source_assets/{content_id}/assets/
    ├── extraction/
    │   ├── description/{gemini|qwen}/{scenes|summaries}/
    │   └── graph/{gemini|qwen}/{scenes|summaries}/
    └── validation/
        ├── representations/
        ├── recommendations/{date}/seed_{seed}/{arm}/
        └── diagnosis/diagnosis.json
```

공유 프레임은 `artifacts_root` 바로 아래에 있으며 run마다 복사하지 않습니다. 각 run의 timestamp와 duration cache는 `cohort/source_assets`에 저장합니다. 새 run에 timestamp가 없어도 정상 PNG는 재사용하고 누락된 프레임만 콘텐츠별 잠금과 원자적 게시로 추가합니다. `--force`도 정상 공유 이미지를 덮어쓰지 않습니다. 손상되거나 해상도가 다른 기존 PNG는 오류로 알리므로 직접 정리한 뒤 실행합니다.

실패 파일은 실패가 있을 때만 생성합니다. 완료된 recovery journal과 임시 checkpoint·dirty·pending 표식은 정리하고, 최종 장면·요약의 provenance와 간단한 생성 이력은 남깁니다. 별도 migration manifest, 전체 설정 snapshot, 요약별 `.inputs` 및 `.changed`, run별 이미지, media preflight·metadata missing 문서는 만들지 않습니다. 준비 통계는 콘솔, 결측 title 진단은 최종 diagnosis에 포함합니다. embedding별 `.inputs`는 실제 입력·fallback·truncation 및 캐시 검증 상태이므로 보존합니다.

각 추천 조합의 `training.json`, `per_event_metrics.jsonl`, `complete.json`, **최종 `sasrec.pt`**를 보존합니다. 기존 run이나 수동 보관한 archive는 자동 삭제하지 않습니다.

## 생성 정책과 재사용

- 기본 장면 상한은 1,024 tokens, 요약·교정 상한은 512 tokens입니다.
- 화면 텍스트는 의미 해석의 단서로만 사용하며 문구 전사·인용·번역 출력은 금지하도록 지시합니다. 근거 있는 장르·목적·배경지식 해석을 허용하고 불확실성을 보존합니다.
- Graph는 `entities`, `relations`, `context`입니다. `name`은 자유 어휘 **개체 종류**이고 실명이나 고유 신원이 아닙니다. 중복 `name`은 허용하며 고유한 장면 내부 `id`와 외형·상태·활동 `attributes`로 구분합니다. 관계의 양 끝 ID를 검증합니다. 객체 추적기는 없으며 장면 사이 ID를 연결하지 않습니다.
- 신규 Summary는 자연스러운 영어 한 문단, 100–200단어 권장·최대 200단어입니다. 정보가 적으면 100단어 미만도 허용합니다. 필드별 할당과 summary grammar는 없습니다. 형식·길이 위반은 원본 장면 관찰을 포함해 한 번 교정하고, 여전히 실패하면 마지막 비어 있지 않은 결과를 명시적 Raw로 사용합니다. 빈 결과와 실행 오류는 구분합니다.
- 프롬프트 경로·내용, 모델·설정·입력 hash가 바뀌면 생성 캐시를 재사용하지 않습니다. 로컬 모델 서명은 설정·tokenizer 텍스트의 내용 hash와 가중치 파일명·크기·mtime으로 계산합니다. 완료 journal을 삭제해도 최종 artifact의 provenance로 재개합니다.
- Gemini Summary **파일이 없을 때만** 같은 Run·표현의 Qwen Summary로 fallback합니다. Raw Gemini가 있으면 우선 사용합니다. 손상된 일반 파일은 오류이며 Qwen 요약 누락을 영벡터로 대체하지 않습니다. 실제 사용 경로를 기록하고 Gemini 결과가 추가되면 embedding·추천 캐시를 갱신합니다.
- BGE 입력 상한은 512 tokens이고 실제 truncation 건수를 기록합니다. 빈 Metadata title만 영벡터를 사용합니다.

## 평가와 검증

기본 데이터는 사용자 100,000명·interaction 719,405건·아이템 19,738개입니다. 같은 사건·후보 catalog에서 날짜별 독립 selection → refit → test를 실행합니다. 5개 Arm은 **7일 × 3 seeds × 5 = 105개 조합**입니다.

NDCG@10을 주 지표로 사용자 단위 paired bootstrap 10,000회, seed 평균 후 날짜 균등 평균을 적용합니다. Metadata 대비 4개, 표현 방식 2개, 모델 2개 비교군에 각각 α=0.05와 Bonferroni 양측 신뢰구간을 적용합니다. 부분 target에서도 보정 분모 4·2·2은 유지합니다. Graph 개선 폭의 모델 간 차이는 탐색적으로 보고합니다. Description·Graph의 기존 coverage 기준을 유지합니다. 자세한 규칙은 [전체 실험 가이드](docs/full_rolling.md)에 있습니다.

```bash
python -m pytest -q
ruff check --config pyproject.toml src tests benchmarks
```

v5 mock 전체 흐름, 작은 CPU 학습, backend·분할·통계·재시도 회귀 테스트를 사용합니다. 전체 MicroLens 추론·학습과 실제 GPU/API 출력 품질 평가는 별도 실험 실행 단계입니다.
