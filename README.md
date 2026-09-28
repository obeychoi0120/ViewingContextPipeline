# ViewingContextPipeline v4

![ViewingContextPipeline v4 구성도](docs/design/Diagram_preview.png)

생성 소스 4개에서 제목 없는 Summary를 만들고, 제목과의 결합 여부에 따라 6개 Arm을 평가합니다. 제목은 BGE 임베딩 직전에 결합합니다.

MicroLens-100K 영상의 Description, Scene Graph, 영문 title을 BGE로 임베딩하고 동일한 SASRec 구조로 추천 성능을 평가합니다. 주 비교는 `graph_qwen_meta − meta`로, 제목에 Graph 정보를 추가하는 효과입니다.

## 비교 Arm

| Arm / CLI target | BGE 입력 | 원본 Scene·Summary 소스 |
| --- | --- | --- |
| `meta` | 영어 제목 | 없음 |
| `graph_qwen` | Summary | `graph_qwen` |
| `graph_qwen_meta` | 제목 + `"\n\n"` + Summary | `graph_qwen` |
| `graph_gemini_meta` | 제목 + `"\n\n"` + Summary | `graph_gemini` |
| `desc_qwen_meta` | 제목 + `"\n\n"` + Summary | `desc_qwen` |
| `desc_gemini_meta` | 제목 + `"\n\n"` + Summary | `desc_gemini` |

metadata는 LLM 입력에 넣지 않습니다. 결합 Arm은 제목 없는 Summary를 단독 Arm과 공유하며, 임베딩 직전에 양끝 공백을 제거하고 빈 줄 하나로 결합합니다. 하나만 있으면 해당 텍스트만, 둘 다 없으면 영벡터를 사용합니다. 단독 Arm은 다른 입력으로 대체하지 않습니다.

`protocol.arms` 설정과 기본 실행 Arm은 없습니다. 지원 Arm은 코드 registry의 위 6개이며, embedding·추천·진단마다 `--target`을 명시해야 합니다. 생성 소스는 `graph_qwen`, `desc_qwen`, `graph_gemini`, `desc_gemini` 4개입니다. `desc_qwen`, `graph_gemini`, `desc_gemini`는 생성용 이름이므로 새 추천 `--target`에는 결합 Arm을 지정합니다. 결합 Arm을 생성 명령에 지정하면 원본 소스를 안내합니다. 기존 진단 manifest의 이전 Arm 계약은 읽기 호환성을 유지합니다.

Qwen/Gemini 접미사는 Scene 추출 모델입니다. Summary 모델은 `--model`로 별도 선택하고 같은 Run의 한 소스에는 한 Summary 모델만 저장합니다. 모델을 바꾸거나 기존 성공 결과를 갱신하려면 새 Run 또는 해당 단계의 `--force`를 사용합니다.

Summary는 `summary_description_v5.md`, `summary_graph_v5.md`를 사용합니다. 목표 200–300단어, 최대 350단어는 프롬프트 지침이며 별도 단어 수 검증은 없습니다. 정보가 적으면 더 짧게 작성합니다. 생성 한도는 `extraction.{description,graph}.{qwen,gemini}` 아래에서 장면·요약별로 설정하며 현재 모두 1,024토큰입니다. 요약은 원본 추출 모델이 아니라 실제 `--model`의 설정을 사용합니다. BGE 입력 한도는 512토큰으로 뒷부분을 자릅니다.

## 실행 준비

Context + Actions 방식의 실험용 Graph 프롬프트(`scene_graph_v5.md`, `summary_graph_v6.md`)와 첫 100개 비교 도구는 [설계·검증 가이드](docs/graph_context_actions.md)에 정리되어 있습니다. 기존 실행 경로는 유지하며, 파일럿 없이 기본 프롬프트를 자동 교체하지 않습니다.

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
RUN_ID=experiment_v4

python -m preparation prepare-cohort --run-id "$RUN_ID"
python -m preparation prepare-input-data --run-id "$RUN_ID"

python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/scene_description_v2.md --model qwen --arm desc_qwen
python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/scene_description_v2.md --model gemini --arm desc_gemini
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v4.md --model qwen --arm graph_qwen
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v4.md --model gemini --arm graph_gemini

python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_description_v5.md --model qwen --arm desc_qwen
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_description_v5.md --model gemini --arm desc_gemini
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v5.md --model qwen --arm graph_qwen
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v5.md --model qwen --arm graph_gemini

python -m validation embed-representations --run-id "$RUN_ID" --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
```

`prepare-cohort` 한 번으로 required items 생성 → 원본·보완 CSV의 제목 병합 → 영상·제목 검증을 완료합니다. 원본의 비어 있지 않은 제목을 우선하고, 필요한 아이템의 빈 제목·누락 행만 보완합니다. 끝내 찾지 못한 제목은 빈 값으로 저장해 Metadata embedding에서 영벡터로 처리합니다. 보완 CSV 자체가 없거나 손상된 경우에는 실패합니다.

병합 결과는 `artifacts/preparation/cohort/metadata_titles.jsonl`, 출처·보완 통계는 같은 폴더의 `cohort_plan.json`에 저장합니다. 원본 CSV를 변경하거나 별도 completed CSV·report 파일을 만들지 않습니다. `--plan-only`나 별도 `validation.complete_titles` 실행은 필요하지 않습니다. 재실행하면 최신 CSV를 다시 읽습니다. 이미 완성된 제목 CSV를 사용할 때는 `data.titles_csv`에 지정하고 `data.titles_supplement_csv`를 생략하면 됩니다(이 경우 누락 행은 오류).

기존 `validation.complete_titles` 독립 명령도 유지합니다. 수동 실행 시 `--required-items`의 현재 경로는 `artifacts/preparation/cohort/required_items.jsonl`이며 이전 `data/cohort/` 경로를 사용하지 않습니다. `--plan-only`는 목록만 미리 확인할 때 선택적으로 사용할 수 있습니다.

`--schema`는 **실제 존재하는 Markdown 프롬프트 파일 하나**입니다. 저장소 루트 기준 상대 경로와 절대 경로를 허용하며 와일드카드 문자열은 허용하지 않습니다. Graph 추출은 구조화 가능한 출력을 `entities / relations`로 저장하고, 그 외 출력은 원문 텍스트로 보존해 Summary에 전달합니다.

생성에는 `--schema`, `--model`, `--arm`이 필수입니다. Scene·Summary 명령의 `--arm`은 생성 소스 4개만 받습니다. Scene의 `--model`은 소스의 추출 모델과 같아야 합니다. Summary 프롬프트는 `{scenes}`가 필수이며 `{english_title}`은 허용하지 않습니다.

Validation은 결합 Arm의 원본 `summaries/{source}`를 읽고 실제 Summary 모델과 제목 미사용 provenance를 검증합니다. 제목을 넣어 생성한 과거 Summary는 새 결합 입력으로 사용하지 않습니다. `meta` 없이 부분 실행할 수 있으며, 선택하지 않은 소스는 요구하지 않습니다.

요약은 `summarize --arm ...`으로 실행하며, arm에 따라 graph/description 입력 형식을 자동 선택합니다. 기존 `summarize-graph`, `summarize-description` 명령도 호환용으로 지원합니다. `--model`은 요약 모델, `--schema`는 요약 프롬프트를 지정합니다.

Qwen 추출·요약은 `CUDA_VISIBLE_DEVICES`에 지정된 GPU를 모두 사용하며 `--gpus` 인자를 받지 않습니다. 예를 들어 `CUDA_VISIBLE_DEVICES=0,2 python -m extraction summarize --arm graph_gemini --model qwen --run-id "$RUN_ID" --schema prompts/summary_graph_v5.md`는 두 GPU를 사용합니다. 환경 변수를 설정하지 않으면 CUDA에서 보이는 모든 GPU를 사용하고, 보이는 GPU가 없으면 오류를 냅니다. 추천도 보이는 GPU를 모두 사용하며 `--gpus`를 받지 않습니다. 기본 GPU당 한 작업을 실행하고 `--workers-per-gpu N`으로 GPU당 동시 작업 수를 조절합니다. 추천은 GPU가 없으면 기본 설정에서 CPU로 실행합니다. Gemini는 영상 구분 없이 scene 큐를 공유하며, 전체 장면 동시 실행 수는 `extraction.gemini.threads`입니다.

일부 Arm만 실행하는 예:

```bash
python -m validation embed-representations --run-id "$RUN_ID" --target desc_qwen_meta graph_qwen_meta meta
python -m validation run-recommendation --run-id "$RUN_ID" --target desc_qwen_meta graph_qwen_meta meta
python -m validation run-diagnosis --run-id "$RUN_ID" --target desc_qwen_meta graph_qwen_meta meta
```

Validation은 preparation의 전체 catalog·item 순서·events·평가 구간을 Run별 `validation/cohort/`에 저장합니다. Summary 실패나 누락으로 아이템·interaction을 삭제하지 않습니다. 기존 이력 마스킹과 평가 이벤트 자격 규칙을 유지합니다. `--target`은 처리 Arm만 제한합니다.

일부 Scene이 실패하면 성공 장면만 Summary에 사용합니다. 성공 장면이 없거나 Summary가 최종 실패하면 `status="failed", text="", word_count=0`을 저장합니다. 실패 로그의 새 `raw_output`은 빈 문자열입니다. 기존 `raw_fallback`은 빈 표현으로 읽고 기존 파일은 일괄 변경하지 않습니다. Summary 파일 부재는 `missing`, 손상·잘못된 Arm/모델 provenance는 오류입니다. 빈 표현은 인코더에 보내지 않고 영벡터로 유지하며 고유 item ID와 추천 후보 자격을 보존합니다.

모든 Arm의 embedding과 완료된 추천 결과는 **현재 Run → 공유 캐시 → 새 계산** 순서로 처리합니다. 공유 캐시는 `<artifacts_root>/shared_cache/{embeddings,recommendations}/`에 저장하며 다른 Run 지정이 필요 없습니다. Scene·Summary를 다른 Run에서 자동으로 가져오지는 않습니다. `--force`는 선택한 단계의 로컬·공유 읽기를 우회하고 기존 공유 엔트리를 덮어쓰지 않습니다.

공통 데이터 키는 순서가 있는 item/content 매핑·전체 events·평가 구간입니다. Embedding 키는 해당 Arm의 실제 입력 텍스트·빈 위치, 인코더/토크나이저 식별자·설정, 표현 계약과 생성 provenance입니다. Summary 프롬프트와 기록된 Scene 프롬프트는 파일명이 아닌 본문 해시로 구분합니다. Summary의 생성 provenance(프롬프트·모델·설정)와 `scene_input_hash`가 있으면 Scene 생성 provenance가 없는 간결한 형식도 Run 간 재사용할 수 있습니다. 입력 해시가 없는 과거 Summary는 검증 가능한 Scene 생성 provenance가 있어야 공유하며, 둘 다 없으면 Run 내부에서만 사용합니다. 과거 생성 결과에 현재 프롬프트나 설정을 소급하지 않습니다. Metadata 제목·인코더가 같으면 시각 Arm의 변경과 무관하게 embedding을 재사용합니다.

추천 키는 공통 데이터·embedding 키와 실제 행렬 해시·날짜·seed·학습 설정·모델/학습/평가 계약입니다. 재사용 단위는 날짜 × seed × Arm의 완료 결과이며 epoch 중간 재개는 없습니다. 데이터나 학습 조건이 바뀌면 해당 추천 결과를 다시 계산하고, `graph_gemini` Summary만 바뀌면 `graph_gemini_meta`만 무효화합니다. 제목 변경은 `meta`와 결합 Arm에, Qwen Summary 변경은 해당 단독·결합 Arm에만 영향을 줍니다. 결합 정책과 실제 입력을 캐시 키에 포함하며 과거 제목 조건부 Summary의 캐시와 분리합니다. checkpoint·학습 기록·이벤트별 지표·완료 manifest를 체크섬과 이벤트 대응까지 검증하고 독립 파일로 복사합니다. 현재 Run ID로 metadata를 갱신하며 `reused_from`으로 원본 출처를 남깁니다. 키별 파일 잠금과 임시 디렉터리의 원자적 게시로 공유 엔트리를 보호합니다. 모든 결과가 재사용되면 BGE/GPU 학습 초기화를 하지 않습니다.

기존 Run에서 공유 불가로 기록된 정상 결과를 공유하려면 원본 Run에서 `embed-representations`와 `run-recommendation`을 순서대로 다시 실행하세요(`--target desc_qwen_meta`로 제한 가능). 입력과 학습 조건이 그대로면 기존 임베딩·완료 학습 결과를 재계산하지 않고 공유 캐시에 등록합니다. 새 Run에는 Summary를 직접 준비한 뒤 같은 명령을 실행합니다. `--force`는 사용하지 않습니다.

새 계약은 `full-catalog-zero-vector/v2`, `shared-scenes-representation/v3`, `sasrec-rolling-combination/v3`입니다. 기존 교집합 실험은 읽기 전용 진단을 지원하지만 새 공유 캐시로 소급 승격하지 않습니다. 기존 실험을 보존하려면 새 Run에서 첫 전체 데이터 결과를 만든 뒤 후속 Run에서 재사용하세요. Arm별 정상·실패·누락·영벡터 수와 embedding 재사용 출처는 `.inputs/{arm}.json` 및 진단에, 추천의 최근 실행 재사용 건수는 `recommendations/reuse.json`에 기록합니다.

`diagnosis.json`의 `rolling-diagnosis/v3`는 아이템별 출처 전체 대신 arm별 해시·truncation·출처 분포·단어 수 요약을 저장합니다. 예외와 Gemini fallback 예시는 각각 최대 10개이며 전체 건수와 생략 건수를 함께 기록합니다. 출처 분포도 빈도순 최대 10개 값과 생략된 레코드 수를 기록합니다. `details_path`는 run 디렉터리 기준 상대경로이며, 상세 기록은 `validation/representations/.inputs/{arm}.json`의 `sources`에 보존됩니다. 결과를 옮길 때 상세 추적이 필요하면 이 숨김 디렉터리도 함께 복사하세요. v2의 `representations.*.sources`는 `sources_summary`와 `details_path`로, `gemini_summary_fallbacks.*` 배열은 `count`·`examples`·`omitted_count`·`details_path` 객체로 변경되었습니다.

진단은 저장된 `sasrec-content-v2`와 `sasrec-content-v3` 추천 결과를 지원하며, 실제 버전을 `recommendations.architecture_version`에 기록합니다. 선택한 날짜·seed·arm에 서로 다른 버전이 섞이면 집계하지 않습니다. 추천 학습 재개는 현재 코드의 모델 버전만 재사용하므로, 과거 버전의 결과를 진단할 때는 `run-diagnosis`만 실행하면 됩니다.

## Graph 프롬프트 비교: Run 분리

비교할 프롬프트마다 별도의 Run을 사용하고 동일한 cohort·catalog·평가 날짜·학습 설정·seed를 유지합니다. 각 Run에서 장면 추출 → 요약 → embedding → 추천을 완료한 뒤 비교합니다.

```bash
python -m validation run-diagnosis --run-id "$RUN_ID" --compare-run-id reference_run --target graph_qwen graph_qwen_meta graph_gemini_meta
```

현재 Run에서 reference Run을 뺀 NDCG@10 차이를 `validation/diagnosis/diagnosis.json`의 `run_comparison`에 기록합니다. Run별 사건·catalog·평가 날짜·seed와 사건 수가 일치해야 하며 두 Run의 추천 캐시도 검증합니다. 새 Graph 3개 Arm의 Run 간 비교에는 Bonferroni 보정을 적용합니다. 프롬프트·모델·요약 정책·fallback의 차이가 함께 포함될 수 있으므로 출처를 확인합니다.

`--schema`는 프롬프트 선택이며 Python 출력 검증 계약을 바꾸지 않습니다. 다른 본문 구조의 과거 Graph나 Summary를 그대로 입력하는 ASIS 전용 경로·본문 자동 변환은 제공하지 않습니다. 기존 artifact를 수동 재사용하려면 현재 계약과 provenance를 충족해야 합니다.

## 기존 생성 결과 변환

아래 절차는 과거 v5 → v6 변환 전용이며 v4 결합 Arm으로 변환하지 않습니다. 기존 제목 포함 Summary는 v6의 `*_meta_*` Arm에 해당합니다. `config.yaml`을 v6 계약으로 설정한 뒤 다음 명령으로 명시적으로 복사합니다.

```bash
python -m extraction migrate-arm-layout --run-id "$RUN_ID" --summary-model qwen
```

기존 Scene 4종과 선택한 Summary 모델의 제목 포함 결과를 새 경로로 복사하며 원본은 보존합니다. 제목 사용을 저장된 provenance로 확인할 수 없으면 건너뛰고 사유를 보고합니다. 대상에 다른 파일이 있으면 충돌로 중단하며 동일한 입력으로 재실행할 수 있습니다. `extraction/arm-layout-migration.json`에 변환 출처를 기록합니다. 과거 원문 fallback은 복사본에서 빈 실패 표현으로 처리하고 원본 생성 provenance는 유지합니다.

이미 Summary를 새 `summaries/{arm}` 경로로 옮겼지만 내부 Arm이 과거 이름인 경우에는 다음 보정 명령을 사용합니다. 전체 사전 검증과 원본 백업 후 최상위 `arm`을 디렉터리에 맞추고 `provenance.arm`을 제거합니다. 과거 생성 조건은 그대로 두고 명시적인 변환 이력을 추가하며, 재실행은 변경이 없는 파일을 건너뜁니다.

```bash
python -m extraction.normalize_summary_arms --run-id "$RUN_ID"
```

제목 없는 Summary는 공유 Scene에서 새로 생성합니다. Validation 파일은 변환하지 않으며 embedding부터 다시 실행합니다. 기존 5개 Arm의 캐시는 새 공유 캐시에 자동 승격하지 않습니다. 기존 계약의 진단은 원래 Arm 이름과 경로로 읽습니다.

## Artifact 구조

```text
artifacts/
├── preparation/{cohort,resized_keyframes,source_assets}/
├── shared_cache/{embeddings,recommendations}/
└── runs/{RUN_ID}/
    ├── extraction/
    │   ├── scenes/{SCENE_ARM}/{content_id}.jsonl
    │   │   └── failures/{content_id}.jsonl
    │   ├── summaries/{SUMMARY_ARM}/{content_id}.json
    │   │   └── failures.jsonl
    │   └── arm-layout-migration.json
    └── validation/
        ├── cohort/
        ├── representations/{arm}_embeddings.npz
        ├── recommendations/{date}/seed_{seed}/{arm}/
        └── diagnosis/diagnosis.json
```

`scenes/{SCENE_ARM}/{content_id}.jsonl`은 장면당 한 줄입니다. Graph는 `content_id`, `scene_idx`, `tokens`, `warning`, `scene_graph` 순서로, Description은 `content_id`, `scene_idx`, `tokens`, `description` 순서로 저장합니다. Scene 본문에는 provenance를 저장하지 않습니다. Qwen·Gemini에 같은 형식을 적용합니다.

```json
{"content_id":"123","scene_idx":0,"tokens":25,"description":"A person walks outdoors."}
{"content_id":"123","scene_idx":0,"tokens":40,"warning":[],"scene_graph":{"entities":[],"relations":[],"context":[]}}
```

`tokens`는 저장된 결과를 만든 마지막 시도의 실제 출력 토큰 수입니다. Qwen은 `output_tokens`, Gemini는 `usage_metadata.candidates_token_count`를 사용하며 입력·생각 토큰과 재시도 누적량은 포함하지 않습니다. 계측값이 없으면 `null`로 저장하고 문자열 길이로 추정하지 않습니다. Summary에도 `content_id` 바로 뒤에 `tokens`를 저장하며, 실패 로그에는 측정된 경우 토큰 수를 보존합니다. Description에는 `warning`이 없습니다.

장면 번호 순서로 저장하며 `.metadata`는 만들지 않습니다. Graph의 구조화 결과는 객체, 파싱 실패 원문은 문자열로 `scene_graph`에 저장하며 타입으로 구분합니다. 별도 `graph_format`은 저장하지 않습니다. `warning`에는 정상 출력은 `[]`, raw 출력은 다섯 오류 태그 중 해당 항목을 기록합니다. 태그와 완화된 검증 규칙은 [Graph 규약](docs/graph_context_actions.md)을 참고하세요. 과거 실패 Raw Graph·Desc는 실패 로그를 유지하며 Summary 입력에서 제외합니다. 신규 생성 실패는 본문 대신 실패 로그에 기록합니다. Graph의 토큰 한도 종료는 잘린 응답 원문을 `raw_output`에 보존하고, 나머지 신규 실패는 빈 문자열을 기록합니다. 중단·비동기 완료로 장면이 빠져 있어도 각 행의 `scene_idx`로 정확히 재개합니다. Graph·Description 장면 provenance는 저장하지 않지만 Summary에 기록한 `scene_input_hash`와 Summary 생성 provenance로 임베딩·추천의 Run 간 공유 여부를 판단합니다. 실제 Summary 텍스트와 입력 해시·생성 설정이 같아야 캐시가 일치하며, 프롬프트만 같고 생성 결과가 다르면 재계산합니다. 같은 Run의 정상 결과 재사용은 유지합니다.

기존 전체 필드 JSONL과 두 필드 JSONL도 읽을 수 있습니다. 두 필드 파일은 원래 `.metadata`에서 장면 번호를 읽어야 하므로 먼저 삭제하지 마세요. 추출 재실행 또는 아래 명령으로 장면 번호를 포함한 형식으로 변환하며, 저장이 성공한 파일의 기존 메타데이터만 삭제합니다. 메타데이터가 없거나 본문과 맞지 않는 이전 파일은 장면 번호를 추측하거나 전체 재생성하지 않고 오류를 보고합니다. 별도 변환 명령은 해당 Run의 장면 추출을 중지한 상태에서 실행합니다.

```bash
python -m extraction migrate-scene-schema --run-id "$RUN_ID"
```

준비 산출물은 `<artifacts_root>/preparation/`의 `cohort`, `resized_keyframes`, `source_assets`에 저장하며 모든 Run이 공유합니다. Cohort의 준비 상태와 평가 계획은 run ID에 종속되지 않습니다. Preparation CLI의 `--run-id`는 기존 호출 호환을 위해 유지하지만 저장 위치를 나누지 않습니다. 최초 준비나 필요한 파일이 없는 경우 `prepare-input-data`를 실행합니다. 공유 cohort·timestamp·프레임이 준비되어 있으면 새 Run은 preparation을 다시 실행하지 않고 extraction·validation을 실행할 수 있습니다. 장면 추출은 `prepare-input-data`가 정상 완료됐다고 가정합니다. 이미지·asset 존재 검사, 폴더 스캔, duration·샘플링 재검증 및 이미지 내용 해시 계산을 하지 않습니다. timestamp JSON으로 scene 수와 재사용 여부를 확인하고, 추론 입력을 공급할 때 필요한 경로를 준비 단계의 PNG 파일명 규칙으로 구성합니다. 이미지는 실제 추론 시 읽으며, 필요한 파일이 없으면 해당 입력을 읽는 시점에 실패합니다. 자동 준비 호출은 하지 않습니다.

장면 추출은 추론 전에 timestamp와 저장 결과를 확인해 처리할 전체 scene 수를 계산합니다. 추론용 task와 이미지는 요청을 공급하면서 읽습니다. Gemini는 제한된 수의 scene을 미리 공급하고 빈 worker가 영상 경계 없이 다음 scene을 바로 처리합니다. 처리 속도와 ETA는 추론 단계의 완료 건수와 경과 시간을 기준으로 표시합니다.

기본 6개 keyframe 정책은 `timestamp_fixed_{scene_duration}s.json`, 다른 개수는 `timestamp_fixed_{scene_duration}s_{num_keyframes}kf.json`을 사용하므로 서로 다른 정책이 공유 timestamp를 덮어쓰지 않습니다. 콘텐츠별 잠금과 원자적 저장으로 동시 준비를 보호합니다. `--force`도 정상 이미지를 덮어쓰지 않습니다. `prepare-input-data`를 명시적으로 실행할 때는 공유 duration의 원본 식별 정보가 달라지면 재사용을 거부하므로 변경된 영상은 별도 `artifacts_root`로 분리합니다.

`prepare-input-data`의 `Prepare visual evidence` 진행률은 영상별 준비·검증 건수입니다. `reused_frames`는 재사용 이미지 수, `new_frames`는 실제 신규 추출 이미지 수입니다. 누락된 이미지를 추출할 때만 `[KEYFRAMES] extracting ...`과 누락 timestamp를 출력합니다. 공유 duration과 timestamp가 유효하면 영상 길이를 재확인하거나 timestamp를 다시 만들지 않습니다. 준비 실패 보고서는 공유 `preparation/cohort/preparation_failures.jsonl`에 저장합니다.

기존 산출물은 자동 이동·삭제하지 않습니다. 이전 데이터를 재사용하려면 사용할 cohort 하나를 `artifacts/preparation/cohort/`에, 기존 프레임과 source assets를 각각 `artifacts/preparation/resized_keyframes/`, `artifacts/preparation/source_assets/`에 배치합니다. 과거 cohort에 기록된 run ID는 로드 시 사용하지 않습니다. `prepare-cohort`를 재실행하면 공유 cohort가 갱신되므로 서로 다른 데이터셋·평가 구성을 유지하려면 `artifacts_root`를 분리합니다. `--plan-only` 실행 후에는 전체 `prepare-cohort`가 성공해야 다음 단계에서 사용할 수 있습니다.

Qwen Desc·Graph·Summary는 설정된 repetition penalty별로 전체 pass를 완료하고 실패 항목만 다음 pass에서 재생성합니다. 기본 순서는 `1.00 → 1.05 → 1.10 → 1.15 → 1.20`이며 성공 항목은 제외합니다. Vertex Gemini의 `429 RESOURCE_EXHAUSTED`만 30초 뒤 한 번 재시도하며, Summary 교정 및 `.recovery`, `.pending`, `.checkpoints`, 콘텐츠 진행 cursor를 저장하지 않습니다. 장면 번호는 본문 파일의 `scene_idx`에, 요약의 생성 당시 provenance는 요약 문서에 남깁니다.

장면 실패는 `scenes/failures/{content_id}.jsonl`에 `content_id`, `scene_idx`, `error`, `raw_output`을 저장합니다. Summary 생성 실패는 `summaries/{SUMMARY_ARM}/failures.jsonl`에 `content_id`, `error`, `raw_output`, `summary_model`과 Qwen의 `repetition_penalty`를 저장합니다. Graph 토큰 한도 종료의 `raw_output`에는 응답 원문을 공백·줄바꿈까지 그대로 저장합니다. 나머지 신규 실패는 빈 문자열을 사용하며 생성 당시 `provenance`를 함께 보존합니다. 장면 생성은 재실행 시 첫 penalty부터 실패 항목을 처리합니다. Qwen Summary는 아직 처리하지 않은 콘텐츠를 먼저 초기 penalty로 처리한 뒤, 실패별 마지막 penalty보다 큰 다음 설정값부터 이어갑니다. 마지막 penalty까지 완료했거나 penalty 필드가 없는 기존 실패는 재생성 없이 빈 Summary로 기록합니다. 실패 이유·재시도 정보·실제 생성 provenance를 보존합니다. Scene 입력이 복구되면 실패 Summary를 다시 시도하고, 성공 결과까지 다시 만들려면 `--force`를 사용합니다. 재시도 성공 후 해당 실패 행을 제거합니다. 기존 실패 로그를 읽을 때 과거 원문은 일괄 삭제하지 않습니다.

```json
{"content_id":"123","scene_idx":2,"error":"model produced an empty description","raw_output":""}
{"content_id":"123","error":"max_tokens","raw_output":"","repetition_penalty":1.2}
```

진행률의 `success`와 `failed`는 현재 패스에서 완료된 성공·실패 수이고, 신규 정책에서는 실패 원문을 표현으로 보존하지 않으므로 `raw`는 0입니다. Qwen 재시도는 실패 항목만 다음 패스로 넘기며, 매 패스 시작 시 분모를 실제 처리 대상 수로 바꾸고 완료 건수·경과 시간을 초기화합니다. `pass=N/M`은 설정된 penalty 순서를 나타내며, `scene/s`와 `summary/s` 및 ETA는 현재 패스 기준입니다. 재사용 결과는 집계에서 제외하고 전체 최종 결과는 단계 종료 로그에 표시합니다. Qwen·Gemini Graph와 Summary의 토큰 한도 종료는 실패로 기록합니다.

각 추천 조합의 `training.json`, `per_event_metrics.jsonl`, `complete.json`, **최종 `sasrec.pt`**를 보존합니다. 기존 run이나 수동 보관한 archive는 자동 삭제하지 않습니다.

## 생성 정책과 재사용

- 장면·요약 상한은 `extraction.{description,graph}.{qwen,gemini}`에서 각각 설정하며 현재 모두 1,024 tokens입니다.
- 화면 텍스트는 의미 해석의 단서로만 사용하며 문구 전사·인용·번역 출력은 금지하도록 지시합니다. 근거 있는 장르·목적·배경지식 해석을 허용하고 불확실성을 보존합니다.
- Qwen Graph·Description·Summary는 repetition penalty 목록 순서로 실패 항목만 재생성합니다. 마지막까지 실패하면 빈 표현으로 처리하고 실패 기록을 남깁니다. 성공한 항목은 재생성하지 않습니다.
- Graph의 구조화 필드는 `entities`, `relations`이며 과거 `context`도 허용합니다. `name`은 자유 어휘 **개체 종류**이고 실명이나 고유 신원이 아닙니다. 중복 `name`은 허용하며 고유한 장면 내부 `id`와 외형·상태·활동 `attributes`로 구분합니다. 현재 E2E 실행에서는 ID 중복·관계 참조 일치 여부를 검증하지 않으며, 생성된 ID와 관계를 그대로 저장합니다. 구조 검증에 맞지 않는 응답은 `scene_graph` 문자열로 정상 저장하며 별도 `graph_format` 필드는 쓰지 않습니다. 객체 추적기는 없으며 장면 사이 ID를 연결하지 않습니다.
- Graph 생성에는 JSON Schema를 강제하지 않습니다. `scene_graph_v3.md`의 줄 단위 출력을 두 모델의 공통 파서가 기존 JSON 구조로 변환합니다. 화살표·하이픈 구분자, 헤더, bullet 등 명확한 형식 변형은 Repair하고, 누락·모호한 행 등 형식 문제는 실패로 처리하지 않고 원문으로 저장합니다. `[End]` 생략을 허용하며 출력 잘림은 실제 backend의 토큰 한도 종료로만 판단합니다. 토큰 한도 종료와 API 오류는 실패로 기록합니다. 완전한 JSON 응답도 지원하며, 잘린 내용을 추측해서 채우지 않습니다. 상세 규칙은 [Qwen 실행 가이드](docs/qwen_vllm.md)에 있습니다.
- 신규 Summary는 문단·목록·마크업·구조화 필드를 허용하며 줄바꿈을 보존합니다. 단어 수는 기록만 하고 실패 조건으로 사용하지 않습니다. Qwen은 빈 출력과 `finish_reason=length`를 실패로 기록하고 다음 penalty에서 재생성합니다. 최종 실패 Summary는 빈 텍스트이며 사용할 성공 Scene이 없으면 `scene_count=0`을 허용합니다.
- Qwen Graph·Desc와 Gemini Graph는 설정·프롬프트·경로가 달라도 기존 성공 결과를 그대로 재사용합니다. Summary도 동일한 요약 모델(qwen/gemini) 안에서는 같은 정책을 따릅니다. 일반 실행은 결과가 없는 항목과 재시도가 남은 실패만 생성합니다. 장면을 갱신해도 이미 성공한 Summary는 보존하며, 성공 결과까지 다시 만들려면 해당 단계에 `--force`를 지정합니다. 실패 기록은 재생성 성공 후 제거합니다. 다른 run ID의 결과를 자동으로 가져오지는 않습니다.
- Validation은 Arm에 연결된 제목 없는 Summary를 읽습니다. 결합 Arm은 Summary 실패·누락 시 제목만, 제목 누락 시 Summary만 사용합니다. 둘 다 없으면 영벡터로 처리합니다. 손상된 파일이나 잘못된 출처는 오류입니다.
- BGE 입력 상한은 512 tokens이고 실제 truncation 건수를 기록합니다. 빈 Metadata 제목과 빈 시각 표현은 모두 영벡터입니다. SASRec 구조는 유지하며 영벡터 아이템 사이의 동일 점수와 기존 catalog 순서 기반 동점 처리를 유지합니다.


## 평가와 검증

기본 데이터는 사용자 100,000명·interaction 719,405건·아이템 19,738개입니다. 같은 사건·후보 catalog에서 날짜별 독립 selection → refit → test를 실행합니다. 6개 Arm은 **7일 × 3 seeds × 6 = 126개 조합**입니다. 부분 target은 선택한 Arm 수만큼 처리합니다.

NDCG@10을 주 지표로 사용자 단위 paired bootstrap 10,000회, seed 평균 후 날짜 균등 평균을 적용합니다. Meta 대비 5개, Qwen 표현 방식 2개, 제목 추가 효과 2개 비교군에 각각 α=0.05와 Bonferroni 양측 신뢰구간을 적용합니다. 부분 target에서도 보정 분모 5·2·2를 유지합니다. `graph_qwen_meta − meta`를 주 결과로, `graph_gemini_meta − graph_qwen_meta`를 teacher 참조 격차(탐색적 95% 구간)로 표시합니다. 일반 모델 우열 및 제거된 Arm이 필요한 교호작용 비교는 생성하지 않습니다.

진단은 제목+Summary·제목만·Summary만·둘 다 없음의 건수, Summary 성공 확보율, 실패·누락 상태와 BGE 잘림 건수를 기록합니다. 제목 fallback이 시각 정보 실패를 가리지 않으며, 실패나 빈 표현 때문에 평가 아이템을 제외하지 않습니다. 과거 v5/v6 Run 진단은 저장된 계약으로 해석하며 원본 산출물을 변경하지 않습니다.

```bash
python -m pytest -q
ruff check --config pyproject.toml src tests benchmarks
```

v5/v6 호환 회귀 테스트와 v4 입력·캐시·진단 테스트, 작은 CPU 추천 학습을 사용합니다. 실제 모델/API 호출과 전체 데이터 Run은 별도로 실행합니다.
