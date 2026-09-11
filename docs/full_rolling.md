# MicroLens 전체 데이터 rolling 실행 계약

기본 `config/pipeline.yaml`은 v4입니다. 원본 CSV 100,000명·719,405개 사건·19,738개 아이템을 모두 보존하며, 7일 × 4 Arm × seed 42/43/44의 84개 조합을 실행합니다. 작은 fixture는 구현 검증용이며 별도 연구 Pilot을 만들지 않습니다.

## 데이터와 시간

`data.pairs_csv`는 `user,item,timestamp` 헤더를 가진 정수 밀리초 CSV입니다. `event_id`는 헤더 다음 첫 행을 0으로 하는 원본 행 인덱스입니다. 중복 사건도 고유 event_id로 남으며 중복 수를 보고합니다. 결측·잘못된 값·cardinality 불일치는 실패합니다. `data.pairs_tsv` 파일이 있으면 사용자별 interaction 정합성을 확인합니다.

정답 시각보다 **엄격히 작은 timestamp**만 이력에 들어갑니다. 같은 시각 사건에는 선후를 부여하지 않습니다. 이전 시각의 동시 사건은 원본 행 순서로 정렬합니다. 이력 없는 사건은 정답에서만 제외되고 원본과 이후 이력에는 남습니다. 문맥은 최근 10개지만 학습 기간의 모든 적격 전이는 epoch당 한 번 학습합니다. Batch 256은 전이 256개입니다.

| 평가일 d의 단계 | 시작(포함) | 종료(미포함) |
|---|---|---|
| epoch 선택 학습 | 원본 시작 | d−1 UTC 00:00 |
| validation | d−1 UTC 00:00 | d UTC 00:00 |
| refit | 원본 시작 | d UTC 00:00 |
| test | d UTC 00:00 | d+1 UTC 00:00 |

최종 관측일은 제외하고 직전 7개 달력 날짜를 평가합니다. 공식 CSV는 2022-09-12에 끝나므로 test 날짜는 2022-09-05~11입니다. 전날 validation으로 epoch를 정한 다음 **같은 seed로 새 모델**을 초기화하여 refit합니다. 날짜 간 checkpoint 학습을 이어가지 않습니다. Validation/test 파라미터는 고정되고 같은 날 앞선 사건은 문맥에 반영됩니다.

후보군은 매일 19,738개입니다. 평가에서는 모든 이전 관측 아이템을 마스킹하되 정답은 보존합니다. 학습의 in-batch negative 마스킹은 기존처럼 입력 문맥의 아이템과 중복 정답에 적용합니다. 점수 동률은 numeric item ID로 정렬한 고정 행 인덱스로 해결합니다. 인기도 보정은 selection/refit의 해당 positive 예제 분포를 사용합니다. 진단의 아이템 빈도는 refit 경계 이전 원본 사건에서 계산합니다.

## Keyframe 준비와 재사용

`protocol.sampling: fixed_windows`는 `scene_duration / num_keyframes` 길이의 구간별 중앙점을 추출합니다. 기본 30초·6장은 Scene 시작 기준 `[2.5, 7.5, 12.5, 17.5, 22.5, 27.5]`초입니다. 마지막 짧은 Scene도 같은 간격을 유지하고 마지막 구간만 영상 끝에서 잘라 중앙점을 구합니다. 12초가 남으면 `[2.5, 7.5, 11]`초를 사용하며 영상 길이를 정수 초로 올리지 않습니다.

추출 시각은 영상 전체 기준이며 소수점 둘째 자리부터 버립니다(`1.666… → 1.6`). JSON과 추출 요청에 같은 시각을 사용합니다. `scene_duration`과 `num_keyframes`는 양의 정수이고 `num_keyframes <= scene_duration × 10`이어야 합니다. 마지막 짧은 구간에서 버림 후 같은 시각이 생기면 한 번만 추출합니다.

아래 경로는 모두 `artifacts/{run_id}/` 기준입니다.

| 입력 산출물 | 경로 |
| --- | --- |
| 이미지 | `data/resized_keyframes/{content_id}/0002_5.png`. 정수 초는 `0005.png` |
| Scene timestamp | `data/cohort/source_assets/{content_id}/assets/timestamp_fixed_{scene_duration}s.json` |
| 영상 길이 cache | `data/cohort/source_assets/{content_id}/assets/video_duration.json` |

이전 이미지 배치를 옮길 때는 실행을 멈춘 뒤 `data/fixed_30s/resized_keyframes/`를 `data/resized_keyframes/`로 이동합니다(새 경로가 없을 때). 다른 Scene 길이는 해당 `fixed_*s` 폴더를 사용합니다. Timestamp JSON은 그대로 둡니다. 이전 경로를 자동 탐색·이동하지 않으며 서로 다른 sampling 결과를 한 디렉터리에 합치지 않습니다.

v4의 `prepare-cohort`는 영상 파일 존재·크기·중복과 title을 검사하고 길이는 조회하지 않습니다. `prepare-input-data`의 4개 worker가 영상별로 ffprobe → Scene 계산 → keyframe 추출을 수행하며 진행률을 표시합니다. 길이는 원본 경로·크기·수정 시각과 함께 저장하며, 같은 원본이면 재실행에서 재사용합니다. 원본 영상을 교체했다면 `prepare-cohort`로 파일 정보를 갱신합니다.

새 cohort의 catalog/inventory에서 `duration_seconds`는 `null`일 수 있습니다. 기존 run에 숫자 길이가 있으면 사용할 수 있으며 이후 조회한 길이는 별도 duration 파일에 기록합니다. 실패는 `preparation_failures.jsonl`에서 확인하고 같은 명령으로 재개합니다. 전체 `data/cohort/media_preflight.json`은 입력 영상 추출·검증이 모두 끝난 뒤 작성하고, cohort 또는 입력 준비 재실행 시 이전 통계는 제거합니다.

## 실행과 환경 간 전달

[README](../README.md)의 Ubuntu/Bash 단계별 명령을 사용합니다. 추천은 옵션을 생략하면 visible GPU 중 첫 장을 사용하며 조합을 순차 실행합니다. CUDA가 없으면 CPU를 사용합니다.

### 추천 조합 병렬 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m validation run-recommendation \
  --run-id "$RUN_ID" --gpus 4 --workers-per-gpu 2
```

- `--gpus`는 visible CUDA 장치 개수입니다. `CUDA_VISIBLE_DEVICES=2,3`과 `--gpus 2`를 쓰면 물리 GPU 2·3에 배정되며 로그에는 논리 장치 `cuda:0`·`cuda:1`로 표시됩니다. 요청한 개수보다 visible GPU가 적으면 실패합니다.
- `--workers-per-gpu` 기본값은 1입니다. 위 예는 GPU당 2개, 총 8개 독립 프로세스를 실행합니다. 각 프로세스가 한 조합의 selection → refit → test를 끝내고 공유 대기열에서 다음 조합을 가져갑니다. 모델을 GPU 간 분할하지 않습니다.
- 프로세스는 `spawn`으로 시작하며 GPU를 지정하고 조합별 seed를 재설정합니다. CPU thread 과다 경합을 줄이기 위해 worker의 PyTorch 연산 thread는 1개입니다. 원본 사건 테이블은 worker마다 한 번 로딩하므로 worker 수에 비례해 호스트 RAM도 사용합니다.
- Batch 256, 학습 순서, loss·마스킹, epoch 선택, refit 초기화 및 평가 조건은 유지합니다. `selection`, `refit epochs=...`, `test` 로그에 날짜·seed·Arm·device를 표시합니다. 전체 진행률은 완료·재사용 조합 수로 갱신됩니다.
- 병렬 worker의 로그·경고는 부모가 모아서 stdout으로 출력하고, 로그 아래에 전체 progress bar를 다시 표시합니다. 아직 완료 조합이 없어도 터미널에서는 약 1초마다 경과 시간을 갱신합니다. 파일로 출력하면 로그 발생 시와 약 30초 간격으로 표시하며, 완료 수는 조합의 test·저장이 끝난 뒤 증가합니다.
- 기존 추천 프로세스를 중단하고 종료를 확인한 다음 같은 run ID로 위 명령을 실행합니다. 완료 조합은 검증 후 재사용하고 미완료 조합만 처음부터 재학습합니다. epoch 중간부터 재개하지 않습니다. 같은 run ID의 추천 명령을 여러 개 동시에 실행하지 않습니다.
- worker 오류나 Ctrl-C가 발생하면 부모가 worker들을 종료·회수합니다. 이미 저장된 정상 완료 조합은 다음 실행에서 재사용됩니다. `--force`는 선택한 Arm의 모든 날짜·seed 조합을 재생성합니다.
- GPU당 2개는 초기 실행 예이며 최적값이나 배속을 보장하지 않습니다. CPU·메모리 대역폭 경합이 생길 수 있으므로 GPU당 1개와 2개의 조합 완료 처리량으로 비교합니다. CUDA 실제 속도와 수치 재현성은 Ubuntu 장비에서 검증해야 합니다.

### 환경 간 전달

1. **Ubuntu GPU 환경:** 전체 CSV 검사·title 보완·cohort·영상 준비, Qwen Graph와 Description 추출·요약.
2. **Ubuntu Gemini API 환경:** 같은 run의 `experiment.json`, `data/cohort/`, `data/resized_keyframes/`를 전달하고 같은 코드 revision을 사용합니다. `conda activate vc_cloud` 후 README의 Gemini 명령을 실행합니다. 원본 MP4·로컬 모델·GPU는 사용하지 않습니다.
3. **Ubuntu GPU 환경:** Gemini 환경의 `extraction/graph/gemini/`를 같은 run으로 전달합니다. Gemini Graph 요약 → BGE → 추천 → 진단을 실행합니다.

두 역할은 같은 장비에서 실행해도 됩니다. 별도 장비라면 `artifacts_root`를 각 호스트에 맞추고 run ID·sampling·prompt·생성 조건을 유지합니다. Gemini 추출은 run 내부 경로에서 timestamp와 이미지를 찾으며, cohort에 남은 원본 영상 절대경로는 출처 기록으로 보존합니다. Gemini VM에서 cohort를 다시 만들 필요는 없습니다.

전달할 `data/cohort/`에는 `eligibility.json`, `cohort_plan.json`, `catalog.jsonl`, `item_inventory.jsonl`, `required_items.jsonl`, `metadata_titles.jsonl`, `events.jsonl`과 `source_assets/`가 포함됩니다. 실행 재개 시에는 해당 환경의 기존 Gemini 산출물과 남아 있는 `.recovery`도 보존합니다. 양쪽에서 동일 파일을 동시에 쓰지 않습니다. `artifacts/`는 Git에서 제외되므로 별도 전달이 필요합니다.

현재 추출 CLI는 전체 catalog의 시각 입력을 검사하며 샘플 선택 옵션은 없습니다. 일부 keyframe만 전달한 상태에서 전체 명령을 실행하면 누락 입력 오류로 중단됩니다. 샘플 검증 결과는 전체 추출 완료로 간주하지 않습니다. v3 leave-two-out run과 v4 full rolling run, 서로 다른 sampling 조건의 산출물을 섞지 않습니다.

Gemini summary의 대체·재시도 규칙은 아래 [생성 검증과 복구](#생성-검증과-복구)를 따릅니다.

### 모드 선택

`run-recommendation`과 후속 `run-diagnosis`는 `--target`으로 실행할 모드를 선택합니다. `METADATA`, `GRAPH_QWEN`, `GRAPH_GEMINI`, `DESC_QWEN` 중 원하는 값만 공백으로 구분해 나열합니다. 대괄호나 쉼표는 넣지 않습니다. `DESC_QWEN`의 기존 저장 이름은 `SASRec_DESC` / `desc`입니다.

```bash
# 예: Graph Qwen, Description Qwen, Metadata만 실행 (7일 × 3개 모드 × 3 seeds = 63조합)
python -m validation run-recommendation --run-id "$RUN_ID" --gpus 4 --workers-per-gpu 3 --target GRAPH_QWEN DESC_QWEN METADATA
python -m validation run-diagnosis --run-id "$RUN_ID" --target GRAPH_QWEN DESC_QWEN METADATA

# 예: Metadata 없이 두 Graph 모드만 선택
python -m validation run-recommendation --run-id "$RUN_ID" --target GRAPH_QWEN GRAPH_GEMINI
python -m validation run-diagnosis --run-id "$RUN_ID" --target GRAPH_QWEN GRAPH_GEMINI
```

- v3와 v4 모두 적용됩니다. `--target`을 생략하면 전체 4개 모드를 대상으로 합니다. 앞 명령의 선택을 자동으로 상속하지 않으므로 진단 명령에도 지정합니다.
- 준비된 공통 cohort와 item index, 선택한 모드의 임베딩이 필요합니다. 제외한 모드의 임베딩·추천 checkpoint·장면 결과는 요구하지 않습니다. 공통 cohort의 title 목록은 그대로 보존합니다. 선택한 Gemini 임베딩이 Qwen summary를 대체 입력으로 사용했다면 그 Qwen summary 의존성은 계속 검증합니다. `embed-representations`의 실행 범위는 기존과 같습니다.
- 제외한 모드의 기존 추천 결과는 보존합니다. 같은 run에서 대상을 추가하면 정상 완료 결과를 재사용하고 추가·미완료 대상을 학습합니다. `--force`도 선택한 모드에만 적용됩니다.
- 진단의 기대 조합·결과 행 수와 장면 coverage 검사는 선택 범위에 맞춥니다. Metadata만 선택하면 장면 검사는 `not_applicable`입니다. 한 모드만 선택하면 해당 모드의 지표를 보고하고 모드 간 비교는 빈 목록으로 남깁니다.
- 비교 양쪽 모드가 모두 선택된 경우에만 기존 비교를 수행합니다. Metadata 우월성의 보정 분모 3과 Graph–Description 비열등성의 보정 분모 2는 그대로 유지합니다. 정책의 `evaluated_comparisons`와 `skipped_comparisons`에 수행·제외한 비교를 구분합니다. Metadata가 제외되면 논문 Metadata HR@10과의 차이도 계산하지 않습니다.
- 결과는 기존 `diagnosis.json`에 저장되며 `target_sources`, `selected_arms`, `excluded_arms`로 범위를 표시합니다. 부분 선택의 `pass`는 그 선택 범위에 대한 검증 통과이며, 아래의 전체 84조합 실험 완료를 뜻하지 않습니다.

## 생성 검증과 복구

생성 제약은 단계별로 다릅니다. Qwen의 구조 제약은 vLLM `xgrammar`를 사용하며, 모든 penalty 재시도에도 유지합니다.

| Step | 생성 제약과 최종 처리 |
| --- | --- |
| Qwen Graph 장면 | 필수 필드·타입·enum·nullable JSON Schema → Graph Repair → penalty 재시도 → Raw |
| 모든 Graph/Description Summary | 7개 라벨 순서의 Grammar → Summary Repair → penalty 재시도 → Raw. Gemini Graph의 요약도 Qwen이 실행 |
| Description 장면 | 자유 텍스트. 빈 응답만 penalty 재시도하고 전부 비면 실패 |
| Gemini Graph 장면 | SDK HTTP retry·Graph Repair. 오류 없이 받은 비어 있지 않은 응답만 최종 Raw 허용 |

`graph_repetition_penalty`, `description_repetition_penalty`, `summary_repetition_penalty`는 단일 숫자 또는 유한한 `[1, 2]` 범위의 비어 있지 않은 목록을 받습니다. 기본 `[1.00, 1.05, 1.10, 1.15, 1.20]`은 최초 1회와 재시도 최대 4회입니다. 같은 Qwen pool에서 실패 항목만 다음 penalty로 제출하며 입력·프롬프트·토큰 한도는 유지합니다. 장면은 전체 대기 Scene을 현재 penalty로 처리한 뒤 실패 Scene을 재시도하고, 요약은 묶음별로 재시도합니다.

Summary Repair는 전체 코드 펜스 제거, canonical/display 라벨 표기 정규화, 7개 라벨이 모두 있고 중복이 없을 때의 순서 복구만 합니다. 필드·내용을 만들어 채우지 않습니다. Graph 개수·ID 참조는 의미 경고로 남기며, Summary 문장 품질·단어 수를 추가 실패 기준으로 삼지 않습니다.

최종 Raw는 현재 재시도 묶음의 마지막 비어 있지 않은 **Repair 전 원문**입니다. 같은 출력 경로에 Graph는 `graph-scene-raw/v1`과 `raw_response`, Summary는 `video-summary-raw/v1`과 `text`를 저장하며 `status=raw_fallback`으로 구분합니다. 가짜 `graph`·`sections`는 만들지 않습니다. Graph 요약에는 정상·Raw 장면을 시간순으로 전달하고 Raw를 관찰 원문으로 표시합니다. `scene_count`는 두 종류를 합한 실제 입력 수입니다. 사용 가능한 장면이 없는 영상은 요약하지 않습니다.

### Gemini Summary 대체

`embed-representations`는 정상 Summary 또는 명시적으로 저장된 Raw Summary의 `text`를 읽습니다. Gemini Graph Summary **파일이 없을 때만** 같은 영상의 Qwen Graph Summary를 사용합니다. 정상·Raw Gemini 파일이 있으면 우선 사용하고, 잘못된 기존 파일을 Raw나 누락으로 간주하지 않습니다. BGE 로딩 전에 대체 항목을 출력하고 `validation/representations/graph_gemini_fallbacks.json`에 목록을 저장합니다. Qwen Graph·Description Summary 누락에는 대체 정책이 없습니다.

v4에서 기존 파일이 없는 Gemini 소스 Summary가 모든 생성 시도에서 비어 있으면 실패 기록과 `[SUMMARY FALLBACK]`을 남기고 다음 단계로 진행할 수 있습니다. 기존 파일을 갱신하다 실패한 경우나 엔진·파일 저장 오류에는 이 예외를 적용하지 않습니다. 대체 후에도 Gemini 원본 장면 coverage는 별도로 검사합니다.

Gemini 빈 응답은 콘솔과 `extraction/graph/gemini/scenes/failures/{content_id}.jsonl`의 `response_diagnostics`에서 확인합니다. `candidates[].finish_reason`, `finish_message`, `prompt_feedback`, `usage_metadata`가 원인 판단의 근거입니다. `MAX_TOKENS`는 출력 한도, `SAFETY`나 `prompt_feedback.block_reason`은 차단 여부를 확인하는 단서이며, 빈 텍스트만으로 차단으로 분류하지 않습니다. 과거 실패 기록에는 진단 필드가 없을 수 있습니다.

### 재시도와 복구 이력

정상·Raw 산출물은 재사용합니다. 중단된 재시도 묶음은 저장된 다음 시도부터 이어가고, 저장된 응답은 재호출 없이 게시합니다. 응답 저장 전 중단된 요청은 다시 호출될 수 있습니다. 최종 실패 항목은 다음 명령 실행에서 첫 penalty부터 새 묶음을 시작합니다. Qwen의 OOM·worker 종료·설정/문법 컴파일·저장 오류는 penalty 재시도나 Raw 성공으로 바꾸지 않습니다.

출력 디렉터리의 `.recovery`에 입력·생성 정책 식별값과 시도 이력을 저장합니다. 각 장면 추출·요약 Step이 실패 없이 완료되면 이 폴더를 삭제하고, 예외·중단·부분 실패(허용된 Gemini Summary 누락 포함)에서는 보존합니다. 정상·Raw 산출물과 요약의 `.inputs`는 삭제하지 않습니다. `--force`는 지정 Step의 기존 완료 결과까지 새로 생성하며, 중단 후에는 같은 명령으로 재개합니다. 세부 실행 이력과 진행률은 [Qwen 실행 가이드](qwen_vllm.md)에 있습니다.

`diagnosis.json`의 `generation_recovery`는 남아 있는 이력으로 정상 생성·Repair·Raw·최종 실패·중단을 집계합니다. `artifact_counts.legacy_unknown`은 대응 복구 이력이 없는 정상 결과이며, 정상 완료 후 `.recovery`가 삭제된 새 결과도 포함합니다. 따라서 이 값만으로 과거 실행 여부나 Repair 비율을 판단할 수 없습니다. Raw 상태는 정식 산출물로 확인하고, `summary_inputs`의 정상·Raw 입력 장면 수는 `.inputs`에서 읽습니다. Raw Graph는 정상 장면 coverage에 포함하지 않습니다.

## 저장과 재개

### 공백 Metadata의 영벡터 정책

공식 title 보완에는 `--unresolved-policy zero-vector`를 사용합니다. 보완 가능한 항목은 정상 title로 채우고, 여전히 없는 항목은 CSV에 빈 필드로 유지합니다. 완료 보고서 v2의 `unresolved_item_ids`에 목록을 남깁니다. 옵션이 없는 독립 title 보완 도구는 기존처럼 미해결 항목을 오류로 처리합니다.

v4 설정의 `validation.cohort.metadata_missing_policy: zero_vector`는 원본 title이 공백인 아이템을 catalog와 모든 interaction에 유지합니다. 정상 title만 BGE로 인코딩하고 빈 title 행은 1024차원 영벡터로 남깁니다. 다른 Arm에 영벡터를 전파하지 않습니다. `data/cohort/metadata_missing.json`에 정책·개수·아이템 ID·임베딩 행 인덱스를 기록하며, 추천 재사용 검증 및 diagnosis에서 실제 영벡터인지 검사합니다. CSV 파일 자체나 필수 아이템 행이 누락된 경우는 입력 오류로 구분합니다.

영벡터는 SASRec 입력 feature에만 적용합니다. 실제 아이템 ID와 위치 정보는 유지하며, projection bias와 후속 네트워크 때문에 최종 아이템 표현까지 영벡터로 고정되는 것은 아닙니다. 이전 정책에서 전환하면 title 보완·cohort 준비 후 임베딩과 추천을 재생성합니다. 기존 산출물은 자동 변환하거나 삭제하지 않습니다.

### 산출물과 후속 단계 갱신

- `experiment.json`: 최초 설정 snapshot. 재실행 시 비교하거나 덮어쓰지 않습니다.
- `data/cohort/events.jsonl`: 전체 사건 한 벌. `cohort_plan.json`에 7개 split 경계·원본/적격/무이력 수·범위 밖 사건 수를 저장합니다.
- `media_preflight.json`: 영상 길이 합계·원본 byte·scene·keyframe 수·RGB payload 추정. PNG 압축률 및 모델 출력 크기는 별도이며 정확한 총 디스크 사용량 보장은 아닙니다.
- `validation/recommendations/{date}/seed_{seed}/{arm}/`: `sasrec.pt`, `training.json`, `per_event_metrics.jsonl`, `complete.json`.

추천은 조합마다 결과를 임시 파일에 순차 기록하고 완성 후 rename합니다. 마지막에 `complete.json`을 확정합니다. 재실행하면 완료 기록, 비어 있지 않은 필수 파일, 학습 기록, 결과 행의 식별자·순위·사건 수·중복 여부를 검사하여 유효한 조합을 유지합니다. 이 검사는 checkpoint 내부 손상이나 형식을 유지한 값 변경을 모두 검출하지는 않습니다. 불완전한 조합만 처음부터 실행하며 `--force`는 선택한 Arm의 모든 조합을 다시 실행합니다. epoch 중간 checkpoint 재개는 없습니다.

장면 내용이 바뀌면 해당 영상의 요약을 갱신 대상으로 표시하며, 장면 수가 같아도 입력 hash로 변경을 확인합니다. 요약을 다시 만든 뒤 `embed-representations`를 실행하면 텍스트·catalog·encoder 설정의 hash와 변경 표식으로 영향받은 Arm을 갱신합니다. 실제 임베딩 배열이 달라지면 추천의 `embedding_hash`도 바뀌어 해당 Arm의 날짜·seed 조합을 다시 실행합니다. 후속 명령은 자동 실행되지 않으므로 **요약 → 임베딩 → 추천 → 진단** 순서로 실행합니다.

이 추적은 run 전체의 설정·파일을 검증하는 전역 fingerprint와 다릅니다. 재개 중 입력·생성 정책 식별값이 달라지면 새 재시도 묶음을 시작하지만, 그 사실만으로 기존 정상 결과를 모두 무효화하지 않습니다. 모델 파일·prompt·코드를 바꾸어 실험 조건을 비교하려면 새 run ID를 사용합니다. 같은 run에서 의도적으로 변경한다면 해당 생성 Step에 `--force`를 적용하고 후속 단계를 실행합니다. 원본 수·시간 경계·영벡터·임베딩 행 매핑·유한 값·결과 완전성 검사는 계속 적용합니다.

## 지표·통계와 완료 판정

날짜·seed·Arm별 **적격 사건 평균**, seed 평균, 7개 날짜 균등 평균 순서로 집계합니다. 모든 100k 사용자가 매일 평가 사건을 가져야 하는 것은 아닙니다. 결과 행 수는 `eligible_test_count × 선택한 Arm 수 × seed 수`이며, 기본 전체 실행은 `eligible_test_count × 12`입니다.

10,000회 paired bootstrap은 원본 사용자를 복원 추출하고 동일 사용자의 모든 날짜·사건 및 모든 Arm에 같은 가중치를 사용합니다. 사용자별 seed 평균 사건 합계와 날짜별 사건 수를 유지하여 날짜 평균의 분모도 다시 계산합니다. 작업 배열은 128MiB를 넘지 않도록 배치를 조절합니다. 빈 날짜/재표집 분모나 0인 상대효과 분모를 버리고 계속하지 않으며 실패로 보고합니다.

주 지표는 NDCG@10입니다. Metadata 우월성 3개 비교는 Bonferroni 보정 양측 CI 하한 > 0, Description 비열등성 2개 비교는 보정 단측 상대 CI 하한 > −0.05로 판정합니다. Gemini–Qwen 비교는 기존 탐색적 양측 상대 CI입니다. HR/NDCG@4·8·10·20·30도 보고합니다.

완료는 **전체 원본 보존 + 동일 조건 + 84개 정상 조합 + 유효한 진단 통계**입니다. `runtime_decision.status=pass`는 실험 실행·진단의 유효성을 뜻하며 가설 채택과는 다릅니다. 실제 우월/비열등 판정은 `statistics.comparisons`를 읽습니다. 논문 Metadata HR@10 약 4.6% 차이는 별도 참고값이며 논문의 미명시 세부 설정까지 동일하다고 주장하지 않습니다.

단위·fixture CPU 검증은 실제 MP4·Qwen·BGE·SASRec과 Gemini API 실행을 대체하지 않습니다.

## 2026-09-08 Windows 검증 기록

`llmjg` Python 3.11 / PyTorch 2.14 CPU에서 전체 테스트 **408개가 통과**했습니다. 작은 fixture의 축소 SASRec으로 84개 조합의 실제 학습·평가·진단을 실행했습니다. Test 기록 중 강제 중단, 재실행, 손상된 한 조합만 재학습, 결과 중복 검출을 검증했습니다. BGE와 영상/API 출력은 fixture이므로 모델 품질 또는 실제 영상 추출 성공을 입증하지 않습니다.

공식 CSV SHA-256은 `cc5972c79e8a24c8c7c8e229ce48b0d56c3719f804debb84df7382293c56bd4e`입니다. 실제 `prepare-cohort --plan-only` 경로로 100,000명·719,405개 사건·19,738개 아이템을 보존하여 저장했습니다. 중복 원본 행은 0개이며 전체 무이력 사건은 100,000개입니다.

| 평가일 | 원본 사건 | 적격 사건 | 이전 이력 없음 |
|---|---:|---:|---:|
| 2022-09-05 | 6,264 | 6,220 | 44 |
| 2022-09-06 | 6,485 | 6,452 | 33 |
| 2022-09-07 | 6,601 | 6,579 | 22 |
| 2022-09-08 | 7,176 | 7,160 | 16 |
| 2022-09-09 | 11,483 | 11,460 | 23 |
| 2022-09-10 | 11,262 | 11,246 | 16 |
| 2022-09-11 | 8,260 | 8,257 | 3 |
| 합계 | 57,531 | **57,374** | 157 |

따라서 본 실험의 예상 지표 행 수는 **688,488개**입니다. 입력 검증 산출물은 해당 VM의 `artifacts/full_input_audit/source_audit.json` 및 `artifacts/full_rolling_inputcheck_260908/data/cohort/`에 있습니다. 입력검증 run은 개발 중 snapshot이며, 본 실험은 새 run ID로 시작합니다.

100k 사용자·7일·4 Arm 합성 배열에서 10,000회 bootstrap을 실행했습니다. 자동 배치는 60회였으며 입력 배열을 포함한 추적 peak는 125,777,165 bytes로 128MiB 미만이었습니다. 이 수치는 bootstrap 배열 검증이며 전체 Python 프로세스 RSS 상한은 아닙니다. `artifacts/full_input_audit/bootstrap_scaling.json`에 기록했습니다.

당시 Windows VM에는 설정된 원본 영상·Qwen/BGE checkpoint와 CUDA가 없어 전체 미디어 준비, Ubuntu 학습, Gemini API 및 실제 84개 조합의 통계 결과는 실행하지 않았습니다. 위 fixture 결과로 본 실험의 완료나 연구 가설 성립을 주장하지 않습니다.
