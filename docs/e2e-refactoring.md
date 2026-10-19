# 주요 Flow를 보존하는 E2E 리팩토링 결과

기준 commit `e9aff2f`에서 시작하여 `codex/e2e-flow-preserving-refactor`에 구현했다. 사용자·catalog 확정 → keyframe 준비 → Graph Qwen / Graph Gemini / Description 추출·요약 → BGE → SASRec selection·refit → 평가·진단 순서를 유지한다.

최종 Windows 검사: **395 passed, 0 failed, 0 skipped**, Ruff 통과, `git diff --check` 통과. PyTorch Transformer에서 기존 `norm_first` 설정에 따른 nested-tensor 최적화 경고 8건이 발생했다. 설정과 모델 구조는 변경하지 않았다.

## AS-IS → 구현 결과

| 영역 | AS-IS | 구현한 TO-BE | 주요 파일 |
|---|---|---|---|
| 요약 | Graph·Description 함수에 cache, task, 성공·실패 저장 반복 | `SummaryBranch`와 공통 실행기. 분기별 schema·식별자·prompt·validator 유지 | `src/extraction/summary_executor.py`, `steps.py` |
| 장면 추출 | backend 분기, 재개 대상, 변환, 저장이 한 함수에 혼재 | 재개 대상을 먼저 계산하고 Qwen/Gemini 실행을 분리. 응답 변환은 파일 I/O 없는 함수 | `src/extraction/scene_executor.py`, `steps.py` |
| 설정 | 긴 검증 함수와 중복 `ValidationConfig` 조립 | protocol·extraction·models·validation 검증 분리, 공통 입력 조립 | `src/pipeline_runtime.py`, `src/validation/config.py` |
| Embedding | fallback, cache, 입력 검증, encode, 저장 혼재 | `_embedding_work` → `_embedding_documents` → `_encode_representations` → `_persist_representations` | `src/validation/steps.py` |
| 진단 | 1,634줄 모듈에 artifact·추천·장면 검증 집중 | support, cohort/representation, recommendation/training, scene 모듈로 분리. 진입점은 검사 순서·통계 가능 여부·보고서 조립 담당 | `src/validation/diagnosis*.py` |
| 추천 | 데이터 준비·selection·refit·평가·저장 혼재 | `_RecommendationInputs`, `_prepare_inputs`, `_select_epoch`, `_refit_model`, `_evaluate_arm`, `_save_refit_checkpoint` | `src/validation/recommendation.py` |
| Worker | 테스트가 생성자를 우회하고 private 필드 수동 설정 | `_start_worker` 경계와 fake process/queue를 사용해 실제 생성자 경로 검사 | `src/extraction/backends/qwen_workers.py`, worker 테스트 |
| 테스트 | 큰 통합 파일에 설정·요약·장면·Embedding 검증 혼재 | 역할별 5개 파일과 `pipeline_fixtures.py`. Graph/Description 실패·재개 시나리오 공통화 | `tests/integration/test_pipeline_*.py` |
| 불필요한 결합 | 과거 README 문구·빈 줄 순서·private 속성 부재 고정 | README 명령 dispatch·필수 옵션·상대 링크 검사. 표시 전용 검사와 private 속성 부재 검사 제거 | README/monitoring/runtime 관련 테스트 |
| 테스트 전용 래퍼 | 사용되지 않는 마지막 frame 래퍼와 자기 구현을 반복하는 assertion | `_last_decodable_frame_timestamp_seconds` 제거. 실제 tail timestamp 계산 및 EOF 복구 검증 유지 | `video_processor.py`, `test_data_preparation.py` |

## 보존한 경계

- 기존 CLI·옵션, stage 함수 호출, 종료 코드, artifact 경로·필드·schema를 유지했다. 새로운 pipeline 실행 옵션이나 orchestration framework는 없다.
- Qwen은 성공+실패가 전체 장면을 덮으면 재사용하고, 불완전한 content는 전체 장면을 다시 처리한다. 각 장면 callback에서 기존 순서로 checkpoint를 저장한다.
- Gemini는 성공 장면을 재사용하고 실패·누락 장면만 요청한다. content별 요청 결과가 모두 수집되면 저장한다. SDK retry 설정은 그대로다.
- 요약은 유효 cache 재사용, 불일치·누락 재생성, 성공 즉시 저장, 실패 기록 제거 시점을 유지한다. 빈 scene과 batch 실패의 기존 동작도 유지한다.
- Gemini summary가 없을 때만 Qwen으로 fallback한다. 잘못된 Gemini summary는 오류다. 모든 처리 대상 입력을 검증한 뒤 모델을 로드하고, 모든 encode가 끝난 뒤 저장한다.
- 추천의 seed → Arm 반복, seed 초기화, 모델·optimizer 생성, RNG permutation, selection/refit 데이터와 popularity 분포, 동점 처리, 학습 수식은 유지한다.
- 진단의 오류 수집 순서와 통계 계산 조건을 유지한다. runtime 판정 실패가 통계 계산을 무조건 막도록 바꾸지 않았다. 하위 진단 모듈은 최상위 진단 모듈을 import하지 않는다.

## 변경 전후 동등성 검증

[기계 판독 가능한 결과와 케이스별 hash](e2e-refactoring-comparison.json)에 실제 관측값을 기록했다. `implementation_source_sha256`는 경로 순서로 정렬한 `src/**/*.py`의 경로와 LF로 정규화한 파일 내용으로 계산했다.

| 비교 | 결과 | 비교한 내용 |
|---|---|---|
| 비학습 계약 56개 | 전체 일치 | 11단계 CLI 연결, 요약 재개·실패, 장면 checkpoint·Gemini 재개, Embedding fallback, 진단 오류·coverage·통계 |
| 생성 요청 98회 | 전체 일치 | task 순서·ID, prompt, 이미지 경로와 순서, 생성 설정 |
| encode 호출 48회 | 전체 일치 | BGE에 전달한 문자열 목록과 순서. 실제 BGE 추론은 fake |
| JSON/JSONL 쓰기 이벤트 1,248건 | 전체 일치 | 경로, 값, durable 설정, 호출 순서. 테스트 준비 쓰기도 포함 |
| PyTorch 테스트 8개 | 전체 일치 | NumPy 기준 loss 등 기존 학습 검사와 4개 Arm × 3개 seed smoke |
| 실제 CPU 학습 checkpoint 12개 | 전체 일치 | metadata 및 **492개 state_dict tensor**의 dtype·shape·값 hash, 학습 이력, 추천 ranking |

비교는 JSON 객체 내용을 사용하며, NPZ·checkpoint는 압축 파일 바이트 대신 내부 배열·tensor를 비교했다. 손상 cache 및 진단 테스트의 mock checkpoint는 원본 파일 hash로 비교한다. 임시 루트·저장소 루트(Windows 예외 문자열의 이스케이프 경로 포함), `elapsed_seconds`, 매 실행 새로 만드는 fixture MP4의 `source_mtime_ns`를 정규화했다. 이 정규화가 실제 source fingerprint 검증을 대체하지는 않으며, donor·sampling 변경·cache 무효화 테스트는 별도로 유지했다.

기준 source는 `git archive e9aff2f`로 보관하고, 동일한 기준 테스트에 baseline/current source 경로를 명시해 실행했다. 관측기는 `pipeline_runtime.__file__`이 지정 source 아래인지 확인한다. 기준 테스트의 private `_minimal_graph_failures` 직접 호출 한 곳만 인자 수 차이에 맞추는 shim을 적용했다. 이 helper의 파일 저장 책임이 호출자로 이동했으므로 필요한 테스트 호환 처리이며, 기준 source는 수정하지 않았다.

추가한 회귀 검사:

- validation 점수 `[0.1, 0.3, 0.3, 0.2]`, patience 2에서 best epoch 2, 종료 epoch 4, refit 2회. selection과 refit의 모델·optimizer가 별개이고 refit 입력에 valid target이 추가되는지 확인.
- Graph·Description 각각 저장 실패 시 기존 summary·실패 파일 보존, 저장 후 중단 시 성공 파일 유지·실패 기록 제거, generator 종료 확인.
- Worker 생성 시 즉시 process 시작, 정상 종료의 sentinel·join·queue close 순서, 중복 close 방지 확인. 기존 interrupt/강제 종료 테스트 유지.

## 재현 방법

이번 검증은 Windows의 `C:/miniconda3/envs/llmjg/python.exe`에서 수행했다. 실제 환경은 Python 3.11.15, NumPy 2.4.3, **PyTorch 2.14.0+cpu**다. 기존 환경에 없던 CPU PyTorch를 학습 검증용으로 설치했다.

전체 검사는 저장소 루트에서 실행한다.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
C:/miniconda3/envs/llmjg/python.exe -m pytest -q
C:/miniconda3/envs/llmjg/python.exe -m ruff check .
git diff --check
```

동등성 원본 snapshot은 현재 로컬 `.git/refactor-audit/` 아래의 `before.json`, `after.json`, `training-before.json`, `training-after.json`이다. 약 10MB의 원본은 Git에 추가하지 않았고, 비교 관측기와 케이스별 hash 결과는 커밋했다. snapshot은 다음으로 다시 비교할 수 있다.

```powershell
C:/miniconda3/envs/llmjg/python.exe tests/refactor_capture.py .git/refactor-audit/before.json .git/refactor-audit/after.json
C:/miniconda3/envs/llmjg/python.exe tests/refactor_capture.py .git/refactor-audit/training-before.json .git/refactor-audit/training-after.json
```

관측기는 일반 테스트 실행에 개입하지 않는다. snapshot을 새로 만들 때만 `-p refactor_capture`로 로드한다. 기준 archive의 테스트 디렉터리에서 실행하되 `PYTHONPATH`는 현재 저장소의 `tests`, `VCP_CAPTURE_OUTPUT`은 snapshot 경로, `VCP_CAPTURE_SOURCE_ROOT`와 `-o pythonpath=...`는 비교할 구현의 `src`로 지정한다. Windows의 `-o pythonpath`에는 **슬래시(`/`) 경로**를 사용한다. 관측기는 CPU tensor 비교를 위해 PyTorch thread 수를 1로 고정한다.

비학습 snapshot에 사용한 선택식:

```text
tests/integration/test_pipeline_v2.py tests/integration/test_user_cohort_pipeline.py tests/validation/test_four_arm_diagnosis.py
-k "summary_resume or summary_failure or eleven_stage or embedding or diagnosis or statistical or sparse_control or gemini_default or qwen_graph_scene or qwen_description_scene"
```

학습 snapshot은 기준 archive의 `tests/validation/test_model.py` 전체 8개 케이스를 사용한다. 현재 테스트의 추가된 early-stopping 케이스는 전체 회귀 검사에 포함된다.

## 커밋 단계

| 단계 | commit | 검증 |
|---|---|---|
| 1. README와 비교 기준 | `1af381c` | README 검사 14개 |
| 2. 요약 공통화 | `4c0a698` | 관련 검사 53개 |
| 3. 장면 추출 분리 | `d5600ee` | 관련 검사 161개 |
| 4. 설정·Embedding·Worker | `b8436ab` | 관련 검사 146개 |
| 5. 진단 분리 | `29f58fc` | 진단 검사 19개 및 통합 재계산 검사 |
| 6. 추천 실행 분리 | `783088f` | 학습 검사 9개, 변경 전후 8개 비교 |
| 7. 최종 정리 | 본 문서를 포함한 마지막 커밋 | 전체 395개, 64개 snapshot 비교, Ruff, diff 검사 |

## 환경별 남은 검증과 제외 범위

이번 결과는 고정 fixture에서 코드·계약·재개 동작을 보존했다는 근거다. 11단계 연결 검사는 keyframe 처리·LLM·BGE·추천 실행을 fake로 대체한다. 추천은 별도의 CPU PyTorch 검사에서 실제 학습했다. 실제 데이터/GPU/Cloud E2E 성공 또는 실험 가설의 유효성을 증명하지 않는다.

- **Ubuntu/CUDA 미실행:** 실제 Qwen/BGE 모델, GPU process·메모리 종료, CUDA 학습 결과, 실제 MP4/ffmpeg fractional seek·EOF 복구.
- **Cloud 미실행:** 실제 Vertex/Gemini 요청·SDK retry·인증·할당량. 관련 계약은 fake 응답으로 검사했다.
- **별도 변경:** 기본 6장 sampling을 cohort 통계가 3장으로 보고하는 문제는 수정하지 않았다. sampling 전달과 기존 artifact/run 호환성 정책을 함께 설계해야 한다.
- **별도 변경:** `run.sh`의 고정 run ID와 주석 처리된 Gemini 단계는 그대로다. 이번 E2E 기준은 README 명령 구성이다.
- **유지:** 평가 임계값, 비교 family, fallback 정책, prompt/schema, 사용자 선정·split·catalog·sampling·Arm·seed 설정.
