# 실행·저장 계약

현재 leave-one-out 설정의 실행 계약입니다. 과거 Run 보고서는 생성 당시 조건을 따릅니다. 빠른 실행 순서는 [README](../README.md), 직접 Graph 모델은 [Graph 학습](graph_training.md)을 참고하세요.

## 입력 준비와 실행 자원

`preparation`은 원본 CSV·제목·영상 검증, 공유 cohort 입출력, duration·keyframe 준비를 담당합니다. `validation.leave_one_out_data`는 사용자별 순서와 마지막 2개 보류 정책을 담당합니다. 전체 원본은 공유 preparation에 보존하고, 최소 이력 5개·최근 13개 절단과 catalog 구성은 Run별 validation에서 수행합니다. 준비 결과는 `<artifacts_root>/preparation/`에 공유되므로 다른 원본 데이터셋은 별도 `artifacts_root`를 사용합니다.

Qwen은 로컬 checkpoint와 vLLM 0.28.0을 사용하고, 보이는 GPU별 worker를 실행합니다. `CUDA_VISIBLE_DEVICES`를 지정하지 않으면 보이는 모든 GPU를 사용합니다. GPU가 없으면 Qwen은 실패합니다. 추천은 GPU가 없으면 CPU로 실행할 수 있습니다. `--gpus` 옵션은 없습니다. Gemini의 전체 Scene 동시성은 `extraction.gemini.threads`로 정하며, 콘텐츠 구분 없이 worker가 대기 중인 장면을 처리합니다.

Scene·Summary 토큰 한도는 `extraction.{graph,description}.{qwen,gemini}`에 있습니다. Summary는 원본 추출 모델이 아니라 실제 요약 모델의 한도를 사용합니다. Qwen Scene은 greedy 생성이며 Summary의 샘플링은 `greedy_decoding`과 `summary_sampling` 설정을 따릅니다. Gemini는 설정된 생성 옵션을 사용합니다.

## Scene·Summary 저장

키프레임은 `extraction.visual_evidence.image_resolution: [width, height]`에 따라
`preparation/resized_keyframes/<width>_<height>/<content_id>/`에 저장하고 읽습니다.
예를 들어 `[640, 352]`이면 `resized_keyframes/640_352/`를 사용합니다.
해상도를 바꾸면 별도 디렉터리에 준비하며, 같은 해상도로 돌아오면 기존 프레임을 재사용합니다.
기존 `resized_keyframes/<content_id>/` 파일은 자동 이동하거나 읽지 않습니다.
새 경로의 프레임은 `prepare-input-data`를 실행해 준비합니다.

- Scene: `runs/<RUN_ID>/extraction/scenes/<source>/<content_id>.jsonl`.
- Description 행: `content_id`, `scene_idx`, `tokens`, `description`.
- Graph 행: `content_id`, `scene_idx`, `tokens`, `warning`, `scene_graph`. `scene_graph`는 구조화 객체 또는 보존한 원문 문자열입니다.
- Summary: `runs/<RUN_ID>/extraction/summaries/<source>/<content_id>.json`. `video-summary/v5`이며 `complete`·`failed` 상태와 명시적인 `provenance.summary_model`(`qwen` 또는 `gemini`)을 요구합니다. 텍스트·장면 수·단어 수·토큰 수·생성 provenance를 저장합니다.
- Scene 실패: 해당 Scene 디렉터리의 `failures/<content_id>.jsonl`. Summary 실패: 해당 Summary 디렉터리의 `failures.jsonl`.

Graph 출력은 JSON Schema로 강제하지 않습니다. 완전한 구조화 출력과 명확한 형식 변형은 파싱하고, 구조화되지 않는 출력은 원문으로 보존합니다. 토큰 한도 종료는 실제 backend 종료 사유로 판단합니다. 원문 Graph는 text Summary에 사용할 수 있지만 직접 graph 학습에서는 구조·warning 검사를 통과한 장면만 사용합니다. Parser `graph-text/v10`은 현재 Context·Entities·Actions 형식과 이에 대응하는 JSON만 구조화합니다. 보수적인 구두점 복구는 유지하지만 Relations·receiver·location 누락·구형 필드를 변환하지 않습니다.

Summary에는 성공 장면만 넣습니다. 성공 장면이 없거나 최종 생성에 실패하면 `status="failed"`, `text=""`, `word_count=0`입니다. 문단·목록·줄바꿈을 허용하며 단어 수는 실패 기준이 아닙니다. 신규 실패의 `raw_output`은 빈 문자열이고 Graph 토큰 한도 종료는 원문을 보존합니다.

출력 저장은 임시 파일을 작성한 뒤 원자적으로 교체합니다. 완료된 실패 시도는 실패 로그에 먼저 기록해, 결과 저장 중 중단되어도 같은 최종 시도를 불필요하게 반복하지 않습니다.

## 재사용과 재시도

Scene은 위 compact 필드만 읽으며 저장된 `graph`·`raw_response` 별칭이나 과거 메타데이터·상태를 추론하지 않습니다. Summary v4·`raw_fallback`·모델 필드 누락도 지원하지 않습니다. 지원 밖 기존 입력은 경로와 원인을 명시하여 거부하고 일반 재실행에서 원본을 변환하거나 덮어쓰지 않습니다. Run 이름이나 생성 당시 프롬프트 버전만으로 현재 형식의 입력을 차단하지 않습니다.

일반 생성은 같은 Run의 성공 결과를 보존합니다. 프롬프트·모델 설정·Scene 입력을 변경해도 이미 성공한 Summary는 자동으로 갱신하지 않습니다. 생성 당시 provenance를 현재 설정으로 바꾸지 않습니다. 성공 결과 갱신은 `--force`를 사용합니다.

Qwen Scene은 설정된 penalty마다 전체 pass를 마친 뒤 실패 항목만 다음 pass로 넘깁니다. 재실행은 첫 penalty부터 시작합니다. Qwen Summary는 신규 콘텐츠를 첫 penalty로 처리하고, 기존 실패는 마지막 기록보다 큰 다음 penalty부터 처리합니다. 마지막 penalty까지 완료한 실패와 penalty 기록이 없는 과거 실패는 빈 Summary를 유지합니다. 실패 당시 Scene 입력 해시가 현재 입력과 달라지면 재시도합니다.

Gemini의 `429 RESOURCE_EXHAUSTED`는 30초 후 한 번 재시도합니다. 다른 오류는 해당 요청 실패로 처리합니다. 실패를 기록한 생성 단계는 오류를 반환할 수 있으며 같은 명령으로 다시 실행합니다. 성공적으로 복구한 항목의 실패 로그는 제거합니다.

현재 Run의 기존 Description Summary에는 다른 Run에서 생성된 provenance가 들어 있을 수 있습니다. 원래 모델·프롬프트·입력 해시는 유지하고, 경로가 과거 Run을 가리킨다는 이유만으로 재생성하지 않습니다.

## Text 표현과 공유 캐시

`meta`는 제목을, `graph_qwen`은 제목 없는 Summary를 사용합니다. 결합 Arm은 양쪽 공백을 제거한 제목과 Summary를 빈 줄 하나로 합칩니다. 하나만 있으면 해당 값만 사용하고, 둘 다 없으면 영벡터입니다. BGE 최대 길이는 `validation.encoder.max_length`이며 truncation 수를 기록합니다.

공유 캐시는 `<artifacts_root>/shared_cache/{embeddings,graph_inputs,recommendations}/`입니다. 로컬 결과를 먼저 검증하고, 공유 캐시를 확인한 뒤 필요한 계산만 수행합니다. 키별 파일 잠금·체크섬·원자적 게시로 완료 결과를 보호합니다. `--force`는 읽기를 우회하며 정상적인 기존 공유 엔트리를 덮어쓰지 않습니다.

- 공통 데이터: 순서가 있는 평가 catalog, 보존된 events, 사용자별 분할.
- Text 입력: 실제 결합 텍스트·결측 상태, BGE 식별자·설정, Arm·결합 계약, 생성 provenance.
- 생성 provenance: 프롬프트 본문 해시, 모델, 설정, 실제 Scene 입력 해시. Run 이름·파일 경로·기록 시각은 의미 해시에서 제외합니다.
- Graph 입력: 실제 Scene 파일 바이트, catalog 순서, 제목, BGE 식별자·설정, 변환 계약.
- 추천: 표현 입력·행렬 해시, 데이터·학습 계약, 프로토콜·분할 fingerprint·seed·설정. 직접 graph는 모델 구조와 pooling도 포함합니다.

Summary의 생성 provenance와 `scene_input_hash`를 확인할 수 있으면 간결한 Scene 형식도 Run 간 공유할 수 있습니다. 입력 해시가 없으면 검증 가능한 Scene 생성 provenance가 필요합니다. 검증되지 않는 결과는 로컬에서만 사용하며, 현재 설정을 과거 기록에 소급하지 않습니다.

추천 재사용 단위는 완료된 seed × Arm입니다. `sasrec.pt`, `training.json`, `per_event_metrics.jsonl`, `complete.json`을 검증하며 직접 graph에는 최적 checkpoint의 catalog 벡터도 포함됩니다. 재사용한 결과는 독립 파일로 복사하고 현재 Run metadata와 `reused_from`을 기록합니다. 모든 결과가 재사용되면 모델·학습 worker를 초기화하지 않습니다.

## 유지하는 계약과 지원 경계

설정은 `experiment_config_version: v5`, validation은 `validation-config/v7`, 원본 cohort는 `microlens-full-source/v2`, 평가 cohort는 `microlens-leave-one-out/v1`, 선택은 `leave-one-out-sequence-catalog/v1`, 추천은 `sasrec-leave-one-out-combination/v1`, 진단은 `leave-one-out-diagnosis/v1`입니다. 학습 구현은 `shared-scenes-leave-one-out-training/v1`이며 프로토콜·분할 fingerprint로 과거 rolling 결과의 재사용을 차단합니다. 모델·Arm·Text 표현 계약은 유지합니다. Graph 입력은 `role-graph-inputs/v4`로 갱신하여 이전 정규화 캐시를 차단합니다. receiver 예약 텐서 슬롯과 모델 파라미터 shape는 유지합니다.

주 CLI는 preparation의 두 단계, extraction의 두 Scene 추출 단계와 `summarize`, validation의 임베딩·추천·진단입니다. 과거 요약 별칭, Arm·Scene 일괄 변환 명령, 독립 제목 보완 CLI는 제거했습니다. 제목 보완은 `prepare-cohort`에서 수행합니다. 기존 Run·분석 보고서·프롬프트를 자동 이동하거나 삭제하지 않습니다.

## Leave-one-out 실행과 보존

공유 원본은 `<artifacts_root>/preparation/cohort/leave_one_out/`, Run별 평가 파일은 `runs/<RUN_ID>/validation/leave_one_out/` 아래에 저장합니다. 기존 rolling cohort·Run·캐시는 보존하며 rolling 실행·재진단은 지원하지 않습니다. Scene·Summary·프레임 경로는 그대로이며 새 catalog와 provenance에 맞는 고정 특징만 재사용합니다.

기존 명령 `prepare-cohort → embed-representations → run-recommendation → run-diagnosis`를 사용합니다. 원본 CSV가 있는 환경에서 새 공유 preparation을 먼저 완료하십시오. 날짜 대신 `seed × Arm` 조합을 실행하고 validation 최고 checkpoint를 복원하여 test를 평가합니다. Graph는 복원 후 Item/catalog 캐시를 비웁니다. `training.json`은 `train`, `best_epoch`, `checkpoint_policy`를, test 행은 `source_event_id`와 `train_item_frequency`를 기록합니다.

진단은 `per_seed`와 전체 사용자 평균을 제공하고 사용자별 seed 평균으로 paired bootstrap을 수행합니다. Run 비교는 같은 프로토콜·전처리·원본·분할·catalog·seed를 요구합니다. DWYS의 rolling HR@10=0.046 대비 차이 계산은 제공하지 않습니다. 전처리 수치·fingerprint·제외 이유는 Run의 `reports/leave_one_out_cohort.{json,md}`와 validation manifest에서 확인합니다.

## 현재 형식 지원 정리 (2026-10-07)

호출이 없는 `validation/io.py`, 구형 rolling benchmark와 raw Graph 호환 모듈을 제거했습니다. 미사용 빈도·catalog scoring·구형 bootstrap·top-k·로그 헬퍼와 분할 재계산 wrapper를 제거하고 학습 전용 루프를 직접 계산으로 정리했습니다. 실제 provenance의 `selection_hash`, 사용자 단위 bootstrap·Bonferroni, Qwen의 문자열 경로 logits 모듈과 Graph 수치 참조 구현은 유지합니다. NumPy 손실 검증은 현행 `transition_loss`를 검사합니다.

Run 산출물·과거 보고서·프롬프트·v7 fixture는 보존합니다. 모델 구조·손실함수·평가 조건·CLI는 변경하지 않습니다. 실제 Run 데이터와 전체 GPU 실험은 이 VM에서 검증하지 못했습니다.

검사 결과: `vc_cloud`에서 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q` **296개 통과(72.76초)**. `python -m ruff check src tests benchmarks`, `git diff --check`, 현행 Scene GCN benchmark의 `--help`도 통과했습니다. 현재 형식의 생성·저장·재사용·임베딩·추천·진단, 비지원 입력 거부·원본 보존, 3개 seed·6개 Arm, mean/GNN 수치·gradient·최적 checkpoint·직렬/병렬·중단 복구·캐시 검증을 포함합니다.
