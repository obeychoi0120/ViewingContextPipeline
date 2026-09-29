# 실행·저장 계약

현재 설정과 `260928_v7`에서 사용하는 계약입니다. 빠른 실행 순서는 [README](../README.md), 직접 Graph 모델은 [Graph 학습](graph_training.md)을 참고하세요.

## 입력 준비와 실행 자원

`preparation`은 원본 CSV·제목·영상 검증, 공유 cohort 입출력, duration·keyframe 준비를 담당합니다. `validation.rolling_data`는 시간순 이력과 rolling 평가 분할 정책을 담당합니다. 준비 결과는 `<artifacts_root>/preparation/`에 공유되므로 다른 데이터셋이나 평가 구간은 별도 `artifacts_root`를 사용합니다.

Qwen은 로컬 checkpoint와 vLLM 0.28.0을 사용하고, 보이는 GPU별 worker를 실행합니다. `CUDA_VISIBLE_DEVICES`를 지정하지 않으면 보이는 모든 GPU를 사용합니다. GPU가 없으면 Qwen은 실패합니다. 추천은 GPU가 없으면 CPU로 실행할 수 있습니다. `--gpus` 옵션은 없습니다. Gemini의 전체 Scene 동시성은 `extraction.gemini.threads`로 정하며, 콘텐츠 구분 없이 worker가 대기 중인 장면을 처리합니다.

Scene·Summary 토큰 한도는 `extraction.{graph,description}.{qwen,gemini}`에 있습니다. Summary는 원본 추출 모델이 아니라 실제 요약 모델의 한도를 사용합니다. Qwen Scene은 greedy 생성이며 Summary의 샘플링은 `greedy_decoding`과 `summary_sampling` 설정을 따릅니다. Gemini는 설정된 생성 옵션을 사용합니다.

## Scene·Summary 저장

- Scene: `runs/<RUN_ID>/extraction/scenes/<source>/<content_id>.jsonl`.
- Description 행: `content_id`, `scene_idx`, `tokens`, `description`.
- Graph 행: `content_id`, `scene_idx`, `tokens`, `warning`, `scene_graph`. `scene_graph`는 구조화 객체 또는 보존한 원문 문자열입니다.
- Summary: `runs/<RUN_ID>/extraction/summaries/<source>/<content_id>.json`. `video-summary/v5`이며 텍스트·상태·장면 수·단어 수·토큰 수·생성 provenance를 저장합니다.
- Scene 실패: 해당 Scene 디렉터리의 `failures/<content_id>.jsonl`. Summary 실패: 해당 Summary 디렉터리의 `failures.jsonl`.

Graph 출력은 JSON Schema로 강제하지 않습니다. 완전한 구조화 출력과 명확한 형식 변형은 파싱하고, 구조화되지 않는 출력은 원문으로 보존합니다. 토큰 한도 종료는 실제 backend 종료 사유로 판단합니다. 원문 Graph는 text Summary에 사용할 수 있지만 직접 graph 학습에서는 구조·warning 검사를 통과한 장면만 사용합니다. Parser는 제공된 프롬프트의 관계형 Graph와 Context + Actions 형식을 처리합니다.

Summary에는 성공 장면만 넣습니다. 성공 장면이 없거나 최종 생성에 실패하면 `status="failed"`, `text=""`, `word_count=0`입니다. 문단·목록·줄바꿈을 허용하며 단어 수는 실패 기준이 아닙니다. 신규 실패의 `raw_output`은 빈 문자열이고 Graph 토큰 한도 종료는 원문을 보존합니다.

출력 저장은 임시 파일을 작성한 뒤 원자적으로 교체합니다. 완료된 실패 시도는 실패 로그에 먼저 기록해, 결과 저장 중 중단되어도 같은 최종 시도를 불필요하게 반복하지 않습니다.

## 재사용과 재시도

일반 생성은 같은 Run의 성공 결과를 보존합니다. 프롬프트·모델 설정·Scene 입력을 변경해도 이미 성공한 Summary는 자동으로 갱신하지 않습니다. 생성 당시 provenance를 현재 설정으로 바꾸지 않습니다. 성공 결과 갱신은 `--force`를 사용합니다.

Qwen Scene은 설정된 penalty마다 전체 pass를 마친 뒤 실패 항목만 다음 pass로 넘깁니다. 재실행은 첫 penalty부터 시작합니다. Qwen Summary는 신규 콘텐츠를 첫 penalty로 처리하고, 기존 실패는 마지막 기록보다 큰 다음 penalty부터 처리합니다. 마지막 penalty까지 완료한 실패와 penalty 기록이 없는 과거 실패는 빈 Summary를 유지합니다. 실패 당시 Scene 입력 해시가 현재 입력과 달라지면 재시도합니다.

Gemini의 `429 RESOURCE_EXHAUSTED`는 30초 후 한 번 재시도합니다. 다른 오류는 해당 요청 실패로 처리합니다. 실패를 기록한 생성 단계는 오류를 반환할 수 있으며 같은 명령으로 다시 실행합니다. 성공적으로 복구한 항목의 실패 로그는 제거합니다.

현재 Run의 기존 Description Summary에는 다른 Run에서 생성된 provenance가 들어 있을 수 있습니다. 원래 모델·프롬프트·입력 해시는 유지하고, 경로가 과거 Run을 가리킨다는 이유만으로 재생성하지 않습니다.

## Text 표현과 공유 캐시

`meta`는 제목을, `graph_qwen`은 제목 없는 Summary를 사용합니다. 결합 Arm은 양쪽 공백을 제거한 제목과 Summary를 빈 줄 하나로 합칩니다. 하나만 있으면 해당 값만 사용하고, 둘 다 없으면 영벡터입니다. BGE 최대 길이는 `validation.encoder.max_length`이며 truncation 수를 기록합니다.

공유 캐시는 `<artifacts_root>/shared_cache/{embeddings,graph_inputs,recommendations}/`입니다. 로컬 결과를 먼저 검증하고, 공유 캐시를 확인한 뒤 필요한 계산만 수행합니다. 키별 파일 잠금·체크섬·원자적 게시로 완료 결과를 보호합니다. `--force`는 읽기를 우회하며 정상적인 기존 공유 엔트리를 덮어쓰지 않습니다.

- 공통 데이터: 순서가 있는 catalog, 전체 events, 평가 구간.
- Text 입력: 실제 결합 텍스트·결측 상태, BGE 식별자·설정, Arm·결합 계약, 생성 provenance.
- 생성 provenance: 프롬프트 본문 해시, 모델, 설정, 실제 Scene 입력 해시. Run 이름·파일 경로·기록 시각은 의미 해시에서 제외합니다.
- Graph 입력: 실제 Scene 파일 바이트, catalog 순서, 제목, BGE 식별자·설정, 변환 계약.
- 추천: 표현 입력·행렬 해시, 데이터·학습 계약, 날짜·seed·설정. 직접 graph는 모델 구조와 pooling도 포함합니다.

Summary의 생성 provenance와 `scene_input_hash`를 확인할 수 있으면 간결한 Scene 형식도 Run 간 공유할 수 있습니다. 입력 해시가 없으면 검증 가능한 Scene 생성 provenance가 필요합니다. 검증되지 않는 결과는 로컬에서만 사용하며, 현재 설정을 과거 기록에 소급하지 않습니다.

추천 재사용 단위는 완료된 날짜 × seed × Arm입니다. `sasrec.pt`, `training.json`, `per_event_metrics.jsonl`, `complete.json`을 검증하며 직접 graph에는 최종 catalog 벡터도 포함됩니다. 재사용한 결과는 독립 파일로 복사하고 현재 Run metadata와 `reused_from`을 기록합니다. 모든 결과가 재사용되면 모델·학습 worker를 초기화하지 않습니다.

## 유지하는 계약과 지원 경계

설정 `experiment_config_version: v4`, Arm `title-summary-six-arms/v2`, 표현 `shared-scenes-representation/v3`, 선택 `full-catalog-zero-vector/v2`, 추천 `sasrec-rolling-combination/v3`를 유지합니다. 내부 Python 모듈 이동은 이 식별자와 캐시 키를 바꾸지 않습니다.

주 CLI는 preparation의 두 단계, extraction의 두 Scene 추출 단계와 `summarize`, validation의 임베딩·추천·진단입니다. 과거 요약 별칭, Arm·Scene 일괄 변환 명령, 독립 제목 보완 CLI는 제거했습니다. 제목 보완은 `prepare-cohort`에서 수행합니다. 기존 Run·분석 보고서·프롬프트를 자동 이동하거나 삭제하지 않습니다.
