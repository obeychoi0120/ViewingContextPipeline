# 실행·저장 계약

현재 leave-one-out 설정의 실행 계약입니다. 과거 Run 보고서는 생성 당시 조건을 따릅니다. 빠른 실행 순서는 [README](../README.md), 직접 Graph 모델은 [Graph 학습](graph_training.md)을 참고하세요.

## 입력 준비와 실행 자원

`preparation`은 원본 CSV·제목·영상 검증, 공유 cohort 입출력, duration·keyframe 준비를 담당합니다. `validation.leave_one_out_data`는 사용자별 순서와 마지막 2개 보류 정책을 담당합니다. 전체 원본은 공유 preparation에 보존하고, 최소 이력 5개·최근 13개 절단과 catalog 구성은 Run별 validation에서 수행합니다. 준비 결과는 `<artifacts_root>/preparation/`에 공유되므로 다른 원본 데이터셋은 별도 `artifacts_root`를 사용합니다.

Qwen은 로컬 checkpoint와 vLLM 0.28.0 이상을 사용하고, 보이는 GPU별 worker를 실행합니다. `CUDA_VISIBLE_DEVICES`를 지정하지 않으면 보이는 모든 GPU를 사용합니다. GPU가 없으면 Qwen은 실패합니다. 추천은 GPU가 없으면 CPU로 실행할 수 있습니다. `--gpus` 옵션은 없습니다. Gemini의 전체 Scene 동시성은 `extraction.gemini.threads`로 정하며, 콘텐츠 구분 없이 worker가 대기 중인 장면을 처리합니다.

Scene·Summary 토큰 한도는 `extraction.{graph,description}.{qwen3vl,qwen3.5,gemini}`에 있습니다. Summary는 원본 추출 모델이 아니라 실제 요약 모델의 한도를 사용합니다. 두 Qwen 모델의 장면 추출과 요약은 greedy decoding을 사용합니다(temperature 0.0, top_p 1.0, top_k 0). Qwen 장면 추출·요약 CLI의 `--repetition-penalty FLOAT`는 해당 실행에 적용할 유한한 단일 값(1.0~2.0)을 받습니다. 생략하면 Graph 장면 추출은 **1.05**, Description 장면 추출과 모든 요약은 **1.0**입니다. Gemini에 이 옵션을 지정하면 오류입니다. 생성된 토큰에만 적용하는 기존 logits processor를 사용하며 vLLM 기본 repetition penalty는 1.0으로 두어 중복 적용을 막습니다. 최초 요청·구조 복구·실패 후 재실행에 같은 값을 전달하며 증가 schedule은 사용하지 않습니다. 새 요청·schedule·provenance에 적용값을 기록하며, 기록된 repetition penalty와 다른 값을 지정해도 정상 Scene과 완료 Summary는 재사용하고 누락·실패·warning Scene 및 실패 Summary는 현재 값으로 생성합니다. Qwen3.5는 Thinking을 해제합니다. 출력 한도는 Graph 장면 384, Description 장면·요약 512입니다. 공통 vLLM 자원 설정은 `extraction.qwen`이며 모델 선택과는 별개입니다.

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
- Summary: `runs/<RUN_ID>/extraction/summaries/<source>/<content_id>.json`. `video-summary/v5`이며 `complete`·`failed` 상태와 명시적인 `provenance.summary_model`(`qwen3vl`, `qwen3.5`, `gemini`)을 요구합니다. 텍스트·장면 수·단어 수·토큰 수·생성 provenance를 저장합니다.
- Scene 실패: 정상·warning 행과 같은 `<content_id>.jsonl`에 `content_id`, `scene_idx`, `tokens`, `error`, `raw_output` 다섯 필드로 저장합니다. `tokens`는 실제 출력 토큰 수 또는 null, `error`는 실패 사유, `raw_output`은 실제 응답 원문(없으면 빈 문자열)입니다. Summary 실패는 기존 Summary 디렉터리의 `failures.jsonl`에 저장합니다.
- Graph 추출 집계: `extract-graph-scenes` 완료 후 같은 Scene 디렉터리의 `generation_stats.json`에 `total`, `success`, `warning`, `failed`, `warning_counts`를 저장합니다. 현재 실행 대상의 재사용 Scene과 새 생성 결과를 모두 포함합니다. `success`는 Warning 없는 정상 Graph, `warning`은 Warning이 있는 Scene 수, `failed`는 최종 실패 Scene 수입니다. `warning_counts`에는 `MISSING_REQUIRED`, `PARSE_FAILED`별 Scene 수를 기록하며 여러 태그가 있는 Scene은 각 태그에 포함합니다.

Graph 출력은 JSON Schema로 강제하지 않습니다. Parser `graph-text/v14`는 v9의 `name; attribute` 형식과 기존 `id: kind; attribute` 형식, 현재 스키마의 JSON을 지원합니다. v9 이름은 공백과 번호를 포함해 정확히 대조하고 장면 내 `e1`, `e2` 등의 내부 ID로 변환합니다. 이름·속성은 보존하며 동의어·대소문자로 개체를 추측하지 않습니다. 한 출력에서 두 선언 형식을 혼용하면 원문으로 보존합니다. 필수 필드와 타입을 검사합니다. Warning은 `MISSING_REQUIRED`(필수 요소 누락), `PARSE_FAILED`(형식·타입 파싱 실패) 두 가지입니다. 미선언 역할 참조는 Warning 없이 `unknown`으로 바꾸며 actor·target이 모두 불명이거나 없는 Action도 유지합니다. 중복 Entity ID는 모든 Entity를 보존하고 같은 ID의 참조는 마지막 선언에 연결합니다. 구조화할 수 없는 출력은 원문으로 보존하여 Text Summary에 사용할 수 있지만 직접 Graph 학습에서는 제외합니다. 기존 Scene은 읽을 때 다시 파싱하며 제거된 Warning은 무시하고 `PARSE_ERROR`는 `PARSE_FAILED`로 읽습니다. 원본 파일은 덮어쓰지 않습니다. 토큰 한도 종료는 실제 backend 종료 사유로 판단합니다. 보수적인 구두점 복구는 유지하지만 Relations·receiver·place 누락·구형 필드를 변환하지 않습니다.

Summary에는 성공 장면만 넣습니다. 성공 장면이 없거나 최종 생성에 실패하면 `status="failed"`, `text=""`, `word_count=0`입니다. 문단·목록·줄바꿈을 허용하며 단어 수는 실패 기준이 아닙니다. 실패 Scene은 Graph·Description 모두 토큰 한도 종료 여부와 관계없이 받은 원문을 `raw_output`에 보존합니다. 이 원문은 Summary·임베딩·Graph 구성에 사용하지 않습니다. 실패 Summary의 `raw_output`은 기존처럼 빈 문자열입니다.

출력 저장은 임시 파일을 작성한 뒤 원자적으로 교체합니다. Scene 결과는 콘텐츠별로 정렬하여 원자적으로 교체하며 `(content_id, scene_idx)`당 최신 결과 한 행만 유지합니다. 재시도 결과가 도착하지 않은 기존 warning·실패 행은 보존합니다. 같은 명령을 재실행하면 정상 Scene만 재사용하고 누락·실패·Graph warning을 현재 설정으로 한 번씩 생성합니다. 실행 중 자동 추가 pass는 없습니다.

## 재사용과 재시도

기존 Scene `failures/<content_id>.jsonl`은 추출 재개 준비 시 원본 JSONL에 통합합니다. 같은 Scene이 양쪽에 있으면 기존 실패 로그를 우선하고, 토큰 수가 없으면 null로 둡니다. 원본 JSONL 저장이 완료된 뒤 해당 실패 파일을 제거하므로 저장·정리 중 중단되어도 재개할 수 있습니다. 새 Scene 실패 디렉터리는 만들지 않습니다. Summary는 기존 실패 로그 형식을 유지합니다.

진행바는 `success=정상`, `warning=경고`, `failed=실패`이며 완료 수는 세 값의 합입니다. 처리량·ETA도 같은 완료 수로 계산합니다. 기존 `raw`·`structured` 표시는 제거합니다. 최종 `generation_stats.json`도 같은 분류로 현재 실행 대상 전체를 집계합니다.

Scene은 위 compact 필드만 읽으며 저장된 `graph`·`raw_response` 별칭이나 과거 메타데이터·상태를 추론하지 않습니다. Summary v4·`raw_fallback`·모델 필드 누락도 지원하지 않습니다. 지원 밖 기존 입력은 경로와 원인을 명시하여 거부하고 일반 재실행에서 원본을 변환하거나 덮어쓰지 않습니다. Run 이름·파일 경로는 의미 해시에서 제외합니다. 생성 결과의 재사용과 임베딩·추천 캐시의 호환성 검증은 구분하며, 생성 결과의 재개 규칙은 아래를 따릅니다.



Scene 디렉터리의 `generation_identity.json`은 생성 설정 기록으로 유지하며 재사용을 막는 비교에는 사용하지 않습니다. 기존 기록은 보존하고, 기록이 없으면 현재 설정으로 생성합니다. 이 파일은 폴더 내 모든 Scene이 동일한 설정으로 생성되었다는 보장이 아닙니다. 정상 Scene과 완료 Summary는 생성 설정이 달라도 재사용하고 누락·실패·warning Scene 및 실패 Summary는 현재 설정으로 생성합니다. Summary의 실제 Scene 내용 변경, 모델·소스·제목 및 저장 형식 계약 검증은 유지합니다. `--force`는 기존 결과까지 현재 설정으로 재생성할 때 사용합니다. 구형 `graph_qwen`·`desc_qwen` Run은 오류로 거부하고 원본 파일은 보존합니다.

Graph 추출을 재실행하면 기존 `generation_stats.json`을 제거하고, 모든 요청 처리가 끝난 뒤 원자적으로 다시 저장합니다. 최종 실패 Scene이 남아 있어도 집계하며, 실행 중단·예외로 끝난 경우에는 완료 통계를 저장하지 않습니다. 모든 Scene을 재사용하는 실행에서도 집계 파일을 생성합니다.

Qwen의 Graph·Description 추출과 요약은 greedy decoding을 사용합니다. `--presence-penalty FLOAT`는 Qwen에서만 지원하며 유한한 -2~2 값으로 지정합니다. 생략 시 Scene 1.5·Summary 2.0을 유지하고, 0이면 presence penalty를 끕니다. 적용값은 새 요청과 해당 provenance의 `settings.generation.presence_penalty`에 기록됩니다. 기존 sampling 결과나 다른 penalty 값의 정상 Scene·완료 Summary도 재사용하며, 기존 파일의 생성 기록은 덮어쓰지 않습니다.

Gemini의 `429 RESOURCE_EXHAUSTED`는 30초 후 한 번 재시도합니다. 다른 오류는 해당 요청 실패로 처리합니다. 실패를 기록한 생성 단계는 오류를 반환할 수 있으며 같은 명령으로 다시 실행합니다. 성공적으로 복구한 Scene은 같은 JSONL의 실패 행을 관측 행으로 교체합니다. Summary의 복구된 실패 로그는 제거합니다.

현재 Run의 기존 Description Summary에는 다른 Run에서 생성된 provenance가 들어 있을 수 있습니다. 원래 모델·프롬프트·입력 해시는 유지하고, 경로가 과거 Run을 가리킨다는 이유만으로 재생성하지 않습니다.

## 장면별 BGE와 공유 캐시

Basic 추천 Arm은 `base d0 d1 g0 g1 g2 g3 g4`이며 Ablation은 `g1a g1b g1c g3a`입니다. 모든 Arm에 제목을 포함합니다. D0·D1은 `desc_qwen3vl`, G0·G1·G2·G3·G4·G1a·G1b·G1c·G3a는 `graph_qwen3vl` 장면 자료를 직접 사용합니다. Summary는 추천 입력에 포함되지 않습니다. BGE는 고정된 1024차원 특징을 생성합니다.

Run 아래 특징은 `validation/representations/{metadata,desc_qwen3vl,graph_qwen3vl}/`에 소스별로 저장합니다. 각 디렉터리의 `manifest.json`에는 입력 해시·catalog·파일 checksum·계약을, `statistics.json`에는 장면 성공·제외·fallback 내역을 기록합니다. 변환 계약은 `scene-bge-inputs/v1`입니다. G1b는 공유 Graph를 수정하지 않고 고정된 교란 뷰를 만듭니다.

공유 캐시는 `<artifacts_root>/shared_cache/{scene_inputs,recommendations}/`입니다. 로컬 결과 검증 → 공유 캐시 → 새 계산 순서이며 키별 잠금·checksum·원자적 게시를 사용합니다. `--force`는 캐시 읽기를 우회합니다. 입력 키에는 실제 Scene 파일 바이트·생성 조건 기록, catalog 순서, 제목, BGE 모델·설정과 변환 계약을 포함합니다. 추천 키에는 추천 계약·Arm 구조·표현 해시·분할·학습 설정·seed·G1b 교란 정책을 추가합니다.

추천 재사용 단위는 완료된 seed × Arm입니다. 모든 Arm에서 `sasrec.pt`, `training.json`, `per_event_metrics.jsonl`, `catalog_vectors.npy`, `complete.json`을 검증합니다. Run 간 재사용은 독립 파일로 복사하여 현재 Run identity와 `reused_from`을 기록합니다. 완료 조합은 모델·worker 초기화 없이 건너뜁니다. epoch 중간 재개는 지원하지 않습니다.

## 유지하는 계약과 지원 경계

설정·추출용 Run 계약은 `experiment_config_version: v6`와 기존 추출 계약을 유지합니다. 추천 계약 **`gnn-attn-aggregation/v1`**은 별도로 관리하며 메타데이터·캐시 검증에만 사용합니다. 계약 이름의 하위 디렉터리는 만들지 않습니다. 모델 구조는 `sasrec-scene-content/v1`, 진단은 `leave-one-out-diagnosis/v5`입니다. 기존 Qwen 장면 자료는 재사용하되 과거 추천 Arm·Summary 임베딩·checkpoint는 새 결과로 재사용하지 않습니다.

validation 설정 `validation-config/v8`, 학습 구현 `shared-scenes-leave-one-out-training/v2`, 평가 `microlens-leave-one-out-evaluation/v2`, 조합 `sasrec-leave-one-out-combination/v2`, 이벤트 `sasrec-leave-one-out-per-event/v2`는 유지합니다. Graph의 receiver 예약 텐서 슬롯도 유지합니다. 원본 추출·프롬프트·기존 Run 산출물은 이동·삭제하지 않습니다.

## Leave-one-out 실행과 진단

`embed-representations`, `prepare-module`, `run-recommendation`, `run-diagnosis`는 독립 단계이며 `--target`이 필수입니다. 각 Arm이 입력·집계 방식을 정하므로 표현 모드·집계 옵션은 받지 않습니다. Basic과 Ablation의 실행 순서는 사용자가 정하며 성능에 따른 허용·거부 로직은 없습니다. 추천 명령은 특징·구간 준비나 진단을 자동 실행하지 않습니다.

G3·G4는 `embed-representations` 이후, 학습 전에 `python -m validation prepare-module --run-id "$RUN_ID" --dataset "$DATASET" --target g3`로 공통 구간 파일을 준비합니다. 이후 G3a 학습 전에는 같은 Step의 `--target g3a`로 섞은 구간만 준비합니다. 이 Step의 `--target`은 구간 확장을 사용하는 Arm만 받습니다. G3와 G4는 Graph 소스 bundle의 `scene_segments.npy`·`scene_segments.json`을 공유하며 G3a는 별도 `scene_segments_shuffled.npy`·`scene_segments_shuffled.json`을 사용합니다. 이 단계는 실행할 때마다 선택한 구간 파일만 다시 만들며 VLM·BGE를 실행하지 않습니다. Graph bundle 재생성 또는 구간 생성 규칙 변경 후에는 구간 파일도 다시 준비합니다. 구간 파일의 버전·입력 해시·checksum을 검증하고 추천 identity에 포함하므로 파일이 없거나 현재 bundle과 맞지 않으면 학습할 수 없습니다.

추천 결과는 `validation/recommendations/seed_<seed>/<arm>/`에 저장합니다. 최적 validation HR@10의 최초 checkpoint를 복원한 뒤 test를 한 번 평가하며 refit은 없습니다. `training.json`에는 파라미터 수·G1b 교란 통계·G1c의 비활성 Q/K도 기록합니다.

진단은 `validation/diagnosis/<target-set>/diagnosis.json`, 보고서는 `reports/<target-set>/report.md`에 저장합니다. target-set은 등록 순서로 연결합니다. Basic과 전체 진단은 별도 경로입니다. JSON에는 seed별·평균·표준편차·Base 상대 변화율·입력 확보율·검정 정책을 저장하고, 상세 아이템별 증거는 원래 특징·학습 파일에서 확인합니다.

3개 seed 기준 Basic 8개 Arm은 24개 학습 조합, Ablation 4개 Arm은 12개 학습 조합, 전체 12개 Arm은 36개 학습 조합입니다. NDCG@10은 사용자별 seed 평균을 paired bootstrap하고 선택한 모든 Arm 쌍에 Bonferroni 보정합니다(Basic 28쌍, Ablation 6쌍, 전체 66쌍). Ablation만 선택하면 Base 상대 변화율과 G1·G3 대비 비교는 계산하지 않으며, 각 진단은 선택한 Arm 묶음별로 저장합니다. HR@10은 기술통계만 보고합니다. Run 간 비교는 같은 Arm·사용자·이벤트·원본·분할·catalog·seed·평가 계약을 요구합니다. 진단 pass는 실행 증거 검증이며 성능 우위 판정과 구분합니다.

아래 날짜별 항목은 이전 계약의 변경 기록입니다. 현재 실행에는 위 계약과 README를 적용합니다.

## 현재 형식 지원 정리 (2026-10-07)

호출이 없는 `validation/io.py`, 구형 rolling benchmark와 raw Graph 호환 모듈을 제거했습니다. 미사용 빈도·catalog scoring·구형 bootstrap·top-k·로그 헬퍼와 분할 재계산 wrapper를 제거하고 학습 전용 루프를 직접 계산으로 정리했습니다. 실제 provenance의 `selection_hash`, 사용자 단위 bootstrap·Bonferroni, Qwen의 문자열 경로 logits 모듈과 Graph 수치 참조 구현은 유지합니다. NumPy 손실 검증은 현행 `transition_loss`를 검사합니다.

Run 산출물·과거 보고서·프롬프트·v7 fixture는 보존합니다. 모델 구조·손실함수·평가 조건·CLI는 변경하지 않습니다. 실제 Run 데이터와 전체 GPU 실험은 이 VM에서 검증하지 못했습니다.

검사 결과: `vc_cloud`에서 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q` **296개 통과(72.76초)**. `python -m ruff check src tests benchmarks`, `git diff --check`, 현행 Scene GCN benchmark의 `--help`도 통과했습니다. 현재 형식의 생성·저장·재사용·임베딩·추천·진단, 비지원 입력 거부·원본 보존, 3개 seed·6개 Arm, mean/GNN 수치·gradient·최적 checkpoint·직렬/병렬·중단 복구·캐시 검증을 포함합니다.

## 진단 JSON 축소 (2026-10-08)

Text·Graph mean/GNN에 동일한 `leave-one-out-diagnosis/v2` 저장 정책을 적용합니다. `--compare-run-id`의 비교 결과도 참조 Run·방향·지표·수치 비교만 저장하며 sources를 복사하지 않습니다. 최종 점수와 통계 계산, 원본 증거 검증·실패 처리는 유지합니다. 상세 분할은 `validation/cohort/`, 입력 provenance는 표현 manifest·`.inputs/`, seed별 이벤트 지표·학습 기록은 추천 디렉터리에서 확인합니다. 진단 JSON만 생성하는 방식이 바뀌므로 재학습 없이 기존 `run-diagnosis` 명령을 다시 실행하면 축소 형식으로 저장됩니다. 현재 VM에는 실제 Run 데이터가 없어 기존 대형 진단 파일 자체의 축소량은 미검증입니다.

검증: `vc_cloud` CPU·단일 스레드 전체 pytest **302개 통과(71.54초)**, Ruff(`src tests benchmarks`), `git diff --check` 통과. 최종 점수 일치, 상세 목록 크기와 무관한 출력, Text·mean·GNN 및 비정상 증거의 오류 보존을 확인했습니다.

## IISAN·DFF 공통 평가 규칙 (2026-10-08)

validation/test에서는 입력의 최대 10개 제한과 별개로 앞선 보존 이력 전체를 마스킹하며, 반복 정답도 예외를 두지 않습니다. 각 사용자의 전체 catalog 점수를 `torch.argsort(scores, descending=True)`로 정렬합니다. 마스킹된 정답의 순위도 그대로 계산하므로 작은 catalog에서는 top-k 안에 들어갈 수 있습니다. 동률 순서는 환경 간 동일성을 보장하지 않습니다.

`validation.evaluation.checkpoint_metric: HR@10`은 고정이며, 최종 보고·paired bootstrap은 NDCG@10을 유지합니다. 평가 계약 `microlens-leave-one-out-evaluation/v2`는 마스킹·랭킹·checkpoint·보고 지표를 학습 fingerprint, 추천 identity, checkpoint, 완료 manifest, 공유 캐시에 연결합니다. 모델·손실·popularity·오른쪽 padding·학습률은 유지하고 참조 코드의 주기적 test나 조기 종료 조건은 도입하지 않습니다. 논문 전체 재현은 아닙니다.

이전 평가 결과의 일반 재진단은 거부합니다. 같은 Run에서 기존 `run-recommendation`을 실행하면 선택한 seed × Arm의 구형 추천 파일을 교체하고, 새 계약에서 완료된 단위는 재사용합니다. 이어서 `run-diagnosis`를 실행해 진단을 재생성하십시오. 추출·분할·임베딩은 provenance가 일치하면 재사용하며 다른 Run·과거 보고서·구형 공유 캐시는 삭제하지 않습니다. 위 진단 축소 절의 재학습 없는 재진단은 당시 계약 또는 현재 계약 안에서만 해당합니다.
