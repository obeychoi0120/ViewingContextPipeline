# 실행·저장 계약

현행 실행의 입출력·재사용 규칙입니다. 빠른 시작은 [README](../README.md), 관측 스키마는 [장면 추출](scene_extraction.md), 수식·평가는 [추천 학습](graph_training.md)을 참고하세요. 과거 Run은 생성 당시의 정의와 증거로 해석합니다.

## 데이터 입력과 cohort

모든 명령은 `--dataset microlens_100k` 또는 `--dataset m3l_10m`을 받으며 한 번에 한 데이터셋을 처리합니다. `protocol.dataset`은 지원 목록입니다. `config.yaml`의 상대 경로는 저장소 루트 기준이고, 선택하지 않은 데이터셋의 원본 파일은 없어도 됩니다.

### MicroLens

`datasets.microlens_100k.data`에 interaction CSV·검증 TSV·영상 디렉터리·제목 CSV·선택적 보완 제목 CSV를 지정합니다.

- interaction CSV는 `user,item,timestamp` 형식이며 timestamp는 정수 밀리초입니다.
- 영상은 `{item_id}.mp4`입니다. TSV가 있으면 사용자별 interaction multiset을 CSV와 대조합니다.
- 제목은 원본의 비어 있지 않은 값을 우선하고 보완 CSV로 필요한 빈 제목·누락 행만 채웁니다. 보완 CSV가 있어도 해결되지 않은 제목은 빈 문자열로 남습니다. 보완 CSV가 없을 때 제목 행 누락은 오류입니다.
- 원본 CSV는 수정하지 않으며 `source_counts`로 원본 개수를 검증합니다.

### M³L-10M

`datasets.m3l_10m.data.root` 아래 다음 파일을 사용합니다.

```text
metadata/ratings.dat
metadata/movies.dat
metadata/id_mmrec_to_grouplens.tsv
videos/{original_GroupLens_movieId}.mp4
```

ID 매핑의 영화로 원본 MovieLens 평점을 제한하고 `source_counts`를 검증합니다. 원본 사용자·영화 ID를 유지하며 평점 timestamp(초)를 밀리초로 바꿉니다. 제목은 `movies.dat`의 연도 포함 제목이고 장르·줄거리는 사용하지 않습니다.

모든 평점을 상호작용으로 사용하며 평점 크기를 가중치로 사용하지 않습니다. 영상이 없거나 영상 스트림·양의 재생시간을 확인할 수 없는 영화와 그 상호작용은 제외합니다. `ffprobe` 누락·실행 권한 같은 환경 오류는 준비를 중단합니다. 이는 **평가한 영화 이력의 다음 영화 예측**이며 공식 MMRec 분할 재현은 아닙니다.

`prepare-cohort`는 `cohort_plan.json`에 원본·제외 후 통계와 입력 fingerprint를, `excluded_items.jsonl`에 제외 영화·사유·상호작용 수를 저장합니다. 나중에 확보한 영상은 기존 cohort에 자동 추가되지 않습니다. 확보 목록을 바꾸면 cohort와 영향받는 후속 산출물을 다시 준비합니다. [M³L 입력 처리](../src/preparation/m3l.py).

### 공유 준비와 Run별 분할

`prepare-cohort`는 원본 데이터·영상·제목을 점검해 공유 preparation cohort를 만듭니다. `prepare-input-data`는 영상의 시간 창·키프레임을 준비합니다. 준비 결과는 데이터셋별로 공유하므로 다른 원본 corpus를 사용할 때는 별도 `artifacts_root`를 사용합니다.

Run별 `validation/cohort/`는 원본 이력 5개 이상 사용자의 최근 13개를 보존하고 마지막 두 이벤트를 validation/test로 보류합니다. 후보 catalog는 보존된 시퀀스의 아이템입니다. 초기 이벤트 순서·catalog·분할·입력 해시를 검증하며 실험 추가·삭제만으로 이 cohort를 바꾸지 않습니다. [공유 cohort](../src/preparation/cohort.py) · [Run별 선택](../src/validation/selection.py).

## 설정 로딩과 실행 자원

`config.yaml`은 데이터·모델·추출·학습·평가 설정이고, `rec_experiments_config.yaml`은 추천 실험 목록입니다. validation 명령은 추천 파일 전체를 YAML 순서로 처리하며 `--target`, `--experiments`는 받지 않습니다. 명령 시작에 한 번 해석한 구성을 worker에도 전달하며 파일 수정은 다음 명령부터 적용됩니다. 추출·전처리는 추천 파일 없이 실행할 수 있습니다.

Qwen은 로컬 checkpoint와 vLLM으로 보이는 GPU별 worker를 실행합니다. `CUDA_VISIBLE_DEVICES`가 없으면 보이는 GPU 전체를 사용하고 GPU가 없으면 실패합니다. Gemini의 장면 동시성은 `extraction.gemini.threads`입니다. 추천의 `--workers-per-gpu`는 GPU당 독립 조합 process 수이며 분산 학습으로 한 모델을 나누는 설정은 아닙니다. GPU가 없으면 값 1에서 CPU를 사용합니다. `--gpus` 옵션은 없습니다.

추출 토큰 한도는 `extraction.{graph,description}.{qwen3vl,gemini}.scene_max_new_tokens`, 생성 정책·옵션은 [장면 추출](scene_extraction.md#qwen-생성-정책과-옵션)을 따릅니다. `validation.graph_execution`의 특징 캐시·chunk·checkpoint는 [학습 자원 설명](graph_training.md#공동-학습과-실행-자원)을 참고하세요.

## 산출물 경로

```text
<artifacts_root>/
├── preparation/<dataset>/
│   ├── cohort/leave_one_out/
│   ├── resized_keyframes/<width>_<height>/<content_id>/
│   └── source_assets/
├── runs/<RUN_ID>/<dataset>/
│   ├── extraction/<source>/
│   ├── validation/
│   │   ├── cohort/
│   │   ├── representations/<source>/
│   │   ├── recommendations/seed_<seed>/<label>/
│   │   └── diagnosis/<experiment-set>/diagnosis.json
│   └── reports/
│       ├── leave_one_out_cohort.json
│       ├── leave_one_out_cohort.md
│       └── <experiment-set>/report.md
└── shared_cache/<dataset>/
    ├── scene_inputs/
    └── recommendations/
```

`<experiment-set>`은 YAML의 label 목록을 순서대로 밑줄로 연결합니다. source는 `metadata` 또는 등록된 시각 source입니다. 같은 source의 고정 BGE 특징은 공유하지만 학습 파라미터는 실험별로 독립적입니다.

키프레임 경로는 `extraction.visual_evidence.image_resolution`으로 정합니다. 해상도를 바꾸면 다른 디렉터리에 준비하고 같은 해상도로 돌아오면 기존 프레임을 재사용합니다. 구형 `resized_keyframes/<content_id>/` 경로를 자동으로 이동하거나 읽지는 않습니다.

자동 전처리·진단 보고서는 위의 Run·데이터셋별 `reports/`에 저장합니다. 별도 작성하는 Run 분석 코드·보고서·표·이미지는 [AGENTS.md](../AGENTS.md)에 따라 `artifacts/runs/<RUN_ID>/reports/`에 저장합니다. 일반 설계 문서는 `docs/`에 둡니다.

## 장면 저장과 추적 정보

Graph·Description·실패는 source 디렉터리의 `<content_id>.jsonl`에 저장하며 행 필드는 [장면 추출 문서](scene_extraction.md#저장과-실패-확인)를 따릅니다. `(content_id, scene_idx)`당 최신 결과 한 행을 유지하고 정렬·임시 파일 작성 후 원자적으로 교체합니다. 재시도 결과가 도착하지 않은 이전 warning·실패 행은 보존합니다.

기존 `failures/<content_id>.jsonl`이 있으면 추출 재개 준비에서 원본 JSONL에 통합하고 저장 성공 후 해당 실패 파일을 제거합니다. 양쪽에 같은 장면이 있으면 기존 실패 로그를 우선하고 누락 토큰 수는 `null`로 둡니다. 새 실패 디렉터리는 만들지 않습니다.

진행바의 `success`·`warning`·`failed` 합으로 완료 수·처리량·ETA를 계산합니다. Graph의 `generation_stats.json`은 현재 대상의 재사용·새 결과 전체와 경고 종류별 수를 기록합니다. 실패 원문은 보존하지만 임베딩·Graph 구성에는 사용하지 않습니다.

`generation_identity.json`은 초기 생성 설정 기록입니다. 기존 기록은 보존하고 없으면 현재 설정을 기록하며, 이 파일의 설정 차이로 정상 장면 재사용을 막지는 않습니다. 따라서 폴더 내 모든 장면의 생성 조건이 동일함을 보장하지 않습니다. 생성 옵션을 바꿔 재개하면 서로 다른 조건이 섞일 수 있습니다.

지원 밖 저장 형식을 자동 추론·변환하지 않습니다. 구형 v8 `location` 읽기 호환 등 명시적 지원 범위는 [파싱 계약](scene_extraction.md#파싱과-지원-경계)을 따릅니다. 기존 Summary 파일·실패 로그와 `run_contract.json`은 현행 실행에서 읽거나 수정하지 않으며 새 `run_contract.json`을 요구하지 않습니다.

## 재실행과 캐시

| 단계·상황 | 일반 재실행 | `--force` |
| --- | --- | --- |
| 정상 장면 | 생성 옵션이 바뀌어도 재사용 | 해당 추출 source의 정상 결과까지 제거 후 재생성 |
| 누락·실패·Graph warning 장면 | 현재 조건으로 한 번씩 생성. 자동 추가 pass 없음 | 전체 source 재생성에 포함 |
| 고정 특징 준비 | 유효한 로컬 bundle → 공유 캐시 → 새 계산 | 캐시 읽기를 우회해 재계산 |
| 활동 구간 준비 | 필요한 source의 구간 파일을 다시 생성 | 같은 준비 경로를 수행 |
| 완료 추천 조합 | 입력·구성·학습·평가 조건 검증 후 재사용 | 호환되는 조합을 다시 학습 |
| 미완료 추천 조합 | 처음부터 학습. epoch 중간 재개 없음 | 같은 동작 |
| 진단 | 저장된 학습 증거를 검증하고 보고서 생성 | 다른 모델 정의의 결과를 유효하게 만드는 우회 기능이 아님 |

Scene은 다른 Run에서 자동으로 가져오지 않습니다. 특징·추천 공유 캐시는 키별 잠금·checksum·원자적 게시를 사용합니다. 특징 키에는 Scene 파일 바이트·생성 기록, catalog·제목·BGE 모델·인코딩 설정·변환 계약을 포함합니다.

추천 키에는 개별 실험의 해석된 정의·모듈 버전·표현 해시·분할·학습 조건·seed·구간 checksum을 포함합니다. 다른 실험의 추가·삭제만으로 동일 정의의 결과를 무효화하지 않습니다. label은 결과 경로와 식별에도 포함되므로 이름을 바꾼 결과를 자동으로 재분류하지 않습니다.

추천은 완료된 seed × 실험 단위로 `sasrec.pt`, `training.json`, `per_event_metrics.jsonl`, `catalog_vectors.npy`, `complete.json`을 검증합니다. Run 간 캐시 재사용은 독립 파일로 복사하고 현재 identity와 `reused_from`을 기록합니다. 완료 조합은 모델·worker 초기화 없이 건너뜁니다. `identity.json`은 학습 시작 시 기록해 부분 결과도 보호합니다.

같은 label의 기존 경로에 다른 실험 정의·모델 구조가 있으면 쓰기 전에 실패시키고 새 Run ID를 안내합니다. `--force`도 이 보호를 우회하지 않습니다. 같은 정의의 학습 조건을 바꿔 다시 학습하면 해당 조합과 진단 결과를 다시 생성하므로 이전 결과를 보존하려면 새 Run을 사용합니다. [추천 캐시](../src/validation/recommendation_cache.py) · [결과 호환성 검사](../src/validation/leave_one_out_recommendation.py).

Graph 특징 bundle이나 구간 생성 규칙이 바뀌면 `prepare-module`을 다시 실행합니다. 구간 파일의 계약·입력·생성 코드 해시·checksum은 추천 identity에 연결됩니다. 추출·BGE를 다시 실행할 필요는 없습니다.

## 학습 결과와 진단 필드

| 파일·필드 | 의미 |
| --- | --- |
| `sasrec.pt` | 선택된 checkpoint의 `state_dict`와 metadata |
| `training.json` | validation 이력, best epoch, 학습 조건·파라미터 수·실행 통계·환경 정보 |
| `per_event_metrics.jsonl` | test 이벤트별 순위·지표·결과 identity |
| `catalog_vectors.npy` | 선택된 모델의 catalog 영상 벡터 |
| `complete.json` | 완료 identity·이벤트 수·결과 파일 checksum |
| 진단 `experiments` | label·source·modules·readout·모듈 버전 |
| `recommendations.means`, `standard_deviations`, `per_seed` | 최종 test의 평균·표본 표준편차·seed별 값 |
| `recommendations.baseline`, `base_relative_changes` | metadata source의 label과 상대 변화율. 기준이 없으면 `null` |
| `statistics.comparisons` | 전체 쌍의 NDCG 차이·보정 신뢰구간. 대상은 `left`, `right` 필드로 확인 |
| `runtime_decision` | 실행 증거의 검증 판정과 오류 |

진단은 `leave-one-out-diagnosis/v6`입니다. label에 하이픈이 있어도 비교 대상은 명시적 필드로 읽고 이름을 분할해 추측하지 않습니다. 보고서는 기록된 구성과 모든 쌍을 표시하며 특정 label에 비교 의미를 고정하지 않습니다.

checkpoint 선택은 validation HR@10, 최종 보고·paired bootstrap의 주 지표는 test NDCG@10입니다. 통계는 사용자별 seed 평균·전체 쌍 Bonferroni를 사용하고 HR@10은 기술통계입니다. 상세 규칙은 [평가](graph_training.md#평가와-결과-해석)를 참고하세요.

진단 pass는 실행 증거 검증이며 성능 개선·관측 정확도의 증명이 아닙니다. `--compare-run-id`는 실험 정의·사용자·이벤트·원본·catalog·분할·seed·평가 계약의 호환성을 검사합니다. 전체 입력·분할·provenance는 cohort·표현·추천 원본 파일에서 확인합니다.

## 결정적 실행과 재현성

`validation.model.deterministic`은 [seed 설정 함수](../src/validation/model.py)의 PyTorch 결정적 연산을 제어합니다. `true`, `false` 모두 실험·seed 시작마다 Python·NumPy·PyTorch seed를 고정하며 배치 순서는 별도의 seeded NumPy Generator로 만듭니다. `true`는 결정적 연산을 강제하고 지원하지 않는 비결정적 연산에서는 오류를 발생시킵니다.

CUDA 행렬 연산을 위한 `CUBLAS_WORKSPACE_CONFIG=:4096:8`은 미지정 시 설정하며 외부 지정값은 유지합니다. 하드웨어·드라이버·PyTorch/CUDA 버전·입력·학습 조건까지 일치시키는 기능은 아닙니다. 모델·입력·seed·환경을 고정해 새로 학습한 가중치·벡터·순위를 비교하는 검증과 완료 결과 재사용을 구분합니다.

worker별 조합은 독립 process이고 조합 시작에 seed를 재설정합니다. 다만 작업 배정 GPU와 `checkpoint: auto`의 분기는 실행 시점의 자원 상태에 영향을 받습니다. 여러 seed 간 변동은 `true`에서도 남으며, 설정 변경 전후의 성능 차이는 같은 조건의 기준·추가 모델끼리 비교해야 합니다.

## 현재 계약과 과거 결과

설정·추출의 `experiment_config_version: v6`와 추천 실험 YAML의 `version: 1`은 서로 다른 버전입니다. 추천 identity의 `gnn-attn-aggregation/v1`은 식별자로 유지하며 현재 장면 Attention이나 GCN 사용을 뜻하지 않습니다. 모델 구조는 `sasrec-scene-content/v4`, 활동 구간은 `scene-segments/v2`, 평가는 `microlens-leave-one-out-evaluation/v2`입니다.

이전 Arm 이름·Summary 임베딩·다른 모델 구조의 checkpoint를 새 실험 결과로 변환하지 않습니다. 입력·checksum·계약이 일치하는 장면·고정 특징은 재사용할 수 있습니다. 과거 보고서와 Run은 생성 당시 정의를 보존합니다.

### 기존 MicroLens 디렉터리 이전

데이터셋 구분이 없던 디렉터리는 현재 실행 전에 이전합니다. 해당 artifacts 루트를 사용하는 작업을 종료한 뒤 먼저 미리보기를 확인합니다.

```bash
python -m preparation.migrate_dataset_layout --artifacts-root artifacts
python -m preparation.migrate_dataset_layout --artifacts-root artifacts --apply
```

기존 preparation·Run·공유 캐시를 `microlens_100k` 하위로 이동하며 영상·모델 원본은 이동하지 않습니다. content ID·이벤트·분할·특징·checkpoint 해시와 기존 `run_contract.json` 내용은 유지합니다. 대상이 있으면 병합·덮어쓰지 않으며 `dataset_layout_migration.json`으로 중단 후 재개합니다. 과거 보고서 안의 경로는 당시 기록으로 남습니다. [이전 도구](../src/preparation/migrate_dataset_layout.py).

<details>
<summary>과거 계약·검증 기록</summary>

다음 기록은 작성 당시 코드와 계약 기준입니다. 현재 실험 목록·CLI·스키마·검증 개수와 구분하며 현재 실행에는 위 계약을 적용합니다.

## 과거 검증 기록: 형식 지원 정리 (2026-10-07)

호출이 없는 `validation/io.py`, 구형 rolling benchmark와 raw Graph 호환 모듈을 제거했습니다. 미사용 빈도·catalog scoring·구형 bootstrap·top-k·로그 헬퍼와 분할 재계산 wrapper를 제거하고 학습 전용 루프를 직접 계산으로 정리했습니다. 실제 provenance의 `selection_hash`, 사용자 단위 bootstrap·Bonferroni, Qwen의 문자열 경로 logits 모듈과 Graph 수치 참조 구현은 유지합니다. NumPy 손실 검증은 현행 `transition_loss`를 검사합니다.

Run 산출물·과거 보고서·프롬프트·v7 fixture는 보존합니다. 모델 구조·손실함수·평가 조건·CLI는 변경하지 않습니다. 실제 Run 데이터와 전체 GPU 실험은 이 VM에서 검증하지 못했습니다.

검사 결과: `vc_cloud`에서 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q` **296개 통과(72.76초)**. `python -m ruff check src tests benchmarks`, `git diff --check`, 현행 Scene GCN benchmark의 `--help`도 통과했습니다. 현재 형식의 생성·저장·재사용·임베딩·추천·진단, 비지원 입력 거부·원본 보존, 3개 seed·6개 Arm, mean/GNN 수치·gradient·최적 checkpoint·직렬/병렬·중단 복구·캐시 검증을 포함합니다.

## 과거 검증 기록: 진단 JSON 축소 (2026-10-08)

Text·Graph mean/GNN에 동일한 `leave-one-out-diagnosis/v2` 저장 정책을 적용합니다. `--compare-run-id`의 비교 결과도 참조 Run·방향·지표·수치 비교만 저장하며 sources를 복사하지 않습니다. 최종 점수와 통계 계산, 원본 증거 검증·실패 처리는 유지합니다. 상세 분할은 `validation/cohort/`, 입력 provenance는 표현 manifest·`.inputs/`, seed별 이벤트 지표·학습 기록은 추천 디렉터리에서 확인합니다. 진단 JSON만 생성하는 방식이 바뀌므로 재학습 없이 기존 `run-diagnosis` 명령을 다시 실행하면 축소 형식으로 저장됩니다. 현재 VM에는 실제 Run 데이터가 없어 기존 대형 진단 파일 자체의 축소량은 미검증입니다.

검증: `vc_cloud` CPU·단일 스레드 전체 pytest **302개 통과(71.54초)**, Ruff(`src tests benchmarks`), `git diff --check` 통과. 최종 점수 일치, 상세 목록 크기와 무관한 출력, Text·mean·GNN 및 비정상 증거의 오류 보존을 확인했습니다.

## IISAN·DFF 공통 평가 규칙 (2026-10-08)

validation/test에서는 입력의 최대 10개 제한과 별개로 앞선 보존 이력 전체를 마스킹하며, 반복 정답도 예외를 두지 않습니다. 각 사용자의 전체 catalog 점수를 `torch.argsort(scores, descending=True)`로 정렬합니다. 마스킹된 정답의 순위도 그대로 계산하므로 작은 catalog에서는 top-k 안에 들어갈 수 있습니다. 동률 순서는 환경 간 동일성을 보장하지 않습니다.

`validation.evaluation.checkpoint_metric: HR@10`은 고정이며, 최종 보고·paired bootstrap은 NDCG@10을 유지합니다. 평가 계약 `microlens-leave-one-out-evaluation/v2`는 마스킹·랭킹·checkpoint·보고 지표를 학습 fingerprint, 추천 identity, checkpoint, 완료 manifest, 공유 캐시에 연결합니다. 모델·손실·popularity·오른쪽 padding·학습률은 유지하고 참조 코드의 주기적 test나 조기 종료 조건은 도입하지 않습니다. 논문 전체 재현은 아닙니다.

이전 모델 구조의 결과는 새 Arm 결과로 재사용하거나 재진단하지 않습니다. 같은 추천 경로에 이전 구조 결과가 있으면 새 Run을 사용해야 하며 `--force`도 덮어쓰지 않습니다. 학습 시작 시 `identity.json`을 기록해 부분 결과도 보호합니다. 동일 구조에서 완료된 단위는 재사용하며, 입력·학습 설정 변경은 결과 identity를 무효화합니다. 추출·BGE 특징은 provenance가 일치하면 재사용합니다.

진단은 선택한 Arm 전체 쌍에 Bonferroni 보정을 적용합니다. 5개 전체는 10쌍이며 NDCG@10 사용자 단위 paired bootstrap을 사용합니다. 마지막 콘솔에 NDCG@10·HR@10 평균 ± 표준편차를 출력합니다. G0−D, G1−G0, G2−G1을 우선 표시합니다.

</details>
