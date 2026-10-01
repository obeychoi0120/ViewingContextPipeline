# 직접 Graph 학습 경로

`text`는 기존 요약 텍스트 BGE 경로이고, `graph`는 Scene Graph의 구조를 추천 loss로 학습하는 새 경로입니다. 세 validation 명령 모두 `--representation-mode`가 필수입니다. Graph 추천·진단에는 `--scene-aggregation mean|attention`도 필수이며 embedding 단계나 text 모드에서는 사용할 수 없습니다.

## 실행

로컬 BGE 가중치와 torch·transformers가 있는 환경에서 실행합니다. 필요한 Python 의존성은 `pip install -e ".[train,embedding]"`로 설치할 수 있습니다. 4개 GPU가 보이도록 `CUDA_VISIBLE_DEVICES=0,1,2,3`을 설정하면 BGE는 고유 문자열을 결정적으로 분할하고, 추천은 GPU당 독립 조합 하나를 실행합니다. `--workers-per-gpu` 기본값은 1입니다. 단일 모델 DDP는 사용하지 않습니다.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m validation embed-representations --run-id 260928_v7 --representation-mode graph --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-recommendation --run-id 260928_v7 --representation-mode graph --scene-aggregation mean --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-diagnosis --run-id 260928_v7 --representation-mode graph --scene-aggregation mean --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-recommendation --run-id 260928_v7 --representation-mode graph --scene-aggregation attention --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-diagnosis --run-id 260928_v7 --representation-mode graph --scene-aggregation attention --target meta graph_qwen graph_qwen_meta graph_gemini_meta
```

동일 순서를 한 번에 실행하려면 저장소 루트에서 `bash script_graph_v7.sh`를 사용합니다. 모듈 1은 실행하지 않습니다.

위 추천 명령은 각각 **7일 × 3 seed × 4 Arm = 84개 조합**의 전체 실험입니다. 완료된 조합은 체크섬과 입력 해시를 검증해 재사용하고, 중단된 조합은 초기화부터 다시 실행합니다. `--force`는 완료 조합도 재계산합니다. 실패한 epoch 중간 상태에서 이어 학습하는 방식은 아닙니다.

기존 text 명령에는 `--representation-mode text`만 추가합니다. Description Arm은 text에서만 지원합니다. `graph`의 `meta`는 text와 같은 제목 1024→512 projection + LayerNorm 기준선이며 Graph encoder를 사용하지 않습니다.

## 모델 및 데이터 계약

- BGE 입력: entity의 종류·attributes, action 문구, Context의 medium·format·topics, 필요한 Arm의 영문 제목. 고유 문자열만 인코딩하며 float32 특징은 학습하지 않습니다.
- entity/action 노드 사이 actor·target·tool·location별 양방향 메시지를 1층·128차원에서 전달합니다. `none`/`unknown`은 역할별 서로 다른 상태 벡터이며 공유 개체 노드가 아닙니다.
- entity 평균 128, action 평균 128, projection한 Context 128을 단순 결합해 장면 384차원을 만듭니다. 영상 mean 또는 attention pooling은 순서 불변이며 시간 위치 정보와 장면 간 메시지는 사용하지 않습니다.
- `_meta`는 제목 BGE 1024를 Linear 1024→128로 변환하고 `[제목 128; Graph 384]`를 결합한 뒤 LayerNorm(512)를 적용합니다. 제목 및 장면 concat 직후 별도 LayerNorm과 아이템 MLP는 없습니다. Role Graph Encoder와 SASRec 내부 정규화는 유지합니다.
- 누락된 제목·영상 부분은 최종 LayerNorm 전 영벡터입니다. 부분 결측 영역이 정규화 이후에도 0일 필요는 없습니다. 전체 결측과 padding은 LayerNorm 이후 다시 0으로 마스킹하며 후보·interaction은 제거하지 않습니다. `meta`는 제목 1024→512→LayerNorm을 사용합니다.
- raw·warning·구조 오류 장면은 제외하고 장면별 사유를 저장합니다. 행동이 없는 정상 장면과 고립 entity는 유지합니다. 필수 참조는 검사하고 receiver는 기존 데이터에 있어도 무시하고 location 누락은 `unknown`으로 해석합니다. 모델 텐서의 receiver 슬롯은 형태 호환성을 위해 남기되 항상 `none` 상태로 두고 해당 간선은 만들지 않습니다. 입력 버전을 갱신해 receiver를 사용한 기존 특징 캐시를 재사용하지 않습니다. JSONL 손상·ID 불일치·중복 장면 번호는 중단합니다.
- 개체 ID는 장면 내부 연결에만 쓰며 BGE 입력에 넣지 않습니다. 개체·행동 수나 장면 수를 프롬프트 상한에 맞춰 자르지 않습니다. 가변 길이 배열을 쓰며 영상 단위로 분할 계산합니다. 기본 분할 크기는 256개 영상·32,768개 노드이며 이를 넘는 단일 영상도 그대로 유지합니다. Checkpoint는 아래 실행 설정에 따라 사용하며, 배치 조립과 전송은 backward에서 반복하지 않습니다.
- SASRec의 loss, 인기도 보정, 후보 마스킹, 이력 길이 10, 차원 512, 날짜별 selection/refit/test는 기존과 같습니다. 각 refit은 모든 학습 모듈을 초기화합니다. 학습 배치의 이력·정답 아이템 합집합만 계산하고 optimizer 갱신 후 캐시를 버립니다.

## 산출물과 재현

Run 루트는 `artifacts/runs/<RUN_ID>/`입니다.

| 위치 | 내용 |
|---|---|
| `validation/representations/graph/<arm>_embeddings/` | memory-mapped BGE·노드·연결·offset·제목 배열, manifest, statistics |
| `validation/recommendations/graph/<mean 또는 attention>/<date>/seed_<seed>/<arm>/` | 전체 모듈 checkpoint, 최종 catalog_vectors.npy, training, per-event metrics, complete |
| `validation/diagnosis/graph_<mean 또는 attention>_diagnosis.json` | 동일 모드 내 paired 통계와 장면 제외·결측·제목 사용 집계 |

`catalog_vectors.npy`의 행 순서는 Graph 입력 manifest의 catalog와 같습니다. padding 행은 저장하지 않습니다. checkpoint를 읽을 때는 동일 해시의 Graph 입력으로 `new_graph_model`을 생성하고 state_dict를 읽습니다. BGE 특징은 checkpoint에 중복 저장하지 않습니다.

캐시는 실제 Scene Graph 파일 바이트·catalog 순서·제목·BGE 설정/로컬 모델 식별 정보·변환 버전으로 구분합니다. 추천 결과에는 특징 해시, Graph 구조 설정, aggregation과 기존 학습 해시·구간·seed가 들어갑니다. 변환/모델 의미를 변경하면 해당 버전도 갱신해야 합니다. Text v4는 분리 특징 배열과 새 캐시 키를 사용하며 저장 위치는 `validation/representations/text/`, `validation/recommendations/text/`, `validation/diagnosis/text_diagnosis.json`입니다.

`--compare-run-id`는 선택한 모드와 aggregation의 상대 Run 결과만 읽습니다. 다른 모드 또는 pooling의 결과를 대신 읽지 않습니다. 방식 간 자동 paired 비교와 장면 간 GNN은 [TODO](TODO.md)에 기록했습니다.

검증 명령: `PYTHONPATH=.:src python -m pytest tests/validation/test_direct_graph.py -q`. 실제 BGE 모델 호출을 대체한 작은 CPU 검증입니다. 전체 실험 성능 및 RTX 6000 Ada 자원 측정은 별도 수행해야 합니다.

경로 변경 이전의 Run 내부 산출물은 자동 이동하지 않습니다. 새 경로에서 같은 명령을 실행하면 입력 검증 후 공유 캐시의 유효한 결과를 복구하며, 캐시가 없으면 재계산합니다.

## 학습 비용 조정 (2026-09-30)

공통 `validation.model` 설정을 학습률 `3e-4`, batch size `512`, patience `3`,
`min_delta=1e-4`로 변경했습니다. 이 네 설정은 text와 graph 모두에 적용됩니다.
Graph encoder는 2층을 유지하고 은닉 차원만 256에서 128로 줄였습니다.
장면·추천 공간은 512차원이며, 끝까지 공동 학습하고 selection/refit/test를 유지합니다.
학습률 scheduler나 Graph 동결은 추가하지 않았습니다.

`min_delta`는 validation NDCG@10의 절대값 기준입니다. 마지막 유의미한 개선점보다
`1e-4`를 초과해 개선해야 patience를 초기화하며, 그 사이의 작은 개선은 누적될 수 있습니다.
refit epoch는 이 기준과 별개로 실제 최고 NDCG를 기록한 epoch를 선택합니다.
동점이면 먼저 기록한 epoch를 유지합니다. 키가 없는 이전 설정은 `min_delta=0`으로 해석합니다.

Graph 모델 계약은 `sasrec-role-graph/v2`, 학습 구현 계약은
`shared-scenes-training-evaluation/v4`입니다. 이전 추천 학습 결과는 새 설정의 완료 결과로
재사용하지 않습니다. 고정 BGE·Graph 입력은 재사용할 수 있습니다.
실행 중인 프로세스에는 변경이 자동 적용되지 않으며 다음 실행부터 적용됩니다.
속도 및 추천 성능 개선은 아직 전체 실험으로 검증하지 않았습니다.

## Graph 실행 최적화 (2026-09-30)

`validation.graph_execution`은 학습 알고리즘과 별개인 실행 설정입니다. 생략해도 다음 기본값을 사용합니다.

```yaml
validation:
  graph_execution:
    chunk_items: 256
    chunk_nodes: 32768
    checkpoint: auto
    feature_cache_mb: 2048
```

- 연결을 역할별로 작업 시작 때 정리합니다. 영상·장면·노드별 Python 조립 루프 대신 연속 구간을 일괄 gather하고, 작은 인덱스 배열을 합쳐 GPU로 전송합니다.
- 역할별 edge·예약값 인덱스, 메시지 개수와 노드 그룹을 배치 준비 때 계산하므로 각 GNN 층에서 GPU boolean filtering을 반복하지 않습니다.
- mean은 영상 ID별 합/개수로, attention은 영상 ID별 안정화 softmax/가중합으로 일괄 계산합니다. 장면 삭제, 시간 순서 정보 또는 새 연결은 추가하지 않습니다.
- `checkpoint: auto`는 전체 학습 배치의 노드·edge·장면 수로 활성값 메모리를 추정합니다. CUDA에서 추정치가 `min(8 GiB, 현재 여유 메모리의 35%)`를 넘으면 checkpoint를 적용하고, 그렇지 않으면 재계산을 생략합니다. CPU의 auto는 재계산하지 않습니다. 메모리 사용량을 엄격히 제한해야 하면 `always`, 재계산을 끄려면 `never`를 사용합니다. auto는 추정 규칙이며 최대 메모리를 보장하는 제한값은 아닙니다.
- Checkpoint의 대상은 이미 준비한 입력에 대한 신경망 연산뿐입니다. 배치 조립·고정 특징 조회·GPU 전송은 다시 실행하지 않습니다.
- 고정 BGE 특징 사전 전체가 `min(feature_cache_mb, 현재 여유 GPU 메모리의 20%)`에 들어가면 GPU에 유지합니다. 들어가지 않거나 할당에 실패하면 필요한 특징을 CPU에서 일괄 조회·전송합니다. `feature_cache_mb: 0`으로 GPU 특징 캐시를 끌 수 있습니다. 전체 topology의 GPU 상주나 비동기 prefetch는 이번 구현에 포함하지 않습니다.
- 학습 이력·정답 ID 합집합은 원래 CPU 데이터에서 만들고, 학습된 아이템 벡터는 기존처럼 optimizer 갱신 후 폐기합니다. 평가 catalog는 같은 모델 상태에서 재사용하며 저장 직전에 재계산하지 않습니다.

실행 버전은 `packed-role-graph/v1`이며 `training.json`의 `execution.graph_selection`과
`execution.graph_refit`에 설정·checkpoint 적용 배치 수·chunk 수·GPU 특징 캐시 크기를 기록합니다.
`chunks`는 해당 모델의 학습 및 평가 호출을 합친 수이며, checkpoint/direct 배치 수는 학습 호출만 셉니다.

아래 내용은 v2 실행 최적화 당시의 호환성입니다. 이후 v3 모델 변경의 재학습 범위는 다음 절을 참고하십시오.

v2 실행 최적화에서는 파라미터 구조, loss, seed, 학습률·early stopping·refit과 입력 해시는 유지했습니다.
실행 설정은 학습 캐시 키를 바꾸지 않으므로 **동일한 학습 설정의 완료 조합과 기존 checkpoint는 재사용**합니다.
부동소수점 연산 묶음/합산 순서가 바뀌므로 비트 단위 일치 대신 출력·gradient·점수 오차와 테스트 순위를 검증합니다.
전체 실제 데이터의 추천 성능 및 RTX 6000 Ada 처리량은 아직 검증하지 않았습니다.

CPU 회귀 검사:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src python -m pytest tests/validation/test_graph_execution.py tests/validation/test_direct_graph.py -q
```

v2 당시의 측정 코드·결과는 [실행 최적화 보고서](../artifacts/runs/260928_v7/reports/graph_execution_optimization.md)에 보존했습니다.
v3에서는 가중치 구조가 달라졌으므로 당시 v2 비교 스크립트를 현재 모델에 그대로 적용할 수 없습니다.
현재 실행의 병목은 아래 `--profile-every` 옵션으로 측정하십시오.

## 이전 Graph 모델 v3: 1층 encoder와 128+384 결합 (2026-09-30)

아래는 이전 v3의 변경 이력입니다. 현재 계약은 v4이며 이 절의 구조·재개 설명은 과거 버전에 해당합니다. 당시 모델 계약은 **`sasrec-role-graph/v3`**였습니다.

| 부분 | 이전 v2 | 현재 v3 |
|---|---|---|
| 노드 특징 | 개체별·행동별 BGE 1024 → 공유 projection 128 + 종류 embedding | 동일 |
| 역할별 메시지 전달 | 2층, hidden 128 | **1층, hidden 128** |
| 장면 표현 | 개체/행동/Context 384 → Linear 512 + LayerNorm | **concat 384 + LayerNorm**, 확대 없음 |
| mean/attention | S × 512 → 512 | **S × 384 → 384** |
| 제목 | BGE 1024 그대로 결합 | **Linear 1024→128 + LayerNorm** |
| 영상 결합 | `[제목 1024; Graph 512]` → Linear 1536→512 | **`[제목 128; Graph 384]` = 512**, 변환층 제거 |
| 추천 | residual MLP 512 → SASRec 512 | 동일 |

Context projection은 기존처럼 개체·행동 입력 projection과 가중치를 공유하지만 메시지 전달에는
참여하지 않습니다. 제목 projection은 별도 가중치입니다. None/unknown 역할 상태, 방향별 관계,
결측 영상의 영벡터 처리, 이력 길이·loss·selection/refit/test는 유지합니다.
Attention 점수 MLP는 384→128→1이며, 장면 순서 정보나 장면 간 GNN은 추가하지 않습니다.

- **BGE는 여전히 `bge-large-en-v1.5` / 1024차원**입니다. Small 전환은 반영하지 않았습니다.
- Scene 추출·고정 BGE 특징·Graph 연결 입력 계약은 그대로여서 `embed-representations`를 다시 실행할 필요가 없습니다.
- **추천 학습은 다시 필요합니다.** Graph 모델 버전과 모델 설정이 추천 identity/cache key에 포함되어,
  v2 완료 결과와 checkpoint는 v3 결과로 재사용되지 않습니다. `--force` 없이 실행해도 구버전 조합은 재학습합니다.
- 현재 Graph 결과 계약은 `meta`도 같은 Graph 버전으로 묶으므로 Graph 모드 `meta`의 결과 identity 역시
  갱신됩니다. `meta`의 계산 구조·초기화는 그대로이며 Text 모드 결과와 입력 계약은 변경하지 않았습니다.
- 추천 결과 경로는 그대로 사용하므로, 해당 조합을 실제 재학습하면 Run 내부의 구버전 결과를 대체합니다.
  이번 구현 작업에서는 전체 실험을 실행하거나 기존 Run 결과를 재생성하지 않았습니다.
- 새로운 모델의 실제 추천 성능·전체 epoch 속도는 아직 미측정입니다. 이전 v2 실행 최적화 벤치마크
  수치를 v3의 개선 수치로 해석하지 않습니다.
- 현재 구조 그림: [PPTX](design/recsys_diagram.pptx), [PNG](design/recsys_diagram.png).

## 추천 학습 병목 프로파일링

`run-recommendation`은 기본 로그에 selection epoch의 학습·validation 시간과 학습 events/s,
refit epoch별 시간·events/s, 최종 test 시간을 표시합니다. 더 자세한 측정은
**`--profile-every N`**으로 켭니다. text/graph 모두 지원하며 다른 step에서는 거부합니다.

```bash
python -m validation run-recommendation \
  --run-id 260928_v7 --representation-mode graph --scene-aggregation mean \
  --target graph_qwen --workers-per-gpu 1 --profile-every 100
```

각 selection/refit epoch와 validation/test에서 **첫 3개 배치 및 N의 배수 배치**를 측정합니다.
`--profile-every 1`은 모든 배치를 측정합니다. 모델 초기화·학습 입력 사전 준비·평가 catalog 생성·
최종 catalog/checkpoint 저장은 프로파일링이 켜져 있으면 매번 측정합니다.
각 평가의 catalog 생성은 배치별 추천 점수 계산과 별도 레코드입니다.

| 로그 키 | 측정 대상 |
|---|---|
| `input_prepare` | 이력·정답 준비 및 GPU 전송, 학습 Graph ID 목록 구성 |
| `graph_items` | 해당 배치의 중복 제거된 영상 표현 생성 전체 |
| `graph_items/unique_ids` | 영상 ID 중복 제거 |
| `graph_items/graph_batch/cpu_pack` | CPU에서 Graph 연결·노드 그룹·offset 조립 |
| `…/index_transfer` | 연결 인덱스 전송 및 tensor view 생성 |
| `…/feature_cache_init` | 최초 고정 BGE 특징 캐시 준비·할당 시도 |
| `…/feature_cpu_gather`, `…/feature_transfer` | GPU 캐시 미사용 시 특징 조회·전송 |
| `…/feature_gpu_gather` | GPU 캐시에서 고정 특징 조회 |
| `graph_items/graph_encoder/node_projection` | 노드 projection 및 종류 embedding |
| `…/message_passing` | 역할별 Graph 메시지 전달 |
| `…/scene_readout`, `…/video_pooling` | 장면 벡터 생성 및 영상 mean/attention |
| `graph_items/item_fusion` | 제목 projection·영상 결합 및 최종 LayerNorm |
| `sasrec_forward`, `candidate_vectors`, `logits_loss` | 사용자 표현·학습 후보 벡터·추천 loss |
| `backward`, `clip_grad`, `optimizer` | 역전파·gradient clipping·가중치 갱신 |
| `catalog_encode` | 평가용 전체 catalog 벡터 생성 |
| `catalog_score`, `mask_rank`, `rank_to_cpu` | 전체 후보 점수·마스킹/순위·CPU 결과 복사 |

`[Profile]`에는 상위 단계, `[Profile detail]`에는 Graph 세부 시간이 출력됩니다. 각 줄에
날짜·Arm·seed·device·PID·phase·epoch·batch를 넣어 worker별 로그를 구분합니다.
Graph 표본에는 고유 영상·장면·노드·방향별 간선·chunk 수, checkpoint 사용 여부,
CPU/GPU에서 조회한 특징 행 수도 기록합니다. CUDA 표본에는 해당 프로세스의
allocated/reserved/표본 구간 peak allocated 메모리(MiB)를 기록합니다.

로그 파일은 각 추천 결과 디렉터리의 **`profile.jsonl`**입니다.

```text
artifacts/runs/<RUN_ID>/validation/recommendations/
  text/<date>/seed_<seed>/<arm>/profile.jsonl
  graph/<aggregation>/<date>/seed_<seed>/<arm>/profile.jsonl
```

- `kind=batch`: 선택된 배치, `kind=catalog`: 전체 catalog 생성, `kind=stage`: 준비·초기화·저장,
  `kind=phase`: epoch 또는 test 전체 시간입니다. `kind=start`는 측정 설정입니다.
- `seconds`는 자식 단계를 포함하는 시간입니다. 예를 들어 `graph_items`와 그 하위 시간을
  더하면 중복 계산됩니다. 부모의 독점 시간은 부모 시간에서 직접 자식 경로들의 시간을
  빼서 계산합니다. `calls`는 해당 표본에서 단계가 실행된 횟수입니다. 여러 chunk의 같은 경로는 합산됩니다.
- Checkpoint 재계산은 `backward/graph_encoder/...`로 표시되며 `backward` 시간에 포함됩니다.
  원래 forward와 합쳐 중복 집계하지 않습니다. 전체 backward의 커널별 세분화는 제공하지 않습니다.
- CUDA 측정은 경계에서 synchronize하는 **완료 기준 wall time**입니다. CPU 처리·전송·GPU 실행·
  대기를 포함하며 순수 CUDA kernel time이 아닙니다. 세부 동기화가 실행 겹침을 줄이므로
  측정된 배치와 epoch 시간에는 프로파일링 오버헤드가 있습니다. 정상 처리량 평가는 옵션을 끄고 수행합니다.
- 첫 배치는 캐시 생성·런타임 초기화 영향을 포함합니다. 이후 표본과 구분해 보십시오.
  샘플 시간 합을 전체 epoch 시간으로 해석하거나, 서로 크기가 다른 Graph 배치를 단순 비교하지 않습니다.
- 옵션이 없거나 샘플링하지 않는 배치에는 프로파일러의 CUDA 동기화·메모리 조회·파일 쓰기가 없습니다.
  기존 학습 코드가 원래 수행하던 동기화는 유지합니다.
- 프로파일 형식 v2는 Run·Arm·모델 설정·입력 해시·device·pid·schema_version을 세션의
  `kind=start` 줄에만 기록합니다. 측정 줄의 `session_id`로 해당 시작 줄과 연결합니다.
  측정 줄에는 timestamp, 단계·epoch·배치, 상태, 예제 수, wall time, 작업별 시간·호출 수,
  workload·CUDA 메모리를 남깁니다. null 배치와 빈 딕셔너리는 생략합니다.
  `examples_per_second`는 examples/wall_seconds로 계산하며, `exclusive_seconds`도
  계산 가능한 중복 데이터여서 저장하지 않습니다. 기존 v1 로그는 수정하지 않으며,
  같은 파일에 새 v2 세션이 추가될 수 있습니다.
- 파일은 표본마다 append 후 닫으며, 중단 후 재실행은 새로운 `session_id`로 구분합니다.
  `training.json`의 `execution.profiling`에 이번 session과 측정 간격을 기록합니다.
- 프로파일링 옵션 자체는 학습 캐시 키·모델·BGE 설정을 바꾸지 않습니다. 위 v3 구조 변경은 별도입니다. 완료 조합은 그대로 건너뛰므로
  새 로그도 생성되지 않습니다. 완료된 조합까지 다시 측정할 때만 명시적으로 `--force`를 사용합니다.
  `profile.jsonl`은 로컬 진단 부속 파일이며 공유 추천 캐시에 게시하거나 복원하지 않습니다.
- 이미 실행 중인 프로세스에는 적용되지 않습니다. 다음 실행부터 사용할 수 있습니다.


## 모델 v4: Baseline·Text·Graph 최종 구조

최종 다이어그램과 일치하는 모델 계약은 Text `sasrec-content-v4`, Graph `sasrec-role-graph/v4`입니다.

| 경로 | 아이템 벡터 |
|---|---|
| meta (두 모드 공통) | 제목 BGE 1024 → Linear 512 → LayerNorm 512 |
| text | 제목 BGE 1024→128; Summary BGE 1024→384 → concat 512 → LayerNorm 512 |
| graph | 제목 128; Graph 영상 384 → concat 512 → LayerNorm 512 |

Graph Encoder는 1층·hidden 128입니다. 장면은 Entity 평균·Action 평균·Context 각각 128을 concat한 384차원이며, mean/attention이 영상 384차원을 만듭니다. 최종 아이템 MLP와 제목·장면 별도 LayerNorm은 제거했습니다. Graph layer 내부 정규화, SASRec, User MLP, loss와 평가 프로토콜은 유지합니다. LayerNorm은 affine=True, eps=1e-5입니다.

Text 특징 NPZ는 `title_values`, `video_values` (각 float32 N×1024), `title_available`, `video_available` (각 bool N)를 저장합니다. 제목·Summary는 독립 BGE 입력이며 선택한 Arm 간 새로 인코딩하는 동일 문자열을 공유합니다. `.inputs/<arm>.json`에는 구성 요소별 원문·사용 여부와 truncation을 기록합니다. Text 표현 계약은 `shared-scenes-representation/v4`입니다. 이전 단일 `values` 배열은 사용하지 않습니다.

누락된 부분은 concat 전에 0으로 마스킹합니다. LayerNorm 이후 부분 결측 슬롯에 값이 생길 수 있습니다. 전체 결측·padding은 마지막에 다시 0으로 마스킹합니다. 이력과 추천 후보에 같은 최종 벡터를 사용합니다.

### 재실행

이전 결과가 필요하면 아래 실행 **전에** 별도로 보관하세요. 동일 결과 경로의 기존 조합은 새 학습 결과로 갱신됩니다. 이전 모델의 완료 조합·checkpoint는 새 모델에 재사용되지 않으며 selection부터 재학습합니다. 새 버전으로 완료한 조합은 재개 시 재사용합니다. epoch 중간 재개는 지원하지 않습니다.

```bash
# Text: 분리 BGE 입력 재준비 후 모든 선택 Arm 재학습
python -m validation embed-representations --run-id 260928_v7 --representation-mode text --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-recommendation --run-id 260928_v7 --representation-mode text --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-diagnosis --run-id 260928_v7 --representation-mode text --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta

# Graph: 유효한 기존 입력 캐시 재사용. attention 실험은 mean을 attention으로 변경.
python -m validation run-recommendation --run-id 260928_v7 --representation-mode graph --scene-aggregation mean --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-diagnosis --run-id 260928_v7 --representation-mode graph --scene-aggregation mean --target meta graph_qwen graph_qwen_meta graph_gemini_meta
```

Graph 입력 계약은 그대로입니다. 입력 파일·제목·BGE가 바뀌지 않았다면 재임베딩하지 않습니다. 이전 Gemini 프로파일에서 catalog 전체 Graph가 비어 있었으므로, 전체 실험 전에 해당 입력의 `statistics.json`에서 유효 장면 및 제외 사유를 확인해야 합니다. v4 모델 변경이 그 입력 문제를 해결하지는 않습니다.
