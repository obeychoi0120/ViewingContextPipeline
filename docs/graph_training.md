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

기존 text 명령에는 `--representation-mode text`만 추가합니다. Description Arm은 text에서만 지원합니다. `graph`의 `meta`는 기존 제목 전용 SASRec이며 Graph encoder를 사용하지 않습니다.

## 모델 및 데이터 계약

- BGE 입력: entity의 종류·attributes, action 문구, Context의 medium·format·topics, 필요한 Arm의 영문 제목. 고유 문자열만 인코딩하며 float32 특징은 학습하지 않습니다.
- entity/action 노드 사이 actor·target·tool·location별 양방향 메시지를 2층·128차원에서 전달합니다. `none`/`unknown`은 역할별 서로 다른 상태 벡터이며 공유 개체 노드가 아닙니다.
- entity 평균, action 평균, Context를 결합해 장면 512차원으로 변환합니다. 영상 mean 또는 attention pooling은 순서 불변이며 시간 위치 정보와 장면 간 메시지는 사용하지 않습니다.
- `_meta`는 `[제목 1024; Graph 512]` 순서입니다. 제목 없는 Graph Arm도 같은 1536→512 변환층을 사용하되 제목 부분은 0입니다.
- 유효한 Graph가 없으면 Graph 부분은 0입니다. 제목도 없으면 최종 영상 벡터는 0입니다. 후보·interaction은 제거하지 않습니다. `meta` 기준선은 기존 모델 계산을 그대로 사용합니다.
- raw·warning·구조 오류 장면은 제외하고 장면별 사유를 저장합니다. 행동이 없는 정상 장면과 고립 entity는 유지합니다. 필수 참조는 검사하고 receiver는 기존 데이터에 있어도 무시하고 location 누락은 `unknown`으로 해석합니다. 모델 텐서의 receiver 슬롯은 형태 호환성을 위해 남기되 항상 `none` 상태로 두고 해당 간선은 만들지 않습니다. 입력 버전을 갱신해 receiver를 사용한 기존 특징 캐시를 재사용하지 않습니다. JSONL 손상·ID 불일치·중복 장면 번호는 중단합니다.
- 개체 ID는 장면 내부 연결에만 쓰며 BGE 입력에 넣지 않습니다. 개체·행동 수나 장면 수를 프롬프트 상한에 맞춰 자르지 않습니다. 가변 길이 배열을 쓰며 영상 단위로 분할 계산합니다. 분할당 64개 영상·8,192개 노드를 기준으로 묶되 이를 넘는 단일 영상도 그대로 유지합니다. 학습 시 activation checkpointing으로 분할 내부 활성값을 backward에서 재계산합니다.
- SASRec의 loss, 인기도 보정, 후보 마스킹, 이력 길이 10, 차원 512, 날짜별 selection/refit/test는 기존과 같습니다. 각 refit은 모든 학습 모듈을 초기화합니다. 학습 배치의 이력·정답 아이템 합집합만 계산하고 optimizer 갱신 후 캐시를 버립니다.

## 산출물과 재현

Run 루트는 `artifacts/runs/<RUN_ID>/`입니다.

| 위치 | 내용 |
|---|---|
| `validation/representations/graph/<arm>_embeddings/` | memory-mapped BGE·노드·연결·offset·제목 배열, manifest, statistics |
| `validation/recommendations/graph/<mean 또는 attention>/<date>/seed_<seed>/<arm>/` | 전체 모듈 checkpoint, 최종 catalog_vectors.npy, training, per-event metrics, complete |
| `validation/diagnosis/graph_<mean 또는 attention>_diagnosis.json` | 동일 모드 내 paired 통계와 장면 제외·결측·제목 사용 집계 |

`catalog_vectors.npy`의 행 순서는 Graph 입력 manifest의 catalog와 같습니다. padding 행은 저장하지 않습니다. checkpoint를 읽을 때는 동일 해시의 Graph 입력으로 `new_graph_model`을 생성하고 state_dict를 읽습니다. BGE 특징은 checkpoint에 중복 저장하지 않습니다.

캐시는 실제 Scene Graph 파일 바이트·catalog 순서·제목·BGE 설정/로컬 모델 식별 정보·변환 버전으로 구분합니다. 추천 결과에는 특징 해시, Graph 구조 설정, aggregation과 기존 학습 해시·구간·seed가 들어갑니다. 변환/모델 의미를 변경하면 해당 버전도 갱신해야 합니다. text 캐시 키와 파일 형식은 유지하며 저장 위치는 `validation/representations/text/`, `validation/recommendations/text/`, `validation/diagnosis/text_diagnosis.json`입니다.

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
