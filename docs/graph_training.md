# 5개 Arm의 장면 인코딩과 추천 학습

현재 추천 Arm은 `base d g0 g1 g2`입니다. 추출기는 Qwen3-VL-2B-Instruct이며 고정 BGE 특징을 사용합니다. 추천 계약은 `gnn-attn-aggregation/v1`, 모델 구조는 `sasrec-scene-content/v4`입니다. 계약명은 식별자이며 장면 Attention을 사용한다는 뜻이 아닙니다.

## 실행

```bash
ARMS=(base d g0 g1 g2)
python -m validation embed-representations --run-id "$RUN_ID" --dataset "$DATASET" --target "${ARMS[@]}"
python -m validation prepare-module --run-id "$RUN_ID" --dataset "$DATASET" --target g1 g2
python -m validation run-recommendation --run-id "$RUN_ID" --dataset "$DATASET" --target "${ARMS[@]}"
python -m validation run-diagnosis --run-id "$RUN_ID" --dataset "$DATASET" --target "${ARMS[@]}"
```

`graph_execution.chunk_items`와 `chunk_nodes`로 청크 크기를 제한하고 한 영상 내부의 장면은 분할하지 않습니다. `chunk_scene_pairs`는 기존 설정 파일 호환을 위해 받지만 현재 모델에서는 사용하지 않습니다. 새 학습에는 이전 구조 결과가 없는 새 Run ID를 사용합니다.

각 단계는 독립적이며 `--target`이 필수입니다. `prepare-module`은 G1·G2를 받으며 함께 선택하면 공통 Graph bundle의 구간 파일을 한 번만 준비합니다. 추천은 학습·validation·최종 test를 수행하고 특징·구간 준비나 진단을 자동 실행하지 않습니다. worker 수·profiling·`--force`는 기존 CLI 규칙을 유지합니다.

## 인코더와 B: Role Fusion

| Arm | 인코딩과 집계 |
| --- | --- |
| Base | 제목 BGE 1024 → Linear512 → 기존 Residual MLP |
| D | 제목1024→128 + 장면별 Description1024→384 → 결합 → Scene MLP → Mean |
| G0 | 제목·Entity·Action·Context 각각1024→128 → Entity/Action 장면 내 평균 → 결합 → Scene MLP → Mean |
| G1 | G0의 Scene MLP 출력에 Hypergraph·구간 기여(A)를 적용 |
| G2 | G1의 장면 인코딩에 Role Fusion(B) 추가 → Scene MLP → Hypergraph(A) |

장면 기반 네 Arm 모두 `h = x + ReLU(Wx + b)`의 Residual Scene MLP(512→512)를 사용합니다. Scene MLP는 장면별로 독립 적용합니다. 구조는 공통이지만 Arm별 학습 파라미터를 공유하지 않습니다.

B는 `[Action; actor Entity; target Entity; tool Entity; place Entity]` 640차원을 Linear128·ReLU로 변환해 원래 Action에 더합니다. 연결이 없는 역할은 absent/unknown 상태 임베딩을 사용합니다. Entity 자체는 갱신하지 않습니다. G0·G1은 Fusion층과 상태 임베딩을 만들지 않으며 역할 연결이 출력에 영향을 주지 않습니다.

Description은 장면별 문단을 각각 BGE로 인코딩합니다. Summary는 입력에 포함하지 않습니다. Graph와 Description은 각자의 유효 장면을 사용하며 실패 장면은 제외합니다. catalog 아이템은 유지하고, 유효 장면이 없는 영상은 `SceneMLP([투영 제목; zeros384])`, 제목도 없으면 0 벡터를 사용합니다.

## A: Hypergraph 집계와 Action 구간 기여

활동 구간 생성은 고정 전처리입니다. Action 문구의 참여자 표식·일부 기능어를 제외한 첫 단어에 기존 어미·불규칙형 규칙을 적용합니다. 같은 활동 키가 연속된 원본 장면 번호에 등장하면 최대 연속 구간으로 묶습니다. 번호가 끊기거나 활동이 다시 등장하면 새 구간이며 길이1 구간과 중첩 구간을 허용합니다. 실제 행동 경계나 동일 actor/target을 검증한 결과로 해석하지 않습니다.

노드는 장면, hyperedge는 활동 구간입니다. S는 영상의 유효 장면 수, k는 구간입니다.

```text
u_k = ReLU(W_u [mean(h_s for s in k); log(|k|); |k|/S] + b_u)
h'_s = h_s + gate_scene * ReLU(W_s [h_s; mean(u_k for k containing s)] + b_s)
z = mean(h'_s) + gate_video * W_v mean(u_k)
```

구간에 속하지 않는 장면은 갱신하지 않습니다. 영상 내 구간 기여는 구간별 동일 가중치 평균이며, 두 게이트는 0에서 시작하는 학습 가능한 스칼라입니다. 구간이 없는 영상은 해당 Arm의 장면 평균을 사용합니다. Scene MLP 출력이 A의 입력입니다. G1·G2에서 Hypergraph가 장면 간 전달의 유일한 경로이며 영상 노드는 없습니다. G1은 B 없이 A만, G2는 A와 B를 모두 사용합니다. A/B는 Arm의 추가 순서이며 실제 계산에서 B는 장면 인코딩, A는 Scene MLP 뒤에 적용합니다. SASRec 내부 Attention은 유지합니다.

구간 파일은 `validation/representations/graph_qwen3vl/scene_segments.npy`와 `.json`이며 계약은 `scene-segments/v2`입니다. 입력 해시·파일 checksum을 검사하고 추천 identity에 포함합니다. 구간 준비는 VLM·BGE 재추론 없이 수행합니다. bundle 또는 생성 규칙 변경 후 다시 준비합니다.

## 공동 학습과 평가

BGE는 동결합니다. 투영층·B·Scene MLP·A와 SASRec을 Arm별 추천 loss로 공동 학습합니다. 512차원 영상 벡터를 사용자 이력에 사용하고 SASRec 사용자 벡터와 후보 영상 벡터의 내적으로 점수를 계산합니다. SASRec 구조·loss·split·seed·학습 설정은 이 Arm 정리에서 변경하지 않습니다. 실제 설정은 [config.yaml](../config.yaml)의 `validation`을 따릅니다.

사용자별 마지막 두 이벤트를 validation/test로 보류합니다. validation HR@10으로 checkpoint를 선택하고 동점이면 최초 epoch를 유지합니다. 선택된 checkpoint로 test를 한 번 수행하며 refit하지 않습니다. 기존 인기도 보정 인배치 Cross-Entropy와 이력 마스킹·전체 catalog 랭킹을 유지합니다.

## 재사용과 과거 결과 보호

공유 특징 소스는 `metadata`, `desc_qwen3vl`, `graph_qwen3vl`이며 동일 입력·checksum의 특징은 재사용합니다. 이전 추천 결과나 checkpoint는 새 Arm으로 재분류하지 않습니다. 같은 추천 경로에 다른 구조 결과가 있으면 쓰기 전에 실패시키고 새 Run을 안내합니다. `--force`도 다른 구조를 덮어쓰지 않습니다. 시작 시 `identity.json`을 기록해 부분 결과를 보호합니다.

추천 결과는 `validation/recommendations/seed_<seed>/<arm>/`에 저장합니다. 동일한 새 identity의 완료 조합은 모델 초기화 없이 재사용합니다. epoch 중간 재개는 지원하지 않습니다. 계약명을 경로에 추가하지 않습니다.

## 진단과 해석

| 우선 비교 | 의미 |
| --- | --- |
| G0 − D | Description과 구조화된 Graph 입력·인코딩 차이 |
| G1 − G0 | Role Fusion 없이 A(Hypergraph)를 추가한 효과 |
| G2 − G1 | Hypergraph가 있는 상태에서 B(Role Fusion)를 추가한 효과 |

G0−D는 추출 내용·투영 구조·유효 장면 차이도 포함합니다. G1−G0만으로 Hypergraph 연결이나 두 내부 경로의 효과를 각각 분리할 수 없습니다. A는 1,050,626개, B는 83,072개 파라미터를 추가합니다. 모듈 추가에 따른 파라미터 증가를 함께 보고하며 각 효과를 합산하지 않습니다.

NDCG@10·HR@10의 seed별 값, 평균·표본 표준편차와 Base 대비 변화율을 보고합니다. Base 미선택 시 상대 변화율을 생략합니다. NDCG@10 사용자 단위 paired bootstrap과 선택 Arm 전체 쌍 Bonferroni 보정을 유지합니다. 전체 5개는 10쌍이며 가족 오류율5%에서 쌍별 신뢰수준99.5%입니다. HR@10은 기술통계만 보고합니다.

진단은 `validation/diagnosis/base_d_g0_g1_g2/`, 보고서는 `reports/base_d_g0_g1_g2/`에 저장합니다. 부분 선택은 등록 순서로 target-set을 만듭니다. 사용자·이벤트·split·seed·모델·Arm·평가 계약을 검사하며 `--compare-run-id`는 같은 Arm의 Run 간 비교입니다. 마지막 콘솔에는 Arm별 NDCG@10·HR@10 평균 ± 표준편차를 출력합니다.

## 도식과 검증 범위

[D](design/recsys_diagram_d.png) · [G0](design/recsys_diagram_g0.png) · [G1](design/recsys_diagram_g1.png) · [G2](design/recsys_diagram_g2.png) · [생성 코드](design/generate_graph_arm_diagrams.py)

CPU·가짜 BGE 테스트는 수식·gradient·장면/영상 경계·fallback·작은 데이터 학습/평가·캐시 보호·진단을 검증합니다. 실제 추출의 시각적 정확도, CUDA 처리량과 전체 데이터 성능 우위는 별도 실험으로 확인해야 합니다.

## G5 정합성과 파라미터

G2의 계산·초기화 기준은 `clab-insight`의 `hyper-g5` 커밋 `486c541`에 등록된 G5입니다. 두 모델은 Role Fusion → Scene MLP → Hypergraph → 장면 평균 + 구간 기여를 수행합니다. GCN은 생성하지 않으며 구간 파라미터는 별도 난수 생성기로 초기화합니다. 구현 정합성은 실제 데이터의 NDCG 0.0440 재현을 보장하지 않습니다.

기본 설정에서 전체 학습 파라미터는 Base·D·G0 각각 7,624,704개, G1 8,675,330개, G2 8,758,402개입니다. 고정 BGE 파라미터는 포함하지 않습니다.
