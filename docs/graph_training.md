# 직접 Graph 학습 경로

현재 파이프라인은 `text`의 기존 Summary 입력과 `graph`의 원본 Entities·Actions·Context 입력을 함께 지원합니다. 모든 validation 명령에는 `--representation-mode`가 필요합니다. Graph 추천·진단은 **`--scene-aggregation mean|gnn|transformer|transformer_pos|transformer_local|attn_pool|gnn_attn`**을 필수로 받으며 `attention`은 CLI에서 거절합니다. embedding 단계와 text 모드에는 aggregation을 지정하지 않습니다.

## 실행

로컬 BGE 가중치와 torch·transformers 환경을 사용합니다. `pip install -e ".[train,embedding]"`로 의존성을 설치할 수 있습니다. BGE는 GPU별 고유 문자열 분할, 추천은 GPU별 독립 seed·Arm 조합으로 실행하며 단일 모델 DDP는 사용하지 않습니다.

Run 평가 산출물은 `validation/`에 직접 저장합니다. 기존 파일은 자동 이동하거나 삭제하지 않습니다. 기존 결과를 보존하며 조건을 바꿀 때는 새 Run ID와 config snapshot을 사용하십시오. 아래 명령은 전체 학습을 실행하므로 이 구현 검증에서는 실행하지 않았습니다.

```bash
RUN_ID="새_Run_ID"
python -m validation embed-representations --run-id "$RUN_ID" --representation-mode graph --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
```

Text는 `--representation-mode text`를 사용하며 Description Arm은 text에서만 지원합니다. 두 경로의 `meta`는 동일한 Metadata 모델입니다. 실제 제목 BGE 입력의 동일성은 각 입력 manifest로 별도 확인해야 합니다.

## 모델 및 데이터 계약

현재 GNN 구조: [추천 구조 PNG](design/recsys_diagram_gnn.png) · [편집 가능한 PPTX](design/recsys_diagram_gnn.pptx) · [갱신 코드](design/generate_recsys_diagram_gnn.py). 현재 mean 구조는 [PNG](design/recsys_diagram.png) · [PPTX](design/recsys_diagram.pptx)에서 확인할 수 있습니다. 전체 파이프라인은 [main_diagram](design/main_diagram.png)을 참고하십시오.

영상용 Item·Scene Residual MLP는 `R(x) = x + ReLU(Linear(512,512)(x))`이며 Linear 하나로 구성됩니다. **Item·Scene 모듈 전체에 LayerNorm이 없으며**, SASRec Transformer와 기존 2층 GELU User MLP의 내부 LayerNorm은 유지합니다.

| 경로 | Item 계산 |
|---|---|
| Metadata | 제목 BGE 1024 → Linear 512 → R |
| Text Graph·Desc | 제목 BGE 1024 → Linear 128; Summary BGE 1024 → Linear 384 → concat 512 → R |
| Graph | 네 성분의 독립 Linear 1024→128 → Action 역할 결합 → Scene concat 512 → R_scene → 유효 Scene Mean |

Graph의 네 Projection은 Title·Entities·Actions·Context마다 독립적이고 모든 Scene에서 공유됩니다. 제목은 영상당 한 번 projection합니다. 각 Action은 `[action, actor, target, tool, place]`의 640차원 concat에서 `a + ReLU(Linear(640,128)(concat))`으로 보정합니다. 역할 연결의 Entity projection을 사용하고 `none`과 `unknown`은 역할별 별도 학습 가능한 상태 벡터를 사용합니다. 현재 입력에는 receiver가 없으며 Entity→Action 한 번만 전달합니다. 역방향 갱신·node-type embedding·다층 message passing은 없습니다.

Scene은 `[title; Entity mean; 보정된 Action mean; Context]` 순서로 concat하고 Scene MLP를 적용한 **뒤** 평균냅니다. Entity나 Action이 없으면 해당 슬롯은 0입니다. Scene이 없고 제목만 있으면 `[title; 0; 0; 0]`을 Scene MLP에 넣습니다. 제목과 유효 Scene 모두 없거나 padding이면 최종 Item은 정확히 0입니다. Graph에는 pooling 뒤 Item MLP가 없습니다.

Text의 누락 성분은 projection 이후 0으로 마스킹하며, 전체 누락과 padding은 Item MLP 이후 다시 0으로 마스킹합니다. 부분 누락 슬롯은 MLP를 거친 후에도 0이어야 하는 것은 아닙니다.

BGE Large 1024차원 float32 특징은 동결됩니다. 현재 compact Scene·Context/Entities/Actions를 읽으며 Entity·Action 인덱스, edge·결측 배열의 shape는 유지합니다. 입력 계약은 `role-graph-inputs/v8`입니다. receiver 예약 슬롯은 모델/checkpoint shape를 위해 유지하고 항상 absent 상태입니다. Relations·receiver가 있는 기존 Scene은 오류로 거부하고 원본은 보존합니다. 미선언 역할 참조는 Warning 없이 `unknown`으로 바꾸고 actor·target이 모두 불명인 Action도 유지합니다. 중복 ID의 Entity는 모두 보존하며 참조 Edge는 같은 ID의 마지막 선언에 연결합니다. Warning은 `MISSING_REQUIRED`와 `PARSE_FAILED`만 기록하며, 구조화할 수 없는 Raw Text와 필수 필드·타입 오류 Scene은 제외합니다. 기존 Scene의 제거된 Warning은 무시하고 Raw Text는 읽을 때 다시 파싱합니다. 원본 파일은 변경하지 않습니다. 정상적인 Action 없는 Scene·고립 Entity도 유지합니다. place은 명시해야 하며 `unknown`과 `none`을 구분합니다. Entity ID는 BGE 입력에 넣지 않습니다. Scene·노드 수는 자르지 않습니다.

Entity·Action·Context는 각각 별도로 BGE에 입력합니다. Entity는 `Name` 다음 `Attributes`, Context는 `Format`·`Medium`·`Topics` 순서로 줄바꿈하며, Action은 동작 텍스트만 사용합니다. Attributes와 Topics 원소는 기존처럼 정렬하고 `, `로 연결합니다. 빈 목록은 필드명과 콜론만 남깁니다. 예시 입력은 다음과 같습니다.

```text
Name: cook
Attributes: red apron
```

```text
cooking
```

```text
Format: demonstration
Medium: live_action
Topics: cooking, food preparation
```

Context는 위 세 필드를 합친 문자열 하나를 Scene별로 임베딩합니다. `v2`의 JSON 문자열 입력 캐시는 새 입력으로 재사용하지 않으며, 입력 hash 변경으로 해당 추천 결과와 체크포인트도 구분됩니다. 새 Run에서 Graph embedding → 추천 학습 → 진단을 다시 실행하십시오. 기존 추출 Scene과 Text Summary 입력은 그대로 사용할 수 있습니다. 기존 Run 산출물은 이 코드 변경으로 수정하지 않습니다.

학습 설정·seed·결정적 연산 사용 여부는 [config.yaml](../config.yaml)의 `validation.model`, 평가·통계 설정은 `validation.evaluation`, 사용자 필터·보존 이력은 `validation.cohort`를 참고하십시오. Checkpoint는 validation HR@10으로 선택하고 최종 보고·paired bootstrap의 주 지표는 NDCG@10입니다. 사용자별 마지막 2개를 validation/test로 보류하고 test 이력에는 validation을 포함하며 refit은 수행하지 않습니다. 기존 Run snapshot은 수정하지 않습니다.

## 캐시 및 packed 실행

Text 모델은 `sasrec-content-v8`, Graph mean은 `sasrec-role-graph/v9`, Graph gnn은 `sasrec-role-graph/v10`, 학습 구현은 `shared-scenes-leave-one-out-training/v2`, packed 실행은 `packed-role-graph/v2`입니다. 모델 identity에는 MLP 구조·정규화 제거·역할 결합·Scene 차원·Pooling이 포함되며 학습 설정은 training hash와 `training.json`에 기록됩니다. 과거 checkpoint·추천 결과는 새 결과로 재사용하지 않습니다. 원본·인코더·현재 입력 계약이 동일한 BGE 입력 캐시는 재사용합니다. 이전 Graph 입력 계약 캐시는 새 경로에서 재생성합니다.

배치 준비에서 Action별 actor·target·tool·place의 Entity 인덱스와 결측 상태를 계산합니다. 학습 중 가중치나 상태 벡터는 준비 단계에서 계산하지 않습니다. GPU 고정 특징 캐시, 영상 chunk, checkpoint, 평가 catalog 캐시는 유지하며 optimizer step마다 Item cache를 폐기합니다. checkpoint 재계산은 이미 준비한 정적 입력의 신경망 연산만 반복합니다.

`validation.graph_execution` 기본값은 `chunk_items: 256`, `chunk_nodes: 32768`, `checkpoint: auto`, `feature_cache_mb: 2048`입니다. CUDA auto checkpoint는 새 Projection·역할 결합·512차원 Scene MLP의 전체 배치 활성값을 추정하여 `min(8 GiB, 여유 메모리의 35%)` 초과 시 사용합니다. CPU auto는 checkpoint를 사용하지 않습니다. 이는 메모리 상한 보장이 아닌 추정 규칙입니다. `always`/`never`로 비교할 수 있습니다. 실행 chunk/cache 옵션은 모델 의미와 training identity를 바꾸지 않습니다.

## 산출물과 검증

| Run 아래 위치 | 내용 |
|---|---|
| `validation/representations/graph/<arm>_embeddings/` | 고정 BGE·노드·edge·결측·offset·제목 배열, manifest, statistics |
| `validation/recommendations/graph/<aggregation>/seed_<seed>/<arm>/` | checkpoint, catalog_vectors.npy, training.json, per-event metrics, complete |
| `validation/diagnosis/graph_<aggregation>_diagnosis.json` | 최종 HR/NDCG·수치 비교·평가 계약·실행 판정 |
| `validation/diagnosis/text_diagnosis.json` | Text 진단 |

기존 Attention 산출물과 과거 Run은 수정·이동하지 않습니다. 같은 Run의 기존 결과 경로에서 재학습하면 해당 조합을 대체하므로 보존 실험에는 새 Run을 사용하십시오. `--compare-run-id`는 선택한 경로·aggregation의 결과만 읽고 프로토콜·전처리·원본·분할·catalog·seed가 다른 Run을 거절합니다.

```bash
python -m pytest -q tests
```

검증은 소형 고정 특징 및 학습·평가·진단 fixture를 사용합니다. 전체 데이터 재학습, 추천 성능 개선, 실제 BGE 추론 처리량은 별도 실험 대상입니다.

2026-10-06 검증: `vc_cloud`, PyTorch 2.13.0+cu129, NVIDIA L4에서 CPU validation **156개**, CUDA 모델 **12개**, CUDA profiler **4개**를 통과했습니다. 참조 구현의 출력·gradient, chunk·checkpoint·특징/catalog cache, 결정적 반복 학습, topology·순서 불변성·결측 상태, optimizer 갱신, 두 Metadata adapter의 동일 입력 일치를 검사했습니다. 소형 rolling 학습→평가→진단과 결과 캐시 검증도 포함됩니다.

별도 integration은 **68개 통과, 11개 실패**입니다. 실패는 기존 root 실행 스크립트·그림 링크 누락 및 과거 v7 hash fixture 불일치이며, 변경 전 HEAD를 별도 디렉터리에서 실행해 같은 11개 실패를 확인했습니다. 변경 파일 Ruff 검사와 shell 문법 검사도 통과했습니다. 전체 Run 학습 및 실제 성능 평가는 실행하지 않았습니다.

## 장면 attention aggregation (2026-10-12)

자문 메모의 A–D 비교표에서 비어 있던 두 칸을 채우는 실험 경로입니다. B는 장면 간 교환 없이 attention pooling만 하는 경로이고, D는 GCN과 조건을 맞춘 attention 전달입니다. Transformer 3종과는 이름·계약(`sasrec-role-graph/v12`, leave-one-out 병합 전 `v10`)이 따로이며, `mean`·`gnn`·Transformer의 계약·초기 가중치·출력은 바뀌지 않았습니다. 구현과 단위 검증까지만 했으며 **추천 학습·평가는 실행하지 않았습니다.**

| `--scene-aggregation` | 자문 표 | 장면 간 연산 | 영상 readout | 비교 대상 → 확인하려는 효과 |
|---|---|---|---|---|
| `mean` | A | 없음 | 평균 | 기준 |
| `attn_pool` | B | 없음 | **학습 attention 가중합** | `mean` → 중요한 장면을 고르는 효과 |
| `gnn` | C | cosine GCN 1층 | 평균 | `mean` → 장면 간 정보 교환 |
| `gnn_attn` | D | **학습 attention GCN 1층** | 평균 | `gnn` → 이웃 가중치를 학습하는 효과 |

`gnn_attn` 추천 구조 전체(Baseline·Text·Graph): [PNG](design/gnn_attn/recsys_diagram_gnn_attn.png) · [편집 가능한 PPTX](design/gnn_attn/recsys_diagram_gnn_attn.pptx) · [갱신 코드](design/generate_recsys_diagram_gnn_attn.py). `gnn_attn` 상세 구조: [PNG](design/gnn_attn/gnn_attn_detail_diagram.png) · [갱신 코드](design/generate_gnn_attn_detail_diagram.py). attention 계산 단계별 예시: [PNG](design/gnn_attn/gnn_attn_walkthrough_diagram.png) · [갱신 코드](design/generate_gnn_attn_walkthrough_diagram.py). 두 그림의 장면·가중치 값은 설명용 가상 예시입니다.

`gnn` → `gnn_attn` → `transformer` 순서로 비교하면 효과를 나눠 볼 수 있습니다. `gnn_attn`은 가중치를 정하는 방식만 바꾸고, `transformer`는 여기에 LN·FFN·2 heads·ReZero 용량을 더합니다. 아래 Transformer 절의 C0/C/D는 이 표의 A–D와 다른 라벨입니다.

```text
공통: [Title; Entity 평균; Action 평균; Context] 512 → Scene MLP → H (S × 512)
unit_rms(x) = √dim · x / max(‖x‖, 1e-8)      점수 입력에만 사용, 값 경로는 그대로

attn_pool:  β = 영상 내부 softmax_s( wᵀ tanh(W₁ unit_rms(h_s) + b₁) )
            W₁ 512→128, w 128→1 (bias 없음, 0 초기화);  영상 = Σ_s β_s h_s
gnn_attn:   f = [Entity 평균; Action 평균; Context] 384 (제목 제외, gnn과 동일)
            A = 행 softmax_j( (W_q unit_rms(f_i))ᵀ (W_k unit_rms(f_j)) / √128 ),  W_q·W_k 384→128, bias 없음
            H′ = H + ReLU(A H W + b) → 유효 장면 평균        (gnn: A = D^-½ max(cos, 0) D^-½)
```

### 설계 근거

- **`attn_pool`의 시작점.** 점수 출력 `w`를 0으로 초기화합니다. 따라서 모든 영상이 균등 가중치, 즉 `mean`과 같은 함수에서 출발합니다(Transformer의 α = 0과 같은 원칙). `GraphSASRec`의 공통 Linear 초기화 뒤에 다시 0으로 둡니다.
  - 출력은 장면 벡터의 볼록 결합이라 영상 벡터 크기가 `mean`과 같은 범위에 머뭅니다. 합성 fixture의 초기 노름은 0.710으로 `mean` 0.708과 비슷했습니다.
  - softmax는 영상 안의 공통 이동을 무시하므로 `w`의 bias는 gradient가 항상 0입니다. 그래서 bias를 두지 않았습니다.
  - 과거 v3–v6 attention pooling(384→128→1)과 같은 형태의 점수 MLP를 512 입력으로 옮겼습니다. 과거 산출물 경로를 보호하려고 이름은 `attention`이 아니라 `attn_pool`입니다.
- **`gnn_attn`과 `gnn`의 정렬.** 다음 조건을 `gnn`과 같게 두었습니다: 제목을 뺀 384차원 입력, self-loop 포함, 1층, `W` 512→512 residual ReLU, 영상 내부 전체 장면 쌍, gradient 비차단.
  - 모듈 이름도 `gcn`으로 같아서 checkpoint 차이는 `gcn.query`·`gcn.key` 두 행렬(98,304개)뿐입니다.
  - 정규화가 대칭 degree에서 행 softmax로 바뀌는 점은 가중치 결정 방식의 일부로 봅니다.
  - key 폭 128은 기존 hidden 128을 재사용했습니다. 새 하이퍼파라미터를 들이지 않으려는 선택입니다.
- **점수 입력 정규화(`unit_rms`).** Scene 특징의 노름이 약 1이어서, 크기를 그대로 두고 점수를 계산하면 점수가 0 근처에 머뭅니다.
  - 합성 fixture에서 정규화 없이 300 step 학습하면 정규화 엔트로피가 `gnn_attn` 0.99999, `attn_pool` 0.9995로 사실상 균등했습니다. 이대로라면 실험이 다시 `mean`이나 균등 GCN을 시험하게 됩니다.
  - Transformer가 pre-LN으로 sublayer 입력을 안정시킨 것과 같은 이유로, **점수 계산 입력만** 정규화했습니다. `gnn` cosine과 같은 L2 방향(eps 1e-8)에 `√dim`을 곱해 단위 RMS로 맞춥니다.
  - 파라미터가 없고, 가중치가 입력 크기에 불변입니다(cosine과 같은 성질). 값·residual 경로와 영상 벡터 크기는 바꾸지 않습니다.
  - 같은 조건에서 300 step 후 엔트로피는 `attn_pool` 0.990, `gnn_attn` 0.886이었습니다.
- **`gnn_attn` 초기화.** `W_q`·`W_k`는 `N(0, 1/(2·fan_in))`로 초기화합니다.
  - 단위 RMS 입력에서 초기 attention logit의 표준편차가 0.5가 됩니다. 이 저장소 Transformer의 xavier `in_proj` + pre-LN 초기값(측정 0.49)과 같습니다.
  - 공통 xavier를 그대로 쓰면 1.5(측정 1.50)였고, 초기 attention이 이미 뾰족했습니다. 합성 초기 엔트로피는 약 0.70이었고, 변경 후에는 약 0.95입니다.
- **비용.** `attn_pool`은 O(S·d)이고 장면 쌍 연산이 없습니다. 그래서 scene group과 `chunk_scene_pairs` 분할 없이 `mean`과 같은 실행 경로를 씁니다. `gnn_attn`은 `gnn`과 같은 S² 경로입니다. 추가 파라미터는 `attn_pool` 65,792개이고, `gnn_attn`은 `gnn`의 262,656개에 98,304개를 더합니다.

### 학습 후 확인할 기록

두 aggregation은 `training.json`의 `execution.graph_training.scene_attention_summary`에 기록합니다. 병합 전 rolling에서는 `execution.graph_selection`·`execution.graph_refit`에 기록했습니다.

- 전체 catalog를 no-grad로 한 번 더 인코딩해 계산합니다. item cache와 chunk 집계는 바꾸지 않습니다.
- 장면이 2개 이상인 영상만 집계합니다.
- 비율은 `S × 가중치`입니다. 1이면 균등(mean), S면 한 장면만 선택한 것입니다.
- 정규화 엔트로피는 `엔트로피 / log S`입니다. 1이면 균등, 0이면 한 장면입니다.

| aggregation | 항목 |
|---|---|
| `attn_pool` | `videos`, `normalized_entropy_mean`, `max_weight_ratio_mean`, `first_scene_weight_ratio_mean`·`last_scene_weight_ratio_mean`(첫·끝 장면 편향), `score_weight_norm`(0이면 mean과 동일) |
| `gnn_attn` | `rows`(장면 행 수), `normalized_entropy_mean`, `max_weight_ratio_mean`, `self_weight_ratio_mean`, `query_weight_norm`, `key_weight_norm` |

높은 가중치를 사람이 판단하는 장면 중요도나 사용자 선호의 원인으로 해석하지 않습니다.

### 예외·계약·캐시

- 장면이 1개인 영상은 두 경로 모두 가중치 1입니다.
- 장면 없이 제목만 있으면 기존 fallback을 쓰고 attention을 건너뜁니다. 전체 결측·padding은 최종 0입니다.
- `meta`는 attention을 적용하지 않는 동일 제목 모델입니다. 계약에는 `attention_pool_applied: false`(`attn_pool`) 또는 `gcn_applied: false`(`gnn_attn`)로 기록됩니다.
- 두 경로 모두 장면 순서를 바꿔도 영상 벡터가 같습니다.

모델 계약은 `sasrec-role-graph/v12`입니다. leave-one-out 병합 때 origin의 mean `v9`·gnn `v10`과 겹치지 않도록 기존 `v10`에서 바꿨습니다. 점수 입력·초기화·정규화는 `graph_model.attention_pool` 또는 `graph_model.gcn`에 기록합니다. 현재 입력 계약 `role-graph-inputs/v8`와 BGE 특징 캐시를 `mean`/`gnn`과 공유하므로 aggregation별 **재임베딩이 필요 없습니다.**

추천 결과·checkpoint·shared cache·진단은 aggregation별 경로로 분리됩니다(`recommendations/graph/<attn_pool|gnn_attn>/`, `diagnosis/graph_<aggregation>_diagnosis.json`). Profiler는 `attn_pool`을 `graph_encoder/video_pooling`에, `gnn_attn`을 `graph_encoder/scene_attention`·`scene_gcn`에 기록합니다. auto checkpoint 추정에는 두 경로의 활성값을 추가했습니다.

```bash
RUN_ID="사용할_Run_ID"
for AGG in attn_pool gnn_attn; do
  python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation "$AGG" --workers-per-gpu 1 --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
  python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation "$AGG" --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
done
python -m pytest -q tests/test_models.py tests/test_representations.py
```

### 이번 변경의 검증

로컬 PC(NVIDIA RTX 4090, PyTorch 2.13.0+cu130)에서 확인한 결과입니다.

| 항목 | 결과 |
|---|---|
| 전체 `tests` (CPU+CUDA) | **380개 통과** (변경 전 340개 + 신규 40개). CUDA profiler 4개는 변경 전과 마찬가지로 함께 실행하면 실패하고 별도 프로세스에서 통과 |
| 기존 5개 aggregation | `mean`·`gnn`·`transformer`·`transformer_pos`·`transformer_local`의 계약·초기 state·catalog 벡터·1 step gradient·execution report가 변경 전 작업 트리와 CPU·CUDA에서 비트 단위로 일치 |
| Ruff·`git diff --check` | 변경 파일 통과 |

신규 테스트는 다음을 확인합니다.

- NumPy·수식으로 직접 계산한 segment softmax(큰 점수에서의 안정성), attention pooling, attention GCN과의 일치. 점수 입력 크기에 대한 불변성
- `attn_pool`이 0 초기화에서 mean과 일치하고, 첫 step에는 점수 가중치만 gradient를 받으며 다음 step에 hidden으로 전달됨
- 장면 순열 불변·등변성, 영상 격리, 묶음 계산과 영상별 단독 계산의 일치, 제목이 `gnn_attn` 가중치 입력에 들어가지 않음
- `gnn`과 `gnn_attn`의 checkpoint 차이가 `query`·`key`뿐임, 파라미터 수, `gnn_attn` 초기 logit 표준편차 0.5
- 요약 통계가 직접 계산과 일치하고, 균등 가중치에서 모두 1이며, 학습 후 변함. 요약 계산이 chunk 집계를 바꾸지 않음
- fallback·빈 영상 우회, chunk·checkpoint 출력과 gradient 일치, 저장·재로딩, 다른 aggregation checkpoint 거부
- `attn_pool`이 장면 쌍 분할·scene group을 쓰지 않음
- 소형 selection→refit→test→diagnosis, `training.json` 요약 기록, 계약·cache key 분리, Entity ID·장면 순서 불변

합성 fixture(영상 64, 장면 315, batch 128, lr 1e-4, dropout 0.1)에서 GPU로 300 step 학습한 결과입니다.

| aggregation | 장면 있는 영상 노름 (초기 → 300 step) | 300 step 후 가중치 |
|---|---|---|
| `mean` | 0.708 → 0.948 | — |
| `attn_pool` | 0.710 → 0.968 | 엔트로피 0.990, 최대 비율 1.18, 첫 장면 비율 0.79 |
| `gnn` | 0.903 → 1.243 | — |
| `gnn_attn` | 0.891 → 1.261 | 엔트로피 0.886, 최대 비율 2.06, 자기 비율 1.11 |

합성 목표이므로 추천 품질과는 무관합니다. 같은 fixture로 benchmark를 돌린 학습 batch 시간(RTX 4090, warmup 5·10 step·3회 중앙값)은 `mean` 6.87ms, `attn_pool` 7.38ms, `gnn` 14.60ms, `gnn_attn` 14.31ms, Transformer 3종 16.6–17.5ms였습니다. 소형 측정이라 실제 데이터 처리량으로 일반화하지 않습니다.

추천 품질, 실제 데이터에서의 가중치 변화와 처리량은 검증하지 않았습니다. 전체 추천 구조 그림(`recsys_diagram*`, `main_diagram`)은 갱신하지 않았습니다.

후속 후보(미구현): 교환과 attention pooling의 조합(예: `gnn` + `attn_pool`), UniMP gated residual, CSA 보조 학습, Scene 내부 Entity·Action attention, 교환 전후 장면 다양성(oversmoothing) 진단을 `gnn`·Transformer에도 적용.

## 장면 Transformer aggregation (2026-10-12)

장면 간 상호작용과 장면 순서가 추천 영상 벡터에 각각 기여하는지 분리해 검증하기 위한 실험 경로입니다. `gnn`의 GCN 자리, 즉 **Scene MLP 출력 `H`와 Mean Pooling 사이**에 Transformer encoder를 넣고 나머지는 `mean`/`gnn`과 동일하게 둡니다. 구현과 단위 검증까지만 했으며 **추천 학습·평가는 실행하지 않았습니다.**

| `--scene-aggregation` | 실험 표기 | 장면 간 연산 | 확인하려는 효과 |
|---|---|---|---|
| `mean` | A | 없음 | 기준 |
| `gnn` | 참고 | cosine 유사도 GCN 1층 | 기존 제안 |
| `transformer_local` | C0 | C와 같은 블록, attention을 자기 장면으로 제한 | 파라미터만 늘린 대조군 |
| `transformer` | C | 영상 내부 모든 장면 attention, 위치 없음 | **C − C0 = 장면 간 상호작용** |
| `transformer_pos` | D | C + 유효 장면 순번의 위치 인코딩 | **D − C = 장면 순서** |

```text
[Title; Entity 평균; Action 평균; Context] 512 → Scene MLP → H (S × 512)
블록 1층(pre-LN, 512 · 2 heads · FFN 2048 · ReLU · dropout):
  X  = H + α₁ · Attn(LN₁(H + P))     C0는 attention을 자기 장면으로 제한
  H′ = X + α₂ · FFN(LN₂(X + P))      α₁, α₂: 학습 스칼라, 초기값 0
  C/C0: P = 0      D: P = 단위 노름 sinusoid(유효 장면 순번)
→ 유효 장면 평균 → 영상 512 → SASRec (변경 없음)
```

### 설계 근거

- **삽입 위치.** GCN과 같은 위치에 두어 `mean`·`gnn`·Transformer가 장면 간 연산 하나만 다르도록 했습니다. 입력은 제목이 포함된 Scene MLP 출력입니다. 제목은 영상 내 모든 장면에 같은 값으로 들어가므로 장면 간 attention 비교에 장면별 차이를 만들지 않습니다.
- **블록 크기.** 1층·2 heads·FFN 4배(2048)·ReLU·`validation.model.dropout`은 이 저장소의 SASRec 블록과 같습니다. 새 하이퍼파라미터를 들이지 않으려는 선택이며, 층수는 GCN 1층에 맞췄습니다. dropout 위치와 초기화(Linear xavier normal·bias 0, attention `in_proj` xavier normal)도 SASRec과 같습니다. `nn.TransformerEncoderLayer`에는 sublayer 출력에 스칼라를 곱할 지점이 없습니다. 추론 시 fused 경로가 sublayer 함수를 우회하기도 하므로 같은 구조의 블록을 직접 구현했습니다.
- **LayerNorm: 블록 내부 pre-LN 유지, 최종 LN 없음, 0 초기화 residual scale.** 합성 fixture의 초기화 직후 측정에서 Scene MLP 출력 `‖H‖`는 약 0.97이었습니다. LayerNorm은 이를 약 `√512 ≈ 22.6` 크기로 바꿉니다. 그래서 일반 pre-LN 블록의 attention·FFN 출력은 약 11.5·12.6으로 residual 입력을 압도합니다. 영상 벡터 노름도 16.95가 되어 `mean` 0.73, `gnn` 0.93, 제목만 있는 영상 0.58과 크게 달랐습니다. 이 상태로는 mean/gnn 비교에 크기 차이가 섞이고, 같은 모델 안에서도 장면 있는 영상이 내적 점수에서 구조적으로 유리해질 수 있습니다.
  - 최종 LN을 추가하면 22.49가 되어 해결되지 않습니다.
  - 블록 내부 LN을 제거하면 1.03이 되지만, attention 점수가 `‖H‖` 변화에 직접 노출됩니다.
  - 그래서 pre-LN으로 sublayer 입력 크기를 안정시키고, 출력은 학습 스칼라 `α`(초기값 0, [ReZero](https://arxiv.org/abs/2003.04887) 방식)로 조절합니다. 초기에는 세 변형이 모두 **`mean`과 정확히 같은 함수**이고, 이후 차이는 추천 loss로 학습한 장면 상호작용·순서 사용량입니다.
  - 최종 LN을 두지 않는 이유는 두 가지입니다. v7부터 유지한 Item·Scene LayerNorm 없음 계약을 따르고(v7에서 제거한 이유는 기록되어 있지 않음), 최종 LN이 있으면 `α = 0`에서도 mean과 같아지지 않기 때문입니다.
  - AdamW의 step당 갱신 폭은 대략 학습률(1e-4) 수준이라 `α`는 천천히 커집니다. 합성 300 step에서 `|α|`는 약 0.01이었습니다. early stopping 전에 Transformer가 충분히 쓰이지 않을 가능성이 있으므로 학습 후 `α`를 확인합니다(아래 `training.json` 항목).
- **C0 대조군.** 파라미터는 세 변형 모두 **3,152,386개**(블록 3,152,384 + `α` 2)로 GCN 층(262,656개)의 약 12배입니다. C가 좋아져도 용량 증가 때문일 수 있으므로, 파라미터가 같고 장면 간 정보 교환만 막은 C0를 둡니다. 대각선 mask에서는 attention 가중치가 자기 장면에 1이 되어 장면별 변환만 남습니다.
- **위치 인코딩.** 고정 sinusoid는 학습 파라미터가 없어 C와 D의 파라미터 수가 정확히 같고 장면 수 상한도 필요 없습니다. 원래 sinusoid의 노름은 `√(512/2) = 16`이라 `‖H‖ ≈ 0.97`인 장면 벡터에 그대로 더하면 LayerNorm 입력을 위치가 지배합니다. 초기 attention이 장면 내용을 거의 보지 못하게 되므로 **위치 벡터를 단위 노름으로 스케일**했습니다. 실제 BGE 분포에서 학습 중 `‖H‖`가 어떻게 변하는지는 측정하지 않았습니다.
- **위치는 sublayer 입력에만 주입.** `mean(P₀…P_{S−1})`은 유효 장면 수 `S`만의 함수입니다. 위치가 residual 경로에 남아 평균에 들어가면 D − C가 순서 효과가 아닌 길이 효과를 포함할 수 있습니다. 그래서 위치는 LN₁·LN₂ 입력에만 더합니다. 1층에서는 `Enc(H+P) − P`와 수식상 같고, `α = 0`이면 D도 부동소수점 상쇄 오차 없이 mean과 정확히 같습니다. sublayer를 통한 간접적인 위치·길이 정보까지 막지는 않습니다.
- **위치 기준은 유효 장면 순번.** 장면은 30초 고정 구간(`scene_duration: 30`)이라 원래 `scene_idx`가 시간에 가깝습니다. 하지만 현재 Graph 입력(`role-graph-inputs/v3`)에는 이 값이 저장되지 않아, 원래 위치를 쓰려면 입력 계약 변경과 재임베딩이 필요합니다. 제외된 장면(raw·warning·구조 오류)이 있는 영상에서만 두 기준이 달라집니다.
- **영상 내부 연결과 묶음 실행.** `gnn`과 같은 `scene_groups`로 장면 수가 같은 영상끼리 `(영상 수, S, 512)` 배치를 만듭니다. padding이나 key mask 없이 영상별 독립 계산과 같은 결과를 내며, 영상 간 attention은 없습니다. 장면 수 종류가 많으면 묶음별 kernel 호출이 늘어나는데, 실제 처리량은 측정하지 않았습니다.

### 예외·계약·캐시

장면 1개 영상도 Transformer를 거칩니다(자기 자신만 attention). 장면 없이 제목만 있으면 기존 `[title; 0; 0; 0]` fallback을 쓰고 Transformer를 건너뜁니다. 전체 결측·padding은 최종 0이며, `meta`는 Transformer를 적용하지 않는 동일 제목 모델입니다. C와 C0는 장면 순서를 바꿔도 영상 벡터가 같고 D만 달라집니다.

세 변형의 모델 계약은 `sasrec-role-graph/v11`입니다. leave-one-out 병합 때 origin의 mean `v9`와 겹치지 않도록 기존 `v9`에서 바꿨습니다. `graph_model.transformer`에 층·head·FFN·위치·attention 범위·정규화·residual scale·위치 스케일·위치 주입 방식을 기록하며, `meta`는 `transformer_applied: false`입니다. 현재 입력 계약 `role-graph-inputs/v8`와 BGE 특징 캐시는 `mean`/`gnn`과 공유하므로 aggregation별 **재임베딩이 필요 없습니다.** 추천 결과·checkpoint·shared cache는 aggregation별로 분리됩니다. `mean`/`gnn`의 identity·모델 버전·초기 가중치는 바뀌지 않았습니다. Chunk는 `gnn`과 같은 `chunk_scene_pairs`(S² 합) 기준을 따르고, auto checkpoint 추정에는 attention 점수와 FFN 활성값을 추가했습니다. Profiler는 `graph_encoder/scene_transformer`와 `workload.scene_pairs`를 기록합니다. 학습된 `[α₁, α₂]`는 `training.json`의 `execution.graph_training.scene_transformer_residual_scales`에 기록되며 checkpoint에도 포함됩니다. leave-one-out에는 refit이 없으므로 이전 rolling의 `graph_selection`·`graph_refit` 구분은 없습니다. 0에 가까우면 해당 sublayer를 거의 쓰지 않은 것입니다.

| 경로 | 신규 산출물 |
|---|---|
| 추천 | `validation/recommendations/graph/<transformer\|transformer_pos\|transformer_local>/<date>/seed_<seed>/<arm>/` |
| 진단 | `validation/diagnosis/graph_<aggregation>_diagnosis.json` |

```bash
RUN_ID="사용할_Run_ID"
for AGG in transformer_local transformer transformer_pos; do
  python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation "$AGG" --workers-per-gpu 1 --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
  python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation "$AGG" --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
done
python -m pytest -q tests/test_models.py tests/test_representations.py
```

진단은 aggregation 하나 안에서 arm끼리 비교합니다. C0·C·D 사이의 날짜×seed 짝 비교는 아직 자동화하지 않았으며, Run별 분석 코드와 보고서는 `artifacts/runs/<RUN_ID>/reports/`에 둡니다.

### 이번 변경의 검증

로컬 PC(NVIDIA RTX 4090, PyTorch 2.13.0+cu130)에서 다음이 통과했습니다.

| 항목 | 결과 |
|---|---|
| CPU validation | **134개** (변경 전 HEAD 104개 + 신규 30개) |
| integration | 41개 (HEAD와 동일) |
| Transformer·GCN 테스트 (CPU+CUDA) | 55개 |
| 기존 CUDA 회귀 (`graph_execution`, `large_optimizations`, `-k cuda`) | 9개 (HEAD에서도 같은 명령이 9개 선택) |
| CUDA profiler (별도 프로세스) | 4개 |

신규 테스트는 다음을 확인합니다.

- 원시 가중치와 residual scale로 작성한 pre-LN attention·FFN 계산과의 일치(학습·추론 경로), 단위 노름 sinusoid
- 묶음 계산과 영상별 단독 계산의 일치, 영상 격리
- C/C0의 장면 순열 등변성과 D의 순서 의존성, C0에서 다른 장면이 결과에 섞이지 않음
- `α = 0`이면 블록이 정확한 항등. 세 변형의 초기 catalog 벡터가 같은 가중치의 `mean`과 비트 단위로 일치
- `α = 0`에서 gradient가 `α`로만 흐르고 한 step 뒤 sublayer 가중치로 전달됨, execution report 기록
- 세 변형의 동일 파라미터 수, fallback·빈 영상 우회
- chunk·checkpoint 출력과 gradient 일치, dropout 학습과 checkpoint 재계산의 일치, 저장·재로딩
- 소형 selection→refit→test→diagnosis와 결과 캐시, identity·cache key 분리, CLI 선택지

`mean`·`gnn`·Text Metadata·Text Desc 모델은 같은 seed에서 변경 전 HEAD와 초기 state 전체·catalog 벡터가 비트 단위로 일치했습니다.

합성 fixture(영상 64, 장면 315, batch 128, lr 1e-4, dropout 0.1)에서 GPU로 300 step 학습해 동작을 확인했습니다. 장면 있는 영상 노름은 세 변형 모두 초기 0.728(`mean`과 동일)에서 1.06–1.08로 변했고(`mean` 0.97), `|α|`는 약 0.01이었습니다. 합성 목표이므로 추천 품질과는 무관합니다. benchmark는 CUDA 소형 실행으로 동작만 확인했습니다.

추천 품질, 실제 처리량, 실제 데이터에서의 `α` 성장, 학습 중 `‖H‖` 대비 위치 크기는 검증하지 않았습니다. 추천 구조 그림(`docs/design/`)은 갱신하지 않았습니다.

후속 후보(미구현): 독립 점수 attention pooling(B), 작은 GRU(E), 원래 `scene_idx` 위치, 2층 변형, 장면 단위 Description 입력에 같은 aggregation 적용.

## 장면 GCN aggregation (2026-10-06)

`mean`은 위 현재 구조를 사용하며, `gnn`은 기존 장면 MLP 출력 뒤에 영상 내부 메시지 전달을 추가합니다. [DCGN §3.1](https://arxiv.org/html/1906.00377#S3.SS1)의 유사도 그래프를 참고한 추천용 변형입니다. 시간 convolution·계층적 pooling·분류 head를 재현한 모델은 아닙니다.

```text
[Entity 평균; 역할 보정 Action 평균; Context] 384 → 장면 유사도
[Title; Entity 평균; Action 평균; Context] 512 → Scene MLP → GCN → Mean → 영상 512
```

유사도 입력에는 제목이 없습니다. 각 영상의 모든 유효 장면 쌍에 대해 현재 384차원 특징의 cosine을 계산하고 음수는 0, self-loop는 1로 설정합니다. `eps=1e-8`의 L2 정규화로 영벡터를 처리하고, `A`의 행합 `D`로 `Â = D⁻¹ᐟ² A D⁻¹ᐟ²`를 만듭니다. 1층 residual GCN은 `H′ = H + ReLU(Â H W + b)`이며 `W`는 512→512입니다. 이후 유효 장면 평균을 구합니다. 별도 LayerNorm·attention·gate·dropout과 시간 위치는 없습니다.

유사도는 매 forward 다시 계산하며 gradient를 차단하지 않습니다. 기존 projection·역할 결합·Scene MLP·GCN·SASRec을 추천 loss로 학습하고 BGE는 계속 동결합니다. 단일 장면도 self-loop GCN을 거칩니다. 장면 없이 제목만 있는 영상은 기존 fallback을 사용하고 GCN을 건너뜁니다. 전체 결측·padding은 최종 0입니다. `meta`는 동일 제목 전용 모델이며 GCN을 적용하지 않습니다.

장면 수가 같은 영상들을 묶어 batched matrix 연산을 수행하고 영상 간 연결을 만들지 않습니다. 정적 묶음 인덱스는 checkpoint 재계산에서 재사용하지만 학습된 유사도는 optimizer 갱신 이후 재사용하지 않습니다. `validation.graph_execution.chunk_scene_pairs` 기본값은 **1048576**이며 기존 item/node 제한과 함께 영상별 `S²` 합으로 chunk를 나눕니다. 단일 영상이 기준보다 크면 단독 처리하며 장면을 삭제하지 않습니다. 이 기준은 엄격한 메모리 상한이 아닙니다. Auto checkpoint 추정에는 유사도·GCN 활성값도 포함합니다.

Profiler는 `graph_encoder/scene_similarity`, `graph_encoder/scene_gcn`과 `workload.scene_pairs`를 기록합니다. Checkpoint 재계산이 수집되면 해당 연산·장면 쌍도 누적됩니다.

GCN 최초 구현 당시 Text·mean v7과 GNN v8을 사용했습니다. 영상 MLP를 1층으로 변경한 현재 계약은 Text v8·mean v9·gnn v10이며 유사도·정규화·residual·층수와 `meta`의 GCN 미적용을 계약에 기록합니다. 입력 계약 `role-graph-inputs/v8`와 BGE 특징 캐시는 공유합니다. 다른 aggregation의 결과/checkpoint를 재개에 사용하지 않습니다. `--compare-run-id`는 같은 aggregation·모델 계약끼리 비교합니다. Mean/gnn 자동 통계 비교는 별도 후속 작업입니다.

| 경로 | 신규 산출물 |
|---|---|
| 추천 | `validation/recommendations/graph/gnn/seed_<seed>/<arm>/` |
| 진단 | `validation/diagnosis/graph_gnn_diagnosis.json` |

유효한 현재 Graph 입력이 있다면 재임베딩 없이 추천부터 실행합니다. GNN은 selection부터 새 학습이며 기존 mean·과거 attention 산출물은 보존합니다.

```bash
RUN_ID="사용할_Run_ID"
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation gnn --workers-per-gpu 1 --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation gnn --target meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta
```

아래 과거 mean/gnn 학습 비용 측정에는 같은 synthetic 입력과 공통 가중치를 사용하고 profiler를 껐습니다. BGE나 Run 입력은 읽지 않았으며 실제 추천 품질·RTX 6000 Ada 처리량을 검증한 실험이 아닙니다. 해당 측정 도구 `benchmarks/scene_gcn_benchmark.py`는 2026-10-15 보조 개발 도구 정리에서 제거했습니다. 기존 측정 결과와 검증 기록은 보존합니다.

```bash
GRAPH_TEST_CUDA=cuda:0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/validation/test_scene_gcn.py
```

다음은 2층 영상 MLP를 사용한 2026-10-06 GCN 구현 당시의 검증입니다. `vc_cloud`, PyTorch 2.13.0+cu129에서 CPU validation **171개**, CLI/profiling integration **14개**, GCN CPU/CUDA 검사 **16개**, 기존 mean CUDA 회귀 **12개**가 통과했습니다. GCN 수학은 독립 NumPy 계산과 비교했으며 작은 selection→refit→test→diagnosis, 결과 재사용, checkpoint 재로딩, 유사도 gradient, 영상 격리, 순열 및 결측 처리를 검사했습니다. 변경 전 HEAD의 mean 모델과 같은 seed·입력에서 초기 state 전체·catalog 벡터·추천 점수가 정확히 일치했습니다. 관련 Ruff와 `git diff --check`도 통과했습니다.

Profiler를 끈 synthetic 측정은 영상 64개, 유효 장면 315개, 장면 쌍 2105개, batch 128, 고정 특징 1024, 추천 차원 512, seed 42, dropout 0을 사용했습니다. 공통 모델 가중치를 맞추고 순전파·loss·backward·gradient clipping·AdamW·캐시 초기화를 포함했습니다. 3회 반복의 각 배치 시간 중앙값을 다시 중앙값으로 집계했습니다. CPU는 warmup 3/측정 5배치, CUDA는 warmup 5/측정 10배치이며 CUDA는 배치 전후 동기화했습니다.

| 환경 | mean 배치 시간 | gnn 배치 시간 | mean / gnn CUDA peak allocated |
|---|---:|---:|---:|
| CPU, 1 thread | 609.0 ms | 642.7 ms | 해당 없음 |
| NVIDIA L4 | 21.1 ms | 44.6 ms | 282.0 / 288.4 MiB |

이 소형 CUDA workload에서 GNN 시간이 약 2.1배였습니다. 장면 수별 연산 묶음과 `S²` 비용은 실제 데이터 분포에 따라 달라집니다. 전체 Run, RTX 6000 Ada 처리량, 추천 품질 개선, 대량 BGE 생성은 검증하지 않았습니다. 기존 입력·mean 결과·과거 attention 산출물과 다이어그램은 변경하지 않았습니다.

## 영상 Residual MLP 1층 전환 (2026-10-07)

모듈 2의 Item·Scene MLP를 `x + ReLU(Linear(x))`로 변경했습니다. Metadata·Text는 영상별 Item MLP, Graph mean/gnn은 장면별 Scene MLP에 적용합니다. 기존 2층 GELU User MLP와 SASRec, 역할 결합·GCN, 학습·평가 조건은 유지합니다.

| 항목 | 이전 | 현재 |
|---|---|---|
| 영상 Residual MLP | `x + Linear(ReLU(Linear(x)))` | `x + ReLU(Linear(x))` |
| Linear 수 / 512차원 파라미터 | 2 / 525,312개 | 1 / 262,656개 |
| 모델 계약 | Text v7 / mean v7 / gnn v8 | Text v8 / mean v9 / gnn v10 |

영상 MLP에는 LayerNorm을 추가하지 않았습니다. 전체 결측·padding은 MLP 이후 마스킹하므로 bias가 학습되어도 최종 출력은 0입니다. 제목 없는 Arm도 기존 concat 순서와 역할 처리를 유지합니다. GNN의 유사도는 계속 Scene MLP 이전의 제목 제외 384차원 특징으로 계산합니다.

모델 identity·checkpoint·완료 판정·shared cache·진단은 새 계약을 사용하여 2층 모델 결과와 혼용하지 않습니다. 원본·encoder가 일치하는 Text/Graph BGE 입력은 재사용할 수 있습니다. 추천은 모든 선택 Arm에서 **selection부터 재학습**하고 진단을 다시 생성합니다. 기존 결과를 보존하려면 별도 Run을 사용하거나 재실행 전에 보관하십시오. 모델 변경 자체는 기존 산출물을 삭제하지 않습니다.

```bash
RUN_ID="사용할_Run_ID"
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode text --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation gnn --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta
```

`vc_cloud`의 CPU validation **105개**, NVIDIA L4 CUDA 검사 **14개**, CLI·README integration **28개**가 통과했습니다. 독립 NumPy 수식·gradient, 결측·padding, projection/encoder/SASRec 학습, Text/Graph Meta 일치, mean/gnn chunk·checkpoint 및 재로딩, 구버전 계약 거부, 프로파일링 회귀를 검사했습니다. 변경 파일 Ruff와 `git diff --check`가 통과했습니다. mean·gnn PPTX/PNG를 함께 갱신하고 편집 가능한 객체·텍스트 렌더링·경계를 확인했습니다.

이는 소형 검증이며 전체 Run 학습·추천 품질·학습 시간 감소 폭은 측정하지 않았습니다. 위 2026-10-06 benchmark는 **이전 2층 영상 MLP**의 측정값입니다. MLP 파라미터가 절반이라는 사실만으로 전체 학습이 두 배 빨라졌다고 해석하지 않습니다.

## 이전 학습 비용 조정 (2026-09-30)

이 절은 과거 v2 설정 기록입니다. 현재 학습 조건은 위 모델 계약을 따릅니다.

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

## 이전 Graph 실행 최적화 (2026-09-30)

이 절은 packed v1 구현 기록이며 현재 역할 조회는 위 packed v2 계약을 따릅니다.

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

아래는 이전 v3의 변경 이력입니다. 현재 계약은 위 모델 및 데이터 계약을 따르며 이 절의 구조·재개 설명은 과거 버전에 해당합니다. 당시 모델 계약은 **`sasrec-role-graph/v3`**였습니다.

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

`run-recommendation`은 기본 로그에 train epoch의 학습·validation 시간과 학습 events/s,
최적 checkpoint 복원 후 test 시간을 표시합니다. 더 자세한 측정은
**`--profile-every N`**으로 켭니다. text/graph 모두 지원하며 다른 step에서는 거부합니다.

```bash
python -m validation run-recommendation \
  --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean \
  --target graph_qwen3vl --workers-per-gpu 1 --profile-every 100
```

각 train epoch와 validation/test에서 **첫 3개 배치 및 N의 배수 배치**를 측정합니다.
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
| `graph_items/graph_encoder/title_projection` | 영상별 제목 Linear projection |
| `…/entity_projection`, `…/action_projection` | Entity·Action의 독립 projection |
| `…/role_fusion`, `…/role_fusion/state_lookup` | Entity→Action 역할 결합·결측 상태 조회 |
| `…/scene_readout`, `…/scene_mlp` | Context projection·네 성분 concat·Scene Residual MLP |
| `…/video_pooling` | 유효 Scene Mean Pooling |
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

### Projection 역전파 operator 측정

`--profile-every`에 **`--profile-operators`**를 함께 지정하면 각 조합/session에서
첫 번째 측정 가능한 train batch 3 이상을 한 번만 PyTorch profiler로
기록합니다. 일반적인 배치 수에서는 selection epoch 1 batch 3입니다. epoch당 배치 수가
3 미만이면 operator trace는 생성되지 않습니다. 일반 표본 로그는 기존 간격대로 유지합니다.

```bash
python -m validation run-recommendation \
  --run-id 260930_v8 --representation-mode graph --scene-aggregation mean \
  --target graph_gemini_meta --workers-per-gpu 1 \
  --profile-every 100 --profile-operators
```

각 조합 결과 디렉터리의 `profile_traces/<session_id>/`에 다음 파일을 저장합니다.

- `selection_epoch_1_batch_3.trace.json`: CPU operator·입력 shape·CUDA 실행 trace.
  Chrome/Perfetto 형식이며 `profile::` 구간과 forward/backward 연결을 확인할 수 있습니다.
- `selection_epoch_1_batch_3.operators.json`: shape별 operator 통계와
  `projection_backward` 요약. CPU 실행이면 device 시간은 0이고 device 비율은 null입니다.
- `profile.jsonl`의 해당 배치에는 `operator_profiled`, `operator_trace`,
  `operator_summary`를 기록합니다. start에는 `operator_profiling` 설정을 추가합니다.
  기존 v2 형식의 필드는 유지합니다. 이 파일들은 공유 추천 캐시에 포함하지 않습니다.

`projection_backward.projection_nodes`는 독립 Entity·Action·Context·Title projection의
forward sequence number와 forward thread를 autograd backward node에 연결해 집계합니다. Text 경로는
title/video projection, metadata 경로는 item projection을 기록합니다.
Checkpoint 재계산이 있으면 해당 forward projection은 `backward/...` 경로로 표시됩니다.
요약의 `node_types`와 trace를 함께 확인하십시오. 파라미터 gradient 누적과 checkpoint
재계산 forward 시간은 projection backward node 비용에 포함하지 않습니다.

CPU·device 시간의 단위는 µs입니다. 요약의 projection 비율은 **모든 autograd backward
node의 inclusive CPU/device 시간 합 대비 비율**입니다. Engine wrapper는 제외해
wrapper와 node를 이중 합산하지 않습니다. 이 비율은 기존 `backward` wall time의 비율이
아니며 CPU 시간과 device 시간을 더하거나 inclusive operator 행들을 모두 더하지 않습니다.
특히 checkpoint 중첩 node나 CUDA 실행 겹침이 있으면 node 시간 합은 elapsed time과 다릅니다.
CPU 비율에는 host 실행 비용이 포함되므로 GPU 계산 비용에는 device 비율과 trace를 사용합니다.

이 옵션은 무거운 진단용입니다. Operator 측정 배치의 wall time에는 profiler 시작·종료
비용이 포함되므로 정상 상태 배치 통계에서 제외하십시오. 기록·파일 저장 비용은 epoch 시간에도
영향을 줍니다. 실제 CUDA operator 수집은 서버의 PyTorch/CUPTI 지원과 권한에 의존하며,
device 시간이 수집되지 않았다면 CPU 비율로 GPU projection 비중을 대신 판단하지 않습니다.
측정 구간은 한 배치이므로 전체 epoch의 projection 비중으로 확정하지 않습니다.


## 이전 모델 v6: BGE Large 전용 전환

이 절의 구조·결정성 설정은 과거 기록입니다. 현재 모델은 위 모델 계약을 따릅니다.

현재 지원 가중치는 `/home_nvme/shared/models/bge-large-en-v1.5`이며 BGE 출력은 1024차원입니다.
[Large 구조 PNG](design/recsys_diagram_large.png)와 [편집 가능한 PPTX](design/recsys_diagram_large.pptx)를 기준으로 합니다.
Small/Large 선택 옵션은 없습니다. `EncoderConfig`는 1024만 허용하고 실제 모델의 `hidden_size`와
출력 배열 차원이 설정과 일치하는지 검사합니다. 서버 경로가 다른 환경에서는 테스트용 설정 파일의
`models.bge`만 실제 로컬 가중치 경로로 지정합니다. 자동 다운로드는 하지 않습니다.

| 경로 | 현재 학습 구조 |
|---|---|
| Baseline `meta` | 제목 BGE 1024 → Linear 512 → LayerNorm 512 |
| Text | 제목 BGE 1024 → Linear 128; Summary BGE 1024 → Linear 384 → concat 512 → LayerNorm 512 |
| Graph | Entity·Action·Context BGE 1024 → 공유 Linear 128; Entity·Action만 node-type embedding 및 1층 역할별 Graph Encoder; 장면 concat 384 → mean/attention 384 |
| Graph 제목 결합 | 제목 BGE 1024 → Linear 128; `[제목 128; 영상 384]` → LayerNorm 512 |

Text의 제목과 Summary는 독립적으로 BGE에 입력하며 Summary projection도 추천 loss로 학습합니다.
BGE 특징은 고정합니다. 제목 없는 Arm은 concat 전 왼쪽 128차원을 0으로 마스킹합니다.
부분 결측은 concat 전에, 전체 결측·padding은 최종 LayerNorm 이후 다시 마스킹합니다.
아이템 MLP는 없으며 SASRec 512·2 blocks·2 heads, User MLP, loss·평가·selection/refit 조건은 유지합니다.
역할 상태의 `F.embedding` 조회, chunk/checkpoint/GPU 특징 캐시 최적화와 결정성 기본값 `false`도 유지합니다.

Text 모델은 `sasrec-content-v6`, Graph 모델은 `sasrec-role-graph/v6`입니다.
Text 분리 특징 계약 `shared-scenes-representation/v4`와 Graph 입력 계약 `role-graph-inputs/v2` 및
파일 키·CLI·저장 경로는 유지합니다. Text의 두 특징 배열과 Graph의 `features.npy`는 float32 1024차원입니다.
BGE 설정·가중치 식별·입력 내용·catalog와 실제 배열 차원을 확인해 small 특징을 배제합니다.
과거 large 특징도 현재 입력과 provenance가 모두 일치할 때만 재사용하며 차원만 같다고 허용하지 않습니다.
모델 버전·입력 차원·projection 정책은 결과/checkpoint/profiler identity에 포함되므로 이전 모델 결과로
완료 판정하거나 재개하지 않습니다. 진단과 Run 비교도 현재 모델 계약을 확인합니다.
Text profiler에는 학습 가능한 Summary 변환을 `video_projection` 구간으로 기록합니다.

### 변경 위치와 근거

| 위치 | 변경 및 근거 |
|---|---|
| `config.yaml`, `src/validation/config.py`, `features.py` | Large 경로·1024 고정 및 실제 hidden size 검사: small 입력 혼용 방지 |
| `src/validation/model.py` | Text Summary Identity를 Linear 1024→384로 교체: Large 특징을 최종 영상 슬롯에 학습 투영 |
| `src/validation/graph_model.py`, `graph_context.py` | 공유 projection 기본 입력 1024·Graph v6: 확정 다이어그램과 기존 최적화 유지 |
| `src/validation/recommendation_contracts.py`, `representation_provenance.py` | 모델 v6·input_dim 1024·Text video_transform=linear: 이전 결과/checkpoint 혼용 방지 |
| `src/validation/graph_inputs.py` | 재사용 및 실행 전 실제 feature shape/dtype 검사: 해시·체크섬이 일치해도 384차원 특징 거부 |
| `tests/validation/`, `tests/conftest.py` | production 입력 1024, 학습·결측·캐시·checkpoint·프로파일 회귀 검증 |

### 이전 v6 Text·Graph 재실행 명령

두 모드 모두 **embed-representations부터** 실행하고 선택한 모든 Arm을 selection부터 재학습합니다.
기존 extraction과 Summary를 재생성할 필요는 없습니다. 같은 Run을 사용하면 기존 특징·추천 결과·진단이
갱신되므로 이전 산출물이 필요하면 **실행 전에 별도로 보관**하십시오. 일괄 삭제는 하지 않습니다.
아래 `RUN_ID`는 실제 사용할 Run으로 지정합니다. 명령은 전체 날짜×seed 실험을 실행합니다.

```bash
conda activate vc_cloud
export RUN_ID=260930_v8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Text: 독립 제목·Summary 특징을 Large로 준비한 뒤 전체 선택 Arm 학습
python -m validation embed-representations --run-id "$RUN_ID" --representation-mode text --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode text --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode text --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta

# Graph: 고정 특징 준비를 mean/gnn이 공유
python -m validation embed-representations --run-id "$RUN_ID" --representation-mode graph --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --target meta graph_qwen graph_qwen_meta graph_gemini_meta
```

이 절은 과거 v6 기록입니다. 현재 `attention` 실행은 지원하지 않으며, 신규 GNN 실행은 아래가 아닌 상단의 현재 계약을 따릅니다.
기존 Gemini Graph의 전체 결측 문제는 별도 입력 문제입니다. 재학습 전 해당 Arm의
`statistics.json`에서 유효 장면 수와 제외 사유를 확인해야 하며 이번 모델 변경이 이를 해결하지 않습니다.

### 구현 검증 (2026-10-02)

- `vc_cloud` CPU validation: **160개 통과**. 이후 빈 특징 파일 복구 case를 추가해 Graph 캐시 관련 검사도 별도로 실행했습니다.
- NVIDIA L4 / CUDA: 1024차원 production 모델 및 기존 Graph packing/checkpoint 회귀 **40개 통과**(CPU·CUDA parametrization 포함). CUDA 전용 Text/mean/attention 프로파일 회귀 **6개 통과**. 대량 입력·RTX 6000 Ada 처리량은 측정하지 않았습니다.
- 실제 로컬 `/home/junsu2.choi/workspace/models/bge-large-en-v1.5`로 CUDA 인코딩: `hidden_size=1024`, 짧은 문장 2개의 float32 `(2, 1024)` 출력·finite·L2 norm=1 확인. `vc_cloud`에 embedding 의존성 `transformers` 4.57.6을 설치했으며 모델은 로컬 가중치만 읽었습니다. 서버 기본 경로는 변경하지 않았습니다.
- Ruff 및 `git diff --check` 통과. 프롬프트·그림·기존 Run 산출물은 수정하지 않았으며 전체 임베딩·추천 실험은 실행하지 않았습니다.
- 추가 integration 검사: **68개 통과, 11개 실패**. 실패는 기존 `run_pipeline_v7.sh`/`script_graph_v7.sh` 및 그림 링크 누락, 과거 v7 semantic/representation hash fixture 불일치입니다. 변경 전 HEAD의 임시 복사본에서도 동일 실패를 재현했고 이번 범위에서 고치지 않았습니다.
- 이 환경에서 CPU와 CUDA operator profiler를 같은 프로세스에서 순서대로 실행하면 GPU backward 시간이 0으로 수집되어 6개 검사가 실패합니다. 변경 전 HEAD에서도 재현됐고 CUDA 전용 새 프로세스에서는 6개 모두 통과했습니다. GPU 비용 해석에는 실제 device 시간 수집 여부를 확인해야 합니다.

재현 명령(전체 실험 및 실제 가중치 대량 인코딩 제외):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/validation
CUBLAS_WORKSPACE_CONFIG=:4096:8 GRAPH_TEST_CUDA=cuda:0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/validation/test_large_optimizations.py tests/validation/test_graph_execution.py
# CUDA operator profiler는 CPU 테스트와 분리한 새 프로세스에서 실행
CUBLAS_WORKSPACE_CONFIG=:4096:8 GRAPH_TEST_CUDA=cuda:0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/validation/test_profiling.py -k cuda
python -m ruff check src/validation tests/validation tests/conftest.py
```

## 이전 모델 v5: BGE small 전환과 state lookup 최적화

아래는 과거 small 전환 기록입니다. 현재 실행에는 적용되지 않으며, 위 모델 계약을 따릅니다. 당시 실행은 `/home_nvme/shared/models/bge-small-en-v1.5`의 384차원 특징만 지원합니다.
설정 계약은 `validation-config/v6`, Text 모델은 `sasrec-content-v5`, Graph 모델은
`sasrec-role-graph/v5`, 학습 구현은 `shared-scenes-training-evaluation/v5`입니다.
Text의 분리 입력 포맷(v4)과 Graph 입력 포맷(v2)은 유지하지만 BGE 모델 식별과 차원이
특징 해시에 포함되므로 기존 large 특징을 재사용하지 않습니다. 1024/768차원 Encoder 설정,
실제 모델의 다른 hidden size, 저장 특징 배열의 차원 불일치는 거부합니다.

| 경로 | 학습 구조 |
|---|---|
| Baseline | 제목 BGE 384 → Linear 512 → LayerNorm 512 |
| Text | 제목 384 → Linear 128; Summary 384 → Identity → concat 512 → LayerNorm 512 |
| Graph | 노드·Context 384 → 공유 Linear 128; 역할별 연산 → Entity·Action·Context concat 384 → mean/attention 384 |
| Graph 제목 결합 | 제목 384 → Linear 128 + Graph 영상 384 → concat 512 → LayerNorm 512 |
| 공통 추천기 | SASRec 512, 2 blocks, 2 heads; User MLP 512 유지 |

Text Summary는 frozen BGE 값을 그대로 사용하며 `video_projection`의 학습 가중치는 없습니다.
Graph의 Context는 노드와 **동일한 Linear 모듈**을 사용하며, 제목 projection은 별도입니다.
결측 구성 요소는 concat 전에 마스킹하고, 전체 결측과 padding은 최종 LayerNorm 뒤에
다시 0으로 만듭니다. 제목 없는 Arm은 기존처럼 제목 슬롯 128이 0입니다.

`validation.model.deterministic: false`가 기본값이며 모든 Text·Graph Arm에 적용됩니다.
Python·NumPy·PyTorch seed는 계속 고정합니다. Selection과 refit 각각 초기화할 때,
병렬 worker 안에서도 같은 설정으로 `torch.use_deterministic_algorithms`를 호출합니다.
`true`로 바꾸면 강제 모드를 사용할 수 있으며, CUDA에서는 PyTorch/cuBLAS 실행 조건에 따라
시작 전에 `export CUBLAS_WORKSPACE_CONFIG=:4096:8`이 필요합니다.
설정값은 training/checkpoint metadata와 profiler 시작 레코드에 남습니다.
모델 구조·입력 차원·BGE 모델 식별·결정성 값 및 학습 구현 버전은 결과 캐시 식별에 반영됩니다.
결정성 설정을 바꾸면 같은 특징으로도 추천 결과를 재학습합니다.

역할별 `states`는 이름·shape `(5, 2, 128)`·초기화를 유지하며 lookup만 `F.embedding`으로
교체했습니다. `sparse=False`, `scale_grad_by_freq=False`, `max_norm=None`, `padding_idx=None`으로
일반 optimizer와 누적 gradient의 수학적 의미를 유지합니다. 역할 순서와 메시지 누적도 같습니다.
Profiler의 `graph_encoder/message_passing/state_lookup` 구간과 operator JSON의
`EmbeddingBackward`/`aten::embedding_dense_backward` 비용으로 확인합니다.
Chunk 확대, compile, 추가 캐싱은 적용하지 않았습니다.

### 서버 B에서 특징 재생성과 재학습

기존 extraction·Summary·보고서와 large 그림은 보존합니다. 기존 결과도 보존하려면 새 Run에
기존 extraction만 복사하고 small 특징과 추천 결과를 생성할 수 있습니다. 아래는 그 예입니다.
새 Run 디렉터리가 아직 없는 상태에서 실행하며, 데이터 경로와 shared preparation cohort는
원본 Run과 동일해야 합니다. 모델은 서버에 미리 준비하고 네트워크 다운로드는 사용하지 않습니다.

```bash
conda activate vc_cloud
export RUN_ID=261001_v8_small
mkdir -p "artifacts/runs/$RUN_ID"
cp -a artifacts/runs/260930_v8/extraction "artifacts/runs/$RUN_ID/extraction"

# Text: 모든 frozen 특징을 small로 생성하고 selection부터 학습합니다.
python -m validation embed-representations --run-id "$RUN_ID" --representation-mode text --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode text --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode text --target meta graph_qwen graph_qwen_meta graph_gemini_meta desc_qwen_meta desc_gemini_meta

# 과거 small 특징 생성 명령입니다. 현재 모델은 Large이며 attention 실행은 지원하지 않습니다.
python -m validation embed-representations --run-id "$RUN_ID" --representation-mode graph --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-recommendation --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-diagnosis --run-id "$RUN_ID" --representation-mode graph --scene-aggregation mean --target meta graph_qwen graph_qwen_meta graph_gemini_meta
```

기존 Run에 직접 실행하면 small 특징과 새 추천 결과가 해당 경로를 갱신합니다.
`--force` 없이도 large 캐시는 배제합니다. 기존 large checkpoint는 small 모델에 로드할 수 없습니다.
각 Graph 입력의 `statistics.json`에서 유효 장면 수를 확인하세요. 빈 Graph의 영벡터 처리도
유지되므로 학습 완료만으로 Graph 정보가 사용되었다고 판단할 수 없습니다.

### 두 최적화의 독립 측정

[측정 코드](../artifacts/runs/260930_v8/reports/benchmark_small_graph_optimizations.py)는
동일 small 구조에서 결정성 on/off × indexing/embedding의 4개 조합을 비교합니다.
입력 배치·초기 state_dict·seed·warm-up 횟수가 같고, 반복마다 조합 순서를 섞습니다.
원래 indexing은 측정 코드에만 존재하며 제품 경로에 선택 설정을 추가하지 않습니다.
학습 처리 시간은 **profiler를 끄고** CUDA 동기화 후 측정하며, operator capture는
별도 warm-up 이후 배치에서 수행합니다. 결과에는 입력·가중치 hash, 장치·PyTorch 버전,
평균 배치 시간의 반복별 값과 중앙값, events/s, operator trace를 저장합니다.
실제 small Graph 특징을 생성한 뒤 같은 서버에서 실행하세요.
측정 코드는 Git 제외 경로인 `artifacts/` 아래에 있으므로 서버 B에도 해당 파일을 복사합니다.

```bash
python artifacts/runs/260930_v8/reports/benchmark_small_graph_optimizations.py \
  --run-id "$RUN_ID" --arm graph_qwen_meta --aggregation mean --device cuda:0 \
  --warmup 5 --steps 30 --repeats 3 --operators
```

산출물은 해당 Run의 `reports/small_optimization_benchmark_<timestamp>/`에 저장합니다.
특정 selection 구간은 `--date YYYY-MM-DD`로 지정합니다. 전체 실험 평가나 품질 비교는 하지 않습니다.
`--device cpu --synthetic-items 32 --batch-size 8 --warmup 1 --steps 2 --repeats 1`은
실제 BGE 호출 없이 측정 코드의 동작을 확인하는 smoke 검사이며 성능 근거로 쓰지 않습니다.
CPU와 현재 VM의 단위 검증은 실제 서버 B의 속도 개선이나 NDCG/HR 품질 변화 측정을 대신하지 않습니다.

현재 VM에서는 CPU·CUDA operator 검사를 같은 프로세스에서 실행했을 때 일부 GPU 비용이
0으로 기록되는 현상이 관찰됐습니다. 별도 CUDA 프로세스에서는 kernel 비용을 확인했습니다.
측정 JSON의 `operator_device_timing_available`이 false이면 해당 operator 비용을 해석하지 않고,
profiler를 끈 wall time만 사용합니다.

Small 구조 그림: [편집 가능한 PPTX](design/recsys_diagram_small.pptx),
[PPTX에서 렌더링한 PNG](design/recsys_diagram_small.png).
[생성 코드](design/generate_recsys_diagram_small.py)는 large와 small 모두에서 Entity·Action·Context가
하나의 Shared Linear Projection 상자를 통과하도록 표시합니다. 노드 종류 embedding과 Role Graph
Encoder는 Entity·Action 분기에만 적용하고, Context 분기는 장면 concat에 직접 연결합니다.
두 PPTX를 편집 가능한 도형·연결선으로 저장하고 LibreOffice PDF를 거쳐 PNG로 렌더링합니다.
실행에는 `python-pptx`, `PyMuPDF`, LibreOffice Impress가 필요합니다.

## 이전 모델 v4: Baseline·Text·Graph 최종 구조

아래는 large 특징을 사용한 v4 기록입니다. 현재 실행은 위 모델 계약을 따릅니다.

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

# 과거 v4 명령입니다. 현재 attention 실행은 지원하지 않습니다.
python -m validation run-recommendation --run-id 260928_v7 --representation-mode graph --scene-aggregation mean --workers-per-gpu 1 --target meta graph_qwen graph_qwen_meta graph_gemini_meta
python -m validation run-diagnosis --run-id 260928_v7 --representation-mode graph --scene-aggregation mean --target meta graph_qwen graph_qwen_meta graph_gemini_meta
```

Graph 입력 계약은 그대로입니다. 입력 파일·제목·BGE가 바뀌지 않았다면 재임베딩하지 않습니다. 이전 Gemini 프로파일에서 catalog 전체 Graph가 비어 있었으므로, 전체 실험 전에 해당 입력의 `statistics.json`에서 유효 장면 및 제외 사유를 확인해야 합니다. v4 모델 변경이 그 입력 문제를 해결하지는 않습니다.

## Leave-one-out 전환 (2026-10-07)

현재 평가 조건은 위 leave-one-out 계약입니다. 문서에 남은 과거 rolling·selection/refit 검증 기록은 당시 결과이며 새 평가 방식의 전체 데이터 성능을 입증하지 않습니다. 같은 Run의 기존 extraction 파일을 사용하고 새 validation 경로에서 특징 준비·학습·진단을 실행할 수 있습니다. 날짜가 없는 3 seed별 학습 결과를 사용자 평균으로 집계하며, 기존 paired bootstrap·Bonferroni 비교군을 유지합니다.

Graph 최적 checkpoint를 복원한 직후 Item/catalog 캐시를 폐기하고 test catalog를 다시 계산합니다. `training.json`에는 `execution.graph_training`과 train/validation/checkpoint_restore/test 시간을 기록합니다. profiler는 train·validation·test의 첫 3개 배치와 N의 배수 배치를 측정하며, operator trace는 train batch 3 이상에서 수집합니다. 기존 rolling checkpoint는 새 프로토콜·분할·학습 구현 identity와 schema 검증을 통과할 수 없습니다.

2026-10-07 구현 검증: 현재 VM의 `vc_cloud`에서 CPU·단일 스레드 전체 pytest **284개**가 통과했습니다. 이력 4/5/13/14개, 비정렬 원본·동일 timestamp·중복 보존·원본 매핑, holdout 전용 아이템과 train 빈도 분리, 중간 최적 epoch·동점의 Text/mean/GNN checkpoint 및 catalog 복원, 3 seed × 6 Arm 집계·직렬/병렬·중단 복구·캐시 불일치·기존 rolling 파일 보존을 검사했습니다. Ruff와 `git diff --check`도 통과했습니다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m ruff check src tests
git diff --check
```

현재 VM에는 설정된 원본 CSV가 없으므로 실제 사용자·아이템 감소 수와 전체 GPU 실험 성능은 미검증입니다. 원본 환경에서 `prepare-cohort`와 `embed-representations`를 실행하면 해당 Run의 `reports/leave_one_out_cohort.{json,md}`에 전처리 수치와 fingerprint를 저장합니다. 이번 전환은 전처리·분할·checkpoint 평가 정책을 정렬한 것이며 논문 성능의 완전 재현을 의미하지 않습니다.

## IISAN·DFF 평가 규칙 정렬 (2026-10-08)

Text·Graph mean/GNN 모두 반복 정답을 포함한 앞선 보존 이력 전체를 마스킹하고, 사용자별 PyTorch 기본 내림차순 argsort로 순위를 계산합니다. checkpoint는 validation HR@10 최고 epoch이며 동점이면 최초 epoch를 선택합니다. patience 5·min_delta 0·최적 가중치 복원 후 test 한 번을 유지합니다. Graph Item/catalog 캐시는 복원 직후 초기화하고 선택된 가중치로 다시 계산합니다. 최종 보고와 사용자별 seed 평균 paired bootstrap은 NDCG@10을 유지합니다.

평가 계약이 학습 fingerprint·추천 identity·checkpoint·완료·공유 캐시·Run 비교에 포함됩니다. 이전 leave-one-out 추천은 완료 결과나 공유 캐시로 재사용하지 않습니다. 같은 Run에서 선택 Arm을 재학습하면 기존 추천 파일을 교체하며, 이어서 진단을 재생성하십시오. 현재 계약의 완료 단위는 중단 복구 시 재사용합니다. 모델 구조·손실·popularity·오른쪽 padding은 유지하며, 입력 계약과 provenance가 일치하는 BGE 특징도 재사용합니다. 동률 순서는 환경에 따라 달라질 수 있고, 논문 전체 재현 또는 GPU 성능 개선을 주장하지 않습니다.
