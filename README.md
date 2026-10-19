# ViewingContextPipeline

MicroLens-100K 영상에서 시청 맥락을 추출하고, **영문 제목에 영상 관측 정보를 더했을 때 추천 성능이 어떻게 달라지는지** 비교하는 실험 파이프라인입니다. Qwen3-VL·Qwen3.5·Gemini로 장면 정보를 생성하고, 동일한 사용자·후보 영상 목록(catalog)·사용자별 분할에서 SASRec을 학습·평가합니다.

## 전체 흐름

- **Text 경로:** 키프레임 → Scene Graph 또는 Description → 영상 Summary → 고정 BGE 특징 → 학습한 영상 벡터
- **직접 Graph 경로:** 키프레임 → Scene Graph → 고정 BGE 특징·역할 연결 → 장면 표현·집계 → 학습한 영상 벡터
- **공통 추천:** 과거 시청 영상 벡터 → SASRec 사용자 벡터 → 후보 영상 벡터와 내적 → 추천 순위·평가 지표

**직접 Graph 경로에는 Summary가 필요하지 않습니다.** 제목은 Scene 추출·Summary 생성에 넣지 않고, 제목을 사용하는 Arm(비교 실험 조건)에서 영상 정보와 결합합니다. BGE는 동결하며, 영상 encoder와 SASRec은 추천 loss로 함께 학습합니다.

![ViewingContextPipeline 전체 흐름](docs/design/main_diagram.png)

그림의 Text·Graph는 제목을 함께 쓰는 예시입니다. 제목을 쓰지 않는 Arm은 해당 성분을 마스킹합니다. 직접 Graph의 집계 방식은 선택할 수 있으며, 지원 목록은 아래 모듈 2에서 확인합니다. [편집 가능한 PPTX](docs/design/main_diagram.pptx) · [그림 생성 코드](docs/design/generate_main_diagram.py)

[시작하기](#시작하기) · [단계별 입력과 출력](#단계별-입력과-출력) · [모듈 상세](#모듈별-동작과-세부사항) · [실험 조건](#실험-조건과-결과-해석) · [산출물·재현](#산출물과-재현)

## 시작하기

저장소 루트에서 실행합니다. Ubuntu/Bash·Python 3.11 이상과 `ffmpeg`·`ffprobe`가 필요하며, 데이터와 로컬 모델 가중치는 미리 준비합니다. Qwen은 Linux/CUDA와 비양자화 BF16 체크포인트를 사용하고, Gemini는 Vertex ADC 인증과 호출 권한이 필요합니다.

```bash
python -m pip install -e ".[qwen,gemini,train,embedding,dev]"
python -m pip check
export RUN_ID="new_run_id"
```

먼저 [config.yaml](config.yaml)의 데이터·모델 경로와 Vertex 프로젝트 설정을 실행 환경에 맞게 바꾸세요. **조건을 분리해 비교하거나 기존 결과를 보존하려면 새 Run ID를 사용합니다.** 정상 생성 결과는 프롬프트·penalty를 바꿔도 일반 재실행에서 재사용되므로 [재개 규칙](#산출물과-재현)을 확인하세요.

<details>
<summary>환경별 설치와 확인할 설정</summary>

Gemini 생성 환경은 `.[gemini,dev]`, 임베딩·추천 환경은 `.[train,embedding,dev]`로 나누어 설치할 수 있습니다. Qwen·임베딩·추천의 GPU는 `CUDA_VISIBLE_DEVICES`로 선택합니다. Qwen은 GPU당 하나의 엔진을 실행하고, 추천의 `--workers-per-gpu`는 GPU당 독립 학습 작업 수입니다(기본 1).

| config.yaml 항목 | 확인할 내용 |
| --- | --- |
| `data` | interaction CSV·검증용 TSV, 영상 디렉터리, 제목·보완 제목 CSV의 경로 |
| `models.qwen3vl`, `models.qwen3.5`, `models.bge` | 로컬 모델 디렉터리 |
| `models.gemini` | Vertex 프로젝트·위치·모델·생성 설정 |
| `extraction.visual_evidence` | 시간 창·키프레임 수·해상도 |
| `extraction.qwen`, `extraction.gemini` | 추론 엔진·동시 실행 설정 |
| `extraction.{graph,description}.{qwen3vl,qwen3.5,gemini}` | Scene·Summary 출력 토큰 한도 |
| `validation.cohort` | 사용자 필터·보존 이력 길이 |
| `validation.encoder`, `validation.graph_execution` | BGE 인코딩·Graph 실행 자원 |
| `validation.model` | 학습 설정·seed·결정적 연산 사용 여부 |
| `validation.evaluation` | checkpoint 지표·평가 cutoff·통계 설정 |
| `artifacts_root` | 산출물 상위 디렉터리 |

interaction CSV는 `user,item,timestamp` 형식이며 timestamp는 정수 밀리초입니다. 영상은 `{item_id}.mp4`로 준비합니다. TSV가 있으면 사용자별 interaction multiset을 CSV와 대조합니다. 제목 보완은 원본의 비어 있지 않은 제목을 우선하고 필요한 빈 제목·누락 행만 채웁니다. 보완 CSV를 사용해도 해결되지 않은 제목은 빈 문자열로 저장하며, 보완 CSV가 없을 때 제목 행 누락은 오류입니다. 원본 CSV는 수정하지 않습니다.

</details>

<details>
<summary>1–2단계: 데이터·키프레임 준비</summary>

```bash
python -m preparation prepare-cohort --run-id "$RUN_ID"
python -m preparation prepare-input-data --run-id "$RUN_ID"
```

`prepare-cohort`는 전체 원본 interaction과 catalog를 준비하고 필요한 영상·제목을 검사합니다. 원본 목록만 확인하는 `--plan-only`도 있지만 다음 단계 전에 전체 준비를 완료해야 합니다. `prepare-input-data`는 정상적인 기존 프레임을 재사용합니다. 준비 결과는 Run 간 공유하므로 데이터셋이나 준비 구성이 다른 실험은 `artifacts_root`를 분리합니다.

Run별 leave-one-out 분할·평가 catalog는 이후 `embed-representations`에서 준비하며 `reports/leave_one_out_cohort.{json,md}`에 전처리 요약을 기록합니다.

</details>

<details>
<summary>3–4단계: Scene 추출·Text 경로의 Summary 생성</summary>

생성 명령에는 `--model`, `--arm`, Markdown 프롬프트 경로인 `--schema`를 지정합니다. `--arm`은 생성 소스이며 평가용 `_meta` Arm을 지정하지 않습니다. 상대 경로는 저장소 루트 기준입니다.

| 모델 (`--model`) | Graph 소스 (`--arm`) | Description 소스 (`--arm`) |
| --- | --- | --- |
| `qwen3vl` | `graph_qwen3vl` | `desc_qwen3vl` |
| `qwen3.5` | `graph_qwen3.5` | `desc_qwen3.5` |
| `gemini` | `graph_gemini` | `desc_gemini` |

아래는 Qwen3-VL·Gemini의 두 관측 형식을 준비하는 예시입니다. Qwen3.5는 위 표에 맞추어 모델·소스를 바꾸면 됩니다.

```bash
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v9.md --model qwen3vl --arm graph_qwen3vl
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v9.md --model gemini --arm graph_gemini
python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/scene_description_v3.md --model qwen3vl --arm desc_qwen3vl
python -m extraction extract-description-scenes --run-id "$RUN_ID" --schema prompts/scene_description_v3.md --model gemini --arm desc_gemini
```

Text 경로에는 선택한 소스의 Summary가 필요합니다. 아래는 네 소스 모두 Gemini로 요약하는 예시입니다. 요약 모델을 Qwen으로 선택하려면 `--model qwen3vl` 또는 `--model qwen3.5`를 사용합니다. 소스 이름은 Scene을 생성한 모델을 유지합니다.

```bash
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v9.md --model gemini --arm graph_qwen3vl
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v9.md --model gemini --arm graph_gemini
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_description_v5.md --model gemini --arm desc_qwen3vl
python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_description_v5.md --model gemini --arm desc_gemini
```

같은 Run·소스에는 하나의 Summary 모델을 사용합니다. 모델을 바꾸려면 새 Run 또는 해당 소스의 `summarize --force`가 필요합니다.

</details>

<details>
<summary>5–8단계: Text 특징 준비·추천·진단</summary>

```bash
TARGETS=(meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta desc_qwen3vl_meta desc_gemini_meta)
python -m validation embed-representations --run-id "$RUN_ID" --target "${TARGETS[@]}" --representation-mode text
python -m validation run-recommendation --run-id "$RUN_ID" --target "${TARGETS[@]}" --representation-mode text
python -m validation run-diagnosis --run-id "$RUN_ID" --target "${TARGETS[@]}" --representation-mode text
```

위 예시는 6개 Arm을 선택합니다. 사용할 조건에 맞게 `TARGETS`를 변경하며 선택하지 않은 소스의 Summary는 필요하지 않습니다.

</details>

<details>
<summary>5–8단계: 직접 Graph 특징 준비·집계 방식별 추천·진단</summary>

Summary를 생성하지 않고 Scene Graph에서 고정 입력을 준비합니다. 아래는 `mean`과 `gnn`을 각각 학습·진단하는 예시이며, 모든 집계 방식은 고정 Graph 입력을 공유합니다.

```bash
TARGETS=(meta graph_qwen3vl graph_qwen3vl_meta graph_gemini_meta)
python -m validation embed-representations --run-id "$RUN_ID" --target "${TARGETS[@]}" --representation-mode graph
for AGG in mean gnn; do
  python -m validation run-recommendation --run-id "$RUN_ID" --target "${TARGETS[@]}" --representation-mode graph --scene-aggregation "$AGG"
  python -m validation run-diagnosis --run-id "$RUN_ID" --target "${TARGETS[@]}" --representation-mode graph --scene-aggregation "$AGG"
done
```

Graph 추천·진단에는 모듈 2 표의 7종 중 하나를 `--scene-aggregation`으로 지정합니다. 집계 방식을 바꾸면 추천부터 별도 학습하며 결과도 분리합니다. embedding과 Text에는 aggregation을 지정하지 않습니다. Description Arm은 직접 Graph에서 지원하지 않습니다.

학습 시간은 `run-recommendation --profile-every 100`으로 측정할 수 있습니다. [측정 항목과 해석](docs/graph_training.md#추천-학습-병목-프로파일링)을 참고하세요.

</details>

## 단계별 입력과 출력

실행 패키지는 `preparation → extraction → validation` 순서입니다. 연구 모듈 2·3은 모두 `validation` 안에 있으며 영상·사용자 표현은 `run-recommendation`에서 함께 학습합니다.

| 단계 | 입력 | 출력 | 명령·담당 코드 |
| --- | --- | --- | --- |
| 1. 데이터 준비 | interaction·제목·영상 목록 | 전체 원본 catalog·interaction·제목 매핑 | `prepare-cohort` · [데이터 준비](src/preparation/cohort.py) |
| 2. 영상 관측 준비 | 영상·sampling 설정 | 키프레임 이미지·timestamp | `prepare-input-data` · [입력 준비](src/preparation/input_data.py) |
| 3. 장면 정보 추출 | 키프레임·프롬프트 | Scene Graph 또는 Description JSONL | `extract-graph-scenes` / `extract-description-scenes` · [추출](src/extraction/steps.py) |
| 4. 영상 요약 — Text 경로 | 장면 관측·프롬프트 | 영상 단위 영문 Summary JSON | `summarize` · [요약](src/extraction/summary_executor.py) |
| 5. 평가·고정 특징 준비 | 원본 이력·제목·Summary 또는 Graph | 사용자별 분할·평가 catalog·1024차원 BGE 특징; Graph는 역할 연결·장면 경계 포함 | `embed-representations` · [분할](src/validation/leave_one_out_data.py), [Graph 입력](src/validation/graph_inputs.py) |
| 6. 영상 표현 학습 | 고정 특징·Graph 구조 | 영상당 512차원 벡터 | `run-recommendation` 내부 · [Text 모델](src/validation/model.py), [Graph 모델](src/validation/graph_model.py) |
| 7. 사용자 표현·추천 평가 | 시청 이력·후보 영상 벡터 | checkpoint·학습 기록·이벤트별 HR/NDCG | `run-recommendation` · [학습·평가](src/validation/leave_one_out_recommendation.py) |
| 8. 진단 | 이벤트 지표·입력·학습 기록 | 최종 점수·통계 비교·실행 판정 JSON | `run-diagnosis` · [Text 진단](src/validation/leave_one_out_diagnosis.py), [Graph 진단](src/validation/graph_diagnosis.py) |

**5단계의 고정 입력 특징과 6단계의 학습된 추천용 영상 벡터는 다릅니다.** 영상 안의 장면 순서와 사용자의 시청 순서도 구분합니다. SASRec은 사용자의 시간순 시청 이력을 처리합니다.

## 모듈별 동작과 세부사항

### 모듈 1. 키프레임에서 장면 정보를 추출합니다

Scene은 의미적 장면 전환을 검출한 결과가 아니라 고정 시간 창입니다. 현재 설정은 30초 창·6개 키프레임·640×352 해상도이며 마지막 짧은 창은 키프레임이 적을 수 있습니다. Graph는 Context·Entities·Actions로 관측을 분해하고, Description은 같은 시각 관측을 자연어로 표현합니다. 오디오는 사용하지 않습니다.

<details>
<summary>프롬프트의 역할과 장면 출력 예시</summary>

| 단계 | 프롬프트 | 입력 → 출력 |
| --- | --- | --- |
| Graph 추출 | [Scene Graph](prompts/scene_graph_v9.md) | 시간순 키프레임 → Context·개체 이름·역할별 Action |
| Description 추출 | [Scene Description](prompts/scene_description_v3.md) | 시간순 키프레임 → 주제·구별되는 관측을 담은 영문 문단 |
| Graph 요약 | [Graph Summary](prompts/summary_graph_v9.md) | 장면 Graph 관측 → 영상 단위 영문 요약 |
| Description 요약 | [Description Summary](prompts/summary_description_v5.md) | 장면 Description → 영상 단위 영문 요약 |

아래는 현재 Graph 프롬프트의 출력 형식을 설명하는 예시이며 실제 Run 결과가 아닙니다.

```text
[Context]
medium: live_action
format: demonstration
topics: food preparation
[Entities]
person; wearing apron
tomato
knife
kitchen
[Actions]
person - cutting - tomato; knife; kitchen
```

Action은 `actor - action - target; tool; place`의 다섯 슬롯입니다. 역할에는 선언한 개체 이름을 공백·번호까지 정확히 복사하거나 `none`(부재·비해당), `unknown`(식별 불가)을 사용합니다. 같은 이름의 서로 다른 개체를 구분할 때만 `person1`, `person2`처럼 번호를 붙입니다. Actions가 없으면 `none`을 씁니다.

파서는 이름을 장면 내부 ID로 변환합니다. 위 예시는 `person → e1`, `tomato → e2`, `knife → e3`, `kitchen → e4`가 되며, Action의 역할 참조도 같은 ID로 연결됩니다. `none`은 저장 시 `null`, `unknown`은 문자열로 유지합니다. 이름·번호·ID가 다른 장면에서도 같다고 동일 개체로 간주하지 않습니다.

Description의 짧은 예시는 `A person wearing an apron cuts a tomato with a knife in a kitchen.`입니다. 두 Summary 프롬프트에는 `{scenes}`가 필요하고 제목 자리표시자 `{english_title}`은 허용하지 않습니다.

</details>

<details>
<summary>저장 형식·파싱·warning 처리</summary>

Scene은 `extraction/scenes/<source>/<content_id>.jsonl`에 저장합니다. Graph 행은 `content_id`·`scene_idx`·`tokens`·`warning`·`scene_graph`, Description 행은 `content_id`·`scene_idx`·`tokens`·`description`을 갖습니다. 실패 행은 같은 파일의 `content_id`·`scene_idx`·`tokens`·`error`·`raw_output`이며, 같은 Scene은 최신 결과 한 행으로 교체합니다.

구조화하지 못한 Graph는 원문과 warning을 보존합니다. 이 원문 관측은 Text Summary에 사용할 수 있지만 직접 Graph에서는 구조·warning 검사를 통과한 장면만 사용합니다. 실패 행의 `raw_output`은 관측·임베딩에 사용하지 않습니다. 미선언 역할 참조는 `unknown`으로 정규화합니다. 파싱 성공과 warning 부재는 시각적 사실성 검증을 뜻하지 않습니다.

기존 `id: kind` 형식의 선언도 읽을 수 있지만 Relations·receiver·구형 `location` 필드·place 누락은 자동 보정하지 않습니다. 지원하지 않는 저장 형식은 경로·원인을 포함한 오류로 거부합니다.

Summary는 `extraction/summaries/<source>/<content_id>.json`에 텍스트·상태(`complete`/`failed`)·장면 수·단어 수·토큰 수·생성 출처(provenance)를 저장합니다. 실제 Scene 내용이 바뀐 완료 Summary는 일반 재개에서 거부하므로 새 Run 또는 `summarize --force`로 다시 생성합니다. 자세한 계약은 [실행·저장 계약](docs/runtime.md)을 참고하세요.

</details>

### 모듈 2. 장면 관측을 추천용 영상 벡터로 만듭니다

Text는 제목·Summary의 BGE 특징을 projection과 MLP로 결합합니다. 직접 Graph는 Entity 정보를 actor·target·tool·place 역할에 따라 Action에 반영하고 장면 표현을 만듭니다. 이후 선택한 집계 방식으로 영상당 512차원 벡터를 만듭니다. **Entity→Action 역할 결합과 장면 간 정보 전달은 서로 다른 연산입니다.**

<details>
<summary>장면 집계 방식 7종과 선택 기준</summary>

| `--scene-aggregation` | 장면 처리 → 영상 집계 |
| --- | --- |
| `mean` | 장면 간 교환 없이 평균 |
| `gnn` | 제목을 제외한 장면 특징의 cosine 유사도로 GCN 전달 후 평균 |
| `attn_pool` | 장면 간 교환 없이 학습한 가중치로 집계 |
| `gnn_attn` | 학습한 attention 가중치로 GCN 전달 후 평균 |
| `transformer` | 위치 정보 없이 장면 Transformer 후 평균 |
| `transformer_pos` | 유효 장면 순번의 위치 정보를 넣어 장면 Transformer 후 평균 |
| `transformer_local` | 각 장면이 자기 자신만 보도록 제한한 Transformer 후 평균 |

GCN·Transformer는 한 영상 내부에서만 장면을 처리합니다. Entity ID를 공유해 영상 간 노드를 합치지 않습니다. `transformer_local`은 장면 간 교환의 효과를 비교하기 위한 대조군이며, `transformer_pos`의 위치는 제외된 장면까지 포함한 원본 timestamp가 아닙니다. 과거 이름인 `attention`은 지원하지 않습니다.

구조의 차이와 추천 품질의 차이는 구분해야 합니다. 설계·검증 기록은 [직접 Graph 학습](docs/graph_training.md)을 참고하세요.

</details>

<details>
<summary>영상 encoder의 차원·수식과 모델 도표</summary>

공통 입력은 동결 BGE large-en-v1.5의 1024차원 특징입니다. Projection·MLP·역할 결합·장면 집계와 SASRec은 추천 학습에서 갱신합니다.

| 경로 | 영상 벡터 계산 |
| --- | --- |
| 제목 기준선 `meta` | 제목 1024 → Linear 512 → 1층 Residual MLP |
| Text | 제목 1024 → Linear 128; Summary 1024 → Linear 384 → concat 512 → 1층 Residual MLP |
| Graph `mean` | 제목·Entity·Action·Context 각각 1024 → Linear 128 → 역할 결합 → 장면 concat 512 → 1층 Scene Residual MLP → 유효 장면 평균 |
| Graph의 다른 집계 | 같은 장면 표현 → 선택한 집계 → 영상 벡터 512 |

Graph의 Entity BGE 입력은 이름·속성, Action은 동작 텍스트, Context는 Format·Medium·Topics입니다. ID는 역할 연결에만 사용하며 BGE 입력에는 넣지 않습니다. `none`·`unknown`에는 역할별 학습 가능한 상태 벡터를 사용합니다.

```text
Action 보정 = Action + ReLU(Linear([Action; actor; target; tool; place]))
Scene = R([Title 128; Entity 평균 128; 보정된 Action 평균 128; Context 128])
R(x) = x + ReLU(Linear(x))
```

GCN 유사도 입력은 Scene MLP 이전의 `[Entity 평균; 보정된 Action 평균; Context]` 384차원으로 제목을 제외합니다. 같은 영상의 모든 유효 장면 쌍에 대해 cosine의 음수는 0, self-loop는 1로 두고 대칭 degree 정규화합니다. `H′ = H + ReLU(Â H W + b)`로 전달한 뒤 평균합니다. 유사도는 매 forward 계산하며 gradient를 차단하지 않습니다.

![추천 모델 구조 — Baseline, Text, Graph GNN](docs/design/recsys_diagram_gnn.png)

영상용 Item·Scene MLP는 Linear 512→512 하나와 ReLU를 쓰며 LayerNorm이 없습니다. SASRec과 2층 GELU User MLP의 LayerNorm은 유지합니다. `meta`에는 장면 집계를 적용하지 않습니다.

[GNN PPTX](docs/design/recsys_diagram_gnn.pptx) · [GNN 그림 생성 코드](docs/design/generate_recsys_diagram_gnn.py) · [mean PNG](docs/design/recsys_diagram.png) · [mean PPTX](docs/design/recsys_diagram.pptx) · [세부 수식·결측·캐시 계약](docs/graph_training.md#모델-및-데이터-계약)

</details>

### 모듈 3. 시청 이력으로 사용자 표현과 추천 순위를 만듭니다

과거 시청 영상 벡터를 SASRec으로 인코딩하고 User MLP를 적용해 512차원 사용자 벡터를 만듭니다. 후보 영상 벡터와의 내적으로 추천 순위를 계산합니다. 영상 encoder와 SASRec은 seed·Arm별로 독립 학습합니다.

학습 설정은 [config.yaml](config.yaml)의 **`validation.model`**, 평가·통계 설정은 **`validation.evaluation`**, 사용자 필터·보존 이력은 **`validation.cohort`**를 참고하세요.

<details>
<summary>사용자별 leave-one-out 분할과 평가 규칙</summary>

현재 분할은 원본 이력이 5개 이상인 사용자의 최근 13개 interaction을 보존한 뒤 마지막 2개를 보류합니다. 동일 timestamp는 원본 event_id 순서로 정하며 마지막 날짜도 포함합니다.

| 단계 | 정답·역할 | 입력 이력 |
| --- | --- | --- |
| Train | 마지막 2개를 제외한 다음 아이템 전이 학습 | 각 정답보다 앞선 이력 |
| Validation | 끝에서 두 번째 interaction으로 HR@10 checkpoint 선택 | train 이력 |
| Test | 마지막 interaction으로 선택된 checkpoint 평가 | train + validation 이력 |

각 정답 앞의 최대 10개를 모델에 입력하며 첫 이벤트는 학습 정답으로 쓰지 않습니다. validation 이후 재학습(refit)은 하지 않습니다. 후보 catalog는 보존된 사용자 시퀀스 전체의 아이템이며 앞선 보존 이력 전체를 마스킹합니다. 반복 정답도 마스킹합니다.

순위는 사용자별 `torch.argsort(scores, descending=True)`로 계산하므로 동률 순서는 PyTorch·device 환경에 따라 달라질 수 있습니다. validation/test는 학습 정답·학습 빈도에 넣지 않습니다. checkpoint는 validation **HR@10**으로 선택하고, 최종 보고·paired bootstrap의 주 지표는 **NDCG@10**입니다. 동점이면 최초 epoch를 유지하며 최적 가중치로 test를 한 번 평가합니다.

전처리는 [MicroLens 공식 SASRec 코드](https://github.com/microlens2023/microlens-dataset/blob/master/Code/VideoRec/SASRec/utils/load_data.py)의 최소 이력·절단·마지막 2개 보류 흐름을 따릅니다. 모델·손실함수까지 논문 전체를 재현한 것은 아니며 사용자별 분할은 사용자 사이의 공통 시간 경계를 보장하지 않습니다.

</details>

<details>
<summary>설계 근거와 연구실 자문 반영 범위</summary>

[연구실 자문 기록](docs/feedback_260923.md)의 세 모듈을 현재 구현에 다음과 같이 연결했습니다.

| 자문 내용 | 현재 구현과 반영 범위 |
| --- | --- |
| 모듈 1: 역할 분해 Graph의 추천 활용성 | Context·Entities·Actions와 역할 연결을 사용하며 Description 비교 경로도 유지 |
| 모듈 2: 장면 유사도 기반 GCN aggregation | 한 영상 안에서 장면 GCN 전달 후 집계. 자문 방법 전체의 재현은 아님 |
| 모듈 3: 사용자·아이템의 공통 feature space | SASRec 사용자 벡터와 후보 영상 벡터를 내적으로 비교 |

제목은 관측 생성에서 제외하고 추천 단계에서 결합하며, 고정 BGE 특징과 학습 가능한 영상 encoder를 분리합니다. 영상 간 노드 공유·어휘 정규화, 계층적 DCGN·UniMP·CSA, 사용자 평균·LightGCN·clustering 비교는 후속 검토 범위입니다. 구현 상태·실험 기록은 [프로젝트 이정표](docs/TODO.md)를 참고하세요.

</details>

## 실험 조건과 결과 해석

같은 Arm 이름이라도 `text`는 Summary를, `graph`는 Scene Graph 구조를 사용합니다. 결과에는 **Run·Arm·표현 모드·Graph 집계 방식**을 함께 명시합니다. 모든 validation 명령에 `--target`과 `--representation-mode`가 필요하며 기본 target 목록은 없습니다.

<details>
<summary>전체 평가 Arm과 필요한 입력</summary>

| `--target` | Text 입력 | 직접 Graph 입력 |
| --- | --- | --- |
| `meta` | 영문 제목 | 영문 제목 |
| `graph_qwen3vl` | Qwen3-VL Graph Summary | Qwen3-VL Scene Graph |
| `graph_qwen3vl_meta` | 제목 + Qwen3-VL Graph Summary | 제목 + Qwen3-VL Scene Graph |
| `graph_qwen3.5` | Qwen3.5 Graph Summary | Qwen3.5 Scene Graph |
| `graph_qwen3.5_meta` | 제목 + Qwen3.5 Graph Summary | 제목 + Qwen3.5 Scene Graph |
| `desc_qwen3.5_meta` | 제목 + Qwen3.5 Description Summary | 미지원 |
| `graph_gemini_meta` | 제목 + Gemini Graph Summary | 제목 + Gemini Scene Graph |
| `desc_qwen3vl_meta` | 제목 + Qwen3-VL Description Summary | 미지원 |
| `desc_gemini_meta` | 제목 + Gemini Description Summary | 미지원 |

모델 접미사는 Scene 추출 모델을 뜻하며 Summary 모델은 별도로 선택합니다. `graph_gemini`는 생성 소스이고 평가에서는 `graph_gemini_meta`를 사용합니다. 현재 지원 범위는 Text 9 Arm·직접 Graph 6 Arm입니다.

</details>

Text의 Graph/Description 비교는 두 관측을 요약문으로 사용한 결과입니다. 구조 활용이나 집계 효과는 입력·모델·학습·평가 조건을 맞춘 비교로 확인해야 합니다. 현재 진단은 선택한 표현 모드·aggregation 내부에서 수행하며 Text/Graph·서로 다른 aggregation 사이의 자동 paired 비교는 지원하지 않습니다.

<details>
<summary>학습 조합 수·통계 비교와 검증 범위</summary>

학습 조합 수는 `seed 수 × 선택한 Arm 수`입니다. 현재 설정의 3 seed에서 Text 9 Arm 전체는 27조합이며 시작하기의 Text 6 Arm 예시는 18조합입니다. 직접 Graph는 집계 방식별로 별도 학습합니다.

NDCG@10은 사용자 단위 paired bootstrap으로 비교합니다. 사용자별 seed 평균을 구한 뒤 사용자에 동일 가중치를 두며 각 seed의 사용자 평균도 보고합니다. 각 실행에서 선택한 target의 모든 쌍을 하나의 Bonferroni 비교군으로 두어 비교 수는 `n × (n−1) / 2`입니다. Text 9 Arm은 36쌍, Graph 6 Arm은 15쌍입니다. 비교 방향은 registry에서 뒤의 Arm − 앞의 Arm이며 CLI 순서와 무관합니다. cutoff·bootstrap 횟수·유의수준은 `config.yaml`의 `validation.evaluation`을 확인하세요.

`run-diagnosis --compare-run-id <참조_Run>`은 선택한 Graph Arm에 대해 사용자·이벤트·분할·seed·평가 계약이 일치하는 Run을 비교하며 직접 Graph는 집계 방식도 같아야 합니다. 이때 Bonferroni 분모는 비교하는 Graph Arm 수입니다. [통계 정책 코드](src/validation/diagnosis_statistics.py) · [Run 비교 코드](src/validation/run_comparison.py)

구현 검증과 전체 데이터의 추천 성능 실험은 구분합니다. 현재 회귀 테스트는 다음 명령으로 실행하며 소형 입력·CPU·오프라인 대역을 사용합니다. 실제 모델 추론·CUDA 처리량·전체 데이터 추천 성능은 이 테스트가 검증하지 않습니다.

```bash
python -m pytest -q tests
```

Run별 성능·비용 분석은 해당 입력·모델·평가 조건과 함께 `artifacts/runs/<RUN_ID>/reports/`에 보관합니다.

</details>

## 산출물과 재현

준비 결과는 공유하고 추출·표현·추천·진단은 Run별로 저장합니다. **Run별 분석 코드·보고서·표·이미지는 `artifacts/runs/<RUN_ID>/reports/`에 보관합니다.**

**재실행은 정상 Scene·완료 Summary를 재사용합니다.** 생성 설정을 바꾸어도 자동 갱신하지 않으며, 같은 Run에 서로 다른 생성 조건의 결과가 섞일 수 있습니다. 전체를 현재 조건으로 다시 만들려면 선택한 단계·소스에 `--force`를 사용합니다. 기존 결과를 보존하려면 새 Run을 사용하세요.

<details>
<summary>산출물 디렉터리와 결과 확인 위치</summary>

```text
artifacts/
├── preparation/                         # 공유 cohort·영상 준비
│   ├── cohort/leave_one_out/
│   ├── resized_keyframes/<width>_<height>/
│   └── source_assets/
├── runs/<RUN_ID>/
│   ├── extraction/
│   │   ├── scenes/<source>/
│   │   └── summaries/<source>/
│   ├── validation/
│   │   ├── cohort/
│   │   ├── representations/{text,graph}/
│   │   ├── recommendations/text/
│   │   ├── recommendations/graph/<aggregation>/
│   │   └── diagnosis/
│   └── reports/
└── shared_cache/                        # 임베딩·Graph 입력·추천 결과 캐시
```

| 확인할 내용 | Run 아래 위치 |
| --- | --- |
| 장면 관측·실패 | `extraction/scenes/<source>/<content_id>.jsonl` |
| 영상 Summary | `extraction/summaries/<source>/<content_id>.json` |
| Text 고정 특징 | `validation/representations/text/<arm>_embeddings.npz` |
| Graph 고정 특징·역할 연결 | `validation/representations/graph/<arm>_embeddings/` |
| 추천 학습·test | `validation/recommendations/text/seed_<seed>/<arm>/` 또는 `validation/recommendations/graph/<aggregation>/seed_<seed>/<arm>/` |
| 진단 | `validation/diagnosis/text_diagnosis.json` 또는 `graph_<aggregation>_diagnosis.json` |

추천 결과는 seed × Arm 단위로 `sasrec.pt`, `training.json`, `per_event_metrics.jsonl`, `complete.json`을 저장하며 직접 Graph는 `catalog_vectors.npy`도 저장합니다. `training.json`의 epoch별 validation과 최종 test 지표는 구분합니다.

진단 JSON은 Arm별 최종 점수(`recommendations.means`), NDCG@10 수치 비교·신뢰구간·비교 정책·평가 계약·실행 판정을 저장합니다. 상세 분할·입력 provenance·이벤트 지표는 원래 cohort·표현·추천 파일에서 확인합니다. 진단의 pass는 실행 증거의 검증 통과이며 추천 성능 개선 판정이 아닙니다.

</details>

<details>
<summary>재개·실패 재시도·캐시 규칙</summary>

| 상황 | 재실행 동작 |
| --- | --- |
| 정상 Scene | 생성 설정이 달라도 재사용 |
| 누락·실패·warning Scene | 현재 설정으로 한 번씩 재시도. penalty 증가·자동 추가 pass 없음 |
| 완료 Summary | 동일 소스·요약 모델·실제 Scene 입력을 검증한 뒤 재사용. 생성 설정 차이만으로 차단하지 않음 |
| 실패 Summary | 현재 설정으로 재시도 |
| 생성 명령의 `--force` | 선택한 소스의 기존 출력을 제거하고 완료 결과까지 재생성 |
| 완료 추천 조합 | 입력·모델·학습·평가 계약을 검증한 뒤 재사용 |
| 미완료 추천 조합 | 처음부터 다시 학습. epoch 중간 재개 없음 |
| 임베딩·추천 캐시 | 현재 Run → 공유 캐시 → 새 계산. `--force`는 해당 단계의 캐시 읽기를 우회 |

Scene·Summary는 다른 Run에서 자동으로 가져오지 않습니다. 같은 추천 경로에서 재학습하면 기존 조합을 대체하며 이후 진단도 다시 생성해야 합니다. 현행 평가 계약으로 학습된 결과는 재학습 없이 재진단할 수 있습니다.

Scene 디렉터리의 `generation_identity.json`은 초기 생성 설정 기록이며 폴더 내 모든 Scene의 조건이 같음을 보장하지 않습니다. 일반 재개에서는 기존 기록을 보존하고 없을 때만 생성합니다. 실제 입력·인코더·모델·학습 계약과 provenance로 호환성을 확인합니다. 자세한 저장·캐시 계약은 [실행·저장 계약](docs/runtime.md)을 참고하세요.

</details>

<details>
<summary>실패·warning 확인과 결측 처리</summary>

- **진행바:** `success`(정상)·`warning`(Graph 경고)·`failed`(실패)의 합이 완료 수입니다. 진행률·처리량·ETA도 같은 합으로 계산합니다. Graph의 최종 집계는 소스 디렉터리의 `generation_stats.json`에서 확인합니다.
- **Scene 실패:** JSONL의 `error`·`raw_output`을 확인합니다. 기존 `failures/<content_id>.jsonl`은 추출 재개 시 원본 JSONL에 원자적으로 통합한 뒤 제거합니다.
- **Summary 실패:** `summaries/<source>/failures.jsonl`을 확인합니다. 실패 Summary의 텍스트는 빈 문자열입니다.
- **Text 결측:** 제목·Summary 중 있는 값을 사용하고 누락 성분은 projection 후 0으로 마스킹합니다. 둘 다 없으면 최종 영상 벡터는 0입니다.
- **Graph 결측:** 유효 장면의 Entity·Action이 없으면 해당 슬롯은 0입니다. 장면 없이 제목만 있으면 제목 기반 Scene MLP 경로를 사용하고 장면 집계를 건너뜁니다. 제목·유효 장면이 모두 없으면 최종 영상 벡터는 0입니다.

결측 때문에 아이템·interaction을 제거하지 않습니다. 손상 파일·잘못된 provenance는 오류로 처리하며 지원하지 않는 과거 저장 형식을 자동 변환하지 않습니다. 기존 Run·분석 파일은 보존합니다.

</details>

<details>
<summary>Qwen 생성 정책·기본값·옵션 변경 예시</summary>

`qwen3vl`은 Qwen3-VL-2B-Instruct, `qwen3.5`는 Qwen3.5-2B를 사용합니다. Qwen3.5는 chat template에 `enable_thinking=False`를 명시합니다. 두 모델은 greedy decoding을 사용하며 greedy에서도 presence·repetition penalty는 적용됩니다.

| 작업 | Presence penalty | Repetition penalty | 현재 config의 출력 토큰 한도 |
| --- | ---: | ---: | ---: |
| Graph 장면 추출 | 1.5 | 1.05 | 384 |
| Description 장면 추출 | 1.5 | 1.0 | 512 |
| Graph·Description 요약 | 2.0 | 1.0 | 512 |

Decoding은 `do_sample=False`, `temperature=0.0`, `top_p=1.0`, `top_k=0`, `min_p=0.0`입니다. 생성 정책은 [qwen_config.py](src/extraction/qwen_config.py), 출력 토큰 한도는 `config.yaml`의 `extraction.{graph,description}.{qwen3vl,qwen3.5}`를 참고하세요. 설치 extra `qwen`과 `extraction.qwen`은 두 모델의 공용 런타임 이름이며 CLI 모델 이름은 아닙니다.

| CLI 옵션 | 허용 범위 | 적용 |
| --- | --- | --- |
| `--presence-penalty FLOAT` | 유한한 단일 숫자 -2.0~2.0 | 해당 실행의 presence penalty. `0`은 비활성화 |
| `--repetition-penalty FLOAT` | 유한한 단일 숫자 1.0~2.0 | 해당 실행의 생성 토큰 전용 repetition penalty |

두 옵션은 Qwen의 Scene 추출·요약에만 지정할 수 있으며 생략하면 위 기본값을 사용합니다. Repetition penalty는 vLLM 기본 penalty와 중복 적용하지 않습니다. 다음은 기본 presence penalty를 끄는 **옵션 변경 예시**입니다.

```bash
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v9.md --model qwen3.5 --arm graph_qwen3.5 --presence-penalty 0
```

옵션은 새로 생성하는 결과에 적용됩니다. 기존 정상 결과의 재사용·전체 재생성은 위 재개 규칙을 따릅니다. 이 정책의 형식 준수율·추천 성능 효과는 별도의 비교 실험으로 확인해야 합니다.

</details>
