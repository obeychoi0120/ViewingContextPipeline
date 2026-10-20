# ViewingContextPipeline

MicroLens-100K 영상에서 시청 맥락을 추출하고, **영문 제목에 영상 관측 정보를 더했을 때 추천 성능이 어떻게 달라지는지** 비교하는 실험 파이프라인입니다. Qwen3-VL·Qwen3.5·Gemini로 장면 정보를 생성하고, 동일한 사용자·후보 영상 목록(catalog)·사용자별 분할에서 SASRec을 학습·평가합니다.

## 전체 흐름

- **Text 경로:** 키프레임 → Scene Graph 또는 Description → 영상 Summary → 고정 BGE 특징 → 학습한 영상 벡터
- **직접 Graph 경로:** 키프레임 → Scene Graph → 고정 BGE 특징·역할 연결 → 장면 표현·집계 → 학습한 영상 벡터
- **공통 추천:** 과거 시청 영상 벡터 → SASRec 사용자 벡터 → 후보 영상 벡터와 내적 → 추천 순위·평가 지표

**직접 Graph 경로에는 Summary가 필요하지 않습니다.** 제목은 Scene 추출·Summary 생성에 넣지 않고, 제목을 사용하는 Arm(비교 실험 조건)에서 영상 정보와 결합합니다. BGE는 동결하며, 영상 encoder와 SASRec은 추천 loss로 함께 학습합니다.

![ViewingContextPipeline 전체 흐름](docs/design/main_diagram.png)

그림의 Text·Graph는 제목을 함께 쓰는 예시입니다. 제목을 쓰지 않는 Arm은 해당 성분을 마스킹합니다. 직접 Graph의 집계 방식은 선택할 수 있으며, 지원 목록은 아래 모듈 3에서 확인합니다. [편집 가능한 PPTX](docs/design/main_diagram.pptx) · [그림 생성 코드](docs/design/generate_main_diagram.py)

[시작하기](#시작하기) · [단계별 입력과 출력](#단계별-입력과-출력) · [모듈 상세](#모듈별-동작과-세부사항) · [실험 조건](#실험-조건과-결과-해석) · [산출물·재현](#산출물과-재현)

## 시작하기

저장소 루트에서 실행합니다. Ubuntu/Bash·Python 3.11 이상과 `ffmpeg`·`ffprobe`가 필요하며, 데이터와 로컬 모델 가중치는 미리 준비합니다. Qwen은 Linux/CUDA와 비양자화 BF16 체크포인트를 사용하고, Gemini는 Vertex ADC 인증과 호출 권한이 필요합니다.

```bash
python -m pip install -e ".[qwen,gemini,train,embedding,dev]"
python -m pip check
export RUN_ID="new_run_id"
```

먼저 [config.yaml](config.yaml)의 데이터·모델 경로와 Vertex 프로젝트 설정을 실행 환경에 맞게 바꾸세요. **조건을 분리해 비교하거나 기존 결과를 보존하려면 새 Run ID를 사용합니다.** 정상 생성 결과는 프롬프트·penalty를 바꿔도 일반 재실행에서 재사용되므로 [재개 규칙](#산출물과-재현)을 확인하세요.

MiniCPM Graph 추출은 아래 전용 설치·실행 방법을 사용합니다.

<details>
<summary>MiniCPM-V-4.7-1B: vLLM으로 Scene Graph 추출</summary>

```bash
python -m pip install -e ".[minicpm,dev]"
python -m pip check
export RUN_ID="new_minicpm_run"
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v9.md --model minicpm
```

먼저 공통 cohort·키프레임 준비를 완료해야 합니다. `models.minicpm`은 `~/workspace/models/MiniCPM-V-4.7-1B`의 로컬 비양자화 BF16 체크포인트를 가리킵니다. 다른 환경에서는 경로를 수정하세요. Linux/CUDA가 필요하며, 전용 extra는 `transformers>=5.18.0`과 `vllm==0.32.0`을 설치합니다. 이 vLLM 릴리스에는 [MiniCPM 4.7 지원 PR #58674](https://github.com/vllm-project/vllm/pull/58674)가 포함되어 있습니다.

`--arm` 생략 시 `graph_minicpm`을 사용합니다. MiniCPM은 Graph 장면 추출만 지원하며 요약·Description·추천 평가 소스로는 선택할 수 없습니다. 기존 v6 설정은 그대로 읽을 수 있습니다. MiniCPM을 활성화하려면 `models.minicpm`, `protocol.graph_extractors` 마지막의 `minicpm`, `extraction.graph.minicpm.scene_max_new_tokens` 세 설정을 함께 추가해야 합니다. 기본 config에는 세 항목이 포함되어 있습니다.

입력은 기존 키프레임을 순서대로 사용하며, 16x 시각 토큰 압축과 processor 기본 slicing을 적용합니다. Thinking은 끄고 greedy decoding으로 생성합니다. 기본값은 출력 상한 384토큰, presence penalty 1.5, 생성 토큰 대상 repetition penalty 1.05이며 `--presence-penalty`와 `--repetition-penalty`로 변경할 수 있습니다. GPU 선택은 `CUDA_VISIBLE_DEVICES`, 엔진 자원 설정은 공용 `extraction.qwen`을 사용합니다. 컨텍스트 길이를 초과하면 입력·출력 한도를 자동으로 줄이지 않고 오류를 보고합니다.

결과는 `artifacts/runs/<RUN_ID>/extraction/scenes/graph_minicpm/<content_id>.jsonl`, 수정하지 않은 응답은 같은 디렉터리의 `raw_responses/<content_id>.jsonl`에 저장합니다. 원본을 먼저 저장한 뒤, 코드·인용 문자열·수식의 이스케이프를 보존하면서 일반 텍스트 개행을 복원하여 기존 v9 파서에 전달합니다. 원본 파일은 각 장면의 최근 응답을 보존하며 재시도 시 해당 장면을 교체합니다.

일반 재실행은 설정 변경 여부와 관계없이 정상 결과를 재사용하고 warning·failed 장면을 다시 생성합니다. 변경한 설정은 새로 생성하는 장면에만 적용되며, `generation_identity.json`은 최초 생성 설정을 보존합니다. 전체 결과에 변경한 설정을 적용하려면 `--force` 또는 새 Run ID를 사용합니다. `--force`는 해당 소스의 기존 결과와 원본 응답을 지우고 다시 생성합니다. 추론 실행과 Graph 파싱 성공은 시각적 정확도나 추천 성능 개선을 입증하지 않습니다.

</details>

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

Graph 추천·진단에는 모듈 3 표의 7종 중 하나를 `--scene-aggregation`으로 지정합니다. 집계 방식을 바꾸면 추천부터 별도 학습하며 결과도 분리합니다. embedding과 Text에는 aggregation을 지정하지 않습니다. Description Arm은 직접 Graph에서 지원하지 않습니다.

학습 시간은 `run-recommendation --profile-every 100`으로 측정할 수 있습니다. [측정 항목과 해석](docs/graph_training.md#추천-학습-병목-프로파일링)을 참고하세요.

</details>

## 단계별 입력과 출력

실행 패키지는 `preparation → extraction → validation` 순서입니다. 연구 모듈 2·3·4는 모두 `validation` 안에 있으며 영상·사용자 표현은 `run-recommendation`에서 함께 학습합니다.

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

직접 Graph 경로를 네 가지 연구 질문으로 나눕니다. 각 모듈은 설계 질문·입출력·현재 구현을 설명하고, 계산·저장 규칙은 드롭다운에서 확인합니다.

| 모듈 | 중심 질문 | 입력 → 현재 출력 |
| --- | --- | --- |
| 1. 스키마와 장면 간 관계 | 무엇을 기록하고 어떤 관계로 장면을 연결할 것인가? | 키프레임 → 장면별 Scene Graph |
| 2. Scene Graph의 벡터화 | 관측의 의미와 역할 구조를 어떻게 벡터에 담을 것인가? | Scene Graph → Scene vector |
| 3. 영상 단위 집계 | 어떤 inductive bias로 장면들을 통합할 것인가? | Scene vectors → Video vector |
| 4. 추천 학습과 평가 | 영상 표현을 실제 추천에 어떻게 사용할 것인가? | Video vectors·시청 이력 → 추천 순위·평가 지표 |

이 구분은 연구 질문에 따른 것이며 독립 실행·학습 단계를 뜻하지 않습니다. **BGE는 동결하고 모듈 2·3의 학습 가능한 부분과 모듈 4의 추천기는 추천 loss로 함께 학습합니다.** `embed-representations`는 고정 BGE 특징·역할 연결을 준비하며, 학습된 Scene·Video vectors는 `run-recommendation`의 forward에서 계산합니다.

### 모듈 1. Scene Graph 스키마, 장면 간 관계 정의

추천에 필요한 시각 관측을 Context·Entities·Actions로 정리합니다. 장면 내부의 역할 관계를 정의하고, 장면 사이에는 동일 개체·시간적 연속성·사건 관계를 어떤 근거로 표현할지 검토합니다.

현재 Scene은 의미적 장면 전환을 검출한 결과가 아니라 30초 고정 시간 창입니다. 창당 6개 키프레임·640×352 해상도를 사용하며 마지막 짧은 창은 키프레임이 적을 수 있습니다. 오디오는 사용하지 않습니다. **현재 스키마의 이름·ID는 장면 내부에서만 유효하며, 장면 간 동일 개체 추적·명시적인 사건 관계는 구현되어 있지 않습니다.**

<details>
<summary>Schema·프롬프트·장면 출력 예시</summary>

[Scene Graph 프롬프트](prompts/scene_graph_v9.md)는 시간순 키프레임에서 다음 정보를 추출합니다.

| 성분 | 기록할 내용 |
| --- | --- |
| Context | 표현 매체(medium)·형식(format)·주제(topics) |
| Entities | 중요한 개체의 이름·속성 |
| Actions | 누가(actor) 무엇을 하는지(action), 무엇에(target), 어떤 도구로(tool), 어디에서(place) |

아래는 출력 형식을 설명하기 위한 예시이며 실제 Run 결과가 아닙니다.

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

</details>

<details>
<summary>이름 참조·역할·none/unknown·파싱과 저장 규칙</summary>

Action은 `actor - action - target; tool; place`의 다섯 슬롯입니다. 역할에는 선언한 개체 이름을 공백·번호까지 정확히 복사하거나 `none`(부재·비해당), `unknown`(식별 불가)을 사용합니다. 같은 이름의 서로 다른 개체를 구분할 때만 `person1`, `person2`처럼 번호를 붙입니다. Actions가 없으면 `none`을 씁니다.

파서는 이름을 장면 내부 ID로 변환합니다. 위 예시는 `person → e1`, `tomato → e2`, `knife → e3`, `kitchen → e4`가 되며 Action의 역할 참조도 같은 ID로 연결됩니다. `none`은 저장 시 `null`, `unknown`은 문자열로 유지합니다. 미선언 역할 참조는 `unknown`으로 정규화합니다.

Graph는 `extraction/scenes/<source>/<content_id>.jsonl`에 `content_id`·`scene_idx`·`tokens`·`warning`·`scene_graph`를 저장합니다. 실패 행은 같은 파일의 `content_id`·`scene_idx`·`tokens`·`error`·`raw_output`이며, 같은 Scene은 최신 결과 한 행으로 교체합니다.

구조화하지 못한 Graph는 원문과 warning을 보존합니다. 직접 Graph 경로는 구조·warning 검사를 통과한 장면만 사용하며 실패 행의 `raw_output`은 관측·임베딩에 사용하지 않습니다. 파싱 성공과 warning 부재는 시각적 사실성 검증을 뜻하지 않습니다.

기존 `id: kind` 형식의 선언도 읽을 수 있지만 Relations·receiver·구형 `location` 필드·place 누락은 자동 보정하지 않습니다. 지원하지 않는 저장 형식은 경로·원인을 포함한 오류로 거부합니다. 자세한 계약은 [실행·저장 계약](docs/runtime.md)을 참고하세요.

</details>

<details>
<summary>장면 간 연결: 현재 지원 범위와 남은 설계 질문</summary>

현재 Graph 입력은 장면을 `scene_idx` 순서로 모으지만 개체 이름·번호·ID가 다른 장면에서도 같다고 동일 개체로 간주하지 않습니다. 한 영상이나 여러 영상의 Entity ID를 공유 노드로 합치지 않습니다.

| 관계 | 현재 구현 | 남은 설계 질문 |
| --- | --- | --- |
| 장면 순서 | 유효 장면을 시간순으로 유지 | 원본 시간·장면 간 간격을 어떻게 표현할 것인가? |
| 장면 간 동일 개체 | 장면별 독립 ID 사용. 장면 간 추적 없음 | 어떤 시각 근거로 동일 개체를 연결할 것인가? |
| 사건 간 관계 | 명시적 연속·인과 관계 저장 없음 | 어떤 관계를 관측으로 인정하고 스키마에 기록할 것인가? |

위 질문은 후속 설계 검토 범위입니다. 현재 GNN·Transformer가 벡터 사이에서 수행하는 정보 전달은 모듈 3에서 다룹니다. 모듈 1은 관계의 의미·표현 규칙을, 모듈 3은 관계나 벡터 유사도를 이용한 집계 연산을 설명합니다.

</details>

### 모듈 2. Scene Graph를 Scene Vector Representation으로 인코딩
Entity의 이름·속성, Action의 동작, Context의 자연어 필드를 각각 BGE로 인코딩합니다. 역할에 연결된 Entity 정보를 Action에 반영하고 요소별 표현을 결합해 **512차원 Scene vector**를 만듭니다. BGE는 동결하며 projection·역할 결합·Scene MLP는 추천 학습에서 갱신합니다.

<details>
<summary>자연어 필드별 BGE 입력과 벡터 차원</summary>

| 성분 | BGE에 넣는 자연어 | 고정 특징 → 학습 가능한 projection |
| --- | --- | --- |
| Entity | 이름·속성 | 개체별 1024 → 128 |
| Action | 동작 텍스트 | 동작별 1024 → 128 |
| Context | Format·Medium·Topics | 장면별 1024 → 128 |

BGE는 large-en-v1.5를 사용합니다. ID는 역할 연결에만 사용하며 BGE 입력에는 넣지 않습니다. Entity·Action은 각각 인코딩한 뒤 역할 결합과 평균을 거치고, Context는 세 필드를 합친 문자열 하나로 인코딩합니다.

</details>

<details>
<summary>Entity→Action 역할 결합과 Scene vector 계산</summary>

Action은 actor·target·tool·place에 연결된 Entity의 128차원 벡터를 가져와 보정합니다. Entity 벡터 자체는 이 역할 결합으로 갱신하지 않습니다. `none`·`unknown`에는 역할별 학습 가능한 상태 벡터를 사용합니다.

```text
Action 보정 = Action + ReLU(Linear([Action; actor; target; tool; place]))
Scene = R([Title 128; Entity 평균 128; 보정된 Action 평균 128; Context 128])
R(x) = x + ReLU(Linear(x))
```

역할 결합은 640 → 128차원, Scene concat은 512차원입니다. Scene MLP는 Linear 512→512 하나와 ReLU를 쓰며 LayerNorm이 없습니다. **장면 내부의 Entity→Action 역할 결합과 모듈 3의 장면 간 정보 전달은 서로 다른 연산입니다.**

</details>

<details>
<summary>제목 결합과 결측 처리</summary>

제목을 사용하는 Arm은 제목의 고정 BGE 특징 1024차원을 별도 projection으로 128차원으로 줄여 Scene concat에 넣습니다. 제목은 관측 생성 입력에 넣지 않으며, 제목을 사용하지 않거나 제목이 없으면 projection 뒤 해당 성분을 0으로 마스킹합니다.

유효 장면에 Entity·Action이 없으면 해당 슬롯은 0입니다. 모듈 3의 GCN 가중치 입력은 제목을 제외한 Entity·보정된 Action·Context 성분을 사용합니다. 장면이 전혀 없는 영상의 처리는 모듈 3의 결측 규칙을 따릅니다.

</details>

### 모듈 3. Scene vectors들을 Video vector로 집계
모든 장면을 동등하게 볼지, 중요한 장면에 가중치를 줄지, 장면 사이에서 정보를 교환할지, 순서를 반영할지에 따라 영상 표현이 달라집니다. 현재 구현은 이러한 가정이 다른 일곱 가지 집계 방식으로 **512차원 Video vector**를 만듭니다. 구조적 가정과 추천 성능의 우열은 별도의 비교 실험으로 구분합니다.

<details>
<summary>집계 방식별 inductive bias</summary>

| `--scene-aggregation` | 담고 있는 가정·구조 | 영상 집계 |
| --- | --- | --- |
| `mean` | 모든 유효 장면을 동등하게 사용. 순서 정보 없음 | 장면 평균 |
| `attn_pool` | 장면별 중요도 학습. 장면 간 정보 교환 없음 | 학습한 가중치로 집계 |
| `gnn` | cosine 유사도로 가중한 장면 간 정보 전달. 순서 정보 없음 | GCN 전달 후 평균 |
| `gnn_attn` | 장면 간 전달 가중치를 attention으로 학습. 순서 정보 없음 | attention 가중 GCN 전달 후 평균 |
| `transformer` | 위치 정보 없이 장면 간 상호작용 학습 | Transformer 후 평균 |
| `transformer_pos` | 장면 간 상호작용에 유효 장면 순번의 위치 정보 추가 | Transformer 후 평균 |
| `transformer_local` | 각 장면의 자기 자신에 대한 변환만 허용하는 대조군 | 장면별 Transformer 후 평균 |

각 집계는 한 영상 내부에서 수행합니다. 모든 방식이 고정 Graph 입력을 공유하며, 집계 방식을 바꾸면 추천부터 별도 학습합니다. 과거 이름인 `attention`은 지원하지 않습니다.

</details>

<details>
<summary>GCN·attention·Transformer의 계산과 도표</summary>

GCN 가중치 입력은 Scene MLP 이전의 `[Entity 평균; 보정된 Action 평균; Context]` 384차원으로 제목을 제외합니다. `gnn`은 같은 영상의 모든 유효 장면 쌍에 대해 cosine의 음수는 0, self-loop는 1로 두고 대칭 degree 정규화한 인접 행렬 `Â`를 만듭니다. `H′ = H + ReLU(Â H W + b)`로 전달한 뒤 평균합니다. 유사도는 매 forward 계산하며 gradient를 차단하지 않습니다.

`gnn_attn`은 같은 가중치 입력에서 학습한 query·key로 점수를 계산하고 행별 softmax로 전달 가중치를 정합니다. `attn_pool`은 Scene MLP 출력에서 장면별 점수를 계산하고 영상 안에서 softmax로 정규화해 가중합합니다.

Transformer 계열은 Scene MLP 출력에 1층 Transformer를 적용한 뒤 평균합니다. `transformer`는 위치 정보 없이 모든 장면을 보고, `transformer_pos`는 유효 장면 순번의 위치 정보를 추가하며, `transformer_local`은 자기 자신만 보도록 attention을 제한합니다.

아래 도표는 모듈 2의 장면 표현, 모듈 3의 GCN 집계, 모듈 4의 추천을 연결한 예시입니다.

![추천 모델 구조 — Baseline, Text, Graph GNN](docs/design/recsys_diagram_gnn.png)

[GNN PPTX](docs/design/recsys_diagram_gnn.pptx) · [GNN 그림 생성 코드](docs/design/generate_recsys_diagram_gnn.py) · [mean PNG](docs/design/recsys_diagram.png) · [mean PPTX](docs/design/recsys_diagram.pptx) · [세부 수식·구현·캐시 계약](docs/graph_training.md)

</details>

<details>
<summary>순서 정보·장면 수·결측 처리의 범위</summary>

유효 장면은 시간순으로 모으지만 순서 정보의 사용 여부는 집계 방식에 달려 있습니다. `transformer_pos`의 위치는 유효 장면 순번이며 제외된 장면까지 포함한 원본 timestamp나 시간 간격이 아닙니다. `transformer_local`은 장면 간 교환의 효과를 비교하기 위한 대조군입니다.

장면이 한 개여도 GCN·Transformer의 변환은 적용됩니다. 유효 장면이 없고 제목만 있으면 `[Title; 0; 0; 0]`을 Scene MLP에 넣은 결과를 사용하고 장면 집계는 건너뜁니다. 제목·유효 장면이 모두 없거나 padding이면 최종 영상 벡터는 0입니다. 집계 뒤 별도의 Item MLP는 없습니다. 제목 기준선 `meta`에는 장면 집계를 적용하지 않습니다.

</details>

### 모듈 4. Video Vector들로 추천 학습 및 평가

과거 시청 영상의 Video vectors를 SASRec으로 인코딩하고 User MLP를 적용해 512차원 사용자 벡터를 만듭니다. 사용자 벡터와 후보 영상 벡터의 내적으로 순위를 계산하고 사용자별 leave-one-out 방식으로 평가합니다. 모듈 3의 영상 안 장면 순서와 이 모듈의 사용자 시청 순서는 구분합니다.

학습 설정은 [config.yaml](config.yaml)의 **`validation.model`**, 평가·통계 설정은 **`validation.evaluation`**, 사용자 필터·보존 이력은 **`validation.cohort`**를 참고하세요.

<details>
<summary>사용자 표현·후보 점수 계산과 공동 학습</summary>

SASRec은 과거 시청 영상 벡터와 위치 정보를 처리합니다. 마지막 유효 위치의 출력에 2층 GELU User Residual MLP를 적용해 사용자 벡터를 만듭니다. SASRec과 User MLP 내부의 LayerNorm은 유지합니다. 후보 영상 벡터는 영상 encoder에서 직접 가져옵니다.

```text
추천 점수 = 사용자 벡터 · 후보 영상 벡터
```

모듈 2·3의 학습 가능한 영상 encoder와 SASRec은 추천 loss로 함께 갱신하며 seed·Arm·Graph 집계 방식별로 별도 학습합니다. 고정 BGE 특징과 학습된 Scene·Video vectors를 구분하고, 장면 encoder나 집계를 독립적으로 사전 학습한 것으로 해석하지 않습니다.

</details>

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
<summary>비교 경로: 제목 기준선과 Text Summary</summary>

네 모듈의 직접 Graph 경로와 함께 제목 기준선·Text 경로를 비교합니다. Description은 시각 관측을 자연어로 표현하고, Text 경로는 장면별 Graph 또는 Description을 영상 Summary로 모은 뒤 벡터화합니다.

| 단계 | 프롬프트 | 입력 → 출력 |
| --- | --- | --- |
| Description 추출 | [Scene Description](prompts/scene_description_v3.md) | 시간순 키프레임 → 주제·구별되는 관측을 담은 영문 문단 |
| Graph 요약 | [Graph Summary](prompts/summary_graph_v9.md) | 장면 Graph 관측 → 영상 단위 영문 요약 |
| Description 요약 | [Description Summary](prompts/summary_description_v5.md) | 장면 Description → 영상 단위 영문 요약 |

Description의 짧은 예시는 `A person wearing an apron cuts a tomato with a knife in a kitchen.`입니다. 두 Summary 프롬프트에는 `{scenes}`가 필요하고 제목 자리표시자 `{english_title}`은 허용하지 않습니다. 구조화하지 못해 원문·warning으로 보존한 Graph 관측은 Text Summary에 사용할 수 있습니다.

| 경로 | Video vector 계산 |
| --- | --- |
| 제목 기준선 `meta` | 제목 BGE 1024 → Linear 512 → 1층 Residual MLP |
| Text | 제목 BGE 1024 → Linear 128; Summary BGE 1024 → Linear 384 → concat 512 → 1층 Residual MLP |

Item MLP는 `R(x) = x + ReLU(Linear(x))`이며 LayerNorm이 없습니다. 누락 성분은 projection 뒤 0으로 마스킹하고 제목·Summary가 모두 없으면 최종 영상 벡터는 0입니다. 추천에는 모듈 4의 SASRec·내적·평가 규칙을 사용합니다.

Description 행은 `extraction/scenes/<source>/<content_id>.jsonl`에 `content_id`·`scene_idx`·`tokens`·`description`을 저장합니다. Summary는 `extraction/summaries/<source>/<content_id>.json`에 텍스트·상태(`complete`/`failed`)·장면 수·단어 수·토큰 수·provenance를 저장합니다. 실제 Scene 내용이 바뀐 완료 Summary는 일반 재개에서 거부하므로 새 Run 또는 `summarize --force`로 다시 생성합니다.

</details>

<details>
<summary>설계 근거와 연구실 자문 반영 범위</summary>

[연구실 자문 기록](docs/feedback_260923.md)의 세 모듈 번호는 원문의 구분입니다. 이 README의 네 모듈과 다음과 같이 대응합니다.

| 자문 원문의 구분 | 이 README의 연결 모듈 | 현재 구현과 반영 범위 |
| --- | --- | --- |
| 자문 모듈 1: 역할 분해 Graph의 추천 활용성 | 모듈 1·2 | Context·Entities·Actions를 정의하고 역할 구조를 장면 표현에 반영 |
| 자문 모듈 2: 장면 유사도 기반 GCN aggregation | 모듈 2·3 | 장면 벡터를 구성하고 한 영상 안에서 GCN 전달 후 집계. 자문 방법 전체의 재현은 아님 |
| 자문 모듈 3: 사용자·아이템의 공통 feature space | 모듈 4 | SASRec 사용자 벡터와 후보 영상 벡터를 내적으로 비교 |

제목은 관측 생성에서 제외하고 추천 단계에서 결합하며 고정 BGE 특징과 학습 가능한 영상 encoder를 분리합니다. 장면 간 동일 개체·사건 관계, 영상 간 노드 공유·어휘 정규화, 계층적 DCGN·UniMP·CSA, 사용자 평균·LightGCN·clustering 비교는 후속 검토 범위입니다. 구현 상태·실험 기록은 [프로젝트 이정표](docs/TODO.md)를 참고하세요.

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

두 옵션은 Qwen의 Scene 추출·요약과 MiniCPM Graph 장면 추출에 지정할 수 있으며 생략하면 위 기본값을 사용합니다. Repetition penalty는 vLLM 기본 penalty와 중복 적용하지 않습니다. 다음은 기본 presence penalty를 끄는 **옵션 변경 예시**입니다.

```bash
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --schema prompts/scene_graph_v9.md --model qwen3.5 --arm graph_qwen3.5 --presence-penalty 0
```

옵션은 새로 생성하는 결과에 적용됩니다. 기존 정상 결과의 재사용·전체 재생성은 위 재개 규칙을 따릅니다. 이 정책의 형식 준수율·추천 성능 효과는 별도의 비교 실험으로 확인해야 합니다.

</details>
