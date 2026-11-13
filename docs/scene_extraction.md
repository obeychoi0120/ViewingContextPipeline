# 장면 추출과 관측 스키마

[README](../README.md)의 실험 실행 안내를 보완하는 문서입니다. source 선택은 [추천 실험 설정](../rec_experiments_config.yaml), 추론 자원·모델·출력 토큰 한도는 [config.yaml](../config.yaml)을 따릅니다.

## 장면 정의와 입력

장면은 의미적 장면 전환 검출 결과가 아니라 고정 시간 창입니다. 시간 창·키프레임 수·해상도는 `extraction.visual_evidence`로 정합니다. 마지막 짧은 창은 키프레임이 적을 수 있습니다. 키프레임은 시간순으로 제공하고 오디오는 사용하지 않습니다.

제목은 추출 프롬프트에 넣지 않고 추천 단계에서 결합합니다. 추출·전처리는 추천 실험 설정 파일에 의존하지 않습니다. 어떤 source를 준비할지는 추천 설정을 보고 실행자가 결정합니다.

| 표현 | 프롬프트 | 추출 결과 |
| --- | --- | --- |
| Graph | [scene_graph_v9_2b.md](../prompts/scene_graph_v9_2b.md) · [scene_graph_v9_8b.md](../prompts/scene_graph_v9_8b.md) | 장면별 Context·Entities·Actions |
| Description | [scene_description_v3.md](../prompts/scene_description_v3.md) | 장면별 관측 문단 |

Graph는 `graph_qwen3vl_2b`·`graph_qwen3vl_8b`·`graph_gemini`, Description은 `desc_qwen3vl_2b`·`desc_qwen3vl_8b`·`desc_gemini` source에 저장합니다. 2B와 8B는 별도 source이므로 한 Run에서 함께 추출할 수 있습니다. 생성 명령에는 `--schema`, `--model`, `--arm`이 필요합니다. 상대 프롬프트 경로는 저장소 루트 기준이며 기존의 비어 있지 않은 Markdown 파일을 지정합니다.

## Graph 스키마와 출력 예시

| 성분 | 기록할 내용 |
| --- | --- |
| Context | 표현 매체 `medium`, 형식 `format`, 주제 `topics` |
| Entities | 중요한 개체의 이름·속성 |
| Actions | `actor - action - target; tool: 이름; place: 이름`의 역할 구조 (tool·place는 해당할 때만) |

다음은 출력 형식의 예시이며 실제 Run 결과가 아닙니다.

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
person - cutting - tomato; tool: knife; place: kitchen
[End]
```

역할에는 선언한 개체 이름을 공백·번호까지 정확히 복사하거나 `none`(부재·비해당), `unknown`(식별 불가)을 사용합니다. 같은 이름의 서로 다른 개체를 구분할 때만 `person1`, `person2` 같은 번호를 붙입니다. Actions가 없으면 `none`을 씁니다. `tool:`과 `place:`는 생략할 수 있으며, 생략한 tool은 `none`, 생략한 place는 `unknown`으로 저장합니다. 라벨 없이 `; knife; kitchen`처럼 위치로 쓴 이전 형식도 읽습니다.

파서는 장면 안에서 이름을 내부 ID로 바꿉니다. 예를 들어 `person → e1`, `tomato → e2`로 변환하고 역할 참조도 연결합니다. `none`은 `null`, `unknown`은 문자열로 저장합니다. 미선언 역할 참조는 `unknown`으로 정규화하며 개체를 추측해 추가하지 않습니다.

이름·ID는 장면 내부에서만 유효합니다. 다른 장면에서 이름이나 번호가 같아도 동일 개체로 합치지 않으며 영상 간 Entity 노드도 공유하지 않습니다.

## 파싱과 지원 경계

Graph 출력은 생성 시 JSON Schema로 강제하지 않습니다. 현행 `graph-text/v22` 파서는 위 이름 선언 형식, 기존 `id: kind; attribute` 선언 형식과 현재 구조의 JSON을 읽습니다. 한 출력 안의 두 선언 형식 혼용은 원문으로 보존합니다.

이름 형식 출력이 엄격 파싱과 구분자 복구에 모두 실패하면 줄 모양으로 섹션을 나눕니다. `medium:`·`format:`·`topics:` 줄은 Context, 행동 구분자가 있는 줄은 Action, 나머지는 Entity입니다. 이때 다음과 같이 처리합니다.

- 섹션 헤더가 없거나 `Entities: a; b`처럼 한 줄로 쓴 형식도 읽습니다.
- 행동은 앞 세 구간만 actor·action·target으로 씁니다. 공백 없는 구분자(`man -cutting- fish`)도 인정합니다.
- 라벨 없는 슬롯이 3개 이상이면 tool·place를 구분할 수 없으므로 모두 버립니다. `tool:`·`place:`가 아닌 라벨도 버립니다.
- `unknown`·`none` 이름의 엔티티 줄은 버리고, 이름 안의 콜론은 공백으로 바꿉니다.
- 출력에 그래프가 여러 개 이어져 있으면 첫 그래프만 씁니다.
- ID 형식 출력에는 이 단계를 적용하지 않습니다.

장면 결과는 다음 다섯 가지 중 하나이며, warning은 장면당 하나만 붙습니다(우선순위 `TOKEN_LIMIT` > `PARSE_FAILED` > `TRASH_INFO`).

| 결과 | 조건 | 대상 |
| --- | --- | --- |
| success | 구조화 성공, 정보 있음 | Graph, Description |
| `PARSE_FAILED` | Context 세 필드, Entities(또는 `none`), Actions(또는 `none`) 중 하나라도 분리하지 못함 | Graph |
| `TRASH_INFO` | medium·format·topics가 전부 `unknown`/`none`이거나, 이름 있는 엔티티와 행동이 하나도 없음 | Graph |
| `TOKEN_LIMIT` | 출력 토큰 한도에서 잘림. 원문만 저장하고 파싱하지 않음 | Graph, Description |
| error | 위에 해당하지 않는 생성 실패(예외, API 오류, 빈 출력) | Graph, Description |

이전 버전의 태그는 읽을 때 `MISSING_REQUIRED`·`PARSE_ERROR` → `PARSE_FAILED`, `TRUNCATION_RECOVERED` → `TOKEN_LIMIT`로 변환합니다. `--force` 없이 재실행하면 먼저 저장된 행에 현재 규칙을 적용합니다. 이전의 절단 error 행은 `TOKEN_LIMIT` 행이 되고, `PARSE_FAILED` 원문은 현재 파서로 다시 읽습니다.

기존 v8 JSON의 Action `location`은 읽기용 복사본에서 `place`로 정규화합니다. 두 필드가 함께 있으면 값이 같아도 거부합니다. 원본 파일은 수정하지 않으며 Relations·receiver·필수 장소 필드 누락 등은 자동으로 다른 스키마로 변환하지 않습니다. 지원하지 않는 입력은 경로·원인과 함께 오류로 처리합니다.

warning 장면은 원문 또는 그래프를 보존하지만 BGE 임베딩과 Graph 구성에서 제외하고, 재실행 시 다시 생성합니다. 저장된 warning은 그대로 따르며 읽을 때 다시 판정하지 않습니다. error의 `raw_output`도 임베딩이나 Graph 구성에 사용하지 않습니다. **파싱 성공과 경고 부재는 시각적 사실성 검증을 뜻하지 않습니다.**

장면 순서와 활동 키를 이용하는 Hypergraph는 추출된 사건 엣지와 별도의 전처리·학습 경로입니다. 현재는 장면 간 동일 개체 추적·명시적 사건/인과 관계·시각적 행동 경계 검출을 수행하지 않습니다. 활동 구간의 의미와 한계는 [활동 Hypergraph](graph_training.md#activity_hypergraph)를 참고하세요.

파서와 구조 검증의 근거는 [Graph 파서](../src/extraction/semantic_graph/parser.py), [구조 검증](../src/extraction/structured_output.py), [저장·복구](../src/extraction/scene_storage.py)입니다.

## Qwen 생성 정책과 옵션

CLI 모델 식별자 `qwen3vl_2b`·`qwen3vl_8b`는 각각 Qwen3-VL 2B·8B Instruct checkpoint를 사용합니다. 로컬 checkpoint 경로는 `models.qwen3vl_2b`·`models.qwen3vl_8b`에 지정하며, 폴더 이름(또는 `_name_or_path`)에 요청한 크기만 들어 있어야 합니다. 이전 식별자 `qwen3vl`과 `graph_qwen3vl`·`desc_qwen3vl` source는 지원하지 않으므로, 기존 결과를 이어 쓰려면 source 폴더를 생성 당시 크기의 새 이름으로 옮깁니다. Thinking checkpoint는 지원하지 않습니다. Linux/CUDA의 vLLM 실행이며 설치 요구사항은 [pyproject.toml](../pyproject.toml)의 `qwen` extra를 확인하세요.

Greedy decoding의 설정은 `do_sample=False`, `temperature=0.0`, `top_p=1.0`, `top_k=0`, `min_p=0.0`입니다. 기본 presence penalty는 Graph 0·Description 1.5, repetition penalty는 Graph 1.05·Description 1.0입니다. 출력 토큰 한도는 `extraction.{graph,description}.{qwen3vl_2b,qwen3vl_8b}.scene_max_new_tokens`를 따르며 문서에 별도 값으로 복제하지 않습니다.

| 옵션 | 허용 범위 | 적용 |
| --- | --- | --- |
| `--presence-penalty FLOAT` | 유한한 수 −2.0~2.0 | 해당 실행의 presence penalty. `0`은 비활성화 |
| `--repetition-penalty FLOAT` | 유한한 수 1.0~2.0 | 생성 토큰 전용 repetition penalty |

Repetition penalty는 전용 logits processor를 사용하며 vLLM 기본 repetition penalty는 1.0으로 두어 중복 적용하지 않습니다. 최초 요청·구조 복구·재실행에 같은 옵션값을 사용하고 증가 schedule은 적용하지 않습니다. 예를 들어 Graph에 이전 기본값인 presence penalty 1.5를 적용하려면 다음과 같이 실행합니다.

```bash
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --dataset "$DATASET" --schema prompts/scene_graph_v9_core.md --model qwen3vl_2b --arm graph_qwen3vl_2b --presence-penalty 1.5
```

옵션은 새로 생성하는 결과와 provenance에 적용됩니다. 기존 정상 장면을 갱신하려면 새 Run 또는 해당 추출의 `--force`를 사용합니다. 생성 정책 변경의 형식 준수율·추천 성능 효과는 별도의 비교가 필요합니다.

엔진 설정 `extraction.qwen`은 GPU 메모리 사용량, 동시 요청, 배치 토큰, context 길이와 prefix caching 등을 제어합니다. 구체적 허용값은 [qwen_config.py](../src/extraction/qwen_config.py)를 따릅니다. 설치 extra `qwen`과 이 설정 이름은 CLI의 `--model qwen3vl_2b`·`qwen3vl_8b`와 구분하며 두 크기에 같은 엔진 설정을 사용합니다.

## Gemini 사용

`gemini` extra를 설치하고 `models.gemini`에 Vertex 프로젝트·위치·모델·생성 설정을 지정합니다. 실행 환경에 ADC 인증과 호출 권한이 필요하며 전체 장면 동시성은 `extraction.gemini.threads`로 정합니다. 로컬 Qwen용 penalty 옵션은 Gemini에 지정할 수 없습니다.

```bash
python -m extraction extract-graph-scenes --run-id "$RUN_ID" --dataset "$DATASET" --schema prompts/scene_graph_v9.md --model gemini --arm graph_gemini
python -m extraction extract-description-scenes --run-id "$RUN_ID" --dataset "$DATASET" --schema prompts/scene_description_v3.md --model gemini --arm desc_gemini
```

생성한 source를 추천 YAML에 명시해 특징·추천 실험에 연결합니다. source가 다르면 특징은 별도로 준비합니다. 같은 추천 조건으로 모델 소스를 비교하더라도 실제 생성 관측·장면 확보율 차이를 포함해 해석해야 합니다.

## 저장과 실패 확인

장면은 `<artifacts_root>/runs/<RUN_ID>/<dataset>/extraction/<source>/<content_id>.jsonl`에 저장합니다.

기존 `extraction/scenes/<source>/` 결과를 재사용하려면 source 폴더 전체를 `extraction/<source>/`로 옮깁니다. `generation_identity.json`과 실패·원본 응답 파일도 함께 유지해야 합니다. 기존 폴더는 자동 이동하지 않으며, 새 위치가 이미 있으면 두 결과를 덮어쓰거나 임의로 병합하지 않습니다.

| 행 종류 | compact 필드 |
| --- | --- |
| Description | `content_id`, `scene_idx`, `tokens`, `description`, (`TOKEN_LIMIT`일 때만) `warning` |
| Graph | `content_id`, `scene_idx`, `tokens`, `warning`, `scene_graph` |
| 실패 | `content_id`, `scene_idx`, `tokens`, `error`, `raw_output` |

`scene_graph`는 구조화 객체 또는 보존한 원문 문자열입니다. 실패의 토큰 수는 실제 출력 토큰 수 또는 `null`이며, 응답 원문이 없으면 빈 문자열을 기록합니다. 같은 장면은 최신 결과 한 행으로 원자적으로 교체합니다.

진행바는 `success=… parse_failed=… trash_info=… token_limit=… error=…`로 표시하며 합이 완료 수입니다. Description은 `success`, `token_limit`, `error`만 표시합니다. Graph·Description 모두 source 디렉터리의 `generation_stats.json`에 재사용·신규 결과를 합친 같은 키의 집계를 기록합니다. warning·error가 있으면 해당 JSONL의 원문과 태그를 확인합니다. error가 남으면 CLI는 종료 코드 1을 반환하고 완료된 출력은 보존합니다. warning만 남으면 종료 코드는 0입니다.

정상 장면 재사용, 기존 실패 로그 통합, `generation_identity.json`의 한계는 [실행·저장 계약](runtime.md#장면-저장과-추적-정보)에 설명합니다. 프롬프트 변경 의도와 버전은 [변경 이력](../prompts/update_note.md)을 참고하세요.
