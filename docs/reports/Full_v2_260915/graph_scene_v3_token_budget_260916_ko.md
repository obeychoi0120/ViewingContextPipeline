# Graph v3 출력 길이와 형식 비용 분석

분석일: 2026-09-16. `desc_gemini 생성 속도 저하 원인 분석` 대화의 전체 사용자 요청·분석 결과, 현재 코드, Qwen archive와 Gemini 저장 결과를 확인했다. 이번 작업은 분석이며 프롬프트·실행 코드·추출 결과는 변경하지 않았다. 모델 추론과 유료 API 호출도 실행하지 않았다.

## 결론

1. **`context`를 합계 1–3개의 짧은 문장으로 제한하는 것이 타당하다.** 배열 형태는 유지하고, 문자열 하나에 문장 하나, 전체 최대 60단어, 추가 정보가 없으면 `[]`로 명시하는 안을 권장한다.
2. **현재 출력 형식에는 상당한 토큰 비용이 있다.** 특히 들여쓰기·줄바꿈이 크다. Qwen v3 실패 원문에서 문자열 밖 공백만 제거하면 토큰 수가 평균 약 1,024 → 564로 감소한다. 같은 의미의 완료 graph를 한 줄 JSON으로 표현하는 경우도 비교용 들여쓰기 JSON보다 약 40% 짧다.
3. **그러나 `context`와 형식만 줄여서는 주된 반복 실패를 해결할 수 없다.** v3 실패 372개는 모두 `context` 생성 전에 잘렸다. 개체·속성·관계의 선택 기준과 분량 제한이 함께 필요하다.
4. **우선 프롬프트에서 간결한 내용 선택 + 한 줄 JSON을 적용하고, 다음 비교 후보로 배열형 JSON 또는 줄 단위 출력 + parser를 검토하는 것이 합리적이다.** 후자는 추가 절약 여지가 있지만 프롬프트만 바꾸어서는 작동하지 않는다.

## 1. 이전 대화와 현재 코드의 기준

이전 대화의 최신 결정은 다음과 같다.

- E2E 실행을 우선하며, ID 중복·미등록 참조 때문에 결과를 거부하거나 재생성하지 않는다.
- Desc·Graph·Summary 모두 한 번 생성하고 실패를 기록한다. 자동 재시도와 Summary 교정은 제거했다.
- `.recovery`·`.pending`을 새로 남기지 않는다. 장면 실패는 `scenes/failures/{content_id}.jsonl`에 원문과 함께 저장한다.
- 공개 Graph 결과는 `content_id`와 `scene_graph`를 유지한다.

따라서 archive에 남은 과거 recovery 기록은 **분석 근거**다. 현재도 재시도 대기로 정체된다는 설명을 적용하면 안 된다. 현재 `scene/s`에는 성공과 실패가 모두 포함되므로, 향후 개선 비교에서는 속도와 함께 정상 graph 비율·잘림 비율을 봐야 한다.

현재 Qwen Graph는 [steps.py](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/steps.py:121)에서 `structured_output={"json": GRAPH_JSON_SCHEMA}`를 지정한다. [schema](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/structured_output.py:20)에는 배열 개수나 문자열 길이의 상한이 없다. 현재 검증은 JSON 형태를 검사하며 ID 참조 무결성을 강제하지 않는다.

Gemini는 [백엔드](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/backends/gemini.py:60)에서 출력 토큰 상한 등을 설정하지만 JSON 응답 schema는 전달하지 않는다. JSON 형식은 프롬프트로 요청한다. 두 모델의 형식 강제 방식은 다르다.

## 2. 분석 대상과 측정 방법

| 대상 | 범위 | 용도 |
|---|---|---|
| `graph/qwen/scenes_v3.tar` | 완료 graph 61개 + 실패 원문 372개 | 주 분석 |
| `graph/qwen/scenes_v2.tar` | 완료 graph 220개 + 실패 원문 97개 | 기존 보고서와 수치 교차 확인 |
| `graph/gemini/scenes/` | 파일명순 첫 100개 콘텐츠, 640개 graph | 같은 v3 지시에서 `context` 길이 참고 |

Qwen 완료 61개는 **완료된 결과만의 표본**이다. 실패 372개를 포함한 전체 출력의 평균적인 `context` 길이를 뜻하지 않는다. Gemini 첫 100개도 전체 데이터셋의 무작위 표본이 아니며, Qwen과 동일 장면만 맞춘 모델 성능 비교가 아니다.

### 프롬프트 버전 확인

- Qwen v3 및 Gemini 표본의 provenance는 모두 `dbdc17a2…aa9f04`이며 Git `2bc68ce`의 `graph_scene_v3.md`와 일치한다.
- 현재 파일의 SHA-256은 `b2fa5394…0a621`이다. Git `118d72e`에서 예시 도입부의 `only, not evidence for the supplied frames`를 제거하고 빈 줄을 추가했다.
- 내용 선택·`context`·배열 길이 지시는 동일하다. 아래 결과는 예시 도입부 수정 전 실행 결과이며, 현재 파일로 새로 생성한 실험은 아니다.

### 토큰 계산

[Qwen 공식 tokenizer](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/e2378df056d88153dc44616229fa371fcb87e236/tokenizer.json)를 사용해 CPU에서 재계산했다. revision은 `e2378df056d88153dc44616229fa371fcb87e236`, tokenizer 파일 SHA-256은 `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`이다. special token은 추가하지 않았다.

v3 실패 372개 중 367개는 저장된 1,024 tokens와 정확히 일치하며, 나머지 5개는 원문 재토큰화 값이 1–2 tokens 차이 난다. 아래 형식 비교는 동일 tokenizer로 통일했다. Gemini 출력에 이 tokenizer를 적용한 값은 **Qwen으로 읽었을 때의 길이**이며 Gemini API의 출력 토큰·과금 토큰 수가 아니다.

완료 graph의 형식 비교에서는 `content_id`와 저장용 `scene_graph` wrapper를 제외했다. 이 필드들은 모델 출력 후 코드가 추가하므로 생성 토큰 절약 대상으로 세면 안 된다.

## 3. `context`의 실제 길이

| 지표 | Qwen v3 완료 61개 | Gemini 첫 100개 콘텐츠의 640개 |
|---|---:|---:|
| `context` 항목 수 평균 | 4.11 | 1.39 |
| 항목 수 중앙값 / 최대 | 4 / 11 | 1 / 4 |
| 3개 항목 초과 | 40개, 65.6% | 2개, 0.3% |
| 전체 단어 수 평균 | 69.4 | 25.5 |
| 단어 수 중앙값 / 최대 | 70 / 189 | 25 / 67 |
| 60단어 초과 | 36개, 59.0% | 1개, 0.2% |
| 한 줄 JSON에서 `context` 내용이 차지하는 토큰 비중 | 24.1% | 12.9% |

단어는 공백으로 나눴다. **항목 수는 문장 수와 같지 않다.** 문자열 하나에 여러 문장을 넣을 수 있으므로 프롬프트는 두 조건을 함께 명시해야 한다. 토큰 비중은 전체 한 줄 JSON의 토큰 수와 `context: []`로 바꾼 JSON의 토큰 수 차이를 합산해 계산했다. 내용 제거를 실제 개선안으로 적용한 것은 아니다.

### 사례

- **`microlens_100k_00018 / S0`: 11항목, 189단어.** 남자의 행동을 한 문장씩 다시 설명하고, 뒤에서는 화면 문구를 인용·번역한다. 중복 설명과 기존 문구 출력 금지 위반이 동시에 있다. `context` 내용만 246 tokens로 전체 한 줄 graph 560 tokens의 약 44%다.
- **`microlens_100k_00014 / S11`: 9항목, 174단어.** 두 플레이어가 자동차에 관해 이야기한다는 동일 문장이 네 번 나온다.

이 사례의 시각적 사실 여부를 이번 분석에서 다시 판정한 것은 아니다. 원문 내부의 반복·길이·문구 출력 규칙 위반을 확인했다.

### 현재 프롬프트의 빈틈

[context 지시](/home/junsu2.choi/workspace/ViewingContextPipeline/prompts/graph_scene_v3.md:19)는 추가 설정·표현 방식·근거 있는 해석을 허용하고 중복을 금지하지만, 항목 수·문장 수·총 길이 상한이 없다. `Consider all supplied frames`와 결합하면 프레임마다 부연하는 방향으로 확장될 여지가 있다. 단일 문장의 예시만으로 실제 출력을 제한하지 못했다.

권장 교체 문구 초안:

```text
context:
- An array of 1-3 short sentences in total, with exactly one sentence per string and at most 60 words across the entire array. Prefer one sentence when sufficient. Use [] if no additional context is needed.
- Include only distinctive setting, presentation, or grounded interpretation not already represented by entities and relations. Mention unrelated shot changes briefly when relevant.
- Do not narrate individual frames, repeat entity attributes or relations, paraphrase captions line by line, or add a concluding summary. Retain uncertainty and follow the on-screen wording restriction.
```

60단어는 제안하는 최초 실험 기준이며 검증된 최적값은 아니다. 1–3문장을 모두 채우라는 지시가 아니라 필요한 만큼만 쓰게 해야 한다. 기존 장르·목적 해석 허용과 자막의 의미 활용 정책은 유지한다.

## 4. 실패의 주된 원인은 `context` 이전에 있다

v3 실패 372개 모두 저장된 output token 수가 1,024다. 원문의 마지막 최상위 필드와 반복을 별도로 확인했다.

| 실패 시 생성 중이던 필드 | 개수 |
|---|---:|
| `entities` — 내부 attributes 포함 | 333 |
| `relations` | 39 |
| `context` | 0 |

372개 모두 `context` 필드에 도달하지 않았다. 351개에서는 완성된 개체 객체 조각 중 ID를 제외한 `name`과 `attributes`가 똑같이 반복된다. 이 지표만으로 서로 다른 실물 개체를 동일 개체라고 판정한 것은 아니다.

따라서 **`context` 제한은 완료 결과의 간결함을 개선하지만, 관측된 대부분의 토큰 초과 실패를 직접 해결하지는 못한다.** 프롬프트에는 다음도 필요하다.

- 중요한 인물·사물·행동 대상·도구를 먼저 고르고, 눈에 보이는 모든 배경 요소를 열거하지 않는다.
- 같은 개체의 반복 프레임 때문에 ID를 새로 만들지 않는다. 다만 실제로 다른 두 사람·두 컵은 별도로 유지한다.
- 장면별 개체·관계 수와 개체별 속성 수를 제한하는 실험을 병행한다. 시작 후보는 개체 최대 8개, 관계 최대 8개, 속성 최대 3개이며, 속성은 짧은 구로 쓴다.
- 최대 개수를 채우지 말고 중요한 관찰을 다 썼으면 배열을 닫도록 명시한다. 개체·관계의 근거가 충분하면 적은 수로 끝내는 것도 정상이다.

상한은 복잡한 장면의 정보를 누락시킬 수 있다. 중요한 서로 다른 인물을 하나로 합치거나 컷 간 관계를 발명하는 방식으로 줄이면 안 된다. 필요하면 덜 중요한 배경 개체부터 제외한다. 문장·단어·개수 제한은 생성 지침이며 1,024-token 이내 완성을 보장하는 강제 장치는 아니다.

## 5. JSON 때문에 토큰을 얼마나 더 쓰는가

### 5.1 실제 실패 원문: 들여쓰기·공백 비용

v3 실패 원문 372개에서 따옴표·이스케이프 상태를 추적하며 **문자열 밖의 공백·줄바꿈만 제거**했다. 문자열 내용, 필드명, 순서, 반복 개체, 잘린 마지막 조각을 그대로 유지했다.

| 표현 | 평균 tokens |
|---|---:|
| 실제 실패 원문 재토큰화 | 1,023.99 |
| 문자열 밖 공백 제거 | 563.70 |
| 차이 | 460.29, 45.0% 감소 |

이는 **관측된 동일한 잘린 출력의 표현 비용 비교**다. 공백을 없애면 모델이 그 지점에서 정상 종료한다는 뜻은 아니다. 반복 생성이 계속되면 확보한 공간에서 개체를 더 나열하고 다시 1,024에 도달할 수 있다.

또한 이미 생성한 결과를 저장 단계에서 minify하는 것은 생성 토큰을 되돌려 주지 않는다. **모델이 처음부터 한 줄로 생성하도록** 해야 한다. 현재 예시 JSON은 한 줄이지만 본문에 줄바꿈·들여쓰기 금지가 명시되어 있지 않다.

권장 추가 문구:

```text
Return a single-line compact JSON object. Do not add indentation, line breaks, or spaces outside string values. Preserve normal spaces inside strings. Close every array and the object, then stop.
```

### 5.2 같은 완료 graph를 여러 형식으로 표현

Qwen v3 완료 61개에서 모든 개체·ID·속성·관계·context 값을 그대로 보존하고 표현만 바꿨다.

| 형식 | 평균 tokens | 한 줄 기존 JSON 대비 감소 |
|---|---:|---:|
| 비교용 2칸 들여쓰기 기존 JSON | 597.8 | — |
| **한 줄 기존 JSON** | **359.3** | 기준 |
| 한 줄 JSON + 필드명 단축 | 344.4 | 4.2% |
| 한 줄 JSON + 개체·관계 객체를 배열로 표현 | 276.3 | 23.1% |
| E/R/C 행으로 구분하는 TSV | 249.0 | 30.7% |

들여쓰기 JSON은 저장 객체를 다시 직렬화한 비교 표현이며 실제 생성 원문은 아니다. 참고로 metadata에 저장된 해당 61개 응답의 평균 output tokens는 598.7이다. 한 줄 기존 JSON은 비교용 들여쓰기 JSON 대비 39.9% 짧다.

`subject_id` 같은 이름만 짧게 바꾸는 효과는 제한적이다. 더 큰 추가 절약은 매 객체마다 반복되는 키·객체 구문을 없애는 데서 나온다. 따라서 **JSON 자체를 버려야만 절약되는 것은 아니다.**

배열형 JSON의 예:

```json
{"entities":[["person1","person",["red jacket"]],["cup1","cup",["white"]]],"relations":[["person1","holding","cup1"]],"context":["An informal indoor gathering."]}
```

위 형식은 positional mapping으로 현재 객체 JSON으로 변환할 수 있다. 하지만 기존 schema와 다르므로 생성 schema·파서 변경이 필요하다.

TSV 측정은 `E<TAB>id<TAB>name<TAB>attribute...`, `R<TAB>subject_id<TAB>predicate<TAB>object_id`, `C<TAB>sentence`를 사용했다. Python csv의 탭 구분·따옴표 escaping 규칙으로 직렬화하고 역변환했으며, 61개 모두 원래 graph와 정확히 같았다. 이는 **변환의 무손실성 확인**이고, Qwen이 새 형식을 안정적으로 생성한다는 검증은 아니다. 임의의 pipe 구분 텍스트나 YAML에 이 수치를 그대로 적용할 수 없다.

### 5.3 형식 제약의 연산 비용은 별도 문제

여기서 확인한 것은 **출력 문자열에 들어가는 토큰 수**다. JSON 제약 엔진이 매 토큰 생성에 더 쓰는 계산 시간, 문법 제약에 따른 내용 선택 변화는 측정하지 않았다. 앞선 v2/v3 비교는 양쪽 모두 같은 JSON schema를 사용했기 때문에 제약 엔진의 유무 효과를 분리하지 못한다.

[vLLM 공식 문서](https://docs.vllm.ai/en/latest/features/structured_outputs/)는 JSON schema와 grammar 등 여러 제약 형식을 제공한다. 따라서 JSON을 벗어나더라도 문법 제약 자체를 반드시 없애야 하는 것은 아니다. 다만 새 형식이 더 빠르고 덜 실패하는지는 같은 입력에서 생성 실험으로 확인해야 한다.

## 6. 권장 적용 순서

### 1차: 현재 JSON 유지, 프롬프트에서 내용과 표현 제한

- `context`: 합계 1–3문장, 한 항목 한 문장, 최대 60단어, 필요 없으면 빈 배열.
- 개체·속성·관계: 중요 내용 선택, 반복 프레임의 중복 ID 금지, 짧은 구, 상한 후보를 적용.
- 한 줄 JSON: 문자열 밖 공백·들여쓰기·줄바꿈 금지.
- 기존 근거 정책: 자막 문구 전사 금지, 근거 있는 해석 허용, 컷 간 연속성 가정 금지, unary action의 attributes 배치, 서로 다른 개체 구분 유지.

이 단계는 `graph_scene_v3.md`만으로 시도할 수 있다. 강제 이행이 필요한 경우에는 나중에 생성용 schema의 `maxItems`나 whitespace 제약을 별도 검토할 수 있다. 기존 결과를 거부하는 사후 검증 강화와는 구분해야 한다.

### 2차: 형식 변경이 필요한 경우

우선 배열형 JSON을 비교 후보로 권장한다. 기존 JSON 처리 방식과 가깝고, 이번 표본에서 한 줄 객체 JSON보다 23.1% 짧다. TSV는 추가 절약이 가능하지만 delimiter·escaping·빈 영역·종료·잘린 마지막 행의 규칙까지 설계해야 한다.

어느 쪽이든 다음을 함께 변경해야 한다.

1. 프롬프트의 생성 형식과 예시.
2. Qwen에 전달하는 현재 `GRAPH_JSON_SCHEMA` 강제 조건. 줄 출력이면 제거하거나 대응 grammar로 교체한다.
3. 생성 형식을 기존 `entities / relations / context`로 바꾸는 결정적 parser와 출력 형식의 버전·재사용 식별 정보.

저장 결과의 `content_id / scene_graph`와 downstream summary 입력 계약은 유지할 수 있다. Parser는 문법을 변환하고, 없는 개체 추가·관계 삭제·ID 자동 병합 같은 의미 수정을 하지 않는 것이 기존 요청과 맞다. 자동 재생성이나 ID 참조 검증도 다시 도입할 필요가 없다.

## 7. 실제 생성 검증에서 확인할 지표

이번 분석으로 **형식별 표현 비용은 확인했지만 프롬프트 준수율·생성 속도 개선은 아직 측정하지 않았다.** 기존 keyframes가 동일한 장면에서 현재 프롬프트, 1차 수정안, 필요 시 형식 변경안을 비교해야 한다.

- 출력 token 평균·상위 10%·1,024 도달 비율.
- 정상 graph 비율과 Raw/실패 비율. 성공+실패를 세는 `scene/s`만으로 판단하지 않는다.
- `context` 문장·단어 수와 개체·속성·관계의 반복.
- 중요한 인물·도구·행동·별개 컷의 누락, 잘못된 개체 합치기, 자막 문구 복제 여부.
- 가능하면 GPU tokens/s와 전체 소요 시간도 함께 기록한다. 토큰 절약률을 곧바로 속도 개선률로 해석하지 않는다.

참고: [앞선 v2/v3 처리량 보고서](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/reports/Full_v2_260915/graph_scene_v2_v3_throughput_260916_ko.md), [시각 근거 기반 프롬프트 진단](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/reports/Full_v2_260915/qwen_prompt_diagnosis_ko.md).
