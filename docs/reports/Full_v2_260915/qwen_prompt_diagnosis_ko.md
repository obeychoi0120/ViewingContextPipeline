# Full_v2_260915: Qwen 장면 프롬프트 진단

검토일: 2026-09-15

## 1. 결론

**자유 서술·Open Vocabulary라는 방향은 유지할 만하다. 현재 문제는 허용한 정보를 관찰·해석·개체·관계에 어떻게 배치할지 충분히 정의하지 않은 데 있다.**

Description은 자막 전사·번역과 프레임별 장황한 이야기 확장이 두드러진다. Graph는 같은 문제에 더해 서로 다른 컷의 인물을 연결하거나, 단독 행동에 임의의 대상을 붙이고, 주요 사물을 등록하지 않는 문제가 있다. 정상 JSON에도 심각한 의미 오류가 있다.

우선순위는 다음과 같다.

1. 두 프롬프트에 공통으로 **화면 텍스트를 해석의 단서로 쓰는 방법**을 구체화한다.
2. 입력이 **하나의 연속 장면을 보장하지 않는 시간순 샘플**임을 명시한다.
3. Graph에서 **단독 행동 / 두 개체 간 관계 / 장면 해석**의 역할을 구체적인 예로 구분한다.
4. 동일 개체의 중복 생성과 서로 다른 개체의 병합을 함께 막는다.
5. 반복과 근거 없는 서사 확장을 줄이면서 콘텐츠를 구별하는 실제 사물·행동을 보존한다.

출력에서 확인한 오류와 프롬프트 원인에 대한 추정은 구분해야 한다. 아래 진단은 실제 프레임과 현재 출력의 대조이며, 수정 프롬프트의 개선 효과를 실험으로 확인한 결과는 아니다.

## 2. 기존 대화에서 반영한 기준

관련 작업 기록을 직접 읽었다.

- **Revise pipeline with new arms** — 작업 ID `01a0a358-465a-7833-8286-05e6cac7cc33`. 초기 방향 논의, 사용자의 주석, 구현 계획, `context` 설명까지 확인했다.
- **Analyze graph summary gaps** — 작업 ID `01a09ec0-7ef1-7de1-830b-391ae3e30c0e`. 기존 Graph의 정보 손실과 관계 변형을 살펴보려던 목적을 확인했다.

이번 진단에 적용한 사용자의 기준:

| 강조한 내용 | 이번 진단에 적용한 기준 |
|---|---|
| 화면에서 얻을 수 있는 정보를 활용 | 장르·목적·배경지식 해석 자체를 오류로 취급하지 않는다. |
| 화면 Text는 그대로 읽어 출력하지 않고 clue로 활용 | 원문·번역 출력과, 단서를 이용한 의미 해석을 구분한다. |
| Description의 서술 제약 완화 | 장면 Description의 여러 문단·나열 자체를 위반으로 취급하지 않는다. |
| Allowed Values 대신 Open Vocabulary | 고정 어휘·이전 개체 수 제한으로 되돌리는 것을 해법으로 삼지 않는다. |
| 개체 구분, `person1 → person2` 관계 방향 보존 | 이름 일치보다 실제 주체·대상과 개별 개체 참조를 확인한다. |
| Summary는 중요한 내용 중심의 한 문단, 100–200단어, 필드별 할당 제거 | 이 제한을 장면 추출에 소급 적용하지 않는다. |
| 표현 방식 효과와 모델 효과 비교 | Description과 Graph에 동일한 근거 사용 정책을 적용한다. |

화면 문구 출력은 **사용자가 정한 출력 정책 위반**으로 평가했다. 법률적 판단을 한 것은 아니다.

## 3. 범위와 실행 출처

실제 파일 위치는 요청에 적힌 `artifacts/runs/.../extractions`가 아니라 [artifacts/Full_v2_260915/extraction](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction)이다.

`S0` 등의 번호는 JSONL의 `scene_idx`이며 **0부터 시작**한다. 이 자료에서는 JSONL 줄 번호가 `scene_idx + 1`이다.

| 콘텐츠 | 화면 내용 | Description | Graph | 확인한 서로 다른 프레임 |
|---|---|---:|---:|---:|
| `00001` | 중국어 자막이 있는 판타지 만화·일러스트 서사 | 19개, S0–18 | 19개, S0–18 | 113 |
| `00002` | 야외에서 내장 요리를 준비·볶고 두 사람이 먹는 영상 | 7개, S0–6 | 7개, S0–6 | 41 |
| `00003` | 여러 짧은 상황극·반전 장면을 편집한 모음 영상 | 11개, S0–10 | 8개, S0–7 | 64 |
| **합계** | | **37** | **34** | **218** |

- 제공된 71개 출력과 해당 프레임 218개를 검토했다. 전체 프레임을 시간·구간 라벨이 있는 [프레임 모음 이미지](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/review)로 확인하고, 중요한 사례는 원본 PNG도 다시 열었다.
- 34개 공통 구간의 Description·Graph keyframe 시각은 모두 일치한다. 로컬 프레임은 모두 640×352다.
- `00003` S8–10의 Graph는 제공된 파일에 없다. 이것을 추출 실패로 단정하지 않았다.
- Description의 모든 행은 [description_scene_v2.md](/home/junsu2.choi/workspace/ViewingContextPipeline/prompts/description_scene_v2.md), Graph의 모든 행은 [graph_scene_v3.md](/home/junsu2.choi/workspace/ViewingContextPipeline/prompts/graph_scene_v3.md)의 현재 SHA-256과 일치한다. 다른 버전의 프롬프트를 분석하는 상황은 아니다.
  - Description: `3deebf228d5dbbb45f99ff3058b2098b2f26e0fadb1dcbe5d6171f861613f69e`
  - Graph: `ec73e425236ea0b430dd3b2e8ef2d0b7cd0fb08f0ed79fea61f45112c82e5209`
- 저장된 provenance의 모델은 `Qwen3-VL-2B-Instruct`, 장면 생성 상한은 1,024 tokens, 기본 입력은 30초당 6개 프레임이다.
- 이 Run에 제공된 Summary는 없다. 요약으로의 영향은 코드·프롬프트에서 예상되는 영향이며, 실제 요약 오류로 집계하지 않았다.

### 구조와 생성 상태

| 지표 | 확인 결과 | 해석 |
|---|---:|---|
| 정상 Graph 레코드 | 26/34 | 구조 검사 통과가 시각적 정확성을 보장하지 않는다. |
| Graph Raw fallback | 8/34, 23.5% | 아래 7개는 ID 참조 문제, 1개는 잘린 JSON이다. |
| Raw 중 JSON 파싱은 가능하지만 대상 ID가 미등록 | 7/8 | 단순 괄호 복구만으로 해결되지 않는다. |
| Graph 재시도 발생 | 28/34, 82.4% | 저장된 시도 수 합계 97회. Description은 37개 모두 1회다. |
| `finish_reason=length` | Description 1개, Graph 1개 | 각각 `00001/S16`, `00003/S3`. |
| Description 길이 | 단어 수 중앙값 439, 범위 250–968 | 공백 기준. 단어 수 자체를 위반으로 판정한 것은 아니다. |
| Description 프레임별 열거 표식 | 28/37 | `first/second/…/final image/frame` 정규식으로 센 형식 지표다. 의미 오류율은 아니다. |

재시도별 repetition penalty가 달라지는 구현이므로, 현재 Description–Graph 차이에는 표현 형식 외에 구조 작성 난이도와 재시도 선택의 영향도 포함된다. 97회라는 수치는 기록된 시도 수이며 실제 전체 실행 시간의 비교가 아니다.

## 4. 주요 문제와 실제 근거

### 4.1. 화면 텍스트 활용이 전사·번역으로 바뀐다 — 공통 최우선

두 프롬프트에는 이미 전사·인용·번역 금지가 있다. 따라서 **규칙이 없는 문제가 아니라, 허용되는 대체 행동이 충분히 구체적이지 않고 실제 출력에서 규칙이 지켜지지 않는 문제**다.

- `00001/S7` Graph의 `context`는 7개 중 6개가 자막을 원문·번역으로 옮기는 문장이다. [Graph 8행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:8), [프레임](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/review/microlens_100k_00001_s06-08.jpg)
- `00001/S16` Graph는 `speech bubble`, `dialogue`를 여러 개체로 만들고 `attributes`에 자막 문구를 담는다. 출력 금지가 `context`에만 해당한다고 좁게 해석해서도 안 된다. [Graph 17행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:17)
- `00003/S7` Description은 결제·기부 관련 자막을 원문과 영어 번역으로 연속 출력한다. [Description 8행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/description/qwen/scenes/microlens_100k_00003.jsonl:8)

**현재 지시의 빈틈:** `Use all visual evidence`와 `On-screen text may be used as a clue` 뒤에 금지가 이어지지만, 텍스트에서 취할 의미의 수준과 배치 방법은 없다. Graph 예시도 이 상황을 다루지 않는다.

**수정 방향:** 두 Arm에 동일하게, 텍스트를 내부적으로 참고해 사물·상황·장르를 해석하되 문구 대신 의미를 자신의 말로 간결하게 설명하도록 한다. 자막·워터마크의 문구를 저장하기 위한 개체도 만들지 않도록 명시한다. 말풍선 자체가 만화 형식의 단서라는 관찰은 보존할 수 있다.

### 4.2. 자막의 비교·언급·감탄을 현재 화면의 사실로 바꾼다

가장 분명한 사례는 `00002`의 음식이다.

- 22.5초에는 길고 관 모양인 재료가 접시에 놓여 있고, 화면 텍스트는 그 재료를 오리 내장으로 지칭한다. 픽셀과 텍스트 단서를 함께 사용하면 `duck intestines`라는 해석을 지지한다. 종 구분을 픽셀만으로 확정한 것은 아니다. [22.5초](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00002/0022_5.png)
- 122.5초의 자막은 닭다리와 **맛을 비교하는 말**이다. 137.5초의 실제 음식도 손가락·발톱 모양이 아닌 관 모양의 볶음 재료다. [122.5초](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00002/0122_5.png), [137.5초](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00002/0137_5.png)
- 그런데 S4 Description은 `possibly chicken feet`, Graph는 `contains chicken feet and vegetables`라고 쓴다. Graph는 같은 출력에서 비교 자막을 번역하면서도 비교 대상을 실제 재료로 올린다. [Description 5행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/description/qwen/scenes/microlens_100k_00002.jsonl:5), [Graph 5행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00002.jsonl:5)
- 다른 구간에서는 Graph의 명칭이 `dried duck intestines`에서 `food`, `noodles`, `chicken feet`로 흔들린다. 핵심 요리의 구체성이 사라지거나 잘못 대체된다.

`00003/S3` Description도 물속 인물의 감탄을 어머니에 대한 개인적인 회상으로 확장한다. 이는 텍스트를 단서로 쓰는 과정에서 문맥을 잘못 해석한 사례다. [Description 4행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/description/qwen/scenes/microlens_100k_00003.jsonl:4)

**수정 방향:** 비교 대상, 언급된 사람·사물, 화자의 주장, 실제 화면에 보이는 대상을 구분한다. 텍스트에 나온다는 이유만으로 개체의 시각적 존재나 행동을 확정하지 않는다. 모호한 경우 실제로 보이는 형태는 보존하면서 종류만 유보한다.

각 30초 구간은 독립 추론이므로 뒤 구간이 앞 구간의 재료명을 자동으로 기억할 수는 없다. 이 한계를 동일 영상 내 명칭 일관성 문제와 구분해야 한다. 프롬프트만으로 이전 구간의 증거를 복원할 수는 없지만, 현재 구간의 비교 대상을 재료로 오인하는 것은 줄여야 한다.

### 4.3. 30초 묶음을 하나의 연속 장면으로 취급해 컷 사이에 관계를 만든다

`00003`은 한 묶음 안에도 서로 다른 장소·등장인물·영상 형식이 여러 번 바뀐다.

**S0의 잘못된 연결:**

- 2.5초: 실내에서 분장한 인물 앞에 다른 사람이 앉아 있다.
- 22.5–27.5초: 전혀 다른 영상에서 벽 앞에 사람들이 서 있다.
- Graph는 첫 인물이 뒤 컷의 파란 옷 인물 앞에서 춤춘다고 연결한다: `person1 — dancing in front of → person2`.
- 첫 컷의 실제 상대는 올바르게 등록하지 못했다. [Graph 1행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00003.jsonl:1), [프레임 묶음](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/review/microlens_100k_00003_s00-02.jpg)

**S1의 잘못된 연결과 누락:**

- 앞 네 프레임은 실내의 붉은 상의 인물·흰 상의 인물과 휴대폰이다. 뒤 두 프레임은 트럭 옆의 다른 두 사람과 긴 막대다.
- Graph는 실내의 흰 상의 인물이 트럭 옆 마스크 쓴 사람을 바라본다고 연결한다. 붉은 상의 인물·휴대폰은 빠지고, `context`는 야외 상황만 남는다. [Graph 2행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00003.jsonl:2)

**Description도 영향받는다:** S4는 돈을 든 손과 흰 옷 인물의 차 안 장면을 뒤의 검은 티셔츠 인물과 합쳐 한 사람의 구매·생활 변화 이야기로 만든다. 실제 두 번째 프레임에는 안경과 청재킷을 착용한 인물이 차 안에 있다. [Description 5행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/description/qwen/scenes/microlens_100k_00003.jsonl:5), [127.5초](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00003/0127_5.png)

**현재 지시의 빈틈:** 두 프롬프트 모두 `this scene`으로 시작한다. 순서·컷 전환·몽타주 처리 규칙이 없다. 백엔드는 이미지를 순서대로 붙이지만 개별 timestamp나 컷 정보를 텍스트로 전달하지 않는다. [입력 구성](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/backends/qwen.py:103)

**수정 방향:** 시간순으로 샘플링한 이미지이며 서로 무관한 컷이 포함될 수 있음을 명시한다. 컷 전환 뒤에 동일 인물·장소·사건이라고 가정하지 않는다. 같은 Graph 안에서 연결되지 않은 개체 집합을 허용하고, 장면 전체 `context`에 모음 영상이라는 의미를 보존한다. 반드시 같은 프레임에 함께 나와야만 관계를 허용하는 과도한 제한도 피한다. 명확한 연속성으로 뒷받침되는 관계는 유지한다.

### 4.4. Graph가 단독 행동을 임의의 두 개체 관계로 바꾼다

[graph_scene_v3.md 3행](/home/junsu2.choi/workspace/ViewingContextPipeline/prompts/graph_scene_v3.md:3)에 unary action을 `attributes`에 넣을 수 있다고 이미 써 있다. 그러나 예시에는 옷 속성과 두 사람 간 관계만 있어 실제 구분 방법이 드러나지 않는다.

| 실제 상황·출력 위치 | 생성된 관계 | 문제 |
|---|---|---|
| `00001/S5`, 인물이 자신의 손을 듦 | `person2 — raising hand → person1` | 손을 드는 행동에 다른 인물을 목적어로 붙인다. |
| `00001/S12`, 얼굴에 손을 댄 인물 | `person1 — covering face with hand → person1` | 별도 관계가 필요 없는 단독 행동을 자기 자신을 향한 간선으로 만든다. |
| `00001/S9`, 서로 다른 컷의 인물 | `person2 — lying on → person3`, `person3 — sitting on → person4` | 바닥 등의 대상을 사람 ID로 대체해 전혀 다른 사건을 만든다. |
| `00001/S10`, 종이 띠를 든 인물 | `person1 — holds → person4`; person4의 종류는 character | 사물 대신 다른 인물 개체를 대상으로 삼는다. |

근거: [S5 Graph](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:6), [S9 Graph](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:10), [S10 Graph](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:11), [S10 실제 프레임](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00001/0322_5.png)

**수정 방향:** `raising a hand`, `smiling`, `running`처럼 대상 없는 행동은 속성에 둔다. 관계는 실제 주체와 대상을 모두 식별할 수 있을 때 작성한다. 대상을 모를 때 다른 사람을 채워 넣지 않는다. 대상이 필요한 사물이라면 그 사물을 개체로 등록한다. 관계가 없는 개체와 빈 `relations`도 정상이라는 예시를 제공한다.

이는 관계 어휘를 제한하는 제안이 아니라, 자유 어휘가 올바른 주체·대상에 연결되도록 하는 제안이다.

### 4.5. 동일 개체를 복제하고, 별개 개체를 합친다

두 문제가 동시에 발생한다.

- **복제:** `00001/S7`의 Graph에 동일한 `white face / black hair / red mark on forehead` 속성을 가진 `person3`, `person4`, `person5`가 생기고 모두 다른 사람을 바라보는 관계가 붙는다. 실제로는 같은 클로즈업의 반복과 다른 각도의 이미지다. [Graph 8행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:8)
- **병합:** `00003/S4` Graph의 `person1`은 흰 옷·돈을 든 속성을 갖는데, 뒤 컷의 검은 티셔츠 인물의 가방과 다른 차를 사용하는 관계까지 받는다. [Graph 5행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00003.jsonl:5)
- **물건의 병합:** `00002/S4`에서 두 사람이 각자 젓가락을 들고 있지만 Graph는 하나의 `chopstick` ID를 두 사람이 함께 드는 것으로 표현한다. [Graph 5행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00002.jsonl:5)
- **관점 변화 오인:** `00001/S12`에서는 데포르메로 바뀐 기존 캐릭터를 다른 남성 개체로 읽는 등 그림체·샷 변화도 식별을 흔든다.

**수정 방향:** 프레임마다 ID를 새로 만드는 것이 아님을 명시한다. 동일성이 분명한 반복은 합치고, 단지 같은 종류·옷 색상이라는 이유만으로 합치지 않는다. 독립된 물건은 종류가 같아도 별개 ID를 갖는다. ID를 가로질러 일관성을 강제하기 위해 외부 추적기나 이전 장면 기억이 있다고 가정해서는 안 된다.

### 4.6. 주요 사물은 빠지는데 관계 대상은 계속 요구한다

Raw 8개 중 아래 7개는 JSON 자체는 파싱되지만 관계 대상이 `entities`에 없다.

| 콘텐츠 / 구간 | 등록하지 않은 관계 대상 |
|---|---|
| `00001/S1` | `cup` |
| `00001/S14` | `throne` |
| `00002/S5` | `food`, `cup`, `text_on_screen` |
| `00002/S6` | `food`, `bag_of_food`, `text_on_screen_3` |
| `00003/S2` | `tail_light`, `pavement_with_water`, `couch` |
| `00003/S5` | `broom`, `bench`, `parking lot` |
| `00003/S6` | `a path near water`, `an office door`, `the man's hand holding a mobile device` |

가령 `00002/S5`는 실제로 음식과 컵이 중요하지만 `entities`에는 두 사람만 남긴다. 다른 경우에는 존재하지 않아야 할 텍스트 대상이나, ID가 아닌 자연어 문장을 `object_id`에 넣는다.

**현재 지시의 빈틈:** “Both IDs must refer to entities”라는 규칙은 있지만, 주체·행동 대상·도구를 먼저 식별하고 그 집합에서 관계를 작성하는 구체적 방법과 사물 예시가 없다. ID를 참조할 수 없는 경우 무엇을 보존하고 무엇을 생략해야 하는지도 약하다.

**수정 방향:** 행동을 이해하는 데 필요한 사물을 개체 목록에 포함한다. 각 관계의 두 ID를 그 목록에서 재사용한다. 모호한 대상을 억지로 등록하지 말고, 확실한 단독 행동이나 맥락만 남긴다. 사람 두 명만 등장하는 현재 예시에 사람–사물 사례를 보완하는 것이 유용하다.

현재 [구조 검증](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/structured_output.py:35)은 키·타입·ID 참조를 확인한다. [semantic_warnings](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/semantic_graph/schema.py:18)도 같은 검사를 반복하므로 `[]`를 의미적 정확성으로 읽으면 안 된다. 프롬프트 진단을 형식 통과율만으로 평가할 수 없다.

### 4.7. 반복·상투적 서사가 구체적인 정보를 밀어낸다

- `00001/S16` Description은 `The reason for the reason ...`을 반복하다 1,024 tokens에서 잘린다. 뒤쪽 프레임의 정보를 충분히 보존하지 못한다. [Description 17행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/description/qwen/scenes/microlens_100k_00001.jsonl:17)
- `00003/S4` Description은 여섯 프레임을 설명하면서 존재하지 않는 일곱 번째 설명까지 덧붙이고, 차·정원·가방을 반복한 뒤 구매·재산·생활 변화 이야기로 확장한다.
- `00003/S7` Graph의 `context` 8개 중 동일 문장 하나가 5번 반복된다. 동시에 작은 녹색 물체와 뒤에 나오는 카드의 차이는 사라진다. [Graph 8행](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00003.jsonl:8), [217.5초의 작은 녹색 물체](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00003/0217_5.png)
- `00001` Description에는 거의 매 구간 판타지·감정적 긴장·중대한 결정이라는 긴 도입과 결론이 붙는다. 반면 실제로 보이는 의상, 손동작, 특정 소품과 표정 변화에는 오독·누락이 남는다.

Description에는 반복을 합치라는 지시가 이미 있지만 선택·종료 기준이 약하다. Graph에는 명시적인 반복 병합 지시도 없다. `context`가 `entities/relations`의 행동을 다시 설명하거나 자막 서사를 담는 두 번째 긴 Description이 된다.

**수정 방향:** 형식 자유는 유지하면서 같은 근거를 여러 문장·항목에서 반복하지 않도록 한다. 장르·목적 해석을 매번 상투적인 결론으로 확장하지 않는다. 구체적 근거를 설명하지 못하는 `possibly`는 정보를 늘리는 근거가 될 수 없음을 명시한다. `context`에는 개체·관계만으로 담기 어려운 추가 맥락을 둔다.

장면 Description을 100–200단어 한 문단으로 제한하라는 결론은 아니다. 그 규칙은 기존 대화에서 Summary에 합의한 것이다. 또한 여러 문단 또는 프레임별 열거 자체를 오류로 판단한 것도 아니다.

### 4.8. 콘텐츠를 구별하는 해석은 오히려 약하게 남는다

해석이 많다고 영상의 특징을 잘 보존한 것은 아니다.

- `00003/S0`는 첫 화면의 구성과 제목 단서를 이용해 “반전·코미디 장면 모음”으로 해석할 근거가 있다. Description은 제목을 옮기면서 의료 환경 같은 세부 추측을 늘리고, Graph는 첫 컷과 마지막 컷을 잘못 연결한다.
- `00002/S6`는 오른쪽 사람이 비닐봉지를 꺼내는 우스운 마무리다. Graph는 이를 음식이 들어 있는 봉지를 건네는 상황으로 쓴다. 실제 마지막 프레임에서는 봉지 안의 음식을 확인할 수 없다. [201.3초](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/resized_keyframes/microlens_100k_00002/0201_3.png)
- `00001`에는 만화 패널, 반복된 정지 그림, 데포르메, 마법 효과가 보인다. 이런 표현 방식의 근거는 보존할 가치가 있지만, 출력은 확인되지 않은 인물 관계와 줄거리 확장에 많은 분량을 쓴다.

**수정 방향:** 화면 텍스트를 포함한 근거를 실제 영상 유형과 핵심 사건을 구별하는 데 사용한다. 장르·목적 해석은 허용하되 화면 밖의 상세한 개인사·사건 전개와 연결하는 데에는 별도의 근거가 필요하다.

## 5. 권장하는 프롬프트 보강 내용

다음은 전체 프롬프트를 대체하는 확정안이 아니라, 현재 지시에 보강할 핵심 문구 초안이다. 기존 스키마와 자유 어휘는 유지한다.

### 두 Arm에 공통으로 적용

> These images are chronological samples from a video segment and may contain unrelated shots. Describe meaningful changes, but do not assume continuity of people, places, or events across cuts.

> Use on-screen text internally to disambiguate visible content and interpret the scene. Express the resulting meaning in your own words; do not output its wording, quotations, translations, or a transcript. Distinguish what is shown from what a caption merely mentions, compares, or claims.

> Keep interpretations tied to specific evidence and retain uncertainty. Do not expand a brief clue into unseen events, personal histories, or unsupported connections. Merge repeated observations while preserving distinctive objects, actions, and differences.

핵심은 “텍스트를 읽지 말라”가 아니라 **단서를 이용해 의미를 파악하되, 언급과 관찰을 혼동하지 않는 것**이다. 위 문구의 효력은 재생성으로 확인해야 한다.

### Graph에 추가

> Put actions without an identified target, such as smiling or raising a hand, in the entity's attributes. Add a relation only when both its subject and target are supported. Do not assign another person as the target just to complete a triple.

> Include the relevant objects needed to represent actions, and reuse their exact entity IDs in relations. Repeated views of the same entity should share an ID when supported; separate instances of the same kind need separate IDs. Unrelated shots may form disconnected parts of the graph.

> Use context for additional setting, presentation, or interpretation that is not already captured by the entities and relations. Do not repeat equivalent context entries or create entities solely to store on-screen wording.

현재 예시는 다음을 보여주는 짧은 예시로 보완하는 편이 좋다.

- 표정·손들기는 속성에 남고 관계가 없어도 되는 경우.
- 인물과 음식·컵 같은 사물이 명확한 ID로 연결되는 경우.
- 같은 `name`을 가진 서로 다른 물건이 각각 다른 사람에게 연결되는 경우.
- 서로 다른 컷의 개체끼리는 연결되지 않는 경우.

고정 predicate 목록, 무조건 6개 이하 개체, 반드시 일정 수의 관계 같은 제약은 이 진단의 개선안이 아니다.

### Description에 추가

프레임별 설명·문단 수를 강제하기보다 **같은 관찰을 도입·본문·결론에 반복하지 않고, 필요한 내용이 끝나면 멈추는 것**을 강조한다. 장면을 구별하는 대상·행동을 서사적 상투어보다 먼저 보존한다.

### Summary는 후속 확인 대상

현재 [공통 Summary 정책](/home/junsu2.choi/workspace/ViewingContextPipeline/prompts/graph_summary_v4.md:1)은 입력의 해석과 불확실성을 보존하게 한다. 요약 입력에는 이미지가 없고, Graph의 Raw도 전달된다. [요약 작업 구성](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/summary_executor.py:126), [Graph 관찰 전달](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/semantic_graph/schema.py:26)

따라서 잘못된 재료명·인물 연결·자막 오역이 장면 추출에서 사실처럼 확정되면, 요약에서 이를 다시 판별하기 어렵다. 자막 문구를 요약에서 지우는 것만으로 잘못 해석된 관계까지 회복되지는 않는다. 다만 이번 자료로 실제 Summary의 실패 빈도나 추천 성능 변화는 판단할 수 없다.

## 6. 다음 비교에서 확인할 기준

프롬프트를 수정한다면 같은 34개 공통 구간에서 다음을 확인하는 것이 우선이다.

1. 원문과 번역 출력이 줄면서도 요리 재료·영상 유형 등의 의미 정보는 유지되는가.
2. 모음 영상의 앞뒤 컷 사이에 잘못된 관계가 사라지는가.
3. 동일 개체 중복과 별개 개체 병합이 모두 줄어드는가.
4. 음식·컵·휴대폰 같은 주요 대상이 등록되고 관계의 실제 주체·대상이 맞는가.
5. 단독 행동이 의미 없는 관계로 바뀌지 않는가.
6. 반복·잘림·미해결 ID가 줄어들면서 고유 정보는 보존되는가.

프롬프트와 입력을 동일하게 맞추고 재시도 수·선택된 시도도 함께 기록해야 한다. 이번 세 콘텐츠는 원인 후보를 찾기 위한 사례이며 전체 데이터의 오류율을 대표한다고 볼 수 없다. 특히 기존 요청의 핵심인 **표현 형태에서 오는 차이**를 보려면 Description과 Graph의 정보 사용 정책을 함께 보강해야 한다.
