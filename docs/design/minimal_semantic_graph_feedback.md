# Minimal Semantic Scene Graph 팀 피드백

## 문제 상황

기존에 억지로 TVTI를 위해서 PPR 등의 내용을 도입하였는데 그것보다는 확실한 scene에 대한 근거를 바탕으로 TVTI를 만드는 것을 추진해보기로 하였음




수정된 파일 

prompt_new.py

taxonomy_new.py

기존 상황
VLM 출력 데이터 (.jsonl)
│
├── scene_type ──────────────► 축1 (sociality), 축4 (utility)
├── setting ─────────────────► 축2 (syntheticity), 축3 (setting_context)
├── scene_function ──────────► 축4 (utility)
├── visual_style_cues
│   ├── media_form ──────────► 축2 (syntheticity)
│   ├── fantasy_element ─────► 축2 (syntheticity)
│   ├── shot_scale ──────────► (제외됨: close shots가 people 아닌 곳에 더 많아 노이즈)
│   ├── graphic_density ─────► (제외됨: 자막/자막오버레이가 63%에 달해 합성 미디어 판별 왜곡)
│   └── composition_density ─► (직접 축에는 미사용, 80-dim 벡터에는 포함)
│
├── mood_bin ─────────────────► 축에는 미사용 (80-dim 벡터에만 포함)
├── affect_cues ──────────────► 축에는 미사용 (80-dim 벡터에만 포함)
├── people_density ───────────► 80-dim 벡터에도 미포함 (그래프 vocabulary 밖)
├── face_prominence ──────────► 80-dim 벡터에도 미포함
│
└── entities
    ├── category ────────────► 80-dim 벡터에 `entity_category:*`로 포함
    ├── DOING ───────────────► hidden node (PPR bridge 역할)
    ├── IS_A ────────────────► hidden node (esig entity signature)
    ├── AT ──────────────────► hidden node (at location)
    └── INTERACTS_WITH ──────► SPO motif 구성 (Object 결정)




Prompt와 Taxonomy의 상호작용 (상세 분석)
🔄 핵심 플로우
prompt.py (Template 정의)
    ↓
    Import taxonomy 데이터 (line 12-24)
    ↓
_allowed_labels_text() 함수 (line 32-52)
    → taxonomy의 모든 카테고리를 프롬프트에 삽입
    ↓
SCENE_EXTRACTION_PROMPT 완성 (line 148-151)
    → VLM에 전달
    ↓
VLM이 taxonomy에 정의된 값만 생성

📋 구체적인 상호작용 3가지
1️⃣ 카테고리 목록 동적 생성

prompt.py의 _allowed_labels_text() 함수:

def _allowed_labels_text() -> str:
    sections: list[str] = [
        f"scene_type:\n{_format_values(SCENE_TYPES)}",  # ← taxonomy.SCENE_TYPES 참조
    ]
    sections.extend(
        f"visual_style_cues.{cue_name}:\n{_format_values(allowed)}"
        for cue_name, allowed in VISUAL_STYLE_CUES.items()  # ← taxonomy.VISUAL_STYLE_CUES 참조
    )


실제 프롬프트에 삽입되는 부분:

Allowed labels:

scene_type:
cheerful | dark | ... | unknown

visual_style_cues.fantasy_element:
high | mid | none | unknown

people_density:
few | many | none | one | unknown

2️⃣ Entity Category와 Role 제약

taxonomy.py 정의:

ENTITY_CATEGORIES: FrozenSet[str] = frozenset({
    "person", "animal", "food", "vehicle", "device", "object", "building", "nature", "text", "unknown"
})

ENTITY_ROLES: FrozenSet[str] = frozenset({
    "main_subject", "object", "setting_element", "background"
})


prompt.py 라인 127:

"role": "main_subject | object | setting_element | background",  # ENTITY_ROLES 사용


프롬프트에서 직접 명시:

{
  "entities": [
    {
      "category": "...",  // ← 반드시 ENTITY_CATEGORIES 중 하나
      "role": "main_subject | object | setting_element | background"  // ← ENTITY_ROLES 중 하나
    }
  ]
}

3️⃣ Relation Slots 설명

taxonomy.py (line 181-187):

RELATION_TYPES: FrozenSet[str] = frozenset({
    "DOING", "WEARING", "IS_A", "AT", "INTERACTS_WITH"
})


prompt.py의 _relation_slots_text() 함수 (line 55-66):

def _relation_slots_text() -> str:
    descriptions = {
        "DOING": 'What is this entity doing? (verb phrase, e.g. "cooking", "running")',
        "WEARING": 'What is this entity wearing? (e.g. "suit", "apron")',
        "IS_A": 'What is this entity\'s specific role/category? (e.g. "chef")',
        "AT": 'Where is this entity located? (e.g. "street", "desk")',
        "INTERACTS_WITH": "Which visible entity/object this entity directly interacts with.",
    }
    ordered = [slot for slot in ("DOING", "WEARING", "IS_A", "AT", "INTERACTS_WITH") 
               if slot in RELATION_TYPES]  # ← RELATION_TYPES 검증
    return "\n".join(...)


프롬프트에 삽입되는 부분:

Entity relation slots (per-entity attributes):
DOING        - What is this entity doing? (verb phrase, e.g. "cooking", "running")
WEARING      - What is this entity wearing? (e.g. "suit", "apron")
IS_A         - What is this entity's specific role/category? (e.g. "chef")
AT           - Where is this entity located? (e.g. "street", "desk")
INTERACTS_WITH - Which visible entity/object this entity directly interacts with.

🔗 정보 흐름 다이어그램
┌─────────────────────────────────────────────────────────────────┐
│                        taxonomy.py                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ SCENE_TYPES = {"people_social", "sport_fitness", ...}    │  │
│  │ MOOD_BINS = {"cheerful", "peaceful", "serious", ...}     │  │
│  │ ENTITY_CATEGORIES = {"person", "animal", "object", ...}  │  │
│  │ RELATION_TYPES = {"DOING", "WEARING", "IS_A", ...}       │  │
│  │ VISUAL_STYLE_CUES = {...}                                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                    ↓ Import (line 12-24)
┌─────────────────────────────────────────────────────────────────┐
│                         prompt.py                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ _allowed_labels_text()                                    │  │
│  │   - SCENE_TYPES → "scene_type: ... | ..."                │  │
│  │   - MOOD_BINS → "mood_bin: ... | ..."                    │  │
│  │   - ENTITY_CATEGORIES → "entity.category: ... | ..."     │  │
│  │                                                           │  │
│  │ _relation_slots_text()                                    │  │
│  │   - RELATION_TYPES → "DOING - ...", "IS_A - ...", ...    │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ↓ Template.substitute() (line 148-151)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ SCENE_EXTRACTION_PROMPT (완성된 프롬프트)                 │  │
│  │ "You are a visual scene-to-graph extractor.               │  │
│  │  Allowed labels:                                          │  │
│  │  scene_type: cheerful | dark | ... | unknown             │  │
│  │  mood_bin: cheerful | peaceful | ... | unknown           │  │
│  │  entity.category: person | animal | ... | unknown        │  │
│  │  Entity relation slots:                                   │  │
│  │  DOING    - What is this entity doing? ...               │  │
│  │  WEARING  - What is this entity wearing? ...             │  │
│  │  ..."                                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                    ↓ scene_extractor.py에서 사용
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen3-VL 모델                                │
│  입력: 이미지 + SCENE_EXTRACTION_PROMPT                         │
│  출력: {"scene_type": "sport_fitness",                          │
│         "mood_bin": "cheerful",                                 │
│         "entities": [{"category": "person",                     │
│                       "relations": {"DOING": "running", ...}}]} │
└─────────────────────────────────────────────────────────────────┘
                    ↓ validator.py 검증
┌─────────────────────────────────────────────────────────────────┐
│                      validator.py                               │
│  VLM 출력값을 taxonomy 정의에 맞춰 검증:                        │
│  - "sport_fitness" ∈ SCENE_TYPES? ✅                           │
│  - "running" is free-form ✅                                    │
│  - "person" ∈ ENTITY_CATEGORIES? ✅                            │
│  - "DOING" ∈ RELATION_TYPES? ✅                                │
│  → 타당성 없는 값은 기본값으로 대체 (예: "sport" → "unknown") │
└─────────────────────────────────────────────────────────────────┘

🎯 핵심 설계 원칙
항목	Taxonomy 역할	Prompt 역할
가능한 값 정의	SCENE_TYPES, MOOD_BINS, ... (enum)	❌ 정의하지 않음
VLM 제약	✅ 어떤 값이 유효한가?	✅ VLM에게 "이 값들 중 선택하세요"
유효성 검증	✅ validator.py가 taxonomy 참조	❌ (validator는 프롬프트 참고 안 함)
마이그레이션	✅ 구형→신형 값 매핑	❌
자유형 필드	DOING, WEARING, IS_A, AT (open)	✅ "verb phrase", "specific role" 가이드 제공
⚙️ 실제 동작 예시

taxonomy.py:

SCENE_TYPES = {"people_social", "sport_fitness", ..., "unknown"}


prompt.py:

Allowed labels:
scene_type:
animal_pet | food_drink | graphic_information | nature_landscape | 
object_product | people_social | person_portrait | sport_fitness | 
unknown | vehicle_transport


VLM 출력 (jsonl):

{"scene_type": "sport_fitness", "mood_bin": "cheerful", ...}


validator.py 검증:

if scene_type in SCENE_TYPES:  # ✅ "sport_fitness" ∈ SCENE_TYPES
    result["scene_type"] = "sport_fitness"
else:
    warnings.append(f"Invalid scene_type: '{scene_type}'")
    result["scene_type"] = "unknown"  # 기본값
현재 사용하고 있는 내용
1. vlm_visual_graph 내부




setting	✅ 사용	ctx:setting:* 노드 (graph.py:66)
scene_function	✅ 사용	ctx:scene_function:* 노드
mood_bin	✅ 사용	ctx:mood:* 노드
entities	✅ 사용	entity/action/place/wearing 노드의 원천
scene_type	⚠️ 추출만	scenes dict엔 담기지만(ingest.py:161) 그래프 ctx 키에 없어 미반영
people_density	⚠️ 추출만	담기지만(ingest.py:167) 그래프 미반영
face_prominence	❌ 미사용	ingest가 추출조차 안 함
affect_cues	❌ 미사용	추출 안 함
2. visual_style_cues 내부




media_form	✅ 사용	ctx:media_form:* 노드
graphic_density	⚠️ 추출만	담기지만(ingest.py:166) 그래프 미반영
fantasy_element	❌ 미사용	추출 안 함
shot_scale	❌ 미사용	추출 안 함
composition_density	❌ 미사용	추출 안 함
3. entities[] 및 relations — 전부 사용됨 ✅




name / IS_A	resolve_ident로 entity 노드 id 결정 (IS_A 우선) (ingest.py:45)
category	entity 노드 속성
role	FEATURES 엣지 속성
local_id	entity_map 구성 + INTERACTS_WITH 대상 해석
relations.DOING	action:* 노드 + DOES 엣지
relations.AT	place:* 노드 + AT 엣지
relations.WEARING	wearing:* 노드 (색상 수식어 제거 후)
relations.INTERACTS_WITH	대상 entity 해석 → ACTS_ON 엣지
요약
완전히 버려지는 필드: duration, raw_data, vlm_visual_graph_warnings, scene_idx(파일값), face_prominence, affect_cues, fantasy_element, shot_scale, composition_density
추출은 되지만 그래프엔 미반영: scene_type, people_density, graphic_density — 이 3개는 ingest.py가 scenes dict에 넣지만 graph.py:66의 ctx 키 목록(setting, scene_function, mood, media_form)에 없어 노드가 안 만들어집니다.
실제 그래프에 반영: setting, scene_function, mood_bin, media_form + entities의 모든 하위 필드와 5개 relations 전부.







Schema v2.0
현재 system에 대한 GPT의 1차 진단 

유효한 내용만 정리




중복되고 있는 내용
중복 영역	필드
사람 존재	people_density, face_prominence, person entity
콘텐츠 종류	scene_type, main entity category
분위기	mood_bin, affect_cues
장소	setting, entity의 AT
합성성	media_form, fantasy_element, graphic_density
행동 목적	scene_function, DOING, entity 구성







현재 hallucination을 강제할 수 있는 부분 

2~5개의 entity를 출력
항상 main subject와 최소 하나의 다른 element를 포함
보이는 경우 setting element를 포함
 

이 규칙은 복잡한 이미지에서는 entity 수를 줄이는 데 도움이 되지만, 다음 장면에서는 문제가 됩니다.
 
 
얼굴 클로즈업
단일 제품 사진
하늘만 보이는 자연 장면
그래픽 로고 화면
단색 배경 위의 음식 하나
 

실제로 두 번째 entity가 명확하지 않은데도 모델은 규칙을 지키기 위해 다음과 같은 것을 만들어 낼 수 있습니다.
 
 
background
table
room
wall
surface
text
 

이들은 대체로 사용자 관심사를 나타내지 않으며, graph에서는 generic hub가 될 가능성이 큽니다.

Relation 구조

현재 JSON 구조에서는 entity 내부에 relation이 들어갑니다.
 
 
{
"local_id": "e1",
"relations": {
"DOING": "...",
"WEARING": "...",
"IS_A": "...",
"AT": "...",
"INTERACTS_WITH": "e2"
}
}
 

관계 종류는 taxonomy에 정의되어 있고, prompt가 각 slot의 의미를 설명합니다.

하지만 현재 구조에는 네 가지 문제가 있습니다.

6.1 아직 생성하지 않은 ID를 미리 참조함

Autoregressive generation 순서는 다음입니다.
 
 
e1 생성
→ e1.INTERACTS_WITH = e2 생성
→ 그 뒤에 e2 entity 생성
 

즉, 모델은 아직 JSON에 정의하지 않은 e2를 먼저 계획해야 합니다. 큰 모델에서는 비교적 가능하지만, 2B 모델에서는 dangling reference나 ID 중복이 발생하기 쉽습니다.

모든 entity를 먼저 출력하고 relation을 나중에 출력하는 구조가 더 안정적입니다.

6.2 Relation이 optional인지 명확하지 않음

첫 번째 entity 예시에는 5개 relation이 모두 들어 있고, 두 번째 entity에는 INTERACTS_WITH만 들어 있습니다.

따라서 모델 입장에서는 다음 중 어느 것이 옳은지 모호합니다.
 
 
"DOING": ""
 
 
 
"DOING": "unknown"
 
 
 
// DOING key 생략
 

가장 좋은 방식은 관계가 없으면 relation record 자체를 출력하지 않는 sparse 구조




6.3 한 entity당 action 하나, target 하나만 표현 가능

현재 슬롯 구조에서는 다음 장면을 표현하기 어렵습니다.
 
 
한 사람이 한 손으로 칼을 들고
다른 손으로 채소를 잡으며
도마 위에서 채소를 자르는 장면
 

현재 구조는 대체로 다음 하나만 표현합니다.
 
 
DOING = cutting
INTERACTS_WITH = e2
 

e2가 knife인지 vegetable인지 애매합니다.

6.4 DOING과 INTERACTS_WITH의 대응 관계가 암묵적임

DOING=holding과 INTERACTS_WITH=e2라면 이해하기 쉽지만, 여러 action이나 object가 있으면 어떤 action이 어떤 object를 대상으로 하는지 알 수 없습니다.




나의 개인적인 분석 + 추가로 줄 context (이건 처음에는 GPT에 먹이지 않음 + 경험 포함)
setting과 relation의 AT이 겹침
wearing이라는 것을 qwen이 제대로 분석하지 못하는 경우가 있음 + fashion에 관련된 주제의 영상이 아닌 경우에는 크게 중요하지 않은 경우가 많음
이제 더이상 taxonomy 전체가 필요하지 않음
화면을 얼마나 작은 단위로 parsing 해낼 수 있을지가 관건이 될 것 같음
게다가 우리는 genre 기반으로 할 것이라는 것도 염두에 두고 있어야함
keyframe만 보고 있는 상황이기 때문이라는 것도 중요
















결론




GPT가 말이 너무 길어져서 따로 

지금 주어진 taxonomy.py를 보면서 전면적으로 하나씩 검토해보라고 시킴




scene_type을 vlm이 뽑아내도록 했을때 부작용이 뭐가 있을까?

media_form은 필요 없을 수도 있을거 같아.

마찬가지로 fantasy_element, shot_scale, graphic_density도 지금 화면에서 어떤게 일어나는지와는 상관이 없어보여

대신 우리는 화면에 있는 내용을 통해서 vlm이 최소한의 semantic한 주제를 알아낼 수 있었으면 해. 그러기 위해서는 지금 현재 이 파일에 정의되어 있지 않더라도 어떤 내용을 추출해야할지에 대해서 알려줘

Entity_categories에 대해서는 원래는 entity의 개념적인 의미를 파악할수 있으면 도움이 될거라고 생각한건데 이것도 필요할까에 대해서 말해줘

Relation에 대해서는 vlm이 작아서 자유형으로 관계를 뽑기보다는 해당하는 항목에 대해서만 뽑도록 해야 hallucination이 덜 생길거라고 생각했는데 어떻게 생각하는지에 대해서 말해줘

affect와 mood는 어떤면에서는 서사를 이해하기 위해서 장면적으로 알아낼 수 있는 최소한의 vlm 기능이라고 생각하는데 이 부분에 대해서는 어떻게 생각해?




최종적으로는 qwen이 아래와 같이만 뽑을 수 있도록 함




필요 없는것들 제거함

제거 항목	제거 이유
scene_type	대상·활동·구도·콘텐츠 목적이 하나의 분류 축에 혼합됨
media_form	내용보다는 표현 형식에 가까움
fantasy_element	구체적인 entity/topic으로 표현하는 편이 유용함
shot_scale	촬영 문법이며 semantic interest와 직접 관련이 적음
graphic_density	그래픽의 양일 뿐 내용은 설명하지 못함
composition_density	화면 스타일 정보
people_density	필요하면 person entity 수로 후처리 가능
face_prominence	관심사 graph에 직접 기여하지 않음
scene_function	review, tutorial, documentary 등은 단일 keyframe으로 판단하기 어려움
mood_bin	affect와 중복되며 장면 전체 분위기를 과도하게 추론할 수 있음
entity.category	구체적인 entity name보다 의미가 약하고 잘못된 type이 validation을 방해할 수 있음
entity.type	이번 설계에서 완전히 제거
IS_A	entity type/category가 없으므로 불필요
INTERACTS_WITH	의미가 지나치게 넓고 대부분의 관계를 뭉뚱그림
confidence·score	사용하지 않기로 한 프로젝트 원칙 반영




{
  "setting_context": "indoor",
  "entities": [
    {
      "local_id": "e1",
      "name": "person",
      "role": "primary"
    },
    {
      "local_id": "e2",
      "name": "vegetables",
      "role": "secondary"
    },
    {
      "local_id": "e3",
      "name": "knife",
      "role": "secondary"
    },
    {
      "local_id": "e4",
      "name": "kitchen",
      "role": "context"
    },
    {
      "local_id": "e5",
      "name": "apron",
      "role": "secondary"
    }
  ],
  "events": [
    {
      "local_id": "ev1",
      "actor_id": "e1",
      "action": "slice",
      "target_id": "e2",
      "instrument_id": "e3",
      "location_id": "e4"
    }
  ],
  "static_relations": [
    {
      "subject_id": "e1",
      "relation": "WEARING",
      "object_id": "e5"
    }
  ],
  "semantic_topics": [
    {
      "label": "vegetable preparation",
      "evidence_entity_ids": ["e1", "e2", "e3"],
      "evidence_event_ids": ["ev1"]
    },
    {
      "label": "home cooking",
      "evidence_entity_ids": ["e2", "e4", "e5"],
      "evidence_event_ids": ["ev1"]
    }
  ],
  "affect": {
    "subject_ids": ["e1"],
    "valence": "neutral",
    "arousal": "medium"
  }
}
