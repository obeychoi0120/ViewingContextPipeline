# Context + Actions Graph 실험 규약

## 설계 철학

추천에 필요한 맥락과 정확한 활동·대상·도구를 선택적으로 보존합니다. 외형을 자세히 나열하거나 관계 수를 채우는 것보다 행동 대상과 동일 개체를 정확히 표현하는 것을 우선합니다. 화면 텍스트는 일반적 주제·활동 판별에만 사용하며 대사·가사·상세 줄거리의 재서술은 제외합니다. 구조화 자체가 저작권 안전성을 보장한다는 가정은 하지 않습니다.

실험 프롬프트는 `prompts/scene_graph_v5.md`, `prompts/summary_graph_v6.md`입니다. 기존 프롬프트와 실행 결과는 보존합니다. 기본 경로 승격과 전체 재생성은 자동으로 하지 않습니다. Description 및 추천 모델은 변경하지 않습니다. 생성 예산은 Scene 1,024 / Summary 2,048토큰이며, Summary는 영문 200–300단어 권장·350단어 최대 지침을 유지합니다. 근거가 적으면 짧게 끝냅니다.

## 저장 및 호환성

필수 섹션은 `[Context]`, `[Entities]`, `[Actions]` 세 개이며 이 순서로 출력합니다. 마지막의 `[End]`는 선택 사항이고, 출력된 경우 그 뒤에 내용이 올 수 없습니다. `medium`, `format`, `topics`는 저장되는 `scene_graph` 객체의 **최상위 필드**입니다.

```json
{
  "medium": "live_action",
  "format": "demonstration",
  "topics": ["seafood preparation"],
  "entities": [
    {"id": "p1", "name": "person", "attributes": []},
    {"id": "f1", "name": "sea urchin", "attributes": []},
    {"id": "t1", "name": "knife", "attributes": []}
  ],
  "actions": [{"actor": "p1", "action": "cutting", "target": "f1", "tool": "t1"}]
}
```

정식 Actions 문법은 `actor - action - target; tool`입니다. 화살표는 방향을 바꾸지 않는 보정 경로로 허용합니다. ID·단어 내부 하이픈은 허용합니다. `none` 참조는 `null`, `topics: none`과 빈 섹션은 빈 배열로 저장합니다. 최소 하나의 actor/target이 필요하며 모든 참조는 선언된 고유 ID여야 합니다.

medium/format 허용 목록과 topics 3개·각 4단어, entities 6개·속성 2개·각 6단어, actions 4개 제한은 프롬프트 지침입니다. 저장 검증에서는 이 제한을 적용하지 않고, 초과 항목 및 목록 밖의 문자열도 삭제·변환 없이 보존합니다. 필수 구조·Actions 문법·참조 무결성·고유 ID만 검사합니다. 세 필수 섹션과 내용이 완전하면 `[End]` 없이 끝나도 정상으로 처리합니다. 백엔드가 토큰 종료를 보고하면 `[End]` 유무와 관계없이 기존 실패·재시도 흐름을 적용합니다.

파서는 `graph-text/v7`, 새 추출 규약 provenance는 `graph/v4`, 내부 schema 상수는 `scene-graph/v4`입니다. 구형 프롬프트의 provenance는 `graph/v3`를 유지합니다. 구형 Entities/Relations·Context 목록·JSON은 읽을 수 있으며 Actions로 추측 변환하지 않습니다. Graph 저장 행에는 `content_id`, `warning`, `scene_idx`, `scene_graph` 순서로 네 필드를 저장합니다. `provenance`와 `graph_format`은 저장하지 않습니다. 파싱·검증 실패 원문은 `scene_graph` 문자열로 보존하고 값의 타입으로 판별해 Summary에 전달합니다. 과거 실패 원문의 재시도 상태는 기존 실패 로그에 보존합니다. 이전 추가 필드가 있는 파일도 읽을 수 있으며, 다시 저장하거나 `migrate-scene-schema`를 실행하면 현재 네 필드로 변환합니다. 이 원문은 `structured` 진행 수와 Summary의 `normal_scene_count`에서 제외하며 `raw_scene_count` 및 `text_scene_count`에 반영합니다. `success` 진행 수는 저장에 성공한 전체 응답 수이므로 원문도 포함합니다.

## Warning 태그

`warning`은 중복 없는 태그 배열입니다. 정상·보정 성공 결과는 `[]`이며, raw 원문에는 확인된 오류만 기록합니다. 파싱 후 중복 ID와 미선언 참조를 함께 확인할 수 있으면 두 태그를 모두 기록합니다. 오류 설명 문자열은 태그로 저장하지 않습니다.

| 태그 | 조건 |
| --- | --- |
| `MISSING_REQUIRED` | 필수 섹션·필드 누락 |
| `INVALID_ACTION_SYNTAX` | Actions 구분자·tool 자리 오류, 빈 action, actor/target 모두 없음 |
| `INVALID_REFERENCE` | 선언되지 않은 actor/target/tool ID |
| `DUPLICATE_ENTITY_ID` | 중복 엔티티 ID |
| `PARSE_ERROR` | 깨진 JSON, 잘못된 타입, 섹션 순서·중복, 모호한 행 등 나머지 파싱 오류 |

```json
{"content_id":"video","warning":["INVALID_REFERENCE"],"scene_idx":0,"scene_graph":"...original model output..."}
```

Actions의 tool 자리 누락은 `MISSING_REQUIRED` 대신 `INVALID_ACTION_SYNTAX`로 분류합니다. 빈 응답·공백만 있는 응답은 Qwen과 Gemini 모두 raw로 저장하지 않고 생성 실패로 처리합니다. Qwen은 기존 penalty 순서로 재시도합니다. 토큰 한도 종료·API 오류도 warning이 아닌 기존 생성 실패·재시도 경로로 처리합니다. warning이 없는 과거 문자열은 읽을 때 현재 파서로 재검사하여 태그를 복원하며, 파일에는 다음 저장·명시적 마이그레이션 때 반영합니다. 따라서 복원된 태그는 과거 실행 당시의 정확한 진단을 보장하지 않습니다. 경고는 Summary 관찰 내용에 추가하지 않습니다.

## 첫 100개 파일럿 도구

아래 명령은 **모델을 호출하지 않고** 비교 manifest, 키프레임 경로·해시, 장면 ID별 비교 입력과 환경 사전 점검 결과만 만듭니다. 준비된 공유 cohort를 content ID 순으로 정렬해 처음 100개를 고정하며 공유 준비 데이터는 수정하지 않습니다.

```bash
PYTHONPATH=src python -m benchmarks.graph_prompt_pilot --run-id graph_context_actions_first100_v2 --baseline-run v4_260922
```

모델 경로가 이동했다면 `--model-path /absolute/path/to/checkpoint`를 지정할 수 있습니다. 기존 provenance의 모델 서명은 파일 수정 시각도 포함하므로 복사된 동일 모델도 불일치할 수 있습니다. 불일치를 동일 가중치의 증거로 취급하지 않으며, 동일 조건 확인이 끝나기 전 실행을 차단합니다.

나중에 생성할 때만 `--execute`를 추가합니다. Qwen Scene 추출 후 **baseline에 기록된 Summary 모델**을 사용합니다. 기존 `v4_260922`의 graph_qwen은 Qwen 추출·Gemini Summary 조합입니다. 첫 100개 이외의 영상이나 Description은 생성하지 않으며 `--force`를 사용하지 않습니다. 동일 manifest는 재개 가능하고 입력·프롬프트가 달라지면 새 run ID가 필요합니다.

`graph_prompt_pilot/` 산출물:

- `manifest.json`: 선택 ID, 프롬프트·baseline·키프레임 해시, 모델·추출 설정.
- `preflight.json`: 실행 전 조건 불일치. API 접근 권한 및 과거 키프레임 파일 내용의 동일성까지 보장하는 것은 아닙니다.
- `scene_pairs.jsonl`: 장면 ID로 연결한 baseline/candidate와 시각적 검토용 키프레임. 누락 장면도 포함합니다. 재실행 시 다시 작성하므로 수동 평가는 별도 파일에 기록합니다.
- `metrics.json`: 구조화·원문·누락 수와 전체 예상 장면 수 대비 구조화율. 아직 생성하지 않은 candidate의 0은 품질 평가 결과가 아닙니다.
- `attempts.jsonl`: 실행 시 Qwen 각 시도의 출력 토큰·종료 사유. 재시도를 포함합니다. Gemini 및 계측이 없는 과거 결과의 토큰 수·종료율은 추정하지 않습니다.

## 품질 검토와 채택 기준

자동 테스트는 문법·저장·전달을 검증합니다. 시각적 의미의 정확성이나 표현 잔존 감소를 입증하지 않습니다. 실제 생성은 이번 구현 검증에 포함하지 않습니다.

동일 장면의 baseline과 candidate를 비교해 주제·형식, 행동·대상·도구, 동일 대상 중복, 가짜 상호작용, 원문 표현 잔존, 기존 중요 정보 보존을 평가합니다. 불일치는 Description을 정답으로 고정하지 않고 키프레임으로 판정합니다. 관찰 불가능한 항목은 미확인으로 남깁니다. 주요 회귀 사례는 다음과 같습니다.

| 영상 번호 / scene_idx | 확인 사항 |
| --- | --- |
| 8 / 1 | 성게 손질의 대상이 성게인지, 사람으로 연결하지 않는지 |
| 48 / 0 | 반복된 동일 캐릭터를 여러 명으로 만들지 않는지 |
| 54 / 2, 84 / 2 | 게임 강화·캐릭터 편집을 개체 간 상호작용으로 만들지 않는지 |
| 98 / 0 | 내용물 변화와 동일 컵의 정체성을 함께 보존하는지 |
| 96 | 메이크업 활동·도구를 보존하고 인물을 중복하지 않는지 |
| 14 / 14 | 처치 주체·대상 및 수동형 표현의 방향을 보존하는지 |
| 32 / 1 | 기존 백팩·태블릿 수납 정보가 유지되는지 |

맥락·행동 정확도, 표현 잔존 감소, 보존 사례가 확인된 뒤 동일 조건의 추천 평가로 기본 경로 채택을 판단합니다. 이 도구는 추천 평가나 전체 데이터 재생성을 자동 실행하지 않습니다.
