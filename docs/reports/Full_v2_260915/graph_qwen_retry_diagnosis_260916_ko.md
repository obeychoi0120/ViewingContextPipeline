# graph_qwen 첫 10개 영상 재시도 진단

분석일: 2026-09-16. 대상: `Full_v2_260915`, 파일명 순서 `microlens_100k_00001`–`00010`의 전체 84 scene. 실행 코드와 최종 저장 결과를 읽었으며 모델 재실행이나 설정 변경은 하지 않았다.

## 집계

| 영상 | Scene | 총 생성 시도 | 재시도한 Scene | 최종 Raw fallback |
|---|---:|---:|---:|---:|
| 00001 | 19 | 56 | 17 | 5 |
| 00002 | 7 | 19 | 6 | 2 |
| 00003 | 11 | 43 | 10 | 7 |
| 00004 | 6 | 24 | 6 | 3 |
| 00005 | 12 | 31 | 10 | 2 |
| 00006 | 6 | 23 | 5 | 3 |
| 00007 | 5 | 17 | 5 | 1 |
| 00008 | 7 | 26 | 6 | 4 |
| 00009 | 8 | 28 | 7 | 2 |
| 00010 | 3 | 13 | 3 | 1 |
| **합계** | **84** | **280** | **75** | **30** |

- 최초 시도 통과 9/84 (10.7%). 재시도 발생 75/84 (89.3%).
- 생성 요청 수는 1회씩 생성하는 경우의 3.33배: 추가 생성 196회. 이는 요청 수 비율이며 실제 소요 시간 배율은 아니다.
- 1/2/3/4/5회 생성한 scene 수: 9/25/12/5/33. 5회까지 간 33개 중 3개만 구조 검증 통과, 30개는 Raw fallback.
- 최종 구조 검증 통과 54개: native 52, syntax repaired 2. Raw fallback 30개 (35.7%).
- 첫 10개 **scene**만으로 좁혀도 총 30회 생성, 재시도 8개, Raw fallback 3개.

## 직접 재검증한 오류

Raw fallback 30개의 원문을 현재 `graph_scene_result(..., strict=True)`로 재검증했다. 28개는 `graph: unresolved relation reference`. 나머지 2개는 1,024 tokens에서 JSON이 잘렸으며 복구 후 `graph: missing or extra fields`로 거부된다. 최종 선택된 출력 전체에서는 4개가 `finish_reason=length`이고 그중 2개는 문법 복구로 통과했다.

### 대표 사례 (scene_idx는 0부터 시작)

- [00001/S0](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/runs/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:1): entities에는 person1/person2/eyes3/woman4만 있지만 관계가 text1, text2를 참조한다.
- [00002/S0](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/runs/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00002.jsonl:1): entities에는 person1/person2만 있지만 관계가 food, vegetables, celery를 참조한다.
- [00003/S5](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/runs/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00003.jsonl:6): 등록하지 않은 man2, person3를 관계 주체로 사용하고 trash bin도 미등록이다.
- [00008/S2](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/runs/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00008.jsonl:3): entities에는 person1/person2만 있지만 plate, egg mixture, lid, cabinet door, bench, shoes, onion bunch를 관계 대상으로 쓴다.
- [00001/S2](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/runs/Full_v2_260915/extraction/graph/qwen/scenes/microlens_100k_00001.jsonl:3): 1,024 tokens까지 관계를 나열하다 JSON이 잘린다. value_emotions_above_materials, belief_system 등의 추상 대상을 관계 ID로 확장한다.

## 왜 반복되는가

1. [JSON schema](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/structured_output.py:20)는 ID를 자유 문자열로 정의한다. 생성 시 JSON 구조를 제약해도, 값이 앞서 생성한 entities ID에 포함되는지까지 이 schema가 제한하지 않는다. [후처리 검증](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/structured_output.py:50)은 참조 무결성을 검사하므로 이 단계에서 거부된다.
2. [재시도](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/recovery.py:249)는 같은 task의 repetition_penalty만 1.0 → 1.05 → 1.1 → 1.15 → 1.2로 바꾼다. 오류 메시지나 실패한 graph를 수정 입력으로 전달하지 않는다. 따라서 반복 억제가 미등록 ID 문제를 직접 해결하지 못한다.
3. [Graph 실행 경로](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/scene_executor.py:173)는 영상별로 최대 8개 scene 묶음을 처리하고, 묶음의 재시도까지 완료해야 다음 묶음을 시작한다. GPU별 max_num_seqs=32여도 상위에서 공급하는 총 요청은 최대 8개이고 재시도 라운드마다 더 줄어든다. 선택된 결과의 GPU 분포도 0/1/2/3 = 28/26/16/14다. 이는 최종 선택 요청 분포이며 GPU 이용률 실측은 아니다.
4. [최종 fallback](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/recovery.py:280)은 5번 실패 후 비어 있지 않은 원문을 결과로 게시한다. 게시 콜백에서는 정상 record로 취급하므로 progress의 success에 포함된다. `failed=0`은 구조 검증 실패가 없다는 뜻이 아니다.

## 해석 범위와 개선 우선순위

- 저장 결과의 prompt hash는 현재 graph_scene_v3.md의 SHA-256 `dbdc17a2c01093f3b7363e1875698bad36eda7c9f9da285a6b24f40d72aa9f04`와 일치한다. 현재 프롬프트에도 정확한 ID 재사용과 사물 등록 지시가 있지만 위 결과에서 지켜지지 않는다.
- 완료한 scene의 recovery journal은 삭제된다. 따라서 196회 과거 추가 시도 각각의 오류 종류·token 수·소요 시간은 복원할 수 없다. 28/2 분류는 **최종 Raw 30개**의 분류이지 전체 실패 시도 226회의 분류가 아니다.
- 우선 scene 공급을 영상 경계·재시도 묶음 대기에서 분리하고, unresolved reference와 출력 길이 초과에 다른 대응을 적용하는 방향이 타당하다. 참조 오류에는 생성된 ID 집합을 반영하는 제약 또는 오류와 원문을 주는 제한적 수정 단계를 별도 검증할 수 있다.
- 단순히 토큰 상한을 높이는 것은 미등록 ID 오류를 해결하지 않으며, 잘못된 관계를 자동 삭제하거나 미등록 대상을 자동 추가하면 평가 대상 graph의 의미가 바뀐다. 그런 변경은 별도 품질 검증이 필요하다.
- progress에는 valid graph / raw fallback / retries를 구분해 표시해야 현재 품질과 추가 생성량을 볼 수 있다.
