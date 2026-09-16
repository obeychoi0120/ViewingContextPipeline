# Graph v2/v3 프롬프트 처리량 비교

분석일: 2026-09-16. 사용자가 제공한 `scenes_v2.tar`, `scenes_v3.tar`의 본문, `.metadata`, `.recovery`를 함께 분석했다. 압축을 기존 scenes 디렉터리에 풀지 않고 tar에서 직접 읽었다. 이전 Run 결과는 이번 집계에 포함하지 않았다.

## 결론

v3는 첫 생성에서 개체를 반복 나열하다 1,024 tokens에 도달하는 경우가 매우 많다. 잘린 JSON은 필수 필드가 없거나 파싱할 수 없어 다음 pass 대기로 남는다. 생성 토큰은 많이 소비하지만 완료 장면 수는 거의 늘지 않아 표시되는 `scene/s`가 낮아진다.

두 archive의 모든 기록은 `attempt_count=1` 또는 recovery attempt 1이다. 아직 재생성은 실행되지 않았다. recovery에 기록된 repetition penalty는 모두 1.00이며 상태는 `running`이다. 즉 이번 차이는 한 장면을 즉시 여러 번 재생성하는 루프 때문이 아니다. 사용자가 요청한 전체 1.00 pass 후 실패분 재시도 정책과 일치한다.

## 동일 조건 확인

- 저장된 v2/v3 prompt hash는 각각 현재 프롬프트 파일과 일치한다.
- 모델 signature, 생성 설정, 실제 engine 설정은 두 archive에서 동일하다. RTX 6000 Ada GPU 0–3, vLLM 0.28.0, `max_num_seqs=32`, `max_new_tokens=1024`, greedy decoding을 사용한다.
- 두 실행 모두 강제 JSON 구조는 `graph/v3`의 `entities`, `relations`, `context`이다. v2 파일을 사용해도 옛 v2 출력 구조로 바뀌지 않는다. 실제 v2 결과 역시 이 세 필드로 생성되었다.
- 공개 JSONL과 `.metadata`의 payload hash는 모두 일치하며, 완료 결과와 recovery journal 사이 task ID 중복은 없다.
- 아래 현재 실행 비교에서 양쪽 모두 완료된 공통 22개 장면은 keyframes가 정확히 일치한다. 나머지 실패 journal에는 keyframes가 없어, 공통 비교는 `content_id:scene_idx`를 기준으로 한다.

## 이전 실행 결과를 제외한 공통 scene ID 238개 비교

v2 archive에는 현재 execution `90dd5a4c-35ff-4ef9-90c7-e15ca2241119`의 응답 243개와 이전 execution `d6c2c2c7-76e8-4336-828c-200a98a0abd6`의 완료 결과 74개가 함께 있다. 진행률은 재사용 결과를 제외하므로, 이번 처리량 비교에서도 이전 execution의 74개를 제외한다. v3의 433개는 모두 execution `31bd59fa-d725-4f8a-bdba-24a1e7e25dfd`다. 현재 execution은 각 archive의 모든 대기 journal이 기록한 execution ID로 식별했다.

이전 결과까지 포함하면 공통 ID가 312개지만, 아래 비교는 두 현재 execution에서 실제 응답이 기록된 공통 238개로 제한했다.

| 항목 | v2 프롬프트 | v3 프롬프트 |
|---|---:|---:|
| 생성 응답이 기록된 공통 장면 | 238 | 238 |
| 장면당 생성 횟수 | 1 | 1 |
| 저장 완료 | 141 (59.2%) | 27 (11.3%) |
| 실패 후 다음 pass 대기 | 97 (40.8%) | 211 (88.7%) |
| `finish_reason=length` | 103 (43.3%) | 214 (89.9%) |
| 응답당 평균 output tokens, 실패 포함 | 629.9 | 980.4 |
| 생성한 output tokens 합계 | 149,927 | 233,329 |
| 저장 완료 장면당 소비한 output tokens | 1,063.3 | 8,641.8 |

공통 장면 중 v2만 완료된 것은 119개, v3만 완료된 것은 5개, 양쪽 완료는 22개, 양쪽 실패 대기는 92개다. 양쪽 완료된 22개만 비교해도 출력 길이는 평균 336.0 → 616.4 tokens로 늘었다.

현재 snapshot에서 저장 완료 장면당 output token 소비는 약 8.1배다. 이 값은 실패한 생성의 토큰도 포함한 첫 pass 비용 비율이며, 실제 시간 비율이나 최종 E2E 비용은 아니다. archive에는 실행 시작·종료 시각과 GPU tokens/s 기록이 없어 사용자 관측 3.5 → 0.5 scene/s의 정확한 시간 배율을 재계산할 수는 없다.

공통 238개에서 `length` 응답 중 v2 6개와 v3 3개는 구문 복구 후 게시되었으므로 길이 한도 도달 수와 실패 대기 수는 같지 않다. 게시 완료는 JSON 구조 통과를 뜻하며 의미적 정확성을 보장하지 않는다.

## Archive 전체 집계

아래는 v2의 이전 execution 결과 74개까지 포함한 archive 전체이며, 현재 실행만의 기록은 v2 243개(완료 146, 대기 97), v3 433개(완료 61, 대기 372)다.

| 항목 | scenes_v2.tar | scenes_v3.tar |
|---|---:|---:|
| 기록된 생성 응답 | 317 | 433 |
| 저장 완료 | 220 | 61 |
| 다음 pass 대기 | 97 | 372 |
| 1,024 tokens 도달 | 103 | 376 |
| 응답당 평균 output tokens | 541.5 | 964.1 |
| 대기 오류: missing or extra fields | 81 | 333 |
| 대기 오류: JSON repair failed | 16 | 39 |

모든 대기 응답은 정확히 1,024 tokens에서 잘렸다. ID 참조 검증 오류는 없다. v3 실패 원문 372개에서 완성된 개체 객체 조각을 읽어 보면, 351개에 ID를 제외하면 같은 `name`과 `attributes`를 가진 개체 객체 반복이 있으며 33개에는 완전히 같은 관계 객체 반복이 있다. 이는 문자열 반복 지표이며, 실제 개체 동일성 검증을 새로 적용한 것이 아니다.

## 실제 사례

### microlens_100k_00037 / scene_idx=0

- v2: 169 tokens로 종료. 개체 2개와 관계 1개를 출력했다.
- v3: 1,024 tokens에서 중단. 사람과 자동차를 쓴 다음, `name: window`, `attributes: [car window]`를 `window15`, `window16`, …, `window28`로 계속 나열하고 `window29` 객체 중간에서 잘렸다.
- v3 tar 내 원문: `scenes/.recovery/188b5626063331505c8ada21b2f50878ac74231726f1bcd14737f3505763c64e.json`.

### microlens_100k_00047 / scene_idx=0

- v2: 392 tokens로 종료. 개체 6개를 출력했다.
- v3: 같은 `name: player`, `attributes: [in game, playing]`를 `player1`부터 `player28`까지 반복하고, `player29` 객체를 쓰는 중 1,024 tokens에서 잘렸다. `relations`, `context`까지 도달하지 못했다.
- v3 tar 내 원문: `scenes/.recovery/1db58f85d5b16b100103ac854b95f96b3600297395a992c3ac1212e0dba128f3.json`.

## 프롬프트 차이와 해석

v2는 개체 최대 6개, 이벤트 최대 4개, 토픽 최대 3개를 요구하고 비슷한 개체를 묶도록 지시한다. v3는 개별 개체 ID 분리, 여러 shot의 중요 내용 보존, 자유로운 attributes와 context를 요구하며 개수 상한이 없다. 현재 강제 JSON schema 역시 배열의 개수 상한이 없다.

따라서 이번 출력에서 관찰되는 차이는 v3가 더 많은 개체를 열거하도록 유도하면서 모델이 반복 나열을 끝내지 못하는 현상과 부합한다. 어느 문구가 얼마만큼 기여했는지는 문구별 대조 실험이 없으므로 확정할 수 없다. v2도 같은 종류의 반복 실패가 97개 있으므로 v2가 반복 문제를 해결한 것은 아니다.

두 실행 모두 첫 pass의 penalty=1.00에서는 반복 penalty가 적용되지 않는다. 이 조건은 같지만, 프롬프트에 따라 반복 생성으로 빠지는 빈도가 크게 다르다. 실패분의 1.05 pass는 전체 첫 pass 이후 실행된다.

## 표시 속도가 낮아지는 이유

현재 `InferenceProgress`의 `scene/s`는 게시 완료 또는 최종 실패 장면 수를 단계 경과 시간으로 나눈 값이다. 첫 pass에서 검증 실패 후 대기 중인 응답은 numerator에 포함되지 않는다. `failed=0`도 중간 검증 실패가 없다는 의미가 아니다.

따라서 v3 archive에서는 433개 응답을 생성하고도 완료 수가 61에 머문다. 이는 output tokens/s 또는 생성 요청/s와 다른 지표다. 코드 근거는 `src/extraction/recovery.py`의 실패 보류 경로, `src/extraction/scene_executor.py`의 게시 후 완료 집계, `src/extraction/progress.py`의 rate 계산이다.

상세 수치·설정·입력 archive SHA-256은 같은 디렉터리의 `graph_scene_v2_v3_throughput_260916_stats.json`에 기록했다.
