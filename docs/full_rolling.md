# v5 전체 rolling 실험

설정은 저장소 루트의 [config.yaml](../config.yaml), 프롬프트는 [prompts](../prompts/)입니다. 실행 명령과 Run 간 프롬프트 비교 절차는 [README](../README.md)를 따릅니다.

`prepare-cohort`는 단일 실행에서 필요한 아이템 목록을 만든 뒤 `data.titles_csv`의 빈 제목·누락 행을 `data.titles_supplement_csv`로 보완합니다. 원본 제목은 유지하고 미해결 제목은 영벡터용 빈 값으로 남깁니다. 결과는 `cohort/metadata_titles.jsonl`, 보완 출처·통계는 `cohort/cohort_plan.json`에 저장하므로 별도 completed CSV나 사전 `--plan-only` 실행은 필요하지 않습니다.

## 시간·학습 계약

원본 CSV의 모든 interaction과 중복 행을 보존합니다. 정수 밀리초 timestamp를 UTC로 해석하고 마지막 관측일을 제외한 직전 7일을 평가합니다. 각 사건의 정답보다 **엄격히 앞선 시각**의 최근 10개 아이템을 입력합니다. 같은 timestamp의 사건은 서로의 이력이 되지 않습니다. 무이력 사건은 학습·평가에서 제외하지만 이후 사건의 이력과 원본에는 보존합니다.

각 날짜·seed·Arm은 독립 모델입니다. 평가 전날 이전 사건으로 학습하고 전날 validation NDCG@10으로 epoch를 선택합니다. 동일 seed의 새 모델을 평가일 이전 사건으로 선택 epoch 수만큼 refit하고 평가일을 test합니다. Test 중 파라미터를 갱신하지 않으며 당일의 앞선 사건도 이력에 반영합니다. 전체 catalog를 점수화하고 과거 관측 아이템은 정답을 제외하고 마스킹합니다.

BGE 1,024차원 특징은 고정하고 projection과 SASRec을 학습합니다. 별도 item ID embedding을 더하지 않습니다. SASRec은 hidden 512, 2 blocks, 2 heads, sequence 10, batch 256, 기본 seeds 42·43·44입니다. 5개 고정 Arm은 105개 조합입니다. `--target`은 embedding·추천·진단에서 동일하게 지원합니다.

추천도 `CUDA_VISIBLE_DEVICES`에서 보이는 모든 GPU를 자동으로 사용하며 `--gpus`는 받지 않습니다. 예를 들어 `CUDA_VISIBLE_DEVICES=0,2 python -m validation run-recommendation --run-id "$RUN_ID"`는 물리 GPU 0·2에서 날짜 × seed × Arm 조합을 병렬 학습합니다. 환경 변수를 생략하면 CUDA에서 보이는 모든 GPU를 사용합니다. 기본 GPU당 동시 작업은 1개이며 `--workers-per-gpu 2`이면 각 GPU에 2개씩 배치합니다. 남은 작업이 적으면 필요한 수의 워커만 실행합니다. GPU가 없으면 기본 설정에서 CPU로 실행하고, 이때 `--workers-per-gpu`가 1보다 크면 오류를 냅니다.

## 통계 계약

주 지표 NDCG@10과 HR/NDCG @4·8·10·20·30을 보고합니다. 사건 지표를 seed 평균하고 날짜별 적격 사건 평균을 계산한 뒤 날짜에 동일 가중치를 줍니다. 사용자를 단위로 복원 추출하는 paired bootstrap 10,000회를 수행하며 한 사용자의 모든 날짜·seed·Arm을 함께 재표집합니다.

| 비교군 | 사전 지정 비교 | 보정 분모 |
| --- | --- | --- |
| Metadata 대비 | 시각 표현 4개 각각 − Metadata | 4 |
| 표현 방식 | 모델별 Graph−Desc | 2 |
| 추출 모델 | Desc, Graph 각각 Gemini−Qwen | 2 |

각 비교군은 α=0.05, Bonferroni 양측 구간을 사용합니다. 부분 target에서는 필요한 Arm이 없는 비교를 사유와 함께 생략하고 분모는 유지합니다. 주 효과는 절대 NDCG 차이입니다. 기준값 0이면 상대 차이는 정의되지 않은 값으로 남깁니다. 구간 하한이 0보다 클 때만 우월성을 표시합니다. 기존 5% 비열등성 규칙은 제거했습니다.

Graph−Desc 개선 폭의 모델 간 차이는 보정 없는 95% 탐색적 구간입니다. 서로 다른 Graph 프롬프트는 별도 Run으로 실행하고 `run-diagnosis --compare-run-id`로 비교합니다. 두 Run의 원본 사건·catalog·평가 날짜·seed가 일치해야 합니다. 모델별 현재 Run−reference Run 비교의 보정 분모는 2이며, 부분 target에서도 유지합니다. 개선 폭의 모델 간 차이는 탐색적 95% 구간입니다. 진단의 `run_comparison`에 결과와 실제 입력 출처를 기록합니다. 요약 정책·모델·fallback이 다르면 그 차이도 포함됩니다.

## Coverage와 fallback

Description·Graph의 정상 장면 coverage는 기본 0.95 이상, Arm 간 차이는 0.05 이하여야 합니다. Raw Graph를 정상 장면으로 세지 않습니다. Metadata만 선택한 평가에는 장면 timestamp 분모가 필요하지 않습니다.

Gemini 요약 파일 부재에만 같은 Run·표현의 Qwen 요약으로 대체합니다. Raw Gemini를 우선 사용하고 손상된 파일은 실패시킵니다. 대체는 생성 coverage를 올리지 않습니다.

`representations/.inputs/{arm}.json`에 실제 요약 경로, 실제 모델 Arm, 원본 Arm·문서 hash, 요약 길이·Raw·교정 정보와 BGE truncation 집계를 기록합니다. 이 입력 정보와 벡터 hash가 추천 identity에 포함됩니다. 같은 문장·벡터라도 Gemini 결과가 추가되어 출처가 바뀌면 그 Arm의 추천 조합을 갱신합니다. 변경된 학습 설정·원본 사건도 추천 재사용을 무효화합니다.

## Artifact와 재개

`artifacts/runs/{RUN_ID}/`에는 `cohort`, `extraction`, `validation`만 둡니다. cohort 문서는 `cohort_plan.json`, `eligibility.json`, `events.jsonl`, `required_items.jsonl`, `item_inventory.jsonl`, `catalog.jsonl`, `metadata_titles.jsonl`입니다. `artifacts_root/source_assets/{content_id}/`에 timestamp·영상 길이 cache를 공유합니다.

`artifacts_root/resized_keyframes`는 모든 run이 공유합니다. 정상 PNG는 자동 덮어쓰지 않으며 콘텐츠별 디렉터리 잠금으로 동시 쓰기를 보호합니다. 프레임은 임시 공간에서 검증한 후 누락 파일만 원자적으로 게시합니다. 새 Run은 공유 PNG와 duration·timestamp를 함께 재사용합니다. 준비 정보가 이미 있으면 `prepare-input-data`를 다시 실행할 필요가 없습니다. 장면 추출은 준비된 assets를 신뢰하며 폴더 스캔·파일 존재 검사·이미지 해시 계산을 생략합니다. timestamp는 작업 구성에 필요한 입력으로 한 번 읽습니다. 이미지 내용 변경 후 재추론은 `--force`로 요청합니다. 준비 결과 통계는 콘솔로 출력합니다.

장면 JSONL은 `content_id`와 본문(`description` 또는 `scene_graph`)만 저장합니다. 장면 번호·keyframes·provenance·생성 이력은 `scenes/.metadata/{content_id}.json`에 같은 행 순서로 보존합니다. Raw Graph의 `scene_graph`는 원문 문자열입니다. 기존 JSONL도 읽을 수 있으며, `python -m extraction migrate-scene-schema --run-id "$RUN_ID"`로 본문과 메타데이터를 분리할 수 있습니다. 해당 Run의 장면 추출을 중지한 상태에서 실행하며, 형식만 바뀐 결과의 요약·임베딩은 재사용합니다.

Qwen 응답은 journal에 먼저 저장하고 최종 결과를 게시합니다. 완료한 작업의 journal은 지우고 장면 `.metadata` 또는 요약 문서에 입력 key·force 실행 ID·시도 수·repair mode를 보존합니다. 중단 시 미완료 journal과 dirty/pending/checkpoint 표식을 보존하고 같은 명령으로 재개합니다. 장면 본문과 메타데이터 저장 사이의 중단도 journal로 복구합니다. Gemini는 영상 경계 없이 scene을 병렬 처리하고, 콘텐츠별 장면이 모두 끝나면 해당 결과를 함께 게시합니다. 완료 순서가 뒤바뀌어도 저장 완료한 콘텐츠는 재사용하며, 중단 시 미완료 콘텐츠는 다시 처리합니다. `--force`는 해당 단계 생성을 새로 시작하지만 공유 이미지는 보존합니다.

신규 Summary는 `video-summary/v4` 문서 하나에 `text`, `status`, `word_count`, `violations`, `correction_count`, 입력·프롬프트·모델·생성 설정 provenance를 담습니다. 한 번 생성한 결과의 길이·형식 위반은 즉시 `summaries/failure.jsonl`에 콘텐츠 ID와 이유만 기록하고 비어 있지 않은 출력은 Raw로 남깁니다. 자동 재시도나 교정·임시 복구 기록은 만들지 않습니다. 엔진/OOM/저장 오류는 Raw로 숨기지 않고 전파합니다.

추천은 `recommendations/{date}/seed_{seed}/{arm}/`에 사건별 결과, 학습 이력, `sasrec.pt`를 저장한 뒤 `complete.json`을 마지막으로 게시합니다. 완료 검증에 실패한 조합만 재실행합니다. `training.json`에 전체 아이템 빈도 사전을 중복 저장하지 않으며 검증에 쓰는 사건별 `refit_item_frequency`는 유지합니다.

## 환경 간 전달

GPU와 Gemini 장비에서 같은 코드·run ID·설정을 사용합니다. `artifacts/runs/RUN_ID/cohort/`와 **공유** `artifacts/resized_keyframes/`, `artifacts/source_assets/`를 같은 상대 위치에 배치하고, Gemini 결과는 `extraction/description/gemini`와 `extraction/graph/gemini`로 돌려보냅니다. `scenes/.metadata` 숨김 폴더도 포함해야 합니다. 생성 provenance의 경로가 달라지면 재생성 명령의 캐시가 무효화될 수 있으므로 같은 저장소 경로를 사용합니다. 재개할 때 진행 중 journal·cursor도 함께 보존합니다.

Run 간 이동용 migration, donor-run 옵션, manifest 또는 전체 config snapshot은 없습니다. 과거 archive와 과거 run은 자동 정리하지 않습니다.
