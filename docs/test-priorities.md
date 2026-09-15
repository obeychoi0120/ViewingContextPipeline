# 테스트 함수 우선순위 분류

분류 기준일: 2026-09-15. 삭제 전 작업 트리의 `tests/**/test_*.py` 전체 183개 함수를 대상으로 분류했다. 당시 커밋되지 않은 변경과 새 `tests/preparation/`도 포함했다.

## 기준

- **P0 · 중요**: 기본 실행, 입력·결과 무결성, 평가 수식·시간 분할, 캐시 일관성, 중단·저장 실패 복구를 지키는 테스트. 깨지면 전체 실행 또는 실험 결론을 신뢰하기 어렵다.
- **P1 · 추가 조건**: 경계값, 선택 옵션, 호환 경로, 운영 진단, 성능·자원 제약의 추가 검증. 유지할 가치가 있다.
- **P2 · 불필요**: 독립 테스트로는 중복·형식 고정이거나 현재 파이프라인에서 쓰이지 않는 경로의 테스트. 아래 통합·코드 정리 조건을 충족한 뒤 제거하는 후보이다.

함수에 여러 검증이 섞여 있으면 가장 중요한 실제 검증을 기준으로 분류했다. 테스트 이름, 길이, mock 사용 여부, 실행 시간만으로 중요도를 정하지 않았다. 특히 실패 처리라도 데이터 손실·잘못된 결과·무한 대기를 막는 경우 P0이다.

## 제거 결과

후속 요청에 따라 **P1 84개와 P2 13개, 총 97개 테스트 함수**를 제거했다. **P0 86개**의 함수 본문과 decorator는 그대로 유지했다. 테스트가 남지 않은 파일 7개와 불필요한 import·전용 helper도 정리했다. 제품 구현은 변경하지 않았다.

아래 분류표와 P2 판단 근거는 삭제 전 검토 기록이다. P1·P2는 이미 제거된 함수이며, P0 링크만 현재 소스 위치를 가리킨다. 이전의 통합·조건부 삭제 제안보다 후속 요청의 전체 제거 지시를 적용했다.

## 삭제 전 집계

개수는 **함수 정의 기준**이다. parametrize로 확장되는 실행 사례 수와 다르다.

| 영역 | P0 | P1 | P2 | 합계 |
| --- | ---: | ---: | ---: | ---: |
| extraction | 34 | 47 | 2 | 83 |
| integration | 5 | 2 | 1 | 8 |
| preparation | 0 | 6 | 1 | 7 |
| v5 | 25 | 13 | 1 | 39 |
| validation | 22 | 16 | 8 | 46 |
| **전체** | **86** | **84** | **13** | **183** |

## 삭제 전 P2 정리 근거

### 중복 검증 통합

- `test_bootstrap_paired_clusters_equal_date_and_seeds` → `test_equal_date_mean_and_paired_user_multiplicity`. 동일한 입력 배열로 같은 `cluster_bootstrap` 함수를 검사한다. 전자의 aggregation 문자열 assertion을 옮기면 두 함수를 따로 유지할 필요가 작다. 전자도 실제 seed별 값을 평균하는 경로를 검사하지는 않는다.
- `test_recommendation_metric_passes_target_row_to_ranker` → `test_metrics_and_repeated_target_history_mask`. 같은 점수·history·target의 직접 함수 조합이다. 정확한 NDCG assertion을 옮긴 뒤 통합한다.
- `test_preparation_steps`, `test_documented_validation_step_matrix_is_stable` → `test_documented_commands_dispatch`와 `test_preparation_commands_removed_from_old_clis`를 중심으로 통합. 등록 순서는 제거하고, 허용 명령을 정확히 제한하려면 집합 검사는 보존한다.
- `test_readme_documents_complete_stage_matrix`의 명령 개수 17 고정은 제거하고, 위 문서 dispatch 테스트에 필요한 stage/model/source 조합 집합 검사를 통합한다. dispatch 검사만으로 명령 누락까지 탐지되지는 않는다.

### 현재 실행 경로와 떨어진 검증

- `test_metrics.py`의 P2 6개는 `paired_bootstrap_ci`, `paired_relative_bootstrap_ci`, `bonferroni_alpha`를 검사한다. 저장소의 `src`·`benchmarks` 검색에서 정의 외 호출자가 없으며, 현재 진단은 `rolling_diagnosis.cluster_bootstrap`, `comparisons`, `run_comparison.compare_graph_runs`를 사용한다. 관련 함수와 테스트를 함께 정리하는 후보이다. 저장소 밖에서 사용하는 API인지까지 확인한 것은 아니므로 이 helper를 지원 API로 유지한다면 해당 테스트는 P1로 유지한다.
- `test_scene_evidence_indexes_frame_directory_once`는 `prepared=False` 스캔 경로를 검사한다. 현재 추출은 `scene_generation_rows`에서 `prepared=True`로 호출한다. 호환 스캔 경로를 제거할 경우 함께 제거하며, 현재 경로의 `test_extraction_builds_paths_without_probing_assets`는 유지한다.

### 출력 모양 고정

- `test_description_scene_message_format`은 단일 설명 문자열의 출력 모양을 고정한다. 삭제하거나 monitoring 입력 사례 표에 통합한다. 영상·모델 식별, 오류 전달, 완료·실패 집계 검증은 P1/P0로 유지한다.

## 함수별 분류 기록

각 함수는 아래에서 한 번씩만 기재한다. P0는 현재 소스 링크, P1·P2는 제거된 함수 이름으로 표시한다.

### tests/extraction/tests/test_backends.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_gemini_backend_uses_images_prompt_and_operational_controls](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_backends.py:35) | Gemini 요청에 이미지·프롬프트·생성 제한이 올바르게 전달되는 핵심 연결. |
| P1 | `test_gemini_empty_response_preserves_sdk_diagnostics` (제거됨) | 빈 SDK 응답의 차단 사유·토큰 진단 보존. |
| P1 | `test_gemini_empty_response_without_metadata` (제거됨) | 진단 메타데이터도 없는 빈 응답 처리. |

### tests/extraction/tests/test_data_preparation.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_fixed_30s_sampling_uses_5_15_25_second_keyframes` (제거됨) | 공개된 기존 3프레임 진입점의 호환성; 기본 6프레임 정책 검증은 별도로 유지. |
| P1 | `test_direct_keyframe_uses_preceding_frame_for_safe_trailing_seek` (제거됨) | 영상 끝 seek 실패 시 직전 디코딩 가능 프레임으로 복구. |
| P1 | `test_failed_trailing_seek_does_not_log_success` (제거됨) | 복구도 실패하면 예외를 내고 성공으로 기록하지 않는 조건. |
| P0 | [test_verified_keyframe_cache_rejects_corrupt_or_wrong_size_images](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_data_preparation.py:16) | 손상·잘못된 크기의 공유 이미지를 정상 캐시로 재사용하지 않도록 검증. |
| P1 | `test_last_decodable_frame_uses_latest_ffprobe_frame_timestamp` (제거됨) | ffprobe 출력이 정렬되지 않거나 중복돼도 마지막 프레임을 올바르게 선택. |
| P0 | [test_prepare_visual_item_reuses_catalog_duration_and_uses_shared_root](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_data_preparation.py:43) | 준비 단계의 duration·timestamp·공유 프레임 경로 연결. |
| P0 | [test_prepare_catalog_processes_exact_cohort](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_data_preparation.py:70) | 지정 cohort만 준비하고 이전 실패 보고서를 정리하는 기본 동작. |

### tests/extraction/tests/test_evidence.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P2 | `test_scene_evidence_indexes_frame_directory_once` (제거됨) | 현재 추출은 prepared=True여서 스캔하지 않음. prepared=False 경로를 정리할 때 함께 제거; 해당 호환 경로를 유지하면 P1. |

### tests/extraction/tests/test_gemini_workers.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_gemini_pool_completes_out_of_order_and_captures_errors](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_gemini_workers.py:14) | Gemini 병렬 작업의 완료 순서·결과 대응과 개별 API 실패 격리. |
| P1 | `test_gemini_pool_rejects_duplicate_task_ids` (제거됨) | 중복 task ID 입력 거부. |
| P1 | `test_gemini_pool_supports_sixteen_simultaneous_scenes` (제거됨) | 16개 요청의 실제 동시 시작 조건; 단순 결과 개수 검사보다 강하므로 유지. |
| P1 | `test_gemini_pool_propagates_empty_response_diagnostics` (제거됨) | worker 경계를 넘어 빈 응답 진단을 보존. |
| P0 | [test_gemini_pool_propagates_ctrl_c_without_waiting_for_http_call](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_gemini_workers.py:68) | 응답이 멈춘 API 호출 중에도 Ctrl-C를 전달하는 종료 경로. |

### tests/extraction/tests/test_inference_progress.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_throughput_excludes_initialization_and_requires_warmup` (제거됨) | 초기화 시간을 처리량에서 제외하고 warmup 이후 계산. |
| P1 | `test_wall_clock_window_handles_bursts_and_no_completion_periods` (제거됨) | 동시 완료와 무응답 구간에서 처리량 계산의 경계 조건. |
| P1 | `test_progress_uses_only_pending_scenes_and_separates_failures` (제거됨) | 재시도·캐시를 제외한 장면 진행률과 성공·실패 집계; 문자열 모양 검사는 축소 가능. |
| P1 | `test_scene_rate_does_not_reset_when_gemini_starts_next_content` (제거됨) | 다음 콘텐츠 시작 시 전체 처리량·ETA가 초기화되지 않는 회귀. |
| P1 | `test_timer_recalculates_every_five_seconds_even_without_completions` (제거됨) | 완료 이벤트가 없어도 타이머가 ETA를 갱신하고 종료 시 정리됨. |
| P1 | `test_finished_cached_and_interrupted_progress` (제거됨) | 전부 캐시됨·전부 실패·중단 상태의 진행률과 ticker 정리. |

### tests/extraction/tests/test_monitoring.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_video_names_use_source_basename_and_fallback` (제거됨) | 운영 로그의 원본 영상 식별과 경로 형식별 fallback. |
| P1 | `test_graph_scene_messages_are_sorted_and_use_normalized_json` (제거됨) | 여러 장면의 순서와 JSON을 읽을 수 있는 로그 형식 검증. |
| P2 | `test_description_scene_message_format` (제거됨) | 단일 설명 로그의 접두사·공백·줄바꿈 문자열 고정. 함수 단독 테스트는 제거하거나 monitoring 사례 표에 통합. |
| P1 | `test_graph_skip_message_flattens_error_lines` (제거됨) | 다중 행 오류와 Gemini 오류 출처를 알아볼 수 있는 로그 검증. |
| P1 | `test_graph_monitoring_can_identify_source` (제거됨) | 모델별 결과를 구분하는 source 표기 검증. |
| P0 | [test_generator_uses_visible_gpu_pool_and_passes_completed_request_metadata](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_monitoring.py:10) | 비동기 완료 결과에 해당 요청의 runtime metadata를 붙이고 다음 요청과 섞지 않음. |

### tests/extraction/tests/test_qwen_backend.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_native_images_order_and_greedy_settings](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:72) | 이미지 순서·프롬프트·greedy 설정·생성 토큰 penalty·종료 토큰 연결. |
| P0 | [test_structured_output_and_generated_token_penalty_are_passed_together](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:96) | 구조화 출력 제약과 생성 토큰 penalty의 동시 적용. |
| P0 | [test_structured_output_compilation_error_never_generates_unconstrained](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:107) | 제약 컴파일 실패를 제약 없는 생성으로 숨기지 않음. |
| P0 | [test_text_only_sampling](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:118) | 요약에 쓰는 텍스트 전용 요청과 sampling 설정 전달. |
| P0 | [test_expanded_image_context_is_checked_without_truncation](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:128) | 이미지 확장 후 입력 길이가 상한을 넘으면 조용히 잘라내지 않고 실패. |
| P1 | `test_loader_parallelism_is_bounded_and_images_are_closed` (제거됨) | 이미지 로딩 동시성 제한과 정상 완료 후 이미지 해제. |
| P1 | `test_cancelled_preparation_releases_images_when_loader_finishes` (제거됨) | 로딩 도중 취소되는 경쟁 상황에서 이미지 해제. |
| P0 | [test_batch_penalty_matches_original_and_tracks_live_tokens](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:135) | 생성된 토큰에만 penalty 적용; 기준 구현과 수치 비교. |
| P0 | [test_penalty_batch_removal_replacement_move_and_swap](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:152) | 연속 배치의 이동·교환·삭제 시 다른 요청에 penalty가 적용되는 오류 방지. |
| P1 | `test_invalid_penalty` (제거됨) | 잘못된 penalty 값 거부. |
| P1 | `test_invalid_runtime_settings` (제거됨) | 잘못된 runtime 값·미지원 필드·충돌하는 설정 거부. |
| P1 | `test_old_config_uses_same_defaults` (제거됨) | 기본 설정과 명시적 override의 호환성; 기본값 자체가 불변임을 검증하는 테스트는 아님. |
| P0 | [test_engine_configuration_and_eos_are_explicit](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_backend.py:170) | 실제 backend 생성 시 엔진 설정·이미지 제한·EOS 선택이 의도대로 연결됨. |

### tests/extraction/tests/test_qwen_benchmark.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_benchmark_excludes_warmup_and_records_partial_failures` (제거됨) | 부가 benchmark의 warmup 제외·부분 실패·결과 보존. |
| P1 | `test_output_validation_uses_stage_contract` (제거됨) | benchmark 출력 검증이 장면·요약의 계약을 따름. |
| P1 | `test_export_uses_first_penalty_and_the_graph_schema` (제거됨) | benchmark export의 penalty와 Graph schema 일치. |
| P1 | `test_transformers_reference_rejects_constraints_before_loading_any_model` (제거됨) | 기준 backend가 지원하지 않는 제약 요청을 모델 로드 전에 거부. |

### tests/extraction/tests/test_qwen_runtime.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_result_hash_is_independent_of_key_order` (제거됨) | benchmark 요청 식별 hash가 key 순서와 무관하고 내용 변경을 반영. |
| P1 | `test_checkpoint_metadata_is_kept_without_creating_runtime_log` (제거됨) | checkpoint provenance 보존과 별도 runtime 로그를 생성하지 않는 조건. |

### tests/extraction/tests/test_qwen_workers.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_completion_refills_fast_gpu_before_saving_without_batch_barrier](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_workers.py:82) | 빠른 GPU에 후속 요청을 공급하고 결과를 요청별로 올바르게 저장. |
| P0 | [test_initial_admission_is_bounded_and_runtime_events_are_delivered](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_workers.py:104) | 대규모 입력을 한꺼번에 적재하지 않는 admission 제한과 ready 이벤트 전달. |
| P1 | `test_progress_starts_after_all_engines_ready_without_blocking_fast_gpu` (제거됨) | 엔진별 초기화 시간 차이에도 빠른 GPU를 막지 않고 처리량을 계산. |
| P1 | `test_progress_refreshes_during_waits_without_completions` (제거됨) | 결과가 없는 대기 중 progress callback 갱신. |
| P0 | [test_callback_failure_keeps_prior_saves_and_stops_all_workers](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_workers.py:132) | 저장 실패·중단 시 이전 결과를 보존하고 모든 worker·queue를 정리. |
| P0 | [test_engine_failure_is_fatal](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_workers.py:154) | OOM·예상 밖 결과·worker 종료를 오류로 전달하고 pool 종료. |
| P1 | `test_duplicate_ids_rejected` (제거됨) | 중복 요청 ID 거부. |
| P1 | `test_engine_readiness_can_be_measured_before_request_submission` (제거됨) | benchmark용 사전 ready 대기와 준비된 엔진 재사용. |
| P0 | [test_all_visible_devices_preserved](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_workers.py:170) | CUDA_VISIBLE_DEVICES의 번호·UUID·MIG·마스크를 보존한 GPU 배정. |
| P1 | `test_no_visible_gpu_fails_clearly` (제거됨) | GPU가 없는 Qwen 환경에서 명확한 실패. |
| P0 | [test_worker_reuses_one_engine_and_passes_requests_concurrently](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_qwen_workers.py:179) | worker마다 엔진 한 개를 재사용하며 요청을 동시에 처리. |
| P1 | `test_spawn_uses_non_daemon_and_bounded_queue` (제거됨) | vLLM 하위 프로세스를 허용하는 non-daemon 설정과 queue 용량 제한. |
| P1 | `test_partial_startup_failure_disposes_first_worker` (제거됨) | 일부 worker만 시작된 뒤 실패할 때 자원 정리. 현재 테스트는 이미 종료한 객체를 재사용하므로 정리 여부 assertion 강화 필요. |

### tests/extraction/tests/test_semantic_graph_json_repair.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_deterministic_repair_cases](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_semantic_graph_json_repair.py:17) | 흔한 비정상 JSON을 정해진 규칙으로 복구하고 parse mode 기록. |
| P1 | `test_fenced_and_noisy_json_is_parsed_without_repair` (제거됨) | 코드펜스·주변 설명이 있는 정상 JSON 처리. |
| P1 | `test_later_valid_object_is_selected_after_malformed_object` (제거됨) | 앞선 객체가 깨져 있어도 뒤의 유효 객체를 선택. |
| P0 | [test_unrepairable_output_is_reported](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_semantic_graph_json_repair.py:24) | 빈 응답·복구 불가 응답을 정상 Graph로 통과시키지 않음. |

### tests/extraction/tests/test_structured_recovery.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_penalty_rejects_invalid_schedule` (제거됨) | penalty 재시도 목록의 잘못된 타입·범위 거부. |
| P1 | `test_penalty_preserves_order_and_scalar_compatibility` (제거됨) | penalty 순서 보존과 scalar 입력 호환. |
| P0 | [test_only_failed_tasks_retry_constraints_stay_enabled_and_raw_is_last_nonempty](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:43) | 실패 작업만 재시도하고 제약 유지·중복 callback 무시·마지막 비어 있지 않은 Raw 선택. |
| P0 | [test_interrupt_resumes_next_penalty_and_failed_cycle_restarts](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:65) | 중단 시 다음 시도부터 재개하고 완전히 실패한 cycle을 다시 시작. |
| P0 | [test_publish_failure_replays_durable_response_and_force_starts_fresh](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:87) | 최종 저장 실패 시 durable 응답을 재사용하고 force 실행 가능 여부 확인. |
| P0 | [test_changed_input_does_not_use_previous_raw](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:100) | 입력 변경 후 과거 Raw가 새 결과로 섞이지 않도록 검증. 기존 실행의 journal은 정리되므로 미완료 journal 사례는 별도 보강 여지. |
| P0 | [test_engine_error_does_not_become_raw_or_consume_budget](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:107) | 엔진 실행 오류를 Raw 응답이나 재시도 소진으로 오인하지 않음. |
| P1 | `test_middle_penalty_succeeds_and_stops_retrying` (제거됨) | 중간 penalty에서 성공하면 추가 추론을 멈추는 경계 조건. |
| P1 | `test_scene_pass_finishes_all_batches_before_increasing_penalty` (제거됨) | 라운드별 재시도 순서와 성공 작업 제외. |
| P1 | `test_scene_pass_resume_finishes_unstarted_scenes_before_retry` (제거됨) | 재개 시 미시작 장면을 재시도 장면보다 먼저 처리하는 순서. |
| P0 | [test_scene_stream_prepares_only_admitted_tasks_and_refills_before_completion](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:116) | 전체 장면 선처리 없이 stream을 소비하고 진행 중 요청과 후속 요청을 함께 처리. |
| P0 | [test_graph_repair_never_invents_required_fields](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:145) | JSON 복구가 필수 Graph 필드를 지어내지 않도록 검증. |
| P0 | [test_checkpoint_write_failure_preserves_last_durable_attempt](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_structured_recovery.py:155) | checkpoint 저장 실패 후 마지막으로 디스크에 저장된 시도부터 복구. |
| P1 | `test_first_inference_does_not_wait_for_all_input_hashes_or_checkpoints` (제거됨) | 비-stream batch 경로도 전체 hash·checkpoint 선처리 없이 추론 시작; stream 검사와 경로가 달라 유지. |

### tests/extraction/tests/test_visual_sampling.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_sampling_keeps_full_bins_and_clips_only_the_tail](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_visual_sampling.py:35) | 장면 분할·기본 6프레임·짧은 꼬리 구간의 timestamp 정확성. |
| P1 | `test_sampling_rejects_invalid_duration` (제거됨) | 0·음수·NaN 등 잘못된 duration 거부. |
| P0 | [test_timestamp_truncation_and_filename_agree](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_visual_sampling.py:62) | 소수 timestamp와 파일명 규칙 일치; 잘못되면 이미지 연결 실패. |
| P0 | [test_fractional_seek_json_cache_and_image_selection_agree](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/extraction/tests/test_visual_sampling.py:67) | seek·timestamp JSON·프레임 파일·evidence 선택의 값 일치. |
| P1 | `test_invalid_keyframes_are_rejected_before_extracting` (제거됨) | 잘못된 정밀도·중복·역순 timestamp를 추출 전에 거부. |

### tests/integration/test_graph_scene_scheduling.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_graph_scenes_finish_content_before_next](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/integration/test_graph_scene_scheduling.py:17) | 콘텐츠별 장면 저장·실패 기록·다음 콘텐츠 진행의 연결. |
| P0 | [test_gemini_refills_workers_and_saves_only_finished_contents](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/integration/test_graph_scene_scheduling.py:91) | Gemini worker 재공급과 콘텐츠가 끝난 시점의 저장을 실제 스레드로 검증. |
| P0 | [test_gemini_interrupt_discards_memory_and_restarts_unfinished_contents](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/integration/test_graph_scene_scheduling.py:165) | Gemini 중단 시 미완성 콘텐츠만 다시 시작하고 완료 결과 보존. |

### tests/integration/test_optional_imports.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_public_clis_import_without_optional_backend_modules](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/integration/test_optional_imports.py:9) | 선택적 모델 의존성 없이 공개 CLI를 import할 수 있는 설치 계약. |
| P1 | `test_plan_only_cli_runs_without_optional_imports_assets_or_external_processes` (제거됨) | plan-only가 모델·영상 asset·외부 프로세스 없이 동작하는 별도 사용 조건. |

### tests/integration/test_readme.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_readme_relative_links_exist` (제거됨) | README의 깨진 상대 링크 탐지. |
| P2 | `test_readme_documents_complete_stage_matrix` (제거됨) | 예제 명령 총수 17개 고정은 문서 편집에 취약. 명령 dispatch 검사를 유지하고 필요한 stage 조합 집합 검사로 통합. |
| P0 | [test_documented_commands_dispatch](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/integration/test_readme.py:12) | 문서의 실제 명령을 parsing해 각 CLI handler까지 전달하는 사용자 진입점 검증. |

### tests/preparation/test_cli.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P2 | `test_preparation_steps` (제거됨) | STEP_HANDLERS 키와 순서를 그대로 비교. 문서 명령 dispatch 및 제거된 명령 거부 테스트로 통합; 추가 명령 금지까지 필요하면 set 검사를 보존. |
| P1 | `test_plan_only_is_forwarded_only_to_prepare_cohort` (제거됨) | prepare-cohort의 plan-only 옵션 전달. |
| P1 | `test_plan_only_is_rejected_on_input_preparation` (제거됨) | prepare-input-data에 plan-only를 사용했을 때 거부. |
| P1 | `test_preparation_cli_syntax_errors_keep_exit_code_two` (제거됨) | 잘못된 CLI 문법의 종료 코드 2 검증. |
| P1 | `test_cohort_interrupt_returns_130` (제거됨) | cohort 준비 중 Ctrl-C의 종료 코드 130 전달. |
| P1 | `test_force_is_forwarded` (제거됨) | 두 준비 단계의 force 옵션 전달. |
| P1 | `test_preparation_commands_removed_from_old_clis` (제거됨) | 이동된 준비 명령을 예전 CLI에서 실행할 때 명확히 거부. |

### tests/v5/test_contracts.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_required_generation_options_and_prompt_paths](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_contracts.py:19) | 생성 명령의 필수 프롬프트·모델/입력 source와 유효 프롬프트 경로 검증. |
| P1 | `test_no_donor_option_and_no_prompt_in_preparation` (제거됨) | 준비 명령에 donor·프롬프트 옵션을 허용하지 않는 계약. |
| P1 | `test_removed_version_settings_rejected` (제거됨) | 제거한 Graph 버전 설정을 조용히 무시하지 않고 거부. |
| P0 | [test_dynamic_targets_and_custom_artifact_root](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_contracts.py:47) | Arm 선택과 사용자 지정 artifact root의 실제 경로 연결. |
| P1 | `test_versioned_targets_rejected` (제거됨) | 지원하지 않는 버전형 target 거부. |
| P1 | `test_duplicate_resolved_arms_rejected` (제거됨) | 중복 Arm 설정 거부. |
| P1 | `test_target_forwarding` (제거됨) | 선택한 target 목록이 embedding·추천·진단 handler로 전달됨. |
| P0 | [test_run_id_cannot_escape_or_claim_the_shared_frame_directory](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_contracts.py:66) | run ID를 통한 경로 탈출과 공유 asset 이름 충돌 방지. |
| P0 | [test_missing_metadata_row_keeps_catalog_scope_and_records_failure](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_contracts.py:71) | Metadata 누락 때문에 catalog를 축소하지 않고 준비 실패로 기록. |

### tests/v5/test_evaluation.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_fixed_families_partial_targets_zero_control_and_effects](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_evaluation.py:14) | 다중비교 보정 4·2·2, 부분 target, 0인 대조군, 효과 방향 검증. |
| P2 | `test_bootstrap_paired_clusters_equal_date_and_seeds` (제거됨) | test_equal_date_mean_and_paired_user_multiplicity와 같은 배열·함수·기대값. aggregation 문자열 검사를 옮겨 통합; 이름과 달리 실제 seed 평균은 검증하지 않음. |
| P0 | [test_105_combinations_real_small_training_resume_and_diagnosis](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_evaluation.py:44) | 105개 조합의 작은 실제 학습·결과 저장·재개·진단 통합 검증. 실행 비용과 별개로 핵심 회귀 테스트. |

### tests/v5/test_pipeline.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_default_flow_and_artifact_lifecycle](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:37) | 생성→요약→embedding의 기본 흐름, 재실행 재사용, 영벡터와 artifact 수명주기. |
| P0 | [test_prompt_changes_invalidate_cache_without_changing_arm](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:68) | 프롬프트 변경 시 추출·요약·embedding 캐시 무효화와 Arm 식별 유지. |
| P0 | [test_fallback_missing_only_raw_precedence_and_identical_vectors](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:99) | Gemini 파일이 없을 때만 fallback, Raw 우선, 손상 파일 오류 및 provenance 갱신. 이름과 달리 동일 벡터 여부는 직접 비교하지 않음. |
| P1 | `test_asis_directory_is_not_a_supported_input` (제거됨) | 과거 ASIS 디렉터리를 현재 표현 입력으로 받지 않는 계약. |
| P0 | [test_summary_length_correction_and_resume](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:135) | 요약 길이 경계·한 번 교정·완료 결과 재사용. |
| P0 | [test_summary_keeps_last_nonempty_raw](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:167) | 교정 실패·빈 교정 시 마지막 비어 있지 않은 Raw를 embedding까지 전달. |
| P0 | [test_tobe_ids_and_directed_relations](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:193) | Graph ID 유일성·관계 참조·속성 타입 및 요약 문단 검증. 고정 GRAPH 상수의 방향 assertion 자체는 약함. |
| P0 | [test_changed_model_settings_refresh_but_prepared_frame_bytes_are_trusted](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_pipeline.py:209) | 모델 설정·모델 파일 변경 시 캐시 갱신, 준비 이미지 신뢰 및 force 재추론. |
| P1 | `test_completed_legacy_image_hash_results_reused_without_rewriting` (제거됨) | 기존 image hash 형식 결과의 호환 재사용과 파일 불필요 수정 방지. |

### tests/v5/test_prepare_titles.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_single_prepare_command_completes_titles_without_extra_artifacts](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_prepare_titles.py:18) | 단일 준비 명령의 제목 병합→영벡터 정책→원본 보존→변경 시 embedding 갱신. |
| P0 | [test_bad_supplement_blocks_preparation_instead_of_silent_zero_vectors](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_prepare_titles.py:54) | 보완 CSV 누락·중복 오류를 영벡터로 숨기지 않고 준비 차단. |
| P1 | `test_optional_supplement_setting_validated` (제거됨) | 선택적 supplement 경로 설정의 정상·빈 값 검증. |

### tests/v5/test_recovery_lifecycle.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_completed_scene_journals_removed_and_interrupted_force_resumes](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_recovery_lifecycle.py:10) | 완료 장면 journal 정리 후에도 중단된 일반/force 실행을 정확히 재개. |
| P0 | [test_summary_correction_resume_replays_response_after_publish_failure](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_recovery_lifecycle.py:54) | Summary 교정의 최종 저장 실패 후 교정 응답을 재추론 없이 복구. |
| P0 | [test_all_empty_summary_is_failure_and_gemini_missing_can_fall_back](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_recovery_lifecycle.py:97) | 전부 빈 요약을 실패로 처리하며 Qwen과 Gemini의 실패 정책을 구분. 실제 fallback 수행은 test_fallback_missing_only_raw_precedence_and_identical_vectors가 담당. |
| P1 | `test_compact_generation_history_does_not_duplicate_response_text` (제거됨) | 생성 provenance는 보존하되 원본 응답을 중복 저장하지 않는 조건. |

### tests/v5/test_run_comparison.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_paired_run_effect_direction_interaction_and_partial_family](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_run_comparison.py:58) | Run 간 효과 방향·모델 상호작용·부분 target의 보정 유지. |
| P0 | [test_cross_run_mismatch_rejected](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_run_comparison.py:72) | 사건·catalog·seed·날짜가 다른 실험을 잘못 paired 비교하지 않음. |
| P1 | `test_same_run_and_no_graph_target_rejected` (제거됨) | 같은 Run 및 Graph가 없는 target의 비교 거부. |

### tests/v5/test_shared_frames.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_second_run_reuses_shared_timestamps_and_frames](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_shared_frames.py:14) | Run 간 공유 timestamp·이미지 재사용과 force 시 정상 이미지 보존. |
| P0 | [test_missing_frames_concurrent_writers_and_invalid_existing](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_shared_frames.py:48) | 동시 쓰기에서 누락 프레임만 한 번 생성하고 기존 파일 보호. |
| P0 | [test_failed_preparation_resumes_without_new_probe_for_completed_duration](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_shared_frames.py:81) | 일부 영상 준비 실패 후 완료 작업을 반복하지 않고 재개. |
| P0 | [test_extraction_builds_paths_without_probing_assets](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_shared_frames.py:116) | 추출 단계가 준비 asset을 재탐색하지 않고 timestamp를 콘텐츠당 한 번 읽는 현재 실행 계약. |
| P1 | `test_shared_sampling_policies_coexist` (제거됨) | 서로 다른 샘플링 정책의 공유 timestamp 공존. |
| P1 | `test_concurrent_runs_share_one_duration_probe` (제거됨) | 여러 Run 동시 준비에서 duration을 한 번만 probe하는 추가 경쟁 조건. |
| P0 | [test_preparation_rejects_changed_source_without_overwriting_shared_metadata](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/v5/test_shared_frames.py:148) | 원본 영상이 바뀌었을 때 공유 metadata를 덮어쓰지 않고 재사용 거부. |
| P1 | `test_extraction_does_not_revalidate_shared_duration_or_sampling` (제거됨) | duration 파일이 없어도 준비된 timestamp로 추출; 별도 sampling 재계산 방지. |

### tests/validation/test_cohort.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_title_csv_accepts_bom_and_splits_only_the_first_comma` (제거됨) | BOM과 제목 안의 쉼표 처리. |
| P1 | `test_title_csv_rejects_invalid_rows` (제거됨) | 중복·잘못된 ID·깨진 제목 CSV 행 거부. |
| P0 | [test_title_csv_omits_blank_titles_for_catalog_coverage_check](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_cohort.py:4) | 빈 제목을 누락 title로 취급하는 Metadata 입력 정책. |

### tests/validation/test_config_v5.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_validation_config_v5_accepts_only_the_declared_contract` (제거됨) | 유효한 v5 설정의 해석과 고정 실험 설정 확인. |
| P0 | [test_validation_config_rejects_legacy_missing_and_extra_fields](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_config_v5.py:26) | 학습 차원·batch·필수 설정의 계약 위반을 거부하여 실험 조건 이탈 방지. |

### tests/validation/test_features.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_bge_encoder_reuses_one_loaded_runtime` (제거됨) | 여러 배치·encode 호출에서 BGE를 한 번만 로드하는 성능 회귀. |

### tests/validation/test_metrics.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_metrics_and_repeated_target_history_mask](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_metrics.py:10) | 반복 target을 history에서 잘못 제거하지 않는 ranking·metric 핵심 동작. |
| P2 | `test_recommendation_metric_passes_target_row_to_ranker` (제거됨) | test_metrics_and_repeated_target_history_mask와 같은 점수·history·target의 직접 함수 호출. NDCG 정확값 assertion을 그 테스트에 통합; 이름과 달리 추천 호출부 wiring 검사는 아님. |
| P0 | [test_ranker_rejects_non_finite_scores_instead_of_reporting_rank_one](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_metrics.py:19) | NaN 점수를 1위로 보고해 평가 지표를 부풀리는 오류 방지. |
| P0 | [test_top_k_uses_the_same_index_tie_break_as_target_rank](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_metrics.py:24) | 동점의 tie-break가 rank와 top-k에서 일치하여 Arm 비교의 결정성 유지. |
| P2 | `test_paired_bootstrap_positive_ci` (제거됨) | paired_bootstrap_ci는 현재 src·benchmarks의 호출자가 없음. 미사용 helper 정리 시 함께 삭제; 현재 rolling 통계는 별도 P0 유지. |
| P2 | `test_bonferroni_alpha_and_invalid_alpha` (제거됨) | bonferroni_alpha helper는 현재 실행 경로의 호출자가 없음. helper 정리와 함께 삭제; 실제 보정 분모는 test_fixed_families_partial_targets_zero_control_and_effects로 유지. |
| P2 | `test_relative_effect_boundary` (제거됨) | 현재 실행에서 쓰지 않는 paired_relative_bootstrap_ci의 과거 상대효과 경계 검사. 해당 helper 정리와 함께 제거. |
| P2 | `test_one_sided_lower_uses_alpha_quantile_not_alpha_over_two` (제거됨) | 현재 rolling 통계에서 쓰지 않는 단측 상대효과 구간 검사. 해당 helper 정리와 함께 제거. |
| P2 | `test_relative_bootstrap_resamples_sparse_positive_control` (제거됨) | 현재 실행에서 쓰지 않는 상대 bootstrap의 양수 대조군 재추출 정책 검사. 해당 helper 정리와 함께 제거. |
| P2 | `test_relative_bootstrap_rejects_zero_full_control_mean` (제거됨) | 현재 실행에서 쓰지 않는 상대 bootstrap의 0 대조군 예외 검사. 해당 helper 정리와 함께 제거; 현재 0 대조군 정책은 v5 평가 테스트 유지. |

### tests/validation/test_model.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_all_item_towers_use_frozen_features_and_trainable_projection](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_model.py:32) | 모든 Arm의 frozen feature·학습 가능한 projection·실제 역전파 검증. |
| P1 | `test_xavier_normal_and_zero_bias_initialization` (제거됨) | 초기화 분산·bias 규약 검증; 확률적 허용 오차가 있어 핵심 학습 정확성과 구분. |
| P0 | [test_causal_right_padding_and_last_valid_user_position](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_model.py:73) | causal mask·padding·마지막 유효 위치로 미래 정보 유출 방지. |
| P0 | [test_popularity_corrected_duplicate_mask_matches_numpy_reference](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_model.py:111) | 인기 보정과 중복 negative mask를 NumPy 기준 수식과 수치 비교. |
| P0 | [test_popularity_distribution_and_non_finite_guards](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_model.py:169) | popularity 분포 및 비정상 확률·feature로 학습 결과가 오염되는 문제 방지. |

### tests/validation/test_rolling.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_all_transitions_strict_ties_and_full_history](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling.py:17) | 시간상 이전 사건만 사용하고 동률 시각·전체 history·입력 순서 처리. |
| P0 | [test_csv_retains_duplicates_and_rejects_fractional_timestamps](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling.py:29) | 중복 interaction을 보존하고 정수 밀리초 계약을 지켜 평가 데이터 의미 유지. |
| P0 | [test_utc_rolling_boundaries_and_same_day_context](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling.py:43) | UTC 기준 selection/refit/test 경계와 같은 날 이전 사건의 context 처리. |
| P0 | [test_equal_date_mean_and_paired_user_multiplicity](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling.py:58) | 날짜 균등 평균과 사용자 단위 paired 재표집을 기대값·가능한 표본값으로 검증. |
| P1 | `test_100k_bootstrap_arrays_are_bounded` (제거됨) | 10만 사용자에서 bootstrap 작업 배열의 추정 메모리 상한 확인; 실제 RSS 측정은 아님. |
| P0 | [test_evaluation_masks_history_older_than_the_ten_item_context](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling.py:73) | 모델 입력 10개보다 오래된 시청 이력도 평가 후보에서 제거. |
| P0 | [test_evaluation_keeps_parameters_and_caches_catalog](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling.py:98) | 평가 중 모델 파라미터 불변과 후보 catalog 재사용; 평가가 학습을 바꾸지 않음. |

### tests/validation/test_rolling_parallel.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_spawned_training_matches_serial_parameters_and_metrics](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling_parallel.py:61) | 실제 spawned 학습과 직렬 학습의 파라미터·metric·artifact 일치. |
| P0 | [test_worker_failure_is_reported_and_all_children_are_joined](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling_parallel.py:113) | 실제 child 실패 전파와 자식 프로세스 정리·완료 표식 미생성. |
| P0 | [test_parent_interrupt_keeps_completed_artifacts_for_resume](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling_parallel.py:124) | 학습 중 부모 중단 후 완료 artifact를 보존하고 남은 조합만 재개. |
| P0 | [test_abrupt_worker_exit_does_not_hang](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling_parallel.py:161) | worker 비정상 종료 시 무한 대기 없이 오류로 종료. |
| P0 | [test_dispatch_is_unique_and_skips_completed_work](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_rolling_parallel.py:169) | 조합 중복 실행 방지, 완료 조합 건너뛰기, force 재실행. |
| P1 | `test_gpu_assignment_respects_visible_devices` (제거됨) | 추천 GPU별 worker 배정과 GPU 없는 환경의 CPU fallback. |

### tests/validation/test_steps.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P0 | [test_representation_cache_must_match_full_catalog](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_steps.py:7) | embedding index가 전체 catalog와 정확히 일치해야 캐시 재사용. |

### tests/validation/test_title_completion.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P1 | `test_completion_fills_only_required_blank_titles_and_writes_provenance` (제거됨) | 문서에 유지된 독립 제목 보완 도구의 원본 우선·필요 title만 병합·출처 보고서 계약. |
| P1 | `test_completion_adds_required_item_missing_from_primary` (제거됨) | 독립 제목 보완 도구가 원본 CSV에 없는 required item을 추가. |
| P1 | `test_completion_refuses_unresolved_required_title_without_output` (제거됨) | 독립 도구의 strict 정책에서 미해결 제목을 거부하고 출력 미생성. |
| P1 | `test_zero_vector_policy_retains_unresolved_rows_and_reports_actual_supplements` (제거됨) | 독립 도구의 명시적 zero-vector 정책에서 빈 행과 보완 통계를 보존. |
| P1 | `test_completion_validates_required_item_contract` (제거됨) | required item의 content ID·정렬 계약 거부 조건. |
| P0 | [test_completion_refuses_to_overwrite_a_source](/home/junsu2.choi/workspace/ViewingContextPipeline/tests/validation/test_title_completion.py:25) | 독립 도구에서 출력 경로로 원본을 지정해 데이터를 덮어쓰는 사고 방지. |
| P1 | `test_completion_cli_returns_nonzero_for_unresolved_title` (제거됨) | 독립 CLI가 미해결 title을 실패 종료 코드로 전달. |

### tests/validation/test_validation_cli.py

| 우선순위 | 테스트 함수 | 판단 근거 |
| --- | --- | --- |
| P2 | `test_documented_validation_step_matrix_is_stable` (제거됨) | handler 이름·등록 순서만 고정. 문서 명령 dispatch 및 제거된 명령 거부 검사로 통합; 정확한 허용 집합이 필요하면 set assertion 보존. |
| P1 | `test_recommendation_gpu_options_are_forwarded` (제거됨) | 추천 workers-per-gpu 옵션이 handler에 전달됨. |
| P1 | `test_gpu_options_are_scoped_and_positive` (제거됨) | GPU worker 옵션을 허용하는 단계와 양수 범위 검증. |
| P1 | `test_recommendation_gpu_count_flag_removed` (제거됨) | 제거된 --gpus 옵션을 거부하는 CLI 계약. |

## 제거 후 검증

- AST 기준으로 P1·P2 97개가 제거되고 P0 86개만 남았음을 확인했다. 남은 함수의 본문·decorator는 제거 전과 동일하다.
- `python -m ruff check --config pyproject.toml src tests benchmarks`: 통과.
- `git diff --check`: 통과.
- 실행 명령: `python -m pytest -q -m 'not torch' --ignore=tests/extraction/tests/test_qwen_backend.py`
- 결과: **143 passed, 1 failed, 1 skipped, 6 deselected**. 개수는 parametrize가 확장된 실행 사례 기준이다.
- 실패는 `test_dispatch_is_unique_and_skips_completed_work`가 `run_rolling()` 내부에서 PyTorch를 요구하지만 현재 환경에 `torch`가 없어 발생했다. 이 테스트에는 기존부터 `torch` marker가 없어서 위 실행에 포함됐다. 해당 P0 함수와 제품 구현은 변경하지 않았다.
- Qwen backend 모듈은 직접 `torch`를 import하므로 실행에서 제외했다. PyTorch 의존 P0 테스트의 전체 통과 여부는 확인하지 못했다.

## 삭제 전 검토의 검증 범위와 한계

- AST로 발견한 183개 함수와 이 표를 1:1로 대조했다. fixture·helper는 개수에 포함하지 않았다.
- 분류는 테스트 본문, 관련 구현, README에 적힌 현재 계약, 저장소 내부 호출 관계를 근거로 한 검토 결과다. 테스트 통과를 보증하는 결과는 아니다.
- `python -m pytest --collect-only -q`는 **279개 사례 수집 후 1개 collection error**로 끝났다. `tests/extraction/tests/test_qwen_backend.py`가 `torch`를 직접 import하지만 현재 환경에 설치되어 있지 않다. 전체 실행 사례 수로 279를 사용하면 안 된다.
- 전체 테스트 실행, 실제 GPU 추론, 외부 API 호출은 수행하지 않았다.
- `test_fallback_missing_only_raw_precedence_and_identical_vectors`는 실제 벡터 동일성을 비교하지 않고, `test_bootstrap_paired_clusters_equal_date_and_seeds`는 seed 평균 계산 경로를 직접 검증하지 않는다. 이름의 표현을 검증 범위로 간주하지 않았다.
- `test_partial_startup_failure_disposes_first_worker`는 이미 `abort()`한 fake process/queue의 `killed`·`closed` 상태를 초기화하지 않는다. 현재 assertion만으로는 두 번째 시작 실패 때 정리가 실제 수행됐는지 구분하기 어렵다. 시나리오의 가치는 P1이나 테스트 보강이 필요하다.
