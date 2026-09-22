# 테스트 함수 우선순위 — 2026-09-22

실행 시점 작업 트리의 테스트 **198개 함수**를 본문과 관련 구현으로 재분류했다. parametrize 실행 사례 수가 아닌 함수 정의 수이다. 이전 기록은 [2026-09-15 분류](test-priorities.md)에 보존한다.

## 결과

| 분류 | 분류 수 | 제거 수 | 남은 수 |
| --- | ---: | ---: | ---: |
| P0 | 182 | 0 | 182 |
| P1 | 15 | 15 | 0 |
| P2 | 1 | 1 | 0 |
| 합계 | 198 | 16 | 182 |

- P0 함수의 본문과 decorator를 보존하고 P1·P2 함수 16개를 제거했다.
- 모든 테스트가 제거된 `tests/v5/test_diagnosis_representations.py`를 삭제했다. 남은 파일에서 불필요해진 import 9개를 정리했다.
- 제거 함수 안의 전용 중첩 helper도 함께 제거했다. 모듈 수준 helper/fixture는 남은 테스트에서 계속 사용하므로 유지했다.
- 제품 코드·설정·prompt는 수정하지 않았다. 기존 사용자 변경은 보존했다.

## 기준 및 중요한 판단

- **P0**: 핵심 실행·평가 정확성·입력/결과 무결성·실패/중단/쓰기 오류 복구. 추가 조건과 섞여 있어도 P0 검증이 있으면 함수 전체를 유지한다.
- **P1**: 추가 옵션·형식 변형·자원 상한·운영 보고 조건. 단순 경계값인지 여부보다 실패의 실제 영향을 우선한다.
- **P2**: 같은 fixture와 실행 경로의 주요 assertion을 남은 테스트가 포함하거나 구현 모양만 고정하는 중복 검증.
- `test_popularity_cache_validation_and_bounded_lifetime`는 cache 상한과 함께 NaN/Inf/0 등의 학습 확률을 거부하므로 P0이다. `test_progress_counts_only_pending_scenes`와 `test_scene_total_is_known_before_streaming_inference`도 실제 저장·부분 재개/stream 실행 검증을 포함해 P0로 유지했다.
- parser의 모호한 입력 거부·필드 보존·복구, cutoff 출력의 실패 격리, legacy artifact 무결성, zero vector 및 전체 cohort 보존은 단순 형식/호환 테스트로 낮추지 않았다.
- 진단 보고서 크기/예시 개수 테스트는 평가 수식과 별개이다. 실제 통계, architecture 근거, 손상 scene 거부 및 소형 학습 진단 P0는 유지한다.
- P2인 `test_new_run_reuses_all_embeddings_without_encoder`는 동일 nine fixture와 9개 arm 공유 재사용을 포함하는 `test_nine_arm_training_reuse_partial_diagnosis_and_run_comparison`이 대체한다. 이 P0는 torch marker가 있으므로 torch를 제외한 실행은 이 중복 범위의 검증을 대신하지 못한다.
- `test_returned_results_are_published_once`는 반환 dict 최종값만 확인한다. 실제 callback 경로의 누락/중복/잘못된 요청 거부 P0는 그대로 유지한다.

## 사용자 변경 및 동시 작업

시작 시점의 미커밋 변경과 테스트 원문을 작업 트리 외부에 보관하고, 삭제 직전 원문과도 비교했다. 검토 중 외부 작업이 설정 계약을 변경하고 `tests/v7/test_concat_arms.py`를 `tests/v4/test_concat_arms.py`로 옮겼다. 이동 및 수정된 P0 본문을 그대로 보존했다. 아래는 현재 경로를 사용한다.

새 사용자 함수 `test_scene_schema_migration_uses_three_sources`는 이미 최신인 12개 파일의 무변경 집계만 확인하므로 P1로 분류했으나, 사용자 변경 보호를 위해 먼저 삭제를 보류했다. 사용자가 “이 P1 함수도 제거”를 명시 승인한 뒤 이동된 파일에서 이 함수만 제거했다. 미결정 보류 항목은 없다.

## 함수별 분류

각 함수는 한 번씩 기재한다. 삭제된 함수도 실행 시점 경로와 이름을 남긴다. 관련 구현 링크는 각 파일 표 위에 표시한다.

### tests/extraction/tests/test_backends.py

관련 구현: [extraction/backends/gemini.py](../src/extraction/backends/gemini.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_gemini_client_has_one_http_attempt_and_empty_failure_excludes_diagnostics` | 보존 | SDK 중복 재시도를 차단하고 빈 생성 응답을 성공으로 취급하지 않는 backend 계약. |
| P0 | `test_gemini_backend_uses_images_prompt_and_operational_controls` | 보존 | Gemini 요청에 이미지·프롬프트·생성 제한이 올바르게 전달되는 핵심 연결. |
| P0 | `test_gemini_backend_retries_resource_exhausted_once_after_30_seconds` | 보존 | 429 실패를 한 번 재시도해 실제 응답으로 복구하는 경로와 호출 횟수 검증. |
| P0 | `test_gemini_backend_does_not_retry_beyond_policy` | 보존 | 재시도 소진 및 비대상 오류를 전파하여 실패 은폐·무제한 재시도를 방지. |

### tests/extraction/tests/test_data_preparation.py

관련 구현: [extraction/data_preparation/fixed30.py](../src/extraction/data_preparation/fixed30.py), [extraction/data_preparation/microlens.py](../src/extraction/data_preparation/microlens.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_verified_keyframe_cache_rejects_corrupt_or_wrong_size_images` | 보존 | 손상·잘못된 크기의 공유 이미지를 정상 캐시로 재사용하지 않도록 검증. |
| P0 | `test_prepare_visual_item_reuses_catalog_duration_and_uses_shared_root` | 보존 | 준비 단계의 duration·timestamp·공유 프레임 경로 연결. |
| P0 | `test_prepare_catalog_processes_exact_cohort` | 보존 | 지정 cohort만 준비하고 이전 실패 보고서를 정리하는 기본 동작. |

### tests/extraction/tests/test_gemini_workers.py

관련 구현: [extraction/backends/gemini_workers.py](../src/extraction/backends/gemini_workers.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_gemini_pool_completes_out_of_order_and_captures_errors` | 보존 | Gemini 병렬 작업의 완료 순서·결과 대응과 개별 API 실패 격리. |
| P0 | `test_gemini_pool_propagates_ctrl_c_without_waiting_for_http_call` | 보존 | 응답이 멈춘 API 호출 중에도 Ctrl-C를 전달하는 종료 경로. |
| P1 | `test_gemini_pool_bounds_input_read_ahead_and_reuses_workers` | 제거 | GeminiWorkerPool.generate의 선행 입력 4개 상한·worker 재사용 수 검증. 결과 대응과 실패 격리는 별도 P0가 검사한다. |

### tests/extraction/tests/test_monitoring.py

관련 구현: [extraction/summary_executor.py](../src/extraction/summary_executor.py), [extraction/qwen_runtime.py](../src/extraction/qwen_runtime.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_generator_uses_visible_gpu_pool_and_passes_completed_request_metadata` | 보존 | 비동기 완료 결과에 해당 요청의 runtime metadata를 붙이고 다음 요청과 섞지 않음. |

### tests/extraction/tests/test_qwen_workers.py

관련 구현: [extraction/backends/qwen_workers.py](../src/extraction/backends/qwen_workers.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_completion_refills_fast_gpu_before_saving_without_batch_barrier` | 보존 | GPU별 비동기 완료와 task/result 대응·저장 순서가 유지되는 핵심 스케줄러 동작. |
| P1 | `test_initial_admission_is_bounded_and_runtime_events_are_delivered` | 제거 | Qwen 초기 admission 용량과 ready 알림 횟수의 추가 운영 조건. 완료 순서·오류 전파·저장 보존 P0는 유지한다. |
| P0 | `test_callback_failure_keeps_prior_saves_and_stops_all_workers` | 보존 | 저장 실패/중단 시 이전 결과를 보존하고 모든 worker·queue를 종료. |
| P0 | `test_engine_failure_is_fatal` | 보존 | 미등록 완료·OOM·worker 소멸을 성공으로 처리하거나 무한 대기하지 않고 실패 전파. |
| P1 | `test_all_visible_devices_preserved` | 제거 | CUDA_VISIBLE_DEVICES의 UUID·MIG·음수 구분자 등 장치 선택 옵션 조합 검증. |
| P0 | `test_worker_reuses_one_engine_and_passes_requests_concurrently` | 보존 | 실제 worker 진입점의 GPU/model/request 연결, 동시에 입장해야 끝나는 요청의 진행과 종료 검증. |

### tests/extraction/tests/test_semantic_graph_json_repair.py

관련 구현: [extraction/semantic_graph/json_repair.py](../src/extraction/semantic_graph/json_repair.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_deterministic_repair_cases` | 보존 | 흔한 비정상 JSON을 정해진 규칙으로 복구하고 parse mode 기록. |
| P0 | `test_unrepairable_output_is_reported` | 보존 | 빈 응답·복구 불가 응답을 정상 Graph로 통과시키지 않음. |

### tests/extraction/tests/test_semantic_graph_text_parser.py

관련 구현: [extraction/semantic_graph/parser.py](../src/extraction/semantic_graph/parser.py), [extraction/structured_output.py](../src/extraction/structured_output.py), [extraction/scene_executor.py](../src/extraction/scene_executor.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_line_graph_preserves_fields_and_hyphenated_values` | 보존 | 정상 line graph의 entity·relation·context 및 하이픈 값의 무손실 파싱. |
| P0 | `test_repairs_only_relation_delimiters` | 보존 | 관계 구분자를 복구하면서 entity ID·속성·predicate의 의미를 보존. |
| P0 | `test_surface_repairs_preserve_every_field` | 보존 | 표면 형식 복구 후 전체 그래프가 원본 기대값과 동일함을 검증. |
| P1 | `test_explicit_empty_sections` | 제거 | 레거시 Context를 포함한 빈 섹션의 NONE/None/[] 표기 변형. 기본 빈 그래프와 잘못된 구조 거부 P0는 유지한다. |
| P0 | `test_does_not_invent_discard_or_select_ambiguous_content` | 보존 | 중복 섹션·불완전 관계·추가 내용을 조용히 버리거나 창작해 파싱하지 않음. |
| P0 | `test_no_semantic_id_validation_or_count_truncation` | 보존 | ID 중복·미등록 관계·개수 초과를 임의 수정하거나 잘라내지 않고 보존. |
| P0 | `test_complete_json_remains_supported` | 보존 | 저장된 JSON graph 및 fenced JSON을 실제 parser 진입점에서 원래 구조로 복구. |
| P0 | `test_graph_without_context` | 보존 | 현재 Context 없는 graph의 native/repaired/JSON 경로와 스키마 일치. |
| P0 | `test_v4_example_passes_scene_validation_and_summary` | 보존 | 실제 현재 scene prompt 예제가 scene 검증과 summary 입력까지 무손실 연결됨. |
| P0 | `test_contextless_graph_still_requires_complete_ordered_sections` | 보존 | 필수 섹션 누락·중복·후행 설명을 거부하여 부분 그래프를 정상으로 오인하지 않음. |
| P0 | `test_empty_contextless_graph_and_invalid_legacy_context` | 보존 | 현재 빈 그래프의 정상 표현과 잘못된 legacy context 타입 거부. |
| P0 | `test_missing_end_is_accepted_without_inventing_context` | 보존 | EOF 종결의 현재 계약을 지키면서 없는 context를 만들어 넣지 않음. |
| P0 | `test_optional_context_does_not_replace_required_sections` | 보존 | 빈 입력과 End 없는 필수 Relations 누락도 검출하며 optional Context가 누락을 대체하지 않음. |

### tests/extraction/tests/test_structured_recovery.py

관련 구현: [extraction/generation.py](../src/extraction/generation.py), [extraction/scene_executor.py](../src/extraction/scene_executor.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_streaming_once_refills_before_completion_and_never_retries` | 보존 | 지연 task 생성과 중복 callback 억제, 빈/잘못된 응답의 원문 전달 및 단일 generation 호출 검증. |
| P0 | `test_single_generation_propagates_execution_and_publication_errors` | 보존 | publication에서 발생한 중단·I/O·runtime 예외가 상위로 전파됨. 이름과 달리 backend 내부 실패를 직접 주입하지는 않음. |
| P0 | `test_single_generation_rejects_invalid_backend_results` | 보존 | 누락 결과·미등록 task·중복 요청으로 결과 대응이 깨지면 즉시 거부. |
| P1 | `test_returned_results_are_published_once` | 제거 | generate_once의 callback 없는 반환 dict 어댑터 경로. 실제 scene/summary 호출은 callback을 사용하며, 이 함수는 dict 최종값만 확인해 발행 횟수 자체는 세지 않는다. |
| P0 | `test_graph_repair_never_invents_required_fields` | 보존 | 복구 가능한 graph는 복구하고 필수 구조가 없는 출력은 원문으로 보존. |
| P0 | `test_graph_format_never_fails_generation_and_survives_summary_roundtrip` | 보존 | 비정형 graph 원문을 scene 저장·summary·진단에 보존하고 backend truncation만 명시적 실패로 구분. |

### tests/extraction/tests/test_visual_sampling.py

관련 구현: [visual_sampling.py](../src/visual_sampling.py), [extraction/evidence.py](../src/extraction/evidence.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_sampling_keeps_full_bins_and_clips_only_the_tail` | 보존 | 장면 분할·기본 6프레임·짧은 꼬리 구간의 timestamp 정확성. |
| P0 | `test_timestamp_truncation_and_filename_agree` | 보존 | 소수 timestamp와 파일명 규칙 일치; 잘못되면 이미지 연결 실패. |
| P0 | `test_fractional_seek_json_cache_and_image_selection_agree` | 보존 | seek·timestamp JSON·프레임 파일·evidence 선택의 값 일치. |

### tests/integration/test_graph_scene_scheduling.py

관련 구현: [extraction/scene_executor.py](../src/extraction/scene_executor.py), [extraction/generation.py](../src/extraction/generation.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_qwen_graph_finishes_each_penalty_pass_before_retrying_failures` | 보존 | pass가 끝난 뒤 실패만 다음 penalty로 재시도하고 성공/최종 실패의 저장을 분리. |
| P0 | `test_qwen_resume_reuses_successes_and_retries_failures_from_first_penalty` | 보존 | 중단 후 성공 scene은 재사용하고 실패 scene만 초기 penalty부터 재시도. |
| P0 | `test_gemini_refills_across_contents_and_saves_finished_contents` | 보존 | 한 콘텐츠의 대기·API 실패가 다른 콘텐츠 완료 및 저장을 막지 않음. |
| P0 | `test_gemini_interrupt_preserves_each_completed_scene` | 보존 | 중단 전에 저장한 각 scene을 보존하고 force 여부와 관계없이 미완료 요청만 재개. |
| P0 | `test_scene_total_is_known_before_streaming_inference` | 보존 | 총량 표시뿐 아니라 다음 콘텐츠 입력 생성 전에 이전 콘텐츠가 저장되고 전체 stream을 빠짐없이 실행하는지 검증하므로 P0. |
| P0 | `test_gemini_resume_preserves_out_of_order_completions_across_interrupts` | 보존 | 순서 밖 완료 결과를 연속 중단 뒤에도 보존하고 정확한 미완료 scene만 재실행. |

### tests/integration/test_optional_imports.py

관련 구현: [extraction/cli.py](../src/extraction/cli.py), [preparation/cli.py](../src/preparation/cli.py), [validation/cli.py](../src/validation/cli.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_public_clis_import_without_optional_backend_modules` | 보존 | 선택적 모델 의존성 없이 공개 CLI를 import할 수 있는 설치 계약. |

### tests/integration/test_readme.py

관련 구현: [extraction/cli.py](../src/extraction/cli.py), [preparation/cli.py](../src/preparation/cli.py), [validation/cli.py](../src/validation/cli.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_documented_commands_dispatch` | 보존 | 문서의 실제 명령을 parsing해 각 CLI handler까지 전달하는 사용자 진입점 검증. |

### tests/v4/test_concat_arms.py

관련 구현: [arm_registry.py](../src/arm_registry.py), [validation/representation_inputs.py](../src/validation/representation_inputs.py), [validation/selection.py](../src/validation/selection.py), [validation/diagnosis_statistics.py](../src/validation/diagnosis_statistics.py), [extraction/scene_storage.py](../src/extraction/scene_storage.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_registry_and_config` | 보존 | 현재 6-arm 설정과 생성 source·평가 target 구분을 검증하고 잘못된 조합을 거부. |
| P0 | `test_shared_title_free_generation` | 보존 | 세 source의 scene/summary 공유와 title-free prompt·출처를 두 모델에 대해 검증. |
| P0 | `test_composition_cases_and_zero_vectors` | 보존 | title+summary/summary만/title만/둘 다 없음의 실제 텍스트·embedding·coverage 의미 검증. |
| P0 | `test_failed_summary_fallback_and_corruption` | 보존 | 실패 summary의 title fallback과 깨진 JSON 오류를 구분. |
| P0 | `test_reject_title_conditioned_summary` | 보존 | title-free source에 title-conditioned 생성 근거가 섞인 경우 거부. |
| P0 | `test_cache_invalidation_and_run_rename` | 보존 | run 이름 변경은 cache를 재사용하고 title/summary 변경은 해당 arm만 재생성. |
| P0 | `test_missing_summary_cache_then_available` | 보존 | 누락 summary의 title-only cache 공유 후 실제 summary가 생기면 무효화. |
| P0 | `test_new_signature_separates_old_policy` | 보존 | 옛 title-conditioned 정책과 현재 title concat 정책의 cache identity를 분리. |
| P0 | `test_comparisons_and_partial_targets` | 보존 | 6-arm 비교 family·teacher reference 및 부분 분석의 보정 분모 정확성. |
| P0 | `test_historical_diagnosis_context` | 보존 | 현재 설정을 변조하지 않고 과거 manifest 계약으로 진단 문맥 복원. |
| P0 | `test_six_arm_training_diagnosis_and_shared_recommendations` | 보존 | 6-arm 실제 소형 학습·진단·부분 분석 및 다른 run에서 126개 결과 공유 재사용. |
| P0 | `test_cli_source_and_target_contract` | 보존 | 실제 CLI로 generation source와 evaluation concat target을 구분하여 실행/거부. |
| P0 | `test_v7_rejects_title_prompt_and_historical_migration` | 보존 | 함수 이름은 과거 명칭이나 현재 6-arm 정책에서 title prompt와 과거 전용 migration을 실행 전에 거부. |
| P1 | `test_scene_schema_migration_uses_three_sources` | 제거 | 현재 형식으로 생성한 파일 12개를 migration으로 스캔하여 converted=0·unchanged=12만 확인하는 추가 조건. 실제 변환·원본 보존·쓰기 실패 복구는 scene_storage P0가 검사한다. |

### tests/v5/test_contracts.py

관련 구현: [pipeline_runtime.py](../src/pipeline_runtime.py), [arm_registry.py](../src/arm_registry.py), [extraction/cli.py](../src/extraction/cli.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_required_generation_options_and_prompt_paths` | 보존 | 생성 명령의 필수 프롬프트·모델/입력 source와 유효 프롬프트 경로 검증. |
| P0 | `test_dynamic_targets_and_custom_artifact_root` | 보존 | Arm 선택과 사용자 지정 artifact root의 실제 경로 연결. |
| P0 | `test_run_id_cannot_escape_or_claim_the_shared_frame_directory` | 보존 | run ID를 통한 경로 탈출과 공유 asset 이름 충돌 방지. |
| P0 | `test_missing_metadata_row_keeps_catalog_scope_and_records_failure` | 보존 | Metadata 누락 때문에 catalog를 축소하지 않고 준비 실패로 기록. |

### tests/v5/test_diagnosis_architecture.py

관련 구현: [validation/rolling_diagnosis.py](../src/validation/rolling_diagnosis.py), [validation/recommendation_contracts.py](../src/validation/recommendation_contracts.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_historical_diagnosis_and_strict_current_resume` | 보존 | 과거 architecture의 지표는 원래 의미로 진단하되 현재 학습 재개로 잘못 사용하지 않음. |
| P0 | `test_mixed_unknown_or_missing_versions_fail` | 보존 | 서로 다른/알 수 없는 architecture 결과를 섞어 통계 계산하지 않음. |
| P0 | `test_historical_results_still_require_valid_evidence` | 보존 | 과거 결과도 embedding·checkpoint·rank·metric 근거가 손상되면 거부. |

### tests/v5/test_diagnosis_representations.py

관련 구현: [validation/diagnosis_representations.py](../src/validation/diagnosis_representations.py), [validation/rolling_diagnosis.py](../src/validation/rolling_diagnosis.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P1 | `test_report_bounds_details_and_preserves_full_evidence` | 제거 | representation_report의 예시 10개 제한·누락 개수·표시용 출처 분포·원본 경로를 검사한다. 읽기 전용 보고의 원본 불변 assertion도 있지만 학습 입력 검증·평가 수식은 실행하지 않는다. |
| P1 | `test_mixed_provenance_is_bounded_and_metadata_needs_no_summary_fields` | 제거 | 다양한 provenance 및 metadata/누락 arm에 대한 보고서 분포 길이·빈 통계 표시의 추가 조건. |
| P1 | `test_diagnose_writes_compact_v3_with_statistics` | 제거 | collect_metrics를 상수로 대체한 2만 행 진단 JSON의 크기·스키마·통계 상태 표시 검사. 실제 통계 정확성은 evaluation/rolling P0가 담당한다. |

### tests/v5/test_evaluation.py

관련 구현: [validation/rolling_diagnosis.py](../src/validation/rolling_diagnosis.py), [validation/rolling_recommendation.py](../src/validation/rolling_recommendation.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_fixed_families_partial_targets_zero_control_and_effects` | 보존 | 다중비교 보정 4·2·2, 부분 target, 0인 대조군, 효과 방향 검증. |
| P0 | `test_105_combinations_real_small_training_resume_and_diagnosis` | 보존 | 105개 조합의 작은 실제 학습·결과 저장·재개·진단 통합 검증. 실행 비용과 별개로 핵심 회귀 테스트. |

### tests/v5/test_gemini_retry_failures.py

관련 구현: [extraction/scene_executor.py](../src/extraction/scene_executor.py), [extraction/summary_executor.py](../src/extraction/summary_executor.py), [extraction/failures.py](../src/extraction/failures.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_gemini_scene_retry_preserves_other_failures_and_removes_file_after_last_success` | 보존 | 부분 실패 복구 및 재중단 중 다른 실패와 성공 기록을 보존하고 마지막 성공 때만 실패 파일 제거. |
| P0 | `test_gemini_summary_restores_legacy_failures_without_generation` | 보존 | 이름과 달리 Gemini scene→Qwen summary 경로의 legacy terminal failure 재료를 재생성 없이 복구. |

### tests/v5/test_nine_arms.py

관련 구현: [arm_registry.py](../src/arm_registry.py), [extraction/arm_migration.py](../src/extraction/arm_migration.py), [validation/representation_inputs.py](../src/validation/representation_inputs.py), [validation/rolling_diagnosis.py](../src/validation/rolling_diagnosis.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_nine_arms_share_four_scene_sources` | 보존 | 9개 arm과 4개 scene source의 생성·출처·title 격리 및 전체 cohort embedding 연결. |
| P0 | `test_partial_targets_require_no_other_outputs` | 보존 | 선택 arm만 생성하되 요약 누락에도 전체 catalog·zero vector 의미를 유지. |
| P0 | `test_schema_contract_and_model_change_isolation` | 보존 | 잘못된 scene/model/title 결합을 거부하고 모델 강제 변경 시 다른 arm 결과 보존. |
| P0 | `test_title_change_does_not_invalidate_title_free_arm` | 보존 | title 변경 시 title 사용 arm만 embedding을 갱신하여 실험 입력 격리. |
| P2 | `test_new_run_reuses_all_embeddings_without_encoder` | 제거 | 동일 nine fixture의 generate→embed→다른 run에 extraction 복사→BGE 금지→9개 공유 재사용을 test_nine_arm_training_reuse_partial_diagnosis_and_run_comparison이 포함한다. 별도 v2 캐시 디렉터리 존재 assertion은 구현 경로 고정이다. |
| P0 | `test_explicit_migration_preserves_originals_and_normalizes_raw` | 보존 | 명시적 migration의 원본 보존·failed 정규화·멱등성·모델 및 충돌 검증. |
| P0 | `test_migration_skips_unverified_title_provenance` | 보존 | title 근거 없는 원본을 title-conditioned 결과로 잘못 이관하지 않음. |
| P0 | `test_comparison_families_and_partial_dimensions` | 보존 | 9-arm 비교 family 및 부분 분석의 다중비교 보정 분모 정확성. |
| P0 | `test_cli_uses_explicit_arms` | 보존 | 실제 CLI→scene/summary/embedding의 명시적 arm 선택 연결. |
| P0 | `test_nine_arm_training_reuse_partial_diagnosis_and_run_comparison` | 보존 | 실제 소형 학습·부분 및 전체 진단·run 간 checkpoint/embedding 공유·효과 차이 0 검증. |
| P0 | `test_failure_only_wrong_provenance_is_not_treated_as_missing` | 보존 | 실패 기록만 있어도 출처가 틀리면 누락 입력으로 숨겨 평가하지 않음. |
| P0 | `test_migration_rejects_conflicting_destination_failure` | 보존 | 목적지 실패 기록 충돌 시 이관을 거부하여 덮어쓰기 방지. |
| P0 | `test_new_config_diagnoses_historical_metadata_without_reinterpreting` | 보존 | 새 설정에서도 과거 metadata 실험의 title 정책과 결과 의미를 보존. |
| P0 | `test_unstructured_graph_is_successful_summary_input` | 보존 | 비정형 scene graph가 두 종류 summary와 nonzero embedding으로 정상 전달됨. |
| P0 | `test_relocated_summary_normalization_preserves_generation_evidence` | 보존 | summary 위치/arm 정규화가 생성 원문·모델·prompt·generation identity를 바꾸지 않음. |
| P0 | `test_normalization_command_backs_up_and_is_idempotent` | 보존 | 실제 CLI가 원본 byte backup을 만들고 반복 실행에도 결과를 보존. |

### tests/v5/test_penalty_passes.py

관련 구현: [extraction/generation.py](../src/extraction/generation.py), [extraction/summary_executor.py](../src/extraction/summary_executor.py), [extraction/scene_executor.py](../src/extraction/scene_executor.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_scene_passes_retry_only_failures_and_keep_empty_failure` | 보존 | scene 실패만 penalty별 재시도하며 불완전 결과는 성공 파일에 쓰지 않고 이후 복구. |
| P0 | `test_summary_passes_retry_only_failures_and_keep_empty_failure` | 보존 | summary 실패만 penalty별 재시도하고 최종 실패 텍스트는 빈 표현으로 격리. |

### tests/v5/test_pipeline.py

관련 구현: [extraction/steps.py](../src/extraction/steps.py), [extraction/summary_validation.py](../src/extraction/summary_validation.py), [validation/representation_inputs.py](../src/validation/representation_inputs.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_default_flow_and_artifact_lifecycle` | 보존 | 생성→요약→embedding의 기본 흐름, 재실행 재사용, 영벡터와 artifact 수명주기. |
| P0 | `test_scene_prompt_changes_reuse_success_without_changing_arm` | 보존 | prompt 변경에도 기존 성공 결과와 arm 식별을 보존하고 embedding을 불필요하게 바꾸지 않음. |
| P0 | `test_missing_and_raw_are_empty_but_corruption_is_an_error` | 보존 | 누락/실패와 손상을 구별하고 cohort 고정 및 복구 후 stale 검출을 유지. |
| P0 | `test_summary_word_count_is_prompt_only_and_cached_resume` | 보존 | 단어 수 제한으로 정상 생성 결과를 실패화하지 않고 저장·재개·embedding까지 연결. |
| P0 | `test_summary_accepts_multiple_paragraphs_without_correction` | 보존 | 복수 문단 원문을 보존하며 추가 교정 생성 없이 실제 summary 저장과 embedding으로 전달. |
| P0 | `test_graph_preserves_ids_and_only_validates_json_shape` | 보존 | JSON 구조 오류를 거부하되 entity/relation ID 의미를 임의 수정하지 않음. |
| P0 | `test_graph_id_mismatches_do_not_retry_or_block_downstream` | 보존 | 두 backend의 ID 불일치 원문을 재시도 없이 scene→summary→embedding으로 보존. |
| P0 | `test_changed_model_settings_and_frame_bytes_reuse_success_until_force` | 보존 | 생성 조건과 frame 변경에도 성공 결과는 유지하고 명시적인 force에서만 재생성. |

### tests/v5/test_prepare_titles.py

관련 구현: [preparation/steps.py](../src/preparation/steps.py), [validation/complete_titles.py](../src/validation/complete_titles.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_single_prepare_command_completes_titles_without_extra_artifacts` | 보존 | 준비 CLI가 원본 title을 보존해 보완·누락 zero vector를 연결하고 보완 변경 시 embedding 갱신. |
| P0 | `test_bad_supplement_blocks_preparation_instead_of_silent_zero_vectors` | 보존 | 보완 파일 누락·중복을 빈 입력으로 숨기지 않고 준비 실패 및 blocked 상태로 기록. |

### tests/v5/test_recovery_lifecycle.py

관련 구현: [extraction/summary_executor.py](../src/extraction/summary_executor.py), [extraction/failures.py](../src/extraction/failures.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_summary_single_pass_records_first_failure_without_scene_or_correction` | 보존 | 중단 후 성공·실패를 그대로 재개하고 terminal 실패 기록 및 force 복구 정책 검증. |
| P0 | `test_summary_publish_error_requires_regeneration_without_journal` | 보존 | summary 쓰기 실패가 성공 artifact로 남지 않고 재실행 시 누락 결과 재생성. |
| P0 | `test_legacy_failures_migrate_with_raw_output_and_no_temporary_state` | 보존 | legacy 실패의 content/scene 식별 및 raw 원문을 유지하며 현재 저장 형식으로 이관. |
| P0 | `test_current_failure_format_wins_over_old_records_after_partial_migration` | 보존 | 부분 migration 뒤 오래된 실패가 최신 실패 상태를 덮어쓰지 않음. |

### tests/v5/test_run_comparison.py

관련 구현: [validation/run_comparison.py](../src/validation/run_comparison.py), [validation/rolling_diagnosis.py](../src/validation/rolling_diagnosis.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_paired_run_effect_direction_interaction_and_partial_family` | 보존 | Run 간 효과 방향·모델 상호작용·부분 target의 보정 유지. |
| P0 | `test_cross_run_mismatch_rejected` | 보존 | 사건·catalog·seed·날짜가 다른 실험을 잘못 paired 비교하지 않음. |

### tests/v5/test_scene_failures.py

관련 구현: [extraction/scene_executor.py](../src/extraction/scene_executor.py), [extraction/steps.py](../src/extraction/steps.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_scene_failures_are_retried_on_resume` | 보존 | 성공 scene을 보존하면서 명시적 실패만 재실행하여 정상 결과로 복구. |
| P0 | `test_description_cutoff_is_saved_only_as_failure` | 보존 | 두 backend의 토큰 제한 출력을 성공 description으로 저장하지 않음. |
| P0 | `test_graph_length_cutoff_logs_identity_reason_and_raw_output` | 보존 | 완전해 보이는 graph도 length 종료면 실패로 격리하고 재실행 때 다시 시도. |
| P0 | `test_graph_text_repair_and_cutoff_share_storage_and_failure_policy` | 보존 | native/복구/비정형 graph 원문 저장과 backend cutoff 실패 구분이 두 backend에서 일치. |

### tests/v5/test_scene_progress.py

관련 구현: [extraction/progress.py](../src/extraction/progress.py), [extraction/steps.py](../src/extraction/steps.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P1 | `test_retry_pass_resets_counts_elapsed_rate_and_eta` | 제거 | 재시도 pass의 진행률 카운터·초당 처리량·ETA 표시 초기화. 실제 실패 재시도·결과 복구 P0는 유지한다. |
| P1 | `test_progress_displays_step_rate_for_each_unit` | 제거 | scene/summary 단위별 진행률 문자열·속도·ETA 표시 조건. |
| P0 | `test_progress_counts_only_pending_scenes` | 보존 | 진행률뿐 아니라 부분 성공 scene 재사용·누락 재생성·force 저장 scene 식별 보존을 확인하므로 P0. |
| P1 | `test_skip_logs_do_not_redraw_progress_before_next_tick` | 제거 | skip 로그 출력 중 progress refresh 빈도 및 ticker 렌더링 횟수의 운영 조건. |

### tests/v5/test_scene_storage.py

관련 구현: [extraction/scene_storage.py](../src/extraction/scene_storage.py), [extraction/steps.py](../src/extraction/steps.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_scenes_have_explicit_indices_without_metadata` | 보존 | scene index·원문·출처를 정렬 저장하고 metadata 없이 읽기 roundtrip 유지. |
| P0 | `test_interrupted_publication_preserves_readable_previous_results` | 보존 | scene publication 실패 시 기존 파일 byte와 읽기 가능한 결과 보존. |
| P0 | `test_legacy_missing_identity_is_not_silently_guessed` | 보존 | legacy metadata 누락/변조 시 scene identity를 추측해 복원하지 않음. |
| P0 | `test_legacy_migration_deletes_metadata_only_after_successful_write` | 보존 | 이관 쓰기가 성공한 뒤에만 metadata 제거; 실패 시 원본과 보조 근거 보존. |
| P0 | `test_migration_preserves_existing_summary_and_embedding_reuse` | 보존 | scene 형식 변환 뒤 요청 내용·요약 및 embedding 재사용이 동일하고 반복 이관은 무변경. |
| P0 | `test_changed_settings_retry_only_explicit_scene_and_summary_failures` | 보존 | 설정/prompt 변경에도 성공 결과를 보존하고 명시적 실패만 재생성. |

### tests/v5/test_shared_cache.py

관련 구현: [validation/shared_cache.py](../src/validation/shared_cache.py), [validation/cache_identity.py](../src/validation/cache_identity.py), [validation/recommendation_cache.py](../src/validation/recommendation_cache.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_all_arm_reuse_and_independent_invalidation` | 보존 | run 간 모든 embedding 공유, 한 arm 실패/복구의 선택적 무효화와 cohort 보존. |
| P0 | `test_visual_keys_bind_actual_input_and_generation` | 보존 | 텍스트·prompt·모델·설정 변경이 해당 embedding cache key에 반영됨. |
| P0 | `test_unknown_visual_provenance_stays_local_and_force_does_not_overwrite` | 보존 | 검증 못한 출처는 공유하지 않고 force도 기존 공유 캐시를 덮어쓰지 않음. |
| P0 | `test_atomic_concurrent_publication_and_corruption` | 보존 | 동시 cache publication에서 파일 조합 일관성을 유지하고 손상 cache를 거부·복구. |
| P0 | `test_metadata_recommendation_reuses_checkpoint_and_metrics` | 보존 | 실제 학습 checkpoint·metric을 다른 run에 안전하게 재사용하고 양쪽 손상 시 재학습. |
| P0 | `test_summary_uses_only_successful_scenes_and_recovers_empty` | 보존 | 실패 scene 원문을 summary 입력에 섞지 않고 성공 scene 복구 후 빈 summary만 재생성. |
| P0 | `test_keys_separate_data_model_and_arm_conditions` | 보존 | data·encoder·학습 조건별 cache identity 변경 범위를 분리해 잘못된 재사용 방지. |

### tests/v5/test_shared_frames.py

관련 구현: [extraction/data_preparation/video_processor.py](../src/extraction/data_preparation/video_processor.py), [preparation/input_data.py](../src/preparation/input_data.py), [extraction/evidence.py](../src/extraction/evidence.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_second_run_reuses_shared_timestamps_and_frames` | 보존 | Run 간 공유 timestamp·이미지 재사용과 force 시 정상 이미지 보존. |
| P0 | `test_missing_frames_concurrent_writers_and_invalid_existing` | 보존 | 동시 쓰기에서 누락 프레임만 한 번 생성하고 기존 파일 보호. |
| P0 | `test_failed_preparation_resumes_without_new_probe_for_completed_duration` | 보존 | 일부 영상 준비 실패 후 완료 작업을 반복하지 않고 재개. |
| P0 | `test_extraction_builds_paths_without_probing_assets` | 보존 | 추출 단계가 준비 asset을 재탐색하지 않고 timestamp를 콘텐츠당 한 번 읽는 현재 실행 계약. |
| P0 | `test_preparation_rejects_changed_source_without_overwriting_shared_metadata` | 보존 | 원본 영상이 바뀌었을 때 공유 metadata를 덮어쓰지 않고 재사용 거부. |

### tests/v5/test_shared_preparation.py

관련 구현: [preparation/steps.py](../src/preparation/steps.py), [pipeline_runtime.py](../src/pipeline_runtime.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_new_run_consumes_preparation_without_rebuilding` | 보존 | 두 run이 동일 준비 원본을 byte/mtime 변경 없이 소비하고 생성·검증 결과는 분리. |
| P1 | `test_shared_cohort_accepts_legacy_run_identity` | 제거 | eligibility에 남아 있는 옛 run_id를 무시하는 추가 호환 조건. 공유 준비 데이터 재사용·실패 시 ready 무효화 P0는 유지한다. |
| P0 | `test_plan_only_invalidates_previous_shared_ready_state` | 보존 | plan-only 실행 후 기존 ready를 재사용하지 않고 재준비가 성공해야 복구. |
| P0 | `test_failed_shared_cohort_refresh_is_not_readable` | 보존 | 공유 준비 갱신 실패 뒤 이전 ready 데이터가 계속 정상으로 읽히는 것을 방지. |

### tests/v5/test_summary_models.py

관련 구현: [extraction/summary_executor.py](../src/extraction/summary_executor.py), [validation/representation_inputs.py](../src/validation/representation_inputs.py), [extraction/cli.py](../src/extraction/cli.py), [validation/cli.py](../src/validation/cli.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_summary_cli` | 보존 | 추출 source arm과 summary model을 분리해 필수 식별값을 handler에 전달. |
| P0 | `test_gemini_text_only_and_reuse` | 보존 | Gemini summary가 이미지/GPU 없이 scene 텍스트를 사용하고 모델 출처를 저장하며 재개 시 재생성하지 않음. |
| P0 | `test_failures_retry_and_preserve_empty_expressions` | 보존 | Gemini 오류·빈 응답·token cutoff를 빈 표현으로 격리하고 다른 모델 결과 및 실패 기록 보존 후 복구. |
| P0 | `test_models_coexist_interrupt_resume_and_embedding_selection` | 보존 | 두 summary 모델의 결과·중단 재개·force를 격리하고 선택 모델별 embedding/recommendation identity 검증. |
| P0 | `test_legacy_failure_mismatch_precedes_migration` | 보존 | 모델 불일치 검증을 legacy 실패 이관보다 먼저 수행해 기존 실패 byte 보존. |
| P0 | `test_failure_model_survives_migration` | 보존 | 실패 이관/갱신 시 summary_model 식별이 소실되지 않음. |
| P1 | `test_legacy_config_optional_and_ignored` | 제거 | 더 이상 동작에 사용하지 않는 protocol.graph_summarizer 옵션을 읽는 호환 조건. |
| P0 | `test_cli_returns_failure_for_gemini_error` | 보존 | Gemini 생성 실패가 CLI 성공으로 보고되지 않고 실패 파일에 남음. |
| P1 | `test_embedding_cli_reads_model_from_arm_artifact` | 제거 | 이름과 달리 artifact를 읽지 않고 handler를 stub으로 대체해 force 기본값 전달과 폐기된 --summary-source 옵션 거부만 확인한다. |
| P0 | `test_embedding_does_not_fall_back_to_other_summary_model` | 보존 | 선택 모델 요약이 없으면 다른 모델 결과로 대체하지 않고 zero vector 유지. |
| P0 | `test_selected_summary_provenance_and_staleness` | 보존 | 선택 모델의 변경만 stale로 검출하고 다른 모델 출처로 위장된 요약은 거부. |

### tests/v5/test_summary_resume.py

관련 구현: [extraction/summary_executor.py](../src/extraction/summary_executor.py), [extraction/failures.py](../src/extraction/failures.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_resume_after_out_of_order_results` | 보존 | 순서 밖 summary 완료 후 중단 위치별 재개에서 중복 요청 없이 penalty·terminal 원문 보존. |
| P0 | `test_mixed_legacy_terminal_and_pending_failures` | 보존 | legacy/소진/재시도 가능 실패를 구분해 정확한 penalty로 재개하고 force 때 전체 복구. |
| P0 | `test_terminal_write_failure_recovers_without_regeneration` | 보존 | terminal 결과 쓰기 실패 후 persisted failure에서 복구하여 이미 생성한 요청을 반복하지 않음. |
| P0 | `test_failure_penalty_survives_migration_and_update` | 보존 | 실패 이관·갱신에 저장된 penalty를 유지해 재개 위치를 잃지 않음. |
| P0 | `test_invalid_scene_input_reports_error` | 보존 | 손상 scene은 오류로, 누락/빈 scene은 빈 failed summary로 구분해 가짜 성공 방지. |

### tests/v5/test_summary_titles.py

관련 구현: [extraction/summary_prompt.py](../src/extraction/summary_prompt.py), [extraction/summary_executor.py](../src/extraction/summary_executor.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_title_rendering_preserves_input_literals` | 보존 | title/scene 안의 placeholder·인용부호·개행을 재해석하지 않아 실제 입력 원문 보존. |
| P0 | `test_titles_reach_every_summary_and_input_hash` | 보존 | 행 순서가 달라도 content ID로 title을 연결하고 title 변경을 생성 input hash에 반영. |
| P0 | `test_legacy_terminal_failure_stays_empty_without_invented_provenance` | 보존 | 재시도 소진 실패를 복원하면서 새 prompt 근거를 창작하지 않고 빈 텍스트 유지. |

### tests/v5/test_validation_selection.py

관련 구현: [validation/selection.py](../src/validation/selection.py), [validation/representation_inputs.py](../src/validation/representation_inputs.py), [validation/rolling_workers.py](../src/validation/rolling_workers.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_full_catalog_applies_to_all_targets_and_preserves_shared_data` | 보존 | 누락 요약에도 모든 arm의 전체 catalog·event/history·zero vector와 공유 준비 원본 유지. |
| P0 | `test_empty_representations_and_no_cross_arm_fallback` | 보존 | 누락/실패 표현은 비우되 손상/잘못된 모델 출처는 거부하고 다른 arm으로 대체하지 않음. |
| P0 | `test_recovery_deletion_source_changes_and_partial_cache_invalidation` | 보존 | 요약 복구·삭제 및 모델 선택 변경에서 cohort는 고정하고 영향받는 cache만 무효화. |
| P0 | `test_missing_manifest_metadata_only_and_outside_target` | 보존 | manifest 없는 평가를 막고 설정 밖 target을 거부하며 metadata-only 평가의 전체 cohort 유지. |
| P0 | `test_fixed_dates_and_history_survive_generation_failures` | 보존 | 생성 실패와 무관하게 test 날짜 경계·대상 event 수·과거 history 유지. |
| P0 | `test_all_missing_summaries_preserve_the_full_cohort` | 보존 | 전부 요약이 없더라도 표본을 버리지 않고 전체 cohort의 zero vector를 생성. |
| P0 | `test_worker_uses_same_full_table_and_rejects_changed_selection` | 보존 | worker가 동일 cohort/phase ID를 사용하고 변경된 selection signature 작업은 거부. |
| P0 | `test_diagnosis_reports_full_selection_and_rejects_corrupt_scenes` | 보존 | 요약 누락으로 손상 scene을 감추지 않고 진단 실패와 근거를 기록. |
| P0 | `test_tampered_or_interrupted_selection_is_not_readable` | 보존 | selection 변조 및 manifest 발행 중단 후 혼합된 cohort를 읽지 못하게 차단. |

### tests/validation/test_cohort.py

관련 구현: [validation/cohort.py](../src/validation/cohort.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_title_csv_omits_blank_titles_for_catalog_coverage_check` | 보존 | 빈 제목을 누락 title로 취급하는 Metadata 입력 정책. |

### tests/validation/test_config_v5.py

관련 구현: [validation/config.py](../src/validation/config.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_validation_config_rejects_legacy_missing_and_extra_fields` | 보존 | 학습 차원·batch·필수 설정의 계약 위반을 거부하여 실험 조건 이탈 방지. |

### tests/validation/test_metrics.py

관련 구현: [validation/metrics.py](../src/validation/metrics.py), [validation/scoring.py](../src/validation/scoring.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_metrics_and_repeated_target_history_mask` | 보존 | 반복 target을 history에서 잘못 제거하지 않는 ranking·metric 핵심 동작. |
| P0 | `test_ranker_rejects_non_finite_scores_instead_of_reporting_rank_one` | 보존 | NaN 점수를 1위로 보고해 평가 지표를 부풀리는 오류 방지. |
| P0 | `test_top_k_uses_the_same_index_tie_break_as_target_rank` | 보존 | 동점의 tie-break가 rank와 top-k에서 일치하여 Arm 비교의 결정성 유지. |

### tests/validation/test_model.py

관련 구현: [validation/model.py](../src/validation/model.py), [validation/recommendation.py](../src/validation/recommendation.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_all_item_towers_use_frozen_features_and_trainable_projection` | 보존 | 모든 Arm의 frozen feature·학습 가능한 projection·실제 역전파 검증. |
| P0 | `test_causal_right_padding_and_last_valid_user_position` | 보존 | causal mask·padding·마지막 유효 위치로 미래 정보 유출 방지. |
| P0 | `test_popularity_corrected_duplicate_mask_matches_numpy_reference` | 보존 | 인기 보정과 중복 negative mask를 NumPy 기준 수식과 수치 비교. |
| P0 | `test_popularity_distribution_and_non_finite_guards` | 보존 | popularity 분포 및 비정상 확률·feature로 학습 결과가 오염되는 문제 방지. |

### tests/validation/test_rolling.py

관련 구현: [validation/rolling_data.py](../src/validation/rolling_data.py), [validation/rolling_diagnosis.py](../src/validation/rolling_diagnosis.py), [validation/rolling_recommendation.py](../src/validation/rolling_recommendation.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_all_transitions_strict_ties_and_full_history` | 보존 | 시간상 이전 사건만 사용하고 동률 시각·전체 history·입력 순서 처리. |
| P0 | `test_csv_retains_duplicates_and_rejects_fractional_timestamps` | 보존 | 중복 interaction을 보존하고 정수 밀리초 계약을 지켜 평가 데이터 의미 유지. |
| P0 | `test_utc_rolling_boundaries_and_same_day_context` | 보존 | UTC 기준 selection/refit/test 경계와 같은 날 이전 사건의 context 처리. |
| P0 | `test_equal_date_mean_and_paired_user_multiplicity` | 보존 | 날짜 균등 평균과 사용자 단위 paired 재표집을 기대값·가능한 표본값으로 검증. |
| P0 | `test_evaluation_masks_history_older_than_the_ten_item_context` | 보존 | 모델 입력 10개보다 오래된 시청 이력도 평가 후보에서 제거. |
| P0 | `test_evaluation_keeps_parameters_and_caches_catalog` | 보존 | 평가 중 모델 파라미터 불변과 후보 catalog 재사용; 평가가 학습을 바꾸지 않음. |

### tests/validation/test_rolling_execution.py

관련 구현: [validation/rolling_execution.py](../src/validation/rolling_execution.py), [validation/rolling_recommendation.py](../src/validation/rolling_recommendation.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_prepared_sequences_and_vector_mask_exact` | 보존 | 최적화된 sequence 및 negative mask가 기준 history 처리와 정확히 동일. |
| P0 | `test_popularity_cache_validation_and_bounded_lifetime` | 보존 | cache 크기뿐 아니라 0·NaN·Inf·음수·underflow popularity를 거부하므로 학습 정확성 P0. |
| P0 | `test_rank_parity_full_history_ties_repeated_targets` | 보존 | 최적화 ranking이 full history·동점·반복 target에서 기준 구현과 같고 비유한 점수 거부. |
| P0 | `test_loss_gradient_and_multiple_updates` | 보존 | 기준 구현과 최적화 구현의 loss 및 실제 gradient 수치 일치. |
| P0 | `test_training_epoch_and_validation_parity` | 보존 | 실제 epoch 학습 결과·event별 rank/metric·validation NDCG가 기준 구현과 일치. |
| P0 | `test_nonfinite_loss_prevents_optimizer_update` | 보존 | NaN loss로 optimizer가 파라미터를 오염시키기 전에 중단. |
| P0 | `test_selection_refit_test_and_old_bundle_compatibility` | 보존 | selection/refit/test의 기준 구현 동등성과 과거 bundle의 손상 없는 재개·공유 재사용. |

### tests/validation/test_rolling_parallel.py

관련 구현: [validation/rolling_workers.py](../src/validation/rolling_workers.py), [validation/rolling_recommendation.py](../src/validation/rolling_recommendation.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_spawned_training_matches_serial_parameters_and_metrics` | 보존 | 실제 spawned 학습과 직렬 학습의 파라미터·metric·artifact 일치. |
| P0 | `test_worker_failure_is_reported_and_all_children_are_joined` | 보존 | 실제 child 실패 전파와 자식 프로세스 정리·완료 표식 미생성. |
| P0 | `test_parent_interrupt_keeps_completed_artifacts_for_resume` | 보존 | 학습 중 부모 중단 후 완료 artifact를 보존하고 남은 조합만 재개. |
| P0 | `test_abrupt_worker_exit_does_not_hang` | 보존 | worker 비정상 종료 시 무한 대기 없이 오류로 종료. |
| P0 | `test_dispatch_is_unique_and_skips_completed_work` | 보존 | 조합 중복 실행 방지, 완료 조합 건너뛰기, force 재실행. |

### tests/validation/test_steps.py

관련 구현: [validation/steps.py](../src/validation/steps.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_representation_cache_must_match_full_catalog` | 보존 | embedding index가 전체 catalog와 정확히 일치해야 캐시 재사용. |

### tests/validation/test_title_completion.py

관련 구현: [validation/complete_titles.py](../src/validation/complete_titles.py).

| 우선순위 | 함수 | 처리 | 실제 검증과 판단 근거 |
| --- | --- | --- | --- |
| P0 | `test_completion_refuses_to_overwrite_a_source` | 보존 | 독립 도구에서 출력 경로로 원본을 지정해 데이터를 덮어쓰는 사고 방지. |

## 검증 결과

검증 진행 중. 완료 결과로 갱신한다.
