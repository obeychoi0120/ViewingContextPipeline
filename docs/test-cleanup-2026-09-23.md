# 핵심 테스트 정리 — 2026-09-23

사용자가 선택한 기준은 **핵심 시나리오만 유지, 과거 호환성은 대표 검증만 유지**입니다. 이전 기록의 P0 분류를 삭제 금지 조건으로 재사용하지 않았습니다. 아래 대체 검증은 핵심 동작을 보존한다는 뜻이며, 삭제된 세부 검증을 모두 포함한다는 뜻은 아닙니다.

## 결과

| 구분 | 이전 | 정리 후 | 감소 |
| --- | ---: | ---: | ---: |
| 테스트 파일 (`test_*.py`) | 42 | 38 | 4 |
| 테스트 함수 정의 | 184 | 111 | 73 (39.7%) |
| pytest 수집 사례 | 468 | 208 | 260 (55.6%) |

기존 이름 77개를 삭제하고 현재 계약에 맞춘 대체 함수 이름 4개를 추가했습니다. 따라서 함수 순감소는 73개입니다. helper/fixture는 함수 수에 포함하지 않습니다. 실행 사례는 이 환경의 기본 CPU 수집 기준이며 `ROLLING_TEST_CUDA=1`을 지정하면 별도 CUDA 사례가 추가됩니다.

## 구조 및 유지한 범위

- `tests/conftest.py`의 `current_context`는 저장소의 현재 v4/6-arm 설정을 사용합니다. `ready_context`는 실제 준비 단계를 작은 임시 데이터와 가짜 ffmpeg로 수행합니다.
- `generate_all` callable fixture는 현재 3개 scene/summary source를 생성합니다. 테스트 모듈에서 다른 테스트 모듈의 준비 함수를 import하지 않습니다.
- 남긴 일반 pipeline 테스트는 `tests/v4/`로 이동했습니다. 과거 v5/v6 설정은 `tests/compatibility/conftest.py`의 `legacy_ready_context`에서만 구성합니다.
- 현재 6-arm의 생성·임베딩·작은 CPU 학습·진단·공유 결과 재사용, 입력/결과 무결성, backend 요청과 오류 전파, 대표 중단·쓰기 실패·재시도를 유지합니다.
- 시간 분할, full history masking, NDCG/동점, bootstrap/다중비교, causal masking, loss/gradient 기준 구현 비교, NaN 업데이트 차단, 실제 spawned worker와 직렬 결과 일치를 유지합니다. 수치 비교용 `rolling_reference.py` 원문은 변경하지 않았습니다.
- 제품 코드·공개 API·config·prompt는 변경하지 않았습니다. 실제 모델 다운로드나 외부 모델 API 호출은 하지 않습니다.

## 계획에서 조정한 seed 범위

제품의 `validation.config.ModelConfig`는 서로 다른 seed **정확히 3개**를 요구합니다. 이 계약을 우회하거나 제품 코드를 바꾸지 않기 위해 실제 설정을 읽는 6-arm 통합 실행은 42/43/44와 7개 날짜, 총 **126개 조합**을 유지합니다. 독립적인 reference/optimized 학습 비교의 arm×seed 9개 조합은 현재 `graph_qwen_meta`/42 한 사례로 줄였습니다. 병렬 job 비교는 42/43을 사용합니다.

## 실행 검증

| 검증 | 결과 |
| --- | --- |
| 수정 전 전체 pytest | 467 passed, 1 failed (488.43초) |
| 수정 후 `python -m pytest --collect-only -q` | 208 cases |
| 수정 후 전체 pytest | **208 passed** (45.46초), CPU 소형 학습·실제 spawned worker 포함 |
| `python -m ruff check tests` | All checks passed |
| `git diff --check` | 통과 |

전체 실행 명령은 전후 모두 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`입니다. 최초 기본 스레드 실행은 219개 통과 후 장시간 실행 때문에 중단했으며, 위 수정 전 결과는 이후 완료한 전체 실행입니다. 수정 전 일부 테스트가 보이는 GPU를 자동 선택했고, 수정 후 대표 통합 학습은 CPU를 명시하므로 시간 차이를 동일 조건의 성능 개선 비율로 해석하지 않습니다. 별도 `ROLLING_TEST_CUDA=1` 행렬은 실행하지 않았습니다.

수정 전 실패는 `tests/v5/test_shared_cache.py::test_metadata_recommendation_reuses_checkpoint_and_metrics`의 checkpoint 가중치 비교입니다. `torch.equal`이 GPU 원본과 CPU 재사용 텐서를 비교하면서 `Expected all tensors to be on the same device`를 발생시켰습니다. 이 테스트는 계획상 제거 대상이었지만 **가중치 보존 assertion은 삭제하지 않고** 현재 `test_six_arm_training_diagnosis_and_shared_recommendations`로 옮겼습니다. 126개 checkpoint의 키·모든 텐서 값·원본 run 출처를 비교하며 양쪽 `torch.load`에 `map_location="cpu"`를 명시합니다. 제품의 캐시 동작을 수정한 것은 아닙니다.

## 삭제·대체 함수 기록

경로는 정리 전 경로입니다. 링크는 정리 후 유지한 테스트를 가리킵니다.

### `tests/extraction/tests/test_semantic_graph_text_parser.py` — 6개

정상 텍스트·대표 구분자 복구·모호한 입력·필수 section 누락·현재 prompt 예제와 JSON 입력은 유지합니다. BOM/개행/헤더/공백의 표기 변형, End 생략과 legacy context 및 ID/개수 세부 조건은 제외합니다.

남긴 검증: [`test_line_graph_preserves_fields_and_hyphenated_values`](../tests/extraction/tests/test_semantic_graph_text_parser.py), [`test_repairs_only_relation_delimiters`](../tests/extraction/tests/test_semantic_graph_text_parser.py), [`test_does_not_invent_discard_or_select_ambiguous_content`](../tests/extraction/tests/test_semantic_graph_text_parser.py), [`test_graph_without_context`](../tests/extraction/tests/test_semantic_graph_text_parser.py), [`test_optional_context_does_not_replace_required_sections`](../tests/extraction/tests/test_semantic_graph_text_parser.py).

- `test_complete_json_remains_supported`
- `test_contextless_graph_still_requires_complete_ordered_sections`
- `test_empty_contextless_graph_and_invalid_legacy_context`
- `test_missing_end_is_accepted_without_inventing_context`
- `test_no_semantic_id_validation_or_count_truncation`
- `test_surface_repairs_preserve_every_field`

### `tests/integration/test_graph_scene_scheduling.py` — 3개

Qwen 재개와 Gemini 순서 밖 완료·재개가 대표 복구를 담당합니다. 진행률 선계산·로그 표시와 별도 penalty 스케줄 조합은 축소했습니다.

남긴 검증: [`test_qwen_resume_reuses_successes_and_retries_failures_from_first_penalty`](../tests/integration/test_graph_scene_scheduling.py), [`test_gemini_resume_preserves_out_of_order_completions_across_interrupts`](../tests/integration/test_graph_scene_scheduling.py), [`test_scene_passes_retry_only_failures_and_keep_empty_failure`](../tests/v4/test_penalty_passes.py).

- `test_gemini_interrupt_preserves_each_completed_scene`
- `test_qwen_graph_finishes_each_penalty_pass_before_retrying_failures`
- `test_scene_total_is_known_before_streaming_inference`

### `tests/v4/test_concat_arms.py` — 3개

현재 제목 결합·실패 표현·캐시 변경과 입력 계약은 남겼습니다. 과거 정책 해시 비교, 누락→복구의 추가 조합, 과거 마이그레이션 거부 옵션 조합은 제외했습니다.

남긴 검증: [`test_cache_invalidation_and_run_rename`](../tests/v4/test_concat_arms.py), [`test_registry_and_config`](../tests/v4/test_concat_arms.py), [`test_reject_title_conditioned_summary`](../tests/v4/test_concat_arms.py).

- `test_missing_summary_cache_then_available`
- `test_new_signature_separates_old_policy`
- `test_v7_rejects_title_prompt_and_historical_migration`

### `tests/v5/test_contracts.py` — 2개

현재 명령의 dispatch와 성공/실패를 유지합니다. 과거 dynamic arm/custom root 및 명령×모델 옵션 오류 전체 곱은 제외합니다.

남긴 검증: [`test_documented_commands_dispatch`](../tests/integration/test_readme.py), [`test_cli_source_and_target_contract`](../tests/v4/test_concat_arms.py), [`test_registry_and_config`](../tests/v4/test_concat_arms.py).

- `test_dynamic_targets_and_custom_artifact_root`
- `test_required_generation_options_and_prompt_paths`

### `tests/v5/test_diagnosis_architecture.py` — 2개

과거 결과를 읽으면서 원문을 변경하지 않는 대표 시나리오만 유지합니다. 과거 architecture 버전별·손상 위치별 거부 행렬은 제외합니다.

남긴 검증: [`test_historical_diagnosis_and_strict_current_resume`](../tests/compatibility/test_diagnosis_architecture.py), [`test_historical_diagnosis_context`](../tests/compatibility/test_diagnosis_architecture.py).

- `test_historical_results_still_require_valid_evidence`
- `test_mixed_unknown_or_missing_versions_fail`

### `tests/v5/test_evaluation.py` — 2개

과거 5-arm 105개 조합 학습을 제거합니다. 현재 6-arm 학습·진단을 유지하고, 효과 방향·0인 대조군·다중비교 보정의 수치 assertion을 현재 통계 테스트로 옮겼습니다.

남긴 검증: [`test_six_arm_training_diagnosis_and_shared_recommendations`](../tests/v4/test_concat_arms.py), [`test_comparisons_and_partial_targets`](../tests/v4/test_concat_arms.py).

- `test_105_combinations_real_small_training_resume_and_diagnosis`
- `test_fixed_families_partial_targets_zero_control_and_effects`

### `tests/v5/test_gemini_retry_failures.py` — 1개

현재 Gemini 실패 복구는 유지하고 과거 summary 실패 파일만으로 결과를 복원하는 경로는 제외합니다.

남긴 검증: [`test_gemini_scene_retry_preserves_other_failures_and_removes_file_after_last_success`](../tests/v4/test_gemini_retry_failures.py), [`test_failures_retry_and_preserve_empty_expressions`](../tests/v4/test_summary_models.py).

- `test_gemini_summary_restores_legacy_failures_without_generation`

### `tests/v5/test_nine_arms.py` — 12개

9-arm 생성/전체 학습·과거 제목 조건·부분 선택·세부 provenance 이관은 제외합니다. 원본 보존·충돌 거부·백업 및 멱등성은 유지합니다. 비정형 graph 후속 요약은 현재 graph_qwen으로 옮겼습니다.

남긴 검증: [`test_explicit_migration_preserves_originals_and_normalizes_raw`](../tests/compatibility/test_nine_arms.py), [`test_migration_rejects_conflicting_destination_failure`](../tests/compatibility/test_nine_arms.py), [`test_normalization_command_backs_up_and_is_idempotent`](../tests/compatibility/test_nine_arms.py), [`test_unstructured_graph_reaches_summary_and_embedding`](../tests/v4/test_concat_arms.py).

- `test_cli_uses_explicit_arms`
- `test_comparison_families_and_partial_dimensions`
- `test_failure_only_wrong_provenance_is_not_treated_as_missing`
- `test_migration_skips_unverified_title_provenance`
- `test_new_config_diagnoses_historical_metadata_without_reinterpreting`
- `test_nine_arm_training_reuse_partial_diagnosis_and_run_comparison`
- `test_nine_arms_share_four_scene_sources`
- `test_partial_targets_require_no_other_outputs`
- `test_relocated_summary_normalization_preserves_generation_evidence`
- `test_schema_contract_and_model_change_isolation`
- `test_title_change_does_not_invalidate_title_free_arm`
- `test_unstructured_graph_is_successful_summary_input`

### `tests/v5/test_pipeline.py` — 8개

과거 5-arm 전체 흐름을 현재 6-arm 흐름으로 대체합니다. 성공 결과 재사용·실패만 재시도는 남기고, 단어 수·문단·ID 변형·원본 frame byte/설정 조합은 제외합니다.

남긴 검증: [`test_six_arm_training_diagnosis_and_shared_recommendations`](../tests/v4/test_concat_arms.py), [`test_changed_settings_retry_only_explicit_scene_and_summary_failures`](../tests/v4/test_scene_storage.py), [`test_missing_summary_does_not_fall_back_to_another_arm`](../tests/v4/test_validation_selection.py).

- `test_changed_model_settings_and_frame_bytes_reuse_success_until_force`
- `test_default_flow_and_artifact_lifecycle`
- `test_graph_id_mismatches_do_not_retry_or_block_downstream`
- `test_graph_preserves_ids_and_only_validates_json_shape`
- `test_missing_and_raw_are_empty_but_corruption_is_an_error`
- `test_scene_prompt_changes_reuse_success_without_changing_arm`
- `test_summary_accepts_multiple_paragraphs_without_correction`
- `test_summary_word_count_is_prompt_only_and_cached_resume`

### `tests/v5/test_recovery_lifecycle.py` — 3개

현재 저장 실패와 정상 실패/재시도는 남깁니다. 과거 failure 포맷 우선순위·이관과 단일-pass 진행률·출력 형태 조합은 제외합니다.

남긴 검증: [`test_summary_publish_error_requires_regeneration_without_journal`](../tests/v4/test_recovery_lifecycle.py), [`test_summary_passes_retry_only_failures_and_keep_empty_failure`](../tests/v4/test_penalty_passes.py), [`test_failures_retry_and_preserve_empty_expressions`](../tests/v4/test_summary_models.py).

- `test_current_failure_format_wins_over_old_records_after_partial_migration`
- `test_legacy_failures_migrate_with_raw_output_and_no_temporary_state`
- `test_summary_single_pass_records_first_failure_without_scene_or_correction`

### `tests/v5/test_run_comparison.py` — 1개

같은 함수를 현재 3개 graph arm의 run 비교로 대체·개명했습니다. 효과 방향·신뢰구간·부분 선택 보정은 유지하고 과거 모델 interaction 기대값은 제외합니다.

남긴 검증: [`test_paired_run_effect_direction_and_partial_family`](../tests/v4/test_run_comparison.py).

- `test_paired_run_effect_direction_interaction_and_partial_family`

### `tests/v5/test_scene_failures.py` — 2개

출력 잘림은 결과 형식 처리 테스트에서, 실패만 재시도하는 동작은 backend별 대표 복구에서 유지합니다. 별도 로그·force 조합은 축소합니다.

남긴 검증: [`test_graph_text_repair_and_cutoff_share_storage_and_failure_policy`](../tests/v4/test_scene_failures.py), [`test_gemini_scene_retry_preserves_other_failures_and_removes_file_after_last_success`](../tests/v4/test_gemini_retry_failures.py), [`test_scene_passes_retry_only_failures_and_keep_empty_failure`](../tests/v4/test_penalty_passes.py).

- `test_graph_length_cutoff_logs_identity_reason_and_raw_output`
- `test_scene_failures_are_retried_on_resume`

### `tests/v5/test_scene_progress.py` — 1개

ETA·진행률·표시 형식은 의도적으로 제외합니다. 이 함수에 섞여 있던 부분 재개와 성공 결과 보존은 복구 테스트에 남겼습니다.

남긴 검증: [`test_qwen_resume_reuses_successes_and_retries_failures_from_first_penalty`](../tests/integration/test_graph_scene_scheduling.py), [`test_changed_settings_retry_only_explicit_scene_and_summary_failures`](../tests/v4/test_scene_storage.py).

- `test_progress_counts_only_pending_scenes`

### `tests/v5/test_scene_storage.py` — 3개

현재 결과 저장 실패 시 원본 보존은 유지합니다. 과거 sidecar 식별 복원과 저장 형식 이관 단계별 장애 검증은 포기합니다. 대표 arm 마이그레이션 원본 보존은 호환 테스트에서 유지합니다.

남긴 검증: [`test_interrupted_publication_preserves_readable_previous_results`](../tests/v4/test_scene_storage.py), [`test_explicit_migration_preserves_originals_and_normalizes_raw`](../tests/compatibility/test_nine_arms.py).

- `test_legacy_migration_deletes_metadata_only_after_successful_write`
- `test_legacy_missing_identity_is_not_silently_guessed`
- `test_migration_preserves_existing_summary_and_embedding_reuse`

### `tests/v5/test_shared_cache.py` — 6개

현재 arm의 제목/summary 변경 시 재생성 범위, run 간 임베딩/학습 재사용, 원자적 저장과 손상 거부를 유지합니다. 키 구성 필드별 변형과 출처 미상 캐시의 세부 정책은 제외합니다.

남긴 검증: [`test_cache_invalidation_and_run_rename`](../tests/v4/test_concat_arms.py), [`test_six_arm_training_diagnosis_and_shared_recommendations`](../tests/v4/test_concat_arms.py), [`test_atomic_concurrent_publication_and_corruption`](../tests/v4/test_shared_cache.py).

- `test_all_arm_reuse_and_independent_invalidation`
- `test_keys_separate_data_model_and_arm_conditions`
- `test_metadata_recommendation_reuses_checkpoint_and_metrics`
- `test_summary_uses_only_successful_scenes_and_recovers_empty`
- `test_unknown_visual_provenance_stays_local_and_force_does_not_overwrite`
- `test_visual_keys_bind_actual_input_and_generation`

### `tests/v5/test_shared_frames.py` — 2개

실제 프레임 재사용·동시 저장·손상 거부·부분 준비 재개는 유지합니다. 개별 파일 탐색 호출 금지와 준비 후 원본 영상 교체 조합은 제외합니다.

남긴 검증: [`test_second_run_reuses_shared_timestamps_and_frames`](../tests/v4/test_shared_frames.py), [`test_missing_frames_concurrent_writers_and_invalid_existing`](../tests/v4/test_shared_frames.py), [`test_failed_preparation_resumes_without_new_probe_for_completed_duration`](../tests/v4/test_shared_frames.py).

- `test_extraction_builds_paths_without_probing_assets`
- `test_preparation_rejects_changed_source_without_overwriting_shared_metadata`

### `tests/v5/test_shared_preparation.py` — 2개

공유 프레임 재사용과 준비 실패 상태 거부는 유지합니다. plan-only 뒤 재준비와 별도 두-run 전체 생성 조합은 제외합니다.

남긴 검증: [`test_second_run_reuses_shared_timestamps_and_frames`](../tests/v4/test_shared_frames.py), [`test_failed_shared_cohort_refresh_is_not_readable`](../tests/v4/test_shared_preparation.py).

- `test_new_run_consumes_preparation_without_rebuilding`
- `test_plan_only_invalidates_previous_shared_ready_state`

### `tests/v5/test_summary_models.py` — 6개

과거 모델별 summary 디렉터리 공존·선택과 실패 파일 이관을 제외합니다. 현재 모델 변경은 force 요구 및 원본 보존으로 검증하고, CLI 성공/실패와 Gemini 텍스트 요청은 유지합니다.

남긴 검증: [`test_summary_model_change_requires_force`](../tests/v4/test_summary_models.py), [`test_cli_source_and_target_contract`](../tests/v4/test_concat_arms.py), [`test_cli_returns_failure_for_gemini_error`](../tests/v4/test_summary_models.py), [`test_gemini_text_only_and_reuse`](../tests/v4/test_summary_models.py).

- `test_embedding_does_not_fall_back_to_other_summary_model`
- `test_failure_model_survives_migration`
- `test_legacy_failure_mismatch_precedes_migration`
- `test_models_coexist_interrupt_resume_and_embedding_selection`
- `test_selected_summary_provenance_and_staleness`
- `test_summary_cli`

### `tests/v5/test_summary_resume.py` — 2개

현재 penalty 재개와 terminal 저장 실패 복구는 유지합니다. 과거 실패 파일 이관 및 legacy/terminal/pending 혼합 조합은 제외합니다.

남긴 검증: [`test_resume_after_out_of_order_results`](../tests/v4/test_summary_resume.py), [`test_terminal_write_failure_recovers_without_regeneration`](../tests/v4/test_summary_resume.py).

- `test_failure_penalty_survives_migration_and_update`
- `test_mixed_legacy_terminal_and_pending_failures`

### `tests/v5/test_summary_titles.py` — 3개

현재 요약은 제목을 입력하지 않으므로 과거 제목 입력·placeholder·terminal 출처 조합을 제외합니다. 현재 요약의 제목 미사용과 임베딩 단계 제목 결합은 유지합니다.

남긴 검증: [`test_shared_title_free_generation`](../tests/v4/test_concat_arms.py), [`test_composition_cases_and_zero_vectors`](../tests/v4/test_concat_arms.py).

- `test_legacy_terminal_failure_stays_empty_without_invented_provenance`
- `test_title_rendering_preserves_input_literals`
- `test_titles_reach_every_summary_and_input_hash`

### `tests/v5/test_validation_selection.py` — 7개

현재 실제 생성 결과로 catalog·event·시간 분할 보존과 다른 arm 대체 금지를 검증합니다. 과거 summary 모델 선택·전용 provenance 및 worker/진단 세부 조합은 제외합니다.

남긴 검증: [`test_full_catalog_applies_to_all_targets_and_preserves_shared_data`](../tests/v4/test_validation_selection.py), [`test_missing_summary_does_not_fall_back_to_another_arm`](../tests/v4/test_validation_selection.py), [`test_tampered_or_interrupted_selection_is_not_readable`](../tests/v4/test_validation_selection.py).

- `test_all_missing_summaries_preserve_the_full_cohort`
- `test_diagnosis_reports_full_selection_and_rejects_corrupt_scenes`
- `test_empty_representations_and_no_cross_arm_fallback`
- `test_fixed_dates_and_history_survive_generation_failures`
- `test_missing_manifest_metadata_only_and_outside_target`
- `test_recovery_deletion_source_changes_and_partial_cache_invalidation`
- `test_worker_uses_same_full_table_and_rejects_changed_selection`

## 매개변수 및 준비 코드 축소

- 공통 summary의 source×표현 전체 곱은 graph/Qwen, description/Qwen으로 축소했습니다. Qwen·Gemini의 실제 backend 연결과 graph cutoff 처리, summary 모델 변경은 별도 테스트로 남겼습니다.
- Gemini 동시성은 대표 2-thread 사례, 순서 밖 중단 재개는 기본 force=False, summary penalty 중단은 중간 pass의 5번째 완료를 사용합니다.
- 파서 화살표 변형은 한 사례, 모호한 텍스트와 JSON 거부는 각각 3개 대표 사례로 축소했습니다. 한 값만 남은 parametrize decorator는 제거했습니다.
- 사용하지 않는 import, 생성/캐시 전용 helper, 진행률 전용 fixture와 함수, 빈 테스트 모듈을 제거했습니다. 남긴 shared frame·parallel·penalty 테스트에서도 로그/진행률 모양 assertion을 걷어냈습니다.

세부 입력 변형, 과거 설정/파일 조합, 운영 표시 및 자원 상한의 회귀 탐지 범위가 줄었습니다. 원래 전체 커버리지의 보존이나 속도 개선 비율을 보장하는 변경은 아닙니다.
