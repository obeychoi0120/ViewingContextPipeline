# 핵심 시나리오 중심 테스트 정리 — 2026-10-06

사용자 요청: 필수적인 테스트만 남기고 함수 수를 절반 정도 줄입니다. 정상 실행, 결과 정확성, 대표 실패·복구를 독립적인 테스트로 유지하며, 큰 함수로 병합해서 함수 수만 줄이지 않습니다.

## 전후 비교

| 기준 | 정리 전 | 정리 후 | 감소 |
|---|---:|---:|---:|
| 테스트 파일 (`test_*.py`) | 52 | 47 | 5 |
| 테스트 함수 (`test_*` 정의) | 233 | 120 | 113 (48.5%) |
| pytest 수집 사례 (매개변수 포함) | 528 | 272 | 256 (48.5%) |

| 영역 | 정리 전 함수 | 정리 후 함수 |
|---|---:|---:|
| `extraction` | 82 | 39 |
| `integration` | 23 | 10 |
| `preparation` | 14 | 11 |
| `validation` | 114 | 60 |

함수 수는 AST에서 `test_`로 시작하는 함수 정의를 셌습니다. 실행 사례 수는 전체 pytest의 수집 결과 기준입니다. 남긴 함수의 매개변수 조합은 유지했으며, 함수 병합은 하지 않았습니다.

## 삭제 방향과 남긴 검증

| 줄인 항목 | 남긴 대체·핵심 검증 | 의도적으로 포기한 범위 |
|---|---|---|
| v7 Actions의 optional slot·reserved ID·JSON 필드별 전수 변형 | `test_context_actions_v8`: 정상 roles 저장·요약, 현재 참조 검증, 현재 프롬프트 예제·잘림; `test_context_actions`: 필수 section·구조 검증 및 legacy 원문 round trip | 과거 v7 기본값·receiver 형식의 각 변형을 모두 독립 검증하는 범위 |
| native/repair 텍스트 파서 표기 및 JSON 추가 변형 | 모호한 텍스트 거부, JSON 대표 복구·복구 불가, 생성→저장→후속 요약의 structured/raw/legacy 보존 | 과거 Relations 화살표·context optional·End 표기의 각 조합 |
| v7 payload와 semantic hash의 고정 스냅샷, 부분 실패·복사 재개 중복 | `test_v7_success_reuse_and_force`: 네 source의 기존 정상 결과·provenance·원본 bytes 보존과 강제 재생성; 현재 cache 변경/Run 이름 변경 및 요약 재개 | 과거 버전의 모든 hash가 현재 구현과 동일하다는 보장, 과거 복사/실패 이력 세부 변형 |
| 출력 token 필드 순서·unknown usage·오류 값별 별도 검증 | `test_cutoff_failure_retains_tokens_after_failure_log_reload`: Qwen·Gemini 모두 잘림 raw/token 재로딩; 모델별 token limit 요청·provenance 전달 | usage extractor의 필드별 전수 검증과 로그 표시 전용 변형 |
| 중복 backend streaming·engine 재사용·추출 scheduling 변형 | Qwen refill/저장 실패/engine 실패 전파, Gemini 순서 밖 완료·중단, 실패만 재시도, 두 모델 요청 및 retry 상한 | 같은 scheduling 동작의 대용량 변형, engine 수명/성능 전용 보장 |
| 레거시 설정·경계값·부분 CLI 옵션 조합 | 현재 명시적 target/source 설정, config 구조 거부, README의 공개 명령 dispatch | 제거된 shell script 경로 검증, 문서 링크 존재 전수 검사, retired 명령 및 profiling CLI 옵션별 오류 메시지 |
| 중복 catalog/title 및 sampling timestamp 문자열 검증 | 전체 catalog 보존, title 보완과 누락 zero vector, 손상 이미지, 프레임 sampling·공유 준비·동시 저장·중단 재개 | 별도 blank-title catalog 사례, timestamp 표기 경계값별 검증 |
| Text/Graph adapter·cache identity·입력 문자열/순서 중복 | 제목/영상의 별도 임베딩·마스크·경계 hash, 다른 source로 대체하지 않는 전체 catalog, 입력/캐시 손상·변경 무효화·원자적 저장 | 과거 adapter의 동일성, cache 재공개 등 중복 경로와 세부 identity 필드별 변형 |
| profiling 표시·operator 분류·checkpoint 자동 추정 전용 검사 | profiling 사용 유무에 따른 학습 weight/rank 보존, chunk/cache/checkpoint의 수치 및 gradient 기준값 일치 | trace 분류·표시/계측 context·자동 메모리 추정의 모든 경계값 |
| 과거 학습 bundle 호환·준비 배열·확률 cache·determinism 옵션 중복 | causal masking, rolling 시간 분할/full history/tie/repeated target, NDCG·bootstrap·비교 통계, loss/gradient/update 기준값, NaN 업데이트 차단 | 과거 bundle별 완료 호환, cache lifetime·RNG 옵션별 계약 |
| GNN 보조 설정·profiling·scene special case 중복 | similarity/residual GCN의 NumPy 기준값, ragged/permutation 독립성, title isolation/fallback, joint loss·chunk·checkpoint·reload; Mean/GNN 통합 학습·진단·입력 invariance | GNN pair count 프로파일링·자동 checkpoint 추정 및 빈 scene 변형 각각의 별도 테스트 |
| 병렬 dispatch/일반 worker 오류 별도 검사 | 직렬/병렬 weight·metrics 일치, 부모 중단 후 완료 artifact 재사용, 비정상 worker 종료 시 모든 child 종료 | 정상 오류 메시지의 별도 변형과 dispatch 구현 단독 검사 |
| graph prompt pilot 도구 전용 3개 테스트 | 현재 parser와 run 비교 통계를 검증하며, pilot 도구 자체의 대체 검증은 추가하지 않음 | pilot subset 고정·baseline Run 보호·pilot 전용 join 검증 |

6개 현재 arm의 생성→요약→임베딩→CPU 학습→진단→재사용 통합 검증은 유지합니다. 실제 `ModelConfig`가 요구하는 서로 다른 seed 42/43/44는 통합 fixture에서 유지하고, 별도 병렬 작업 구분에는 42/43을 사용합니다. Mean/GNN의 수치 기준 코드(`rolling_reference.py`, `graph_reference.py`, `graph_v7_reference.py`)와 `graph_fixture.py`는 변경하지 않았습니다.

## 기존 실패와 보정

수정 전 전체 pytest: **515 passed / 13 failed**, 528개 사례, 100.31초. 기존 실패를 통과시키기 위한 제품 수정은 하지 않았습니다.

- 유지한 2개 실패를 현재 계약에 맞게 보정했습니다. `test_prompt_examples_and_truncation`의 삭제된 Summary 지시문 문자열 기대값을 실제 전달된 scene JSON의 내용 일치 검증으로 바꿨습니다. `test_single_prepare_command_completes_titles_without_extra_artifacts`는 폐기된 `values` 배열 대신 현재 메타데이터 저장 계약인 `title_values`를 검증합니다. 제목 누락 zero vector와 변경 후 무효화 검증은 유지했습니다.
- 기존 실패 중 11개는 축소 범위에 해당하는 검사였습니다: 문서 링크 검사 2개, 현재 저장소에 없는 `run_pipeline_v7.sh`/`script_graph_v7.sh` 실행 검사 2개, 고정 v7 hash 검사 7개. 해당 검사들은 삭제했습니다. 문서의 잘못된 링크나 과거 고정 hash와의 차이를 수정한 것으로 해석하면 안 됩니다. README 명령 dispatch와 v7 정상 결과 재사용·원본 보존은 별도로 유지했습니다.
- 위 2개 테스트 외에는 남긴 함수 본문·decorator·signature의 AST가 변경 전과 동일함을 확인했습니다. 사용하지 않게 된 import 58개와 상수(`GRAPH`, `MANIFEST`, `CASES`)를 제거했습니다.

## 사용하지 않는 fixture 삭제

남긴 v7 reuse 테스트는 네 source 각각의 `scenes_*_0.json` / `summaries_*_0.json`만 사용합니다. 그 8개 파일은 보존했습니다. 고정 hash·실패/복사 변형 삭제에 따라 아래 7개 파일을 삭제했습니다.

- `tests/fixtures/v7/manifest.json`
- `tests/fixtures/v7/representation_hashes.json`
- `tests/fixtures/v7/summary_failures.json`
- `tests/fixtures/v7/scenes_graph_qwen_1.json`
- `tests/fixtures/v7/scenes_graph_gemini_1.json`
- `tests/fixtures/v7/summaries_desc_qwen_1.json`
- `tests/fixtures/v7/summaries_desc_gemini_1.json`

공통 준비 fixture는 기존 `tests/conftest.py`에 있으며 변경하지 않았습니다. 남긴 테스트 간 직접 helper import가 없고, 수치 reference와 `config_data` 준비 helper의 기존 구조를 보존했습니다.

## 검증

- 환경: 이 VM의 `vc_cloud`, CPU 테스트 (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). 생성·임베딩 backend는 fake를 사용합니다. 실제 모델 inference, 외부 API, 전체 실험의 품질·CUDA 성능을 검증한 결과는 아닙니다.
- 수집: `python -m pytest --collect-only -q` → **272 tests collected**.
- 전체 실행: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q` → **272 passed**, 90.32초. 소형 CPU 학습과 multiprocessing 검증을 포함합니다.
- 정적 검사: `python -m ruff check tests` → **All checks passed!**.
- patch 검사: `git diff --check` → **통과**.
- 변경 범위: `tests/`와 이 정리 기록만 변경합니다. 제품 코드·공개 API·설정·프롬프트 및 기존 Run 산출물은 변경하지 않았습니다.

## 파일별 삭제 함수 목록

아래는 실제 삭제한 113개 함수입니다. 완전 삭제 모듈은 `[모듈 삭제]`로 표시합니다. 대체 검증과 축소 범위는 위 표에 기록했습니다.

### `tests/extraction/test_context_actions.py`

- `test_direction_and_internal_hyphens`
- `test_empty_references_and_actions`
- `test_empty_sections`
- `test_ambiguous_or_incomplete_lines_are_preserved_as_text`
- `test_prompt_examples_and_summary_template`
- `test_end_is_optional_for_native_and_repaired_actions`
- `test_token_termination_is_failure_even_with_complete_graph`
- `test_summary_receives_raw_but_does_not_count_it_as_structured`
- `test_prompt_limits_do_not_reject_or_trim_graph`
- `test_duplicate_and_reference_tags_are_collected_without_duplicates`
- `test_json_structure_uses_specific_tags`

### `tests/extraction/test_context_actions_v7.py` [모듈 삭제]

- `test_positional_defaults_and_explicit_values`
- `test_json_missing_keys_only`
- `test_unknown_allowed_without_entity`
- `test_unknown_is_not_a_known_endpoint`
- `test_invalid_optional_values_preserve_raw`
- `test_reserved_entity_ids`
- `test_required_slots_and_extra_slots_are_not_repaired`
- `test_v7_examples_and_storage_summary_roundtrip`
- `test_mixed_optional_slots_and_json_repairs`

### `tests/extraction/test_context_actions_v8.py`

- `test_legacy_receiver_validation_disabled_without_mutation`

### `tests/extraction/test_graph_prompt_pilot.py` [모듈 삭제]

- `test_first100_is_stable_and_does_not_modify_shared_cohort`
- `test_cannot_use_baseline_run_as_pilot`
- `test_comparison_joins_scene_ids_and_counts_raw_invalid_missing`

### `tests/extraction/test_output_tokens.py`

- `test_scene_output_tokens_and_exact_field_order`
- `test_summary_tokens_cover_sources_backends_and_failure_resume`
- `test_unknown_usage_is_not_estimated_or_inherited`
- `test_invalid_persisted_tokens_are_rejected`

### `tests/extraction/test_qwen_workers.py`

- `test_worker_reuses_one_engine_and_passes_requests_concurrently`

### `tests/extraction/test_scene_failures.py`

- `test_empty_graph_is_a_generation_failure`

### `tests/extraction/test_scene_storage.py`

- `test_compact_text_graph_remains_summary_input_without_format_marker`
- `test_changed_settings_retry_only_explicit_scene_and_summary_failures`

### `tests/extraction/test_semantic_graph_text_parser.py`

- `test_line_graph_preserves_fields_and_hyphenated_values`
- `test_repairs_only_relation_delimiters`
- `test_graph_without_context`
- `test_v4_example_passes_scene_validation_and_summary`
- `test_optional_context_does_not_replace_required_sections`

### `tests/extraction/test_structured_recovery.py`

- `test_streaming_once_refills_before_completion_and_never_retries`
- `test_graph_repair_never_invents_required_fields`
- `test_nonempty_graph_format_errors_survive_summary_roundtrip`

### `tests/extraction/test_summary_models.py`

- `test_summary_model_change_requires_force`

### `tests/extraction/test_token_limits.py`

- `test_invalid_model_limit_rejected`
- `test_missing_model_limits_rejected`

### `tests/integration/test_contracts.py`

- `test_missing_metadata_row_keeps_catalog_scope_and_records_failure`

### `tests/integration/test_entrypoints.py` [모듈 삭제]

- `test_document_links_exist`
- `test_scripts_dispatch_supported_commands`
- `test_retired_extraction_commands_are_rejected`

### `tests/integration/test_graph_scene_scheduling.py`

- `test_gemini_refills_across_contents_and_saves_finished_contents`

### `tests/integration/test_profiling_cli.py` [모듈 삭제]

- `test_profile_rejected_before_loading_other_steps`
- `test_profile_interval_must_be_positive`
- `test_profile_cli_passes_through_step_to_rolling`
- `test_operator_profile_requires_recommendation_and_interval`

### `tests/integration/test_v7_artifacts.py`

- `test_v7_payloads_and_semantic_hashes`
- `test_all_six_representation_keys_match_pre_refactor`
- `test_v7_failed_summary_retries_without_rewriting_log_on_read`
- `test_v7_scene_copy_resumes_after_interruption`

### `tests/preparation/test_catalog.py` [모듈 삭제]

- `test_title_csv_omits_blank_titles_for_catalog_coverage_check`

### `tests/preparation/test_data_preparation.py`

- `test_prepare_visual_item_reuses_catalog_duration_and_uses_shared_root`

### `tests/preparation/test_visual_sampling.py`

- `test_timestamp_truncation_and_filename_agree`

### `tests/validation/test_component_model.py`

- `test_metadata_array_and_component_adapters_agree`

### `tests/validation/test_concat_arms.py`

- `test_registry_and_config`
- `test_reject_title_conditioned_summary`
- `test_cli_source_and_target_contract`
- `test_summary_reuses_existing_output`
- `test_unified_summary_rejects_invalid_source`
- `test_unstructured_graph_reaches_summary_and_embedding`

### `tests/validation/test_direct_graph.py`

- `test_node_embedding_text_format_and_list_order`
- `test_json_input_version_cannot_reuse_labeled_input_cache`
- `test_cli_rejects_invalid_modes_before_loading`
- `test_empty_and_single_scene_pooling`
- `test_four_gpu_feature_partition_and_order`
- `test_optional_explicit_role_states_and_scene_order`
- `test_source_identity_title_model_and_catalog`
- `test_deprecated_receiver_is_ignored_in_graph_inputs`
- `test_v7_model_contract_reuses_inputs_and_rejects_old_checkpoint`
- `test_title_only_meta_uses_common_v7_baseline`
- `test_none_unknown_states_are_role_specific_and_receiver_is_ignored`
- `test_scene_without_entities_or_actions_keeps_context`

### `tests/validation/test_early_stopping.py`

- `test_zero_delta_preserves_previous_stopping_rule`
- `test_batch_size_and_legacy_delta_defaults`
- `test_invalid_delta_rejected`
- `test_training_changes_invalidate_identity_without_changing_input`

### `tests/validation/test_graph_execution.py`

- `test_checkpoint_does_not_repeat_preparation_and_catalog_reuses_cache`
- `test_execution_tuning_does_not_invalidate_training_identity`
- `test_missing_and_title_only_batches`
- `test_repeated_deterministic_training_is_identical`
- `test_checkpoint_auto_uses_total_batch_memory`

### `tests/validation/test_large_optimizations.py`

- `test_seed_policy_preserves_rng_and_switches_determinism`
- `test_determinism_default_and_change_invalidate_training_cache`
- `test_graph_rejects_small_stored_features_before_training`
- `test_large_runtime_and_actual_output_contract`

### `tests/validation/test_metrics.py`

- `test_top_k_uses_the_same_index_tie_break_as_target_rank`

### `tests/validation/test_model.py`

- `test_all_item_towers_use_frozen_features_and_trainable_projection`
- `test_popularity_distribution_and_non_finite_guards`

### `tests/validation/test_profiling.py`

- `test_disabled_is_inert_and_sampling_resets_after_error`
- `test_compact_records_preserve_session_context_and_measurements`
- `test_operator_backward_attribution_separates_shared_projection_and_other_linear`
- `test_operator_profile_error_preserves_original_and_resets_context`

### `tests/validation/test_rolling.py`

- `test_csv_retains_duplicates_and_rejects_fractional_timestamps`
- `test_evaluation_masks_history_older_than_the_ten_item_context`
- `test_evaluation_keeps_parameters_and_caches_catalog`

### `tests/validation/test_rolling_execution.py`

- `test_prepared_sequences_and_vector_mask_exact`
- `test_popularity_cache_validation_and_bounded_lifetime`
- `test_selection_refit_test_and_old_bundle_compatibility`

### `tests/validation/test_rolling_parallel.py`

- `test_worker_failure_is_reported_and_all_children_are_joined`
- `test_dispatch_is_unique_and_skips_completed_work`

### `tests/validation/test_scene_gcn.py`

- `test_single_and_actionless_scene_zero_similarity_and_empty_nodes`
- `test_profile_preserves_outputs_gradients_and_counts_pairs`
- `test_contracts_cli_and_scene_pair_execution_config`
- `test_gnn_auto_checkpoint_accounts_for_quadratic_pairs`

### `tests/validation/test_shared_cache.py`

- `test_existing_local_embeddings_can_be_published_without_encoding`

### `tests/validation/test_steps.py`

- `test_legacy_embedding_is_not_reused`

### `tests/validation/test_validation_selection.py`

- `test_missing_summary_does_not_fall_back_to_another_arm`
