# Qwen vLLM 실행과 처리량 확인

v5 Qwen 추출과 모든 요약은 로컬 비양자화 checkpoint, BF16, vLLM 0.28.0 AsyncLLM을 사용합니다. [config.yaml](../config.yaml)의 `extraction.qwen`에서 GPU 메모리·동시 요청·context 길이를 설정합니다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `gpu_memory_utilization` | 0.90 | GPU당 엔진 메모리 예산 |
| `max_num_seqs` | 32 | GPU당 동시 스케줄링 요청 수 |
| `max_num_batched_tokens` | 16384 | 엔진 iteration의 토큰 예산 |
| `max_model_len` | auto | vLLM 메모리 기반 결정. 양의 정수 지정도 가능 |
| `renderer_num_workers` | 4 | 비동기 전처리·이미지 로딩 동시성 |
| `enable_chunked_prefill` | true | 긴 prefill 분할 |
| `enable_prefix_caching` | true | 공통 접두부 KV cache 재사용 |
| `enforce_eager` | false | 컴파일/CUDA graph 허용 |
| `async_scheduling` | null | vLLM 호환성 판단과 기본값 사용 |

Qwen의 네 생성 단계는 `--gpus` 없이 `CUDA_VISIBLE_DEVICES`에서 보이는 GPU를 모두 사용합니다. 변수를 생략하면 CUDA에서 보이는 모든 GPU를 사용합니다. 빈 값이나 `-1`로 GPU를 숨기면 실행에 실패합니다. GPU 번호와 GPU/MIG UUID의 순서를 보존합니다. GPU마다 독립 엔진을 복제하고 tensor parallel은 1입니다. GPU당 미완료 요청 공급은 `2 × max_num_seqs`로 제한합니다. 워커는 이미지 경로·프롬프트를 받아 처리하고 완료 GPU에 다음 요청을 먼저 공급합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m extraction extract-graph-scenes --model qwen --run-id "$RUN_ID" --schema prompts/graph_scene_v3.md
CUDA_VISIBLE_DEVICES=0,2 python -m extraction summarize-graph --source qwen --run-id "$RUN_ID" --schema prompts/graph_summary_v4.md
```

Qwen Graph는 콘텐츠별 최대 8개 장면 묶음의 재시도·저장을 완료한 뒤 다음 묶음을 처리합니다. Qwen Description은 장면을 연속 공급합니다. Gemini 추출은 `extraction.gemini.threads`로 콘텐츠 내부 동시성을 제한하며 콘텐츠 완료 시 결과를 게시합니다.

## 생성과 복구

Graph에는 TOBE JSON Schema 제약을 적용합니다. 구조·ID 중복·미해결 관계 참조를 검증하며 어휘 enum은 없습니다. Description과 Summary에는 구조 grammar를 적용하지 않습니다. 임의 프롬프트 파일을 선택해도 Python Graph 출력 계약은 TOBE로 고정됩니다.

장면 상한은 1,024 tokens, 요약·교정 상한은 512 tokens입니다. Qwen 반복 패널티는 생성 토큰에만 한 번 적용합니다. 기본 장면 재시도 순서는 1.00·1.05·1.10·1.15·1.20입니다. 실패 Graph는 마지막 비어 있지 않은 결과를 Raw로 남길 수 있으며 정상 coverage에서 제외합니다. Summary는 빈 응답의 제한적 재시도와 길이·형식 교정 한 번을 구분합니다. 교정 후에도 계약을 만족하지 못하면 마지막 비어 있지 않은 결과를 Raw로 사용합니다.

진행률은 재사용을 제외한 요청 수를 기준으로 하며 장면과 요약 단위를 구분합니다. 장면 추출은 전체 입력을 미리 순회하지 않으므로 입력 공급이 끝나기 전까지 전체 건수는 미정이고 ETA는 `estimating`입니다. 출력·저장·재시도까지 포함한 처리 속도와 ETA입니다. 요약 교정은 같은 요약 작업의 일부입니다.

각 Qwen 응답을 durable journal에 먼저 저장한 후 최종 artifact를 게시합니다. 저장 오류나 중단 시 응답을 재추론하지 않고 게시를 재개합니다. 완료 journal은 삭제하고 최종 artifact에 입력 key·force 실행 ID·시도 수·repair mode를 보존합니다. 입력·프롬프트·설정 변경은 기존 캐시를 무효화합니다. 엔진/OOM/저장 오류는 그대로 중단하며 원인을 해결한 뒤 같은 명령으로 재개합니다.

## 선택적 처리량 측정

측정 도구는 명시적으로 요청한 경로에 요청 manifest와 보고서를 만듭니다. 운영 pipeline이 자동 생성하는 artifact는 아닙니다. 전용 GPU 한 장을 사용합니다.

```bash
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-scenes --schema prompts/graph_scene_v3.md --limit 144 --output /tmp/qwen-scenes.json
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-summary-qwen --schema prompts/graph_summary_v4.md --limit 144 --output /tmp/qwen-summaries.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --backend vllm --output /tmp/scenes-vllm.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-summaries.json --backend vllm --output /tmp/summaries-vllm.json
```

처음 16개는 warmup, 나머지는 측정입니다. 요청이 적으면 `--warmup 1`처럼 조절합니다. 같은 고정 요청·이미지 hash를 사용하며 생성 1회 처리량을 측정합니다. 운영 단계의 전체 재시도·교정·Raw 게시 비용은 포함하지 않습니다. 실제 CUDA/API 호환성·속도·내용 품질은 CPU 및 mock 회귀 테스트와 별도로 검증해야 합니다.

벤치마크의 vLLM backend도 보이는 GPU를 모두 사용합니다. Transformers 비교용 backend는 단일 GPU 전용이므로 `CUDA_VISIBLE_DEVICES`로 하나만 지정합니다.
