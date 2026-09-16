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

Qwen Graph와 Description은 영상 경계 없이 장면을 연속 공급하고 각 요청을 한 번만 생성합니다. Gemini 추출도 영상 경계 없이 장면을 공급하며 `extraction.gemini.threads`로 전체 동시성을 제한하고 장면 완료 시 결과를 게시합니다. Gemini SDK의 HTTP 요청도 한 번만 시도합니다.

고정 개수로 요청을 잘라 묶음 전체의 완료를 기다리는 처리는 추출·요약 단계에 없습니다. GPU 메모리용 동시 요청 한도 안에서 완료할 때마다 다음 요청을 공급합니다.

## 생성과 실패 기록

Graph에는 TOBE JSON Schema 제약을 적용합니다. JSON 구조만 검증하며 ID 중복·미해결 관계 참조는 허용합니다. Description과 Summary에는 구조 grammar를 적용하지 않습니다.

장면 상한은 1,024 tokens, 요약 상한은 512 tokens입니다. Desc·Graph·Summary 모두 한 번 생성한 결과로 성공·실패를 확정합니다. 기본 repetition penalty는 `1.00`이고 기존 목록 설정은 첫 값만 사용합니다. Summary의 형식·길이 위반도 즉시 실패이며 별도 교정 요청은 하지 않습니다. Qwen Graph·Summary의 `finish_reason=length`는 토큰 한도 실패입니다. 실패한 Graph·Summary의 비어 있지 않은 원문은 E2E 입력용 Raw로 보존합니다.

각 장면 출력 폴더의 `failure.jsonl`에는 `content_id`, `scene_idx`, `error`만 저장합니다. 각 Summary 출력 폴더의 같은 파일에는 `content_id`, `error`만 저장합니다. `failures/`, `.recovery`, `.pending`, `.checkpoints` 및 콘텐츠 진행 cursor는 생성하지 않습니다. 기존 `failures/`는 실행 시작 시 최소 필드로 통합하고 이전 임시 기록을 제거합니다.

정상 결과와 실패 기록은 다음 실행에서 재사용합니다. 실패 항목을 다시 처리하려면 `--force`로 해당 단계를 실행합니다. 중단 후에는 저장된 장면·요약과 실패 기록을 건너뛰고 미완료 항목을 생성합니다. 저장 전에 유실된 응답은 다시 생성하며, 이전 시도나 교정 초안을 복구하지 않습니다. 엔진/OOM/저장 오류는 그대로 전파합니다.

진행률은 이번 실행의 요청을 기준으로 하며 `failed`에 첫 응답 실패를 포함하고 `raw`는 그 부분집합입니다. 장면 `scene/s`에는 실패 결과도 포함합니다. Qwen과 Gemini 모두 장면별로 결과를 저장합니다. 장면 JSONL은 `content_id`와 `description` 또는 `scene_graph`만 담으며, 식별 정보와 캐시 provenance는 `scenes/.metadata/{content_id}.json`에 보존합니다.

## 선택적 처리량 측정

측정 도구는 명시적으로 요청한 경로에 요청 manifest와 보고서를 만듭니다. 운영 pipeline이 자동 생성하는 artifact는 아닙니다. 전용 GPU 한 장을 사용합니다.

```bash
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-scenes --schema prompts/graph_scene_v3.md --limit 144 --output /tmp/qwen-scenes.json
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-summary-qwen --schema prompts/graph_summary_v4.md --limit 144 --output /tmp/qwen-summaries.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --backend vllm --output /tmp/scenes-vllm.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-summaries.json --backend vllm --output /tmp/summaries-vllm.json
```

처음 16개는 warmup, 나머지는 측정입니다. 요청이 적으면 `--warmup 1`처럼 조절합니다. 같은 고정 요청·이미지 hash를 사용하며 생성 1회 처리량을 측정합니다. 운영 단계의 검증·결과 저장 비용은 포함하지 않습니다. 실제 CUDA/API 호환성·속도·내용 품질은 CPU 및 mock 회귀 테스트와 별도로 검증해야 합니다.

벤치마크의 vLLM backend도 보이는 GPU를 모두 사용합니다. Transformers 비교용 backend는 단일 GPU 전용이므로 `CUDA_VISIBLE_DEVICES`로 하나만 지정합니다.
