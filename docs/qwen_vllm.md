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

Qwen Graph와 Description은 영상 경계 없이 장면을 연속 공급합니다. 한 pass가 모두 끝나면 실패한 장면만 다음 repetition penalty로 재생성하고 같은 GPU 엔진을 재사용합니다. Gemini 추출도 영상 경계 없이 장면을 공급하며 `extraction.gemini.threads`로 전체 동시성을 제한하고 장면 완료 시 결과를 게시합니다. Vertex Gemini가 `429 RESOURCE_EXHAUSTED`를 반환하면 해당 요청만 30초 대기한 뒤 한 번 재시도합니다. 두 번째 429와 그 밖의 API 오류는 장면 실패로 기록합니다.

고정 개수로 요청을 잘라 묶음 전체의 완료를 기다리는 처리는 추출·요약 단계에 없습니다. GPU 메모리용 동시 요청 한도 안에서 완료할 때마다 다음 요청을 공급합니다.

## 생성과 실패 기록

Graph는 JSON Schema 강제 생성 없이 `[Entities]`, `[Relations]`, `[Context]`, `[End]` 줄 형식을 요청합니다. Qwen과 Gemini가 공통 파서로 기존 `entities / relations / context` JSON 객체를 구성한 뒤 필드·타입을 검증합니다. 완전한 JSON 응답도 계속 지원합니다. ID 중복·미해결 관계 참조는 허용하며 개체·관계 개수 제한은 프롬프트 지침입니다. Description과 Summary에도 구조 grammar를 적용하지 않습니다.

Repair는 화살표 변형(`→`, `⇒`, `-->`, `=>`), 공백으로 구분된 하이픈·대시, 헤더 대소문자·콜론·Markdown 제목, 항목 앞 bullet·번호, 전각 구분자, 마지막 세미콜론, 응답 전체 코드 펜스를 처리합니다. 관계는 주체·관계·대상이 유일하게 분리될 때만 변환하며 ID·속성·관계 문구 안의 하이픈은 보존합니다. 빈 섹션은 `none` 또는 `[]`를 명시해야 합니다. 필수 섹션·종료 표식 누락, 역방향·모호한 관계, 중복 섹션, 종료 후 추가 내용은 실패입니다. JSON의 작은따옴표·구문 따옴표·마지막 쉼표는 복구할 수 있지만 잘린 괄호를 닫거나 여러 객체 중 하나를 선택하지 않습니다. Parser 버전은 Graph 생성 provenance에 포함합니다.

장면·요약의 현재 상한은 각각 1,024 tokens입니다. Qwen Graph·Desc 및 모든 Summary는 `1.00 → 1.05 → 1.10 → 1.15 → 1.20` 순서로 각 pass의 실패 항목만 재생성합니다(설정 목록 사용). 프롬프트는 그대로 사용하며 별도 교정 프롬프트는 만들지 않습니다. Qwen Graph·Desc·Summary의 `finish_reason=length`와 Gemini Graph·Desc의 `MAX_TOKENS`는 완전해 보이는 본문이라도 토큰 한도 실패입니다. 마지막까지 실패한 Qwen Graph·Desc·Summary의 비어 있지 않은 최종 원문은 E2E 입력용 Raw로 보존합니다. 빈 원문은 실패 기록만 남깁니다. Raw Desc는 `description`에 원문을, 장면 메타데이터에 `status=raw_fallback`을 저장합니다.

장면 실패는 `scenes/failures/{content_id}.jsonl`에 `content_id`, `scene_idx`, `error`, `raw_output`을 저장합니다. Summary 실패는 `summaries/failures.jsonl`에 `content_id`, `error`, `raw_output`을 저장합니다. 원문은 공백·줄바꿈을 포함해 그대로 보존하며 응답이 없으면 빈 문자열입니다. `.recovery`, `.pending`, `.checkpoints` 및 콘텐츠 진행 cursor는 생성하지 않습니다. 실행 시작 시 이전 실패 파일을 새 경로·필드로 옮기며, 예전 기록에 원문이 없으면 빈 문자열을 사용합니다.

정상 결과는 다음 실행에서 재사용합니다. Graph·Desc·Summary는 기존 실패 항목을 다음 실행에서 다시 처리합니다. Qwen 생성은 첫 penalty부터 다시 시작합니다. 정상 결과 저장 후 해당 실패 행을 지우고, 마지막 실패가 해결되면 파일도 삭제합니다. 재시도 중단·재실패 시에는 실패 기록을 유지하거나 최신 오류·원문으로 갱신합니다. 저장 전에 유실된 응답은 다시 생성하며, 이전 시도나 교정 초안을 복구하지 않습니다. 엔진/OOM/저장 오류는 그대로 전파합니다.

진행률은 이번 실행의 요청을 기준으로 하며 `failed`는 각 항목의 최신 결과를 나타내며 재생성에 성공하면 `success`로 옮깁니다. `raw`는 실패의 부분집합입니다. 장면 `scene/s`에는 실패 결과도 포함합니다. Qwen과 Gemini 모두 장면별로 결과를 저장합니다. 장면 JSONL은 `content_id`와 `description` 또는 `scene_graph`만 담으며, 식별 정보와 캐시 provenance는 `scenes/.metadata/{content_id}.json`에 보존합니다.

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
