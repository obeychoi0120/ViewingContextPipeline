# Qwen vLLM 실행과 처리량 확인

Qwen 추출과 Qwen 요약은 로컬 비양자화 checkpoint, BF16, vLLM 0.28.0 AsyncLLM을 사용합니다. [config.yaml](../config.yaml)의 `extraction.qwen`에서 GPU 메모리·동시 요청·context 길이를 설정합니다.

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
CUDA_VISIBLE_DEVICES=0 python -m extraction extract-graph-scenes --model qwen --run-id "$RUN_ID" --schema prompts/scene_graph_v3.md --arm graph_qwen
CUDA_VISIBLE_DEVICES=0,2 python -m extraction summarize --run-id "$RUN_ID" --schema prompts/summary_graph_v5.md --model qwen --arm graph_qwen
```

Qwen Graph와 Description은 영상 경계 없이 장면을 연속 공급합니다. 한 pass가 모두 끝나면 실패한 장면만 다음 repetition penalty로 재생성하고 같은 GPU 엔진을 재사용합니다. Gemini 추출도 영상 경계 없이 장면을 공급하며 `extraction.gemini.threads`로 전체 동시성을 제한하고 장면 완료 시 결과를 게시합니다. Vertex Gemini가 `429 RESOURCE_EXHAUSTED`를 반환하면 해당 요청만 30초 대기한 뒤 한 번 재시도합니다. 두 번째 429와 그 밖의 API 오류는 장면 실패로 기록합니다.

고정 개수로 요청을 잘라 묶음 전체의 완료를 기다리는 처리는 추출·요약 단계에 없습니다. GPU 메모리용 동시 요청 한도 안에서 완료할 때마다 다음 요청을 공급합니다.

## 생성과 실패 기록

Graph는 JSON Schema 강제 생성 없이 `[Entities]`, `[Relations]`, `[End]` 줄 형식을 요청합니다. Qwen과 Gemini가 공통 파서로 `entities / relations` JSON 객체를 구성합니다. 파싱이나 구조 검증이 불가능하면 원문을 정상 Scene 텍스트로 저장하여 Summary에 전달합니다. 완전한 JSON 응답도 계속 지원합니다. ID 중복·미해결 관계 참조는 허용하며 개체·관계 개수 제한은 프롬프트 지침입니다. Description과 Summary에도 구조 grammar를 적용하지 않습니다.

Repair는 화살표 변형(`→`, `⇒`, `-->`, `=>`), 공백으로 구분된 하이픈·대시, 헤더 대소문자·콜론·Markdown 제목, 항목 앞 bullet·번호, 전각 구분자, 마지막 세미콜론, 응답 전체 코드 펜스를 처리합니다. 관계는 주체·관계·대상이 유일하게 분리될 때만 변환하며 ID·속성·관계 문구 안의 하이픈은 보존합니다. 빈 섹션은 `none` 또는 `[]`를 명시해야 합니다. 필수 데이터 섹션 누락, 역방향·모호한 관계, 중복 섹션 등으로 구조화할 수 없는 출력도 생성 실패로 처리하지 않고 원문을 보존합니다. JSON의 작은따옴표·구문 따옴표·마지막 쉼표는 복구할 수 있지만 잘린 괄호를 닫거나 여러 객체 중 하나를 선택하지 않습니다. `[End]`는 생략할 수 있으며 응답 끝에서 파싱을 마칩니다. 출력 잘림은 종료 표식 유무 대신 Qwen의 `finish_reason=length`, Gemini의 `MAX_TOKENS`로 판단합니다. 이전 `[Context]` 출력은 호환용으로 허용합니다. 정상 종료한 비정형 출력은 `scene_graph` 문자열과 `graph_format="text"`로 저장하며 과거 실패 원문과 구분합니다. API·엔진 오류는 계속 오류로 처리합니다. Parser 버전은 Graph 생성 provenance에 포함합니다.

장면·요약의 기본 상한은 각각 1,024 tokens와 2,048 tokens입니다. Qwen Graph·Desc 및 모든 Summary는 `1.00 → 1.05 → 1.10 → 1.15 → 1.20` 순서로 각 pass의 실패 항목만 재생성합니다(설정 목록 사용). 프롬프트는 그대로 사용하며 별도 교정 프롬프트는 만들지 않습니다. Qwen Graph·Desc·Summary의 `finish_reason=length`와 Gemini Graph·Desc·Summary의 `MAX_TOKENS`는 완전해 보이는 본문이라도 토큰 한도 실패입니다. 최종 실패 장면은 Summary 입력에서 제외하며, 성공 장면이 없거나 Summary가 최종 실패하면 빈 Summary를 저장합니다. 실패 원문으로 내용을 채우지 않습니다.

장면 실패는 `scenes/{SCENE_ARM}/failures/{content_id}.jsonl`에 `content_id`, `scene_idx`, `error`, `raw_output`을 저장합니다. Summary 생성 실패는 `summaries/{SUMMARY_ARM}/failures.jsonl`에 `content_id`, `error`, `raw_output`, `repetition_penalty`를 저장합니다. 신규 실패의 `raw_output`은 빈 문자열이고 생성 provenance를 함께 보존합니다. 기존 원문 파일은 일괄 삭제하지 않습니다. `.recovery`, `.pending`, `.checkpoints` 및 콘텐츠 진행 cursor는 생성하지 않습니다. 실행 시작 시 이전 실패 파일을 새 경로·필드로 옮기며, 예전 기록에 원문이 없으면 빈 문자열을 사용합니다.

Qwen Graph·Desc, Gemini Graph와 Summary의 정상 결과는 다음 실행에서 재사용합니다. 한 Summary Arm의 생성 모델을 바꾸려면 새 Run 또는 해당 Arm의 `--force`가 필요합니다. 일반 실행은 실패 및 결과가 없는 항목만 처리합니다. 장면을 갱신해도 성공한 Summary는 보존하며 전체 재생성은 해당 단계의 `--force`로 요청합니다. 장면 Graph·Desc는 기존 실패 항목을 다음 실행에서 다시 처리하며 Qwen 장면 생성은 첫 penalty부터 시작합니다. Summary는 미처리 콘텐츠를 초기 penalty로 완료한 뒤 실패별 마지막 penalty보다 큰 다음 설정값부터 재개합니다. 마지막 penalty까지 완료했거나 penalty 필드가 없는 기존 실패는 재생성 없이 빈 실패 표현을 콘텐츠별 Summary JSON으로 복구합니다. 최종 실패 로그는 유지하며 일반 재개에서 재생성하지 않습니다. Summary 검증은 빈 출력과 토큰 한도 종료만 실패로 보고 문단·목록·마크업·구조화 필드는 허용합니다. 정상 결과 저장 후 해당 실패 행을 지우고, 마지막 실패가 해결되면 파일도 삭제합니다. 재시도 중단·재실패 시에는 실패 기록을 유지하거나 최신 오류·빈 raw_output·provenance로 갱신합니다. 저장 전에 유실된 응답은 다시 생성하며, 교정 초안은 복구하지 않습니다. Summary 최종 실패는 실패 이유·빈 raw_output·provenance와 penalty를 먼저 저장하므로 문서 저장 중 중단돼도 로그에서 복구합니다. 엔진/OOM/저장 오류는 그대로 전파합니다.

진행률은 이번 실행의 요청을 기준으로 하며 `failed`는 각 항목의 최신 결과를 나타내며 재생성에 성공하면 `success`로 옮깁니다. 신규 실패를 Raw 표현으로 사용하지 않으므로 `raw`는 0입니다. 장면 `scene/s`에는 실패 결과도 포함합니다. Qwen과 Gemini 모두 장면별로 결과를 저장합니다. Graph 장면 JSONL은 `content_id`, `scene_idx`, `scene_graph` 세 필드만 저장합니다. Description은 기존대로 `content_id`, `scene_idx`, `description`, `provenance`를 저장합니다. `.metadata`는 생성하지 않습니다. 이전 두 필드 파일의 장면 번호는 기존 `.metadata`에서 읽어 옮기고 새 파일 저장 후 해당 메타데이터를 삭제합니다. 이전 메타데이터가 없으면 장면 번호를 추측하지 않고 오류를 보고합니다.

## 선택적 처리량 측정

측정 도구는 명시적으로 요청한 경로에 요청 manifest와 보고서를 만듭니다. 운영 pipeline이 자동 생성하는 artifact는 아닙니다. 전용 GPU 한 장을 사용합니다.

```bash
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-scenes --schema prompts/scene_graph_v3.md --limit 144 --output /tmp/qwen-scenes.json
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-summary-qwen --schema prompts/summary_graph_v5.md --limit 144 --output /tmp/qwen-summaries.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --backend vllm --output /tmp/scenes-vllm.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-summaries.json --backend vllm --output /tmp/summaries-vllm.json
```

처음 16개는 warmup, 나머지는 측정입니다. 요청이 적으면 `--warmup 1`처럼 조절합니다. 같은 고정 요청·이미지 hash를 사용하며 생성 1회 처리량을 측정합니다. 운영 단계의 검증·결과 저장 비용은 포함하지 않습니다. 실제 CUDA/API 호환성·속도·내용 품질은 CPU 및 mock 회귀 테스트와 별도로 검증해야 합니다.

벤치마크의 vLLM backend도 보이는 GPU를 모두 사용합니다. Transformers 비교용 backend는 단일 GPU 전용이므로 `CUDA_VISIBLE_DEVICES`로 하나만 지정합니다.

## v7 생성 소스와 추천 Arm

Scene·Summary 생성의 `--arm`은 `graph_qwen`, `desc_qwen`, `graph_gemini` 중 하나입니다. 추천에는 `meta`, `graph_qwen`, `desc_qwen`, `graph_qwen_meta`, `desc_qwen_meta`, `graph_gemini_meta`만 사용합니다. 결합 Arm 전용 Summary를 생성하지 않습니다.

`summary_description_v5.md`, `summary_graph_v5.md`는 제목 없이 200–300단어를 목표로 하며 최대 350단어를 프롬프트에만 지정합니다. Summary 한도 2,048은 Qwen의 `max_tokens`, Gemini의 `max_output_tokens`에 전달됩니다. metadata는 생성 후 임베딩 직전에 제목 + `"\n\n"` + Summary로 결합합니다. 하나만 있으면 그것만 사용하며, BGE는 512토큰을 넘는 뒷부분을 자릅니다. 다음 Run에는 v5 프롬프트를 명시해야 하며 프롬프트 변경만으로 기존 성공 Summary가 자동 갱신되지는 않습니다.
