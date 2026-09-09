# Qwen vLLM 실행과 처리량 확인

Qwen 장면 추출과 Graph/Description 요약은 Ubuntu/CUDA의 **vLLM 0.28.0 AsyncLLM**을 사용합니다. Gemini가 만든 Graph의 요약도 같은 경로입니다. 모델은 `models.qwen`의 기존 비양자화 로컬 checkpoint를 BF16으로 읽고, KV cache도 모델 dtype을 따릅니다. Transformers FC patch는 운영 경로에서 제거했습니다.

## GPU와 설정

`config/pipeline.yaml`에 아래 선택적 설정을 둡니다. v3/v4 설정에 없거나 일부 키가 빠져 있으면 같은 기본값을 적용합니다. RTX 6000 Ada 48GB에서 측정하기 위한 시작값이며 최적값으로 검증된 수치는 아닙니다.

```yaml
extraction:
  qwen:
    gpu_memory_utilization: 0.90
    max_num_seqs: 32
    max_num_batched_tokens: 16384
    max_model_len: auto
    renderer_num_workers: 4
    enable_chunked_prefill: true
    enable_prefix_caching: true
    enforce_eager: false
    async_scheduling: null
```

| 설정 | 의미 |
| --- | --- |
| `gpu_memory_utilization` | GPU당 엔진 메모리 예산 비율 `(0, 1]`. GPU 연산 사용률 목표가 아님 |
| `max_num_seqs` | GPU당 엔진이 동시에 스케줄링할 요청 수 |
| `max_num_batched_tokens` | 엔진 한 실행 단계에서 처리할 토큰 예산 |
| `max_model_len` | `auto`는 vLLM의 메모리 기반 결정(`-1`)에 대응. 양의 정수로 직접 제한 가능 |
| `renderer_num_workers` | GPU당 비동기 토큰화·멀티모달 전처리 worker 수. 이미지 로딩 스레드도 같은 수로 제한 |
| `enable_chunked_prefill` | 긴 입력의 prefill을 여러 엔진 실행 단계로 분할 |
| `enable_prefix_caching` | 같은 접두부의 KV cache 재사용 |
| `enforce_eager` | `false`는 vLLM의 컴파일/CUDA graph 사용을 허용 |
| `async_scheduling` | `null`은 vLLM 호환성 판단과 기본 선택, 명시적 `true/false` 허용 |

애플리케이션은 GPU당 최대 `2 × max_num_seqs`개의 미완료 요청만 공급합니다. 큐에는 이미지 경로·프롬프트·작업 정보만 들어갑니다. 로딩된 이미지의 최대 보유 요청 수도 이 한도 안에 있고, 파일 로딩 동시성은 `renderer_num_workers`입니다. 완료된 GPU에 다음 요청을 먼저 넣은 뒤 부모 프로세스가 결과를 검증·저장합니다. 진행 순서가 바뀌어도 영상·장면 ID에 따라 저장합니다.

`--gpus N`은 **사용할 GPU 개수**이고 기본값은 1입니다. 각 GPU에 독립 엔진을 복제하며 tensor parallel 크기는 1입니다. 특정 GPU를 고를 때는 다음처럼 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m extraction extract-graph-scenes --model qwen --run-id "$RUN_ID"
CUDA_VISIBLE_DEVICES=0,2 python -m extraction summarize-graph --source qwen --run-id "$RUN_ID" --gpus 2
```

이미지 개수 한도는 `visual_evidence.num_keyframes`입니다. 같은 Scene의 이미지는 기존 순서로 한 요청에 전달합니다. 입력의 이미지 토큰까지 포함한 길이와 출력 토큰 상한의 합이 context 한도를 넘으면 작업 ID·필요 길이·한도를 포함한 오류로 중단합니다. 프롬프트·프레임·출력 상한을 자동 축소하지 않습니다. 시작 로그에서 실제 `max_model_len`과 `async_scheduling`을 확인합니다.

Greedy는 `temperature=0`, 출력 상한은 `max_tokens`로 전달합니다. 샘플링 설정과 요청별 seed를 전달하고, 모델 generation config가 파이프라인의 생성 설정을 덮어쓰지 않도록 합니다. 기존 EOS 목록을 종료 토큰으로 사용하며 반복 패널티는 **생성된 토큰에만** 한 번 적용합니다. 엔진 차이 때문에 출력의 글자 단위 일치를 보장하지 않습니다.

Graph 장면은 JSON Schema, 모든 Summary는 7줄 Grammar를 `SamplingParams.structured_outputs`로 전달합니다. 엔진의 구조 제약 backend는 `xgrammar`로 고정합니다. Graph Schema의 중첩 필드·enum·nullable은 검증하고, 개수·ID 참조는 의미 경고를 유지합니다. Summary는 빈 값이 가능하지만 모든 값이 비면 실패합니다. 구조 제약 초기화·컴파일 실패를 자유 생성으로 전환하지 않습니다. [vLLM 0.28 Structured Output API](https://docs.vllm.ai/en/v0.28.0/features/structured_outputs/)를 기준으로 합니다.

## 진행률과 ETA

`extract-graph-scenes`(Qwen/Gemini), `extract-description-scenes`는 이번 실행의 미완료 장면 수를, `summarize-graph`/`summarize-description`은 미완료 요약 요청 수를 진행 바의 분모로 사용합니다. 분모는 실행 시작 시 고정되며, `scene`/`summary` 단위를 붙입니다. 분자는 이번 실행의 성공과 실패를 합한 처리 수입니다. 예를 들어 `382/47597 scene`은 이번 처리 대상 47,597개 장면 중 382개를 처리했다는 뜻이며, 영상 수가 아닙니다. 재사용 결과와 요약할 장면이 없는 영상은 시작 로그에서 확인할 수 있고 요청 수에서는 제외합니다. ETA는 이번 처리 시도가 끝나는 예상 시간이며, 실패가 모두 복구되는 시간은 아닙니다.

진행 바는 20칸으로 표시하고, 경과 시간·ETA·`success`/`failed`만 남깁니다. Qwen은 출력 토큰 처리량을 `tok/s`로 추가 표시합니다. 경과 시간에는 초기화 시간이 포함됩니다. Qwen은 모든 GPU 엔진이 준비된 뒤부터 처리량을 측정합니다. 먼저 준비된 GPU는 계속 요청을 처리하며 다른 GPU를 기다리지 않습니다. Gemini는 요청 공급 시점부터 측정합니다.

ETA는 최근 최대 180초의 완료 요청 수를 실제 경과 시간으로 나눈 처리량으로 계산합니다. 측정 60초 경과 및 최근 구간의 완료 8개 이상일 때 표시하며, 이전에는 `ETA=initializing` 또는 `ETA=estimating`을 표시합니다. 처리량은 30초마다 갱신하고 완료가 없는 대기 구간도 포함합니다. 표본이 부족하거나 최근 구간에 완료가 없으면 다시 추정 중으로 표시합니다. 출력은 단일 ETA이며 변동 범위는 표시하지 않습니다. 정상 종료 시 `ETA=00:00`, 중단·오류 시 `ETA=--`를 표시합니다.

## 재개와 실행 이력

기존 정상 결과와 명시적인 `raw_fallback`은 재사용하고 실패·누락 항목을 처리합니다. 정상 스키마는 유지하며 Raw는 별도 스키마입니다. `[1.00, 1.05, 1.10, 1.15, 1.20]` 순서로 생성하고 각 응답을 검증·Repair한 뒤 실패 task만 다음 penalty로 제출합니다. 모델 pool은 Step 안에서 계속 사용합니다. 전부 실패하면 마지막 비어 있지 않은 원문을 Raw로 저장하고, 모두 비면 최종 실패로 남깁니다. Description 장면은 자유 텍스트이며 빈 응답만 재시도합니다. [전체 처리 규칙](../README.md)을 참고하세요.

각 출력 디렉터리의 `.recovery`에 입력·생성 정책 hash, task ID, 재시도 묶음과 시도 이력을 원자적으로 저장합니다. 시도에는 Repair 전 원문, penalty, 검증 오류와 Repair 모드, 출력 토큰 수, `finish_reason`·`stop_reason`을 기록합니다. 응답을 저장한 다음 정식 산출물을 게시하므로 게시 중 오류는 저장된 응답으로 재개합니다. 재시도 중 중단은 남은 시도부터, 최종 실패 후 재실행은 새 묶음의 첫 penalty부터 시작합니다. `--force`는 해당 Step 전체에 새 묶음을 시작합니다. 정상 산출물은 생성 방식 변경만으로 무효화하지 않습니다.

`artifacts/$RUN_ID/extraction/qwen_runtime.jsonl`은 부모 프로세스만 추가 기록합니다.

- `engine_ready`: 실행 ID, 단계, 모델 경로, GPU, 라이브러리 버전, 실제 적용 설정.
- `result`: 저장 후 단계·작업 ID·실행 ID·상태·결과 해시와 토큰/종료 정보. 저장된 응답으로 게시를 재개하면 `replayed_from_execution_id`와 `original_engine`으로 원래 실행을 구분합니다.
- 정상 산출물의 현재 해시와 일치하는 성공 이력이 없으면 `legacy_unknown`으로 집계합니다. 옛 결과가 Transformers에서 실행됐다고 추정하지 않습니다.

마지막 줄의 불완전한 append만 재실행 시 제거합니다. 중간 줄 손상 또는 구조가 잘못된 레코드는 오류입니다. 결과 저장 후 이력 저장 전에 중단됐다면 결과는 재사용되지만 이력은 `legacy_unknown`일 수 있습니다. Ctrl+C, 엔진 종료, OOM, 저장 오류 시 이미 저장한 결과를 보존하고 워커와 vLLM 하위 프로세스를 정리합니다. 엔진 오류는 숨기지 않고 중단하므로 원인을 해결한 뒤 같은 명령으로 재개합니다.

## Linux 실측

Windows CPU/모의 엔진 테스트는 CUDA 호환성·속도·내용 품질을 증명하지 않습니다. 아래 도구로 Linux에서 실제 이미지와 텍스트를 각각 측정합니다. 운영 artifact는 수정하지 않으며, 고정 요청 파일과 별도 보고서만 생성합니다. 측정 도구는 **전용 GPU 한 장**을 사용합니다. 다중 GPU 분배 동작은 통합 테스트와 실제 단계 실행에서 확인합니다.

```bash
python -m pip install -e ".[qwen,dev]"
python -m pip check
python -c "import torch, vllm; print(vllm.__version__, torch.cuda.get_device_name(0))"

# 기준 요청을 한 번 고정합니다. 앞의 16개는 warmup, 나머지는 측정용입니다.
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-scenes --limit 144 --output /tmp/qwen-scenes.json
python -m benchmarks.qwen_benchmark export --run-id "$RUN_ID" --stage graph-summary-qwen --limit 144 --output /tmp/qwen-summaries.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --backend vllm --output /tmp/scenes-vllm-32-16384.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-summaries.json --backend vllm --output /tmp/summary-vllm-32-16384.json
```

장면 JSON이 준비된 뒤 요약 요청을 export합니다. `description-scenes`, `description-summary`, `graph-summary-gemini`도 지원합니다. 실제 요청 수가 16개 이하라면 `--warmup 1` 등을 지정해 측정 요청을 남깁니다. 같은 요청 파일을 모든 비교에서 사용하고, 이미지가 바뀌면 해시 검사가 실패합니다. 샘플링 모드에서 운영처럼 seed가 `None`이면 확률적 결과 차이도 발생합니다. 출력 파일이 이미 있으면 덮어쓰지 않습니다.

새로 export하는 요청은 현재 Structured Output 제약과 첫 penalty를 포함합니다. 이 도구는 고정 요청 1회 생성의 처리량을 측정하며 Step의 Repair/재시도/Raw 게시를 실행하지 않습니다. 실제 Linux/CUDA 검증에서는 Graph·Summary를 각각 여러 요청으로 export한 뒤 아래처럼 1보다 큰 penalty와 제약을 동시에 확인할 수 있습니다. Windows 테스트만으로 실제 컴파일·모델 출력이 검증된 것은 아닙니다.

```bash
python -c 'import json; from pathlib import Path; p=Path("/tmp/qwen-scenes.json"); d=json.loads(p.read_text()); [r.update(repetition_penalty=1.10) for r in d["requests"]]; p.with_name("qwen-scenes-structured-110.json").write_text(json.dumps(d))'
python -c 'import json; from pathlib import Path; p=Path("/tmp/qwen-summaries.json"); d=json.loads(p.read_text()); [r.update(repetition_penalty=1.10) for r in d["requests"]]; p.with_name("qwen-summaries-structured-110.json").write_text(json.dumps(d))'
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes-structured-110.json --backend vllm --output /tmp/scenes-structured-110-report.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-summaries-structured-110.json --backend vllm --output /tmp/summaries-structured-110-report.json
```

변경 전 Transformers 구현은 `benchmarks/qwen_transformers_reference.py`에 보존했습니다. 이 구현은 구조 제약을 지원하지 않으므로 **과거의 제약 없는 요청 파일**로만 실행합니다. 구조 제약이 있는 파일은 오류로 거부하며 제약을 제거해서 실행하지 않습니다. 기준 환경에 `accelerate`가 필요하며, 가능하면 이전 실행 환경을 유지하고 보고서의 라이브러리 버전도 함께 비교합니다. 아래 `legacy-*` 파일은 변경 전에 export해 둔 파일입니다.

```bash
python -m pip install accelerate
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/legacy-qwen-scenes.json --backend transformers --output /tmp/scenes-transformers.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/legacy-qwen-summaries.json --backend transformers --output /tmp/summary-transformers.json
```

이후 `max_num_seqs`는 16/32/64, 토큰 예산은 8192/16384/32768을 비교합니다. 예를 들어 기본값 측정 뒤 아래처럼 한 축씩 바꿉니다. 매번 별도 프로세스를 실행해 모델 로딩·컴파일과 steady-state 시간을 분리합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --max-num-seqs 16 --max-num-batched-tokens 16384 --output /tmp/scenes-16-16384.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --max-num-seqs 64 --max-num-batched-tokens 16384 --output /tmp/scenes-64-16384.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --max-num-seqs 32 --max-num-batched-tokens 8192 --output /tmp/scenes-32-8192.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --max-num-seqs 32 --max-num-batched-tokens 32768 --output /tmp/scenes-32-32768.json
```

보고서에는 `startup_seconds`, `warmup_seconds`, `steady_seconds`, 요청/s, 출력 토큰/s, 미완료 요청 수, 유효하지 않은 출력 수, 실패율과 전체 응답 텍스트를 저장합니다. `peak_gpu_memory_mib`는 엔진 하위 프로세스까지 포함한 `nvidia-smi` 0.5초 간격 관측 최대값이며 순간 피크를 놓칠 수 있습니다. startup에는 로딩·컴파일을, warmup에는 첫 요청의 추가 초기화를 포함합니다. 검증·보고서 저장은 steady 시간에서 제외합니다. OOM 등 실패 시에도 수집된 결과와 실패 단계를 보고서에 남깁니다.

같은 `requests_hash`끼리 비교하고, 실패율 0과 충분한 context 길이를 확인한 설정 중 장면·요약 각각의 처리량을 비교합니다. JSON·요약 구조 검사와 별도로 `task_id`별 실제 내용 차이를 검토합니다. 동일 입력·생성 조건에서 안정성과 내용 검토를 마친 뒤 측정 결과를 운영 YAML에 반영합니다. 이 저장소 변경 시점에는 RTX 6000 Ada의 실제 처리량 및 최적값을 측정하지 않았습니다.

API 기준: [vLLM 0.28.0 AsyncLLM](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/engine/async_llm.py), [모델 설정](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/model.py), [배치 logits processor](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/logits_processor/interface.py).
