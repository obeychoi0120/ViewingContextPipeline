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

## 재개와 실행 이력

기존 성공 결과는 재사용합니다. 장면 추출은 성공한 scene index만 완료로 보고 실패·누락 장면을 다시 요청합니다. 요약은 기존 검증·재시도 규칙을 따릅니다. 엔진 변경만으로 기존 산출물을 무효화하지 않으며, `experiment.json`과 장면·요약 JSON 스키마도 유지합니다. `--force`는 해당 단계 전체를 의도적으로 다시 생성할 때만 사용합니다.

`artifacts/$RUN_ID/extraction/qwen_runtime.jsonl`은 부모 프로세스만 추가 기록합니다.

- `engine_ready`: 실행 ID, 단계, 모델 경로, GPU, 라이브러리 버전, 실제 적용 설정.
- `result`: 저장 후 단계·작업 ID·실행 ID·상태·결과 해시와 토큰 수. `artifact_id`는 저장된 scene index를 기준으로 하며 제출 ID와 구분합니다.
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

변경 전 Transformers 구현은 `benchmarks/qwen_transformers_reference.py`에 그대로 보존했습니다. 같은 checkpoint와 입력으로 기존 실행 경로를 측정할 때만 사용합니다. 기준 환경에 `accelerate`가 필요하며, 가능하면 이전 실행 환경을 유지하고 보고서의 라이브러리 버전도 함께 비교합니다.

```bash
python -m pip install accelerate
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-scenes.json --backend transformers --output /tmp/scenes-transformers.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.qwen_benchmark run --requests /tmp/qwen-summaries.json --backend transformers --output /tmp/summary-transformers.json
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
