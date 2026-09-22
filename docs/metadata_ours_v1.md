# V1 Metadata + Ours Graph (LoRA)

사내 저장소의 `807afdbc90e7da6703ecdb12017c63c060100dad`를 기준으로 이 실험에 필요한 기능만 추가한다.
여기서 **V1은 Graph/요약 실험 버전**이다. 전체 데이터의 7일 rolling 평가는 기존 `viewing-context-config/v4`를 사용한다.

## 재현 대상

| Arm | NDCG@10 | HR@10 | Metadata 대비 NDCG@10 |
| --- | ---: | ---: | ---: |
| Metadata + Ours v1 Graph (LoRA) | 0.03694 | 6.93% | +3.10% |

2026-09-21 검증한 기존 관측값이다. 이 변경으로 전체 실험을 다시 실행해 얻은 값은 아니다.

- Graph: Qwen3-VL-2B-Instruct에 LoRA 2.1을 병합한 **merged checkpoint**.
- Graph 프롬프트: `config/prompts/graph_scene_v2.md`의 이벤트형 스키마.
- Summary: **base Qwen3-VL-2B-Instruct**, `config/prompts/graph_summary_v3.md`.
- BGE 입력: 아래 문자열 전체를 BGE-large-en-v1.5로 한 번 인코딩한다. 결과는 1,024차원이다.

```text
Title: {영문 title}

Visual context:
{LoRA Graph를 base Qwen으로 요약한 텍스트}
```

기존 SASRec의 구조·학습·epoch 선택·후보 마스킹·집계는 그대로 사용한다.
콘텐츠 19,738개, 장면 115,947개, 사용자 100,000명, interaction 719,405개이며
2022-09-05~11의 7일 × seed 42/43/44 = 새 Arm 21개 조합이다.

## 준비

Ubuntu/Bash, 저장소 루트 기준이다. 기본 설치와 데이터 경로 설정은 [README](../README.md)를 따른다.
원본 영상, 데이터, base Qwen/BGE checkpoint, merged LoRA checkpoint는 별도로 준비해야 한다.

1. `config/pipeline.yaml`의 `data` 경로를 실험 데이터에 맞춘다.
   특히 `data.titles_csv`에는 당시 사용한 영문 title을 지정한다.
   기존 기록은 `MicroLens-100k_title_en.csv`를 사용했고, 빈 title은 기존 `zero_vector` 정책을 따른다.
   README의 title 보완 절차로 만든 다른 title을 사용하면 같은 입력이 아니다.
2. `models.qwen`에는 **base 모델**을, `models.bge`에는 BGE 모델을 지정한다.
   LoRA는 아래 추출 명령의 `--qwen-model-path`로만 전달한다.
3. V1 프롬프트와 기본 추천 설정을 유지한다. 기록된 추출 환경은 vLLM 0.28.0,
   torch 2.13.0, transformers 5.17.0이며, GPU별 요청 수 32,
   `extraction.qwen.gpu_memory_utilization: 0.85`, `async_scheduling: true`였다.
   `max_model_len: auto`는 기존 단계별 길이 계산을 사용한다.
4. 새 run ID를 사용한다. 기존 vanilla Graph run에 LoRA 추출을 이어서 실행하면
   완료 장면 캐시가 재사용되므로 두 모델의 결과가 섞일 수 있다.

`--qwen-model-path`는 `extract-graph-scenes --model qwen`에서만 허용된다.
해당 호출의 모델과 로그만 바뀌고 YAML과 다음 요약 호출의 base 모델은 유지된다.
adapter만 있는 디렉터리가 아니라 병합을 완료한 모델 디렉터리를 전달해야 한다.

## 실행

```bash
RUN=Metadata_Ours_V1
LORA_MODEL=/path/to/LoRA_weight/2.1/merged

python -m validation prepare-cohort --run-id "$RUN"
python -m extraction prepare-input-data --run-id "$RUN"
python -m extraction extract-graph-scenes --run-id "$RUN" --model qwen --qwen-model-path "$LORA_MODEL" --gpus 2
python -m extraction summarize-graph --run-id "$RUN" --source qwen --gpus 2

CUDA_VISIBLE_DEVICES="" python -m validation embed-representations --run-id "$RUN" --target METADATA_GRAPH_QWEN
CUDA_VISIBLE_DEVICES="" python -m validation run-recommendation --run-id "$RUN" --target METADATA_GRAPH_QWEN
python -m validation run-diagnosis --run-id "$RUN" --target METADATA_GRAPH_QWEN
```

이 행의 기존 BGE/SASRec 평가는 CPU에서 실행됐다. 위 명령도 CPU를 선택한다.
GPU 평가가 필요하면 `CUDA_VISIBLE_DEVICES=""`를 제거하고 추천 단계에 기존 GPU 옵션을 사용할 수 있지만,
실행 환경 차이로 수치가 완전히 같지 않을 수 있다.

장면 추출 실패가 남으면 **같은 LoRA 옵션으로** 추출 명령을 재실행한다.
기존 성공 장면은 보존하며 실패·미완료 장면을 복구한다. 기록된 V1은 최종 실패 장면 0개,
완료 요약 19,738개였다. 기본 95% 진단 통과만으로 이 조건을 대신하지 않는다.

임베딩·추천 명령도 같은 target으로 재실행하면 완료 결과를 재사용한다.
title이나 요약이 바뀌면 임베딩 입력 해시가 달라져 다시 생성하며,
장면 변경 후 아직 요약을 갱신하지 않았다면 임베딩을 거부한다.

동일 run에서 Metadata 대조군도 새로 평가해 상대 향상률을 계산하려면:

```bash
CUDA_VISIBLE_DEVICES="" python -m validation embed-representations --run-id "$RUN" --target METADATA
CUDA_VISIBLE_DEVICES="" python -m validation run-recommendation --run-id "$RUN" --target METADATA
python -m validation run-diagnosis --run-id "$RUN" --target METADATA METADATA_GRAPH_QWEN
```

이때 총 조합은 42개다. 별도의 기존 대조군을 재사용한다면 먼저 catalog, events,
metadata titles와 날짜별 split이 같은지 확인해야 한다.

## 결과 확인

`artifacts_root` 기본값에서는 다음 파일을 확인한다.

- `artifacts/$RUN/validation/representations/graph_qwen_metadata_embeddings.npz`
- `artifacts/$RUN/validation/diagnosis/diagnosis.json`

진단 JSON의 `runtime_decision.status`가 `pass`이고,
`recommendations.means.SASRec_METADATA_GRAPH_QWEN`에서 `NDCG@10`과 `HR@10`을 읽는다.
상대 향상률은 `(결합 NDCG@10 / Metadata NDCG@10 - 1) × 100`이다.

기존 결과의 반올림 전 값:

| 값 | 기록 |
| --- | ---: |
| 결합 NDCG@10 | 0.03694211966445207 |
| 결합 HR@10 | 0.06931704092534996 |
| Metadata NDCG@10 | 0.03583306162718162 |
| 결합 입력 최대 BGE 토큰 수 | 255 / 512 |
| 잘린 결합 입력 | 0 / 19,738 |

원래 입력을 대조할 때 사용할 SHA-256:

| 파일 | SHA-256 |
| --- | --- |
| `metadata_titles.jsonl` | `931b8c352c725c8bc73de3946f9347be9a8ce7ab927ad32a4b8caf8cb78f107f` |
| `graph_scene_v2.md` (LF) | `832eda946a3cb8cd8d7678b06f4bb8f3751fbeb0980fd10342870668cc898e7b` |
| `graph_summary_v3.md` (LF) | `e9d7fe16384bec2345dc80ae97ce27b372b11c5d6567ece91efbd6ba3a89cda6` |

완전히 같은 관측값을 재평가하려면 당시 cohort와 생성된 LoRA Graph/요약, BGE 입력,
모델 및 라이브러리 버전까지 같아야 한다. 기존 전체 실행은 먼저 완성한 1,000개 콘텐츠의
Graph/요약을 재사용했으므로 새 추론의 결과가 바이트 단위로 같다고 보장하지 않는다.
위 숫자는 결과를 강제로 맞추기 위한 상수가 아니며 실행 코드에서 사용하지 않는다.

## 변경 범위

추가한 것은 추출 호출의 모델 경로 선택, 명시적 `METADATA_GRAPH_QWEN` target,
Title+Graph 텍스트 결합과 의존성 검사, 해당 Graph의 장면·복구 진단 연결이다.
target을 생략하면 기존 네 Arm을 실행한다.
SASRec, rolling 학습, BGE, V1 프롬프트, 기본 YAML은 기준 커밋의 내용을 유지한다.
