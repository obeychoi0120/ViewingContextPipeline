# 프로젝트 구현 이정표

마지막 갱신: **2026-10-07**. 현재 코드·작업 트리와 저장된 검증·진단 기록을 기준으로 정리합니다. 구현 완료, 소형 검증 완료, 전체 데이터 실험 결과를 구분합니다.

Run별 분석 코드·표·보고서는 `artifacts/runs/<RUN_ID>/reports/`에 저장하고 여기에는 링크만 남깁니다. 설계와 단계별 입출력은 [README](../README.md), 자문 원문은 [연구실 피드백](feedback_260923.md), 실행·모델 계약은 [직접 Graph 학습](graph_training.md)과 [실행·저장 계약](runtime.md)을 참고하세요.

## 현재 구현·문서 상태

| 항목 | 현재 상태 | 근거·확인 범위 |
| --- | --- | --- |
| Text·직접 Graph 추천 경로 | 구현·소형 CPU/CUDA 검증 완료 | Graph 입력 준비, Entity→Action 역할 결합, Scene MLP, mean, SASRec 공동 학습·진단. 실제 성능은 해당 모델 계약의 Run별 결과로 확인. [모델·데이터 계약](graph_training.md#모델-및-데이터-계약) |
| BGE Large·현재 영상 encoder | 구현·소형 검증 완료 | 동결 BGE 1024차원, 독립 projection·1층 Item/Scene Residual MLP, 최종 영상 512차원. Item/Scene LayerNorm은 없고 SASRec/User MLP 내부 LayerNorm은 유지. Text는 `sasrec-content-v8`, mean은 `sasrec-role-graph/v9`. 기존 2층 User MLP는 유지 |
| 영상 Residual MLP 1층 전환 | 구현·CPU/CUDA 소형 검증 완료; 전체 시간·품질 비교 대기 | Item·Scene은 `x + ReLU(Linear(x))`; 512차원 블록 파라미터 525,312→262,656개. CPU validation 105개, L4 CUDA 14개, CLI·README 28개 통과. 입력 캐시 재사용, 추천 selection부터 재학습. [변경·재실행·검증](graph_training.md#영상-residual-mlp-1층-전환-2026-10-07) |
| Graph v8 프롬프트·receiver 제거 | 구현·회귀 검증 완료 | 추출·Summary·직접 Graph에서 receiver 제외. 텐서 슬롯은 형태 호환성을 위해 유지. 현재 Graph 입력 계약은 `role-graph-inputs/v3`. 실제 추출 정확도·추천 효과는 별도 평가. [프롬프트 변경 이력](../prompts/update_note.md) |
| DCGN 참고 장면 similarity graph·1층 residual GCN | 구현·소형 CPU/CUDA 검증 완료; 전체 데이터 비교 대기 | 제목 제외 384차원 cosine, 영상 내부 전체 장면 쌍, 512차원 전달. 현재 gnn 계약은 `sasrec-role-graph/v10`. 수학·순열·영상 격리·gradient·chunk/checkpoint·캐시·진단 검증. [GCN 구현·검증](graph_training.md#장면-gcn-aggregation-2026-10-06) |
| 실행 최적화·프로파일링 | packed 입력·GPU 특징/catalog cache·chunk/checkpoint·계측 구현 완료 | 기존 Run의 일부 selection 프로파일과 소형 CPU/L4 benchmark 기록 있음. 현재 모델의 RTX 전체 시간·메모리 측정은 대기. [실행 최적화 기록](../artifacts/runs/260928_v7/reports/graph_execution_optimization.md), [프로파일 분석](../artifacts/runs/260930_v8/reports/recommendation_profile_analysis.md) |
| 진단 포맷·Run 간 paired 비교 | 구현·소형 검증 완료 | Text/Graph 진단은 `rolling-diagnosis/v3`. 같은 경로의 Graph Arm Run 비교를 지원하며 직접 Graph는 동일 aggregation을 사용. 교차 Text/Graph·mean/gnn 비교는 후속 항목. [진단 포맷 기록](../artifacts/runs/260930_v8/reports/diagnosis_format_update.md), [Run 비교 코드](../src/validation/run_comparison.py) |
| 연구실 자문용 README·다이어그램 안내 | 재작성 완료 | 피드백 반영 범위, 8단계 입출력, 코드 연결, 11개 접기/펼치기. mean 전체 흐름과 GNN 모델 그림의 범위를 구분. 공개 명령 18개 dispatch·링크/앵커·Graph 예시 파싱 검사 완료. [README](../README.md) |
| 핵심 시나리오 중심 테스트 정리 | 현재 작업 트리에 반영·정리 당시 CPU 전체 검사 완료 | 함수 233→120개, 정리 당시 272개 사례 통과. README의 GNN 명령 2개 추가 후 현재 274개 수집 확인; 274개 전체 실행은 이번 문서 갱신에서 수행하지 않음. 과거 v7 고정 해시 검사는 제거하고 정상 결과 재사용·원본 보존 검사는 유지. [정리 기록](test-cleanup-2026-10-06.md) |

현재 Scene aggregation은 `mean`·`gnn`입니다. 과거 Scene attention pooling 구현·참조 테스트는 제거했으며, SASRec 내부 attention과 과거 attention Run 산출물은 유지합니다. UniMP·CSA는 현재 GCN 구현에 포함하지 않습니다.

## 저장된 실험 결과와 현재 모델의 구분

Run 이름의 v8, 프롬프트 버전, 모델 architecture 버전은 서로 다른 식별자입니다. 아래 조합 수는 저장된 진단의 날짜 × seed × Arm 수이며, 이 결과를 현재 Text v8·mean v9·gnn v10 모델의 실험 완료로 표시하지 않습니다.

| Run | 저장된 진단과 모델 계약 | 해석 범위·보고서 |
| --- | --- | --- |
| `260930_v8` | Text v5: 126조합; Graph mean v5: 84조합; 과거 Graph attention v5: 84조합 | 각 경로 내부 기준선으로 비교한 과거 모델 결과. Text/Graph Metadata 값이 달라 경로 간 차이를 표현만의 효과로 단정하지 않음. [성능 비교](../artifacts/runs/260930_v8/reports/diagnosis_baseline_comparison.md) |
| `261002_v8_large` | Text v6: 126조합; Graph mean v6: 63조합 | Graph mean에는 `meta` 실측값이 없음. Text Metadata와의 차이는 경로 간 참고 비교이며 유의성 검정은 제공되지 않음. [성능 비교](../artifacts/runs/261002_v8_large/reports/diagnosis_baseline_comparison.md) |

위 두 Run은 진단의 실제·기대 이벤트 수 일치를 확인했으며 GNN 진단 파일은 없습니다. 기존 문서 갱신에서 Text/mean v7·GNN v8의 전체 추천 결과는 확인하지 못했습니다. 이번 1층 MLP 계약(Text v8·mean v9·gnn v10)의 전체 실험도 실행하지 않았으므로 대기로 유지합니다. 사내 서버의 실행 진행 상태는 이번 갱신에서 확인하지 않았습니다.

기존 `260930_v8` 프로파일은 과거 Graph v4·mean의 한 날짜·seed·Arm selection 일부를 다룹니다. 전체 Run, 현재 GNN, refit/test 처리량으로 일반화하지 않습니다. [배치 프로파일 분석](../artifacts/runs/260930_v8/reports/recommendation_profile_analysis.md) · [projection 연산 분석](../artifacts/runs/260930_v8/reports/projection_operator_analysis.md)

## 남은 작업

기존 연구·실행 항목의 우선순위를 유지합니다. 실험 설계나 대표 주장에 영향을 주는 조건은 아래 선행 조건을 먼저 확정합니다. 이 목록의 갱신은 새 실험을 실행하거나 설계 선택을 채택한 것을 뜻하지 않습니다.

| 우선순위 | 항목 | 상태 | 선행 조건 | 완료 기준 |
| --- | --- | --- | --- | --- |
| P0 | 현재 Text v8·Graph mean v9 전체 추천 학습·진단 | 저장소에서 현재 계약의 결과 미확인 | 현재 BGE Large·입력 계약 확인, 보존할 기존 결과와 분리한 Run | 선택한 날짜·seed·Arm의 완료 manifest·test 지표·진단 검증, Run별 보고서 저장 |
| P1 | 장면 GCN v10 전체 실험·mean 비교 | 소형 검증 완료, 전체 비교 대기 | 유효한 현재 Graph 입력, 동일 cohort·평가 구간·seed·공통 학습 조건 | mean/gnn 추천 품질·비용 결과 보관. 통계 비교는 사전 정의한 비교군과 paired 집계 기준 적용 |
| P1 | 실제 GPU 시간·메모리 측정과 병목 최적화 | 과거 일부 프로파일·소형 benchmark 있음; 현재 RTX 전체 측정 대기 | 대상 RTX 6000 Ada 4장, 전체 입력, 모델·동시 작업 수·계측 조건 기록 | BGE 준비·selection·refit·test 단계별 시간/peak 메모리 기록, 최적화 전후 수치·학습 동등성 확인. 처리량은 profiler 영향을 분리해 측정 |
| P2 | Text/Graph·mean/gnn 자동 paired 비교와 구조 제거 실험 | 대기; 동일 경로 Run 비교는 구현됨 | 모드별 결과 확보; 비교군·구조 제거 범위 사전 결정 **[직접 결정 필요]** | 사용자·이벤트·날짜·seed 일치 검증과 paired CI, 구조 변경·재학습 범위를 포함한 보고서 |
| P2 | UniMP 참고 Graph Transformer attention·gated residual | 후속 대기 | GCN 기준선 결과; attention 재도입·비교 조건 결정 **[직접 결정 필요]** | cosine 기반 GCN과 학습 attention 비교. 라벨 전파는 별도 라벨·누수 정책 확정 필요 |
| P2 | CSA 참고 attention 보조 학습 | 후속 대기 | attention 경로; 반사실 attention·추천 목적함수·추가 계산량 정의 **[직접 결정 필요]** | 보조 학습 유무의 추천 품질·비용 비교. 실제 사용자 인과성 주장과 구분 |
| P2 | 계층적 DCGN | 후속 대기 | 장면 GCN 기준선; 시간 convolution·pooling 계층 결정 **[직접 결정 필요]** | 장면 간 전달과 시간·계층 구조의 효과를 분리한 비교 |
| P2 | 사용자 평균 aggregation | 대기 | 직접 Graph 영상 표현 기준선 확보 | 같은 후보·이력·평가 분할로 SASRec과 비교 |
| P2 | LightGCN·clustering 기반 사용자 표현 | 대기 | 사용자–아이템 학습 구간·clustering 갱신 규칙 결정 **[직접 결정 필요]** | test leakage 없이 학습·추론·비교 재현 |

## 이전 구현 이력

아래는 당시의 구조·검증 기록입니다. 현재 지원 구조와 남은 실험은 위 표를 기준으로 확인합니다. 기존 보고서·프롬프트·Run 산출물은 이동하거나 수정하지 않습니다.

- `260928_v7` Graph 입력·학습 구현: [구현 검증 보고서](../artifacts/runs/260928_v7/reports/graph_implementation.md).
- `260928_v7` 실행 최적화: [변경 사항·수치 검증·벤치마크](../artifacts/runs/260928_v7/reports/graph_execution_optimization.md).
- Graph 모델 v3: 1층 encoder·장면 384·제목 128 결합과 추천 단계별 프로파일링 구현. 당시 실제 추천 성능·RTX 처리량 비교는 대기. [이전 모델 v3 이력](graph_training.md#이전-graph-모델-v3-1층-encoder와-128384-결합-2026-09-30) · [변경 위치·재학습 범위·검사 결과](../artifacts/runs/260928_v7/reports/graph_v3_profiling_implementation.md).
- 모델 v4: Meta 제목 512 기준선, Text 제목·Summary 독립 BGE, 공통 최종 LayerNorm, 아이템 MLP 제거. 당시 전체 추천 성능 실험은 대기. [구현·검증 보고서](../artifacts/runs/260928_v7/reports/model_v4_implementation.md).
- Small v5: BGE 384차원·학습 최적화의 과거 구현 검증. 현재 설정은 BGE Large 1024차원입니다. [당시 구현 검증](../artifacts/runs/260930_v8/reports/small_implementation_validation.md).
- v3~v6와 Small v5 설명은 과거 이력으로 보존합니다. 2층 영상 MLP(Text/mean v7·GNN v8)는 과거 구조입니다. 현재 1층 MLP(Text v8·mean v9·gnn v10)의 구조·재실행 조건은 [직접 Graph 학습](graph_training.md)에 정리했습니다.
