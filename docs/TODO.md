# 프로젝트 구현 이정표

이 문서는 프로젝트 공통 구현 상태와 다음 단계를 관리합니다. Run별 분석 코드·표·보고서는 `artifacts/runs/<RUN_ID>/reports/`에 저장하고 여기에는 링크만 남깁니다.

| 우선순위 | 항목 | 상태 | 선행 조건 | 완료 기준 |
|---|---|---|---|---|
| P0 | Graph 입력 준비, Entity→Action 역할 결합, Scene MLP, mean, SASRec 공동 학습·진단 | v7 구현·검증 완료, 전체 실험 별도 | 현재 Scene Graph·BGE Large | 고정 특징·역할·결측·재개·진단 검증. 실제 추천 성능은 Run별 확인 |
| P0 | BGE Large 1024 특징·현재 Text/Graph v7 모델 | 구현·CPU/CUDA 검증 완료, 전체 실험 별도 | 로컬 bge-large-en-v1.5 및 현재 Run 입력 | 현재 독립 projection·Residual MLP·결측 계약으로 재현 |
| P1 | DCGN 참고 장면 similarity graph + 1층 residual GCN | 구현·CPU/CUDA 검증 완료, 전체 실험 대기 | 제목 제외 384차원 cosine·영상 내부 전체 쌍·512차원 전달 | 수학·순열·영상 격리·gradient·chunk/checkpoint·캐시·진단 검증 완료. 전체 데이터에서 동일 조건 mean 비교 대기 |
| P2 | UniMP의 Graph Transformer attention·gated residual | 후속 대기 | GCN 기준선 결과; attention 재도입 및 비교 조건 결정 | cosine 기반 GCN과 학습 attention 비교. 라벨 전파는 별도 라벨·누수 정책 확정 필요 |
| P2 | CSA 참고 attention 보조 학습 | 후속 대기 | attention 경로; 반사실 attention·추천 목적함수·추가 계산량 정의 | 보조 학습 유무의 추천 품질·비용 비교; 실제 사용자 인과성 주장과 구분 |
| P2 | 계층적 DCGN | 후속 대기 | 장면 GCN 기준선; 시간 convolution·pooling 계층 결정 | 장면 간 전달과 시간·계층 구조의 효과를 분리한 비교 |
| P2 | 사용자 평균 aggregation | 대기 | 직접 Graph 영상 표현 기준선 확보 | 같은 후보·이력·평가 분할로 SASRec과 비교 |
| P2 | LightGCN·clustering 기반 사용자 표현 | 대기 | 사용자–아이템 학습 구간과 clustering 갱신 규칙 결정 | test leakage 없이 학습·추론·비교 재현 |
| P2 | text/graph·mean/gnn 자동 paired 비교와 구조 제거 실험 | 대기 | 모드별 결과 확보, 비교 family 사전 정의 | 일치하는 사용자·날짜·seed 검증 및 paired CI 제공 |
| P1 | 실제 GPU 시간·메모리 측정과 병목 최적화 | 배치 처리 최적화 완료, RTX 전체 측정 대기 | 대상 RTX 6000 Ada 4장과 전체 입력 준비 | BGE 준비·selection·refit·평가 단계별 시간/peak 메모리 기록, 정확도 유지 확인 |

- 사용법: [직접 Graph 학습](graph_training.md)
- `260928_v7` 검증 기록: [구현 검증 보고서](../artifacts/runs/260928_v7/reports/graph_implementation.md)

- `260928_v7` 실행 최적화: [변경 사항·수치 검증·벤치마크](../artifacts/runs/260928_v7/reports/graph_execution_optimization.md)

- Graph 모델 v3: 1층·장면 384·제목 128 결합과 추천 단계별 프로파일링 구현. 실제 추천 성능/RTX 처리량 비교는 대기. [이전 모델 v3 이력](graph_training.md#이전-graph-모델-v3-1층-encoder와-128384-결합-2026-09-30).
- `260928_v7` Graph v3·프로파일링 검증: [변경 위치·재학습 범위·검사 결과](../artifacts/runs/260928_v7/reports/graph_v3_profiling_implementation.md).

- 모델 v4: Meta 제목 512 기준선, Text 제목·Summary 독립 BGE, 공통 최종 LayerNorm, 아이템 MLP 제거. 전체 추천 성능 실험은 대기. [구현·검증 보고서](../artifacts/runs/260928_v7/reports/model_v4_implementation.md).

- 현재 mean/Text는 v7: 독립 projection·Residual MLP이며 Item/Scene LayerNorm은 없습니다. 신규 gnn 실험은 v8 계약을 사용합니다. [현재 구조·GNN 재실행](graph_training.md). v3~v6와 Small v5 설명은 과거 이력으로 보존합니다.
- Scene attention pooling의 과거 구현·참조 테스트는 제거했습니다. SASRec 내부 attention은 유지하며 과거 Run 산출물은 보존합니다. UniMP·CSA는 이번 GCN 구현에 포함하지 않습니다.
