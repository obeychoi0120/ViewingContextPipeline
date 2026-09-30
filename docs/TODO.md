# 프로젝트 구현 이정표

이 문서는 프로젝트 공통 구현 상태와 다음 단계를 관리합니다. Run별 분석 코드·표·보고서는 `artifacts/runs/<RUN_ID>/reports/`에 저장하고 여기에는 링크만 남깁니다.

| 우선순위 | 항목 | 상태 | 선행 조건 | 완료 기준 |
|---|---|---|---|---|
| P0 | Graph 입력 준비, 역할별 encoder, mean/attention, SASRec 공동 학습, 진단 | 구현·검증 완료, 전체 실험 대기 | v7 Scene Graph, BGE 로컬 가중치 | 입력·구조·gradient·재개·진단 CPU 검사와 GPU smoke 통과. 전체 실험 결과는 별도 확인 |
| P1 | 장면 간 GNN aggregation | 다음 단계 | 장면 연결 기준과 시간 정보 사용 결정 | 동일 데이터·seed에서 mean/attention과 비교, 시간·연결 ablation 포함 |
| P2 | 사용자 평균 aggregation | 대기 | 직접 Graph 영상 표현 기준선 확보 | 같은 후보·이력·평가 분할로 SASRec과 비교 |
| P2 | LightGCN·clustering 기반 사용자 표현 | 대기 | 사용자–아이템 학습 구간과 clustering 갱신 규칙 결정 | test leakage 없이 학습·추론·비교 재현 |
| P2 | text/graph·mean/attention 자동 paired 비교와 구조 제거 실험 | 대기 | 모드별 결과 확보, 비교 family 사전 정의 | 일치하는 사용자·날짜·seed 검증 및 paired CI 제공 |
| P1 | 실제 GPU 시간·메모리 측정과 병목 최적화 | 배치 처리 최적화 완료, RTX 전체 측정 대기 | 대상 RTX 6000 Ada 4장과 전체 입력 준비 | BGE 준비·selection·refit·평가 단계별 시간/peak 메모리 기록, 정확도 유지 확인 |

- 사용법: [직접 Graph 학습](graph_training.md)
- `260928_v7` 검증 기록: [구현 검증 보고서](../artifacts/runs/260928_v7/reports/graph_implementation.md)

- `260928_v7` 실행 최적화: [변경 사항·수치 검증·벤치마크](../artifacts/runs/260928_v7/reports/graph_execution_optimization.md)

- Graph 모델 v3: 1층·장면 384·제목 128 결합과 추천 단계별 프로파일링 구현. 실제 추천 성능/RTX 처리량 비교는 대기. [현재 모델·프로파일링 사용법](graph_training.md#graph-모델-v3-1층-encoder와-128384-결합-2026-09-30).
- `260928_v7` Graph v3·프로파일링 검증: [변경 위치·재학습 범위·검사 결과](../artifacts/runs/260928_v7/reports/graph_v3_profiling_implementation.md).

- 모델 v4: Meta 제목 512 기준선, Text 제목·Summary 독립 BGE, 공통 최종 LayerNorm, 아이템 MLP 제거. 전체 추천 성능 실험은 대기. [구현·검증 보고서](../artifacts/runs/260928_v7/reports/model_v4_implementation.md).
