**Graph_Qwen · Desc_Qwen · Metadata 비교 보고서**

실험 `Full_v1_260908` · 작성일 2026-09-14 · 분석 원본: [diagnosis.json](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json) · 스키마 `rolling-diagnosis/v1`

**이번 실험에서 추천 품질은 Desc_Qwen > Metadata > Graph_Qwen 순이다.** HR/NDCG@4·8·10·20·30의 10개 전체 평균 지표에서 같은 순서가 관측됐다. 주 지표 NDCG@10에서 Desc_Qwen은 Metadata보다 **2.58% 높고 보정된 우월성 기준을 통과**했다. Graph_Qwen은 Metadata보다 **19.22%**, Desc_Qwen보다 **21.26% 낮았으며**, Desc_Qwen 대비 상대 손실을 5% 이내로 허용하는 비열등성 기준을 통과하지 못했다.

품질을 우선한 후속 실험 기준 Arm으로는 Desc_Qwen이 적절하다. 다만 Metadata 대비 절대 개선은 NDCG@10 **+0.00092573**, HR@10 **+0.1484%p**로 작고, 날짜별 편차와 집계 가중치에 민감하다. Metadata는 성능이 근접하고 영상 표현 생성 단계가 없어 실용적인 기준선이다. 현재 Graph_Qwen을 Desc_Qwen 대신 사용할 추천 품질상의 근거는 부족하다. 이는 이번 데이터·기간·생성 방식의 결과이며, Graph 표현 전반이나 Qwen 모델 자체의 일반적인 우열을 뜻하지 않는다.

![전체 NDCG@10, Metadata 대비 보정 신뢰구간, 날짜별 NDCG@10 비교](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/reports/Full_v1_260908/arm_comparison.png)

그림의 막대는 공식 전체 평균, 오차 막대는 Metadata 대비 NDCG@10 차이의 Bonferroni 보정 양측 98.333% 신뢰구간이다. 날짜별 선은 seed 3개의 평균이며, 선의 높이 차이에 대한 날짜별 유의성 검정은 수행하지 않았다.

분석 범위와 Arm의 의미는 다음과 같다. `SASRec_DESC`는 사용자가 지칭한 **Desc_Qwen**의 저장 이름이다. Metadata 입력은 영문 title이다. Graph_Qwen은 Graph를 영상 요약 텍스트로 바꾼 뒤 임베딩하므로, 결과는 Graph→요약→임베딩→추천 경로 전체에 대한 평가다. Arm 정의와 공통 BGE/SASRec 구조는 [저장소 README](/home/junsu2.choi/workspace/ViewingContextPipeline/README.md:19)와 [실행 계약](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/full_rolling.md:78)으로 보충했다.

| 보고서 이름 | 진단의 Arm 이름 | 저장소에 정의된 표현 입력 |
| --- | --- | --- |
| Graph_Qwen | `SASRec_GRAPH_QWEN` | Qwen 장면 Graph → 영상 요약 텍스트 → BGE |
| Desc_Qwen | `SASRec_DESC` | Qwen 장면 Description → 영상 요약 텍스트 → BGE |
| Metadata | `SASRec_METADATA` | 영문 title → BGE |

**평가 원본은 사용자 100,000명, 아이템 19,738개, interaction 719,405건이다.** 평가일은 2022-09-05~2022-09-11의 7일이고 seed는 42·43·44다. 이번 선택 범위는 3 Arm으로, **7일 × 3 seed × 3 Arm = 63개 조합**이다. `SASRec_GRAPH_GEMINI`는 제외됐다. 선택 범위에서 `runtime_decision.status=pass`, 오류 목록은 비어 있고 `statistics.status=computed`다. 이 pass는 실행·진단 유효성 판정이며, 세 Arm의 가설이 모두 성립했다는 뜻은 아니다. 근거: [cohort](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:2), [실행 판정](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:1391).

평가기간의 원본 사건 57,531건 중 이전 이력이 없는 157건을 제외한 **57,374개 적격 사건**을 공통으로 평가했다. 이를 seed와 Arm별로 반복한 평가 행은 **516,366개 = 57,374 × 3 × 3**로 기대치와 실제치가 일치한다. Arm당 평가 행은 172,122개다. 516,366개를 서로 독립적인 고유 사건 수로 해석하면 안 된다. 사용자 100,000명도 원본 cohort 크기이며, 평가기간에 실제로 적격 사건을 가진 고유 사용자 수는 이 JSON에 따로 제시되지 않는다. 원본 전체의 무이력 사건 100,000건과 평가기간의 제외 157건은 집계 범위가 다르다.

공식 평균은 **각 날짜·seed의 사건 평균 → seed 균등 평균 → 7개 날짜 균등 평균**으로 계산한다. 날짜마다 사건 수가 달라도 최종 가중치는 1/7이다. 따라서 사용자별 점수를 단순 평균한 값이나 전체 사건을 한 번에 합친 평균과 다를 수 있다. 실행 계약상 평가일마다 이전 날짜의 데이터로 epoch 선택·refit 후 당일 test를 수행하며, 동일한 catalog와 이력 조건으로 비교한다. 근거: [시간 분할 계약](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/full_rolling.md:5), [집계·통계 계약](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/full_rolling.md:154).

아래는 공식 평균 10개 지표다. HR@K는 정답이 상위 K개 안에 포함된 비율이며, NDCG@K는 정답의 순위가 높을수록 큰 점수를 준다. **표의 점수는 모두 0~1 원척도**, 상대차는 `(비교 Arm / 기준 Arm − 1) × 100`이다. 예를 들어 Desc_Qwen HR@10의 0.068699는 약 **6.8699%**다. 근거: [원본 평균](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:1351), [지표 계산](/home/junsu2.choi/workspace/ViewingContextPipeline/src/validation/metrics.py:42).

| 지표 · 원척도 | Graph_Qwen | Desc_Qwen | Metadata | Desc / Metadata 상대차 | Graph / Metadata 상대차 |
| --- | --- | --- | --- | --- | --- |
| NDCG@4 | 0.020191 | 0.026754 | 0.026018 | +2.83% | -22.40% |
| NDCG@8 | 0.026767 | 0.034303 | 0.033463 | +2.51% | -20.01% |
| NDCG@10 | 0.028945 | 0.036759 | 0.035833 | +2.58% | -19.22% |
| NDCG@20 | 0.035873 | 0.044566 | 0.043477 | +2.50% | -17.49% |
| NDCG@30 | 0.040170 | 0.049234 | 0.047901 | +2.78% | -16.14% |
| HR@4 | 0.029590 | 0.038873 | 0.038003 | +2.29% | -22.14% |
| HR@8 | 0.048335 | 0.060388 | 0.059191 | +2.02% | -18.34% |
| HR@10 | 0.055706 | 0.068699 | 0.067215 | +2.21% | -17.12% |
| HR@20 | 0.083235 | 0.099701 | 0.097612 | +2.14% | -14.73% |
| HR@30 | 0.103440 | 0.121648 | 0.118410 | +2.73% | -12.64% |

Desc_Qwen의 NDCG@10은 **0.03675879**, Metadata는 **0.03583306**, Graph_Qwen은 **0.02894466**다. HR@10은 각각 **6.8699%, 6.7215%, 5.5706%**다. Desc_Qwen의 Metadata 대비 HR@10 상대 개선은 **+2.21%**이고, 절대 개선은 **+0.1484%p**다. Graph_Qwen의 Metadata 대비 HR@10 상대차는 **−17.12%**, 절대차는 **−1.1509%p**다. 상대 변화율과 퍼센트포인트 차이를 구분해야 한다.

Graph_Qwen의 Metadata 대비 손실은 상위 순위 구간에서 더 크다. HR@4 상대차는 **−22.14%**, HR@30은 **−12.64%**이며, NDCG@4는 **−22.40%**, NDCG@30은 **−16.14%**다. 추천 목록을 늘리면 상대 격차가 줄지만 모든 보고 cutoff에서 낮다. 이는 전체 평균의 기술적 비교이며, NDCG@10 외 지표의 통계적 유의성은 이 파일로 판정하지 않는다.

**통계 판정의 주 지표는 NDCG@10**이다. 사용자 단위 paired bootstrap 10,000회, bootstrap seed 42를 사용했다. 같은 사용자의 날짜·사건과 Arm 간 대응관계를 유지하며, seed 평균 후 날짜 균등 집계를 적용한다. 검정군별 familywise α는 0.05다. Metadata 우월성은 사전 정의된 3개 비교에 Bonferroni 보정을 적용해 비교당 α=0.016667, 양측 신뢰수준은 **98.333%**다. Graph–Description 비열등성은 사전 정의된 2개 비교에 대해 비교당 α=0.025, **단측 97.5% 하한**을 쓴다. Gemini가 제외되어도 보정 분모 3과 2는 유지돼 있다. 근거: [통계 및 보정 정책](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:1447).

| 비교 · NDCG@10 | 관측 효과 | 원본의 보정 구간 / 하한 | 판정 |
| --- | --- | --- | --- |
| Desc_Qwen − Metadata | +0.00092573 · 상대 +2.58% | 양측 CI [+0.00003455, +0.00182087] | 우월성 통과 · 하한 > 0 |
| Graph_Qwen − Metadata | −0.00688841 · 상대 −19.22% | 양측 CI [−0.00783290, −0.00596355] | 우월성 미통과 · CI 전체가 0 미만 |
| Graph_Qwen / Desc_Qwen − 1 | 상대 −21.26% | 단측 상대 하한 −22.9474% · 허용 경계 −5% | 비열등성 미통과 · 하한 ≤ −5% |

Desc_Qwen의 우월성은 보정 후에도 성립하지만, CI 하한은 **+0.00003455**로 0에 가깝다. 따라서 평균상 우위의 근거는 있으나 개선 폭이 크게 보장되지는 않는다. Graph_Qwen은 Metadata 대비 CI 전체가 음수이므로 낮은 NDCG@10을 뒷받침한다. Graph_Qwen–Desc_Qwen에는 단측 비열등성 하한이 저장돼 있으므로, 이를 임의의 양측 CI나 별도의 열등성 검정 결과로 바꾸어 제시하지 않는다. 명시적인 p값은 원본에 없다.

날짜별 NDCG@10을 보면 전체 평균의 차이가 일정하지 않다. 아래는 날짜 안에서 seed 3개를 균등 평균한 결과다.

| 평가일 | 적격 사건 | Graph_Qwen | Desc_Qwen | Metadata | Desc / Metadata 상대차 |
| --- | --- | --- | --- | --- | --- |
| 2022-09-05 | 6,220 | 0.027627 | 0.036915 | 0.033984 | +8.63% |
| 2022-09-06 | 6,452 | 0.027022 | 0.038580 | 0.033016 | +16.86% |
| 2022-09-07 | 6,579 | 0.031158 | 0.037398 | 0.038428 | -2.68% |
| 2022-09-08 | 7,160 | 0.024790 | 0.029321 | 0.028096 | +4.36% |
| 2022-09-09 | 11,460 | 0.025316 | 0.032377 | 0.035794 | -9.55% |
| 2022-09-10 | 11,246 | 0.029721 | 0.038512 | 0.040729 | -5.44% |
| 2022-09-11 | 8,257 | 0.036978 | 0.044208 | 0.040785 | +8.39% |

Desc_Qwen은 Metadata를 **7일 중 4일** 앞섰다. 상대차는 **−9.55%~+16.86%**로 변하며, 9월 7·9·10일에는 뒤졌다. 특히 사건 수가 많은 9월 9·10일은 전체 적격 사건의 **39.58%**를 차지한다. Graph_Qwen은 날짜별 평균 NDCG@10에서 두 Arm보다 7일 모두 낮다.

seed별 7일 균등 평균 NDCG@10은 다음과 같다.

| seed | Graph_Qwen | Desc_Qwen | Metadata | Desc / Metadata 상대차 |
| --- | --- | --- | --- | --- |
| 42 | 0.028960 | 0.035438 | 0.035339 | +0.28% |
| 43 | 0.029706 | 0.036859 | 0.035221 | +4.65% |
| 44 | 0.028168 | 0.037980 | 0.036939 | +2.82% |

Desc_Qwen은 3개 seed의 기간 평균에서 모두 Metadata보다 높지만, seed 42에서는 상대차가 **+0.28%**에 그친다. 날짜·seed 조합으로 나누면 승패는 다음과 같다. 날짜 승리는 seed 평균끼리, 조합 승리는 동일 날짜·동일 seed끼리 비교한다. 동률은 없다.

| 비교 방향 | NDCG@10 날짜 승리 | NDCG@10 조합 승리 | HR@10 날짜 승리 | HR@10 조합 승리 |
| --- | --- | --- | --- | --- |
| Desc_Qwen > Metadata | 4 / 7 | 12 / 21 | 4 / 7 | 9 / 21 |
| Graph_Qwen > Metadata | 0 / 7 | 1 / 21 | 0 / 7 | 0 / 21 |
| Graph_Qwen > Desc_Qwen | 0 / 7 | 0 / 21 | 0 / 7 | 0 / 21 |

Desc_Qwen은 전체 평균 HR@10이 가장 높아도 개별 21개 조합 중 Metadata를 앞선 경우는 9개다. 평균 개선 폭은 승리 횟수와 다른 정보를 준다. Graph_Qwen은 Desc_Qwen보다 NDCG@10·HR@10 모두 21개 조합에서 낮다. 이 승패 집계는 안정성을 살펴보는 기술 통계이며, 조합을 독립 표본으로 간주한 추가 유의성 검정은 아니다.

집계 방식에 대한 민감도를 확인하기 위해 원본 `daily`의 사건 수로 가중한 평균도 계산했다. 이 보조 집계는 seed를 동일하게 반영하면서 사건이 많은 날짜에 더 큰 가중치를 준다.

| 집계 | 지표 | Graph_Qwen | Desc_Qwen | Metadata | Desc / Metadata 상대차 |
| --- | --- | --- | --- | --- | --- |
| 날짜 균등 · 공식 | NDCG@10 | 0.028945 | 0.036759 | 0.035833 | +2.58% |
| 사건 수 가중 · 보조 | NDCG@10 | 0.028904 | 0.036666 | 0.036312 | +0.97% |
| 날짜 균등 · 공식 | HR@10 | 0.055706 | 0.068699 | 0.067215 | +2.21% |
| 사건 수 가중 · 보조 | HR@10 | 0.055670 | 0.068626 | 0.068114 | +0.75% |

순위는 유지되지만 Desc_Qwen의 Metadata 대비 NDCG@10 개선이 **+2.58% → +0.97%**, HR@10 개선이 **+2.21% → +0.75%**로 줄어든다. 따라서 날짜를 균등하게 중시하는 공식 평가에서의 우위를 실제 사건 빈도 전체에서 같은 크기의 효과로 해석하면 안 된다. **보조 집계에 대한 bootstrap CI는 재계산하지 않았으며, 공식 집계의 우월성 판정을 그대로 적용할 수 없다.**

생성 품질·완료 현황도 함께 확인했다. 장면 분모는 고정 30초 timestamp 산출물의 **115,947개 장면**이다. 근거: [장면 coverage](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:1395), [생성 복구 이력](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:253).

| 항목 | Graph_Qwen | Desc_Qwen | 해석 |
| --- | --- | --- | --- |
| native 장면 산출물 | 115,932 | 115,947 | 최종 산출물의 생성 방식 분류 |
| repaired 장면 산출물 | 8 | 0 | Graph 정상 성공 장면에 포함 |
| raw fallback 장면 | 7 | 0 | Graph 정상 성공 coverage에서는 제외 |
| 정상 성공 장면 | 115,940 / 115,947 | 115,947 / 115,947 | 99.99396% / 100% |
| 장면 결과 기록 coverage | 100% | 100% | raw fallback까지 포함한 결과 기록 완전성 |
| 최종 실패 / 누락 / 중복 결과 | 0 / 0 / 0 | 0 / 0 / 0 | 최종 장면 기록 기준 |
| 의미 규칙 경고 장면 | 44,941 / 115,940 · 38.7623% | 해당 지표 없음 | 정상 성공 Graph 중 경고가 있는 장면 비율 |
| native 영상 요약 | 19,738 | 19,738 | 각 Arm의 전체 아이템 수와 일치 |
| raw 장면을 입력받은 요약 | 7 / 19,738 · 0.03546% | 0 | 요약 출력 자체는 native |

두 Arm의 공통 정상 성공 장면은 **115,940개**이며, coverage 차이는 **0.00604%p**다. 설정된 최소 성공률 95%와 최대 Arm 간 차이 5%p 조건을 충족한다. `common_scene_intersection_required=false`이므로 두 Arm을 공통 성공 장면만으로 잘라 평가한 것은 아니다. 현재 coverage 집계에는 대규모 장면 누락이 나타나지 않는다. 다만 소수 raw 입력이나 특정 아이템의 영향 크기는 해당 사건을 연결해 보기 전에는 단정할 수 없다.

**Graph_Qwen의 의미 규칙 경고율 38.76%는 우선 점검할 신호다.** 성공적으로 파싱된 Graph에서도 경고가 기록되므로 `issue_counts={}`, 최종 실패 0건과 동시에 성립할 수 있다. 저장소의 [경고 규칙 구현](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/semantic_graph/schema.py:142)은 필드·타입, 개체 수, 이름 표기, 참조 관계 등 규칙 위반을 검사한다. 이 JSON에는 경고 종류별 분포나 추천 오차와의 연결이 없다. 따라서 **38.76%를 의미적으로 틀린 장면의 비율이나 성능 저하의 입증된 원인으로 해석할 수 없다.** Desc_Qwen에 동일 경고 지표가 없으므로 0%라고 비교해서도 안 된다.

Metadata는 **19,738개 중 3개 아이템의 title이 공백**이며 해당 입력만 영벡터로 처리했다. 비율은 **0.01520%**, item ID는 `4688`, `6026`, `12701`이다. 아이템과 interaction은 유지됐다. 이 3개 아이템에 평가 사건이 얼마나 집중되는지는 JSON에 없어, 성능 영향이 없다고 단정하거나 손실량을 계산할 수 없다. 근거: [Metadata 결측 기록](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:309).

생성 시도 횟수는 다음과 같다. 이는 `current_cycle_attempts`를 읽은 값이며, [집계 구현](/home/junsu2.choi/workspace/ViewingContextPipeline/src/extraction/recovery_report.py:22)상 각 기록의 마지막 재시도 묶음에 남은 시도를 합산한다.

| 단계 | Graph_Qwen 기록된 시도 | Desc_Qwen 기록된 시도 | 산출물당 시도 · Graph / Desc |
| --- | --- | --- | --- |
| 장면 | 129,008 | 115,947 | 1.113 / 1.000 |
| 영상 요약 | 19,918 | 21,113 | 1.009 / 1.070 |
| 합계 | 148,926 | 137,060 | 단계별 작업을 합한 참고 건수 |

Graph_Qwen의 장면 시도는 Desc_Qwen보다 **11.26% 많고**, 장면·요약 합산 시도는 **8.66% 많다**. 반면 영상 요약 시도는 Graph_Qwen이 더 적다. `native`는 최종 산출물의 생성 방식이며 한 번의 시도로 성공했다는 뜻은 아니다. 이 숫자는 전체 과거 실행의 누적 호출·토큰·GPU 시간·비용이 아니고, 두 경로는 프롬프트와 출력 형식·토큰 예산도 다르다. 따라서 시도 횟수 차이를 비용 차이로 치환하지 않는다. Metadata에는 영상 장면·요약 생성 단계가 없지만, 공통 임베딩·학습 비용까지 0이라는 뜻은 아니다.

결과를 바탕으로 후속 작업의 우선순위를 제안한다.

1. **Desc_Qwen을 품질 기준으로 두고 Metadata와 함께 유지한다.** 평균 개선은 확인됐지만 효과가 작으므로, 실제 생성 시간·토큰·GPU 사용량을 계측하고 추가 평가 기간에서 개선이 유지되는지 확인한 뒤 채택 비용을 판단한다. 별도 평가 데이터 없이 이 결과만으로 온라인 효과를 주장하지 않는다.
2. **Graph_Qwen의 경고와 정보 손실을 분리해 진단한다.** 장면별 경고 유형·개수와 아이템별 경고 비율을 집계한 뒤, 해당 아이템의 추천 순위·NDCG 차이에 연결한다. 영상 길이·장면 수·아이템 인기도도 함께 확인한다. 별도로 Graph→요약 단계에서 개체·행동·주제가 보존되는지, Description 요약과 길이·중복도·표현 다양성이 어떻게 다른지 비교한다. 이는 원인 후보를 검증할 제안이며 현재 파일에서 확인된 원인은 아니다.
3. **시간·집계 민감도를 사전에 정한 조건으로 재검증한다.** 추가 날짜와 반복 seed에서 날짜 균등 및 사건 수 가중 지표를 함께 보고, 각각에 맞는 사용자 단위 paired bootstrap을 계산한다. title+Description 결합이나 Graph 요약 정책 변경은 새 Arm으로 평가해야 하며, 이번 결과는 그 효과를 검증하지 않는다.

판정 범위는 이번 7일과 관측된 3개 seed다. 저장된 bootstrap은 사용자를 재표집하고 seed는 먼저 평균하므로, 미래 날짜의 분포 변화나 새 학습 seed의 변동을 모두 포괄하는 신뢰구간으로 볼 수 없다. JSON에는 사용자·아이템별 순위, 추천 다양성·catalog 노출, 인기도 구간별 성능, 경고 종류별 분포, 시간·토큰·비용 수치가 없어 해당 비교는 수행하지 않았다. 장면 정상 완료율도 표현의 의미적 정확성을 직접 측정하지 않는다.

파일의 `paper_reference`에는 Metadata HR@10 참고값 **4.6%**가 있고, 이번 **6.7215%**와의 차이는 **+2.1215%p**다. 이 값은 JSON에 저장된 참고값으로만 인용했다. 같은 파일이 실험 설정의 완전한 동일성을 보장하지 않는다고 명시하므로 논문 재현의 성공·실패 판정에 사용하지 않는다. 근거: [paper_reference](/home/junsu2.choi/workspace/ViewingContextPipeline/artifacts/Full_v1_260908/validation/diagnosis/diagnosis.json:332).

보고서 작성 시 63개 조합의 고유성·완전성, 날짜별 사건 수, 총 평가 행 수, 30개 Arm별 평균, 비교 효과·판정식, 장면 수 분모와 coverage를 원본 JSON 안에서 교차 확인했다. 평균 재계산의 최대 부동소수점 오차는 **1.39e-17**다. 개별 추천 산출물이나 사용자 단위 bootstrap을 다시 실행한 것은 아니며, CI는 원본 값을 사용했다. 날짜·seed·가중치 보조 집계의 원척도 수치는 [aggregated_metrics.csv](/home/junsu2.choi/workspace/ViewingContextPipeline/docs/reports/Full_v1_260908/aggregated_metrics.csv)에 저장했다. CSV의 `evaluation_rows_including_seeds`는 seed 반복을 포함한 평가 행 수다.

원본 SHA-256: `feab56e19bc63af3b16cd6498e202c4c551cf79c0538dd02141b10b141367369`
