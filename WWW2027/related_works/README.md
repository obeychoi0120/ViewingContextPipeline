# HGVRec 관련 문헌과 Short Paper 작성 참고

HGVRec의 **장면 간 Action 연속 구간 집계인 ASHA**와 **장면 내부 역할 관계를 인코딩하는 REAF**를 설명하기 위한 문헌 19편입니다. 기술·데이터셋 문헌 16편과 작성 형식 참고 3편을 구분했습니다. 서지 확인일은 2026-11-10이며, 아래의 HGVRec 비교는 현재 구현과 원문을 대조한 해석입니다.

핵심 비교는 DWYS의 설명 기반 비디오 표현, MHCR의 사용자·아이템 수준 하이퍼그래프, AllSet 계열의 일반적인 집계 구조입니다. HGVRec은 **역할 정보를 반영한 장면 표현을 연속 Action 구간으로 묶어 추천용 비디오 표현을 학습하는 구성**으로 설명할 수 있습니다. 하이퍼그래프 왕복 집계나 역할 개념 자체를 최초 제안으로 서술하지 않습니다.

- [영어 Related Works 초안](../related_works.tex): Introduction 다음의 독립 섹션으로 구성한 세 문단, 핵심 11편 인용.
- [BibTeX 라이브러리](../references.bib): 19편 전체. 실제 인용한 문헌만 원고의 References에 출력됩니다.
- 기존 [DWYS PDF](../../paper/Describe%20What%20You%20See.pdf), [MHCR PDF](../../paper/MHCR.pdf), [ItemRAG PDF](../../paper/ItemRAG.pdf)를 활용했습니다.

## 문헌 목록과 인용 위치

‘우선’은 현재 Related Work 초안에 인용한 문헌입니다. ‘보완’은 Methods·Experiments의 설명에 맞춰 추가할 후보이며, ‘형식’은 글 구성 참고입니다.

| 번호 | 논문·약칭 | 발표 | HGVRec에서의 용도 | 구분 | BibTeX 키 |
|---|---|---|---|---|---|
| 1 | Describe What You See, DWYS | RecSys 2025 | MLLM 설명 기반 비디오 추천과 비교 | 우선 | `denadai2025dwys` |
| 2 | MHCR | ICASSP 2025 | 비디오 추천의 기존 하이퍼그래프 적용과 비교 | 우선 | `lyu2025mhcr` |
| 3 | Frozen LVLMs / DFF | arXiv, 2026 v2 | 설명과 hidden state, ID 결합의 대안 | 우선 | `sun2026dff` |
| 4 | MMGCN | ACM MM 2019 | 모달리티별 사용자·아이템 그래프 | 우선 | `wei2019mmgcn` |
| 5 | IISAN | SIGIR 2024 | 동결된 기반 모델의 추천용 표현 적응 | 보완 | `fu2024iisan` |
| 6 | DHCN | AAAI 2021 | 세션 하이퍼엣지와 Action 구간의 단위 차이 | 우선 | `xia2021dhcn` |
| 7 | Hypergraph Neural Networks, HGNN | AAAI 2019 | 하이퍼그래프 전파의 기본 선행연구 | 우선 | `feng2019hgnn` |
| 8 | HNHN | ICML 2020 GRL+ 워크숍 | 하이퍼엣지 표현·크기 정규화 | 보완 | `dong2020hnhn` |
| 9 | You are AllSet | ICLR 2022 | 두 multiset 함수로 ASHA를 설명 | 우선 | `chien2022allset` |
| 10 | ED-HNN | ICLR 2023 | 수신 노드별 메시지와 ASHA의 차이 | 우선 | `wang2023edhnn` |
| 11 | ZEN | NeurIPS 2025 | 선형화된 HNN 비교와 설계 요소 분해 | 우선 | `bae2025zen` |
| 12 | Deep Sets | NeurIPS 2017 | 순열 불변 집계와 집합 크기 정보 | 보완 | `zaheer2017deepsets` |
| 13 | Action Genome | CVPR 2020 | Action의 관계·장면 그래프 표현 | 우선 | `ji2020actiongenome` |
| 14 | Grounded Situation Recognition | ECCV 2020 | Action 내 Entity의 의미 역할 | 우선 | `pratt2020gsr` |
| 15 | SASRec | ICDM 2018 | 순차 추천 backbone 출처 | 보완 | `kang2018sasrec` |
| 16 | MicroLens | CIKM 2025 | 비디오 추천 데이터셋 출처 | 보완 | `ni2025microlens` |
| 17 | ItemRAG | SIGIR 2026 Short | 문제·기여·실험 연결 방식 | 형식 | `kim2026itemrag` |
| 18 | TESMR | SIGIR 2026 Short | 표현 학습 중심의 기여 전개 | 형식·보완 | `shin2026tesmr` |
| 19 | ReFuGe | WWW 2026 Short | 참고문헌 포함 4페이지 구성 | 형식 | `kim2026refuge` |

## 비디오 표현과 추천

### 1. DWYS

**Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations** — Marco De Nadai, Andreas Damianou, Mounia Lalmas. RecSys 2025.

동결된 MLLM으로 비디오·오디오의 의미를 자연어로 설명하고, 텍스트 임베딩을 여러 추천 모델에 입력합니다. HGVRec과는 ‘영상에서 의미 정보를 추출해 추천용 입력을 만든다’는 지점이 가깝습니다. 비교할 부분은 텍스트에 역할·시간 정보가 들어 있는지 여부보다, 추출 후 표현 학습에서 역할 연결과 연속 구간을 **명시적인 연산 구조로 사용하는지**입니다. DWYS의 설명도 Entity와 Action, 시간적 정보를 포함할 수 있으므로 ‘DWYS는 행동이나 시간 정보를 전혀 표현하지 못한다’고 쓰면 부정확합니다.

원문의 §4–5를 보면 입력 모달리티와 추천 모델, 분할 방식도 비교에 영향을 줍니다. 논문에 보고된 수치를 현재 프로젝트 결과와 바로 대조하지 않습니다.

출처: [논문](https://arxiv.org/abs/2508.09789), [출판 DOI](https://doi.org/10.1145/3705328.3759303). 로컬 PDF의 §4–5도 확인했습니다.

### 2. MHCR

**Multi-view Hypergraph-based Contrastive Learning Model for Cold-Start Micro-video Recommendation** — Sisuo Lyu, Xiuze Zhou, Xuming Hu. ICASSP 2025.

사용자–아이템 그래프, 모달리티별 아이템 유사도 그래프, 학습되는 하이퍼엣지 표현을 결합하고, 모달리티 간 및 그래프–하이퍼그래프 대조 학습을 사용합니다. 원문 §II-B의 하이퍼그래프 소속 행렬은 아이템 특징과 학습되는 하이퍼엣지 벡터로 구성되며, 사용자 표현에는 상호작용 정보도 반영됩니다.

HGVRec의 ASHA는 **개별 비디오 내부의 장면**을 노드로 삼고, Action key가 유지되는 연속 장면 구간으로 소속을 정합니다. MHCR과 비교할 핵심은 노드 단위, 하이퍼엣지 구성 규칙, 추천 목적함수이며, 단순히 ‘비디오에 하이퍼그래프를 적용했다’는 차별화는 성립하지 않습니다. 현재 HGVRec에 MHCR의 cold-start 성능이나 대조 학습 효과를 전이해서 주장하지 않습니다.

출처: [원문 v2](https://arxiv.org/abs/2409.09638v2), [출판 DOI](https://doi.org/10.1109/ICASSP49660.2025.10887695), [출판사 등록 서지](https://api.crossref.org/works/10.1109/ICASSP49660.2025.10887695). 로컬 PDF의 §II-B–C, 식 (3)–(5)를 확인했습니다. 기존 `docs/paper/MHCR.md`의 arXiv 정보에 더해, 이번 라이브러리에는 ICASSP 출판 정보를 반영했습니다.

### 3. DFF

**Frozen LVLMs for Micro-Video Recommendation: A Systematic Study of Feature Extraction and Fusion** — Huatuan Sun, Yunshan Ma, Changguang Wu, Yanxin Zhang, Pengfei Wang, Xiaoyu Du. arXiv:2512.21863, v2 2026-05-05.

동결된 video-language model의 생성 설명과 중간 hidden state, 그리고 콘텐츠 표현의 ID 대체·결합 전략을 비교합니다. DFF는 여러 층의 hidden state와 item ID 표현을 결합합니다. HGVRec이 사용하는 명시적 구조화 의미 표현과 구분되는 강한 콘텐츠 표현 대안으로 읽을 수 있습니다.

원문은 자체 실험에서 hidden state 및 ID 결합의 이점을 보고하지만, 이것이 모든 데이터셋에서 텍스트 기반 접근보다 우수하다는 일반 명제는 아닙니다. 향후 비교한다면 ID 사용 여부와 동결 특징의 조건을 맞춰야 합니다. 최초 공개는 2025년이며, BibTeX의 2026년은 검토한 v2의 연도입니다. 확인된 학회 출판 정보 없이 학회명을 붙이지 않았습니다.

출처: [논문 v2](https://arxiv.org/abs/2512.21863v2).

### 4. MMGCN

**MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video** — Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, Tat-Seng Chua. ACM MM 2019.

각 모달리티에서 사용자–아이템 이분 그래프를 구성하고 상호작용을 통해 사용자와 비디오 표현을 학습합니다. 추천 시스템에서 ‘graph-based representation’이 이미 사용된 맥락을 짧게 소개하기 좋습니다. 그래프의 노드는 사용자·비디오이므로 ASHA의 비디오 내부 장면 그래프와 구분합니다. MMGCN을 장면 그래프나 Action 역할 집계의 직접 선행 구현으로 설명하지 않습니다.

출처: [저자 소속기관 서지·초록](https://research.monash.edu/en/publications/mmgcn-multi-modal-graph-convolution-network-for-personalized-reco/), [DOI](https://doi.org/10.1145/3343031.3351034).

### 5. IISAN

**IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT** — Junchen Fu, Xuri Ge, Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, Jie Wang, Joemon M. Jose. SIGIR 2024.

기반 모델과 분리된 side network에서 모달리티 내부·사이의 표현을 적응시킵니다. HGVRec의 ‘동결 특징 위에 추천 목적의 모듈을 학습한다’는 구성에 관련됩니다. 다만 IISAN의 핵심은 멀티모달 적응과 학습 효율이며, ASHA처럼 연속 Action 구간을 구성하는 방법은 아닙니다. Methods의 학습 설정을 설명하거나 학습 비용 비교를 실제로 수행할 때 인용할 후보입니다. 논문의 GPU 학습 효율을 모바일·TV 추론 효율의 근거로 사용하지 않습니다.

출처: [논문](https://arxiv.org/abs/2404.02059), [SIGIR 공식 proceedings](https://sigir-2024.github.io/proceedings), [DOI](https://doi.org/10.1145/3626772.3657725).

### 6. DHCN

**Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation** — Xin Xia, Hongzhi Yin, Junliang Yu, Qinyong Wang, Lizhen Cui, Xiangliang Zhang. AAAI 2021.

아이템을 노드로, 한 세션의 아이템 집합을 하이퍼엣지로 구성합니다. 하이퍼그래프와 그 line graph의 두 채널에서 얻는 세션 표현을 자기지도 학습으로 연결합니다. ‘여러 아이템의 공동 출현’을 묶는 선행연구와 ‘한 비디오의 여러 장면에서 Action이 지속되는 구간’을 묶는 ASHA의 차이를 드러내기 좋습니다. 세션 단위의 사용자 행동과 영상 속 Action은 의미가 다른 대상입니다.

출처: [AAAI 원문](https://cdn.aaai.org/ojs/16578/16578-13-20072-1-2-20210518.pdf), [공개 논문](https://arxiv.org/abs/2012.06852). 원문의 hypergraph construction과 dual-channel 구성을 확인했습니다.

## ASHA의 하이퍼그래프 집계

### 7. HGNN

**Hypergraph Neural Networks** — Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao. AAAI 2019.

incidence matrix와 정규화된 하이퍼그래프 전파를 통해 고차 관계를 표현하는 기본 문헌입니다. ASHA의 장면→구간→장면 정보 전달이 기존 하이퍼그래프 학습과 연결된다는 근거로 인용합니다. 일반적인 왕복 전파 자체와, 어떤 장면을 하나의 하이퍼엣지로 묶을지에 대한 HGVRec의 구성 규칙을 구별해야 합니다.

출처: [논문](https://arxiv.org/abs/1809.09401), [AAAI DOI](https://doi.org/10.1609/aaai.v33i01.33013558).

### 8. HNHN

**HNHN: Hypergraph Networks with Hyperedge Neurons** — Yihe Dong, Will Sawin, Yoshua Bengio. Graph Representation Learning and Beyond Workshop at ICML 2020.

노드와 하이퍼엣지 양쪽에 비선형 변환을 적용하고, 하이퍼엣지 크기와 노드 차수의 영향을 조절하는 정규화를 제시합니다. ASHA가 구간 표현을 만들고 장면으로 되돌려 보내는 구조와, 구간 크기를 취급하는 방법을 검토하는 보완 문헌입니다. HNHN의 정규화와 ASHA의 명시적 크기 특징 `[log |e|, |e|/S]`는 서로 다른 설계입니다. ICML 본회의 논문으로 표기하지 않습니다.

출처: [저자 기관의 워크숍 정보](https://www.microsoft.com/en-us/research/publication/hnhn-hypergraph-networks-with-hyperedge-neurons/), [논문](https://arxiv.org/abs/2006.12278).

### 9. AllSet

**You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks** — Eli Chien, Chao Pan, Jianhao Peng, Olgica Milenkovic. ICLR 2022.

노드→하이퍼엣지와 하이퍼엣지→노드의 두 multiset 함수로 다양한 HNN 집계를 일반화합니다. §3의 식 (5)는 각 집계 함수가 기존 노드·하이퍼엣지 상태를 조건으로 사용할 수 있도록 정의합니다. ASHA의 mean 기반 구간 집계와 장면 갱신을 설명하는 가장 직접적인 구조적 참고입니다.

ASHA는 이 일반적인 틀 안에서 **연속 Action으로 정해지는 소속**, **구간 크기 특징**, **구간 단위 비디오 readout**을 사용하는 구체적인 설계로 해석할 수 있습니다. 이 해석이 AllSetTransformer와의 연산상 동일성이나 보편 근사 능력의 증명은 아닙니다. 서지는 arXiv 메타데이터의 ‘Multiset Function’ 제목을 사용했으며, 공개 PDF 첫 페이지에는 ‘Multiset Learning’이라는 제목 변형이 있습니다.

출처: [ICLR 공개 기록](https://openreview.net/forum?id=hpBTIv2uy_E), [논문](https://arxiv.org/abs/2106.13264). §3–4와 식 (5)를 확인했습니다.

### 10. ED-HNN

**Equivariant Hypergraph Diffusion Neural Operators** — Peihao Wang, Shenghao Yang, Yunyu Liu, Zhangyang Wang, Pan Li. ICLR 2023.

하이퍼엣지의 집계 표현뿐 아니라 수신 노드 자신의 표현을 사용해 **각 incidence마다** 메시지를 만듭니다. Algorithm 1은 `m(e→v) = rho(h_v, m_e)`를 계산한 뒤 노드에서 메시지를 집계하며, 반복 층의 파라미터를 공유합니다.

현재 ASHA는 구간 표현을 장면별로 먼저 평균한 후 `[scene; received]`를 변환합니다. 따라서 노드 상태를 함께 쓴다는 유사성은 있지만, ED-HNN의 ‘수신 노드별 변환 후 집계’와 ASHA의 ‘집계 후 결합’은 다릅니다. ED-HNN의 표현력 보장이나 깊은 모델의 결과를 ASHA에도 적용된다고 쓰지 않습니다. 겹치는 구간을 통한 다층 전파를 향후 검토할 때 유용한 문헌입니다.

출처: [ICLR 공개 기록](https://openreview.net/forum?id=RiTjKoscnNd), [논문](https://arxiv.org/abs/2207.06680). §3.2–3.3, Algorithm 1, 식 (5)를 확인했습니다.

### 11. ZEN

**Parameter-Free Hypergraph Neural Network for Few-Shot Node Classification** — Chaewoon Bae, Doyun Choi, Jaehyun Lee, Jaemin Yoo. NeurIPS 2025.

HGNN, HNHN, UniGCNII, AllDeepSets, ED-HNN의 **선형화된 형태**를 하나의 전파 관점으로 정리합니다. 이후 닫힌 형태의 가중치 해와 중복 자기 정보의 영향을 조절하는 전파를 사용합니다. §3.1, Table 1과 식 (2)는 ASHA에서 소속 행렬, 정규화, 변환, residual/readout을 구분해서 설명할 때 유용합니다.

‘기존의 모든 비선형 HNN이 원래부터 같은 식이다’라는 뜻은 아닙니다. ZEN의 선형화 과정은 비선형 함수를 제거하는 가정을 둡니다. 현재 ASHA는 학습 가능한 비선형 모듈이며 ZEN의 parameter-free 모델과 다릅니다. few-shot 노드 분류의 속도·성능 결과도 비디오 추천의 결과로 해석하지 않습니다.

출처: [NeurIPS 공식 논문](https://proceedings.neurips.cc/paper_files/paper/2025/hash/65f46f5c816b2fd5a2f08ab04a8bc4a3-Abstract-Conference.html), [공개 논문](https://arxiv.org/abs/2510.21462).

### 12. Deep Sets

**Deep Sets** — Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, Alexander J. Smola. NeurIPS 2017.

집합 입력에 대한 순열 불변 함수와 집계 표현의 기초 문헌입니다. ASHA의 구간 내부 mean과 비디오 readout이 순서에 불변인 이유를 설명할 때 참고할 수 있습니다. 현재 구현에 대한 해석으로, 동일한 장면 벡터가 반복되면 mean은 반복 횟수를 구별하지 못하므로 별도의 구간 크기 특징이 정보를 보완합니다. 이는 크기 특징의 실험적 효과를 확인했다는 뜻은 아닙니다. 또한 구간을 시간상 연속으로 구성하는 것과 구간 내부 순서를 인코딩하는 것은 구분해야 합니다.

출처: [NeurIPS 공식 논문](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).

## REAF의 역할 관계와 시각 표현

### 13. Action Genome

**Action Genome: Actions as Compositions of Spatio-Temporal Scene Graphs** — Jingwei Ji, Ranjay Krishna, Li Fei-Fei, Juan Carlos Niebles. CVPR 2020.

Action을 사람·객체 사이의 관계와 그 시간적 변화로 분해하는 장면 그래프 표현 및 데이터셋을 제시합니다. ‘Action을 하나의 라벨로만 다루지 않고 참여 객체와 관계를 함께 표현한다’는 배경에 적합합니다. HGVRec의 actor·target·tool·place 역할 스키마와 Action Genome의 관계 분류 체계는 동일하지 않습니다. 또한 Action Genome의 행동 인식·그래프 예측 결과는 REAF의 추천 성능을 검증하지 않습니다.

출처: [CVPR 원문·서지](https://openaccess.thecvf.com/content_CVPR_2020/html/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.html), [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.pdf).

### 14. Grounded Situation Recognition

**Grounded Situation Recognition** — Sarah Pratt, Mark Yatskar, Luca Weihs, Ali Farhadi, Aniruddha Kembhavi. ECCV 2020.

이미지의 주요 활동, 그 활동에 참여하는 Entity의 의미 역할, Entity의 bounding box를 함께 예측하는 과제를 정의합니다. 역할에 따라 Entity–Action 관계를 구별하는 REAF의 동기를 설명하기 좋습니다.

GSR은 역할과 위치를 **예측**하지만, REAF는 VLM이 이미 추출한 역할 연결에 따라 특징을 **결합**합니다. 따라서 REAF를 새 역할 검출기나 grounding 모델로 서술하지 않습니다. REAF의 추천 목적 표현 학습과 GSR의 시각적 구조 예측은 목적함수도 다릅니다.

출처: [ECCV 공식 기록](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1987_ECCV_2020_paper.php), [논문](https://arxiv.org/abs/2003.12058).

## 추천 backbone과 데이터셋

### 15. SASRec

**Self-Attentive Sequential Recommendation** — Wang-Cheng Kang, Julian McAuley. ICDM 2018.

사용자 상호작용 이력의 self-attention으로 다음 아이템을 예측합니다. HGVRec의 추천 backbone을 설명하는 Methods 또는 Experiments에서 인용합니다. 사용자 이력의 순서 모델링과 비디오 내부의 Action 구간 집계는 서로 다른 수준의 시간 구조입니다. ASHA·REAF의 효과를 논의할 때 backbone 변경 효과와 구분하는 기준이 됩니다.

출처: [논문·ICDM 채택 정보](https://arxiv.org/abs/1808.09781), [DOI](https://doi.org/10.1109/ICDM.2018.00035).

### 16. MicroLens

**A Content-Driven Micro-Video Recommendation Dataset at Scale** — Yongxin Ni, Yu Cheng, Xiangyan Liu, Junchen Fu, Youhua Li, Xiangnan He, Yongfeng Zhang, Fajie Yuan. CIKM 2025.

원본 영상과 여러 콘텐츠 모달리티를 활용하는 비디오 추천 데이터셋·벤치마크입니다. Experiments의 데이터셋 출처로 인용합니다. 2023년 arXiv 초고와 2025년 출판본이 있으므로 BibTeX는 정식 출판본을 기준으로 정리했습니다.

초고와 출판본의 초록에는 전체 데이터 규모도 다르게 기술되어 있습니다. 실험 표의 사용자·아이템·상호작용 수는 논문 초록에서 가져오지 않고, 실제 사용하는 MicroLens subset과 전처리 후 cohort에서 확인해야 합니다. 이 문헌은 HGVRec의 성능 근거가 아니라 데이터 출처입니다.

출처: [저자 소속기관의 정식 출판 정보](https://www.researchwithrutgers.org/en/publications/a-content-driven-micro-video-recommendation-dataset-at-scale/), [출판 DOI](https://doi.org/10.1145/3746252.3761655), [2023년 초고](https://arxiv.org/abs/2309.15379).

## KAIST 연구실의 Short Paper 작성 예시

아래 세 편은 [신기정 교수님의 공개 논문 목록](https://kijungs.github.io/)에서 Short Paper 여부를 확인했습니다. ‘형식 참고’만을 이유로 원고에 인용할 필요는 없습니다. 페이지 수와 섹션은 확인한 공개 PDF 기준입니다.

| 논문 | 페이지 | 본문 섹션 | HGVRec 작성 시 참고할 부분 |
|---|---:|---|---|
| ItemRAG | 5, References는 5쪽 | Introduction → Related work and preliminary → Proposed method → Experiment → Conclusion and discussion | 기존 입력·검색 단위의 문제에서 구체적인 설계 차이로 이어지는 설명 |
| TESMR | 5, References는 5쪽 | Introduction → Related Work & Problem Formulation → Preliminary Data Analysis → Proposed Method → Experiments → Conclusions and Future Directions | 표현의 문제를 제시하고 각 학습 단계의 역할을 연결하는 방식 |
| ReFuGe | 4, References는 4쪽 | Introduction & Related Works → Task Formulation → Proposed Framework → Experiments → Conclusions | Related Work를 Introduction에 통합하고 방법·실험에 공간을 배분하는 방식 |

### 17. ItemRAG

**ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation** — Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, Kijung Shin. SIGIR 2026 Short Paper.

추천 프롬프트에 넣을 정보를 사용자 이력 단위가 아니라 아이템 단위로 검색하고, 의미 정보와 공동 구매 정보를 결합합니다. HGVRec의 직접적인 장면 하이퍼그래프 선행연구는 아니지만, 기존 접근의 표현 단위를 짚고 새 단위를 제안하는 논리 전개를 참고할 수 있습니다. 5페이지 PDF이므로 4페이지 전체 분량의 예시로 사용하지 않습니다.

출처: [논문](https://arxiv.org/abs/2511.15141), [출판 DOI](https://doi.org/10.1145/3805712.3809942), [로컬 PDF](../../paper/ItemRAG.pdf).

### 18. TESMR

**From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation** — Jeeho Shin, Kyungho Kim, Kijung Shin. SIGIR 2026 Short Paper.

콘텐츠 기반 특징 개선, 사용자–레시피 관계 전파, 대조 학습의 세 단계로 멀티모달 특징을 추천용 임베딩으로 바꿉니다. 표현 학습 중심의 기여를 설명하는 형식과 모듈별 역할을 실험 질문에 연결하는 방식을 참고하기 좋습니다. HGVRec에 같은 세 단계나 대조 학습이 있다는 뜻은 아닙니다. 데이터 분석을 별도 섹션으로 둔 5페이지 예시이므로 HGVRec에서 섹션 수를 그대로 따라갈 필요는 없습니다.

출처: [논문](https://arxiv.org/abs/2511.19176), [출판 DOI](https://doi.org/10.1145/3805712.3809863).

### 19. ReFuGe

**ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents** — Kyungho Kim, Geon Lee, Juyeon Kim, Dongwon Choi, Shinhwan Kang, Kijung Shin. WWW 2026 Short Paper.

관계형 데이터베이스 예측을 위한 특징을 LLM agent로 생성·선택합니다. 추천이나 Action 표현의 직접 선행연구는 아니지만, 참고문헌까지 4페이지에 들어가는 WWW 원고 형식의 구체적인 예시입니다. Introduction 안에 관련 연구를 통합하고, 짧은 문제 정의, 프레임워크 그림, 실험 질문, 짧은 결론으로 연결합니다. HGVRec에서는 Related Works를 독립 섹션으로 두면서 방법·실험에 공간을 배분하는 방식을 참고할 수 있습니다.

출처: [논문·Short Paper 채택 정보](https://arxiv.org/abs/2601.17735), [출판 DOI](https://doi.org/10.1145/3774904.3792895).

## ASHA와 REAF를 서술할 때의 비교 기준

| 비교 대상 | 기존 연구에서 사용하는 요소 | 현재 HGVRec에서 설명할 요소 |
|---|---|---|
| DWYS | 자연어 설명의 텍스트 표현 | 추출된 역할과 Action 구간을 명시적으로 사용하는 표현 학습 |
| MHCR·DHCN | 사용자·아이템 또는 세션 수준의 하이퍼그래프 | 개별 비디오 내부의 장면과 연속 Action 구간 |
| HGNN·AllSet | 노드–하이퍼엣지 전파와 multiset 집계 | Action key에 따른 연속 구간 소속, 중첩, 크기 특징, 구간 readout |
| ED-HNN | 수신 노드별 하이퍼엣지 메시지, 층간 파라미터 공유 | 구간 메시지 평균 후 장면 특징과 결합하는 현재의 단일 왕복 |
| HNHN·Deep Sets | 크기·차수 처리, 순열 불변 집계 | mean이 직접 보존하지 않는 구간 크기를 별도 특징으로 전달 |
| GSR·Action Genome | 의미 역할 또는 시공간 관계의 시각적 표현 | 이미 추출된 actor·target·tool·place를 추천 표현에 반영 |

이 표는 설계의 비교이며, 성능 우위나 신규성의 최종 증명은 아닙니다. 현재 코드에서 확인한 경계는 다음과 같습니다.

- ASHA의 구간은 정규화된 Action key와 연속된 scene index로 구성됩니다. 같은 key가 나중에 재등장하면 별도 구간이며, 여러 Action을 가진 장면은 여러 구간에 속할 수 있습니다. 학습된 의미 기반 이벤트 검출이나 Entity identity 추적과 구별합니다.
- 구간 크기 특징은 `[log |e|, |e|/S]`입니다. 장면 수에 기반한 범위 정보이며, 초 단위 실제 지속 시간이나 구간 내부의 세밀한 순서를 나타내지 않습니다.
- REAF는 추출된 역할에 따라 Entity 특징을 Action 특징에 결합합니다. 역할 발견이나 양방향 Entity–Action 갱신을 수행하는 모듈로 서술하지 않습니다.
- ZEN·AllSet·ED-HNN에서 성립하는 이론·실험 결과는 각 논문의 가정과 구조에 대한 결과입니다. 현재 HGVRec으로 그대로 옮기지 않습니다.

구현 대조: [장면·역할·구간 인코더](../../../src/validation/graph_model.py), [Action key와 연속 구간 구성](../../../src/validation/scene_segments.py). 인용 우선순위는 4페이지 원고의 공간을 위한 편집 제안이며, 실험 결과와 Methods가 채워지면 필요한 근거에 맞춰 조정할 수 있습니다.
