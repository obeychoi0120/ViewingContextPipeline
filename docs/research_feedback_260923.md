교수님께서 제안해 주시고 기훈님께서 답변해 주신 모듈 1~3과 관련하여, 선행 연구 및 향후 검토해 보면 좋을 실험 방향을 정리하여 공유드립니다.

## 모듈 1 관련 (Scene-to-Graph Generation)

현재 방식처럼 소형 VLM으로 keyframe에서 schema 기반 JSON을 생성하고 이를 graph로 변환하는 접근은 관련 연구 흐름과도 잘 맞닿아 있는 것으로 보입니다. 특히 현재 schema(actor/action/target/instrument/location)는 situation recognition 분야(VidSitu)의 동사–의미역 구조와 유사합니다.
따라서 동사 어휘나 역할 정의를 설계하실 때 해당 연구를 참고하시면 좋을 것 같습니다. 최근에는 2B급 소형 VLM을 SFT한 뒤 RL로 추가 학습하여 JSON 형태의 scene graph를 생성하는 연구(R1-SGG, SceneGraphVLM)도 나오고 있습니다. 추출 품질 개선이 필요할 때 참고하실 수 있습니다.

다만 같은 VLM에서 나온 정보라면 graph와 description에 담기는 정보량 자체는 크게 다르지 않을 것으로 보입니다. 
그렇지만, 여러 비디오가 같은 노드를 공유하고, 정보가 역할별로 분해되어 있으며, 노드 단위로 검증하고 필터링할 수 있다는 가정 하에선 그래프를 활용하는게 장점이 될 수 있을거 같습니다.
이를 위해선 노드가 정규화되어 비디오 간에 실제로 공유하도록 설계 되는 것이 필요해보입니다. (예를 들어 "frying pan / pan / skillet"이 서로 다른 노드로 남는다면 graph는 압축된 caption과 크게 다르지 않다). 
이때 어휘 정규화는 LLM4SGG의 방식을, 노드 단위 검증은 Davidsonian Scene Graph(DSG)의 방식을 참고하실 수 있습니다.

정리하면, 모듈 1을 디벨롭하시면서, 추출 정확도를 넘어 "추천에 활용성이 있는 graph인가"를 기준으로 함께 평가하는 것이 적절해 보입니다.


## 모듈 2 관련 (Representation Generation from Scene Graph)

scene 그래프를 aggregate 해서 video representation을 만드는 방법에는 feature vector similarity 를 이용한 aggregation 방법이 존재합니다 (예시, DCGN).
간단하게 말씀드리면, 각 scene graph를 한 개의 node로 취급하고, 거기에서 추출한 feature vector similarity에 기반한 GCN (= GNN)을 활용해 '프레임->샷->이벤트->비디오' 순으로 representation을 만들어 나가는 방법이며 이는 GNN의 message propagation 과정에서 이웃끼리 feature information을 전파하는 것으로 video representation 정확도를 올릴 수 있을 것으로 기대됩니다. 단순 matching을 통한 merge도 있지만, 해당 방법은 비교 행위 자체의 computational complexity가 존재하여 on-device에는 적합하지 않아보여 앞서 언급한 방법으로 먼저 검증을 진행하는게 좋다고 생각합니다.

또한, 교수님께서 언급하셨던 video 사이를 합치는 것으로 인해 의미론적으로 정보를 잃는 것(Oversmoothing) 에 대해서는 attention-based aggregation 방법으로 완화할 수 있을 거 같습니다.
예시 1. Message Passing을 통한 semi-supervised classification (UniMP)
예시 2. causal-based supervision을 통한 attention function (CSA)

장르라는 메타데이터는 존재하니, 이를 통해서 앞선 vector similarity aggregation 방법과 attention-based aggregation 방법의 성능을 비교하는 쪽으로 방향을 잡는 것이 적합해 보입니다.
아래 (연관 Reference) 부분에 앞서 언급한 방법들을 언급한 논문과 추가로 attention-based graph에 대한 연구 및 Oversmoothing 문제에 대한 연구도 첨부하겠습니다.

## 모듈 3 관련 (User Representation Generation and Recommendation)

특정 비디오의 scene graph와 사용자가 보유한 video scene graph 집합 간의 직접적인 그래프 비교를 기반으로 추천을 수행하는 방식은 구현 가능할 것으로 보입니다. 이 경우 embedding-based 또는 kernel-based graph comparison을 활용할 수 있습니다. (구글 스칼라에 Graph Isomorphism Test, Graph Matching 키워드로 검색하시면 그래프 사이 비교 연산과 관련한 다양한 논문들을 찾아보실 수 있습니다.) 다만, 사용자가 보유한 scene graph의 수가 많아질수록 사용자별 연산량이 증가하며, 그래프 비교 알고리즘 자체의 계산 복잡도도 높은 편입니다. 따라서 직접 비교 방식보다는, 일반적인 추천 시스템에서 널리 활용되는 것처럼 사용자와 아이템을 동일한 의미 공간의 embedding/feature space로 투영하는 방향을 우선적으로 검토하는 것이 적절해 보입니다.

구체적으로는 다음과 같은 실험을 단계적으로 진행해 볼 수 있습니다.

1. Naive, non-learnable 방식
    사용자별 scene graph 집합을 사용자와 scene graph 간의 edge로 구성된 그래프로 해석한 뒤, 각 scene graph representation을 aggregation하여 user representation을 구성하는 방식입니다.
2. Learnable 방식
    위 방식의 확장으로, user embedding 자체를 학습 가능한 parameter로 두고 user–scene graph 사이 주어진 interaction을 바탕으로 user embedding을 학습하는 방식입니다.
3. Scene graph clustering 기반 방식
    각 scene graph를 하나의 node로 보고, scene graph 간의 연결 관계를 edge로 해석하여 graph of scene graphs를 구성한 뒤, clustering algorithm을 적용하는 방식입니다. 이때 이미 확보한 scene graph별 representation을 활용하거나, scene graph 간 연결 구조만 활용하는 방식 모두를 고려할 수 있습니다. 이후 각 cluster의 대표 representation(예: cluster 내 embedding의 평균)을 구하고, 각 사용자가 보유한 scene graph들이 속한 cluster를 기반으로 cluster embedding의 weighted sum을 user embedding으로 설정할 수 있습니다.

우선 위 3가지와 같은 방식들을 통해 MicroLens 데이터셋에서의 성능을 확인하고, 이를 바탕으로 사용자와 scene graph를 투영하는 feature space를 지속적으로 고도화해 나가면 좋을 것 같습니다.


## 연관 Reference

모듈1 관련
(DFF는 MicroLens 데이터셋에서 VLM 중간 layer latent를 활용한 최근 연구로, 성능 비교 시 참고하실 수 있을 것 같아 첨부하였습니다.)

VidSitu (https://arxiv.org/abs/2104.00990)
R1-SGG (https://arxiv.org/abs/2504.13617)
SceneGraphVLM (https://arxiv.org/abs/2605.13667)
LLM4SGG (https://arxiv.org/abs/2310.10404)
Davidsonian Scene Graph (https://arxiv.org/abs/2310.18235)
Frozen LVLMs for Micro-Video Recommendation (DFF) (https://arxiv.org/abs/2512.21863)

모듈2 관련

DCGN (https://arxiv.org/abs/1906.00377)
UniMP (https://arxiv.org/abs/2009.03509)
CSA (https://arxiv.org/abs/2305.13115)
Attention-based GNN (https://arxiv.org/abs/2605.08679)
Oversmoothing in GNN (https://arxiv.org/abs/2305.16102)

모듈3 관련
(주로 추천에서 많이쓰이는 그래프 알고리즘을 활용한 유저 데이터 표현 구축 관련 논문이고, PULSE의 경우 클러스터링 기반 임베딩 제작에 도움이 될 거 같아 첨부하였습니다.)

LightGCN (https://dl.acm.org/doi/abs/10.1145/3397271.3401063)
LightGCL (https://arxiv.org/abs/2302.08191)
SimGCL (https://dl.acm.org/doi/abs/10.1145/3477495.3531937)
SGL (https://dl.acm.org/doi/abs/10.1145/3404835.3462862)
PULSE (https://dl.acm.org/doi/abs/10.1145/3774904.3792228)