# ViewingContext Validation

동일한 MicroLens cohort와 SASRec protocol에서 ID, visual-only Graph, visual-only Description item representation을 독립 학습·비교하는 evaluator component입니다.

![MicroLens SASRec validation](../docs/validation_pipeline.png)

이 결과는 고정된 next-item ranking protocol의 증거입니다. semantic correctness, CTR, 만족도 또는 인과효과로 해석하지 않습니다.

## 실험 흐름

1. interaction pairs와 실제 MP4의 교집합에서 deterministic cohort를 만든다.
2. leave-two-out train/validation/test split과 catalog를 고정한다.
3. paired `VP_graph`·`VP_desc`의 completeness와 evidence fingerprint를 검사한다.
4. local BGE로 두 1024D L2-normalized representation을 만든다.
5. ID, Graph, Description SASRec을 같은 split과 세 seed로 독립 학습한다.
6. HR/NDCG와 Graph-vs-Description paired bootstrap non-inferiority를 보고한다.

## 설정

- `config/pilot_1k.yaml`: 1K user pilot
- `config/canonical_100k.yaml`: 100K user canonical
- `contracts/cohort_manifest.schema.json`: cohort provenance
- `contracts/representation_manifest.schema.json`: profile/encoder provenance
- `contracts/report.schema.json`: 평가 결과와 readiness

Root runner는 데이터·profile·model·output 경로를 run별 runtime config로 주입합니다.

## Component CLI 독립 실행

모든 명령은 이 디렉터리에서 실행합니다. package를 설치하지 않았다면 `PYTHONPATH=src`를 설정합니다.

```powershell
conda activate llmjg
$env:PYTHONPATH = "src"

python -m vc_validation.cli --config config/pilot_1k.yaml preflight
python -m vc_validation.cli --config config/pilot_1k.yaml prepare-cohort
python -m vc_validation.cli --config config/pilot_1k.yaml materialize-representations
python -m vc_validation.cli --config config/pilot_1k.yaml run-experiment
```

`materialize-representations`는 catalog의 모든 콘텐츠에 Graph와 Description profile이 존재하고 두 문서의 `content_id`, `status`, non-empty `text`, evidence fingerprint가 일치할 때만 진행합니다.

## 테스트

```powershell
conda activate llmjg
python -m pytest -q
```
