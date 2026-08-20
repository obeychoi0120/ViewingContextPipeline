# ViewingContext Extraction

영상에서 fixed-interval keyframe과 visual graph/description profile을 만드는 producer component입니다. 현재 MicroLens canonical track은 `img_only + fixed_30s + Qwen3-VL`입니다.

![Viewing Context extraction](../docs/extraction_pipeline.png)

## Canonical contract

| 표기 | 의미 | 출력 |
| --- | --- | --- |
| `SC_graph` | Scene별 visual Context Graph | `scene_context_graph_qwen/*_scene_context.jsonl` |
| `VC_graph` | 성공 Scene의 결정론적 영상 집계 | `video_context_graph_qwen/*_context_graph_ond.json` |
| `VP_graph` | `VC_graph`의 고정 순서 영어 직렬화 | `video_profile_graph_qwen/*_vp_graph.json` |
| Scene Description | 같은 keyframe의 detailed English description | `scene_description_qwen/*_scene_descriptions.jsonl` |
| `VP_desc` | Scene Description의 영상 단위 text-only 요약 | `video_profile_desc_qwen/*_vp_desc.json` |

Graph와 Description은 같은 `visual-evidence-fingerprint/v1`을 가져야 합니다. fingerprint는 Scene boundary, timestamp, 이미지 SHA-256·크기와 sampling contract를 포함합니다. `img_only` payload에는 ASR/OCR·제목·장르·카테고리를 넣지 않습니다.

## 환경과 설정

```powershell
conda activate llmjg
pip install -r requirements-test.txt
pip install -r requirements-ondevice.txt
```

- `config/microlens_config.json`: source와 import output
- `config/video_data_collection.json`: fixed interval, resize, ASR/OCR
- `config/scene_context_extraction_ondevice.json`: Graph VLM
- `config/scene_description_generation.json`: Description VLM
- `config/.env`: `OUTPUT_SAVE_PATH` 등 로컬 실행값

Root runner는 원본 설정을 수정하지 않고 run별 runtime copy를 전달합니다.

## Component CLI 독립 실행

모든 명령은 이 디렉터리에서 실행합니다.

```powershell
conda activate llmjg

python -m src.video_data_collection.cli import-microlens `
  --config config/microlens_config.json --scope pilot

python -m src.scene_context_extraction.ondevice.cli `
  --manifest output/manifests/catalog_manifest.csv `
  --settings config/scene_context_extraction_ondevice.json

python -m src.scene_description_generation.graph_profile_cli `
  --manifest path/to/vce_selection.jsonl `
  --settings config/scene_context_extraction_ondevice.json

python -m src.scene_description_generation.cli `
  --manifest path/to/vce_selection.jsonl `
  --settings config/scene_description_generation.json
```

각 CLI는 기존 fingerprint가 같은 결과를 resume합니다. `--force`는 해당 CLI가 선택한 콘텐츠만 다시 생성합니다. terminal Scene failure나 evidence mismatch가 남으면 완전한 profile을 쓰지 않습니다.

## 호환 경로

YouTube collection, `fixed_15s`, `shot_wise`, Gemini Reference, Gauss와 legacy multimodal `VP_ref` 코드는 기존 사용자를 위해 유지합니다. 이 경로들은 현재 MicroLens Graph-vs-Description root workflow에서 호출하지 않습니다.

## 테스트

```powershell
conda activate llmjg
python -m pytest -q
```
