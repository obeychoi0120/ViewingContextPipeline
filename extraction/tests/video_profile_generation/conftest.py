from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


CONTENT_ID = "Tech_Auto_001_dx8QCYU4kvQ"


def write_manifest(
    root: Path,
    text: str | None = None,
    content_ids: list[str] | None = None,
) -> Path:
    path = root / "contracts" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        selected_ids = content_ids or [CONTENT_ID]
        text = "content_id,url\n" + "".join(
            f"{content_id},https://example.com/{index}\n"
            for index, content_id in enumerate(selected_ids)
        )
    path.write_text(
        text,
        encoding="utf-8",
    )
    return path


def profile_payload() -> dict[str, object]:
    return {
        "profile": {
            "topic": [
                {
                    "concept_id": "products/consumption",
                    "extension_label": "앱테크와 영수증 적립",
                }
            ],
            "content_type": [
                {
                    "concept_id": "tutorial/explanation",
                    "extension_label": "스마트폰 앱 사용 설명",
                }
            ],
            "intent": [
                {"concept_id": "teach", "extension_label": None},
                {"concept_id": "inform", "extension_label": None},
            ],
            "media_form": ["live_action"],
            "presentation": {
                "pace": "medium",
                "information_density": "high",
                "complexity": "basic",
            },
        },
        "descriptive_context": {
            "key_entities": [],
            "intended_audience": ["간단한 절약과 앱테크에 관심 있는 시청자"],
            "languages": ["ko"],
            "formats": ["진행자 설명", "스마트폰 화면 시연"],
            "tones": ["친근함", "실용적"],
            "content_warnings": [],
        },
    }


def summary_payload() -> dict[str, object]:
    return {
        "summary": "영수증과 여러 리워드 앱을 활용해 소액을 적립하는 방법을 설명하는 영상이다."
    }


def details_payload() -> dict[str, object]:
    return profile_payload()


def ontology_payload() -> dict[str, object]:
    def facet_spec(facet: str, value_kind: str, **shape: object) -> dict[str, object]:
        return {
            "facet": facet,
            "ko_label": facet,
            "description": f"{facet} description",
            "value_kind": value_kind,
            "selection_rule": f"{facet} selection rule",
            **shape,
        }

    def closed_spec(value_id: str, rank: int | None) -> dict[str, object]:
        return {
            "value_id": value_id,
            "ko_label": value_id,
            "description": f"{value_id} description",
            "allowed_boundary": f"{value_id} allowed",
            "forbidden_boundary": f"{value_id} forbidden",
            "ordinal_rank": rank,
        }

    def concept(concept_id: str, facet: str) -> dict[str, object]:
        return {
            "concept_id": concept_id,
            "facet": facet,
            "parent_id": None,
            "ko_label": concept_id,
            "description": f"{concept_id} description",
            "allowed_boundary": f"{concept_id} allowed",
            "forbidden_boundary": f"{concept_id} forbidden",
            "aliases": [],
            "assignable_as_preference": True,
        }

    media_forms = [
        "live_action",
        "live_action_cg",
        "animation",
        "graphics",
    ]
    pace = ["slow", "medium", "fast"]
    density = ["low", "medium", "high"]
    complexity = ["general", "basic", "intermediate", "advanced"]
    payload = {
        "ontology_version": "viewing-ontology-contract/v3",
        "facet_specs": [
            facet_spec("topic", "canonical_concept", min_items=0, max_items=6),
            facet_spec(
                "content_type",
                "canonical_concept",
                min_items=0,
                max_items=3,
            ),
            facet_spec("intent", "canonical_concept", min_items=0, max_items=3),
            facet_spec(
                "media_form",
                "closed_enum",
                min_items=1,
                max_items=2,
                allow_unknown=True,
            ),
            facet_spec(
                "presentation",
                "ordinal",
                dimensions=["pace", "information_density", "complexity"],
                allow_unknown=True,
            ),
        ],
        "concept_id_policy": "stable IDs",
        "unmatched_concept_policy": "omit unmatched concepts",
        "unknown_value_policy": "unknown is observation-only",
        "closed_axes": {
            "media_form": media_forms,
            "presentation": {
                "pace": pace,
                "information_density": density,
                "complexity": complexity,
            },
        },
        "closed_axis_definitions": {
            "media_form": [closed_spec(value, None) for value in media_forms],
            "presentation": {
                "pace": [closed_spec(value, rank) for rank, value in enumerate(pace)],
                "information_density": [
                    closed_spec(value, rank) for rank, value in enumerate(density)
                ],
                "complexity": [
                    closed_spec(value, rank) for rank, value in enumerate(complexity)
                ],
            },
        },
        "canonical_concepts": {
            "topic": [concept("products/consumption", "topic")],
            "content_type": [concept("tutorial/explanation", "content_type")],
            "intent": [
                concept("teach", "intent"),
                concept("inform", "intent"),
            ],
        },
    }
    return payload


def write_ontology_contract(tmp_path) -> str:
    path = tmp_path / "viewing_ontology_v3.json"
    path.write_text(
        json.dumps(ontology_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def write_local_inputs(
    root: Path,
    objects: dict[str, str],
) -> Path:
    output_root = root / "output"
    metadata_path = output_root / "asset" / "metadata" / f"{CONTENT_ID}.json"
    ref_path = output_root / "asset" / "fixed_15s" / "ref_jsonl" / f"{CONTENT_ID}_ref.jsonl"
    metadata_path.unlink(missing_ok=True)
    ref_path.unlink(missing_ok=True)
    for mode in ("fixed_15s", "shot_wise"):
        shutil.rmtree(
            output_root / "asset" / mode / "resized_keyframes" / CONTENT_ID,
            ignore_errors=True,
        )
    local_prefixes = {
        "ViewingContextPipeline/asset/metadata/": output_root / "asset" / "metadata",
        "ViewingContextPipeline/asset/fixed_15s/ref_jsonl/": output_root / "asset" / "fixed_15s" / "ref_jsonl",
        "ViewingContextPipeline/asset/shot_wise/ref_jsonl/": output_root / "asset" / "shot_wise" / "ref_jsonl",
        "ViewingContextPipeline/asset/fixed_15s/resized_keyframes/": output_root / "asset" / "fixed_15s" / "resized_keyframes",
        "ViewingContextPipeline/asset/shot_wise/resized_keyframes/": output_root / "asset" / "shot_wise" / "resized_keyframes",
    }
    for name, text in objects.items():
        for prefix, directory in local_prefixes.items():
            if name.startswith(prefix):
                path = directory / name.removeprefix(prefix)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix.lower() == ".png":
                    path.write_bytes(text.encode("utf-8"))
                else:
                    path.write_text(text, encoding="utf-8")
                break
    return output_root


class FakeModels:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        response_index = len(self.calls)
        self.calls.append(kwargs)
        if response_index >= len(self.responses):
            raise AssertionError("unexpected generate_content call")
        return self.responses[response_index]


@pytest.fixture
def input_objects(tmp_path) -> dict[str, str]:
    metadata = {
        "title": "매일 사용하는 고효율 앱테크",
        "channel": "예브라",
        "upload_date": "20260701",
        "duration": 807,
        "description": "광고 없이 실제 사용하는 앱테크를 소개합니다.\n#앱테크",
        "view_count": 1234,
        "url": "https://example.com",
    }
    ref_scene = {
        "scene_idx": 0,
        "timeline": [
            {
                "shot_idx": 0,
                "timestamp": 0,
                "raw_asr": "영수증 앱테크를",
                "raw_ocr": "영수증 적립",
            },
            {
                "shot_idx": 1,
                "timestamp": 34,
                "raw_asr": "소개합니다.",
                "raw_ocr": "영수증 적립",
            },
        ],
    }
    objects = {
        f"ViewingContextPipeline/asset/metadata/{CONTENT_ID}.json": json.dumps(
            metadata, ensure_ascii=False
        ),
        f"ViewingContextPipeline/asset/fixed_15s/ref_jsonl/{CONTENT_ID}_ref.jsonl": json.dumps(
            ref_scene, ensure_ascii=False
        ),
        f"ViewingContextPipeline/asset/fixed_15s/resized_keyframes/{CONTENT_ID}/0000.png": "image",
        f"ViewingContextPipeline/asset/fixed_15s/resized_keyframes/{CONTENT_ID}/0034.png": "image",
    }
    write_local_inputs(tmp_path, objects)
    return objects


@pytest.fixture
def fake_client():
    return SimpleNamespace(
        models=FakeModels(
            SimpleNamespace(parsed=summary_payload(), text=None),
            SimpleNamespace(parsed=details_payload(), text=None),
        )
    )
