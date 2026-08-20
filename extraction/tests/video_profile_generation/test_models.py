from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.video_profile_generation.models import (
    VideoProfileDetails,
    VideoProfileDocument,
    VideoProfileSummary,
    build_video_profile_details_model,
)

from .conftest import ontology_payload, profile_payload, summary_payload


def test_summary_rejects_over_1000_characters() -> None:
    payload = summary_payload()
    payload["summary"] = "가" * 1001

    with pytest.raises(ValidationError):
        VideoProfileSummary.model_validate(payload)


def test_summary_schema_describes_retrieval_not_plot_synopsis() -> None:
    description = VideoProfileSummary.model_json_schema()["properties"]["summary"][
        "description"
    ]

    assert "Semantic retrieval" in description
    assert "시간순 줄거리보다 핵심 주제" in description


def test_profile_deduplicates_labels_without_reordering() -> None:
    payload = profile_payload()
    payload["profile"]["topic"] = [
        {"concept_id": "products/consumption", "extension_label": "앱테크"},
        {"concept_id": "products/consumption", "extension_label": "영수증 적립"},
    ]

    with pytest.raises(ValidationError):
        VideoProfileDetails.model_validate(payload)


def test_profile_uses_und_when_language_is_unknown() -> None:
    payload = profile_payload()
    payload["descriptive_context"]["languages"] = []

    with pytest.raises(ValidationError):
        VideoProfileDetails.model_validate(payload)

    payload["descriptive_context"]["languages"] = ["und"]
    assert VideoProfileDetails.model_validate(
        payload
    ).descriptive_context.languages == ["und"]


def test_details_response_schema_is_derived_from_contract() -> None:
    model = build_video_profile_details_model(ontology_payload())
    schema = model.model_json_schema()
    profile_ref = schema["properties"]["profile"]["$ref"].split("/")[-1]
    profile = schema["$defs"][profile_ref]

    assert set(profile["properties"]) == {
        "topic",
        "content_type",
        "intent",
        "media_form",
        "presentation",
    }
    assert profile["properties"]["topic"]["minItems"] == 0
    assert profile["properties"]["topic"]["maxItems"] == 6
    media_items = profile["properties"]["media_form"]["items"]
    assert media_items["enum"] == [
        "live_action",
        "live_action_cg",
        "animation",
        "graphics",
        "unknown",
    ]


def test_document_rejects_extra_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        VideoProfileDocument.model_validate(
            {
                "title": "매일 사용하는 고효율 앱테크",
                "channel": "예브라",
                "upload_date": "20260701",
                "meta_desc": "설명",
                "summary": summary_payload()["summary"],
                "profile": profile_payload()["profile"],
                "descriptive_context": profile_payload()["descriptive_context"],
                "generation": {},
            }
        )
