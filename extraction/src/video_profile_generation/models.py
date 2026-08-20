"""Strict schemas for generated video profiles."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    field_validator,
)


ShortLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
AudienceLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)
]
LanguageCode = Annotated[str, StringConstraints(pattern=r"^(?:[a-z]{2}|und)$")]
SummaryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalConceptRef(StrictModel):
    concept_id: ShortLabel
    extension_label: ShortLabel | None = None


class Presentation(StrictModel):
    pace: ShortLabel
    information_density: ShortLabel
    complexity: ShortLabel


class CanonicalProfile(StrictModel):
    topic: list[CanonicalConceptRef]
    content_type: list[CanonicalConceptRef]
    intent: list[CanonicalConceptRef]
    media_form: list[ShortLabel]
    presentation: Presentation

    @field_validator("topic", "content_type", "intent")
    @classmethod
    def deduplicate_concepts(
        cls, values: list[CanonicalConceptRef]
    ) -> list[CanonicalConceptRef]:
        ids = [item.concept_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("canonical concept IDs must not be duplicated")
        return values

    @field_validator("media_form")
    @classmethod
    def validate_media_form(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("media_form must not contain duplicates")
        if "unknown" in values and len(values) != 1:
            raise ValueError("unknown media form must be used alone")
        return values


class DescriptiveContext(StrictModel):
    key_entities: list[ShortLabel] = Field(
        max_length=12,
        description="Explicitly evidenced named entities, ordered by importance.",
    )
    intended_audience: list[AudienceLabel] = Field(
        max_length=4,
        description="Audience interests or prerequisite knowledge, never demographics.",
    )
    languages: list[LanguageCode] = Field(
        min_length=1,
        max_length=3,
        description="Observed language codes in importance order; use und if unknown.",
    )
    formats: list[ShortLabel] = Field(
        max_length=4,
        description="Observable presentation formats, ordered by importance.",
    )
    tones: list[ShortLabel] = Field(
        max_length=4,
        description="Observable emotional or rhetorical tones, ordered by importance.",
    )
    content_warnings: list[ShortLabel] = Field(
        max_length=6,
        description="Directly evidenced content warnings.",
    )

    @field_validator(
        "key_entities",
        "intended_audience",
        "languages",
        "formats",
        "tones",
        "content_warnings",
    )
    @classmethod
    def deduplicate_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class VideoProfileSummary(StrictModel):
    summary: SummaryText = Field(
        description=(
            "Semantic retrieval용 근거 기반 한국어 콘텐츠 설명. "
            "시간순 줄거리보다 핵심 주제, 구체적 초점과 범위를 우선한다."
        )
    )


class VideoProfileDetails(StrictModel):
    profile: CanonicalProfile
    descriptive_context: DescriptiveContext


class VideoProfileDocument(StrictModel):
    ontology_version: Literal["viewing-ontology-contract/v3"] = (
        "viewing-ontology-contract/v3"
    )
    title: str
    channel: str
    upload_date: str
    meta_desc: str
    summary: SummaryText
    profile: CanonicalProfile
    descriptive_context: DescriptiveContext

    @field_validator("channel", "upload_date")
    @classmethod
    def require_non_empty_passthrough(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


def _literal(values: tuple[str, ...]) -> Any:
    return Literal[*values]


def build_video_profile_details_model(
    ontology_contract: dict[str, Any],
) -> type[BaseModel]:
    """Build Gemini's structured response schema from the validated contract."""

    canonical_concepts = ontology_contract["canonical_concepts"]
    closed_axes = ontology_contract["closed_axes"]

    profile_fields: dict[str, tuple[Any, Any]] = {}
    for facet_spec in ontology_contract["facet_specs"]:
        facet = facet_spec["facet"]
        field_contract = facet_spec
        if field_contract["value_kind"] == "canonical_concept":
            ids = tuple(item["concept_id"] for item in canonical_concepts[facet])
            model_name = "".join(part.title() for part in facet.split("_"))
            concept_ref = create_model(
                f"{model_name}ConceptRef",
                concept_id=(_literal(ids), ...),
                extension_label=(ShortLabel | None, None),
                __base__=StrictModel,
            )
            profile_fields[facet] = (
                list[concept_ref],
                Field(
                    min_length=field_contract["min_items"],
                    max_length=field_contract["max_items"],
                    description=(
                        f"Canonical {facet} concepts ordered by importance; "
                        "extensions must stay within the selected concept boundary."
                    ),
                ),
            )
        elif field_contract["value_kind"] == "closed_enum":
            values = tuple(closed_axes[facet])
            if field_contract["allow_unknown"]:
                values = (*values, "unknown")
            profile_fields[facet] = (
                list[_literal(values)],
                Field(
                    min_length=field_contract["min_items"],
                    max_length=field_contract["max_items"],
                    description=f"Canonical {facet} values ordered by importance.",
                ),
            )
        elif field_contract["value_kind"] == "ordinal":
            presentation_fields: dict[str, tuple[Any, Any]] = {}
            for dimension in field_contract["dimensions"]:
                values = tuple(closed_axes[facet][dimension])
                if field_contract["allow_unknown"]:
                    values = (*values, "unknown")
                presentation_fields[dimension] = (
                    _literal(values),
                    Field(description=f"Canonical presentation axis: {dimension}."),
                )
            presentation_model = create_model(
                "CanonicalPresentationResponse",
                **presentation_fields,
                __base__=StrictModel,
            )
            profile_fields[facet] = (presentation_model, ...)

    profile_model = create_model(
        "CanonicalViewingProfileResponse",
        **profile_fields,
        __base__=StrictModel,
    )
    return create_model(
        "VideoProfileDetailsResponse",
        profile=(profile_model, ...),
        descriptive_context=(DescriptiveContext, ...),
        __base__=StrictModel,
    )
