from __future__ import annotations

import json
import re
from typing import Any


SUMMARY_VALIDATION_VERSION = "summary-soft-validation/v2"
SUMMARY_WORD_GUIDANCE_MAX = 256
SUMMARY_SECTIONS = (
    "setting_and_environments",
    "main_characters_and_objects",
    "chronological_events",
    "relations",
    "affect_or_topic",
)

_SECTION_LABELS = {
    "setting_and_environments": "Setting and environments",
    "main_characters_and_objects": "Main characters and objects",
    "chronological_events": "Chronological events",
    "relations": "Relations",
    "affect_or_topic": "Affect or topic",
}


class SummaryContractError(ValueError):
    pass


def parse_summary_sections(text: str) -> dict[str, str]:
    """Parse the model-authored five-field JSON without semantic reclassification."""
    raw = str(text or "").strip()
    if not raw:
        raise SummaryContractError("structured video summary must not be empty")
    value = _load_summary_object(raw)
    if value is None:
        raise SummaryContractError("structured video summary must be one JSON object")
    if set(value) != set(SUMMARY_SECTIONS):
        missing = sorted(set(SUMMARY_SECTIONS) - set(value))
        extra = sorted(set(value) - set(SUMMARY_SECTIONS))
        raise SummaryContractError(
            f"structured video summary fields mismatch: missing={missing} extra={extra}"
        )
    if any(not isinstance(value[name], str) for name in SUMMARY_SECTIONS):
        raise SummaryContractError("structured video summary fields must all be strings")
    sections = {name: value[name].strip() for name in SUMMARY_SECTIONS}
    if not any(sections.values()):
        raise SummaryContractError("structured video summary must contain visible evidence")
    return sections


def serialize_summary_sections(sections: dict[str, str]) -> str:
    """Render validated sections as one stable natural-language paragraph for BGE."""
    if tuple(sections) != SUMMARY_SECTIONS:
        raise SummaryContractError("summary sections must use the canonical field order")
    parts = [
        f"{_SECTION_LABELS[name]}: {_with_terminal_punctuation(value)}"
        for name, value in sections.items()
        if value
    ]
    if not parts:
        raise SummaryContractError("structured video summary must contain visible evidence")
    return " ".join(parts)


def summary_soft_warnings(text: str) -> list[str]:
    """Report advisory summary-length deviations without rejecting output."""
    word_count = len(str(text or "").strip().split())
    if word_count <= SUMMARY_WORD_GUIDANCE_MAX:
        return []
    return [
        "summary_word_guidance_exceeded: "
        f"observed={word_count} guidance_max={SUMMARY_WORD_GUIDANCE_MAX}"
    ]


def _load_summary_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE
        )
        if match.strip()
    )
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _with_terminal_punctuation(value: str) -> str:
    return value if value.endswith((".", "!", "?")) else f"{value}."
