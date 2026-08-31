from __future__ import annotations

from typing import Any

from extraction.summary_validation import SummaryContractError, parse_summary_sections


SCENE_SCHEMA_VERSION = "scene-description/v1"
SUMMARY_SCHEMA_VERSION = "description-video-summary/v3"


class DescriptionError(RuntimeError):
    pass


def description_summary_prompt(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise DescriptionError("description summary requires scene records")
    lines: list[str] = []
    for record in records:
        if (
            record.get("schema_version") != SCENE_SCHEMA_VERSION
            or not str(record.get("description", "")).strip()
        ):
            raise DescriptionError("description summary received an invalid scene record")
        lines.append(f"Scene {record['scene_idx']}: {record['description']}")
    return template.format(scenes="\n".join(lines))


def validate_summary(text: str) -> dict[str, str]:
    try:
        return parse_summary_sections(text)
    except SummaryContractError as exc:
        raise DescriptionError(str(exc)) from exc
