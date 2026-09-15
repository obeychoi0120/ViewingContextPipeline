from __future__ import annotations
from extraction.summary_validation import (
    SUMMARY_SCHEMA_VERSION as SUMMARY_SCHEMA_VERSION,
    validate_summary as validate_summary,
)

SCENE_SCHEMA_VERSION = "scene-description/v2"


class DescriptionError(ValueError):
    pass


def description_summary_prompt(template, records):
    if not records:
        raise DescriptionError("description summary requires scene records")
    lines = []
    for record in records:
        if (
            record.get("schema_version") != SCENE_SCHEMA_VERSION
            or not record.get("description", "").strip()
        ):
            raise DescriptionError("invalid description observation")
        lines.append(f"Scene {record['scene_idx']}: {record['description']}")
    return template.replace("{scenes}", "\n".join(lines))
