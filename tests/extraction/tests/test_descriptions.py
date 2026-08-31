from __future__ import annotations

from pathlib import Path

import pytest

from extraction.descriptions import (
    DescriptionError,
    description_summary_prompt,
    validate_summary,
)
from extraction.summary_validation import SUMMARY_SECTIONS


def test_description_scene_prompt_uses_strict_visible_only_grounding() -> None:
    root = Path(__file__).resolve().parents[3]
    prompt = (root / "config/prompts/description_scene_v1.md").read_text(
        encoding="utf-8"
    )

    assert "using only facts directly" in prompt
    assert "Do not infer story, intent, identity, demographics" in prompt
    assert "purpose, audience" in prompt
    assert "cultural context, or actual" in prompt
    assert "Do not read or transcribe visible text" in prompt


def test_description_summary_preserves_scene_order_and_requires_seven_fields() -> None:
    records = [
        {"schema_version": "scene-description/v1", "scene_idx": 0, "description": "first"},
        {"schema_version": "scene-description/v1", "scene_idx": 1, "description": "second"},
    ]
    prompt = description_summary_prompt("{scenes}", records)
    assert prompt.index("Scene 0") < prompt.index("Scene 1")
    sections = {name: f"visible {name}" for name in SUMMARY_SECTIONS}
    labeled_text = "\n".join(f"{name}: {value}" for name, value in sections.items())
    assert validate_summary(labeled_text) == sections
    with pytest.raises(DescriptionError, match="exactly seven"):
        validate_summary("short")
