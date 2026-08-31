from __future__ import annotations

SUMMARY_SECTIONS = (
    "setting_and_environments",
    "main_characters_and_objects",
    "chronological_events",
    "relations",
    "visual_atmosphere",
    "visible_affect",
    "semantic_topics",
)

_SECTION_LABELS = {
    "setting_and_environments": "Setting and environments",
    "main_characters_and_objects": "Main characters and objects",
    "chronological_events": "Chronological events",
    "relations": "Relations",
    "visual_atmosphere": "Visual atmosphere",
    "visible_affect": "Visible affect",
    "semantic_topics": "Semantic topics",
}


class SummaryContractError(ValueError):
    pass


def parse_summary_sections(text: str) -> dict[str, str]:
    """Parse exactly seven ``field: value`` lines in canonical order."""
    raw = str(text or "").strip()
    if not raw:
        raise SummaryContractError("structured video summary must not be empty")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != len(SUMMARY_SECTIONS):
        raise SummaryContractError(
            "structured video summary must contain exactly seven non-empty labeled lines"
        )
    sections: dict[str, str] = {}
    for expected, line in zip(SUMMARY_SECTIONS, lines):
        label, separator, value = line.partition(":")
        if not separator:
            raise SummaryContractError(
                f"structured video summary line is missing ':' for {expected}"
            )
        normalized_label = label.strip()
        if normalized_label != expected:
            raise SummaryContractError(
                "structured video summary labels must use canonical order: "
                f"expected={expected!r} actual={normalized_label!r}"
            )
        sections[expected] = value.strip()
    if not any(sections.values()):
        raise SummaryContractError("structured video summary must contain visible evidence")
    return sections


def serialize_summary_sections(sections: dict[str, str]) -> str:
    """Render validated sections as stable newline-delimited text for BGE."""
    if tuple(sections) != SUMMARY_SECTIONS:
        raise SummaryContractError("summary sections must use the canonical field order")
    parts = [
        f"{_SECTION_LABELS[name]}: {_with_terminal_punctuation(value)}"
        for name, value in sections.items()
        if value
    ]
    if not parts:
        raise SummaryContractError("structured video summary must contain visible evidence")
    return "\n".join(parts)


def _with_terminal_punctuation(value: str) -> str:
    return value if value.endswith((".", "!", "?")) else f"{value}."
