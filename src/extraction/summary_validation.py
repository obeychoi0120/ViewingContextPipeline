"""Require nonempty summary text; formatting and word counts are prompt guidance."""

from __future__ import annotations

SUMMARY_SCHEMA_VERSION = "video-summary/v4"


class SummaryContractError(ValueError):
    pass


def inspect_summary(text: str) -> tuple[str, list[str]]:
    if not isinstance(text, str) or not text.strip():
        return "", ["empty"]
    return text.strip(), []


def validate_summary(text: str) -> str:
    normalized, violations = inspect_summary(text)
    if violations:
        raise SummaryContractError(", ".join(violations))
    return normalized
