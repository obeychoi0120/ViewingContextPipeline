"""One paragraph, up to 200 whitespace-delimited English words."""

from __future__ import annotations
import re

SUMMARY_SCHEMA_VERSION = "video-summary/v4"


class SummaryContractError(ValueError):
    pass


def inspect_summary(text: str) -> tuple[str, list[str]]:
    if not isinstance(text, str) or not text.strip():
        return "", ["empty"]
    lines = text.strip().splitlines()
    violations = []
    if any(re.match(r"^\s*(?:[-*#] |[0-9]+[.)] |```)", line) for line in lines):
        violations.append("list_or_markup")
    if (
        text.lstrip().startswith(("{", "["))
        or sum(bool(re.match(r"^\s*[A-Za-z][A-Za-z_ ]{0,50}:", line)) for line in lines) >= 2
    ):
        violations.append("structured_fields")
    if re.search(r"\n\s*\n", text.strip()):
        violations.append("multiple_paragraphs")
    normalized = " ".join(text.split())
    if len(normalized.split()) > 200:
        violations.append("over_200_words")
    return normalized, violations


def validate_summary(text: str) -> str:
    normalized, violations = inspect_summary(text)
    if violations:
        raise SummaryContractError(", ".join(violations))
    return normalized
