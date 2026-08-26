from __future__ import annotations


SUMMARY_VALIDATION_VERSION = "summary-soft-validation/v1"
SUMMARY_WORD_GUIDANCE_MAX = 150


def summary_soft_warnings(text: str) -> list[str]:
    """Report advisory summary-length deviations without rejecting output."""
    word_count = len(str(text or "").strip().split())
    if word_count <= SUMMARY_WORD_GUIDANCE_MAX:
        return []
    return [
        "summary_word_guidance_exceeded: "
        f"observed={word_count} guidance_max={SUMMARY_WORD_GUIDANCE_MAX}"
    ]
