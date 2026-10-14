"""Stable tags for Graph structure recovery; generation failures use their own log."""

MISSING_REQUIRED = "MISSING_REQUIRED"
PARSE_FAILED = "PARSE_FAILED"
WARNING_TAGS = (MISSING_REQUIRED, PARSE_FAILED)
_REMOVED_TAGS = {"INVALID_ACTION_SYNTAX", "INVALID_REFERENCE", "DUPLICATE_ENTITY_ID"}


def warning_tags(values):
    """Read current tags; discard retired warnings and rename the old parse failure tag."""
    if not isinstance(values, list) or any(
        not isinstance(v, str)
        or v not in {*WARNING_TAGS, *_REMOVED_TAGS, "PARSE_ERROR"} for v in values
    ):
        raise ValueError("warning must be an array of Graph warning tags")
    return list(dict.fromkeys(
        PARSE_FAILED if value == "PARSE_ERROR" else value
        for value in values if value not in _REMOVED_TAGS
    ))
