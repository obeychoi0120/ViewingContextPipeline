"""Stable tags for Graph structure recovery; generation failures use their own log."""

MISSING_REQUIRED = "MISSING_REQUIRED"
INVALID_ACTION_SYNTAX = "INVALID_ACTION_SYNTAX"
INVALID_REFERENCE = "INVALID_REFERENCE"
DUPLICATE_ENTITY_ID = "DUPLICATE_ENTITY_ID"
PARSE_ERROR = "PARSE_ERROR"
WARNING_TAGS = (MISSING_REQUIRED, INVALID_ACTION_SYNTAX, INVALID_REFERENCE,
                DUPLICATE_ENTITY_ID, PARSE_ERROR)


def warning_tags(values):
    """Validate and deduplicate persisted tags without treating descriptions as tags."""
    if not isinstance(values, list) or any(not isinstance(v, str) or v not in WARNING_TAGS
                                           for v in values):
        raise ValueError("warning must be an array of Graph warning tags")
    return list(dict.fromkeys(values))
