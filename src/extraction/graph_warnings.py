"""Stable tags for Graph structure recovery; failed Scene rows carry an error instead."""

MISSING_REQUIRED = "MISSING_REQUIRED"
PARSE_FAILED = "PARSE_FAILED"
WARNING_TAGS = (MISSING_REQUIRED, PARSE_FAILED)
# A usable Graph salvaged from output cut at the token limit. It marks the scene as
# recovered; unlike WARNING_TAGS it neither blocks reuse nor excludes the scene.
TRUNCATION_RECOVERED = "TRUNCATION_RECOVERED"
_REMOVED_TAGS = {"INVALID_ACTION_SYNTAX", "INVALID_REFERENCE", "DUPLICATE_ENTITY_ID"}


def warning_tags(values):
    """Read current tags; discard retired warnings and rename the old parse failure tag."""
    if not isinstance(values, list) or any(
        not isinstance(v, str)
        or v not in {*WARNING_TAGS, TRUNCATION_RECOVERED, *_REMOVED_TAGS, "PARSE_ERROR"}
        for v in values
    ):
        raise ValueError("warning must be an array of Graph warning tags")
    return list(dict.fromkeys(
        PARSE_FAILED if value == "PARSE_ERROR" else value
        for value in values if value not in _REMOVED_TAGS
    ))


def blocking_tags(values):
    """Tags that make a scene unusable; the recovery marker is not one of them."""
    return [value for value in values or () if value != TRUNCATION_RECOVERED]


def is_recovered(values):
    return TRUNCATION_RECOVERED in (values or ()) and not blocking_tags(values)
