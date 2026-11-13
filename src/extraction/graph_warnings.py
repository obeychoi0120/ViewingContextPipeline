"""Scene outcome tags. Errors carry no output; a warning keeps the output.

A usable warning (REPEAT_OCCURRED) is counted separately but the scene is used and is not
generated again on a rerun; within one run a later penalty pass still retries it. Every other
warning excludes the scene and is generated again.
"""

PARSE_FAILED = "PARSE_FAILED"  # [Context], [Entities] or [Actions] cannot be separated.
TOKEN_LIMIT = "TOKEN_LIMIT"  # Cut at the output token limit; the text is never parsed.
REPEAT_OCCURRED = "REPEAT_OCCURRED"  # Stopped before repeating a line; the graph before it is used.
# One tag per scene; when several apply, the first in this order wins.
WARNING_TAGS = (TOKEN_LIMIT, PARSE_FAILED, REPEAT_OCCURRED)
USABLE_TAGS = frozenset({REPEAT_OCCURRED})
DESCRIPTION_WARNING_TAGS = (TOKEN_LIMIT,)
OUTCOMES = ("success", "repeat_occurred", "parse_failed", "token_limit", "error")
USABLE_OUTCOMES = ("success", "repeat_occurred")
DESCRIPTION_OUTCOMES = ("success", "token_limit", "error")

# Tags written by earlier parsers, read as their current equivalent.
_LEGACY_TAGS = {
    "PARSE_ERROR": PARSE_FAILED,
    "MISSING_REQUIRED": PARSE_FAILED,
    "TRUNCATION_RECOVERED": TOKEN_LIMIT,
}
# TRASH_INFO (no usable information) is no longer checked; such scenes are used.
_REMOVED_TAGS = {"INVALID_ACTION_SYNTAX", "INVALID_REFERENCE", "DUPLICATE_ENTITY_ID", "TRASH_INFO"}
# Errors that earlier versions stored for outputs cut at the token limit.
LEGACY_TRUNCATION_ERRORS = {
    "graph: output truncated at token limit",
    "description: output truncated at token limit",
}


def warning_tags(values):
    """Read stored tags as the single current tag; discard retired ones."""
    if not isinstance(values, list) or any(
        not isinstance(v, str) or v not in {*WARNING_TAGS, *_LEGACY_TAGS, *_REMOVED_TAGS}
        for v in values
    ):
        raise ValueError("warning must be an array of scene warning tags")
    tags = {_LEGACY_TAGS.get(value, value) for value in values if value not in _REMOVED_TAGS}
    return primary_tags(tags)


def primary_tags(tags):
    """The highest-priority tag as a list, or []."""
    return next(([tag] for tag in WARNING_TAGS if tag in tags), [])


def blocking_tags(tags):
    """Tags that exclude a scene from inputs and make it generate again."""
    return [tag for tag in tags or [] if tag not in USABLE_TAGS]


def needs_generation(record):
    """A missing, failed or warned scene, except for usable warnings."""
    return record is None or "error" in record or bool(blocking_tags(record.get("semantic_warnings")))


def outcome(record):
    """Name the single outcome of an in-memory scene record (None counts as an error)."""
    if record is None or "error" in record:
        return "error"
    tags = record.get("semantic_warnings") or record.get("warning") or []
    return tags[0].lower() if tags else "success"
