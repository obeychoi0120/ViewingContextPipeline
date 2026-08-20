from __future__ import annotations


BOT_CHECK_RETRY_DELAY_SEC = 5 * 60
BOT_CHECK_MAX_RETRIES = 1
BOT_CHECK_PATTERNS = (
    "not a bot",
    "sign in to confirm",
    "confirm you're not",
    "confirm you are not",
)


def is_youtube_bot_check_error(exc: BaseException) -> bool:
    message = exception_text(exc).lower().replace("’", "'")
    return any(pattern in message for pattern in BOT_CHECK_PATTERNS)


def exception_text(exc: BaseException) -> str:
    parts = [str(exc)]
    cause = getattr(exc, "__cause__", None)
    context = getattr(exc, "__context__", None)
    if cause is not None:
        parts.append(str(cause))
    if context is not None:
        parts.append(str(context))
    return " ".join(parts)
