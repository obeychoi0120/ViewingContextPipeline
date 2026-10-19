"""Actual output token counts supplied by the backend, never text-length estimates."""


def validate_tokens(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("tokens must be a nonnegative integer or null")
    return value


def qwen_output_tokens(event):
    value = (event or {}).get("output_tokens")
    return value if type(value) is int and value >= 0 else None


def gemini_output_tokens(diagnostics):
    usage = (diagnostics or {}).get("usage_metadata") or {}
    value = usage.get("candidates_token_count", usage.get("candidatesTokenCount"))
    return value if type(value) is int and value >= 0 else None
