from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract one JSON object from fenced, noisy, or truncated model output."""

    if not text or not text.strip():
        return None
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, re.sub(r",(\s*[}\]])", r"\1", text)):
        parsed = _balanced_json(candidate)
        if parsed is not None:
            return parsed
    return _repair_truncated_json(text)


def _balanced_json(text: str) -> dict[str, Any] | None:
    start = -1
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if not stack:
                start = index
            stack.append(char)
        elif char == "}" and stack:
            stack.pop()
            if not stack and start >= 0:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    fragment = re.sub(r",(\s*[}\]])", r"\1", text)
    start = fragment.find("{")
    if start < 0:
        return None
    fragment = fragment[start:].rstrip()
    fragment = re.sub(r',\s*"[^"]*$', "", fragment)
    fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)
    fragment = re.sub(r",\s*$", "", fragment)
    braces = brackets = 0
    in_string = escaped = False
    for char in fragment:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        braces += (char == "{") - (char == "}")
        brackets += (char == "[") - (char == "]")
    fragment += "]" * max(0, brackets) + "}" * max(0, braces)
    try:
        value = json.loads(re.sub(r",(\s*[}\]])", r"\1", fragment))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
