from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
from typing import Any, Literal


JSON_REPAIR_VERSION = "semantic-graph-json-repair/v1"


@dataclass(frozen=True)
class GraphParseResult:
    graph: dict[str, Any] | None
    status: Literal["parsed", "repaired", "failed"]
    error: str | None = None


def parse_or_repair_graph(text: str) -> GraphParseResult:
    """Parse one JSON object, then apply one deterministic repair pass."""
    raw = str(text or "")
    if not raw.strip():
        return GraphParseResult(None, "failed", "empty VLM output")
    parsed = _parse_json_candidates(raw)
    if parsed is not None:
        return GraphParseResult(parsed, "parsed")
    repaired = repair_graph_json_once(raw)
    if repaired is not None:
        return GraphParseResult(repaired, "repaired")
    return GraphParseResult(None, "failed", "JSON repair failed")


def repair_graph_json_once(text: str) -> dict[str, Any] | None:
    """Repair syntax only; never add semantic fields or alter references."""
    normalized = str(text or "").translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    )
    candidates = _candidate_texts(normalized)

    without_trailing_commas = [
        re.sub(r",(\s*[}\]])", r"\1", candidate)
        for candidate in candidates
    ]
    parsed = _parse_json_candidates("\n".join(without_trailing_commas))
    if parsed is not None:
        return parsed

    for candidate in candidates:
        parsed = _parse_python_dict(candidate)
        if parsed is not None:
            return parsed

    truncated = _close_truncated_json(normalized)
    if truncated is None:
        return None
    return _load_json_dict(re.sub(r",(\s*[}\]])", r"\1", truncated))


def _candidate_texts(text: str) -> list[str]:
    fenced = re.findall(r"```(?:json|python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [item.strip() for item in fenced if item.strip()]
    candidates.extend(_object_candidates(text))
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    return _unique(candidates)


def _parse_json_candidates(text: str) -> dict[str, Any] | None:
    for candidate in _candidate_texts(text):
        parsed = _load_json_dict(candidate)
        if parsed is not None:
            return parsed
    return None


def _object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
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
            if not stack and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return sorted(_unique(candidates), key=len, reverse=True)


def _parse_python_dict(text: str) -> dict[str, Any] | None:
    for candidate in [*_object_candidates(text), text.strip()]:
        try:
            value = ast.literal_eval(candidate)
            normalized = json.loads(json.dumps(value, ensure_ascii=False))
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(normalized, dict):
            return normalized
    return None


def _close_truncated_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:].split("```", 1)[0].rstrip()
    fragment = re.sub(r",\s*$", "", fragment)
    fragment = re.sub(r',\s*"[^"\\]*$', "", fragment)

    stack: list[str] = []
    in_string = False
    escaped = False
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
        if char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
    if escaped:
        fragment = fragment[:-1]
    if in_string:
        fragment += '"'
    fragment = re.sub(r",(\s*)$", r"\1", fragment)
    fragment += "".join("]" if opener == "[" else "}" for opener in reversed(stack))
    return fragment


def _load_json_dict(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
