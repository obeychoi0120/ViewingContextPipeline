from __future__ import annotations

import ast
from dataclasses import dataclass
import io
import json
import tokenize
from typing import Any


@dataclass(frozen=True)
class GraphParseResult:
    graph: dict[str, Any] | None
    error: str | None = None
    parse_mode: str | None = None
    warning: tuple[str, ...] = ()


def parse_or_repair_graph(text: str) -> GraphParseResult:
    """Parse one JSON object, then apply one deterministic repair pass."""
    raw = str(text or "")
    if not raw.strip():
        return GraphParseResult(graph=None, error="empty VLM output", warning=("MISSING_REQUIRED",))
    parsed = _load_json_dict(raw)
    if parsed is not None:
        return GraphParseResult(graph=parsed, parse_mode="native")
    repaired = repair_graph_json_once(raw)
    if repaired is not None:
        return GraphParseResult(graph=repaired, parse_mode="repaired")
    return GraphParseResult(graph=None, error="JSON repair failed", warning=("PARSE_ERROR",))


def repair_graph_json_once(text: str) -> dict[str, Any] | None:
    """Repair punctuation in a complete object; never close a truncated response."""
    normalized = _normalize_punctuation(str(text or "").strip())
    parsed = _load_json_dict(normalized)
    if parsed is not None:
        return parsed
    try:
        # Python literal syntax must not silently join adjacent strings or overwrite keys.
        previous = None
        for token in tokenize.generate_tokens(io.StringIO(normalized).readline):
            if token.type == tokenize.COMMENT or token.type == previous == tokenize.STRING:
                return None
            if token.type not in (tokenize.NL, tokenize.NEWLINE):
                previous = token.type
        tree = ast.parse(normalized, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [ast.literal_eval(key) for key in node.keys]
                if len(keys) != len(set(keys)):
                    return None
        value = ast.literal_eval(tree)
        normalized_value = json.loads(json.dumps(value, ensure_ascii=False))
    except (SyntaxError, ValueError, TypeError, tokenize.TokenError, RecursionError):
        return None
    return normalized_value if isinstance(normalized_value, dict) else None


def _normalize_punctuation(text):
    # Only syntax punctuation outside string values may be removed or replaced.
    output = []
    closing = None
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if closing is not None:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == closing:
                output.append(quote)
                closing = None
            else:
                output.append(char)
        elif char in ('"', "'", "“", "‘"):
            closing, quote = {"“": ("”", '"'), "‘": ("’", "'")}.get(char, (char, char))
            output.append(quote)
        elif char == "," and text[index + 1:].lstrip().startswith(("}", "]")):
            continue
        else:
            output.append(char)
    return "".join(output)


def _load_json_dict(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result
