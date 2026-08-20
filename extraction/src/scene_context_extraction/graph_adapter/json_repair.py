from __future__ import annotations

import json
import re
from typing import Any


def preprocess_raw_text(text: str, max_entities: int = 5) -> str:
    """Pre-process raw VLM output to truncate entities array before parsing.
    The model often ignores entity count limits and generates 40+ entities,
    causing JSON truncation. This function finds the entities array in the
    raw text and truncates it to max_entities entries before JSON parsing.
    """
    if not text or max_entities <= 0:
        return text

    # Find the "entities" key and its opening bracket
    entities_match = re.search(r'"entities"\s*:\s*\[', text)
    if not entities_match:
        return text

    array_start = entities_match.end()  # position right after '['
    before_entities = text[:array_start]
    after_array_start = text[array_start:]

    # Find individual entity objects (top-level { ... } within the entities array)
    entity_starts: list[tuple[int, int]] = []
    depth = 0
    in_str = False
    esc = False
    entity_start = None

    for i, ch in enumerate(after_array_start):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            if depth == 0:
                entity_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and entity_start is not None:
                entity_starts.append((entity_start, i + 1))
                entity_start = None

    if len(entity_starts) <= max_entities:
        return text  # No truncation needed

    # Keep only the first max_entities entities
    kept_entities = after_array_start[:entity_starts[max_entities][0]]
    # Remove trailing comma
    kept_entities = kept_entities.rstrip()
    if kept_entities.endswith(','):
        kept_entities = kept_entities[:-1].rstrip()

    # Close the entities array and the rest of the JSON
    result = before_entities + kept_entities + ']'

    # Close the outer JSON object
    result += '\n}'

    return result



def extract_json(text: str) -> dict[str, Any] | None:
    """Extract one JSON object from raw VLM text."""
    if not text or not text.strip():
        return None

    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    for candidate_text in (text, re.sub(r",(\s*[}\]])", r"\1", text)):
        parsed = _balanced_json(candidate_text)
        if parsed is not None:
            return parsed

    repaired = _repair_truncated_json(text)
    if repaired is not None:
        return repaired

    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    matches.sort(key=len, reverse=True)
    for match_text in matches:
        for candidate in (match_text, re.sub(r",(\s*[}\]])", r"\1", match_text)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _balanced_json(text: str) -> dict[str, Any] | None:
    start = -1
    stack = []
    for index, char in enumerate(text):
        if char == "{":
            if not stack:
                start = index
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
            if not stack and start != -1:
                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    fixed = re.sub(r",(\s*[}\]])", r"\1", text)
    start_idx = fixed.find("{")
    if start_idx == -1:
        return None

    fragment = fixed[start_idx:].rstrip()
    fragment = re.sub(r',\s*"[^"]*$', "", fragment)
    fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)
    fragment = re.sub(r",\s*$", "", fragment)

    open_braces = 0
    open_brackets = 0
    in_str = False
    escaped = False
    for char in fragment:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if char == "{":
            open_braces += 1
        elif char == "}":
            open_braces -= 1
        elif char == "[":
            open_brackets += 1
        elif char == "]":
            open_brackets -= 1

    fragment += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    fragment = re.sub(r",(\s*[}\]])", r"\1", fragment)
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return None
