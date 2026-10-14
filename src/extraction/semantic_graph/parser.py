"""Parse complete line-form graphs, repairing delimiters without inventing content."""

from __future__ import annotations

import re

from extraction.graph_warnings import MISSING_REQUIRED, PARSE_FAILED
from extraction.structured_output import OutputValidationError, normalize_graph_references

from extraction.semantic_graph.json_repair import (
    GraphParseResult,
    parse_or_repair_graph as parse_json_graph,
)

GRAPH_PARSER_VERSION = "graph-text/v14"
_SECTIONS = ("Context", "Entities", "Actions", "End")
_ID = re.compile(r"[\w.-]+")
_BULLET = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)")
_DELIMITER = re.compile(r"-+>|=+>|→|⇒|－＞|(?<=\s)[-–—](?=\s)")
_ACTION_MARK = re.compile(r"[<>=→⇒←↔⇐⟵⟷⟺－＞]|(?<=\s)[-–—](?=\s)")


class GraphTextError(ValueError):
    def __init__(self, message, tag=PARSE_FAILED):
        super().__init__(message)
        self.tag = tag


def _header(line, *, repair):
    if not repair:
        return next((name for name in _SECTIONS if line == f"[{name}]"), None)
    match = re.fullmatch(
        r"(?:\[\s*(entities|context|actions|end)\s*\]"
        r"|(entities|context|actions|end)\s*[:：]"
        r"|#{1,6}\s+(entities|context|actions|end)\s*[:：]?)",
        line,
        re.IGNORECASE,
    )
    return next(value for value in match.groups() if value).capitalize() if match else None


def _unwrap(text):
    text = text.strip().lstrip("\ufeff").strip()
    lines = text.splitlines()
    if lines and re.fullmatch(r"```[\w-]*\s*", lines[0]):
        if len(lines) < 2 or lines[-1].strip() != "```":
            raise GraphTextError("unclosed code fence or content after code fence")
        text = "\n".join(lines[1:-1]).strip()
    return text


def _entity(line, *, repair, named=False):
    parts = re.split(r"[:：]" if repair else ":", line, maxsplit=1)
    if named:
        parts = [re.split(r"[;；]", line, maxsplit=1)[0].strip(), line]
        if not _valid_reference(parts[0], named=True) or parts[0].casefold() in {"none", "unknown"}:
            raise GraphTextError("expected a non-reserved entity name")
    if len(parts) != 2 or (not named and not _ID.fullmatch(parts[0].strip())):
        raise GraphTextError("expected entity 'id: kind; attribute'")
    values = [value.strip() for value in re.split(r"[;；]" if repair else ";", parts[1])]
    if not repair and any(char in parts[1] for char in "：；"):
        raise GraphTextError("nonstandard entity delimiter")
    if repair and len(values) > 1 and not values[-1]:
        values.pop()  # One trailing delimiter; never discard an empty middle field.
    if not values or not values[0]:
        raise GraphTextError("missing entity kind", MISSING_REQUIRED)
    if any(not value for value in values):
        raise GraphTextError("empty entity kind or attribute")
    return {"id": parts[0].strip(), "name": values[0], "attributes": values[1:]}


def _valid_reference(value, *, named):
    if not named:
        return bool(_ID.fullmatch(value))
    return bool(value and not re.search(r"[;；:：\r\n]", value) and not _ACTION_MARK.search(value))


def _action_endpoints(line, *, named=False):
    delimiters = list(_DELIMITER.finditer(line))
    if len(delimiters) != 2:
        raise GraphTextError("expected exactly two unambiguous action delimiters")
    left, right = delimiters
    subject = line[: left.start()].strip()
    predicate = line[left.end() : right.start()].strip()
    target = line[right.end() :].strip()
    if (
        not _valid_reference(subject, named=named)
        or not _valid_reference(target, named=named)
        or not predicate
        or _ACTION_MARK.search(predicate)
    ):
        raise GraphTextError(
            "incomplete or ambiguous action; expected actor - action - target"
        )
    return subject, predicate, target


def _action(line, *, repair, named=False):
    parts = re.split(r"[;；]" if repair else ";", line)
    if len(parts) != 3:
        raise GraphTextError("expected actor - action - target; tool; place")
    body, tool, place = (part.strip() for part in parts)
    if repair:
        actor, action, target = _action_endpoints(body, named=named)
    else:
        fields = body.split(" - ")
        if len(fields) != 3:
            raise GraphTextError("expected exactly two ' - ' action delimiters")
        actor, action, target = (part.strip() for part in fields)
        if not action or _ACTION_MARK.search(action):
            raise GraphTextError("empty or ambiguous action")
    if any(not _valid_reference(value, named=named) for value in (actor, target, tool, place)):
        raise GraphTextError("expected entity references, none, or unknown")

    def reference(value):
        token = value.casefold() if repair else value
        return None if token == "none" else "unknown" if token == "unknown" else value

    return {
        "actor": reference(actor),
        "action": action,
        "target": reference(target),
        "tool": reference(tool),
        "place": reference(place),
    }


def _parse_context_actions(text, *, repair):
    required = ("Context", "Entities", "Actions")
    headers = {
        _header(_BULLET.sub("", line.strip(), count=1) if repair else line.strip(), repair=repair)
        for line in text.splitlines()
    }
    missing = [name for name in required if name not in headers]
    if missing:
        raise GraphTextError(f"missing [{missing[0]}]", MISSING_REQUIRED)
    order = (*required, "End")
    sections, current, seen = {}, None, 0
    for number, source in enumerate(text.splitlines(), 1):
        line = source.strip()
        if not line:
            continue
        if current == "End":
            raise GraphTextError(f"line {number}: content after [End]")
        if repair:
            line = _BULLET.sub("", line, count=1)
        header = _header(line, repair=repair)
        if header:
            if seen >= len(order) or header != order[seen]:
                raise GraphTextError(f"line {number}: duplicate or out-of-order section {header}")
            current = header
            sections[header] = []
            seen += 1
        elif (
            current is None
            or _header(line, repair=True)
            or re.fullmatch(r"\[.*\]", line)
            and line != "[]"
        ):
            raise GraphTextError(f"line {number}: unexpected section or formatting")
        else:
            sections[current].append(line)
    if seen < len(required):
        raise GraphTextError(f"missing [{required[seen]}]", MISSING_REQUIRED)
    context = {}
    for line in sections["Context"]:
        parts = re.split(r"[:：]" if repair else ":", line, maxsplit=1)
        if len(parts) != 2:
            raise GraphTextError("expected context key: value")
        key, value = (part.strip() for part in parts)
        if key not in {"medium", "format", "topics"} or key in context:
            raise GraphTextError("unknown or duplicate context field")
        if not value:
            raise GraphTextError("missing context value", MISSING_REQUIRED)
        if key == "topics":
            context[key] = (
                []
                if value == "none"
                else [part.strip() for part in re.split(r"[;；]" if repair else ";", value)]
            )
            if any(not part or part.casefold() == "none" for part in context[key]):
                raise GraphTextError("empty topic or none mixed with topics")
        else:
            context[key] = value
    if set(context) != {"medium", "format", "topics"}:
        raise GraphTextError("missing context field", MISSING_REQUIRED)
    # A whole entity section uses either v9 names or legacy id: kind rows.
    entity_styles = {bool(re.search(r"[:：]", re.split(r"[;；]", row, maxsplit=1)[0]))
                     for row in sections["Entities"]}
    if len(entity_styles) > 1:
        raise GraphTextError("mixed entity declaration formats")
    named = entity_styles == {False}
    graph = dict(context)
    for section, parse in (("Entities", _entity), ("Actions", _action)):
        rows = sections[section]
        if not rows:
            raise GraphTextError(f"empty [{section}]; write none explicitly")
        empty = [line == "none" or repair and line.casefold() in {"none", "[]"} for line in rows]
        if any(empty):
            if len(rows) != 1:
                raise GraphTextError(f"[{section}]: none mixed with other items")
            graph[section.lower()] = []
        else:
            graph[section.lower()] = [parse(line, repair=repair, named=named) for line in rows]
    return graph, named


def parse_or_repair_graph(text: str) -> GraphParseResult:
    """Parse complete Actions graphs; undeclared role IDs become unknown."""
    raw = str(text or "")
    if not raw.strip():
        return GraphParseResult(None, error="empty VLM output", warning=(MISSING_REQUIRED,))
    try:
        unwrapped = _unwrap(raw)
        if unwrapped.startswith("{"):
            parsed = parse_json_graph(unwrapped)
            if parsed.graph is not None:
                mode = "repaired" if unwrapped != raw.strip() else parsed.parse_mode
                return _validated_graph(parsed.graph, mode)
            return parsed
        try:
            graph, named = _parse_context_actions(raw, repair=False)
            mode = "native"
        except GraphTextError:
            graph, named = _parse_context_actions(unwrapped, repair=True)
            mode = "repaired"
        return _validated_graph(graph, mode, named=named)
    except OutputValidationError as exc:
        return GraphParseResult(None, error=str(exc), warning=tuple(exc.tags))
    except GraphTextError as exc:
        return GraphParseResult(None, error=f"graph text repair failed: {exc}", warning=(exc.tag,))


def _validated_graph(graph, mode, *, named=False):
    graph, changed = normalize_graph_references(graph)
    if named:
        # Resolve exact names before assigning IDs so an undeclared e1 cannot
        # accidentally reference the first generated entity. Keep duplicate
        # declarations and last-reference behavior consistent with legacy input.
        ids = {entity["id"]: f"e{index}" for index, entity in enumerate(graph["entities"], 1)}
        for index, entity in enumerate(graph["entities"], 1):
            entity["id"] = f"e{index}"
        for action in graph["actions"]:
            for role in ("actor", "target", "tool", "place"):
                value = action[role]
                action[role] = value if value in (None, "unknown") else ids[value]
    return GraphParseResult(
        graph,
        parse_mode="repaired" if changed else mode,
    )
