"""Parse complete line-form graphs, repairing delimiters without inventing content."""

from __future__ import annotations

import re

from extraction.semantic_graph.json_repair import (
    GraphParseResult,
    parse_or_repair_graph as parse_json_graph,
)

GRAPH_PARSER_VERSION = "graph-text/v6"
_REQUIRED_SECTIONS = ("Entities", "Relations", "End")
# Context is accepted only for compatibility with earlier prompts and saved outputs.
_SECTIONS = (*_REQUIRED_SECTIONS, "Context", "Actions")
_ID = re.compile(r"[\w.-]+")
_BULLET = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)")
_DELIMITER = re.compile(r"-+>|=+>|→|⇒|－＞|(?<=\s)[-–—](?=\s)")
_RELATION_MARK = re.compile(r"[<>=→⇒←↔⇐⟵⟷⟺－＞]|(?<=\s)[-–—](?=\s)")


class GraphTextError(ValueError):
    pass


def _header(line, *, repair):
    if not repair:
        return next((name for name in _SECTIONS if line == f"[{name}]"), None)
    match = re.fullmatch(
        r"(?:\[\s*(entities|relations|context|actions|end)\s*\]"
        r"|(entities|relations|context|actions|end)\s*[:：]"
        r"|#{1,6}\s+(entities|relations|context|actions|end)\s*[:：]?)",
        line, re.IGNORECASE,
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


def _entity(line, *, repair):
    parts = re.split(r"[:：]" if repair else ":", line, maxsplit=1)
    if len(parts) != 2 or not _ID.fullmatch(parts[0].strip()):
        raise GraphTextError("expected entity 'id: kind; attribute'")
    values = [value.strip() for value in re.split(r"[;；]" if repair else ";", parts[1])]
    if not repair and any(char in parts[1] for char in "：；"):
        raise GraphTextError("nonstandard entity delimiter")
    if repair and len(values) > 1 and not values[-1]:
        values.pop()  # One trailing delimiter; never discard an empty middle field.
    if not values or any(not value for value in values):
        raise GraphTextError("empty entity kind or attribute")
    return {"id": parts[0].strip(), "name": values[0], "attributes": values[1:]}


def _relation(line, *, repair):
    delimiters = list(_DELIMITER.finditer(line))
    if len(delimiters) != 2:
        raise GraphTextError("expected exactly two unambiguous relation delimiters")
    if not repair and any(delimiter.group() != "->" for delimiter in delimiters):
        raise GraphTextError("nonstandard relation delimiter")
    left, right = delimiters
    subject = line[:left.start()].strip()
    predicate = line[left.end():right.start()].strip()
    target = line[right.end():].strip()
    if (not _ID.fullmatch(subject) or not _ID.fullmatch(target)
            or not predicate or _RELATION_MARK.search(predicate)):
        raise GraphTextError("incomplete or ambiguous relation; expected subject -> relation -> object")
    return {"subject_id": subject, "predicate": predicate, "object_id": target}


def _parse_lines(text, *, repair):
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if _header(first, repair=repair) == "Context":
        return _parse_context_actions(text, repair=repair)
    sections = {}
    current = None
    seen = 0
    for number, source in enumerate(text.splitlines(), 1):
        line = source.strip()
        if not line:
            continue
        if current == "End":
            raise GraphTextError(f"line {number}: content after [End]")
        if _BULLET.match(line):
            if not repair:
                raise GraphTextError(f"line {number}: unexpected list marker")
            line = _BULLET.sub("", line, count=1)
        header = _header(line, repair=repair)
        if header:
            legacy_context = header == "Context" and current == "Relations"
            if not legacy_context and (seen >= len(_REQUIRED_SECTIONS)
                                       or header != _REQUIRED_SECTIONS[seen]):
                raise GraphTextError(f"line {number}: duplicate or out-of-order section {header}")
            current = header
            if header != "End":
                sections[header] = []
            if not legacy_context:
                seen += 1
            continue
        if (_header(line, repair=True) or (re.fullmatch(r"\[.*\]", line) and line != "[]")
                or line.startswith(("```", "#"))):
            raise GraphTextError(f"line {number}: unexpected section or formatting")
        if current is None:
            raise GraphTextError(f"line {number}: expected [Entities]")
        sections[current].append((number, line))
    # EOF after Relations (or legacy Context) is a valid terminator. Token-limit
    # truncation is detected from backend finish reasons by the scene executor.
    if seen < len(_REQUIRED_SECTIONS) - 1:
        raise GraphTextError(f"missing [{_REQUIRED_SECTIONS[seen]}] section or terminator")

    graph = {}
    for section, rows in sections.items():
        if not rows:
            raise GraphTextError(f"empty [{section}]; write none explicitly")
        empty = [line == "none" or repair and line.casefold() in {"none", "[]"}
                 for _, line in rows]
        if any(empty):
            if len(rows) != 1:
                raise GraphTextError(f"[{section}]: none mixed with other items")
            graph[section.lower()] = []
            continue
        values = []
        for number, line in rows:
            try:
                if section == "Entities":
                    values.append(_entity(line, repair=repair))
                elif section == "Relations":
                    values.append(_relation(line, repair=repair))
                else:
                    if not repair and line.casefold() in {"none", "[]"}:
                        raise GraphTextError("nonstandard empty section marker")
                    values.append(line)
            except GraphTextError as exc:
                raise GraphTextError(f"line {number}: {exc}") from exc
        graph[section.lower()] = values
    return graph


def _action(line, *, repair):
    parts = re.split(r"[;；]" if repair else ";", line)
    if len(parts) != 2:
        raise GraphTextError("expected actor - action - target; tool (use none for missing values)")
    body, tool = (part.strip() for part in parts)
    if repair:
        relation = _relation(body, repair=True)
        actor, action, target = (relation[key] for key in ("subject_id", "predicate", "object_id"))
    else:
        fields = body.split(" - ")
        if len(fields) != 3:
            raise GraphTextError("expected exactly two ' - ' action delimiters")
        actor, action, target = (part.strip() for part in fields)
        if not action or _RELATION_MARK.search(action):
            raise GraphTextError("empty or ambiguous action")
    if any(not _ID.fullmatch(value) for value in (actor, target, tool)):
        raise GraphTextError("expected endpoint IDs or none")
    def reference(value):
        return None if value == "none" or repair and value.casefold() == "none" else value
    return {"actor": reference(actor), "action": action,
            "target": reference(target), "tool": reference(tool)}


def _parse_context_actions(text, *, repair):
    required = ("Context", "Entities", "Actions")
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
        elif (current is None or _header(line, repair=True)
              or re.fullmatch(r"\[.*\]", line) and line != "[]"):
            raise GraphTextError(f"line {number}: unexpected section or formatting")
        else:
            sections[current].append(line)
    if seen < len(required):
        raise GraphTextError(f"missing [{required[seen]}]")
    context = {}
    for line in sections["Context"]:
        parts = re.split(r"[:：]" if repair else ":", line, maxsplit=1)
        if len(parts) != 2:
            raise GraphTextError("expected context key: value")
        key, value = (part.strip() for part in parts)
        if key not in {"medium", "format", "topics"} or key in context or not value:
            raise GraphTextError("missing, unknown or duplicate context field")
        if key == "topics":
            context[key] = ([] if value == "none" else
                            [part.strip() for part in re.split(r"[;；]" if repair else ";", value)])
            if any(not part or part.casefold() == "none" for part in context[key]):
                raise GraphTextError("empty topic or none mixed with topics")
        else:
            context[key] = value
    if set(context) != {"medium", "format", "topics"}:
        raise GraphTextError("missing context field")
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
            graph[section.lower()] = [parse(line, repair=repair) for line in rows]
    return graph


def parse_or_repair_graph(text: str) -> GraphParseResult:
    """Consume the whole response; retain JSON compatibility for complete old outputs."""
    raw = str(text or "")
    if not raw.strip():
        return GraphParseResult(None, error="empty VLM output")
    try:
        unwrapped = _unwrap(raw)
        if unwrapped.startswith("{"):
            parsed = parse_json_graph(unwrapped)
            if parsed.graph is not None and unwrapped != raw.strip():
                return GraphParseResult(parsed.graph, parse_mode="repaired")
            return parsed
        try:
            graph = _parse_lines(raw, repair=False)
        except GraphTextError:
            graph = _parse_lines(unwrapped, repair=True)
            return GraphParseResult(graph, parse_mode="repaired")
        return GraphParseResult(graph, parse_mode="native")
    except GraphTextError as exc:
        return GraphParseResult(None, error=f"graph text repair failed: {exc}")
