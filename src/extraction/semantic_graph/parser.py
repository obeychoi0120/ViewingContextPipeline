"""Parse complete line-form graphs, repairing delimiters without inventing content."""

from __future__ import annotations

import re

from extraction.graph_warnings import PARSE_FAILED
from extraction.structured_output import OutputValidationError, normalize_graph_references

from extraction.semantic_graph.json_repair import (
    GraphParseResult,
    parse_or_repair_graph as parse_json_graph,
)

GRAPH_PARSER_VERSION = "graph-text/v23"
_SECTIONS = ("Context", "Entities", "Actions", "End")
_ID = re.compile(r"[\w.-]+")
_BULLET = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)")
_DELIMITER = re.compile(r"-+>|=+>|→|⇒|－＞|(?<=\s)[-–—](?=\s)")
_ACTION_MARK = re.compile(r"[<>=→⇒←↔⇐⟵⟷⟺－＞]|(?<=\s)[-–—](?=\s)")
# Name-form tool and place slots are labelled and optional; these are the values of an omitted slot.
_SLOT_DEFAULTS = {"tool": "none", "place": "unknown"}
_SLOT = re.compile(r"(tool|place):\s*(.*)")
_LOOSE_SLOT = re.compile(r"(tool|place)\s*[:：]\s*(.*)", re.IGNORECASE)


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
        raise GraphTextError("missing entity kind")
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
    labels = [_SLOT.fullmatch(part.strip()) for part in parts[1:]]
    if named and not repair and len(parts) <= 3 and all(labels):
        # Labelled slots: "; tool: name; place: name", each optional.
        slots = dict(_SLOT_DEFAULTS, **{match[1]: match[2] for match in labels})
        if len({match[1] for match in labels}) != len(labels) or not all(slots.values()):
            raise GraphTextError("duplicate or empty tool/place slot")
        parts = [parts[0], slots["tool"], slots["place"]]
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


def _split_joined_actions(rows):
    """Regroup name-form actions that were joined with semicolons or lost slots.

    A piece containing an action delimiter starts an action; the pieces after it
    are its tool and place, labelled ("tool: knife") or in that order. Missing
    slots take the prompt defaults (tool none, place unknown), and
    "actor - action" keeps the action with an unknown target.
    """
    actions = []
    for row in rows:
        pieces = [piece.strip() for piece in re.split(r"[;；]", row)]
        while len(pieces) > 1 and not pieces[-1]:
            pieces.pop()
        current = None
        for piece in pieces:
            delimiters = len(_DELIMITER.findall(piece))
            if delimiters:
                if delimiters > 2:
                    raise GraphTextError("expected actor - action - target; tool; place")
                current = {"body": piece if delimiters == 2 else f"{piece} - unknown"}
                actions.append(current)
                continue
            label = _LOOSE_SLOT.fullmatch(piece)
            slot = label[1].lower() if label else next((s for s in _SLOT_DEFAULTS if s not in (current or {})), None)
            if current is None or not piece or slot is None or slot in current:
                raise GraphTextError("expected actor - action - target; tool; place")
            current[slot] = (label[2] if label else piece) or _SLOT_DEFAULTS[slot]
    return ["; ".join([action["body"], *(action.get(slot, default) for slot, default in _SLOT_DEFAULTS.items())])
            for action in actions]


# Lenient fallback: name-form output whose sections still separate by line shape.
_LOOSE_DELIMITER = re.compile(
    r"-+>|=+>|→|⇒|－＞|\s[-–—]+\s|(?<=\s)[-–—](?=\S)|(?<=\S)[-–—](?=\s)"
)
_LOOSE_HEADER = re.compile(
    r"(?:#{1,6}\s*)?(?:\[\s*(context|entities|actions|end)\s*\]"
    r"|(context|entities|actions|end)\s*(?:[:：]|$))\s*(.*)",
    re.IGNORECASE,
)
_CONTEXT_FIELD = re.compile(r"(medium|format|topics)\s*[:：]\s*(.*)", re.IGNORECASE)
_LABEL = re.compile(r"(\w+)\s*[:：]\s*(.*)")
_ID_FORM_ROW = re.compile(r"[A-Za-z]{1,3}\d+\s*[:：]")
_EMPTY = {"none", "[]"}


def _pieces(text):
    return [piece.strip() for piece in re.split(r"[;；]", text) if piece.strip()]


def _endpoints(piece):
    """actor, action, target of one action piece, or None; pieces after the target are dropped."""
    parts = [part.strip() for part in _LOOSE_DELIMITER.split(piece)]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1], (parts[2] if len(parts) > 2 else "") or "unknown"


def _is_action_row(line):
    parts = _LOOSE_DELIMITER.split(_pieces(line)[0]) if _pieces(line) else []
    return len(parts) >= 3 and all(part.strip() for part in parts[:2])


def _lenient_name(value):
    return re.sub(r"\s*[:：]\s*", " ", value).strip()


def _lenient_reference(value, names):
    """A role value: none, unknown, or a name (a colon name matches its declared entity)."""
    value = value.strip()
    token = value.casefold()
    if token == "none":
        return None
    if _lenient_name(value) in names:
        return _lenient_name(value)
    if not value or token == "unknown" or not _valid_reference(value, named=True):
        return "unknown"
    return value


def _lenient_actions(line):
    """Actions on one line; slots that cannot be told apart are dropped, never guessed."""
    actions = []
    for piece in _pieces(line):
        ends = _endpoints(piece)
        if ends:
            actions.append({"ends": ends, "labels": {}, "slots": []})
            continue
        if not actions:
            continue  # A piece before the first action belongs to none.
        label = _LABEL.fullmatch(piece)
        if label:
            key = label[1].lower()
            if key in _SLOT_DEFAULTS and label[2].strip():
                actions[-1]["labels"].setdefault(key, label[2].strip())
            continue  # Other labels (target:, 12:00) are dropped.
        actions[-1]["slots"].append(piece)
    result = []
    for action in actions:
        slots = dict(action["labels"])
        # More than two unlabelled slots is a list of entities, not tool and place.
        for piece in action["slots"] if len(action["slots"]) <= 2 else []:
            key = next((slot for slot in _SLOT_DEFAULTS if slot not in slots), None)
            if key:
                slots[key] = piece
        actor, verb, target = action["ends"]
        result.append({
            "actor": actor, "action": verb, "target": target,
            **{slot: slots.get(slot, default) for slot, default in _SLOT_DEFAULTS.items()},
        })
    return result  # Role values are resolved once all entity names are known.


def _parse_lenient(text):
    """Separate Context, Entities and Actions by line shape, or raise GraphTextError."""
    context, entity_rows, actions = {}, [], []
    answered = {"entities": False, "actions": False}
    section, started = None, False
    for source in str(text or "").splitlines():
        line = _BULLET.sub("", source.strip(), count=1).strip()
        if not line or re.fullmatch(r"```[\w-]*", line):
            continue
        header = _LOOSE_HEADER.fullmatch(line)
        if header:
            name, rest = (header[1] or header[2]).lower(), header[3].strip()
            if name == "end":
                break
            if name == "context" and (entity_rows or actions):
                break  # A second graph follows; keep the first.
            section = name
            if not rest:
                continue
            if name == "context":  # "Context: medium; format; topics"
                for key, value in zip(("medium", "format", "topics"), re.split(r"[;；]", rest)):
                    context.setdefault(key, value.strip())
                continue
            if name == "entities" and not _is_action_row(rest):  # "Entities: a; b; c"
                answered["entities"] = True
                entity_rows += [] if rest.casefold() in _EMPTY else _pieces(rest)
                continue
            line = rest
        field = _CONTEXT_FIELD.fullmatch(line)
        if field and section in (None, "context") and not started:
            context.setdefault(field[1].lower(), field[2].strip())
            continue
        started = True
        if line.casefold() in _EMPTY:
            # Without headers, the first none answers Entities and the next one Actions.
            target = section if section in answered else (
                "actions" if answered["entities"] else "entities")
            answered[target] = True
            continue
        if section == "actions" or _is_action_row(line):
            parsed = _lenient_actions(line)
            actions += parsed
            answered["actions"] |= bool(parsed)
            continue
        answered["entities"] = True
        entity_rows.append(line)
    if any(not context.get(key) for key in ("medium", "format", "topics")):
        raise GraphTextError("missing context field")
    if not answered["entities"]:
        raise GraphTextError("missing [Entities]")
    if not answered["actions"]:
        raise GraphTextError("missing [Actions]")
    if any(_ID_FORM_ROW.match(row) for row in entity_rows):
        raise GraphTextError("ID-form entities are not repaired leniently")
    entities = []
    for row in entity_rows:
        name, *attributes = [part.strip() for part in re.split(r"[;；]", row)]
        name = _lenient_name(name)
        if not name or name.casefold() in {"none", "unknown"}:
            continue
        entities.append({"id": name, "name": name,
                         "attributes": list(dict.fromkeys(part for part in attributes if part))})
    names = {entity["name"] for entity in entities}
    topics = context["topics"]
    graph = {
        "medium": context["medium"],
        "format": context["format"],
        "topics": [] if topics.casefold() == "none" else [
            topic for topic in _pieces(topics) if topic.casefold() != "none"],
        "entities": entities,
        "actions": [
            {role: value if role == "action" else _lenient_reference(value, names)
             for role, value in action.items()}
            for action in actions
        ],
    }
    return graph


def _parse_context_actions(text, *, repair):
    required = ("Context", "Entities", "Actions")
    headers = {
        _header(_BULLET.sub("", line.strip(), count=1) if repair else line.strip(), repair=repair)
        for line in text.splitlines()
    }
    missing = [name for name in required if name not in headers]
    # Name-form output sometimes starts at "medium:" without [Context], and then
    # often without [Entities] too: the context lines end where the entities begin.
    field = re.compile(r"\s*(medium|format|topics)\s*[:：]")
    headless = repair and missing in (["Context"], ["Context", "Entities"]) and bool(field.match(text))
    if headless:
        lines = text.strip().splitlines()
        if "Entities" in missing:
            split = next((i for i, line in enumerate(lines) if not field.match(line)), len(lines))
            lines.insert(split, "[Entities]")
        text = "\n".join(["[Context]", *lines])
    elif missing:
        raise GraphTextError(f"missing [{missing[0]}]")
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
        raise GraphTextError(f"missing [{required[seen]}]")
    context = {}
    for line in sections["Context"]:
        parts = re.split(r"[:：]" if repair else ":", line, maxsplit=1)
        if len(parts) != 2:
            raise GraphTextError("expected context key: value")
        key, value = (part.strip() for part in parts)
        if key not in {"medium", "format", "topics"} or key in context:
            raise GraphTextError("unknown or duplicate context field")
        if not value:
            raise GraphTextError("missing context value")
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
        raise GraphTextError("missing context field")
    # A whole entity section uses either v9 names or legacy id: kind rows.
    entity_styles = {bool(re.search(r"[:：]", re.split(r"[;；]", row, maxsplit=1)[0]))
                     for row in sections["Entities"]}
    if len(entity_styles) > 1:
        raise GraphTextError("mixed entity declaration formats")
    named = entity_styles == {False}
    if headless and not named:
        raise GraphTextError("missing [Context]")
    graph = dict(context)
    for section, parse in (("Entities", _entity), ("Actions", _action)):
        rows = sections[section]
        if not rows:
            raise GraphTextError(f"empty [{section}]; write none explicitly")
        empty = [line == "none" or repair and line.casefold() in {"none", "[]"} for line in rows]
        if repair and named and section == "Actions" and any(empty) and not all(empty):
            # A stray none beside real actions: keep the actions.
            rows = [line for line, skip in zip(rows, empty) if not skip]
            empty = [False] * len(rows)
        if any(empty):
            if len(rows) != 1:
                raise GraphTextError(f"[{section}]: none mixed with other items")
            graph[section.lower()] = []
        else:
            if repair and named and section == "Actions":
                rows = _split_joined_actions(rows)
            graph[section.lower()] = [parse(line, repair=repair, named=named) for line in rows]
    return graph, named


def parse_or_repair_graph(text: str) -> GraphParseResult:
    """Parse Actions graphs strictly, then with delimiter repair, then by line shape.

    Name-form roles missing from [Entities] are kept as attribute-less entities;
    undeclared ID-form roles still become unknown.
    """
    raw = str(text or "")
    if not raw.strip():
        return GraphParseResult(None, error="empty VLM output", warning=(PARSE_FAILED,))
    try:
        try:
            unwrapped = _unwrap(raw)
            if unwrapped.startswith("{"):
                parsed = parse_json_graph(unwrapped)
                if parsed.graph is None:
                    return parsed
                mode = "repaired" if unwrapped != raw.strip() else parsed.parse_mode
                return _validated_graph(parsed.graph, mode)
            try:
                graph, named = _parse_context_actions(raw, repair=False)
                mode = "native"
            except GraphTextError:
                graph, named = _parse_context_actions(unwrapped, repair=True)
                mode = "repaired"
        except GraphTextError as exc:
            try:
                graph, named, mode = _parse_lenient(raw), True, "repaired"
            except GraphTextError:
                raise exc from None
        return _validated_graph(graph, mode, named=named)
    except OutputValidationError as exc:
        return GraphParseResult(None, error=str(exc), warning=tuple(exc.tags))
    except GraphTextError as exc:
        return GraphParseResult(None, error=f"graph text repair failed: {exc}", warning=(exc.tag,))


def _declare_referenced_names(graph):
    """Keep name-form roles that [Entities] omitted, as attribute-less entities."""
    names = {entity["id"] for entity in graph["entities"]}
    declared = False
    for action in graph["actions"]:
        for role in ("actor", "target", "tool", "place"):
            value = action[role]
            if value is None or value == "unknown" or value in names:
                continue
            token = value.casefold()
            if token in {"none", "unknown"}:
                action[role] = None if token == "none" else "unknown"
            else:
                graph["entities"].append({"id": value, "name": value, "attributes": []})
                names.add(value)
            declared = True
    return declared


def _validated_graph(graph, mode, *, named=False):
    declared = named and _declare_referenced_names(graph)
    graph, changed = normalize_graph_references(graph)
    changed = changed or declared
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
