"""
graph_v4/prompt.py - Fixed on-device VLM scene extraction prompt.

The VLM extracts visible atoms only. It must not output preference scores,
style archetype scores, or content-axis scores.
"""

from string import Template

from .ontology import (
    AFFECT_CUES,
    ENTITY_CATEGORIES,
    ENTITY_ROLES,
    FACE_PROMINENCE,
    MOOD_BINS,
    PEOPLE_DENSITY,
    RELATION_DEFINITIONS,
    RELATION_TYPES,
    SCENE_FUNCTIONS,
    SCENE_TYPES,
    SETTINGS,
    VISUAL_STYLE_CUES,
)


def _format_values(values: frozenset[str]) -> str:
    ordered = sorted(values, key=lambda value: (value == "unknown", value))
    return " | ".join(ordered)


def _allowed_labels_text() -> str:
    sections: list[str] = [
        f"scene_type:\n{_format_values(SCENE_TYPES)}",
    ]
    sections.extend(
        f"visual_style_cues.{cue_name}:\n{_format_values(allowed)}"
        for cue_name, allowed in VISUAL_STYLE_CUES.items()
    )
    sections.extend(
        [
            f"people_density:\n{_format_values(PEOPLE_DENSITY)}",
            f"face_prominence:\n{_format_values(FACE_PROMINENCE)}",
            f"mood_bin:\n{_format_values(MOOD_BINS)}",
            f"affect_cues:\n{_format_values(AFFECT_CUES)}",
            f"scene_function:\n{_format_values(SCENE_FUNCTIONS)}",
            f"setting:\n{_format_values(SETTINGS)}",
            f"entity.category:\n{_format_values(ENTITY_CATEGORIES)}",
            f"entity.role:\n{_format_values(ENTITY_ROLES)}",
        ]
    )
    return "\n\n".join(sections)


def _relation_slots_text() -> str:
    return "\n".join(
        f"{slot:<14}- {definition['description']}"
        for slot, definition in RELATION_DEFINITIONS.items()
        if slot in RELATION_TYPES
    )


_SCENE_EXTRACTION_PROMPT_TEMPLATE = Template("""You are a visual scene-to-graph extractor.

Analyze only the visible content and visible visual form of the image.
Choose only from the allowed labels for enum fields.
Return valid JSON only.

Allowed labels:

$allowed_labels

Entity relation slots (per-entity attributes):
$relation_slots

Rules:
- Choose only from the allowed labels. Use unknown when unclear if the field allows unknown.
- Use none for people_density when no people are visible.
- Use none for face_prominence when no face is visible.
- mood_bin and affect_cues must be based only on visible tone, expression, action, and atmosphere.
- scene_function describes the scene's visible communicative role.
- fantasy_element means visibly impossible, supernatural, magical, surreal, or speculative elements.
- Keep entity names generic and canonical, e.g. coffee, dog, phone, table.
- Use coarse entity categories from the allowed list; put finer details in entity names or IS_A.
- For digital overlays, UI, HUDs, charts, captions, subtitles, or info panels, use entity.category=text.
- For person entities, fill DOING when visible. Fill IS_A only if the role is clear.
- INTERACTS_WITH must be a target entity local_id, e.g. "e2"; put the action itself in DOING.
- A small background person should not make scene_type=people_social.

IMPORTANT entity rules:
- List 2 to 5 visible entities.
- Always include the main subject and at least one other visible element.
- Include setting elements when visible.

Return exactly this JSON shape:

{
  "scene_type": "...",
  "visual_style_cues": {
    "media_form": "...",
    "fantasy_element": "...",
    "shot_scale": "...",
    "graphic_density": "...",
    "composition_density": "..."
  },
  "people_density": "...",
  "face_prominence": "...",
  "mood_bin": "...",
  "affect_cues": ["..."],
  "scene_function": "...",
  "setting": "...",
  "entities": [
    {
      "local_id": "e1",
      "category": "...",
      "name": "generic canonical name",
      "role": "main_subject | object | setting_element | background",
      "relations": {
        "DOING": "...",
        "WEARING": "...",
        "IS_A": "...",
        "AT": "...",
        "INTERACTS_WITH": "e2"
      }
    },
    {
      "local_id": "e2",
      "category": "...",
      "name": "generic canonical name",
      "role": "main_subject | object | setting_element | background",
      "relations": {
        "INTERACTS_WITH": "e1"
      }
    }
  ]
}""")

SCENE_EXTRACTION_PROMPT = _SCENE_EXTRACTION_PROMPT_TEMPLATE.substitute(
    allowed_labels=_allowed_labels_text(),
    relation_slots=_relation_slots_text(),
)

USER_MESSAGE = "Extract visible content and visual evidence from this scene image."
