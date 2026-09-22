"""Render summary inputs without interpreting placeholders inside observations."""

import re


def render_summary_prompt(template, scenes, *, english_title=None):
    title = (english_title or "").strip() or "(unavailable)"
    values = {"scenes": scenes, "english_title": title}
    rendered = re.sub(r"\{(scenes|english_title)\}",
                      lambda match: values[match.group(1)], template)
    return rendered


def validate_summary_template(template, uses_title):
    if "{scenes}" not in template:
        raise ValueError("Summary schema requires {scenes}")
    if ("{english_title}" in template) != uses_title:
        raise ValueError("Summary schema {english_title} must match the Arm title policy")
