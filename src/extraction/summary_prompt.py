"""Render summary inputs without interpreting placeholders inside observations."""

import re


def render_summary_prompt(template, scenes, *, english_title=None):
    title = (english_title or "").strip() or "(unavailable)"
    values = {"scenes": scenes, "english_title": title}
    rendered = re.sub(r"\{(scenes|english_title)\}",
                      lambda match: values[match.group(1)], template)
    # Older/custom templates and raw fallbacks also receive the title when supplied.
    if english_title is not None and "{english_title}" not in template:
        return f"English Title: {title}\n\n{rendered}"
    return rendered
