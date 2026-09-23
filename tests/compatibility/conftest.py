"""Historical configuration is restricted to representative compatibility tests."""

import pytest


@pytest.fixture
def legacy_ready_context(ready_context):
    context = ready_context
    context.config.pop("experiment_config_version")
    context.config["schema_version"] = "viewing-context-config/v5"
    context.config["protocol"]["description_extractors"] = ["qwen", "gemini"]
    context.config["protocol"]["arms"] = [
        "metadata",
        "graph_qwen",
        "graph_gemini",
        "desc_qwen",
        "desc_gemini",
    ]
    for kind in ("graph", "description"):
        template = (context.root / f"prompts/summary_{kind}_v4.md").read_text()
        (context.root / f"prompts/summary_{kind}_v4_meta.md").write_text(
            template.replace(
                "Scene observations:", "English Title: {english_title}\n\nScene observations:"
            )
        )
    return context
