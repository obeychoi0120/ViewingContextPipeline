from __future__ import annotations

from pathlib import Path
import re

import yaml

from validation.recommendation_contracts import RECOMMENDATION_ARMS


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def _relative_links(text: str) -> list[str]:
    matches = re.findall(r"\[[^\]]+\]\((?:<([^>]+)>|([^)]+))\)", text)
    return [first or second for first, second in matches]


def test_readme_relative_links_exist() -> None:
    text = README.read_text(encoding="utf-8")

    for target in _relative_links(text):
        if "://" in target or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        assert (ROOT / path).exists(), target


def test_readme_documents_current_dag_cli_and_schema_contracts() -> None:
    text = README.read_text(encoding="utf-8")
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))

    assert text.index("```mermaid") < text.index("2026-08-27 pipeline PNG snapshot")
    assert config["schema_version"] == "viewing-context-config/v2" in text
    assert config["extraction"]["graph"]["scene_prompt"] in text
    assert config["extraction"]["summary_repetition_penalty"] == 1.05
    assert "summary_repetition_penalty=1.05" in text
    assert "summary_repetition_penalty=1.00" not in text
    for arm in RECOMMENDATION_ARMS:
        assert arm in text
    for schema in (
        "validation-config/v2",
        "microlens-cohort-eligibility/v2",
        "metadata-title/v1",
        "sasrec-training-run/v2",
        "diagnosis/v3",
    ):
        assert schema in text

    stage_commands = re.findall(
        r"^python -m (?:extraction|validation) [^\r\n]+--run-id [^\r\n]+$",
        text,
        flags=re.MULTILINE,
    )
    assert len(stage_commands) == 11
    assert sum("extract-graph-scenes" in command for command in stage_commands) == 2
    assert sum("summarize-graph" in command for command in stage_commands) == 2
