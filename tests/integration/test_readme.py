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
    assert config["schema_version"] == "viewing-context-config/v3" in text
    assert config["extraction"]["graph"]["scene_prompt"] in text
    assert config["extraction"]["summary_repetition_penalty"] == 1.05
    assert "summary_repetition_penalty=1.05" in text
    assert "summary_repetition_penalty=1.00" not in text
    for stage, default in (("graph", 1.05), ("description", 1.0)):
        assert config["extraction"][f"{stage}_repetition_penalty"] == default
        assert f"extraction.{stage}_repetition_penalty={default}" in text
    for arm in RECOMMENDATION_ARMS:
        assert arm in text
    for schema in (
        "validation-config/v3",
        "microlens-user-cohort-plan/v1",
        "microlens-cohort-eligibility/v3",
        "metadata-title/v1",
        "sasrec-training-run/v2",
        "diagnosis/v4",
    ):
        assert schema in text

    stage_commands = re.findall(
        r"^python -m (?:extraction|validation) [^\r\n]+--run-id [^\r\n]+$",
        text,
        flags=re.MULTILINE,
    )
    auxiliary = [
        command
        for command in stage_commands
        if "--plan-only" in command or "--reuse-run-id" in command
    ]
    assert len(auxiliary) == 2
    stage_commands = [command for command in stage_commands if command not in auxiliary]
    assert len(stage_commands) == 11
    assert sum("extract-graph-scenes" in command for command in stage_commands) == 2
    assert sum("summarize-graph" in command for command in stage_commands) == 2


def test_readme_explains_user_first_readiness_scope_and_reruns() -> None:
    text = README.read_text(encoding="utf-8")
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    opening = text.split("## 이 PoC가 답하려는 질문")[0]
    assert "user-1K" in opening and "사용자를 먼저" in opening
    assert "성능 개선을 보장하지 않으며" in opening
    assert "이번 구현에 포함되지 않습니다" in opening
    for key in ("cohort_sampling", "catalog_scope"):
        assert config["protocol"][key] in text
    assert config["validation"]["cohort"]["user_count"] == 1000
    assert "MicroLens-100k_pairs.tsv" in config["data"]["pairs_tsv"]
    assert config["data"]["titles_csv"].endswith("MicroLens-100k_title_en_completed.csv")
    assert "python -m validation.complete_titles" in text
    assert "MicroLens-50k_titles.csv" in text
    assert "metadata-title-completion/v1" in text
    assert "user_count=1000" in text and "user_count=59" not in text
    for artifact in ("cohort_plan.json", "selected_users.jsonl", "required_items.jsonl"):
        assert artifact in text
    assert "`planned`/`blocked` 상태에서는 downstream이 실행되지 않습니다" in text
    assert "절대 NDCG 차이를 사용자 증가만의 효과로 해석하지 않습니다" in text
    assert "PNG와 timestamp만" in text
    assert "cohort.statistics.selection" in text and "cohort.statistics.refit" in text
    assert text.count("<details>") == text.count("</details>")
    assert "```mermaid\nflowchart TB\n" in text
