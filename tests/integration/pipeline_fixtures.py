from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import validation.steps as validation_steps
from validation.cohort import prepare_cohort
from extraction.summary_validation import SUMMARY_SECTIONS
from pipeline_runtime import RunContext


ROOT = Path(__file__).resolve().parents[2]


def _summary_lines(sections: dict[str, str]) -> str:
    return "\n".join(f"{name}: {sections[name]}" for name in SUMMARY_SECTIONS)


def _ready_cohort(context: RunContext) -> str:
    context.config["validation"]["cohort"]["user_count"] = 1
    context.path("data", "pairs_tsv").write_text("fixture\t1 1 1 1 1\n", encoding="utf-8")
    (context.path("data", "videos_dir") / "1.mp4").write_bytes(b"video")
    prepare_cohort(
        validation_steps.validation_config(context),
        output_dir=context.cohort_dir,
        probe=lambda _: 10.0,
    )
    return "microlens_100k_00001"


@pytest.fixture()
def context(tmp_path: Path) -> RunContext:
    data = tmp_path / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    (data / "pairs.tsv").write_text("fixture\n", encoding="utf-8")
    (data / "titles.csv").write_text("1,Fixture title\n", encoding="utf-8")
    models = tmp_path / "models"
    for name in ("qwen", "bge"):
        (models / name).mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    # These fixtures exercise the preserved v3 artifact contracts.
    config["schema_version"] = "viewing-context-config/v3"
    config["protocol"].update(cohort_sampling="user_first_nested_stratified",
                              catalog_scope="selected_user_sequence_union")
    config["validation"]["cohort"] = dict(user_count=1000, seed=42,
        min_sequence_length=5, max_sequence_length=13, history_strata=[5, 10, 20, 50])
    config["validation"]["evaluation"]["cutoffs"] = [4, 8, 10, 20]
    config["artifacts_root"] = str(tmp_path / "artifacts")
    config["data"] = {
        "videos_dir": str(videos),
        "pairs_tsv": str(data / "pairs.tsv"),
        "titles_csv": str(data / "titles.csv"),
    }
    config["models"] = {
        "qwen": str(models / "qwen"),
        "bge": str(models / "bge"),
        "gemini": {
            "project_id": "test-project",
            "location": "global",
            "model_id": "test-gemini",
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "thinking_level": "low",
            "media_resolution": "MEDIA_RESOLUTION_MEDIUM",
        },
    }
    for arm in ("graph", "description"):
        for key in ("scene_prompt", "summary_prompt"):
            config["extraction"][arm][key] = str(ROOT / config["extraction"][arm][key])
    (config_dir / "pipeline.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return RunContext.load("test_run", root=tmp_path)
