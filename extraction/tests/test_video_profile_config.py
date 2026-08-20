from __future__ import annotations

import json

import pytest

from src.video_profile_generation.config import (
    ConfigError,
    load_video_profile_config,
)


VALID_CONFIG = {
    "shot_interval": "fixed_15s",
    "gemini_location": "global",
    "gemini_model": "gemini-3.5-flash",
    "gemini_thinking_level": "medium",
    "ontology_contract_path": "viewing_ontology_v3.json",
}


def test_load_video_profile_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "project-id")
    monkeypatch.setenv("OUTPUT_SAVE_PATH", "output")
    path = tmp_path / "video_profile_generation.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    config = load_video_profile_config(path)

    assert config.gcp_project_id == "project-id"
    assert config.shot_interval == "fixed_15s"
    assert config.resolve_local_input_dir(tmp_path) == tmp_path / "output/custom"
    assert config.resolve_local_output_dir(tmp_path) == tmp_path / "output/custom/video_profile/fixed_15s"


@pytest.mark.parametrize("thinking_level", ["minimal", "low", "medium", "high"])
def test_supported_thinking_levels(tmp_path, thinking_level: str, monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "project-id")
    monkeypatch.setenv("OUTPUT_SAVE_PATH", "output")
    path = tmp_path / "video_profile_generation.json"
    path.write_text(json.dumps({**VALID_CONFIG, "gemini_thinking_level": thinking_level}), encoding="utf-8")

    assert load_video_profile_config(path).gemini_thinking_level == thinking_level


@pytest.mark.parametrize(
    "updates",
    [
        {"gemini_thinking_level": "auto"},
        {"shot_interval": "fixed_10s"},
        {"gcs_bucket_name": "legacy-bucket"},
        {"gcp_project_id": "legacy-project"},
        {"local_output_dir": "legacy-output"},
        {"unexpected": True},
    ],
)
def test_invalid_video_profile_config(tmp_path, updates: dict[str, object], monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "project-id")
    monkeypatch.setenv("OUTPUT_SAVE_PATH", "output")
    path = tmp_path / "video_profile_generation.json"
    path.write_text(json.dumps({**VALID_CONFIG, **updates}), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_video_profile_config(path)


def test_missing_video_profile_config(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_video_profile_config(tmp_path / "missing.json")


def test_video_profile_config_requires_cloud_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("OUTPUT_SAVE_PATH", raising=False)
    path = tmp_path / "video_profile_generation.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    with pytest.raises(ConfigError, match="GCP_PROJECT_ID"):
        load_video_profile_config(path)


def test_video_profile_config_requires_output_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "project-id")
    monkeypatch.delenv("OUTPUT_SAVE_PATH", raising=False)
    path = tmp_path / "video_profile_generation.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    with pytest.raises(ConfigError, match="OUTPUT_SAVE_PATH"):
        load_video_profile_config(path)
