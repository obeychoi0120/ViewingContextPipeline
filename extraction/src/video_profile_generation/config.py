from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.common.output_paths import custom_output_root


class ConfigError(ValueError):
    """Raised when the video profile configuration is invalid."""


class VideoProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gcp_project_id: str = Field(min_length=1)
    gemini_location: str = Field(min_length=1)
    gemini_model: str = Field(min_length=1)
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"]
    shot_interval: Literal["fixed_15s", "fixed_30s", "shot_wise"] = "fixed_30s"
    ontology_contract_path: str = Field(min_length=1)
    local_output_dir: str = Field(min_length=1)

    @field_validator(
        "gcp_project_id",
        "gemini_location",
        "gemini_model",
        "ontology_contract_path",
        "local_output_dir",
        mode="before",
    )
    @classmethod
    def strip_required_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    def resolve_local_output_dir(self, project_root: Path) -> Path:
        path = Path(self.local_output_dir)
        resolved = path if path.is_absolute() else project_root / path
        return resolved / self.shot_interval

    def resolve_local_input_dir(self, project_root: Path) -> Path:
        path = Path(self.local_output_dir)
        resolved = path if path.is_absolute() else project_root / path
        return resolved.parent

    def resolve_ontology_contract_path(self, project_root: Path) -> Path:
        path = Path(self.ontology_contract_path)
        return path if path.is_absolute() else project_root / path


def load_video_profile_config(path: str | Path) -> VideoProfileConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"video profile config not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in video profile config {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"failed to read video profile config {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"video profile config must be a JSON object: {config_path}")

    legacy_keys = sorted({"gcp_project_id", "local_output_dir"} & raw.keys())
    if legacy_keys:
        raise ConfigError(
            f"move {', '.join(legacy_keys)} from {config_path} to "
            "GCP_PROJECT_ID/OUTPUT_SAVE_PATH in config/.env"
        )

    load_dotenv(config_path.parent / ".env")
    gcp_project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    output_save_path = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not gcp_project_id:
        raise ConfigError("GCP_PROJECT_ID is required in config/.env or the environment")
    if not output_save_path:
        raise ConfigError("OUTPUT_SAVE_PATH is required in config/.env or the environment")
    raw = {
        **raw,
        "gcp_project_id": gcp_project_id,
        "local_output_dir": str(custom_output_root(output_save_path) / "video_profile"),
    }

    try:
        return VideoProfileConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid video profile config {config_path}: {exc}") from exc
