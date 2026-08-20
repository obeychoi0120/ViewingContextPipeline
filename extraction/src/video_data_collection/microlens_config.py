from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class MicroLensConfigError(ValueError):
    """Raised when the MicroLens importer config is invalid."""


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(StrictConfig):
    videos_dir: str = Field(min_length=1)
    titles_csv: str = Field(min_length=1)
    tags_csv: str = Field(min_length=1)
    pairs_tsv: str = Field(min_length=1)
    selection_jsonl: str | None = None
    smoke_selection_jsonl: str | None = None


class PilotConfig(StrictConfig):
    size: int = Field(default=1000, gt=0)
    smoke_size: int = Field(default=32, gt=0)
    seed: int = 42
    minimum_per_category: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> "PilotConfig":
        if self.smoke_size > self.size:
            raise ValueError("smoke_size must not exceed pilot size")
        return self


class AspectRatioConfig(StrictConfig):
    minimum: float = Field(default=4 / 3, gt=0)
    maximum: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "AspectRatioConfig":
        if self.minimum > self.maximum:
            raise ValueError("aspect_ratio.minimum must not exceed maximum")
        return self


class SamplingConfig(StrictConfig):
    shot_interval: Literal["fixed_15s", "fixed_30s"] = "fixed_30s"
    interval_seconds: Literal[10, 15] = 10
    frames_per_scene: Literal[1, 3] = 3
    ocr_fps: Literal[1] = 1
    resize_mode: Literal["contain_pad"] = "contain_pad"
    padding_color: Literal["black"] = "black"

    @model_validator(mode="after")
    def validate_sampling_contract(self) -> "SamplingConfig":
        expected = {
            "fixed_15s": (15, 1),
            "fixed_30s": (10, 3),
        }[self.shot_interval]
        if (self.interval_seconds, self.frames_per_scene) != expected:
            raise ValueError(
                f"{self.shot_interval} requires interval_seconds={expected[0]} "
                f"and frames_per_scene={expected[1]}"
            )
        return self


class MetadataDefaults(StrictConfig):
    channel: str = Field(default="unknown", min_length=1)
    upload_date: str = Field(default="unknown", min_length=1)
    description: str = ""


class OutputConfig(StrictConfig):
    inventory_jsonl: str = "manifests/microlens_100k_inventory.jsonl"
    failures_jsonl: str = "manifests/microlens_100k_import_failures.jsonl"
    selection_json: str = "manifests/microlens_100k_selection.json"
    pilot_manifest_csv: str = "manifests/microlens_100k_pilot_1k.csv"
    smoke_manifest_csv: str = "manifests/microlens_100k_smoke.csv"
    pilot_categories_jsonl: str = "manifests/microlens_100k_pilot_1k_categories.jsonl"
    smoke_categories_jsonl: str = "manifests/microlens_100k_smoke_categories.jsonl"


class MicroLensConfig(StrictConfig):
    schema_version: Literal["microlens-import-config/v1"]
    dataset_id: Literal["microlens-100k"]
    source: SourceConfig
    output_root: str = Field(min_length=1)
    assets_root: str = Field(default="source_assets", min_length=1)
    processing_config_path: str = Field(min_length=1)
    pilot: PilotConfig = Field(default_factory=PilotConfig)
    aspect_ratio: AspectRatioConfig = Field(default_factory=AspectRatioConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    metadata_defaults: MetadataDefaults = Field(default_factory=MetadataDefaults)
    outputs: OutputConfig = Field(default_factory=OutputConfig)

    def resolve_project_path(self, project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    def resolve_output_root(self, project_root: Path) -> Path:
        return self.resolve_project_path(project_root, self.output_root)

    def resolve_output_path(self, project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.resolve_output_root(project_root) / path


def load_microlens_config(path: str | Path) -> MicroLensConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return MicroLensConfig.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise MicroLensConfigError(
            f"invalid MicroLens config {config_path}: {exc}"
        ) from exc
