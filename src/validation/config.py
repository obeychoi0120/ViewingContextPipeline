from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    pairs_tsv: Path
    videos_dir: Path
    titles_csv: Path


class RollingDatasetConfig(DatasetConfig):
    pairs_csv: Path


class EncoderConfig(StrictModel):
    model_path: Path
    embedding_dim: Literal[1024]
    max_length: Literal[512]
    batch_size: int = Field(default=32, gt=0)


class ModelConfig(StrictModel):
    max_sequence_length: Literal[10]
    embedding_dim: Literal[512]
    num_blocks: Literal[2]
    num_heads: Literal[2]
    dropout: float = Field(ge=0, lt=1)
    batch_size: Literal[256]
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    popularity_power: Literal[1.0]
    seeds: list[int]

    @model_validator(mode="after")
    def validate_model(self) -> "ModelConfig":
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("exactly three distinct model seeds are required")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        return self


class EvaluationConfig(StrictModel):
    cutoffs: list[int]
    primary_cutoff: Literal[10]
    bootstrap_samples: int = Field(gt=0)
    familywise_alpha: float = Field(gt=0, lt=1)
    multiple_comparison_correction: Literal["bonferroni"]
    min_scene_coverage: float = Field(ge=0, le=1)
    max_arm_coverage_gap: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> "EvaluationConfig":
        if self.cutoffs not in ([4, 8, 10, 20], [4, 8, 10, 20, 30]):
            raise ValueError("cutoffs must be [4, 8, 10, 20] with optional @30")
        return self


class FullCohortConfig(StrictModel):
    mode: Literal["full_rolling"]
    metadata_missing_policy: Literal["zero_vector"]
    user_count: int = Field(gt=0)
    interaction_count: int = Field(gt=0)
    item_count: int = Field(gt=0)
    timezone: Literal["UTC"]
    evaluation_days: Literal[7]
    exclude_final_day: Literal[True]


class ValidationConfig(StrictModel):
    schema_version: Literal["validation-config/v5"]
    run_id: str
    dataset: RollingDatasetConfig
    cohort: FullCohortConfig
    encoder: EncoderConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output_dir: Path



def build_validation_config(
    *, run_id, dataset, settings, model_path, output_dir
) -> ValidationConfig:
    """Assemble the shared contract after the caller resolves its own paths."""
    return ValidationConfig.model_validate(
        {
            "schema_version": "validation-config/v5",
            "run_id": run_id,
            "dataset": dataset,
            "cohort": settings.get("cohort"),
            "encoder": {**settings["encoder"], "model_path": model_path},
            "model": settings.get("model"),
            "evaluation": settings.get("evaluation"),
            "output_dir": output_dir,
        }
    )
