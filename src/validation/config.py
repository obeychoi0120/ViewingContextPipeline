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


class CohortConfig(StrictModel):
    user_count: int = Field(gt=0)
    seed: int
    min_sequence_length: int = Field(ge=5)
    max_sequence_length: int = Field(ge=5)
    history_strata: list[int]

    @model_validator(mode="after")
    def validate_cohort(self) -> "CohortConfig":
        if (
            not self.history_strata
            or self.history_strata != sorted(set(self.history_strata))
            or self.history_strata[0] != 5
        ):
            raise ValueError("history_strata must be sorted, unique, and start at 5")
        return self


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
    non_inferiority_margin: Literal[0.05]
    familywise_alpha: float = Field(gt=0, lt=1)
    multiple_comparison_correction: Literal["bonferroni"]
    min_scene_coverage: float = Field(ge=0, le=1)
    max_arm_coverage_gap: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> "EvaluationConfig":
        if self.cutoffs != [4, 8, 10, 20]:
            raise ValueError("cutoffs must be exactly [4, 8, 10, 20]")
        return self


class ValidationConfig(StrictModel):
    schema_version: Literal["validation-config/v2"]
    run_id: str
    dataset: DatasetConfig
    cohort: CohortConfig
    encoder: EncoderConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output_dir: Path
