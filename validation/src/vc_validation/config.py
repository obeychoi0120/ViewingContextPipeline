from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    pairs_tsv: Path
    videos_dir: Path
    vp_graph_dir: Path
    vp_desc_dir: Path


class CohortConfig(StrictModel):
    user_count: int = Field(gt=0)
    smoke_user_count: int = Field(gt=0)
    seed: int
    min_sequence_length: int = Field(ge=5)
    max_sequence_length: int = Field(ge=5)
    history_strata: list[int]

    @model_validator(mode="after")
    def validate_cohort(self) -> "CohortConfig":
        if self.smoke_user_count > self.user_count:
            raise ValueError("smoke_user_count must not exceed user_count")
        if not self.history_strata or self.history_strata != sorted(set(self.history_strata)) or self.history_strata[0] != 5:
            raise ValueError("history_strata must be sorted, unique, and start at 5")
        return self


class EncoderConfig(StrictModel):
    model_id: Literal["BAAI/bge-large-en-v1.5"]
    model_path: Path
    embedding_dim: Literal[1024]
    max_length: Literal[512]
    normalize_embeddings: Literal[True]
    batch_size: int = Field(default=32, gt=0)


class ModelConfig(StrictModel):
    max_sequence_length: Literal[10]
    embedding_dim: int = Field(gt=0)
    num_blocks: Literal[2]
    num_heads: Literal[2]
    dropout: float = Field(ge=0, lt=1)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
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

    @model_validator(mode="after")
    def validate_cutoffs(self) -> "EvaluationConfig":
        if self.cutoffs != [4, 8, 10, 20]:
            raise ValueError("cutoffs must be exactly [4, 8, 10, 20]")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal["viewing-context-experiment/v2"]
    run_id: str
    dataset: DatasetConfig
    cohort: CohortConfig
    encoder: EncoderConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(raw)
    base = config_path.parent
    data = config.model_dump()
    for key in ("pairs_tsv", "videos_dir", "vp_graph_dir", "vp_desc_dir"):
        candidate = Path(data["dataset"][key])
        data["dataset"][key] = candidate if candidate.is_absolute() else (base / candidate).resolve()
    encoder_raw = str(raw["encoder"]["model_path"])
    encoder_path = Path(encoder_raw)
    data["encoder"]["model_path"] = encoder_path if encoder_raw.startswith("/") or encoder_path.is_absolute() else (base / encoder_path).resolve()
    output = Path(data["output_dir"])
    data["output_dir"] = output if output.is_absolute() else (base.parent / output).resolve()
    return ExperimentConfig.model_validate(data)
