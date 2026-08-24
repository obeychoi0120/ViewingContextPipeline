from pathlib import Path

from viewing_context_pipeline.validation.config import load_config


def test_pilot_config_locks_user_count_and_protocol() -> None:
    path = Path(__file__).resolve().parents[2] / "config" / "validation" / "pilot_1k.yaml"
    config = load_config(path)
    assert config.cohort.user_count == 1000
    assert config.model.max_sequence_length == 10
    assert config.model.embedding_dim == 128
    assert config.evaluation.cutoffs == [4, 8, 10, 20]
    assert config.evaluation.primary_cutoff == 10
    assert config.encoder.model_path.as_posix() == "/home_nvme/shared/models/bge-large-en-v1.5"
    assert config.schema_version == "validation-config/v1"


def test_canonical_config_locks_100k_and_bge_contract() -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "config" / "validation" / "canonical_100k.yaml")
    assert config.cohort.user_count == 100000
    assert config.encoder.embedding_dim == 1024
    assert config.encoder.model_id == "BAAI/bge-large-en-v1.5"
