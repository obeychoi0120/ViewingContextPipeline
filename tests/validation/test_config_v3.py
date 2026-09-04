from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from validation.config import ValidationConfig

from conftest import config_data


def test_validation_config_v3_accepts_only_the_declared_contract(tmp_path) -> None:
    value = config_data(tmp_path)

    config = ValidationConfig.model_validate(value)

    assert config.schema_version == "validation-config/v3"
    assert config.dataset.titles_csv == tmp_path / "titles.csv"
    assert config.model.embedding_dim == 512
    assert config.model.batch_size == 256
    assert config.model.popularity_power == 1.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version="validation-config/v1"),
        lambda value: value.update(schema_version="validation-config/v2"),
        lambda value: value["dataset"].pop("titles_csv"),
        lambda value: value["dataset"].update(legacy_titles="titles.csv"),
        lambda value: value["model"].update(embedding_dim=8),
        lambda value: value["model"].update(batch_size=2),
        lambda value: value["model"].pop("popularity_power"),
        lambda value: value["model"].update(legacy_id_embedding=True),
    ],
)
def test_validation_config_rejects_legacy_missing_and_extra_fields(
    tmp_path,
    mutation,
) -> None:
    value = deepcopy(config_data(tmp_path))
    mutation(value)

    with pytest.raises(ValidationError):
        ValidationConfig.model_validate(value)
