"""MiniCPM remains an extraction source, outside the eight recommendation arms."""

from dataclasses import replace

import pytest

from arm_registry import generation_registry, registry, select_arms
from extraction.scene_storage import write_scene_records
from helpers import ready_cohort, scene_row
from validation.graph_inputs import build_input, source_identity
from validation.selection import prepare_validation_cohort


def test_minicpm_is_extraction_source_only(context):
    assert "graph_minicpm" in generation_registry(context.config)
    assert "graph_minicpm" not in registry(context.config)
    assert "graph_minicpm_meta" not in registry(context.config)
    with pytest.raises(ValueError):
        select_arms(context.config, ["graph_minicpm"])


def test_extraction_source_identity_remains_distinct(context):
    ready_cohort(context)
    cohort = prepare_validation_cohort(context)
    cohort = {**cohort, "catalog": cohort["catalog"][:1],
              "metadata_titles": cohort["metadata_titles"][:1]}
    cid = cohort["catalog"][0]["content_id"]
    for source in ("graph_minicpm", "graph_qwen3vl"):
        write_scene_records(context.scene_arm_dir(source) / f"{cid}.jsonl", [scene_row(cid)])
    mini = replace(generation_registry(context.config)["graph_minicpm"], uses_title=True)
    qwen = registry(context.config)["g0"]
    _, arrays, stats = build_input(context, cohort, mini)
    assert arrays["titles"][0] != 0 and stats["counts"]["valid_scenes"] == 1
    assert source_identity(context, cohort, mini) != source_identity(context, cohort, qwen)
