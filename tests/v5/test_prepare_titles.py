import numpy as np
import pytest

from pipeline_runtime import read_json, read_jsonl
from preparation.cli import main
from validation.steps import embed_representations, validation_config


def configure_titles(context):
    primary = context.path("data", "titles_csv")
    primary.write_text("1,Original title\n2, \n")
    supplement = context.root / "official_titles.csv"
    supplement.write_text('item,title\n1,Do not replace\n2,"Filled, title"\n3,Third title\n')
    context.config["data"]["titles_supplement_csv"] = str(supplement)
    return primary, supplement


def test_single_prepare_command_completes_titles_without_extra_artifacts(
    v5_context, monkeypatch, fake_models
):
    context = v5_context
    primary, supplement = configure_titles(context)
    originals = {p: p.read_bytes() for p in (primary, supplement)}
    monkeypatch.setattr("preparation.cli.RunContext.load", lambda _: context)
    assert not (context.cohort_dir / "required_items.jsonl").exists()
    assert main(["prepare-cohort", "--run-id", context.run_id]) == 0
    cohort = context.require_ready_cohort()
    assert [r["title"] for r in cohort["metadata_titles"]] == [
        "Original title",
        "Filled, title",
        "Third title",
        "",
    ]
    completion = cohort["plan"]["title_completion"]
    assert completion["supplemented_item_ids"] == ["2", "3"]
    assert completion["unresolved_item_ids"] == ["4"]
    assert validation_config(context).dataset.titles_supplement_csv == supplement
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert not list(context.root.rglob("*.report.json"))
    assert not list(context.root.rglob("*completed.csv"))
    assert {p.name for p in context.run_root.iterdir()} == {"cohort", "extraction", "validation"}
    embed_representations(context, target=["metadata"])
    with np.load(context.representations_dir / "metadata_embeddings.npz") as arrays:
        assert np.all(arrays["values"][3] == 0)
        assert np.any(arrays["values"][1] != 0)
    supplement.write_text("item,title\n2,Updated title\n3,Third title\n4,Fourth title\n")
    assert main(["prepare-cohort", "--run-id", context.run_id]) == 0
    titles = read_jsonl(context.cohort_dir / "metadata_titles.jsonl")
    assert titles[1]["title"] == "Updated title" and titles[3]["title"] == "Fourth title"
    assert embed_representations(context, target=["metadata"])["generated_arms"] == ["metadata"]


@pytest.mark.parametrize("broken", ["missing", "duplicate"])
def test_bad_supplement_blocks_preparation_instead_of_silent_zero_vectors(
    v5_context, monkeypatch, broken
):
    _, supplement = configure_titles(v5_context)
    if broken == "missing":
        supplement.unlink()
    else:
        supplement.write_text("item,title\n2,One\n2,Two\n")
    monkeypatch.setattr("preparation.cli.RunContext.load", lambda _: v5_context)
    assert main(["prepare-cohort", "--run-id", v5_context.run_id]) == 1
    assert read_json(v5_context.cohort_dir / "eligibility.json")["status"] == "blocked"
    assert not (v5_context.cohort_dir / "metadata_titles.jsonl").exists()
