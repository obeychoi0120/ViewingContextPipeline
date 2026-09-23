import pytest

from pipeline_runtime import ConfigError, RunContext


@pytest.mark.parametrize("name", ["..", "a/b", "resized_keyframes", "source_assets", "a\\b", ""])
def test_run_id_cannot_escape_or_claim_the_shared_frame_directory(current_context, name):
    with pytest.raises(ConfigError):
        RunContext.load(name, root=current_context.root)


def test_missing_metadata_row_keeps_catalog_scope_and_records_failure(current_context):
    from preparation.steps import prepare_cohort_step
    from pipeline_runtime import read_json, read_jsonl

    titles = current_context.path("data", "titles_csv")
    titles.write_text("1,First\n2, \n3,Third\n")
    with pytest.raises(RuntimeError, match="unresolved assets"):
        prepare_cohort_step(current_context)
    assert len(read_jsonl(current_context.cohort_dir / "required_items.jsonl")) == 4
    assert read_json(current_context.cohort_dir / "eligibility.json")["status"] == "blocked"
    assert read_jsonl(current_context.cohort_dir / "preparation_failures.jsonl") == [
        {"item_id": "4", "reason": "missing_title"}
    ]
