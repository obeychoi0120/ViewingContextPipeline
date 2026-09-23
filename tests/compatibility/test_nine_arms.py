"""Representative historical migration: preserve originals, reject conflicts, back up normalization."""

from copy import deepcopy
from dataclasses import replace
import shutil

import pytest

from pipeline_runtime import read_json, write_json
from extraction import steps
from extraction.arm_migration import migrate_arm_layout
from validation.steps import embed_representations
from validation.representation_provenance import read_state

NAMES = [
    "meta",
    "graph_qwen",
    "graph_gemini",
    "desc_qwen",
    "desc_gemini",
    "graph_meta_qwen",
    "graph_meta_gemini",
    "desc_meta_qwen",
    "desc_meta_gemini",
]


@pytest.fixture
def nine(legacy_ready_context, fake_models):
    ctx = legacy_ready_context
    ctx.config["schema_version"] = "viewing-context-config/v6"
    ctx.config["protocol"]["arms"] = list(NAMES)
    return ctx


def legacy_generation(ctx):
    settings = deepcopy(ctx.config)
    settings["schema_version"] = "viewing-context-config/v5"
    settings["protocol"]["arms"] = [
        "metadata",
        "graph_qwen",
        "graph_gemini",
        "desc_qwen",
        "desc_gemini",
    ]
    old = replace(ctx, config=settings)
    for representation in ("graph", "description"):
        for model in ("qwen", "gemini"):
            getattr(steps, f"extract_{representation}_scenes")(
                old,
                model=model,
                schema=f"prompts/scene_{representation}_v{3 if representation == 'graph' else 2}.md",
            )
            getattr(steps, f"summarize_{representation}")(
                old,
                source=model,
                model="qwen",
                schema=f"prompts/summary_{representation}_v4_meta.md",
            )
    return old


def test_explicit_migration_preserves_originals_and_normalizes_raw(nine):
    old = legacy_generation(nine)
    raw = next(old.graph_summary_dir("qwen").glob("*.json"))
    doc = read_json(raw)
    doc.update(status="raw_fallback", violations=["max_tokens"])
    write_json(raw, doc)
    before = {p: p.read_bytes() for p in (nine.run_root / "extraction").rglob("*") if p.is_file()}
    result = migrate_arm_layout(nine, summary_model="qwen")
    assert result["file_count"] == 33 and not result["skipped"]
    assert before == {p: p.read_bytes() for p in before}
    migrated = read_json(nine.summary_arm_dir("graph_meta_qwen") / raw.name)
    assert migrated["text"] == "" and migrated["status"] == "failed"
    assert migrated["provenance"] == doc["provenance"]
    assert not nine.summary_arm_dir("graph_qwen").exists()
    assert migrate_arm_layout(nine, summary_model="qwen") == result
    embed_representations(nine)
    assert read_state(nine, "graph_qwen")["zero_vector_count"] == 4
    with pytest.raises(ValueError, match="different Summary model"):
        migrate_arm_layout(nine, summary_model="gemini")
    target = nine.summary_arm_dir("graph_meta_qwen") / raw.name
    changed = read_json(target)
    changed["violations"] = ["changed"]
    write_json(target, changed)
    with pytest.raises(ValueError, match="conflict"):
        migrate_arm_layout(nine, summary_model="qwen")


def test_migration_rejects_conflicting_destination_failure(nine):
    from pipeline_runtime import write_jsonl

    legacy_generation(nine)
    write_jsonl(
        nine.summary_arm_dir("graph_meta_qwen") / "failures.jsonl",
        [
            {
                "content_id": "existing",
                "summary_model": "qwen",
                "error": "different result",
                "raw_output": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="conflict"):
        migrate_arm_layout(nine, summary_model="qwen")
    assert not list((nine.run_root / "extraction/scenes").glob("*/*.jsonl"))


def test_normalization_command_backs_up_and_is_idempotent(nine, monkeypatch):
    import tarfile
    from extraction.normalize_summary_arms import main

    old = legacy_generation(nine)
    dest = nine.summary_arm_dir("graph_meta_qwen")
    shutil.copytree(old.graph_summary_dir("qwen"), dest)
    before = {p.name: p.read_bytes() for p in dest.glob("*.json")}
    monkeypatch.setattr("extraction.normalize_summary_arms.RunContext.load", lambda _: nine)
    assert main(["--run-id", nine.run_id]) == 0
    backup = next(
        (nine.run_root.parents[1] / "backups" / nine.run_id).glob("summary-arm-normalization-*")
    )
    assert read_json(backup / "manifest.json")["state"] == "complete"
    with tarfile.open(backup / "originals.tar.gz") as archive:
        for name, data in before.items():
            assert (
                archive.extractfile(f"extraction/summaries/graph_meta_qwen/{name}").read() == data
            )
    after = {p.name: p.read_bytes() for p in dest.glob("*.json")}
    assert all(read_json(dest / name)["arm"] == "graph_meta_qwen" for name in after)
    assert main(["--run-id", nine.run_id]) == 0
    assert {p.name: p.read_bytes() for p in dest.glob("*.json")} == after
