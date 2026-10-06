"""Reuse and forced regeneration of representative v7 artifacts."""
from pathlib import Path
import json
import shutil

import pytest

from artifact_io import read_json, write_jsonl

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/v7"


@pytest.mark.parametrize("source", ["graph_qwen", "graph_gemini", "desc_qwen", "desc_gemini"])
def test_v7_success_reuse_and_force(current_context, monkeypatch, fake_models, source):
    from extraction import steps

    context = current_context
    fixture = FIXTURES / f"summaries_{source}_0.json"
    doc = read_json(fixture)
    cid = doc["content_id"]
    destination = context.summary_arm_dir(source) / f"{cid}.json"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(fixture, destination)
    scenes = json.loads((FIXTURES / f"scenes_{source}_0.json").read_text())
    write_jsonl(context.scene_arm_dir(source) / f"{cid}.jsonl", scenes)
    monkeypatch.setattr(type(context), "require_ready_cohort", lambda _: {"catalog": [{"content_id": cid}]})
    schema = "prompts/summary_graph_v7.md" if source.startswith("graph") else "prompts/summary_description_v5.md"
    before = destination.read_bytes()
    steps.summarize(context, arm=source, model="gemini", schema=schema)
    assert fake_models == []
    assert destination.read_bytes() == before
    # The copied Description provenance may name another run; it must remain intact.
    assert read_json(destination)["provenance"] == doc["provenance"]
    steps.summarize(context, arm=source, model="gemini", schema=schema, force=True)
    assert len(fake_models) == 1
    assert read_json(destination)["status"] == "complete"
