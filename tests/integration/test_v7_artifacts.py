"""Frozen pre-refactor v7 data and identities, without touching live artifacts."""
from pathlib import Path
import json
import shutil

import pytest

from artifact_io import fingerprint, read_json, write_json, write_jsonl
from arm_registry import registry
from extraction.scene_storage import read_scene_records
from extraction.summary_storage import reuse_summary_document
from validation.cache_identity import generation_identity, semantic_document_hash
from validation.representation_inputs import documents_for_arm, representation_signature

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/v7"
MANIFEST = read_json(FIXTURES / "manifest.json")
CASES = [(group, sample) for group, inventory in MANIFEST.items() for sample in inventory["fixtures"]]


@pytest.mark.parametrize("group,sample", CASES, ids=[sample["file"] for _, sample in CASES])
def test_v7_payloads_and_semantic_hashes(tmp_path, group, sample):
    payload = json.loads((FIXTURES / sample["file"]).read_text())
    assert fingerprint(payload) == sample["fingerprint"]
    if group.startswith("scenes/"):
        path = tmp_path / f"{payload[0]['content_id']}.jsonl"
        write_jsonl(path, payload)
        before = path.read_bytes()
        records = read_scene_records(path)
        assert [r["scene_idx"] for r in records] == sorted(r["scene_idx"] for r in payload)
        assert path.read_bytes() == before
    else:
        path = FIXTURES / sample["file"]
        assert reuse_summary_document(path, content_id=payload["content_id"], arm=payload["arm"]) == payload
        assert generation_identity(payload["provenance"]) == sample["generation_identity"]
        assert semantic_document_hash({"content_id":payload["content_id"], "text":payload["text"],
                                       "source_provenance":payload["provenance"]}) == sample["document_hash"]


def test_all_six_representation_keys_match_pre_refactor(current_context):
    context = current_context
    context.config["models"]["bge"] = "/baseline/bge"
    cohort = {"catalog": [{"item_id": "1", "content_id": "microlens_100k_00001"}],
              "metadata_titles": [{"item_id": "1", "content_id": "microlens_100k_00001", "title": "Baseline title"}]}
    for source in ("graph_qwen", "graph_gemini", "desc_qwen", "desc_gemini"):
        doc = read_json(FIXTURES / f"summaries_{source}_0.json")
        doc["content_id"] = "microlens_100k_00001"
        write_json(context.summary_arm_dir(source) / "microlens_100k_00001.json", doc)
    expected = read_json(FIXTURES / "representation_hashes.json")
    actual = {name: representation_signature(context, cohort["catalog"], arm,
                                             documents_for_arm(context, cohort, arm))
              for name, arm in registry(context.config).items()}
    assert actual == expected


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


@pytest.mark.parametrize("source", ["desc_qwen", "desc_gemini"])
def test_v7_failed_summary_retries_without_rewriting_log_on_read(current_context, monkeypatch, fake_models, source):
    from extraction import steps
    from extraction.failures import FailureLog

    context = current_context
    doc = read_json(FIXTURES / f"summaries_{source}_1.json")
    cid = doc["content_id"]
    directory = context.summary_arm_dir(source)
    write_json(directory / f"{cid}.json", doc)
    failure = read_json(FIXTURES / "summary_failures.json")[source]
    log = directory / "failures.jsonl"
    write_jsonl(log, [failure])
    before = log.read_bytes()
    assert FailureLog(directory).contains(cid, None)
    assert log.read_bytes() == before
    scenes = json.loads((FIXTURES / f"scenes_{source}_0.json").read_text())
    write_jsonl(context.scene_arm_dir(source) / f"{cid}.jsonl",
                [{**row, "content_id": cid} for row in scenes])
    monkeypatch.setattr(type(context), "require_ready_cohort", lambda _: {"catalog": [{"content_id": cid}]})
    steps.summarize(context, arm=source, model="gemini", schema="prompts/summary_description_v5.md")
    assert len(fake_models) == 1
    assert read_json(directory / f"{cid}.json")["status"] == "complete"
    assert not log.exists()


def test_v7_scene_copy_resumes_after_interruption(current_context, monkeypatch):
    from extraction import steps
    from extraction.backends import GeminiGenerationOutcome

    context = current_context
    source = "graph_qwen"
    scenes = json.loads((FIXTURES / f"scenes_{source}_0.json").read_text())
    ids = [scenes[0]["content_id"], "microlens_100k_00002"]
    for cid in ids:
        write_jsonl(context.scene_arm_dir(source) / f"{cid}.jsonl",
                    [{**row, "content_id": cid} for row in scenes])
    monkeypatch.setattr(type(context), "require_ready_cohort", lambda _: {"catalog": [{"content_id": cid} for cid in ids]})
    calls = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                calls.append(task.task_id)
                callback(GeminiGenerationOutcome(task.task_id, "A brief summary."))
                if len(calls) == 1:
                    raise KeyboardInterrupt

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    options = dict(arm=source, model="gemini", schema="prompts/summary_graph_v7.md")
    with pytest.raises(KeyboardInterrupt):
        steps.summarize(context, **options)
    first = context.summary_arm_dir(source) / f"{ids[0]}.json"
    before = first.read_bytes()
    steps.summarize(context, **options)
    assert calls == ids
    assert first.read_bytes() == before
    assert all(read_json(context.summary_arm_dir(source) / f"{cid}.json")["status"] == "complete" for cid in ids)
