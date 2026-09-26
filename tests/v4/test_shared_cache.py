"""Shared cache publication is atomic and rejects damaged bundles."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import shutil

import pytest

from arm_registry import registry
from validation.shared_cache import SharedCache
from validation.cache_identity import shareable_document
from pipeline_runtime import read_json, write_json
from validation.steps import embed_representations
from validation.representation_provenance import read_state, state_path


@pytest.mark.parametrize("scene_hash", [None, "", "unknown", "z" * 64, 123, []])
def test_unidentified_scene_inputs_remain_local(scene_hash):
    prov = {"prompt_hash": "summary", "model": "qwen", "settings": {"tokens": 100},
            "scene_provenance": [{}], "scene_input_hash": scene_hash}
    assert not shareable_document({"source_provenance": prov})
    prov["scene_input_hash"] = "a" * 64
    assert shareable_document({"source_provenance": prov})
    for field in ("prompt_hash", "model", "settings"):
        incomplete = deepcopy(prov)
        incomplete.pop(field)
        assert not shareable_document({"source_provenance": incomplete})
    prov.pop("scene_input_hash")
    prov["scene_provenance"] = [{"prompt_hash": "scene", "model": "qwen",
                                 "settings": {"tokens": 100}}]
    assert shareable_document({"source_provenance": prov})


@pytest.mark.parametrize("changed", ["scene_input_hash", "prompt_hash", "settings", "text"])
def test_description_changes_invalidate_only_description_arms(
    ready_context, fake_models, generate_all, changed
):
    generate_all(ready_context)
    embed_representations(ready_context, target=list(registry(ready_context.config)))
    other = replace(ready_context, run_id="changed", run_root=ready_context.run_root.parent / "changed")
    shutil.copytree(ready_context.run_root / "extraction", other.run_root / "extraction")
    path = next(other.summary_arm_dir("desc_qwen").glob("*.json"))
    doc = read_json(path)
    if changed == "text":
        doc["text"] += " A new detail."
        doc["word_count"] = len(doc["text"].split())
    elif changed == "settings":
        doc["provenance"][changed]["max_new_tokens"] += 1
    else:
        doc["provenance"][changed] = "b" * 64
    write_json(path, doc)
    result = embed_representations(other, target=list(registry(other.config)))
    assert result["generated_arms"] == ["desc_qwen_meta"]
    assert result["reuse"]["shared"] == ["meta", "graph_qwen", "graph_qwen_meta", "graph_gemini_meta", "desc_gemini_meta"]


def test_existing_local_embeddings_can_be_published_without_encoding(
    ready_context, fake_models, generate_all, monkeypatch
):
    generate_all(ready_context)
    targets = ["desc_qwen_meta"]
    embed_representations(ready_context, target=targets)
    # Simulate artifacts produced under the former local-only eligibility rule.
    shutil.rmtree(ready_context.run_root.parent.parent / "shared_cache" / "embeddings")
    original = {}
    for arm in targets:
        state = read_state(ready_context, arm)
        original[arm] = state["recommendation_hash"]
        state["shareable"] = False
        write_json(state_path(ready_context, arm), state)
    monkeypatch.setattr("validation.features.BGETextEncoder",
                        lambda *args: pytest.fail("local/shared reuse must not load BGE"))
    result = embed_representations(ready_context, target=targets)
    assert result["reuse"]["local"] == targets
    for arm in targets:
        state = read_state(ready_context, arm)
        assert state["shareable"]
        assert state["recommendation_hash"] == original[arm]
    other = replace(ready_context, run_id="reuse", run_root=ready_context.run_root.parent / "reuse")
    shutil.copytree(ready_context.run_root / "extraction", other.run_root / "extraction")
    assert embed_representations(other, target=targets)["reuse"]["shared"] == targets
    with pytest.raises(pytest.fail.Exception, match="must not load BGE"):
        embed_representations(other, target=targets, force=True)


def test_atomic_concurrent_publication_and_corruption(current_context, tmp_path):
    cache = SharedCache(current_context, "embeddings", "test-key")
    sources = []
    for i in range(6):
        source = tmp_path / str(i)
        source.mkdir()
        (source / "a").write_text(str(i))
        (source / "b").write_text(str(i))
        sources.append(source)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(
            executor.map(lambda p: cache.publish(p, ["a", "b"], origin={"run_id": p.name}), sources)
        )
    restored = tmp_path / "restored"
    assert cache.restore(restored)
    assert (restored / "a").read_bytes() == (restored / "b").read_bytes()
    (cache.path / "a").write_text("corrupt")
    assert cache.restore(restored) is None
    cache.publish(sources[0], ["a", "b"], origin={"run_id": "0"})
    assert cache.valid()
    (cache.path / "cache.json").unlink()
    assert cache.restore(restored) is None
