from __future__ import annotations


import numpy as np
import pytest

import validation.features as validation_features
import validation.steps as validation_steps
from pipeline_runtime import RunContext, read_json, write_json, write_jsonl


from pipeline_fixtures import context as context, _ready_cohort


def test_embedding_uses_fixed_files_and_no_manifest(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    content_id = _ready_cohort(context)
    for source in ("qwen", "gemini"):
        write_json(
            context.graph_summary_dir(source) / f"{content_id}.json",
            {
                "schema_version": "graph-video-summary/v3",
                "content_id": content_id,
                "status": "complete",
                "text": f"{source} graph",
            },
        )
    write_json(
        context.description_summary_dir / f"{content_id}.json",
        {
            "schema_version": "description-video-summary/v3",
            "content_id": content_id,
            "status": "complete",
            "text": "description",
        },
    )
    dimension = validation_steps.validation_config(context).encoder.embedding_dim
    loads: list[object] = []
    encoded_texts: list[list[str]] = []

    class FakeEncoder:
        def __init__(self, settings):
            loads.append(settings)

        def encode(self, texts):
            encoded_texts.append(list(texts))
            return np.ones((len(texts), dimension), dtype=np.float32)

    monkeypatch.setattr(validation_features, "BGETextEncoder", FakeEncoder)

    validation_steps.embed_representations(context)

    assert (context.representations_dir / "item_index.json").is_file()
    for branch in ("metadata", "graph_qwen", "graph_gemini", "desc"):
        assert (context.representations_dir / f"{branch}_embeddings.npz").is_file()
    assert len(loads) == 1
    assert len(encoded_texts) == 4
    assert encoded_texts[0] == ["Fixture title"]

    (context.representations_dir / "desc_embeddings.npz").unlink()
    validation_steps.embed_representations(context)

    assert len(loads) == 2
    assert len(encoded_texts) == 5
    assert encoded_texts[-1] == ["description"]

    stable = {
        branch: np.load(context.representations_dir / f"{branch}_embeddings.npz")["values"].copy()
        for branch in ("metadata", "graph_qwen", "graph_gemini", "desc")
    }

    class FailingEncoder:
        def __init__(self, _settings):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated second-branch failure")
            return np.full((len(texts), dimension), 2.0, dtype=np.float32)

    monkeypatch.setattr(validation_features, "BGETextEncoder", FailingEncoder)
    with pytest.raises(RuntimeError, match="second-branch failure"):
        validation_steps.embed_representations(context, force=True)
    for branch, expected in stable.items():
        actual = np.load(context.representations_dir / f"{branch}_embeddings.npz")["values"]
        assert np.array_equal(actual, expected)
    assert not (context.representations_dir / "manifest.json").exists()


def test_missing_summary_error_names_the_actual_path(context: RunContext) -> None:
    context.initialize()
    _ready_cohort(context)
    expected = context.graph_summary_dir("qwen")
    with pytest.raises(
        validation_steps.ValidationStepError, match="missing graph_qwen summary directory"
    ) as raised:
        validation_steps.embed_representations(context)
    assert str(expected) in str(raised.value)


@pytest.fixture()
def embedding_fallback_case(context, monkeypatch, capsys):
    catalog = [{"item_id": str(index), "content_id": f"c{index}"} for index in (1, 2)]
    monkeypatch.setattr(RunContext, "require_ready_cohort", lambda _: {"catalog": catalog})
    write_jsonl(
        context.cohort_dir / "metadata_titles.jsonl",
        [{**row, "title": f"title {row['item_id']}"} for row in catalog],
    )
    for row in catalog:
        content_id = row["content_id"]
        write_json(
            context.graph_summary_dir("qwen") / f"{content_id}.json",
            {"content_id": content_id, "text": f"qwen {content_id}"},
        )
        write_json(
            context.description_summary_dir / f"{content_id}.json",
            {"content_id": content_id, "text": f"description {content_id}"},
        )
    write_json(
        context.graph_summary_dir("gemini") / "c2.json",
        {"content_id": "c2", "text": "gemini c2"},
    )
    before_load = []
    encoded = []
    dimension = validation_steps.validation_config(context).encoder.embedding_dim

    class Encoder:
        def __init__(self, _settings):
            before_load.append(capsys.readouterr().out)

        def encode(self, texts):
            encoded.append(list(texts))
            return np.ones((len(texts), dimension), dtype=np.float32)

    monkeypatch.setattr(validation_features, "BGETextEncoder", Encoder)
    return encoded, before_load


def test_embedding_falls_back_only_for_missing_gemini_and_reports_before_load(
    context,
    embedding_fallback_case,
):
    encoded, before_load = embedding_fallback_case
    qwen_path = context.graph_summary_dir("qwen") / "c1.json"
    before = qwen_path.read_bytes()
    validation_steps.embed_representations(context)
    assert encoded == [
        ["title 1", "title 2"],
        ["qwen c1", "qwen c2"],
        ["qwen c1", "gemini c2"],
        ["description c1", "description c2"],
    ]
    assert "graph_gemini -> graph_qwen: 1 items" in before_load[0]
    assert "item_id=1 | content_id=c1" in before_load[0]
    assert "item_id=2" not in before_load[0]
    assert qwen_path.read_bytes() == before
    assert not (context.graph_summary_dir("gemini") / "c1.json").exists()
    assert read_json(context.representations_dir / "graph_gemini_fallbacks.json") == {
        "fallbacks": [
            {
                "item_id": "1",
                "content_id": "c1",
                "source": "graph_qwen",
                "summary_path": "extraction/graph/qwen/summaries/c1.json",
            }
        ],
    }


def test_embedding_falls_back_when_entire_gemini_summary_directory_is_missing(
    context,
    embedding_fallback_case,
):
    encoded, before_load = embedding_fallback_case
    (context.graph_summary_dir("gemini") / "c2.json").unlink()
    context.graph_summary_dir("gemini").rmdir()
    validation_steps.embed_representations(context)
    assert encoded[2] == ["qwen c1", "qwen c2"]
    assert "graph_gemini -> graph_qwen: 2 items" in before_load[0]
    assert "item_id=2 | content_id=c2" in before_load[0]


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        '{"content_id":"c1","text":" "}',
        '{"content_id":"wrong","text":"ok"}',
    ],
)
def test_embedding_does_not_hide_invalid_existing_gemini_summary(
    context,
    embedding_fallback_case,
    contents,
):
    encoded, before_load = embedding_fallback_case
    (context.graph_summary_dir("gemini") / "c1.json").write_text(contents, encoding="utf-8")
    with pytest.raises((ValueError, validation_steps.ValidationStepError)):
        validation_steps.embed_representations(context)
    assert encoded == before_load == []
    assert not (context.representations_dir / "graph_gemini_fallbacks.json").exists()


def test_embedding_fallback_still_requires_qwen_summary_when_qwen_embeddings_are_cached(
    context,
    embedding_fallback_case,
):
    encoded, before_load = embedding_fallback_case
    validation_steps.embed_representations(context)
    fallback_path = context.representations_dir / "graph_gemini_fallbacks.json"
    before = fallback_path.read_bytes()
    qwen_path = context.graph_summary_dir("qwen") / "c1.json"
    qwen_path.unlink()
    (context.representations_dir / "graph_gemini_embeddings.npz").unlink()
    with pytest.raises(
        validation_steps.ValidationStepError, match="graph_qwen (fallback )?summary"
    ) as err:
        validation_steps.embed_representations(context)
    assert str(qwen_path) in str(err.value)
    assert len(encoded) == 4 and len(before_load) == 1
    assert fallback_path.read_bytes() == before


def test_embedding_cache_tracks_fallback_changes_and_prints_cached_list(
    context,
    embedding_fallback_case,
    capsys,
):
    encoded, before_load = embedding_fallback_case
    validation_steps.embed_representations(context)
    stable = {
        branch: (context.representations_dir / f"{branch}_embeddings.npz").read_bytes()
        for branch in ("metadata", "graph_qwen", "desc")
    }
    validation_steps.embed_representations(context)
    output = capsys.readouterr().out
    assert "cached embeddings" in output and "item_id=1 | content_id=c1" in output
    assert len(before_load) == 1

    gemini_path = context.graph_summary_dir("gemini") / "c1.json"
    write_json(gemini_path, {"content_id": "c1", "text": "gemini c1"})
    validation_steps.embed_representations(context)
    assert len(encoded) == 5 and encoded[-1] == ["gemini c1", "gemini c2"]
    assert read_json(context.representations_dir / "graph_gemini_fallbacks.json") == {
        "fallbacks": []
    }
    gemini_path.unlink()
    validation_steps.embed_representations(context)
    assert len(encoded) == 6 and encoded[-1] == ["qwen c1", "gemini c2"]
    for branch, contents in stable.items():
        assert (context.representations_dir / f"{branch}_embeddings.npz").read_bytes() == contents


@pytest.mark.parametrize("sidecar", [None, "broken json"])
def test_embedding_does_not_reuse_untracked_fallback_cache(
    context,
    embedding_fallback_case,
    sidecar,
):
    encoded, _ = embedding_fallback_case
    validation_steps.embed_representations(context)
    fallback_path = context.representations_dir / "graph_gemini_fallbacks.json"
    if sidecar is None:
        fallback_path.unlink()
    else:
        fallback_path.write_text(sidecar, encoding="utf-8")
    validation_steps.embed_representations(context)
    assert len(encoded) == 5 and encoded[-1] == ["qwen c1", "gemini c2"]
    assert read_json(fallback_path)["fallbacks"][0]["content_id"] == "c1"


def test_embedding_failure_preserves_previous_fallback_record_and_matrix(
    context,
    embedding_fallback_case,
    monkeypatch,
):
    validation_steps.embed_representations(context)
    fallback_path = context.representations_dir / "graph_gemini_fallbacks.json"
    embedding_path = context.representations_dir / "graph_gemini_embeddings.npz"
    before = (fallback_path.read_bytes(), embedding_path.read_bytes())
    write_json(
        context.graph_summary_dir("gemini") / "c1.json",
        {"content_id": "c1", "text": "gemini c1"},
    )

    class FailingEncoder:
        def __init__(self, _settings):
            pass

        def encode(self, _texts):
            raise RuntimeError("encoding failed")

    monkeypatch.setattr(validation_features, "BGETextEncoder", FailingEncoder)
    with pytest.raises(RuntimeError, match="encoding failed"):
        validation_steps.embed_representations(context)
    assert (fallback_path.read_bytes(), embedding_path.read_bytes()) == before
