import json

import numpy as np
import pytest

from conftest import config_data
from vc_validation.config import ExperimentConfig
from vc_validation.features import FeatureError, load_paired_profile_texts, materialize_representations


def _profile(path, content_id: str, evidence: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content_id": content_id, "status": "complete", "text": text, "evidence_fingerprint": {"fingerprint": evidence}}), encoding="utf-8")


def test_materialization_enforces_parity_and_normalized_1024d(tmp_path) -> None:
    data = config_data(tmp_path, users=1)
    config = ExperimentConfig.model_validate(data)
    content_id = "microlens_100k_00001"
    cohort = config.output_dir / "cohort"
    cohort.mkdir(parents=True)
    (cohort / "catalog.jsonl").write_text(json.dumps({"item_id": "1", "content_id": content_id}) + "\n", encoding="utf-8")
    config.encoder.model_path.mkdir()
    _profile(config.dataset.vp_graph_dir / f"{content_id}_vp_graph.json", content_id, "same", "graph text")
    _profile(config.dataset.vp_desc_dir / f"{content_id}_vp_desc.json", content_id, "same", "description text")

    def fake_encoder(_, texts):
        values = np.ones((len(texts), 1024), dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    manifest = materialize_representations(config, encoder=fake_encoder)
    assert manifest["dimension"] == 1024
    assert manifest["graph_completeness"] == manifest["desc_completeness"] == 1.0
    graph = np.load(config.output_dir / "representations/vp_graph_embeddings.npz")["values"]
    assert graph.shape == (1, 1024)
    assert np.allclose(np.linalg.norm(graph, axis=1), 1.0)


def test_profile_evidence_mismatch_is_hard_failure(tmp_path) -> None:
    config = ExperimentConfig.model_validate(config_data(tmp_path, users=1))
    content_id = "microlens_100k_00001"
    _profile(config.dataset.vp_graph_dir / f"{content_id}_vp_graph.json", content_id, "graph", "graph")
    _profile(config.dataset.vp_desc_dir / f"{content_id}_vp_desc.json", content_id, "desc", "desc")
    with pytest.raises(FeatureError, match="evidence fingerprint mismatch"):
        load_paired_profile_texts(config, [{"content_id": content_id}])
