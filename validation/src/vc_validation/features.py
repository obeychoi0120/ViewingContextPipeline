from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import EncoderConfig, ExperimentConfig
from .io import atomic_write_json, file_fingerprint, fingerprint, read_jsonl


class FeatureError(RuntimeError):
    pass


def _evidence_id(document: dict[str, Any]) -> str:
    value = document.get("evidence_fingerprint")
    if isinstance(value, dict):
        value = value.get("fingerprint")
    if not isinstance(value, str) or not value:
        raise FeatureError("profile has no evidence_fingerprint")
    return value


def _load_profile(path: Path, content_id: str, arm: str) -> tuple[str, str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureError(f"invalid {arm} profile: {path}") from exc
    if document.get("content_id") != content_id:
        raise FeatureError(f"content id mismatch: {path}")
    if document.get("status") != "complete":
        raise FeatureError(f"incomplete {arm} profile: {path}")
    text = document.get("text")
    if not isinstance(text, str) or not text.strip():
        raise FeatureError(f"empty {arm} profile text: {path}")
    return text.strip(), _evidence_id(document), document


def load_paired_profile_texts(config: ExperimentConfig, catalog: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    graph_texts: list[str] = []
    desc_texts: list[str] = []
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in catalog:
        content_id = row["content_id"]
        graph_path = config.dataset.vp_graph_dir / f"{content_id}_vp_graph.json"
        desc_path = config.dataset.vp_desc_dir / f"{content_id}_vp_desc.json"
        if not graph_path.is_file() or not desc_path.is_file():
            missing.append(content_id)
            continue
        graph_text, graph_evidence, _ = _load_profile(graph_path, content_id, "graph")
        desc_text, desc_evidence, _ = _load_profile(desc_path, content_id, "description")
        if graph_evidence != desc_evidence:
            raise FeatureError(f"evidence fingerprint mismatch for {content_id}")
        graph_texts.append(graph_text)
        desc_texts.append(desc_text)
        sources.append({"content_id": content_id, "evidence_fingerprint": graph_evidence, "graph": file_fingerprint(graph_path), "desc": file_fingerprint(desc_path)})
    if missing:
        raise FeatureError(f"representation completeness is not 100%; missing {len(missing)} paired profiles")
    return graph_texts, desc_texts, sources


def encode_bge_texts(settings: EncoderConfig, texts: list[str]) -> np.ndarray:
    if not settings.model_path.is_dir():
        raise FeatureError(f"local encoder path does not exist: {settings.model_path}")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise FeatureError("torch and transformers are required to materialize BGE representations") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(settings.model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(settings.model_path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), settings.batch_size):
            encoded = tokenizer(texts[start:start + settings.batch_size], padding=True, truncation=True, max_length=settings.max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            batches.append(hidden.cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0) if batches else np.empty((0, settings.embedding_dim), dtype=np.float32)


def _validate_matrix(matrix: np.ndarray, rows: int, dimension: int, arm: str) -> None:
    if matrix.shape != (rows, dimension):
        raise FeatureError(f"unexpected {arm} matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise FeatureError(f"{arm} matrix contains non-finite values")
    if rows and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise FeatureError(f"{arm} embeddings are not L2-normalized")


def encoder_file_manifest(path: Path) -> list[dict[str, Any]]:
    allowed = {".json", ".txt", ".model", ".safetensors", ".bin"}
    return [file_fingerprint(file) for file in sorted(path.rglob("*")) if file.is_file() and file.suffix.lower() in allowed]


def materialize_representations(config: ExperimentConfig, *, encoder: Callable[[EncoderConfig, list[str]], np.ndarray] = encode_bge_texts) -> dict[str, Any]:
    catalog = read_jsonl(config.output_dir / "cohort" / "catalog.jsonl")
    graph_texts, desc_texts, sources = load_paired_profile_texts(config, catalog)
    graph = np.asarray(encoder(config.encoder, graph_texts), dtype=np.float32)
    desc = np.asarray(encoder(config.encoder, desc_texts), dtype=np.float32)
    _validate_matrix(graph, len(catalog), config.encoder.embedding_dim, "graph")
    _validate_matrix(desc, len(catalog), config.encoder.embedding_dim, "description")
    output = config.output_dir / "representations"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "vp_graph_embeddings.npz", values=graph)
    np.savez_compressed(output / "vp_desc_embeddings.npz", values=desc)
    item_index = {row["item_id"]: index for index, row in enumerate(catalog)}
    atomic_write_json(output / "item_index.json", item_index)
    model_files = encoder_file_manifest(config.encoder.model_path)
    encoder_contract = {
        "model_id": config.encoder.model_id, "model_path": str(config.encoder.model_path.resolve()),
        "embedding_dim": config.encoder.embedding_dim, "max_length": config.encoder.max_length,
        "pooling": "cls", "normalize_embeddings": True, "query_instruction": None,
        "local_files_only": True, "files": model_files,
    }
    manifest = {
        "schema_version": "visual-profile-representations/v1", "catalog_size": len(catalog),
        "graph_completeness": 1.0, "desc_completeness": 1.0,
        "failure_count": 0,
        "dimension": config.encoder.embedding_dim, "item_ids": list(item_index),
        "source_fingerprint": fingerprint(sources), "encoder": encoder_contract,
        "encoder_fingerprint": fingerprint(encoder_contract),
    }
    atomic_write_json(output / "representation_manifest.json", manifest)
    return manifest


# Transitional name for callers upgrading from v1.
materialize_features = materialize_representations
