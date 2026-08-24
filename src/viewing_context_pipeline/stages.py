from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
import yaml


class StageError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"invalid JSONL: {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise StageError(f"JSONL rows must be objects: {path}")
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Progress:
    def __init__(self, stage: str, total: int | None = None) -> None:
        self.stage = stage
        self.total = total
        self.done = 0
        self.started = perf_counter()

    def phase(self, name: str) -> None:
        print(f"[PHASE] {self.stage} {name}", flush=True)

    def advance(self, label: str = "") -> None:
        self.done += 1
        elapsed = perf_counter() - self.started
        eta = None if not self.total or not self.done else max(0.0, elapsed / self.done * (self.total - self.done))
        suffix = f" eta={eta:.1f}s" if eta is not None else ""
        print(f"[PROGRESS] {self.stage} {self.done}/{self.total or '?'} elapsed={elapsed:.1f}s{suffix} {label}".rstrip(), flush=True)


def _runtime(path: str | Path) -> dict[str, Any]:
    runtime = _read_json(Path(path))
    if runtime.get("schema_version") != "pipeline-runtime/v1":
        raise StageError("runtime schema must be pipeline-runtime/v1")
    return runtime


def _component_env(runtime: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["OUTPUT_SAVE_PATH"] = runtime["run_root"]
    env["LINUX_ASSETS_SAVE_PATH"] = str(Path(runtime["paths"]["cohort_dir"]) / "source_assets")
    env["GCP_PROJECT_ID"] = runtime["cloud"].get("gcp_project_id", "")
    return env


def _run(argv: list[str], *, cwd: Path, runtime: dict[str, Any]) -> None:
    print("[COMMAND] " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, env=_component_env(runtime), check=True)


def _validation_config(runtime: dict[str, Any]):
    from viewing_context_pipeline.validation.config import ValidationConfig

    raw = yaml.safe_load(Path(runtime["components"]["validation_config"]).read_text(encoding="utf-8"))
    raw["run_id"] = runtime["run_id"]
    raw["output_dir"] = runtime["run_root"]
    raw["dataset"].update({
        "pairs_tsv": runtime["data"]["pairs_tsv"],
        "videos_dir": runtime["data"]["videos_dir"],
    })
    raw["encoder"]["model_path"] = runtime["models"]["bge"]
    return ValidationConfig.model_validate(raw)


def _write_runtime_component_configs(runtime: dict[str, Any]) -> dict[str, Path]:
    root = Path(runtime["run_root"]) / "runtime" / "components"
    root.mkdir(parents=True, exist_ok=True)
    modality = runtime["modality"]
    multimodal = modality == "multimodal"
    paths: dict[str, Path] = {}
    for name, key in (("ondevice_graph", "ondevice_graph_config"), ("ondevice_desc", "ondevice_desc_config"), ("gemini_graph", "gemini_graph_config")):
        document = _read_json(Path(runtime["components"][key]))
        document["shot_interval"] = "fixed_30s"
        document["multimodal"] = multimodal
        if name.startswith("ondevice"):
            document["MODEL_PATH"] = runtime["models"]["qwen"]
        if name == "gemini_graph":
            document["gemini_location"] = runtime["cloud"]["gemini_location"]
            document["gemini_model"] = runtime["cloud"]["gemini_model"]
            document["gemini_thinking_level"] = runtime["cloud"]["gemini_thinking_level"]
        destination = root / f"{name}.json"
        _write_json(destination, document)
        paths[name] = destination
    processing = _read_json(Path(runtime["components"]["processing_config"]))
    processing["shot_interval"] = "fixed_30s"
    processing["multimodal"] = multimodal
    processing.setdefault("asr_config", {})["enabled"] = multimodal
    processing.setdefault("ocr_config", {})["enabled"] = multimodal
    if multimodal:
        processing["asr_config"]["REF_MODEL_PATH"] = runtime["models"]["asr"]
    paths["processing"] = root / "data_preparation.json"
    _write_json(paths["processing"], processing)
    return paths


def prepare_data(runtime_path: str | Path) -> None:
    runtime = _runtime(runtime_path)
    progress = Progress("prepare_data")
    progress.phase("cohort")
    from viewing_context_pipeline.validation.cohort import prepare_cohort

    config = _validation_config(runtime)
    cohort_manifest = prepare_cohort(config, output_dir=Path(runtime["paths"]["cohort_dir"]))
    catalog_path = Path(runtime["paths"]["cohort_dir"]) / "catalog.jsonl"
    catalog = _read_jsonl(catalog_path)
    progress.total = len(catalog)
    progress.phase("fixed_30s_keyframes")
    component_paths = _write_runtime_component_configs(runtime)
    from viewing_context_pipeline.extraction.data_preparation.microlens import prepare_catalog

    result = prepare_catalog(
        catalog,
        titles_csv=runtime["data"]["titles_csv"],
        tags_csv=runtime["data"]["tags_csv"],
        assets_root=Path(runtime["paths"]["cohort_dir"]) / "source_assets",
        output_root=runtime["run_root"],
        processing_config=component_paths["processing"],
        force=True,
    )
    if result["failed"] or result["succeeded"] != len(catalog):
        raise StageError(f"fixed_30s preparation incomplete: {result}")
    progress.phase("canonical_contracts")
    if runtime["modality"] == "multimodal":
        multimodal_root = Path(runtime["paths"]["multimodal_ref_dir"])
        for row in catalog:
            content_id = row["content_id"]
            reference_path = multimodal_root / f"{content_id}_multimodal_ref.jsonl"
            reference_rows = _read_jsonl(reference_path)
            _write_jsonl(reference_path, [
                {"schema_version": "multimodal-reference/v1", "content_id": content_id, **reference}
                for reference in reference_rows
            ])
    progress.phase("visual_manifest")
    visual_rows: list[dict[str, Any]] = []
    for row in catalog:
        content_id = row["content_id"]
        frames_dir = Path(runtime["paths"]["data_dir"]) / "resized_keyframes" / content_id
        timestamp = Path(runtime["paths"]["cohort_dir"]) / "source_assets" / content_id / "assets" / "timestamp_fixed_30s.json"
        frames = sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if not frames or not timestamp.is_file():
            raise StageError(f"missing visual evidence for {content_id}")
        scenes = json.loads(timestamp.read_text(encoding="utf-8"))
        visual_rows.append({
            "schema_version": "visual-manifest/v1", "content_id": content_id,
            "item_id": row["item_id"], "frames_dir": str(frames_dir),
            "timestamp_json": str(timestamp), "frame_count": len(frames), "scenes": scenes,
            "evidence_fingerprint": _hash({"timestamp": _file_hash(timestamp), "frames": [_file_hash(path) for path in frames]}),
        })
        progress.advance(content_id)
    _write_jsonl(Path(runtime["paths"]["visual_manifest"]), visual_rows)
    if runtime["modality"] == "visual_only" and Path(runtime["paths"]["multimodal_ref_dir"]).exists():
        if any(Path(runtime["paths"]["multimodal_ref_dir"]).iterdir()):
            raise StageError("visual_only preparation created multimodal_ref artifacts")
    if runtime["modality"] == "multimodal":
        _validate_multimodal_refs(runtime, visual_rows)
    multimodal_fingerprints = (
        {
            row["content_id"]: _file_hash(Path(runtime["paths"]["multimodal_ref_dir"]) / f"{row['content_id']}_multimodal_ref.jsonl")
            for row in visual_rows
        }
        if runtime["modality"] == "multimodal"
        else None
    )
    manifest = {
        "schema_version": "prepared-data/v1", "run_id": runtime["run_id"],
        "modality": runtime["modality"], "sampling": "fixed_30s", "complete": True,
        "catalog_size": len(catalog), "cohort_fingerprint": cohort_manifest["cohort_fingerprint"],
        "visual_manifest": str(runtime["paths"]["visual_manifest"]),
        "multimodal_ref_dir": runtime["paths"]["multimodal_ref_dir"] if runtime["modality"] == "multimodal" else None,
        "fingerprint": _hash({"cohort": cohort_manifest["cohort_fingerprint"], "visual": visual_rows, "multimodal_refs": multimodal_fingerprints, "modality": runtime["modality"]}),
    }
    _write_json(Path(runtime["paths"]["prepared_data_manifest"]), manifest)
    print(f"[OUTPUT] {runtime['paths']['prepared_data_manifest']} fingerprint={manifest['fingerprint']}", flush=True)


def _timestamps_from_visual(row: dict[str, Any]) -> list[list[float | int]]:
    raw = json.loads(Path(row["timestamp_json"]).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise StageError(f"empty timestamp document: {row['timestamp_json']}")
    return [scene.get("keyframe_timestamps", []) for scene in raw if isinstance(scene, dict)]


def _validate_multimodal_refs(runtime: dict[str, Any], visual_rows: list[dict[str, Any]]) -> None:
    root = Path(runtime["paths"]["multimodal_ref_dir"])
    for visual in visual_rows:
        path = root / f"{visual['content_id']}_multimodal_ref.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            raise StageError(f"missing or empty multimodal_ref: {path}")
        scenes = _read_jsonl(path)
        expected = _timestamps_from_visual(visual)
        if len(scenes) != len(expected):
            raise StageError(f"scene count mismatch for {visual['content_id']}")
        for index, (scene, timestamps) in enumerate(zip(scenes, expected)):
            timeline = scene.get("timeline")
            if not isinstance(timeline, list) or len(timeline) != len(timestamps):
                raise StageError(f"image/reference mismatch for {visual['content_id']} scene {index}")
            for entry in timeline:
                if not isinstance(entry, dict) or not isinstance(entry.get("raw_asr", ""), str) or not isinstance(entry.get("raw_ocr", ""), str):
                    raise StageError(f"invalid multimodal_ref type for {visual['content_id']} scene {index}")


def _extraction_manifest(runtime: dict[str, Any]) -> Path:
    return Path(runtime["paths"]["cohort_dir"]) / "extraction_manifest.csv"


def _canonical_context(runtime: dict[str, Any], branch: str, source: dict[str, Any], source_path: Path, visual: dict[str, Any]) -> dict[str, Any]:
    context_type = "graph" if branch.endswith("graph") else "description"
    text = source.get("text")
    if not isinstance(text, str) or not text.strip():
        if context_type == "graph" and isinstance(source.get("context"), dict):
            from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.context_text import serialize_graph_context
            text = serialize_graph_context(source["context"])
        else:
            raise StageError(f"empty context text: {source_path}")
    evidence = visual["evidence_fingerprint"]
    if runtime["modality"] == "multimodal":
        multimodal_ref_path = Path(runtime["paths"]["multimodal_ref_dir"]) / f"{visual['content_id']}_multimodal_ref.jsonl"
        evidence = _hash({"visual": evidence, "multimodal_ref": _file_hash(multimodal_ref_path)})
    model_config = Path(runtime["run_root"]) / "runtime" / "components" / f"{branch}.json"
    return {
        "schema_version": "video-context/v1", "content_id": visual["content_id"],
        "context_type": context_type, "backend": branch.removesuffix("_graph").removesuffix("_desc"),
        "branch": branch, "modality": runtime["modality"], "status": "complete", "text": text.strip(),
        "evidence_fingerprint": evidence,
        "model_fingerprint": _file_hash(model_config) if model_config.is_file() else _hash(branch),
        "source": {"path": str(source_path), "sha256": _file_hash(source_path)},
    }


def _source_profiles(runtime: dict[str, Any], branch: str) -> list[Path]:
    mode = "multimodal" if runtime["modality"] == "multimodal" else "img_only"
    base = Path(runtime["run_root"]) / "viewing_context" / mode / "fixed_30s"
    patterns = {
        "ondevice_graph": (base / "video_context_graph_qwen", "*_context_graph_ond.json"),
        "ondevice_desc": (base / "video_context_desc_qwen", "*_context_desc_qwen.json"),
        "gemini_graph": (base / "video_context_graph_gemini", "*_context_graph_gemini.json"),
        "gemini_desc": (base / "video_context_desc_gemini", "*_context_desc_gemini.json"),
    }
    directory, pattern = patterns[branch]
    return sorted(directory.glob(pattern))


def extract_context(runtime_path: str | Path, branch: str) -> None:
    runtime = _runtime(runtime_path)
    if branch not in runtime["enabled_branches"]:
        raise StageError(f"branch is disabled: {branch}")
    progress = Progress(f"extract_{branch}_context")
    component_paths = _write_runtime_component_configs(runtime)
    root = Path(runtime["repo_root"])
    manifest = _extraction_manifest(runtime)
    progress.phase("model_inference")
    if branch == "ondevice_graph":
        _run([sys.executable, "-m", "viewing_context_pipeline.extraction.scene_context_extraction.ondevice.cli", "--manifest", str(manifest), "--settings", str(component_paths[branch]), "--force"], cwd=root, runtime=runtime)
    elif branch == "ondevice_desc":
        _run([sys.executable, "-m", "viewing_context_pipeline.extraction.scene_description_generation.cli", "--manifest", str(Path(runtime["paths"]["cohort_dir"]) / "catalog.jsonl"), "--settings", str(component_paths[branch]), "--force"], cwd=root, runtime=runtime)
    elif branch == "gemini_graph":
        _run([sys.executable, "-m", "viewing_context_pipeline.extraction.scene_context_extraction.gemini.cli", "--manifest", str(manifest), "--settings", str(component_paths[branch]), "--gcp-project-id", runtime["cloud"]["gcp_project_id"], "--force"], cwd=root, runtime=runtime)
    else:
        _run([sys.executable, "-m", "viewing_context_pipeline.extraction.scene_description_generation.gemini_cli", "--manifest", str(manifest), "--settings", str(component_paths["gemini_graph"]), "--force"], cwd=root, runtime=runtime)
    progress.phase("canonical_handoff")
    visual_rows = _read_jsonl(Path(runtime["paths"]["visual_manifest"]))
    progress.total = len(visual_rows)
    if runtime["modality"] == "multimodal":
        _validate_multimodal_refs(runtime, visual_rows)
    by_id = {row["content_id"]: row for row in visual_rows}
    source_paths = _source_profiles(runtime, branch)
    sources: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in source_paths:
        document = _read_json(path)
        content_id = str(document.get("content_id", ""))
        if content_id:
            sources[content_id] = (document, path)
    missing = sorted(set(by_id) - set(sources))
    if missing:
        raise StageError(f"{branch} is incomplete; missing {len(missing)} contents")
    output_dir = Path(runtime["paths"]["context_root"]) / branch
    documents = []
    for content_id, visual in by_id.items():
        source, source_path = sources[content_id]
        document = _canonical_context(runtime, branch, source, source_path, visual)
        destination = output_dir / f"{content_id}.json"
        _write_json(destination, document)
        documents.append(document)
        progress.advance(content_id)
    manifest_document = {
        "schema_version": "video-context-manifest/v1", "run_id": runtime["run_id"],
        "branch": branch, "modality": runtime["modality"], "context_type": documents[0]["context_type"],
        "content_count": len(documents), "complete": True,
        "fingerprint": _hash([{key: row[key] for key in ("content_id", "evidence_fingerprint", "model_fingerprint")} for row in documents]),
    }
    _write_json(output_dir / "manifest.json", manifest_document)
    print(f"[OUTPUT] {output_dir / 'manifest.json'} fingerprint={manifest_document['fingerprint']}", flush=True)


def embed_representations(runtime_path: str | Path) -> None:
    runtime = _runtime(runtime_path)
    progress = Progress("embed_representations", len(runtime["enabled_branches"]))
    from viewing_context_pipeline.validation.features import encode_bge_texts, encoder_file_manifest

    config = _validation_config(runtime)
    catalog = _read_jsonl(Path(runtime["paths"]["cohort_dir"]) / "catalog.jsonl")
    content_ids = [row["content_id"] for row in catalog]
    output = Path(runtime["paths"]["representations_manifest"]).parent
    output.mkdir(parents=True, exist_ok=True)
    branches: dict[str, Any] = {}
    common_evidence: dict[str, str] = {}
    for branch in runtime["enabled_branches"]:
        progress.phase(branch)
        context_dir = Path(runtime["paths"]["context_root"]) / branch
        documents = [_read_json(context_dir / f"{content_id}.json") for content_id in content_ids]
        if any(row.get("schema_version") != "video-context/v1" or row.get("modality") != runtime["modality"] or row.get("status") != "complete" for row in documents):
            raise StageError(f"invalid or mixed-modality contexts in {branch}")
        for row in documents:
            previous = common_evidence.setdefault(row["content_id"], row["evidence_fingerprint"])
            if previous != row["evidence_fingerprint"]:
                raise StageError(f"evidence fingerprint mismatch for {row['content_id']}")
        matrix = np.asarray(encode_bge_texts(config.encoder, [row["text"] for row in documents]), dtype=np.float32)
        if matrix.shape != (len(catalog), config.encoder.embedding_dim) or not np.isfinite(matrix).all():
            raise StageError(f"invalid embedding matrix for {branch}: {matrix.shape}")
        filename = f"{branch}_embeddings.npz"
        np.savez_compressed(output / filename, values=matrix)
        branches[branch] = {"path": str(output / filename), "context_type": documents[0]["context_type"], "source_fingerprint": _read_json(context_dir / "manifest.json")["fingerprint"]}
        progress.advance(branch)
    item_index = {row["item_id"]: index for index, row in enumerate(catalog)}
    _write_json(output / "item_index.json", item_index)
    encoder = {"model_path": str(config.encoder.model_path), "dimension": config.encoder.embedding_dim, "files": encoder_file_manifest(config.encoder.model_path)}
    manifest = {
        "schema_version": "representations/v1", "run_id": runtime["run_id"], "modality": runtime["modality"],
        "catalog_size": len(catalog), "dimension": config.encoder.embedding_dim, "branches": branches,
        "encoder": encoder, "complete": True, "fingerprint": _hash({"branches": branches, "encoder": encoder}),
    }
    _write_json(Path(runtime["paths"]["representations_manifest"]), manifest)
    print(f"[OUTPUT] {runtime['paths']['representations_manifest']} fingerprint={manifest['fingerprint']}", flush=True)


def run_recommendation(runtime_path: str | Path) -> None:
    runtime = _runtime(runtime_path)
    progress = Progress("run_recommendation")
    progress.phase("independent_sasrec_training")
    from viewing_context_pipeline.validation.recommendation import train_recommendation_arms
    manifest = train_recommendation_arms(_validation_config(runtime), runtime)
    _write_json(Path(runtime["paths"]["recommendations_manifest"]), manifest)
    progress.total = 1
    progress.advance(f"arms={len(manifest['arms'])}")
    print(f"[OUTPUT] {runtime['paths']['recommendations_manifest']} fingerprint={manifest['fingerprint']}", flush=True)


def run_diagnosis(runtime_path: str | Path) -> None:
    runtime = _runtime(runtime_path)
    progress = Progress("run_diagnosis", 1)
    progress.phase("metrics_and_readiness")
    from viewing_context_pipeline.validation.diagnosis import diagnose_recommendations
    document = diagnose_recommendations(_validation_config(runtime), runtime)
    _write_json(Path(runtime["paths"]["diagnosis"]), document)
    progress.advance("report_ready=true")
    print(f"[OUTPUT] {runtime['paths']['diagnosis']} fingerprint={document['fingerprint']}", flush=True)


STAGE_HANDLERS: dict[str, Callable[[str | Path], None]] = {
    "prepare_data": prepare_data,
    "extract_ondevice_graph_context": lambda path: extract_context(path, "ondevice_graph"),
    "extract_ondevice_desc_context": lambda path: extract_context(path, "ondevice_desc"),
    "extract_gemini_graph_context": lambda path: extract_context(path, "gemini_graph"),
    "extract_gemini_desc_context": lambda path: extract_context(path, "gemini_desc"),
    "embed_representations": embed_representations,
    "run_recommendation": run_recommendation,
    "run_diagnosis": run_diagnosis,
}
