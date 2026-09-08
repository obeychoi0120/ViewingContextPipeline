"""Portable experiment identity and content-addressed stage dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline_runtime import read_json, write_json


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def text_hash(path):
    # Git checkouts may use CRLF on Windows and LF on Ubuntu.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def initialize_run(context):
    config = json.loads(json.dumps(context.config))
    config.pop("artifacts_root")
    config["data"] = sorted(config["data"])
    for name in ("qwen", "bge"):
        config["models"][name] = (
            str(config["models"][name]).replace("\\", "/").rstrip("/").split("/")[-1]
        )
    prompts = {}
    for branch in ("graph", "description"):
        for kind in ("scene_prompt", "summary_prompt"):
            prompts[f"{branch}.{kind}"] = text_hash(context.config_path("extraction", branch, kind))
            config["extraction"][branch][kind] = f"{branch}.{kind}"
    code = {
        p.relative_to(context.root).as_posix(): text_hash(p)
        for p in sorted((context.root / "src").rglob("*.py"))
    }
    identity = {"config": config, "prompts": prompts, "code": code}
    path = context.run_root / "experiment.json"
    digest = fingerprint(identity)
    if path.exists():
        if read_json(path).get("fingerprint") != digest:
            raise RuntimeError("run config, prompts or implementation changed; use a new run ID")
    else:
        if any(context.run_root.iterdir()):
            raise RuntimeError("v4 requires a new run ID; unversioned artifacts cannot be adopted")
        write_json(
            path,
            {
                "schema_version": "rolling-experiment/v1",
                "fingerprint": digest,
                "identity": identity,
                "config_snapshot": context.config,
            },
        )
    return digest


def bind_stage(context, name, dependencies):
    """A changed dependency is an error even with --force, never a cache hit."""
    experiment = read_json(context.run_root / "experiment.json")["fingerprint"]
    value = {"experiment": experiment, "dependencies": dependencies}
    value["fingerprint"] = fingerprint(value)
    path = context.run_root / "fingerprints" / f"{name}.json"
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"{name} dependencies changed; use a new run ID")
    write_json(path, value)
    return value["fingerprint"]


def model_identity(path):
    files = [
        p
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
    ]
    if not files:
        raise RuntimeError(f"model files missing: {path}")
    return {p.relative_to(path).as_posix(): file_hash(p) for p in files}


def bind_extraction(context, name, *, model="qwen", scene_dir=None):
    if context.config["schema_version"] != "viewing-context-config/v4":
        return
    if name.startswith("extract-graph-scenes"):
        output = context.graph_scene_dir(model)
    elif name.startswith("summarize-graph"):
        output = context.graph_summary_dir(name.rsplit("-", 1)[-1])
    elif name == "extract-description-scenes":
        output = context.description_scene_dir
    else:
        output = context.description_summary_dir
    if not (context.run_root / "fingerprints" / f"{name}.json").exists() and any(
        p.is_file() for p in output.glob("*.json*")
    ):
        raise RuntimeError(f"unbound {name} cache; use a new run ID")
    dependencies = {"cohort": context.require_ready_cohort()["eligibility"]["hashes"]}
    if model == "qwen":
        dependencies["model"] = model_identity(context.path("models", "qwen"))
    else:
        dependencies["model"] = context.config["models"]["gemini"]
    if scene_dir is not None:
        dependencies["scenes"] = {p.name: file_hash(p) for p in sorted(scene_dir.glob("*.jsonl"))}
    else:
        dependencies["keyframes"] = {
            p.relative_to(context.evidence_dir).as_posix(): file_hash(p)
            for p in sorted(context.evidence_dir.rglob("*.png"))
        }
        dependencies["timestamps"] = {
            p.relative_to(context.cohort_dir).as_posix(): file_hash(p)
            for p in sorted((context.cohort_dir / "source_assets").rglob("timestamp_*.json"))
        }
    bind_stage(context, name, dependencies)


REPRESENTATION_FILES = ["item_index.json", "graph_gemini_fallbacks.json"] + [
    f"{branch}_embeddings.npz" for branch in ("metadata", "graph_qwen", "graph_gemini", "desc")
]


def complete_representations(context):
    stage = require_stage(context, "representations")
    write_json(
        context.representations_dir / "complete.json",
        {
            "schema_version": "rolling-representations/v1",
            "fingerprint": stage["fingerprint"],
            "hashes": {
                name: file_hash(context.representations_dir / name) for name in REPRESENTATION_FILES
            },
        },
    )


def verify_representations(context):
    stage = require_stage(context, "representations")
    complete = read_json(context.representations_dir / "complete.json")
    if complete.get("schema_version") != "rolling-representations/v1" or (
        complete.get("fingerprint") != stage["fingerprint"]
        or set(complete.get("hashes", {})) != set(REPRESENTATION_FILES)
    ):
        raise RuntimeError("representation completion/provenance mismatch")
    for name, digest in complete["hashes"].items():
        if file_hash(context.representations_dir / name) != digest:
            raise RuntimeError(f"representation artifact changed: {name}")
    if "documents" in stage["dependencies"]:
        from validation.steps import _embedding_documents

        cohort = context.require_ready_cohort()
        sources = {
            "metadata": None,
            "graph_qwen": context.graph_summary_dir("qwen"),
            "graph_gemini": context.graph_summary_dir("gemini"),
            "desc": context.description_summary_dir,
        }
        fallbacks = read_json(context.representations_dir / "graph_gemini_fallbacks.json")[
            "fallbacks"
        ]
        if any((sources["graph_gemini"] / f"{r['content_id']}.json").exists() for r in fallbacks):
            raise RuntimeError("Gemini fallback inputs changed; use a new run ID")
        documents = _embedding_documents(
            context, cohort["catalog"], sources, list(sources), {r["content_id"] for r in fallbacks}
        )
        if {b: fingerprint(d) for b, d in documents.items()} != stage["dependencies"]["documents"]:
            raise RuntimeError("representation source summaries changed; use a new run ID")
    return complete


def require_stage(context, name):
    stage = read_json(context.run_root / "fingerprints" / f"{name}.json")
    experiment = read_json(context.run_root / "experiment.json")["fingerprint"]
    if (
        not isinstance(stage.get("dependencies"), dict)
        or stage.get("experiment") != experiment
        or stage.get("fingerprint")
        != fingerprint({k: v for k, v in stage.items() if k != "fingerprint"})
    ):
        raise RuntimeError(f"{name} fingerprint does not match the current run")
    if "cohort" in stage["dependencies"] and stage["dependencies"]["cohort"] != read_json(
        context.cohort_dir / "eligibility.json"
    ).get("hashes"):
        raise RuntimeError(f"{name} belongs to a different cohort")
    return stage
