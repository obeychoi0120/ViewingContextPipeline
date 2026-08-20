from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml


STAGES = (
    "preflight",
    "prepare-cohort",
    "import-microlens",
    "extract-graph",
    "build-graph-profiles",
    "build-description-profiles",
    "materialize-representations",
    "run-experiment",
)
RUN_SCHEMA = "viewing-context-pipeline/v1"
LOCAL_SCHEMA = "viewing-context-local/v1"
MANIFEST_SCHEMA = "viewing-context-pipeline-run/v1"


class PipelineError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"failed to read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"YAML root must be an object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]

    def document(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "cwd": str(self.cwd)}


@dataclass(frozen=True)
class PipelineContext:
    root: Path
    pipeline_path: Path
    local_path: Path
    pipeline: dict[str, Any]
    local: dict[str, Any]
    run_id: str
    run_root: Path
    extraction_root: Path
    validation_root: Path

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        local_path: str | Path,
        *,
        root: Path | None = None,
    ) -> "PipelineContext":
        repo_root = (root or Path(__file__).resolve().parents[1]).resolve()
        pipeline_path = _resolve(repo_root, str(config_path), "config")
        machine_path = _resolve(repo_root, str(local_path), "local_config")
        pipeline = _load_yaml(pipeline_path)
        local = _load_yaml(machine_path)
        if pipeline.get("schema_version") != RUN_SCHEMA:
            raise PipelineError(f"schema_version must be {RUN_SCHEMA}")
        if local.get("schema_version") != LOCAL_SCHEMA:
            raise PipelineError(f"local schema_version must be {LOCAL_SCHEMA}")
        run_id = pipeline.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_id):
            raise PipelineError("run_id must contain lowercase letters, digits, dot, underscore, or dash")
        protocol = pipeline.get("protocol")
        expected = {
            "modality": "img_only",
            "shot_interval": "fixed_30s",
            "graph_profile": "VP_graph",
            "description_profile": "VP_desc",
        }
        if protocol != expected:
            raise PipelineError(f"canonical protocol must be exactly {expected}")
        if pipeline.get("scope") not in {"pilot", "canonical"}:
            raise PipelineError("scope must be pilot or canonical")
        artifact_root = _resolve(repo_root, pipeline.get("artifacts_root", "artifacts"), "artifacts_root")
        extraction_root = _resolve(repo_root, pipeline.get("extraction_root", "extraction"), "extraction_root")
        validation_root = _resolve(repo_root, pipeline.get("validation_root", "validation"), "validation_root")
        return cls(
            root=repo_root,
            pipeline_path=pipeline_path,
            local_path=machine_path,
            pipeline=pipeline,
            local=local,
            run_id=run_id,
            run_root=artifact_root / run_id,
            extraction_root=extraction_root,
            validation_root=validation_root,
        )

    @property
    def extraction_output(self) -> Path:
        # Existing Extraction CLIs preserve a root whose final directory starts
        # with "microlens"; other names are routed below a custom/ directory.
        return self.run_root / "extraction" / "microlens"

    @property
    def validation_output(self) -> Path:
        return self.run_root / "validation"

    @property
    def runtime_root(self) -> Path:
        return self.run_root / "runtime"

    @property
    def runtime_paths(self) -> dict[str, Path]:
        return {
            "microlens": self.runtime_root / "extraction" / "microlens_config.json",
            "processing": self.runtime_root / "extraction" / "video_data_collection.json",
            "graph": self.runtime_root / "extraction" / "scene_context_extraction_ondevice.json",
            "description": self.runtime_root / "extraction" / "scene_description_generation.json",
            "validation": self.runtime_root / "validation" / "experiment.yaml",
        }

    @property
    def selection_path(self) -> Path:
        return self.validation_output / "cohort" / "vce_selection.jsonl"

    @property
    def smoke_selection_path(self) -> Path:
        return self.validation_output / "cohort" / "vce_smoke_selection.jsonl"

    @property
    def import_manifest(self) -> Path:
        return self.extraction_output / "manifests" / "catalog_manifest.csv"

    @property
    def graph_profile_dir(self) -> Path:
        return self.extraction_output / "viewing_context" / "img_only" / "fixed_30s" / "video_profile_graph_qwen"

    @property
    def description_profile_dir(self) -> Path:
        return self.extraction_output / "viewing_context" / "img_only" / "fixed_30s" / "video_profile_desc_qwen"

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "pipeline_manifest.json"

    def component_path(self, key: str) -> Path:
        components = self.pipeline.get("components")
        if not isinstance(components, dict) or key not in components:
            raise PipelineError(f"components.{key} is required")
        return _resolve(self.root, components[key], f"components.{key}")

    def local_path_value(self, section: str, key: str) -> Path:
        values = self.local.get(section)
        if not isinstance(values, dict) or key not in values:
            raise PipelineError(f"local config requires {section}.{key}")
        return _resolve(self.root, values[key], f"{section}.{key}")


def runtime_documents(context: PipelineContext) -> dict[str, dict[str, Any]]:
    microlens = _load_json(context.component_path("microlens_config"))
    processing = _load_json(context.component_path("processing_config"))
    graph = _load_json(context.component_path("graph_config"))
    description = _load_json(context.component_path("description_config"))
    validation = _load_yaml(context.component_path("validation_config"))

    processing["shot_interval"] = "fixed_30s"
    processing.setdefault("asr_config", {})["enabled"] = False
    processing.setdefault("ocr_config", {})["enabled"] = False

    graph.update({
        "multimodal": False,
        "MODEL_FAMILY": "qwen3_vl",
        "MODEL_PATH": str(context.local_path_value("models", "qwen")),
        "shot_interval": "fixed_30s",
    })
    description.update({
        "multimodal": False,
        "MODEL_FAMILY": "qwen3_vl",
        "MODEL_PATH": str(context.local_path_value("models", "qwen")),
        "shot_interval": "fixed_30s",
    })

    source = microlens.setdefault("source", {})
    source.update({
        "videos_dir": str(context.local_path_value("data", "videos_dir")),
        "titles_csv": str(context.local_path_value("data", "titles_csv")),
        "tags_csv": str(context.local_path_value("data", "tags_csv")),
        "pairs_tsv": str(context.local_path_value("data", "pairs_tsv")),
        "selection_jsonl": str(context.selection_path),
        "smoke_selection_jsonl": str(context.smoke_selection_path),
    })
    microlens["output_root"] = str(context.extraction_output)
    microlens["processing_config_path"] = str(context.runtime_paths["processing"])
    microlens["sampling"] = {
        "shot_interval": "fixed_30s",
        "interval_seconds": 10,
        "frames_per_scene": 3,
        "ocr_fps": 1,
        "resize_mode": "contain_pad",
        "padding_color": "black",
    }
    microlens["outputs"] = {
        "inventory_jsonl": "manifests/item_inventory.jsonl",
        "failures_jsonl": "manifests/import_failures.jsonl",
        "selection_json": "manifests/selection.json",
        "pilot_manifest_csv": "manifests/catalog_manifest.csv",
        "smoke_manifest_csv": "manifests/smoke_manifest.csv",
        "pilot_categories_jsonl": "manifests/catalog_categories.jsonl",
        "smoke_categories_jsonl": "manifests/smoke_categories.jsonl",
    }

    dataset = validation.setdefault("dataset", {})
    dataset.update({
        "pairs_tsv": str(context.local_path_value("data", "pairs_tsv")),
        "videos_dir": str(context.local_path_value("data", "videos_dir")),
        "vp_graph_dir": str(context.graph_profile_dir),
        "vp_desc_dir": str(context.description_profile_dir),
    })
    validation["run_id"] = context.run_id
    validation.setdefault("encoder", {})["model_path"] = str(context.local_path_value("models", "bge"))
    validation["output_dir"] = str(context.validation_output)

    return {
        "microlens": microlens,
        "processing": processing,
        "graph": graph,
        "description": description,
        "validation": validation,
    }


def write_runtime_configs(context: PipelineContext) -> None:
    documents = runtime_documents(context)
    paths = context.runtime_paths
    for key in ("microlens", "processing", "graph", "description"):
        _atomic_text(paths[key], json.dumps(documents[key], ensure_ascii=False, indent=2) + "\n")
    _atomic_text(paths["validation"], yaml.safe_dump(documents["validation"], sort_keys=False, allow_unicode=True))


def preflight(context: PipelineContext) -> dict[str, Any]:
    runtime_documents(context)
    checks: dict[str, bool] = {}
    for key in ("pairs_tsv", "titles_csv", "tags_csv"):
        checks[f"data.{key}"] = context.local_path_value("data", key).is_file()
    checks["data.videos_dir"] = context.local_path_value("data", "videos_dir").is_dir()
    checks["models.qwen"] = context.local_path_value("models", "qwen").is_dir()
    checks["models.bge"] = context.local_path_value("models", "bge").is_dir()
    checks["ffmpeg"] = shutil.which("ffmpeg") is not None
    checks["ffprobe"] = shutil.which("ffprobe") is not None
    checks["python.torch"] = importlib.util.find_spec("torch") is not None
    checks["python.transformers"] = importlib.util.find_spec("transformers") is not None
    checks["extraction"] = context.extraction_root.is_dir()
    checks["validation"] = context.validation_root.is_dir()
    return {
        "schema_version": "viewing-context-preflight/v1",
        "run_id": context.run_id,
        "ready": all(checks.values()),
        "checks": checks,
    }


def _base_env(context: PipelineContext) -> dict[str, str]:
    env = os.environ.copy()
    env["OUTPUT_SAVE_PATH"] = str(context.extraction_output)
    return env


def command_for_stage(context: PipelineContext, stage: str, *, force: bool = False) -> Command:
    if stage not in STAGES or stage == "preflight":
        raise PipelineError(f"stage has no subprocess command: {stage}")
    python = sys.executable
    runtime = context.runtime_paths
    env = _base_env(context)
    if stage in {"prepare-cohort", "materialize-representations", "run-experiment"}:
        current = env.get("PYTHONPATH", "")
        validation_src = str(context.validation_root / "src")
        env["PYTHONPATH"] = validation_src + (os.pathsep + current if current else "")
        command_name = {
            "prepare-cohort": "prepare-cohort",
            "materialize-representations": "materialize-representations",
            "run-experiment": "run-experiment",
        }[stage]
        argv = (python, "-m", "vc_validation.cli", "--config", str(runtime["validation"]), command_name)
        return Command(argv, context.validation_root, env)
    if stage == "import-microlens":
        argv = (
            python, "-m", "src.video_data_collection.cli", "import-microlens",
            "--config", str(runtime["microlens"]), "--scope", "pilot",
        )
        if force:
            argv += ("--force",)
        return Command(argv, context.extraction_root, env)
    if stage == "extract-graph":
        argv = (
            python, "-m", "src.scene_context_extraction.ondevice.cli",
            "--manifest", str(context.import_manifest), "--settings", str(runtime["graph"]),
        )
        if force:
            argv += ("--force",)
        return Command(argv, context.extraction_root, env)
    if stage == "build-graph-profiles":
        argv = (
            python, "-m", "src.scene_description_generation.graph_profile_cli",
            "--manifest", str(context.selection_path), "--settings", str(runtime["graph"]),
        )
        if force:
            argv += ("--force",)
        return Command(argv, context.extraction_root, env)
    if stage == "build-description-profiles":
        argv = (
            python, "-m", "src.scene_description_generation.cli",
            "--manifest", str(context.selection_path), "--settings", str(runtime["description"]),
        )
        if force:
            argv += ("--force",)
        return Command(argv, context.extraction_root, env)
    raise PipelineError(f"unsupported stage: {stage}")


def prerequisites(context: PipelineContext, stage: str) -> tuple[Path, ...]:
    mapping = {
        "prepare-cohort": (),
        "import-microlens": (context.selection_path,),
        "extract-graph": (context.import_manifest,),
        "build-graph-profiles": (context.selection_path,),
        "build-description-profiles": (context.selection_path,),
        "materialize-representations": (context.graph_profile_dir, context.description_profile_dir),
        "run-experiment": (context.validation_output / "representations" / "representation_manifest.json",),
    }
    return mapping.get(stage, ())


def outputs_for_stage(context: PipelineContext, stage: str) -> tuple[Path, ...]:
    mapping = {
        "prepare-cohort": (context.selection_path,),
        "import-microlens": (context.import_manifest,),
        "extract-graph": (
            context.extraction_output / "viewing_context" / "img_only" / "fixed_30s" / "video_context_graph_qwen",
        ),
        "build-graph-profiles": (context.graph_profile_dir,),
        "build-description-profiles": (context.description_profile_dir,),
        "materialize-representations": (
            context.validation_output / "representations" / "representation_manifest.json",
        ),
        "run-experiment": (
            context.validation_output / "experiment" / "report.json",
            context.validation_output / "experiment" / "report_ready.json",
        ),
    }
    return mapping.get(stage, ())


def _path_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        record["sha256"] = _sha256(path)
    return record


def _read_manifest(context: PipelineContext) -> dict[str, Any]:
    if not context.manifest_path.is_file():
        return {
            "schema_version": MANIFEST_SCHEMA,
            "run_id": context.run_id,
            "created_at": _now(),
            "pipeline_config": {"path": str(context.pipeline_path), "sha256": _sha256(context.pipeline_path)},
            "local_config": {"path": str(context.local_path), "sha256": _sha256(context.local_path)},
            "git_head": _git_head(context.root),
            "stages": {},
        }
    value = _load_json(context.manifest_path)
    if value.get("schema_version") != MANIFEST_SCHEMA or value.get("run_id") != context.run_id:
        raise PipelineError(f"invalid pipeline manifest: {context.manifest_path}")
    return value


def _write_manifest(context: PipelineContext, manifest: dict[str, Any]) -> None:
    _atomic_text(context.manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def _stage_state(context: PipelineContext, stage: str) -> dict[str, Any] | None:
    return _read_manifest(context).get("stages", {}).get(stage)


def _record_stage(context: PipelineContext, stage: str, payload: dict[str, Any]) -> None:
    manifest = _read_manifest(context)
    manifest.setdefault("stages", {})[stage] = payload
    manifest["updated_at"] = _now()
    _write_manifest(context, manifest)


def initialize_run(context: PipelineContext) -> None:
    if context.run_root.exists() and any(context.run_root.iterdir()):
        raise PipelineError(f"fresh run directory is not empty: {context.run_root}")
    context.run_root.mkdir(parents=True, exist_ok=True)
    write_runtime_configs(context)
    _write_manifest(context, _read_manifest(context))


def execute_stage(
    context: PipelineContext,
    stage: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if stage == "preflight":
        return preflight(context)
    command = command_for_stage(context, stage, force=force)
    if dry_run:
        return {"stage": stage, "dry_run": True, "command": command.document()}
    if stage == "prepare-cohort" and (
        not context.run_root.exists() or not any(context.run_root.iterdir())
    ):
        initialize_run(context)
    elif not context.run_root.is_dir():
        raise PipelineError(f"run directory does not exist; run prepare-cohort first: {context.run_root}")
    if not all(path.exists() for path in context.runtime_paths.values()):
        raise PipelineError(f"runtime configs are incomplete: {context.runtime_root}")
    missing = [str(path) for path in prerequisites(context, stage) if not path.exists()]
    if missing:
        raise PipelineError(f"{stage} is missing prerequisite inputs: {', '.join(missing)}")
    current = _stage_state(context, stage)
    if current and current.get("status") == "complete" and not force:
        if resume:
            return {"stage": stage, "status": "skipped", "reason": "already complete"}
        raise PipelineError(f"stage is already complete; use --resume or --force: {stage}")
    inputs = [_path_record(path) for path in prerequisites(context, stage)]
    _record_stage(context, stage, {
        "status": "running",
        "started_at": _now(),
        "command": command.document(),
        "inputs": inputs,
    })
    try:
        subprocess.run(command.argv, cwd=command.cwd, env=command.env, check=True)
    except subprocess.CalledProcessError as exc:
        _record_stage(context, stage, {
            "status": "failed",
            "finished_at": _now(),
            "exit_code": exc.returncode,
            "command": command.document(),
            "inputs": inputs,
        })
        raise PipelineError(f"stage failed with exit code {exc.returncode}: {stage}") from exc
    result = {
        "status": "complete",
        "finished_at": _now(),
        "exit_code": 0,
        "command": command.document(),
        "inputs": inputs,
        "outputs": [_path_record(path) for path in outputs_for_stage(context, stage)],
    }
    _record_stage(context, stage, result)
    return {"stage": stage, **result}


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/pipelines/microlens_graph_vs_desc_pilot.yaml")
    parser.add_argument("--local-config", default="config/local.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def stage_cli_main(stage: str, argv: list[str] | None = None) -> int:
    parser = _common_parser(f"Run the {stage} ViewingContextPipeline stage.")
    if stage == "preflight":
        parser.set_defaults(force=False)
    else:
        parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = PipelineContext.load(args.config, args.local_config)
        result = execute_stage(context, stage, dry_run=args.dry_run, resume=args.resume, force=args.force)
    except PipelineError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def runner_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Run the canonical ViewingContextPipeline workflow.")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--force-stage", action="append", choices=STAGES[1:], default=[])
    args = parser.parse_args(argv)
    try:
        context = PipelineContext.load(args.config, args.local_config)
        if args.stage != "all":
            result = execute_stage(
                context,
                args.stage,
                dry_run=args.dry_run,
                resume=args.resume,
                force=args.stage in args.force_stage,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        check = preflight(context)
        if not check["ready"] and not args.dry_run:
            failed = [name for name, passed in check["checks"].items() if not passed]
            raise PipelineError("preflight failed: " + ", ".join(failed))
        if args.dry_run:
            result = {
                "preflight": check,
                "stages": [command_for_stage(context, stage).document() for stage in STAGES[1:]],
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if not context.run_root.exists() or not any(context.run_root.iterdir()):
            initialize_run(context)
        elif not args.resume and not args.force_stage:
            raise PipelineError(f"fresh run directory already exists; use --resume: {context.run_root}")
        first_forced = min((STAGES.index(stage) for stage in args.force_stage), default=len(STAGES))
        results = []
        for index, stage in enumerate(STAGES[1:], start=1):
            results.append(execute_stage(
                context,
                stage,
                resume=args.resume or bool(args.force_stage),
                force=index >= first_forced,
            ))
        print(json.dumps({"preflight": check, "stages": results}, ensure_ascii=False, indent=2))
        return 0
    except PipelineError as exc:
        parser.error(str(exc))
    return 2
