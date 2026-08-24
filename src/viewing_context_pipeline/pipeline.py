from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import yaml


RUN_SCHEMA = "pipeline/v1"
LOCAL_SCHEMA = "viewing-context-local/v1"
MANIFEST_SCHEMA = "pipeline-run/v1"
RUNTIME_SCHEMA = "pipeline-runtime/v1"
TIMEZONE = ZoneInfo("Asia/Seoul")

EXTRACTION_STAGES = (
    "extract_ondevice_graph_context",
    "extract_ondevice_desc_context",
    "extract_gemini_graph_context",
    "extract_gemini_desc_context",
)
STAGES = (
    "prepare_data",
    *EXTRACTION_STAGES,
    "embed_representations",
    "run_recommendation",
    "run_diagnosis",
)
BRANCH_BY_STAGE = {
    "extract_ondevice_graph_context": "ondevice_graph",
    "extract_ondevice_desc_context": "ondevice_desc",
    "extract_gemini_graph_context": "gemini_graph",
    "extract_gemini_desc_context": "gemini_desc",
}
STAGE_BY_BRANCH = {branch: stage for stage, branch in BRANCH_BY_STAGE.items()}
FINAL_STAGES = ("embed_representations", "run_recommendation", "run_diagnosis")


class PipelineError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(TIMEZONE).isoformat()


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=TIMEZONE)
    return current.astimezone(TIMEZONE).strftime("%y%m%d_%H%M")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    started_at: str
    run_root: Path

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        local_path: str | Path,
        *,
        run_id: str | None,
        allow_generate: bool,
        root: Path | None = None,
    ) -> "PipelineContext":
        repo_root = (root or Path(__file__).resolve().parents[2]).resolve()
        pipeline_path = _resolve(repo_root, str(config_path), "config")
        machine_path = _resolve(repo_root, str(local_path), "local_config")
        pipeline = _load_yaml(pipeline_path)
        local = _load_yaml(machine_path)
        if pipeline.get("schema_version") != RUN_SCHEMA:
            raise PipelineError(f"schema_version must be {RUN_SCHEMA}")
        if local.get("schema_version") != LOCAL_SCHEMA:
            raise PipelineError(f"local schema_version must be {LOCAL_SCHEMA}")
        protocol = pipeline.get("protocol")
        if not isinstance(protocol, dict):
            raise PipelineError("protocol must be an object")
        if protocol.get("modality") not in {"visual_only", "multimodal"}:
            raise PipelineError("protocol.modality must be visual_only or multimodal")
        if protocol.get("sampling") != "fixed_30s":
            raise PipelineError("protocol.sampling must be fixed_30s")
        extraction = pipeline.get("extraction")
        expected_branches = set(STAGE_BY_BRANCH)
        if not isinstance(extraction, dict) or set(extraction) != expected_branches:
            raise PipelineError("extraction must define exactly: " + ", ".join(sorted(expected_branches)))
        if any(type(value) is not bool for value in extraction.values()):
            raise PipelineError("every extraction branch flag must be boolean")
        if not any(extraction.values()):
            raise PipelineError("at least one extraction branch must be enabled")
        selected_run_id = (run_id or "").strip()
        if not selected_run_id:
            if not allow_generate:
                raise PipelineError("--run-id is required for downstream or resumed stage execution")
            selected_run_id = generate_run_id()
        if not re.fullmatch(r"\d{6}_\d{4}", selected_run_id):
            raise PipelineError("run_id must use YYMMDD_HHmm, for example 260824_0938")
        artifact_root = _resolve(repo_root, pipeline.get("artifacts_root", "artifacts"), "artifacts_root")
        return cls(
            root=repo_root,
            pipeline_path=pipeline_path,
            local_path=machine_path,
            pipeline=pipeline,
            local=local,
            run_id=selected_run_id,
            started_at=_now(),
            run_root=artifact_root / selected_run_id,
        )

    @property
    def modality(self) -> str:
        return str(self.pipeline["protocol"]["modality"])

    @property
    def sampling(self) -> str:
        return "fixed_30s"

    @property
    def enabled_branches(self) -> tuple[str, ...]:
        configured = self.pipeline["extraction"]
        return tuple(branch for branch in STAGE_BY_BRANCH if configured[branch])

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        extraction = tuple(STAGE_BY_BRANCH[branch] for branch in self.enabled_branches)
        return ("prepare_data", *extraction, *FINAL_STAGES)

    @property
    def runtime_path(self) -> Path:
        return self.run_root / "runtime" / "pipeline.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "pipeline_manifest.json"

    @property
    def cohort_dir(self) -> Path:
        return self.run_root / "data" / "cohort"

    @property
    def data_dir(self) -> Path:
        return self.run_root / "data" / self.sampling

    @property
    def visual_manifest(self) -> Path:
        return self.data_dir / "visual_manifest.jsonl"

    @property
    def multimodal_ref_dir(self) -> Path:
        return self.data_dir / "multimodal_ref"

    @property
    def prepared_data_manifest(self) -> Path:
        return self.data_dir / "prepared_data_manifest.json"

    @property
    def catalog_path(self) -> Path:
        return self.cohort_dir / "catalog.jsonl"

    @property
    def context_root(self) -> Path:
        return self.run_root / "extraction" / "contexts" / self.modality

    def context_dir(self, branch: str) -> Path:
        return self.context_root / branch

    def context_manifest(self, branch: str) -> Path:
        return self.context_dir(branch) / "manifest.json"

    @property
    def representations_manifest(self) -> Path:
        return self.run_root / "validation" / "representations" / "manifest.json"

    @property
    def recommendations_manifest(self) -> Path:
        return self.run_root / "validation" / "recommendations" / "manifest.json"

    @property
    def diagnosis_path(self) -> Path:
        return self.run_root / "validation" / "diagnosis" / "diagnosis.json"

    def component_path(self, key: str) -> Path:
        components = self.pipeline.get("components")
        if not isinstance(components, dict) or key not in components:
            raise PipelineError(f"components.{key} is required")
        return _resolve(self.root, components[key], f"components.{key}")

    def local_value(self, section: str, key: str, *, required: bool = True) -> str:
        values = self.local.get(section)
        value = values.get(key) if isinstance(values, dict) else None
        text = str(value or "").strip()
        if required and not text:
            raise PipelineError(f"local config requires {section}.{key}")
        return text

    def local_path_value(self, section: str, key: str, *, required: bool = True) -> Path | None:
        value = self.local_value(section, key, required=required)
        return _resolve(self.root, value, f"{section}.{key}") if value else None


def runtime_document(context: PipelineContext) -> dict[str, Any]:
    data = {
        key: str(context.local_path_value("data", key))
        for key in ("videos_dir", "titles_csv", "tags_csv", "pairs_tsv")
    }
    models: dict[str, str | None] = {
        "qwen": str(context.local_path_value("models", "qwen")),
        "bge": str(context.local_path_value("models", "bge")),
        "asr": None,
    }
    asr = context.local_path_value("models", "asr", required=context.modality == "multimodal")
    if asr is not None:
        models["asr"] = str(asr)
    cloud = {
        "gcp_project_id": context.local_value("cloud", "gcp_project_id", required=False),
        "gemini_location": context.local_value("cloud", "gemini_location", required=False) or "global",
        "gemini_model": context.local_value("cloud", "gemini_model", required=False) or "gemini-3.6-flash",
        "gemini_thinking_level": context.local_value("cloud", "gemini_thinking_level", required=False) or "high",
    }
    components = {
        key: str(context.component_path(key))
        for key in (
            "processing_config",
            "ondevice_graph_config",
            "ondevice_desc_config",
            "gemini_graph_config",
            "validation_config",
        )
    }
    return {
        "schema_version": RUNTIME_SCHEMA,
        "run_id": context.run_id,
        "started_at": context.started_at,
        "modality": context.modality,
        "sampling": context.sampling,
        "enabled_branches": list(context.enabled_branches),
        "repo_root": str(context.root),
        "run_root": str(context.run_root),
        "paths": {
            "cohort_dir": str(context.cohort_dir),
            "data_dir": str(context.data_dir),
            "visual_manifest": str(context.visual_manifest),
            "multimodal_ref_dir": str(context.multimodal_ref_dir),
            "prepared_data_manifest": str(context.prepared_data_manifest),
            "context_root": str(context.context_root),
            "representations_manifest": str(context.representations_manifest),
            "recommendations_manifest": str(context.recommendations_manifest),
            "diagnosis": str(context.diagnosis_path),
        },
        "data": data,
        "models": models,
        "cloud": cloud,
        "components": components,
        "fingerprint": _fingerprint({
            "pipeline": context.pipeline,
            "local": context.local,
            "run_id": context.run_id,
        }),
    }


def preflight(context: PipelineContext) -> dict[str, Any]:
    runtime = runtime_document(context)
    checks: dict[str, bool] = {}
    for key in ("pairs_tsv", "titles_csv", "tags_csv"):
        checks[f"data.{key}"] = Path(runtime["data"][key]).is_file()
    checks["data.videos_dir"] = Path(runtime["data"]["videos_dir"]).is_dir()
    if any(branch.startswith("ondevice") for branch in context.enabled_branches):
        checks["models.qwen"] = Path(str(runtime["models"]["qwen"])).is_dir()
    if context.modality == "multimodal":
        checks["models.asr"] = bool(runtime["models"]["asr"]) and Path(str(runtime["models"]["asr"])).is_dir()
    checks["models.bge"] = Path(str(runtime["models"]["bge"])).is_dir()
    if any(branch.startswith("gemini") for branch in context.enabled_branches):
        checks["cloud.gcp_project_id"] = bool(runtime["cloud"]["gcp_project_id"])
    checks["ffmpeg"] = shutil.which("ffmpeg") is not None
    checks["ffprobe"] = shutil.which("ffprobe") is not None
    checks["python.torch"] = importlib.util.find_spec("torch") is not None
    checks["python.transformers"] = importlib.util.find_spec("transformers") is not None
    return {
        "schema_version": "pipeline-preflight/v1",
        "run_id": context.run_id,
        "ready": all(checks.values()),
        "checks": checks,
    }


def dependencies(context: PipelineContext, stage: str) -> tuple[str, ...]:
    if stage == "prepare_data":
        return ()
    if stage in EXTRACTION_STAGES:
        return ("prepare_data",)
    if stage == "embed_representations":
        return tuple(STAGE_BY_BRANCH[branch] for branch in context.enabled_branches)
    if stage == "run_recommendation":
        return ("embed_representations",)
    if stage == "run_diagnosis":
        return ("run_recommendation",)
    raise PipelineError(f"unsupported stage: {stage}")


def outputs_for_stage(context: PipelineContext, stage: str) -> tuple[Path, ...]:
    if stage == "prepare_data":
        return (context.prepared_data_manifest,)
    if stage in EXTRACTION_STAGES:
        return (context.context_manifest(BRANCH_BY_STAGE[stage]),)
    if stage == "embed_representations":
        return (context.representations_manifest,)
    if stage == "run_recommendation":
        return (context.recommendations_manifest,)
    if stage == "run_diagnosis":
        return (context.diagnosis_path,)
    raise PipelineError(f"unsupported stage: {stage}")


EXPECTED_OUTPUT_SCHEMA = {
    "prepare_data": "prepared-data/v1",
    "embed_representations": "representations/v1",
    "run_recommendation": "recommendations/v1",
    "run_diagnosis": "diagnosis/v1",
}


def output_is_complete(context: PipelineContext, stage: str) -> bool:
    expected_schema = "video-context-manifest/v1" if stage in EXTRACTION_STAGES else EXPECTED_OUTPUT_SCHEMA[stage]
    for path in outputs_for_stage(context, stage):
        if not path.is_file():
            return False
        try:
            document = _load_json(path)
        except PipelineError:
            return False
        if document.get("schema_version") != expected_schema:
            return False
        if document.get("run_id") != context.run_id:
            return False
        if document.get("modality") != context.modality:
            return False
        if stage == "run_diagnosis":
            if document.get("report_ready") is not True:
                return False
        elif document.get("complete") is not True:
            return False
    return True


def command_for_stage(context: PipelineContext, stage: str) -> Command:
    if stage not in context.enabled_stages:
        raise PipelineError(f"stage is disabled for this run: {stage}")
    env = os.environ.copy()
    source_root = str(context.root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        source_root + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else source_root
    )
    argv = (
        sys.executable,
        "-m",
        "viewing_context_pipeline",
        "_execute-stage",
        "--runtime",
        str(context.runtime_path),
        "--stage",
        stage,
    )
    return Command(argv=argv, cwd=context.root, env=env)


def _initial_manifest(context: PipelineContext) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": context.run_id,
        "started_at": context.started_at,
        "timezone": "Asia/Seoul",
        "modality": context.modality,
        "sampling": context.sampling,
        "enabled_branches": list(context.enabled_branches),
        "pipeline_config": {"path": str(context.pipeline_path), "sha256": _sha256(context.pipeline_path)},
        "local_config": {"path": str(context.local_path), "sha256": _sha256(context.local_path)},
        "git_head": _git_head(context.root),
        "stages": {},
    }


def _read_manifest(context: PipelineContext) -> dict[str, Any]:
    if not context.manifest_path.is_file():
        return _initial_manifest(context)
    value = _load_json(context.manifest_path)
    if value.get("schema_version") != MANIFEST_SCHEMA or value.get("run_id") != context.run_id:
        raise PipelineError(f"invalid pipeline manifest: {context.manifest_path}")
    if value.get("modality") != context.modality:
        raise PipelineError("run modality does not match the existing manifest")
    if value.get("sampling") != context.sampling:
        raise PipelineError("run sampling does not match the existing manifest")
    if value.get("enabled_branches") != list(context.enabled_branches):
        raise PipelineError("enabled extraction branches do not match the existing manifest")
    return value


def _record_stage(context: PipelineContext, stage: str, payload: dict[str, Any]) -> None:
    manifest = _read_manifest(context)
    manifest.setdefault("stages", {})[stage] = payload
    manifest["updated_at"] = _now()
    _atomic_json(context.manifest_path, manifest)


def initialize_run(context: PipelineContext) -> None:
    if context.run_root.exists() and any(context.run_root.iterdir()):
        raise PipelineError(f"run directory already exists and will not be overwritten: {context.run_root}")
    context.run_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.runtime_path, runtime_document(context))
    _atomic_json(context.manifest_path, _initial_manifest(context))


def _stage_state(context: PipelineContext, stage: str) -> dict[str, Any] | None:
    state = _read_manifest(context).get("stages", {}).get(stage)
    return state if isinstance(state, dict) else None


def _required_input_paths(context: PipelineContext, stage: str) -> tuple[Path, ...]:
    return tuple(path for dependency in dependencies(context, stage) for path in outputs_for_stage(context, dependency))


def execute_stage(
    context: PipelineContext,
    stage: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    command = command_for_stage(context, stage)
    if dry_run:
        return {"stage": stage, "dry_run": True, "command": command.document()}
    if stage == "prepare_data" and not context.run_root.exists():
        initialize_run(context)
    if not context.runtime_path.is_file() or not context.manifest_path.is_file():
        raise PipelineError(f"run is not initialized: {context.run_root}")
    missing = [str(path) for path in _required_input_paths(context, stage) if not path.is_file()]
    if missing:
        raise PipelineError(f"{stage} is missing prerequisite outputs: {', '.join(missing)}")
    current = _stage_state(context, stage)
    if current and current.get("status") == "complete" and output_is_complete(context, stage) and not force:
        if resume:
            print(f"[SKIP] {stage} already complete", flush=True)
            return {"stage": stage, "status": "skipped", "reason": "already complete"}
        raise PipelineError(f"stage is already complete; use --resume or --force: {stage}")
    started = perf_counter()
    print(f"[START] {stage} run_id={context.run_id} modality={context.modality}", flush=True)
    _record_stage(context, stage, {
        "status": "running",
        "started_at": _now(),
        "command": command.document(),
    })
    try:
        subprocess.run(command.argv, cwd=command.cwd, env=command.env, check=True)
        if not output_is_complete(context, stage):
            raise PipelineError(f"{stage} exited successfully but its canonical output is missing or incomplete")
    except (subprocess.CalledProcessError, PipelineError) as exc:
        exit_code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
        _record_stage(context, stage, {
            "status": "failed",
            "finished_at": _now(),
            "exit_code": exit_code,
            "command": command.document(),
            "error": str(exc),
        })
        print(f"[FAILED] {stage} elapsed={perf_counter() - started:.1f}s error={exc}", flush=True)
        raise PipelineError(f"stage failed: {stage}: {exc}") from exc
    result = {
        "status": "complete",
        "finished_at": _now(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "exit_code": 0,
        "command": command.document(),
        "outputs": [str(path) for path in outputs_for_stage(context, stage)],
    }
    _record_stage(context, stage, result)
    print(f"[DONE] {stage} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return {"stage": stage, **result}


def descendants(context: PipelineContext, requested: set[str]) -> set[str]:
    affected = set(requested)
    changed = True
    while changed:
        changed = False
        for stage in context.enabled_stages:
            if stage not in affected and any(dependency in affected for dependency in dependencies(context, stage)):
                affected.add(stage)
                changed = True
    return affected


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/pipelines/microlens_graph_vs_desc_pilot.yaml")
    parser.add_argument("--local-config", default="config/local.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def stage_cli_main(stage: str, argv: list[str] | None = None) -> int:
    parser = _common_parser(f"Run the {stage} ViewingContextPipeline stage.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = PipelineContext.load(
            args.config,
            args.local_config,
            run_id=args.run_id,
            allow_generate=stage == "prepare_data" and not args.resume,
        )
        if stage == "prepare_data" and context.run_root.exists() and not args.resume and not args.force and not args.dry_run:
            raise PipelineError(f"run directory already exists and will not be overwritten: {context.run_root}")
        result = execute_stage(context, stage, dry_run=args.dry_run, resume=args.resume, force=args.force)
    except PipelineError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def runner_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Run the ViewingContextPipeline v1 workflow.")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--force-stage", action="append", choices=STAGES, default=[])
    args = parser.parse_args(argv)
    try:
        allow_generate = args.stage in {"all", "prepare_data"} and not args.resume
        context = PipelineContext.load(
            args.config,
            args.local_config,
            run_id=args.run_id,
            allow_generate=allow_generate,
        )
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
            print(json.dumps({
                "run_id": context.run_id,
                "preflight": check,
                "stages": [command_for_stage(context, stage).document() for stage in context.enabled_stages],
            }, ensure_ascii=False, indent=2))
            return 0
        disabled_forces = sorted(set(args.force_stage) - set(context.enabled_stages))
        if disabled_forces:
            raise PipelineError("cannot force disabled stage(s): " + ", ".join(disabled_forces))
        if not context.run_root.exists():
            initialize_run(context)
        elif not args.resume and not args.force_stage:
            raise PipelineError(f"run directory already exists; use --resume or --force-stage: {context.run_root}")
        forced = descendants(context, set(args.force_stage))
        results = []
        for stage in context.enabled_stages:
            results.append(execute_stage(
                context,
                stage,
                resume=args.resume or bool(args.force_stage),
                force=stage in forced,
            ))
        print(json.dumps({"run_id": context.run_id, "preflight": check, "stages": results}, ensure_ascii=False, indent=2))
        return 0
    except PipelineError as exc:
        parser.error(str(exc))
    return 2
