from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import yaml


CONFIG_PATH = Path("config/pipeline.yaml")
CONFIG_SCHEMA = "viewing-context-config/v1"
RUNTIME_SCHEMA = "viewing-context-runtime/v1"


class ConfigError(RuntimeError):
    pass


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"model or data directory not found: {root}")
    rows = [
        {
            "path": file.relative_to(root).as_posix(),
            "size": file.stat().st_size,
            "mtime_ns": file.stat().st_mtime_ns,
        }
        for file in sorted(root.rglob("*"))
        if file.is_file()
    ]
    return fingerprint(rows)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object: {path}")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be an object: {path}")
    return value


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class RunContext:
    root: Path
    run_id: str
    config: dict[str, Any]
    run_root: Path
    config_fingerprint: str

    @classmethod
    def load(cls, run_id: str, *, root: Path | None = None) -> "RunContext":
        repo_root = (root or Path(__file__).resolve().parents[2]).resolve()
        selected = str(run_id or "").strip()
        if not selected or selected in {".", ".."} or Path(selected).name != selected or "\\" in selected:
            raise ConfigError("run_id must be a single non-empty directory name")
        config = _load_yaml(repo_root / CONFIG_PATH)
        _validate_config(config)
        artifact_root = _resolve(
            repo_root,
            config.get("artifacts_root", "artifacts"),
            "artifacts_root",
        )
        return cls(
            repo_root,
            selected,
            config,
            artifact_root / selected,
            fingerprint(config),
        )

    def initialize(self) -> None:
        runtime_path = self.runtime_path
        if runtime_path.is_file():
            runtime = read_json(runtime_path)
            if runtime.get("config_fingerprint") != self.config_fingerprint:
                raise ConfigError("current fixed config does not match the run snapshot")
            return
        if self.run_root.exists() and any(self.run_root.iterdir()):
            raise ConfigError(f"run directory has no v1 runtime snapshot: {self.run_root}")
        document = {
            "schema_version": RUNTIME_SCHEMA,
            "run_id": self.run_id,
            "config_path": str(self.root / CONFIG_PATH),
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
        }
        write_json(runtime_path, document)

    @property
    def runtime_path(self) -> Path:
        return self.run_root / "runtime" / "config_snapshot.json"

    @property
    def pipeline_manifest(self) -> Path:
        return self.run_root / "pipeline_manifest.json"

    def stage_manifest(self, stage: str) -> Path:
        return self.run_root / "manifests" / f"{stage}.json"

    @property
    def cohort_dir(self) -> Path:
        return self.run_root / "data" / "cohort"

    @property
    def evidence_dir(self) -> Path:
        return self.run_root / "data" / "fixed_30s"

    @property
    def visual_manifest(self) -> Path:
        return self.evidence_dir / "visual_manifest.jsonl"

    @property
    def graph_scene_dir(self) -> Path:
        return self.run_root / "extraction" / "graph" / "scenes"

    @property
    def description_scene_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "scenes"

    @property
    def graph_summary_dir(self) -> Path:
        return self.run_root / "extraction" / "graph" / "summaries"

    @property
    def description_summary_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "summaries"

    @property
    def representations_manifest(self) -> Path:
        return self.run_root / "validation" / "representations" / "manifest.json"

    @property
    def recommendations_manifest(self) -> Path:
        return self.run_root / "validation" / "recommendations" / "manifest.json"

    @property
    def diagnosis_path(self) -> Path:
        return self.run_root / "validation" / "diagnosis" / "diagnosis.json"

    def config_path(self, *keys: str) -> Path:
        value: Any = self.config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ConfigError("missing config value: " + ".".join(keys))
            value = value[key]
        return _resolve(self.root, value, ".".join(keys))

    def path(self, section: str, key: str) -> Path:
        values = _require_mapping(self.config, section)
        return _resolve(self.root, values.get(key), f"{section}.{key}")


def _validate_config(value: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol",
        "artifacts_root",
        "data",
        "models",
        "extraction",
        "validation",
    }
    if set(value) != expected_keys:
        raise ConfigError(f"pipeline config must contain exactly {sorted(expected_keys)}")
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA}")
    protocol = _require_mapping(value, "protocol")
    expected = {
        "dataset": "microlens_100k",
        "modality": "visual_only",
        "sampling": "fixed_30s",
        "backend": "qwen3_vl_2b",
        "arms": ["graph", "description"],
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ConfigError(f"protocol.{key} must be {expected_value!r}")
    extraction = _require_mapping(value, "extraction")
    for arm in ("graph", "description"):
        settings = _require_mapping(extraction, arm)
        if settings.get("do_sample") is not False:
            raise ConfigError(f"extraction.{arm}.do_sample must be false")
    _require_mapping(value, "validation")
    data = _require_mapping(value, "data")
    models = _require_mapping(value, "models")
    if set(data) != {"videos_dir", "titles_csv", "tags_csv", "pairs_tsv"}:
        raise ConfigError("data must contain videos_dir, titles_csv, tags_csv, pairs_tsv")
    if set(models) != {"qwen", "bge"}:
        raise ConfigError("models must contain exactly qwen and bge")
