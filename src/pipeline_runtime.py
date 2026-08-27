from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import yaml


CONFIG_PATH = Path("config/pipeline.yaml")
CONFIG_SCHEMA = "viewing-context-config/v1"


class ConfigError(RuntimeError):
    pass


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

    @classmethod
    def load(cls, run_id: str, *, root: Path | None = None) -> "RunContext":
        repo_root = (root or Path(__file__).resolve().parents[1]).resolve()
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
        )

    def initialize(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)

    @property
    def cohort_dir(self) -> Path:
        return self.run_root / "data" / "cohort"

    @property
    def evidence_dir(self) -> Path:
        return self.run_root / "data" / "fixed_30s"

    def graph_scene_dir(self, source: str) -> Path:
        return self.run_root / "extraction" / "graph" / source / "scenes"

    def graph_failure_dir(self, source: str) -> Path:
        return self.run_root / "extraction" / "graph" / source / "failures"

    @property
    def description_scene_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "scenes"

    @property
    def description_failure_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "failures"

    def graph_summary_dir(self, source: str) -> Path:
        return self.run_root / "extraction" / "graph" / source / "summaries"

    @property
    def description_summary_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "summaries"

    @property
    def representations_dir(self) -> Path:
        return self.run_root / "validation" / "representations"

    @property
    def recommendations_dir(self) -> Path:
        return self.run_root / "validation" / "recommendations"

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
        "graph_extractors": ["qwen", "gemini"],
        "graph_summarizer": "qwen",
        "description_model": "qwen",
        "arms": ["graph_qwen", "graph_gemini", "description"],
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ConfigError(f"protocol.{key} must be {expected_value!r}")
    extraction = _require_mapping(value, "extraction")
    if set(extraction) != {"visual_evidence", "graph", "description"}:
        raise ConfigError(
            "extraction must contain visual_evidence, graph, and description"
        )
    visual_evidence = _require_mapping(extraction, "visual_evidence")
    if set(visual_evidence) != {"image_resolution"}:
        raise ConfigError("extraction.visual_evidence must contain image_resolution")
    resolution = visual_evidence.get("image_resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in resolution
        )
    ):
        raise ConfigError(
            "extraction.visual_evidence.image_resolution must be two positive integers"
        )
    generation_keys = {
        "summary_prompt",
        "scene_max_new_tokens",
        "summary_max_new_tokens",
    }
    graph_keys = generation_keys | {"gemini_concurrency"}
    description_keys = generation_keys | {"scene_prompt"}
    for arm in ("graph", "description"):
        settings = _require_mapping(extraction, arm)
        expected = graph_keys if arm == "graph" else description_keys
        if set(settings) != expected:
            raise ConfigError(
                f"extraction.{arm} must contain exactly {sorted(expected)}"
            )
        for key in ("scene_max_new_tokens", "summary_max_new_tokens"):
            setting = settings.get(key)
            if (
                not isinstance(setting, int)
                or isinstance(setting, bool)
                or setting <= 0
            ):
                raise ConfigError(f"extraction.{arm}.{key} must be a positive integer")
        for key in ("summary_prompt", "scene_prompt"):
            if key in settings and (
                not isinstance(settings.get(key), str) or not settings[key].strip()
            ):
                raise ConfigError(f"extraction.{arm}.{key} must be a non-empty path")
    concurrency = extraction["graph"].get("gemini_concurrency")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ConfigError("extraction.graph.gemini_concurrency must be a positive integer")
    _require_mapping(value, "validation")
    data = _require_mapping(value, "data")
    models = _require_mapping(value, "models")
    if set(data) != {"videos_dir", "pairs_tsv"}:
        raise ConfigError("data must contain exactly videos_dir and pairs_tsv")
    if set(models) != {"qwen", "bge", "gemini"}:
        raise ConfigError("models must contain exactly qwen, bge, and gemini")
    gemini = _require_mapping(models, "gemini")
    gemini_keys = {
        "project_id",
        "location",
        "model_id",
        "temperature",
        "max_output_tokens",
        "thinking_level",
    }
    if set(gemini) != gemini_keys:
        raise ConfigError(f"models.gemini must contain exactly {sorted(gemini_keys)}")
    for key in ("project_id", "location", "model_id"):
        if not isinstance(gemini.get(key), str) or not gemini[key].strip():
            raise ConfigError(f"models.gemini.{key} must be a non-empty string")
    temperature = gemini.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not 0 <= temperature <= 2
    ):
        raise ConfigError("models.gemini.temperature must be a number from 0 to 2")
    max_output_tokens = gemini.get("max_output_tokens")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens <= 0
    ):
        raise ConfigError("models.gemini.max_output_tokens must be a positive integer")
    if gemini.get("thinking_level") not in {"low", "medium", "high"}:
        raise ConfigError(
            "models.gemini.thinking_level must be low, medium, or high"
        )
