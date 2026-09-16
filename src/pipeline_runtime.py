from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from artifact_io import atomic_write_json, atomic_write_jsonl
from visual_sampling import validate_sampling


CONFIG_PATH = Path("config.yaml")
CONFIG_SCHEMA = "viewing-context-config/v5"


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
    atomic_write_json(path, value, durable=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows, durable=False)


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
        if (
            not selected
            or selected in {".", "..", "resized_keyframes", "source_assets"}
            or Path(selected).name != selected
            or "\\" in selected
        ):
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
            artifact_root / "runs" / selected,
        )

    def initialize(self) -> None:
        for directory in (self.cohort_dir, self.run_root / "extraction", self.run_root / "validation"):
            directory.mkdir(parents=True, exist_ok=True)

    def require_ready_cohort(self) -> dict[str, Any]:
        from validation.rolling_data import load_cohort
        return load_cohort(self.cohort_dir, self.run_id)

    @property
    def cohort_dir(self) -> Path:
        return self.run_root / "cohort"

    @property
    def evidence_dir(self) -> Path:
        return self.run_root.parent.parent

    @property
    def keyframes_dir(self) -> Path:
        return self.evidence_dir / "resized_keyframes"

    @property
    def source_assets_dir(self) -> Path:
        return self.evidence_dir / "source_assets"

    def extraction_dir(self, representation: str, model: str, phase: str) -> Path:
        if representation not in {"description", "graph"}:
            raise ValueError(f"invalid representation: {representation}")
        if model not in {"qwen", "gemini"} or phase not in {"scenes", "summaries"}:
            raise ValueError("invalid extraction model or phase")
        return self.run_root / "extraction" / representation / model / phase

    def graph_scene_dir(self, source: str) -> Path:
        return self.extraction_dir("graph", source, "scenes")

    def graph_failure_path(self, source: str) -> Path:
        return self.graph_scene_dir(source) / "failure.jsonl"

    def description_scene_dir(self, source: str) -> Path:
        return self.extraction_dir("description", source, "scenes")

    def description_failure_path(self, source: str) -> Path:
        return self.description_scene_dir(source) / "failure.jsonl"

    def graph_summary_dir(self, source: str) -> Path:
        return self.extraction_dir("graph", source, "summaries")

    def graph_summary_failure_path(self, source: str) -> Path:
        return self.graph_summary_dir(source) / "failure.jsonl"

    def description_summary_dir(self, source: str) -> Path:
        return self.extraction_dir("description", source, "summaries")

    def description_summary_failure_path(self, source: str) -> Path:
        return self.description_summary_dir(source) / "failure.jsonl"

    def prompt_path(self, schema: str | Path) -> Path:
        path = _resolve(self.root, str(schema), "--schema")
        if path.suffix.lower() != ".md" or not path.is_file():
            raise ValueError(f"--schema requires one existing Markdown prompt: {path}")
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"empty prompt: {path}")
        return path

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
    _validate_protocol(value)
    _validate_extraction(value)
    _validate_models(value)
    _validate_validation(value)


def _validate_protocol(value: dict[str, Any]) -> None:
    from arm_registry import active_arms
    protocol = _require_mapping(value, "protocol")
    expected = {
        "dataset": "microlens_100k", "modality": "visual_only", "sampling": "fixed_windows",
        "cohort_sampling": "full_rolling", "catalog_scope": "full_source_catalog",
        "graph_extractors": ["qwen", "gemini"], "graph_summarizer": "qwen",
        "description_extractors": ["qwen", "gemini"],
    }
    if set(protocol) != set(expected) | {"arms"}:
        raise ConfigError("invalid protocol keys")
    for key, setting in expected.items():
        if protocol.get(key) != setting:
            raise ConfigError(f"protocol.{key} must be {setting!r}")
    try:
        active_arms(value)
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigError(str(exc)) from exc


def _validate_extraction(value: dict[str, Any]) -> None:
    from extraction.qwen_config import qwen_settings

    extraction = _require_mapping(value, "extraction")
    try:
        qwen_settings(extraction.get("qwen"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if set(extraction) - {"qwen"} != {
        "greedy_decoding",
        "visual_evidence",
        "graph_repetition_penalty",
        "description_repetition_penalty",
        "summary_repetition_penalty",
        "summary_sampling",
        "graph",
        "description",
        "gemini",
    }:
        raise ConfigError(
            "extraction must contain greedy_decoding, visual_evidence, "
            "graph_repetition_penalty, description_repetition_penalty, "
            "summary_repetition_penalty, summary_sampling, graph, description, and gemini"
        )
    if not isinstance(extraction.get("greedy_decoding"), bool):
        raise ConfigError("extraction.greedy_decoding must be true or false")
    for stage in ("graph", "description", "summary"):
        key = f"{stage}_repetition_penalty"
        from extraction.recovery import penalty_schedule
        try:
            penalty_schedule(extraction.get(key))
        except ValueError as exc:
            raise ConfigError(f"extraction.{key}: {exc}") from exc
    visual_evidence = _require_mapping(extraction, "visual_evidence")
    if set(visual_evidence) != {"image_resolution", "scene_duration", "num_keyframes"}:
        raise ConfigError(
            "extraction.visual_evidence must contain image_resolution, scene_duration, "
            "and num_keyframes"
        )
    try:
        validate_sampling(visual_evidence["scene_duration"], visual_evidence["num_keyframes"])
    except ValueError as exc:
        raise ConfigError(f"extraction.visual_evidence: {exc}") from exc
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
    summary_sampling = _require_mapping(extraction, "summary_sampling")
    if set(summary_sampling) != {"temperature", "top_p", "top_k"}:
        raise ConfigError("extraction.summary_sampling must contain temperature, top_p, and top_k")
    sampling_temperature = summary_sampling.get("temperature")
    if (
        not isinstance(sampling_temperature, (int, float))
        or isinstance(sampling_temperature, bool)
        or not 0 < float(sampling_temperature) <= 2
    ):
        raise ConfigError("extraction.summary_sampling.temperature must be in (0, 2]")
    sampling_top_p = summary_sampling.get("top_p")
    if (
        not isinstance(sampling_top_p, (int, float))
        or isinstance(sampling_top_p, bool)
        or not 0 < float(sampling_top_p) <= 1
    ):
        raise ConfigError("extraction.summary_sampling.top_p must be in (0, 1]")
    sampling_top_k = summary_sampling.get("top_k")
    if (
        not isinstance(sampling_top_k, int)
        or isinstance(sampling_top_k, bool)
        or sampling_top_k <= 0
    ):
        raise ConfigError("extraction.summary_sampling.top_k must be a positive integer")
    generation_keys = {
        "scene_max_new_tokens",
        "summary_max_new_tokens",
    }
    for arm in ("graph", "description"):
        settings = _require_mapping(extraction, arm)
        expected = generation_keys
        if set(settings) != expected:
            raise ConfigError(f"extraction.{arm} must contain exactly {sorted(expected)}")
        for key in ("scene_max_new_tokens", "summary_max_new_tokens"):
            setting = settings.get(key)
            if not isinstance(setting, int) or isinstance(setting, bool) or setting <= 0:
                raise ConfigError(f"extraction.{arm}.{key} must be a positive integer")
    gemini = _require_mapping(extraction, "gemini")
    if set(gemini) != {"threads"}:
        raise ConfigError("extraction.gemini must contain exactly threads")
    if type(gemini["threads"]) is not int or gemini["threads"] <= 0:
        raise ConfigError("extraction.gemini.threads must be a positive integer")
    _require_mapping(value, "validation")


def _validate_models(value: dict[str, Any]) -> None:
    data = _require_mapping(value, "data")
    models = _require_mapping(value, "models")
    data_keys = {"videos_dir", "pairs_tsv", "titles_csv"}
    if value["schema_version"] == CONFIG_SCHEMA:
        data_keys.add("pairs_csv")
    if not data_keys <= set(data) or set(data) - data_keys - {"titles_supplement_csv"}:
        raise ConfigError(f"data must contain {sorted(data_keys)} and optionally titles_supplement_csv")
    if "titles_supplement_csv" in data and (not isinstance(data["titles_supplement_csv"], str) or not data["titles_supplement_csv"].strip()):
        raise ConfigError("data.titles_supplement_csv must be a non-empty path")
    if set(models) != {"qwen", "bge", "gemini"}:
        raise ConfigError("models must contain exactly qwen, bge, and gemini")
    gemini = _require_mapping(models, "gemini")
    gemini_keys = {
        "project_id",
        "location",
        "model_id",
        "temperature",
        "thinking_level",
        "media_resolution",
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
    if gemini.get("thinking_level") not in {"low", "medium", "high"}:
        raise ConfigError("models.gemini.thinking_level must be low, medium, or high")
    media_resolutions = {
        "MEDIA_RESOLUTION_UNSPECIFIED",
        "MEDIA_RESOLUTION_LOW",
        "MEDIA_RESOLUTION_MEDIUM",
        "MEDIA_RESOLUTION_HIGH",
    }
    if gemini.get("media_resolution") not in media_resolutions:
        raise ConfigError(
            f"models.gemini.media_resolution must be one of {sorted(media_resolutions)}"
        )


def _validate_validation(value: dict[str, Any]) -> None:
    validation = _require_mapping(value, "validation")
    expected_validation_keys = {"cohort", "encoder", "model", "evaluation"}
    if set(validation) != expected_validation_keys:
        raise ConfigError(f"validation must contain exactly {sorted(expected_validation_keys)}")
    try:
        from pydantic import ValidationError

        from validation.config import build_validation_config

        build_validation_config(
            run_id="config-validation", dataset=value["data"],
            settings={**validation, "encoder": _require_mapping(validation, "encoder")},
            model_path=value["models"].get("bge"), output_dir=value.get("artifacts_root"),
        )
    except ValidationError as exc:
        raise ConfigError(f"invalid validation config: {exc}") from exc
