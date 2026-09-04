from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from artifact_io import atomic_write_json, atomic_write_jsonl


CONFIG_PATH = Path("config/pipeline.yaml")
CONFIG_SCHEMA = "viewing-context-config/v3"


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
            or selected in {".", ".."}
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
            artifact_root / selected,
        )

    def initialize(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)

    def require_ready_cohort(self) -> dict[str, Any]:
        from validation.cohort import load_ready_cohort

        return load_ready_cohort(
            self.cohort_dir,
            run_id=self.run_id,
            settings=self.config["validation"]["cohort"],
            inputs={key: str(self.path("data", key)) for key in self.config["data"]},
        )

    @property
    def cohort_dir(self) -> Path:
        return self.run_root / "data" / "cohort"

    @property
    def evidence_dir(self) -> Path:
        return self.run_root / "data" / "fixed_30s"

    def graph_scene_dir(self, source: str) -> Path:
        return self.run_root / "extraction" / "graph" / source / "scenes"

    def graph_failure_dir(self, source: str) -> Path:
        return self.graph_scene_dir(source) / "failures"

    @property
    def description_scene_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "scenes"

    @property
    def description_failure_dir(self) -> Path:
        return self.description_scene_dir / "failures"

    def graph_summary_dir(self, source: str) -> Path:
        return self.run_root / "extraction" / "graph" / source / "summaries"

    def graph_summary_failure_dir(self, source: str) -> Path:
        return self.graph_summary_dir(source) / "failures"

    @property
    def description_summary_dir(self) -> Path:
        return self.run_root / "extraction" / "description" / "summaries"

    @property
    def description_summary_failure_dir(self) -> Path:
        return self.description_summary_dir / "failures"

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
        "cohort_sampling": "user_first_nested_stratified",
        "catalog_scope": "selected_user_sequence_union",
        "graph_extractors": ["qwen", "gemini"],
        "graph_summarizer": "qwen",
        "description_model": "qwen",
        "arms": ["metadata", "graph_qwen", "graph_gemini", "description"],
    }
    if set(protocol) != set(expected):
        raise ConfigError(f"protocol must contain exactly {sorted(expected)}")
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ConfigError(f"protocol.{key} must be {expected_value!r}")
    extraction = _require_mapping(value, "extraction")
    if set(extraction) != {
        "greedy_decoding",
        "visual_evidence",
        "graph_repetition_penalty",
        "description_repetition_penalty",
        "summary_repetition_penalty",
        "summary_sampling",
        "graph",
        "description",
    }:
        raise ConfigError(
            "extraction must contain greedy_decoding, visual_evidence, "
            "graph_repetition_penalty, description_repetition_penalty, "
            "summary_repetition_penalty, summary_sampling, graph, and description"
        )
    if not isinstance(extraction.get("greedy_decoding"), bool):
        raise ConfigError("extraction.greedy_decoding must be true or false")
    for stage in ("graph", "description", "summary"):
        key = f"{stage}_repetition_penalty"
        repetition_penalty = extraction.get(key)
        if (
            not isinstance(repetition_penalty, (int, float))
            or isinstance(repetition_penalty, bool)
            or not 1 <= float(repetition_penalty) <= 2
        ):
            raise ConfigError(f"extraction.{key} must be in [1, 2]")
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
        "scene_prompt",
        "summary_prompt",
        "scene_max_new_tokens",
        "summary_max_new_tokens",
    }
    graph_keys = generation_keys | {"gemini_concurrency"}
    description_keys = generation_keys
    for arm in ("graph", "description"):
        settings = _require_mapping(extraction, arm)
        expected = graph_keys if arm == "graph" else description_keys
        if set(settings) != expected:
            raise ConfigError(f"extraction.{arm} must contain exactly {sorted(expected)}")
        for key in ("scene_max_new_tokens", "summary_max_new_tokens"):
            setting = settings.get(key)
            if not isinstance(setting, int) or isinstance(setting, bool) or setting <= 0:
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
    if set(data) != {"videos_dir", "pairs_tsv", "titles_csv"}:
        raise ConfigError("data must contain exactly videos_dir, pairs_tsv, and titles_csv")
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
    max_output_tokens = gemini.get("max_output_tokens")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens <= 0
    ):
        raise ConfigError("models.gemini.max_output_tokens must be a positive integer")
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
    validation = _require_mapping(value, "validation")
    expected_validation_keys = {"cohort", "encoder", "model", "evaluation"}
    if set(validation) != expected_validation_keys:
        raise ConfigError(f"validation must contain exactly {sorted(expected_validation_keys)}")
    try:
        from pydantic import ValidationError

        from validation.config import ValidationConfig

        ValidationConfig.model_validate(
            {
                "schema_version": "validation-config/v3",
                "run_id": "config-validation",
                "dataset": data,
                "cohort": validation.get("cohort"),
                "encoder": {
                    **_require_mapping(validation, "encoder"),
                    "model_path": models.get("bge"),
                },
                "model": validation.get("model"),
                "evaluation": validation.get("evaluation"),
                "output_dir": value.get("artifacts_root"),
            }
        )
    except ValidationError as exc:
        raise ConfigError(f"invalid validation config: {exc}") from exc
