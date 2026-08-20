from __future__ import annotations

from pathlib import Path

from src.common.manifest import CANONICAL_MANIFEST_PATH
from src.video_profile_generation.cli import main, parse_args


def test_default_config_path_is_anchored_to_script() -> None:
    args = parse_args([])

    assert args.config == (
        Path(__file__).resolve().parents[1]
        / "config"
        / "video_profile_generation.json"
    )
    assert args.manifest == CANONICAL_MANIFEST_PATH


def test_missing_config_returns_exit_code_2(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.json")]) == 2
