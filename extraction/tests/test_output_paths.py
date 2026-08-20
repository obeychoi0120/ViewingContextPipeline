from pathlib import Path

from src.common.output_paths import custom_output_root


def test_custom_output_root_appends_custom_once() -> None:
    assert custom_output_root("output") == Path("output/custom")
    assert custom_output_root("output/custom") == Path("output/custom")


def test_custom_output_root_preserves_microlens_dataset_root() -> None:
    root = Path("output/microlens_100k_pilot_1k")
    assert custom_output_root(root) == root
