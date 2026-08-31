from __future__ import annotations

import validation.cli as cli_module


def test_documented_validation_step_matrix_is_stable() -> None:
    assert tuple(cli_module.STEP_HANDLERS) == (
        "prepare-cohort",
        "embed-representations",
        "run-recommendation",
        "run-diagnosis",
    )
