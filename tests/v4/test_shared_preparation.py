"""Preparation is shared across runs; extraction and validation remain isolated."""

import pytest

from pipeline_runtime import RunContext
from preparation.steps import prepare_cohort_step


def test_failed_shared_cohort_refresh_is_not_readable(ready_context):
    from pathlib import Path

    other = RunContext.load("other", root=ready_context.root)
    Path(other.config["data"]["videos_dir"]).joinpath("1.mp4").unlink()
    with pytest.raises(RuntimeError, match="unresolved assets"):
        prepare_cohort_step(other)
    with pytest.raises(RuntimeError, match="not ready"):
        ready_context.require_ready_cohort()
