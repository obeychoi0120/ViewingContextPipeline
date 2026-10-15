"""Scene counters remain disjoint when retries replace earlier outcomes."""

from io import StringIO
from types import SimpleNamespace

from extraction.progress import InferenceProgress


def test_progress_counts_outcomes_and_replacements_without_double_counting():
    displayed = []
    bar = SimpleNamespace(
        fp=StringIO(), n=0, set_postfix_str=lambda value, **_: displayed.append(value),
        reset=lambda **_: None,
    )
    progress = InferenceProgress(
        total=4, desc="graph", unit="scene", progress_factory=lambda **_: bar,
        clock=lambda: 10,
    )
    progress._started = 0
    progress.complete(task_id="a", warning=True)
    progress.complete(task_id="b", failed=True)
    progress.complete(task_id="c")
    progress._render(refresh=False)
    assert (progress.success, progress.warning, progress.failed, bar.n) == (1, 1, 1, 3)
    assert "success=1 warning=1 failed=1" in displayed[-1]
    assert "scene/s=0.30" in displayed[-1] and "ETA=00:03" in displayed[-1]
    assert "raw=" not in displayed[-1] and "structured=" not in displayed[-1]
    progress.complete(task_id="a")
    progress.complete(task_id="b", warning=True)
    progress.complete(task_id="c", failed=True)
    progress._render(refresh=False)
    assert (progress.success, progress.warning, progress.failed, bar.n) == (1, 1, 1, 3)
    progress.begin_pass(2, index=2, count=2)
    assert (progress.success, progress.warning, progress.failed, bar.n) == (0, 0, 0, 0)
