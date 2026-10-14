"""Selected-target pairwise comparisons use one Bonferroni family."""

import numpy as np
import pytest

from validation.leave_one_out_diagnosis import comparisons


@pytest.mark.parametrize('size,count', [(1, 0), (2, 1), (3, 3), (6, 15)])
def test_selected_pairs_and_direction(current_context, size, count):
    from arm_registry import registry

    names = list(registry(current_context.config))[:size]
    settings = type('Evaluation', (), {'familywise_alpha': 0.05})()
    values = np.arange(size, dtype=float) + 1
    draws = np.random.default_rng(42).normal(values, 0.1, (1000, size))
    result = comparisons(values, draws, settings, arms=names, config=current_context.config)
    reordered = comparisons(values[::-1], draws[:, ::-1], settings, arms=names[::-1], config=current_context.config)
    assert result == reordered and len(result) == count
    for key, row in result.items():
        left, right = key.split('-')
        assert names.index(left) > names.index(right)
        delta = draws[:, names.index(left)] - draws[:, names.index(right)]
        alpha = settings.familywise_alpha / count
        lo, hi = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
        assert row['ci_low'] == lo and row['ci_high'] == hi
        assert row['family_size'] == count
        assert row['confidence_level'] == pytest.approx(1 - alpha)
        assert 'primary' not in row and row['role'] == 'confirmatory'


