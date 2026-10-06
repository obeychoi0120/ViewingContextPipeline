import math

import pytest

from validation.early_stopping import EarlyStopping


def test_small_improvements_do_not_extend_patience_but_select_actual_best():
    tracker = EarlyStopping(patience=3, min_delta=1e-4)
    scores = [0.03, 0.03002, 0.03004, 0.03006]
    assert [tracker.update(i, s) for i, s in enumerate(scores, 1)] == [False, False, False, True]
    assert tracker.best_epoch == 4
    assert tracker.progress_epoch == 1


def test_accumulated_improvement_resets_patience_and_ties_keep_first_best():
    tracker = EarlyStopping(patience=3, min_delta=1e-4)
    scores = [0.03, 0.03006, 0.03012, 0.03012, 0.0301, 0.03011]
    assert [tracker.update(i, s) for i, s in enumerate(scores, 1)] == [False] * 5 + [True]
    assert tracker.best_epoch == tracker.progress_epoch == 3


@pytest.mark.parametrize('score', [math.nan, math.inf, -math.inf])
def test_nonfinite_validation_scores_rejected(score):
    with pytest.raises(ValueError, match='finite'):
        EarlyStopping(3).update(1, score)
