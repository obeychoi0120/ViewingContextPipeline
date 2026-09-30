import math

import pytest
from pydantic import ValidationError

from conftest import config_data
from validation.config import ValidationConfig
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


def test_zero_delta_preserves_previous_stopping_rule():
    tracker = EarlyStopping(patience=2)
    scores = [0.03, 0.030001, 0.030002, 0.030002, 0.02]
    assert [tracker.update(i, s) for i, s in enumerate(scores, 1)] == [False] * 4 + [True]
    assert tracker.best_epoch == 3


@pytest.mark.parametrize('score', [math.nan, math.inf, -math.inf])
def test_nonfinite_validation_scores_rejected(score):
    with pytest.raises(ValueError, match='finite'):
        EarlyStopping(3).update(1, score)


@pytest.mark.parametrize('batch', [256, 512])
def test_batch_size_and_legacy_delta_defaults(tmp_path, batch):
    data = config_data(tmp_path)
    data['model']['batch_size'] = batch
    data['model'].pop('min_delta', None)
    config = ValidationConfig.model_validate(data)
    assert config.model.batch_size == batch
    assert config.model.min_delta == 0.0


@pytest.mark.parametrize('delta', [-0.001, math.nan, math.inf])
def test_invalid_delta_rejected(tmp_path, delta):
    data = config_data(tmp_path)
    data['model']['min_delta'] = delta
    with pytest.raises(ValidationError):
        ValidationConfig.model_validate(data)


def test_training_changes_invalidate_identity_without_changing_input(current_context):
    from validation.steps import validation_config
    from validation.selection import training_signature
    config = validation_config(current_context)
    cohort = {'events': [], 'manifest': {'selection_hash': 'fixed'}}
    original = training_signature(current_context, cohort, config)
    current_context.config['validation']['model']['min_delta'] *= 2
    assert training_signature(current_context, cohort, config) != original
