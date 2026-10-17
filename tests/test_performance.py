"""
Performance Tests -- sanity checks on the models produced by
`python run_pipeline.py` on the real CDM + Sentinel-2 data. Skipped (with a
reason) until the pipeline has run; they never pass vacuously.

The real labels are weak and the test split is a handful of held-out sites,
so these are floors, not targets: every model must beat a majority-class
predictor, and tuning must not have produced a model that memorises its
training folds.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

REPORTS = ('reports/model_comparison.csv', 'reports/hyperparameter_tuning.csv',
           'data/processed/y_test.npy')
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in REPORTS),
    reason="Pipeline artifacts not found. Run `python run_pipeline.py` first.",
)


@pytest.fixture(scope='module')
def comparison():
    return pd.read_csv('reports/model_comparison.csv')


def majority_macro_f1() -> float:
    y = np.load('data/processed/y_test.npy')
    majority = np.bincount(y).argmax()
    return f1_score(y, np.full_like(y, majority), average='macro', labels=[0, 1, 2],
                    zero_division=0)


def test_models_beat_majority_class(comparison):
    floor = majority_macro_f1()
    models = comparison[~comparison['Model'].str.startswith('Context-only')]
    weak = models[models['f1_macro'] <= floor]
    assert weak.empty, f"Models not better than majority class ({floor:.3f}):\n{weak}"


def test_attention_model_present(comparison):
    assert 'CNN-LSTM-Attention' in set(comparison['Model'])


def test_confidence_intervals_are_ordered(comparison):
    assert (comparison['accuracy_ci_low'] <= comparison['accuracy']).all()
    assert (comparison['accuracy'] <= comparison['accuracy_ci_high']).all()


def test_tuned_models_do_not_memorise_training_folds():
    """Chosen configurations: train-vs-CV macro-F1 gap stays below 0.2."""
    tuning = pd.read_csv('reports/hyperparameter_tuning.csv')
    chosen = tuning[tuning['chosen']]
    assert len(chosen) >= 4
    too_big = chosen[chosen['overfit_gap'] > 0.2]
    assert too_big.empty, f"Overfitting configurations chosen:\n{too_big}"
