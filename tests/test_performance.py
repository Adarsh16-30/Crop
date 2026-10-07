"""
Performance Tests — quality thresholds for the models produced by
`python run_pipeline.py`. Skipped (with a reason) until the pipeline has run;
they never pass vacuously.
"""

import os

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score

ARTIFACTS = ('data/processed/X_test.npy', 'models/checkpoints/lstm_best.pt',
             'models/checkpoints/rf_model.joblib')
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in ARTIFACTS),
    reason="Pipeline artifacts not found. Run `python run_pipeline.py` first.",
)


@pytest.fixture(scope='module')
def test_split():
    return {k: np.load(f'data/processed/{k}.npy')
            for k in ('X_test', 'y_test', 'X_test_agg', 'y_test_agg')}


@pytest.fixture(scope='module')
def lstm_predictions(test_split):
    from src.inference.predictor import Predictor
    try:
        predictor = Predictor('models/checkpoints/lstm_best.pt')
    except ValueError as exc:  # checkpoint from an older training script
        pytest.skip(str(exc))
    return predictor.predict_proba_scaled(test_split['X_test']).argmax(axis=1)


def test_rf_accuracy_threshold(test_split):
    from src.models.classical_models import load_model
    rf = load_model('models/checkpoints/rf_model.joblib')
    if rf.n_features_in_ != test_split['X_test_agg'].shape[1]:
        pytest.skip("RF was trained on a different feature set; re-run the pipeline")
    acc = accuracy_score(test_split['y_test_agg'], rf.predict(test_split['X_test_agg']))
    assert acc >= 0.85, f"RF accuracy {acc:.4f} < 0.85"


def test_lstm_accuracy_threshold(test_split, lstm_predictions):
    acc = accuracy_score(test_split['y_test'], lstm_predictions)
    assert acc >= 0.85, f"LSTM accuracy {acc:.4f} < 0.85"


def test_early_stress_f1_threshold(test_split, lstm_predictions):
    f1 = f1_score(test_split['y_test'] == 1, lstm_predictions == 1)
    assert f1 >= 0.75, f"Early Stress F1 {f1:.4f} < 0.75"


def test_active_disease_precision_threshold(test_split, lstm_predictions):
    prec = precision_score(test_split['y_test'] == 2, lstm_predictions == 2, zero_division=0)
    assert prec >= 0.80, f"Active Disease precision {prec:.4f} < 0.80"
