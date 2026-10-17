"""
Integration Test -- the full run_pipeline on the real fixture data in a temp dir.
"""

import json
import os
import shutil

import pandas as pd
import pytest

import run_pipeline

from .conftest import SAMPLES_CSV, SEQUENCES_CSV


def _stage_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs('data/real/cdm_s2')
    shutil.copy(SEQUENCES_CSV, run_pipeline.RAW_CSV)
    shutil.copy(SAMPLES_CSV, run_pipeline.SAMPLES_CSV)


@pytest.mark.slow
def test_full_pipeline_with_tuning(tmp_path, monkeypatch):
    from src.training import tuning

    _stage_fixture(tmp_path, monkeypatch)
    # shrink the grids: this test checks the wiring, not the search
    monkeypatch.setattr(tuning, 'RF_GRID', {'max_depth': [3, 6], 'min_samples_leaf': [5],
                                            'max_features': ['sqrt']})
    monkeypatch.setattr(tuning, 'XGB_GRID', {'max_depth': [2], 'learning_rate': [0.1],
                                             'n_estimators': [50], 'min_child_weight': [1],
                                             'reg_lambda': [1.0]})
    monkeypatch.setattr(tuning, 'DL_GRID', {'hidden_dim': [8, 16], 'dropout': [0.5],
                                            'weight_decay': [1e-2], 'noise_std': [0.1]})
    rc = run_pipeline.main(['--epochs', '2', '--patience', '2', '--cv-folds', '2',
                            '--max-features', '4', '--skip-mqtt'])
    assert rc == 0

    for path in ('data/processed/scaler_params.json', 'data/processed/sample_meta.csv',
                 'models/checkpoints/lstm_best.pt', 'models/checkpoints/cnn_lstm_best.pt',
                 'models/checkpoints/cnn_lstm_attention_best.pt',
                 'models/checkpoints/rf_qubo_model.joblib', 'reports/model_comparison.csv',
                 'reports/model_significance.csv', 'reports/classical_cv.csv',
                 'reports/hyperparameter_tuning.csv', 'reports/ablation_cv.csv',
                 'reports/feature_selection_comparison.csv', 'reports/disease_risk_map.html'):
        assert os.path.exists(path), path

    comparison = pd.read_csv('reports/model_comparison.csv')
    assert {'Random Forest', 'XGBoost', 'LSTM', 'CNN-LSTM', 'CNN-LSTM-Attention',
            'Context-only RF (date, location)', 'Spectral-only RF'} <= set(comparison['Model'])
    assert comparison['Model'].str.startswith('RF (QUBO').any()

    tuned = pd.read_csv('reports/hyperparameter_tuning.csv')
    assert set(tuned['model']) == {'Random Forest', 'XGBoost', 'lstm', 'cnn_lstm',
                                   'cnn_lstm_attention'}
    assert tuned.groupby('model')['chosen'].sum().eq(1).all()
    assert {'cv_f1_mean', 'train_f1_mean', 'overfit_gap'} <= set(tuned.columns)

    cv = pd.read_csv('reports/classical_cv.csv')
    assert set(cv['cv_split']) == {'site', 'spatial_zone'}

    meta = pd.read_csv('data/processed/sample_meta.csv')
    assert (meta.groupby('site_id')['split'].nunique() == 1).all()   # no site leakage
    preds = pd.read_csv('data/processed/predictions.csv')
    assert set(preds['sample_id']) == set(meta['sample_id'])
    assert json.load(open('reports/qubo_comparison.json'))['n_features_selected'] == 4


@pytest.mark.slow
def test_pipeline_without_tuning(tmp_path, monkeypatch):
    _stage_fixture(tmp_path, monkeypatch)
    rc = run_pipeline.main(['--no-tune', '--epochs', '2', '--patience', '2', '--hidden-dim',
                            '8', '--models', 'cnn_lstm_attention', '--skip-quantum',
                            '--skip-geo', '--skip-mqtt'])
    assert rc == 0
    assert not os.path.exists('reports/hyperparameter_tuning.csv')
    from src.inference.predictor import load_checkpoint_dict
    cfg = load_checkpoint_dict('models/checkpoints/cnn_lstm_attention_best.pt')['config']
    assert cfg['model_type'] == 'cnn_lstm_attention' and cfg['hidden_dim'] == 8


def test_missing_dataset_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match='src.real_data temporal'):
        run_pipeline.main(['--no-tune', '--skip-mqtt'])
