"""
Integration Test — the full run_pipeline on a tiny dataset in a temp dir.
"""

import json
import os

import pandas as pd
import pytest

import run_pipeline


@pytest.mark.slow
def test_full_pipeline_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = run_pipeline.main([
        '--samples', '160', '--seq-len', '10', '--epochs', '2', '--hidden-dim', '16',
        '--patience', '2', '--max-features', '4', '--grid-size-m', '20000', '--skip-mqtt',
    ])
    assert rc == 0

    for path in ('data/processed/scaler_params.json', 'data/processed/sample_meta.csv',
                 'models/checkpoints/lstm_best.pt', 'models/checkpoints/cnn_lstm_best.pt',
                 'models/checkpoints/rf_qubo_model.joblib', 'reports/model_comparison.csv',
                 'reports/model_significance.csv', 'reports/classical_cv.csv',
                 'reports/feature_selection_comparison.csv', 'reports/disease_risk_map.html'):
        assert os.path.exists(path), path

    comparison = pd.read_csv('reports/model_comparison.csv')
    assert {'Random Forest', 'XGBoost', 'LSTM', 'CNN-LSTM'} <= set(comparison['Model'])
    assert comparison['Model'].str.startswith('RF (QUBO').any()
    assert (comparison['accuracy_ci_low'] <= comparison['accuracy']).all()

    cv = pd.read_csv('reports/classical_cv.csv')
    assert set(cv['cv_split']) == {'random', 'spatial_zone'}

    preds = pd.read_csv('data/processed/predictions.csv')
    meta = pd.read_csv('data/processed/sample_meta.csv')
    assert set(preds['split']) == {'train', 'val', 'test'}
    assert set(preds['sample_id']) == set(meta['sample_id'])  # real ids, every sample
    assert preds[['lat', 'lon']].notna().all().all()

    qubo = json.load(open('reports/qubo_comparison.json'))
    assert qubo['n_features_selected'] == 4
