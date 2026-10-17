"""
Hyperparameter tuning: grouped folds, the one-standard-error rule, and
classical / sequence-model searches on the real fixture data.
"""

import numpy as np
import pandas as pd
import pytest

from src.training import tuning


def test_expand_grid():
    combos = tuning.expand_grid({'a': [1, 2], 'b': ['x', 'y', 'z']})
    assert len(combos) == 6 and {'a': 2, 'b': 'z'} in combos


def test_grouped_folds_never_split_a_group():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 5)
    y = rng.integers(0, 3, len(groups))
    for tr, te in tuning.grouped_folds(y, groups, n_splits=4):
        assert not set(groups[tr]) & set(groups[te])


def test_one_standard_error_prefers_simpler_model():
    results = pd.DataFrame({'complexity': [1, 5, 10],
                            'cv_f1_mean': [0.58, 0.60, 0.61],
                            'cv_f1_se': [0.03, 0.03, 0.04]})
    assert tuning.one_standard_error_choice(results)['complexity'] == 1
    results.loc[0, 'cv_f1_mean'] = 0.50          # now outside 1 SE of the best
    assert tuning.one_standard_error_choice(results)['complexity'] == 5


def test_tune_classical_reports_gap_and_choice(tiny_data):
    from src.models.classical_models import make_random_forest

    X = np.concatenate([tiny_data['X_train_agg'], tiny_data['X_val_agg']])
    y = np.concatenate([tiny_data['y_train_agg'], tiny_data['y_val_agg']])
    ids = np.concatenate([tiny_data['ids_train_agg'], tiny_data['ids_val_agg']])
    groups = np.array([i.rsplit('_', 2)[0] for i in ids])     # site id prefix
    params, table = tuning.tune_classical(
        'Random Forest', make_random_forest,
        {'max_depth': [2, None], 'min_samples_leaf': [1, 10], 'max_features': ['sqrt']},
        X, y, groups, n_splits=3)
    assert set(params) == {'max_depth', 'min_samples_leaf', 'max_features'}
    assert table['chosen'].sum() == 1 and len(table) == 4
    # an unconstrained forest fits its training folds far better than it generalises
    deep = table[(table['max_depth'].isna()) & (table['min_samples_leaf'] == 1)].iloc[0]
    assert deep['train_f1_mean'] > 0.95 and deep['overfit_gap'] > 0.2


@pytest.mark.slow
def test_tune_sequence_model_smoke(tiny_data):
    X = np.concatenate([tiny_data['X_train'], tiny_data['X_val']])
    y = np.concatenate([tiny_data['y_train'], tiny_data['y_val']])
    ids = np.concatenate([tiny_data['ids_train'], tiny_data['ids_val']])
    groups = np.array([i.rsplit('_', 2)[0] for i in ids])
    params, table, info = tuning.tune_sequence_model(
        'cnn_lstm_attention', X, y, groups, tiny_data['scaler_params'],
        grid={'hidden_dim': [8], 'dropout': [0.3, 0.5], 'weight_decay': [1e-2],
              'noise_std': [0.1]}, n_splits=2, epochs=2, patience=2)
    assert params['hidden_dim'] == 8 and table['chosen'].sum() == 1
    assert (table['n_params'] > 0).all()
    assert info['oof_logits'].shape == (len(y), 3) and info['epochs'] >= 1
    assert np.abs(info['oof_logits']).sum(axis=1).all()     # every sample predicted once
