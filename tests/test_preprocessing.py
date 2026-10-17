"""
Unit Tests — Preprocessing: split integrity, leakage, padding, aggregates
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    aggregate_feature_names,
    create_aggregated_features,
    create_sequences,
    run_preprocessing_pipeline,
)
from src.features.context import CONTEXT_FEATURES, add_context_columns
from src.features.spectral_indices import compute_indices, get_feature_columns

from .conftest import SEQ_LEN


class TestSplits:

    def test_splits_are_disjoint_and_complete(self, tiny_data):
        ids = [set(tiny_data[f'ids_{s}']) for s in ('train', 'val', 'test')]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        n_samples = pd.read_csv(tiny_data['raw_csv'])['sample_id'].nunique()
        assert sum(len(s) for s in ids) == n_samples

    def test_no_site_in_two_splits(self, tiny_data):
        """Overlapping sequences of one site must not leak across splits."""
        meta = pd.read_csv(f"{tiny_data['processed_dir']}/sample_meta.csv")
        assert (meta.groupby('site_id')['split'].nunique() == 1).all()

    def test_sequence_and_aggregate_splits_identical(self, tiny_data):
        for split in ('train', 'val', 'test'):
            np.testing.assert_array_equal(tiny_data[f'ids_{split}'],
                                          tiny_data[f'ids_{split}_agg'])
            np.testing.assert_array_equal(tiny_data[f'y_{split}'], tiny_data[f'y_{split}_agg'])

    def test_scaler_fitted_on_train_rows_only(self, tiny_data):
        df = add_context_columns(compute_indices(pd.read_csv(tiny_data['raw_csv'])))
        train = df[df.sample_id.isin(tiny_data['ids_train'])]
        cols = get_feature_columns() + CONTEXT_FEATURES
        assert tiny_data['scaler_params']['feature_names'] == cols
        np.testing.assert_allclose(tiny_data['scaler_params']['mean'],
                                   train[cols].mean().to_numpy(), rtol=1e-6)
        # ... and therefore NOT on all rows
        assert not np.allclose(tiny_data['scaler_params']['mean'],
                               df[cols].mean().to_numpy(), rtol=1e-9)

    def test_train_split_is_standardized(self, tiny_data):
        X = tiny_data['X_train'].reshape(-1, tiny_data['X_train'].shape[-1])
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-4)
        np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-2)


class TestSequences:

    def test_short_samples_are_dropped_not_padded(self):
        df = pd.DataFrame({
            'sample_id': [0] * 5 + [1] * 3,
            'timestamp': list(range(5)) + list(range(3)),
            'label': [0] * 5 + [1] * 3,
            'f': np.arange(8, dtype=float) + 1,
        })
        X, y, ids = create_sequences(df, seq_len=5, feature_cols=['f'])
        assert X.shape == (1, 5, 1)
        assert ids.tolist() == [0]
        assert (X != 0).all()

    def test_shapes(self, tiny_data):
        assert tiny_data['X_train'].shape[1:] == (SEQ_LEN, 12 + len(CONTEXT_FEATURES))

    def test_no_context_option(self, tmp_path):
        from .conftest import SEQUENCES_CSV
        data = run_preprocessing_pipeline(SEQUENCES_CSV, str(tmp_path), seq_len=SEQ_LEN,
                                          seed=7, group_col='site_id', context=False)
        assert data['X_train'].shape[1:] == (SEQ_LEN, 12)
        assert data['X_train_agg'].shape[1] == 48


class TestAggregates:

    def test_shape_and_names(self, tiny_data):
        names = aggregate_feature_names(get_feature_columns() + CONTEXT_FEATURES,
                                        n_context=len(CONTEXT_FEATURES))
        assert tiny_data['agg_feature_names'] == names
        assert tiny_data['X_train_agg'].shape[1] == len(names) == 48 + 4
        assert names[:2] == ['red_edge_705_mean', 'red_edge_740_mean']
        assert names[-4:] == CONTEXT_FEATURES

    def test_context_kept_as_last_value_not_summarised(self):
        X = np.zeros((1, 4, 3))
        X[0, :, 0] = [1, 2, 3, 4]          # spectral
        X[0, :, 1:] = [[5, 6], [5, 6], [5, 6], [7, 8]]   # context
        out = create_aggregated_features(X, n_context=2)[0]
        assert out.shape == (4 * 1 + 2,)
        assert out[-2:].tolist() == [7, 8]

    def test_slope_and_delta_on_linear_series(self):
        t = np.arange(10, dtype=float)
        X = np.stack([2 * t + 1], axis=-1)[None]  # (1, 10, 1)
        mean, std, slope, delta = create_aggregated_features(X)[0]
        assert mean == pytest.approx(10.0)
        assert slope == pytest.approx(2.0)
        assert delta == pytest.approx(2 * 7)  # mean(t=7..9) - mean(t=0..2) = 14
        assert std > 0
