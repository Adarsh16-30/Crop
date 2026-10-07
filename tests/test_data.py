"""
Unit Tests — Data Generation & Spectral Indices
"""

import numpy as np
import pandas as pd

from src.data.data_generator import BANDS, generate_dataset, generate_spectral_sequence
from src.features.spectral_indices import (
    REIP_MAX_NM,
    REIP_MIN_NM,
    compute_indices,
    get_feature_columns,
    raw_to_features,
)


class TestSpectralSequenceGeneration:

    def test_sequence_shape(self):
        df = generate_spectral_sequence(n_timesteps=30, label=1)
        assert len(df) == 30
        assert all(band in df.columns for band in BANDS)

    def test_sequence_has_label_and_timestamp(self):
        df = generate_spectral_sequence(n_timesteps=10, label=2)
        assert (df['label'] == 2).all()
        assert df['timestamp'].tolist() == list(range(10))

    def test_values_clamped_to_valid_range(self):
        df = generate_spectral_sequence(n_timesteps=50, label=2, noise=0.1)
        for band in BANDS:
            assert df[band].between(0.0, 1.0).all()

    def test_disease_progression_nir_degrades(self):
        healthy = generate_spectral_sequence(30, label=0, seed=42)
        diseased = generate_spectral_sequence(30, label=2, seed=42)
        assert diseased['nir_842'].mean() < healthy['nir_842'].mean()

    def test_seed_reproducibility(self):
        pd.testing.assert_frame_equal(generate_spectral_sequence(10, label=0, seed=42),
                                      generate_spectral_sequence(10, label=0, seed=42))
        assert not generate_spectral_sequence(10, label=0, seed=1)['nir_842'].equals(
            generate_spectral_sequence(10, label=0, seed=2)['nir_842'])


class TestDatasetGeneration:

    def test_dataset_shape(self):
        df = generate_dataset(n_samples=50, seq_len=10)
        assert len(df) == 500
        assert {'sample_id', 'lat', 'lon'} <= set(df.columns)

    def test_dataset_label_distribution(self):
        labels = generate_dataset(n_samples=1000, seq_len=5, seed=42) \
            .groupby('sample_id')['label'].first()
        assert labels.value_counts()[0] > 300

    def test_gps_coordinates_in_range(self):
        df = generate_dataset(n_samples=100, seq_len=5, center_lat=12.97, center_lon=77.59,
                              gps_spread=0.5)
        assert df['lat'].between(12.47, 13.47).all()
        assert df['lon'].between(77.09, 78.09).all()


class TestSpectralIndices:

    def test_all_indices_finite_on_full_dataset(self):
        """Regression: REIP used to range from -1.8e5 to +4.6e6."""
        df = compute_indices(generate_dataset(n_samples=300, seq_len=30, seed=42))
        values = df[get_feature_columns()].to_numpy()
        assert np.isfinite(values).all()

    def test_reip_is_a_red_edge_wavelength(self):
        df = compute_indices(generate_dataset(n_samples=300, seq_len=30, seed=42))
        assert df['REIP_approx'].between(REIP_MIN_NM, REIP_MAX_NM).all()

    def test_reip_handles_degenerate_bands(self):
        df = pd.DataFrame({b: [0.5] for b in BANDS})  # flat spectrum: old formula -> inf
        out = compute_indices(df)
        assert np.isfinite(out['REIP_approx']).all()

    def test_zero_reflectance_does_not_produce_inf(self):
        df = pd.DataFrame({b: [0.0] for b in BANDS})
        out = compute_indices(df)
        assert np.isfinite(out[get_feature_columns()].to_numpy()).all()

    def test_normalized_indices_range(self):
        df = compute_indices(generate_spectral_sequence(n_timesteps=10, label=0))
        assert df['NDRE'].between(-1, 1).all()
        assert df['NDVI'].between(-1, 1).all()
        assert (df['MSI'] >= 0).all()

    def test_raw_to_features_matches_dataframe_path(self):
        """The array path used at inference equals the DataFrame path used in training."""
        df = compute_indices(generate_spectral_sequence(n_timesteps=15, label=2, seed=3))
        from_array = raw_to_features(df[BANDS].to_numpy())
        np.testing.assert_allclose(from_array, df[get_feature_columns()].to_numpy(), rtol=1e-12)

    def test_feature_columns_helper(self):
        assert len(get_feature_columns()) == 12
        assert len(get_feature_columns(include_indices=False)) == 6

    def test_inplace_false_returns_copy(self):
        df = generate_spectral_sequence(10, label=0)
        out = compute_indices(df, inplace=False)
        assert 'NDRE' in out.columns and 'NDRE' not in df.columns
