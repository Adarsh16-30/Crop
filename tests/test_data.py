"""
Unit Tests — Data Generation & Spectral Indices
"""

import pytest
import numpy as np
import pandas as pd

from src.data.data_generator import (
    generate_spectral_sequence, generate_dataset, BANDS, LABEL_NAMES
)
from src.features.spectral_indices import compute_indices, get_feature_columns


class TestSpectralSequenceGeneration:
    """Tests for single spectral sequence generation."""

    def test_sequence_shape(self):
        df = generate_spectral_sequence(n_timesteps=30, label=1)
        assert len(df) == 30
        for band in BANDS:
            assert band in df.columns

    def test_sequence_has_label_and_timestamp(self):
        df = generate_spectral_sequence(n_timesteps=10, label=2)
        assert 'label' in df.columns
        assert 'timestamp' in df.columns
        assert (df['label'] == 2).all()
        assert df['timestamp'].tolist() == list(range(10))

    def test_values_clamped_to_valid_range(self):
        df = generate_spectral_sequence(n_timesteps=50, label=2, noise=0.1)
        for band in BANDS:
            assert df[band].min() >= 0.0
            assert df[band].max() <= 1.0

    def test_disease_progression_nir_degrades(self):
        """NIR should be lower on average in diseased vs healthy crops."""
        healthy = generate_spectral_sequence(30, label=0, seed=42)
        diseased = generate_spectral_sequence(30, label=2, seed=42)
        assert diseased['nir_842'].mean() < healthy['nir_842'].mean()

    def test_different_seeds_produce_different_data(self):
        df1 = generate_spectral_sequence(10, label=0, seed=1)
        df2 = generate_spectral_sequence(10, label=0, seed=2)
        assert not df1['nir_842'].equals(df2['nir_842'])

    def test_same_seed_reproduces_data(self):
        df1 = generate_spectral_sequence(10, label=0, seed=42)
        df2 = generate_spectral_sequence(10, label=0, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestDatasetGeneration:
    """Tests for full dataset generation."""

    def test_dataset_shape(self):
        df = generate_dataset(n_samples=50, seq_len=10)
        assert len(df) == 50 * 10  # n_samples * seq_len
        assert 'sample_id' in df.columns
        assert 'lat' in df.columns
        assert 'lon' in df.columns

    def test_dataset_label_distribution(self):
        df = generate_dataset(n_samples=1000, seq_len=5, seed=42)
        labels = df.groupby('sample_id')['label'].first()
        # With p=[0.5, 0.3, 0.2], we expect roughly these proportions
        assert labels.value_counts()[0] > 300  # Healthy should be most common

    def test_gps_coordinates_in_range(self):
        df = generate_dataset(n_samples=100, seq_len=5,
                              center_lat=12.97, center_lon=77.59, gps_spread=0.5)
        assert df['lat'].min() >= 12.97 - 0.6
        assert df['lat'].max() <= 12.97 + 0.6
        assert df['lon'].min() >= 77.59 - 0.6
        assert df['lon'].max() <= 77.59 + 0.6


class TestSpectralIndices:
    """Tests for vegetation index computation."""

    def test_ndre_range(self):
        df = generate_spectral_sequence(n_timesteps=10, label=0)
        df = compute_indices(df)
        assert df['NDRE'].between(-1, 1).all()

    def test_ndvi_range(self):
        df = generate_spectral_sequence(n_timesteps=10, label=0)
        df = compute_indices(df)
        assert df['NDVI'].between(-1, 1).all()

    def test_msi_non_negative(self):
        df = generate_spectral_sequence(n_timesteps=10, label=0)
        df = compute_indices(df)
        assert (df['MSI'] >= 0).all()

    def test_all_indices_computed(self):
        df = generate_spectral_sequence(n_timesteps=10, label=0)
        df = compute_indices(df)
        expected = ['NDRE', 'NDVI', 'MSI', 'SWIR_ratio', 'REIP_approx', 'CRI']
        for idx in expected:
            assert idx in df.columns

    def test_no_nan_in_indices(self):
        df = generate_spectral_sequence(n_timesteps=10, label=0)
        df = compute_indices(df)
        for idx in ['NDRE', 'NDVI', 'MSI', 'SWIR_ratio']:
            assert not df[idx].isna().any()

    def test_feature_columns_helper(self):
        cols = get_feature_columns(include_raw=True, include_indices=True)
        assert len(cols) == 12  # 6 raw + 6 indices
        cols_raw = get_feature_columns(include_raw=True, include_indices=False)
        assert len(cols_raw) == 6

    def test_inplace_vs_copy(self):
        df = generate_spectral_sequence(10, label=0)
        df_copy = compute_indices(df, inplace=False)
        assert 'NDRE' in df_copy.columns
        # Original should be modified since we called inplace default=True first time
