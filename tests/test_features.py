"""
Vegetation indices on real Sentinel-2 readings (tests/fixtures).
"""

import numpy as np
import pandas as pd
import pytest

from src.constants import BANDS
from src.features.spectral_indices import (
    REIP_MAX_NM,
    REIP_MIN_NM,
    compute_indices,
    get_feature_columns,
    raw_to_features,
)

from .conftest import SEQUENCES_CSV


@pytest.fixture(scope='module')
def readings():
    return pd.read_csv(SEQUENCES_CSV)


class TestSpectralIndices:

    def test_all_indices_finite_on_real_readings(self, readings):
        values = compute_indices(readings.copy())[get_feature_columns()].to_numpy()
        assert np.isfinite(values).all()

    def test_reip_is_a_red_edge_wavelength(self, readings):
        df = compute_indices(readings.copy())
        assert df['REIP_approx'].between(REIP_MIN_NM, REIP_MAX_NM).all()

    def test_reip_handles_degenerate_bands(self):
        df = pd.DataFrame({b: [0.5] for b in BANDS})  # flat spectrum: old formula -> inf
        assert np.isfinite(compute_indices(df)['REIP_approx']).all()

    def test_zero_reflectance_does_not_produce_inf(self):
        df = pd.DataFrame({b: [0.0] for b in BANDS})
        out = compute_indices(df)
        assert np.isfinite(out[get_feature_columns()].to_numpy()).all()

    def test_normalized_indices_range(self, readings):
        df = compute_indices(readings.copy())
        assert df['NDRE'].between(-1, 1).all()
        assert df['NDVI'].between(-1, 1).all()
        assert (df['MSI'] >= 0).all()

    def test_raw_to_features_matches_dataframe_path(self, readings):
        """The array path used at inference equals the DataFrame path used in training."""
        df = compute_indices(readings.head(50).copy())
        from_array = raw_to_features(df[BANDS].to_numpy())
        np.testing.assert_allclose(from_array, df[get_feature_columns()].to_numpy(), rtol=1e-12)

    def test_feature_columns_helper(self):
        assert len(get_feature_columns()) == 12
        assert len(get_feature_columns(include_indices=False)) == 6

    def test_inplace_false_returns_copy(self, readings):
        df = readings.head(5).copy()
        out = compute_indices(df, inplace=False)
        assert 'NDRE' in out.columns and 'NDRE' not in df.columns


class TestRealFixture:

    def test_reflectances_are_physical(self, readings):
        assert readings[BANDS].gt(0).all().all() and readings[BANDS].le(1).all().all()

    def test_every_sample_has_one_label_and_site(self, readings):
        per = readings.groupby('sample_id').agg(labels=('label', 'nunique'),
                                                sites=('site_id', 'nunique'))
        assert (per['labels'] == 1).all() and (per['sites'] == 1).all()
        assert set(readings['label']) == {0, 1, 2}
