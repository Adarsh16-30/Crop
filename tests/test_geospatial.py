"""
Unit Tests — Geo-Spatial Risk Mapping
"""

import numpy as np
import pandas as pd
import pytest

from src.constants import expected_risk
from src.geospatial.risk_mapper import aggregate_zone_risk, build_geodataframe, get_risk_summary


def _preds(n=100, seed=0, lat=(12.9, 13.1), lon=(77.5, 77.7)):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([1, 1, 1], n)
    return pd.DataFrame({
        'lat': rng.uniform(*lat, n), 'lon': rng.uniform(*lon, n),
        'predicted_label': probs.argmax(1),
        'prob_class_0': probs[:, 0], 'prob_class_1': probs[:, 1], 'prob_class_2': probs[:, 2],
    })


class TestRiskScore:

    def test_expected_risk_values(self):
        assert expected_risk([1, 0, 0]) == 0.0
        assert expected_risk([0, 0, 1]) == 1.0
        assert expected_risk([0, 1, 0]) == 0.5
        # An uncertain "Healthy" prediction still carries risk (old formula gave 0).
        assert expected_risk([0.4, 0.3, 0.3]) == pytest.approx(0.45)

    def test_risk_from_probabilities(self):
        gdf = build_geodataframe(_preds(20))
        expected = gdf[['prob_class_1', 'prob_class_2']].to_numpy() @ np.array([0.5, 1.0])
        np.testing.assert_allclose(gdf['risk_score'], expected)
        assert gdf['risk_score'].between(0, 1).all()

    def test_requires_probabilities_or_risk(self):
        with pytest.raises(ValueError):
            build_geodataframe(pd.DataFrame({'lat': [1.0], 'lon': [1.0],
                                             'predicted_label': [0], 'confidence': [0.9]}))

    def test_missing_gps_is_dropped_not_invented(self):
        df = _preds(5)
        df.loc[2, 'lat'] = np.nan
        gdf = build_geodataframe(df)
        assert len(gdf) == 4
        assert 2 not in gdf.index

    def test_categories_and_crs(self):
        gdf = build_geodataframe(_preds(10))
        assert set(gdf['risk_category'].cat.categories) == {'Low', 'Moderate', 'High', 'Critical'}
        assert gdf.crs.to_epsg() == 4326


class TestZoneAggregation:

    def test_zones_respect_min_samples(self):
        zones = aggregate_zone_risk(build_geodataframe(_preds(200)), grid_size_m=5000,
                                    min_samples=3)
        assert len(zones) > 0
        assert (zones['sample_count'] >= 3).all()
        assert zones['mean_risk'].between(0, 1).all()
        assert zones.crs.to_epsg() == 4326

    def test_grid_cells_are_metric(self):
        """Cells should be ~grid_size_m wide regardless of latitude."""
        zones = aggregate_zone_risk(build_geodataframe(_preds(300)), grid_size_m=5000,
                                    min_samples=1)
        widths = zones.to_crs(zones.estimate_utm_crs()).geometry.bounds
        np.testing.assert_allclose(widths['maxx'] - widths['minx'], 5000, rtol=1e-3)

    def test_sparse_data_returns_empty(self):
        zones = aggregate_zone_risk(build_geodataframe(_preds(3)), grid_size_m=100,
                                    min_samples=5)
        assert len(zones) == 0


def test_summary():
    summary = get_risk_summary(build_geodataframe(_preds(30)))
    assert summary['total_samples'] == 30
    assert 0 <= summary['mean_risk'] <= 1
