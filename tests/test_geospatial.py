"""
Unit Tests — Geo-Spatial Risk Mapping
"""

import pytest
import numpy as np
import pandas as pd

from src.geospatial.risk_mapper import (
    build_geodataframe, aggregate_zone_risk, get_risk_summary
)


class TestGeoDataFrameConstruction:
    """Tests for GeoDataFrame creation."""

    def test_has_geometry_column(self):
        df = pd.DataFrame({
            'lat': [12.97, 12.98],
            'lon': [77.59, 77.60],
            'predicted_label': [0, 2],
            'confidence': [0.9, 0.85]
        })
        gdf = build_geodataframe(df)
        assert 'geometry' in gdf.columns

    def test_has_risk_score(self):
        df = pd.DataFrame({
            'lat': [12.97, 12.98],
            'lon': [77.59, 77.60],
            'predicted_label': [0, 2],
            'confidence': [0.9, 0.85]
        })
        gdf = build_geodataframe(df)
        assert 'risk_score' in gdf.columns

    def test_risk_score_in_valid_range(self):
        df = pd.DataFrame({
            'lat': np.random.uniform(12.0, 13.0, 100),
            'lon': np.random.uniform(77.0, 78.0, 100),
            'predicted_label': np.random.choice([0, 1, 2], 100),
            'confidence': np.random.uniform(0.5, 1.0, 100)
        })
        gdf = build_geodataframe(df)
        assert gdf['risk_score'].between(0, 1).all()

    def test_healthy_has_zero_risk(self):
        df = pd.DataFrame({
            'lat': [12.97],
            'lon': [77.59],
            'predicted_label': [0],
            'confidence': [0.95]
        })
        gdf = build_geodataframe(df)
        assert gdf['risk_score'].iloc[0] == 0.0

    def test_active_disease_has_high_risk(self):
        df = pd.DataFrame({
            'lat': [12.97],
            'lon': [77.59],
            'predicted_label': [2],
            'confidence': [0.9]
        })
        gdf = build_geodataframe(df)
        assert gdf['risk_score'].iloc[0] == pytest.approx(0.9)

    def test_risk_category_assigned(self):
        df = pd.DataFrame({
            'lat': [12.97, 12.98, 12.99, 13.00],
            'lon': [77.59, 77.60, 77.61, 77.62],
            'predicted_label': [0, 1, 2, 2],
            'confidence': [0.9, 0.8, 0.6, 0.95]
        })
        gdf = build_geodataframe(df)
        assert 'risk_category' in gdf.columns
        assert set(gdf['risk_category'].cat.categories) == {'Low', 'Moderate', 'High', 'Critical'}

    def test_crs_set_correctly(self):
        df = pd.DataFrame({
            'lat': [12.97],
            'lon': [77.59],
            'predicted_label': [0],
            'confidence': [0.9]
        })
        gdf = build_geodataframe(df)
        assert gdf.crs.to_epsg() == 4326


class TestZoneAggregation:
    """Tests for spatial grid aggregation."""

    def test_zone_creation(self):
        df = pd.DataFrame({
            'lat': np.random.uniform(12.9, 13.1, 50),
            'lon': np.random.uniform(77.5, 77.7, 50),
            'predicted_label': np.random.choice([0, 1, 2], 50),
            'confidence': np.random.uniform(0.5, 1.0, 50)
        })
        gdf = build_geodataframe(df)
        zones = aggregate_zone_risk(gdf, grid_size=0.05)
        assert len(zones) > 0
        assert 'mean_risk' in zones.columns
        assert 'sample_count' in zones.columns

    def test_zone_risk_in_range(self):
        df = pd.DataFrame({
            'lat': np.random.uniform(12.9, 13.1, 100),
            'lon': np.random.uniform(77.5, 77.7, 100),
            'predicted_label': np.random.choice([0, 1, 2], 100),
            'confidence': np.random.uniform(0.5, 1.0, 100)
        })
        gdf = build_geodataframe(df)
        zones = aggregate_zone_risk(gdf, grid_size=0.05)
        if len(zones) > 0:
            assert zones['mean_risk'].between(0, 1).all()


class TestRiskSummary:
    """Tests for risk summary statistics."""

    def test_summary_keys(self):
        df = pd.DataFrame({
            'lat': [12.97, 12.98, 12.99],
            'lon': [77.59, 77.60, 77.61],
            'predicted_label': [0, 1, 2],
            'confidence': [0.9, 0.8, 0.95]
        })
        gdf = build_geodataframe(df)
        summary = get_risk_summary(gdf)
        assert 'total_samples' in summary
        assert 'mean_risk' in summary
        assert 'high_risk_count' in summary
        assert summary['total_samples'] == 3
