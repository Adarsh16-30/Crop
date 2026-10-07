"""
Geo-Spatial Risk Mapping
=========================
Converts model predictions with GPS coordinates into GeoDataFrames with
probability-weighted risk scores, and aggregates them on a metric grid.
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from src.constants import NUM_CLASSES, RISK_BINS, RISK_CATEGORIES, expected_risk

logger = logging.getLogger(__name__)

PROB_COLUMNS = [f'prob_class_{i}' for i in range(NUM_CLASSES)]


def build_geodataframe(predictions_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert predictions into a GeoDataFrame with risk scores.

    Required columns: lat, lon, predicted_label, and either
    `prob_class_0..2` (risk is computed as the probability-weighted sum) or
    an existing `risk_score` column. Rows with missing coordinates are
    dropped with a warning -- coordinates are never invented.
    """
    df = predictions_df.copy()

    missing_gps = df['lat'].isna() | df['lon'].isna()
    if missing_gps.any():
        logger.warning("Dropping %d predictions with missing GPS coordinates",
                       int(missing_gps.sum()))
        df = df.loc[~missing_gps]

    if all(c in df for c in PROB_COLUMNS):
        df['risk_score'] = expected_risk(df[PROB_COLUMNS].to_numpy())
    elif 'risk_score' not in df:
        raise ValueError(f"Need {PROB_COLUMNS} or a risk_score column to compute risk")

    df['risk_score'] = df['risk_score'].clip(0.0, 1.0)
    df['risk_category'] = pd.cut(df['risk_score'], bins=RISK_BINS, labels=RISK_CATEGORIES)

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']),
                            crs='EPSG:4326')


def aggregate_zone_risk(
    gdf: gpd.GeoDataFrame,
    grid_size_m: float = 5000.0,
    min_samples: int = 3
) -> gpd.GeoDataFrame:
    """
    Aggregate risk into square grid cells of `grid_size_m` metres.

    The grid is built in a local UTM projection (so cells are equal-area,
    unlike degree-based cells) and returned in EPSG:4326 for web maps.
    Cells with fewer than `min_samples` points are dropped as too sparse to
    report a zone-level risk.
    """
    empty = gpd.GeoDataFrame(columns=['geometry', 'mean_risk'], geometry='geometry',
                             crs='EPSG:4326')
    if gdf.empty:
        return empty

    utm = gdf.estimate_utm_crs()
    projected = gdf.to_crs(utm)
    work = pd.DataFrame({
        'x_bin': np.floor(projected.geometry.x / grid_size_m).astype(int),
        'y_bin': np.floor(projected.geometry.y / grid_size_m).astype(int),
        'risk_score': gdf['risk_score'].to_numpy(),
        'predicted_label': gdf['predicted_label'].to_numpy(),
    })

    agg = work.groupby(['x_bin', 'y_bin'], sort=False).agg(
        mean_risk=('risk_score', 'mean'),
        max_risk=('risk_score', 'max'),
        min_risk=('risk_score', 'min'),
        std_risk=('risk_score', 'std'),
        sample_count=('risk_score', 'size'),
        dominant_label=('predicted_label', lambda s: int(s.mode().iloc[0])),
    ).reset_index()
    agg = agg[agg['sample_count'] >= min_samples].copy()
    if agg.empty:
        logger.warning("No grid zones with >= %d samples (grid %.0f m)", min_samples, grid_size_m)
        return empty

    cells = [box(x * grid_size_m, y * grid_size_m, (x + 1) * grid_size_m, (y + 1) * grid_size_m)
             for x, y in zip(agg['x_bin'], agg['y_bin'])]
    zone_gdf = gpd.GeoDataFrame(agg, geometry=cells, crs=utm).to_crs('EPSG:4326')
    centroids = gpd.GeoSeries(cells, crs=utm).centroid.to_crs('EPSG:4326')
    zone_gdf['center_lon'] = centroids.x.to_numpy()
    zone_gdf['center_lat'] = centroids.y.to_numpy()
    zone_gdf['std_risk'] = zone_gdf['std_risk'].fillna(0.0)
    zone_gdf['zone_risk_category'] = pd.cut(zone_gdf['mean_risk'], bins=RISK_BINS,
                                            labels=RISK_CATEGORIES)

    return zone_gdf[['geometry', 'center_lon', 'center_lat', 'mean_risk', 'max_risk',
                     'min_risk', 'std_risk', 'sample_count', 'dominant_label',
                     'zone_risk_category']]


def export_geojson(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Export a GeoDataFrame to GeoJSON (non-numeric columns stringified)."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    export_df = gdf.copy()
    for col in export_df.columns:
        if col != 'geometry' and not pd.api.types.is_numeric_dtype(export_df[col]):
            export_df[col] = export_df[col].astype(str)
    export_df.to_file(output_path, driver='GeoJSON')
    logger.info("GeoJSON exported: %s", output_path)


def get_risk_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Summary statistics of point-level risk."""
    return {
        'total_samples': len(gdf),
        'mean_risk': float(gdf['risk_score'].mean()),
        'max_risk': float(gdf['risk_score'].max()),
        'std_risk': float(gdf['risk_score'].std()),
        'high_risk_count': int((gdf['risk_score'] > 0.7).sum()),
        'risk_distribution': gdf['risk_category'].value_counts().to_dict(),
        'label_distribution': gdf['predicted_label'].value_counts().to_dict(),
    }
