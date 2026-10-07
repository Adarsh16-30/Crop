"""
Geo-Spatial Risk Mapping
=========================
Converts model predictions with GPS coordinates into GeoPandas
GeoDataFrames with risk scores for spatial analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box


# Risk score mapping: disease class → base risk
RISK_MAP = {0: 0.0, 1: 0.5, 2: 1.0}


def build_geodataframe(predictions_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert a predictions DataFrame into a GeoDataFrame with risk scores.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns: lat, lon, predicted_label, confidence.
        Optional columns: sample_id, timestamp, spectral bands.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with geometry (Points), risk_score, and all input columns.
    """
    df = predictions_df.copy()

    # Create geometries in a vectorized way for better performance
    geometry = gpd.points_from_xy(df['lon'], df['lat'])

    # Compute risk score: base_risk(label) * confidence
    df['risk_score'] = df['predicted_label'].map(RISK_MAP) * df['confidence']

    # Clamp risk score to [0, 1]
    df['risk_score'] = df['risk_score'].clip(0.0, 1.0)

    # Risk category for easier filtering / visualization
    df['risk_category'] = pd.cut(
        df['risk_score'],
        bins=[-0.01, 0.2, 0.5, 0.7, 1.01],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def aggregate_zone_risk(
    gdf: gpd.GeoDataFrame,
    grid_size: float = 0.01,
    min_samples: int = 1
) -> gpd.GeoDataFrame:
    """
    Aggregate risk scores into a spatial grid.

    Divides the extent into cells of `grid_size` degrees and computes
    mean risk per cell.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with risk_score and geometry columns.
    grid_size : float
        Grid cell size in degrees (default: ~1.1 km at equator).
    min_samples : int
        Minimum number of samples in a cell to include it.

    Returns
    -------
    gpd.GeoDataFrame
        Grid of zones with mean_risk, max_risk, sample_count, and polygon geometry.
    """
    # Snap points to a regular grid and aggregate in one pass
    bounds = gdf[['lon', 'lat']].copy()
    bounds['x_bin'] = np.floor(bounds['lon'] / grid_size).astype(int)
    bounds['y_bin'] = np.floor(bounds['lat'] / grid_size).astype(int)
    grouped = gdf.copy()
    grouped['x_bin'] = bounds['x_bin']
    grouped['y_bin'] = bounds['y_bin']

    aggregated = grouped.groupby(['x_bin', 'y_bin'], sort=False).agg(
        mean_risk=('risk_score', 'mean'),
        max_risk=('risk_score', 'max'),
        min_risk=('risk_score', 'min'),
        std_risk=('risk_score', 'std'),
        sample_count=('risk_score', 'size'),
        dominant_label=('predicted_label', lambda s: int(s.mode().iloc[0]))
    ).reset_index()

    aggregated = aggregated[aggregated['sample_count'] >= min_samples].copy()

    if len(aggregated) > 0:
        aggregated['lon_min'] = aggregated['x_bin'] * grid_size
        aggregated['lat_min'] = aggregated['y_bin'] * grid_size
        aggregated['center_lon'] = aggregated['lon_min'] + grid_size / 2
        aggregated['center_lat'] = aggregated['lat_min'] + grid_size / 2
        aggregated['geometry'] = [
            box(x, y, x + grid_size, y + grid_size)
            for x, y in zip(aggregated['lon_min'], aggregated['lat_min'])
        ]

    if len(aggregated) == 0:
        print("WARNING: No grid zones created. Check data bounds and grid_size.")
        return gpd.GeoDataFrame(columns=['geometry', 'mean_risk'], crs='EPSG:4326')

    zone_gdf = gpd.GeoDataFrame(aggregated, geometry='geometry', crs='EPSG:4326')

    # Risk category for zones
    zone_gdf['zone_risk_category'] = pd.cut(
        zone_gdf['mean_risk'],
        bins=[-0.01, 0.2, 0.5, 0.7, 1.01],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )

    return zone_gdf[['geometry', 'center_lon', 'center_lat', 'mean_risk', 'max_risk', 'min_risk', 'std_risk', 'sample_count', 'dominant_label', 'zone_risk_category']]


def export_geojson(
    gdf: gpd.GeoDataFrame,
    output_path: str
) -> None:
    """
    Export GeoDataFrame to GeoJSON file.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to export.
    output_path : str
        Output file path.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Drop non-serializable columns
    export_df = gdf.copy()
    for col in export_df.columns:
        if col != 'geometry' and not pd.api.types.is_numeric_dtype(export_df[col]):
            export_df[col] = export_df[col].astype(str)

    export_df.to_file(output_path, driver='GeoJSON')
    print(f"GeoJSON exported: {output_path}")


def get_risk_summary(gdf: gpd.GeoDataFrame) -> dict:
    """
    Generate a summary of risk statistics.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with risk scores.

    Returns
    -------
    dict
        Summary statistics.
    """
    return {
        'total_samples': len(gdf),
        'mean_risk': float(gdf['risk_score'].mean()),
        'max_risk': float(gdf['risk_score'].max()),
        'std_risk': float(gdf['risk_score'].std()),
        'high_risk_count': int((gdf['risk_score'] > 0.7).sum()),
        'risk_distribution': gdf['risk_category'].value_counts().to_dict(),
        'label_distribution': gdf['predicted_label'].value_counts().to_dict(),
    }
