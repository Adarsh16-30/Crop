"""
Spectral Vegetation Index Computation
======================================
Computes standard vegetation stress indices from raw multispectral
band reflectance values.

Indices:
    - NDRE  (Normalized Difference Red Edge)
    - NDVI  (Normalized Difference Vegetation Index)
    - MSI   (Moisture Stress Index)
    - SWIR_ratio
    - REIP_approx  (Red Edge Inflection Point approximation)
    - CRI  (Chlorophyll Red-Edge Index)
"""

import pandas as pd
import numpy as np

# Small epsilon to prevent division by zero
EPS = 1e-8


def compute_ndre(df: pd.DataFrame) -> pd.Series:
    """
    Normalized Difference Red Edge Index.
    Sensitive to chlorophyll content in vegetation canopy.
    NDRE = (783nm - 705nm) / (783nm + 705nm)
    """
    return (df['red_edge_783'] - df['red_edge_705']) / \
           (df['red_edge_783'] + df['red_edge_705'] + EPS)


def compute_ndvi(df: pd.DataFrame) -> pd.Series:
    """
    Normalized Difference Vegetation Index.
    Classic indicator of vegetation health.
    NDVI = (NIR_842 - RedEdge_705) / (NIR_842 + RedEdge_705)
    """
    return (df['nir_842'] - df['red_edge_705']) / \
           (df['nir_842'] + df['red_edge_705'] + EPS)


def compute_msi(df: pd.DataFrame) -> pd.Series:
    """
    Moisture Stress Index.
    Higher values indicate water stress.
    MSI = SWIR_1610 / NIR_842
    """
    return df['swir_1610'] / (df['nir_842'] + EPS)


def compute_swir_ratio(df: pd.DataFrame) -> pd.Series:
    """
    SWIR Band Ratio.
    Changes in this ratio can indicate structural or water content shifts.
    SWIR_ratio = SWIR_2190 / SWIR_1610
    """
    return df['swir_2190'] / (df['swir_1610'] + EPS)


def compute_reip(df: pd.DataFrame) -> pd.Series:
    """
    Red Edge Inflection Point (linear approximation).
    Shifts in REIP indicate stress onset.
    REIP_approx = 705 + 35 * ((705 + 783) / 2 - 740) / (783 - 740 + eps)
    """
    mid = (df['red_edge_705'] + df['red_edge_783']) / 2.0
    return 705 + 35 * (mid - df['red_edge_740']) / \
           (df['red_edge_783'] - df['red_edge_740'] + EPS)


def compute_cri(df: pd.DataFrame) -> pd.Series:
    """
    Chlorophyll Red-Edge Index.
    CRI = (1 / RedEdge_705) - (1 / RedEdge_740)
    """
    return (1.0 / (df['red_edge_705'] + EPS)) - \
           (1.0 / (df['red_edge_740'] + EPS))


def compute_indices(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Compute all vegetation stress indices and add them as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw spectral band columns.
    inplace : bool
        If True, modify df in place. Otherwise, return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with added index columns: NDRE, NDVI, MSI, SWIR_ratio,
        REIP_approx, CRI.
    """
    if not inplace:
        df = df.copy()

    df['NDRE'] = compute_ndre(df)
    df['NDVI'] = compute_ndvi(df)
    df['MSI'] = compute_msi(df)
    df['SWIR_ratio'] = compute_swir_ratio(df)
    df['REIP_approx'] = compute_reip(df)
    df['CRI'] = compute_cri(df)

    return df


def get_feature_columns(include_raw: bool = True,
                        include_indices: bool = True) -> list[str]:
    """
    Get list of feature column names for model input.

    Parameters
    ----------
    include_raw : bool
        Include raw spectral band columns.
    include_indices : bool
        Include computed vegetation indices.

    Returns
    -------
    list[str]
        List of feature column names.
    """
    cols = []
    if include_raw:
        cols.extend([
            'red_edge_705', 'red_edge_740', 'red_edge_783',
            'nir_842', 'swir_1610', 'swir_2190'
        ])
    if include_indices:
        cols.extend(['NDRE', 'NDVI', 'MSI', 'SWIR_ratio', 'REIP_approx', 'CRI'])
    return cols
