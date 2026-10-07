"""
Spectral Vegetation Index Computation
======================================
Computes standard vegetation stress indices from raw multispectral
band reflectance values.

Indices:
    - NDRE  (Normalized Difference Red Edge)
    - NDVI  (Normalized Difference Vegetation Index, red-edge variant)
    - MSI   (Moisture Stress Index)
    - SWIR_ratio
    - REIP_approx  (Red Edge Inflection Point, linear interpolation)
    - CRI  (Chlorophyll Red-Edge Index)

All functions work on anything that supports numpy arithmetic: pandas
Series, numpy arrays or plain floats. This is the single implementation
used by training, the API and the MQTT bridge.
"""

import numpy as np
import pandas as pd

from src.constants import BANDS, INDICES

# Small epsilon to prevent division by zero
EPS = 1e-8

# Reflectance floor used in reciprocal / ratio indices so a band reading of
# exactly 0 (sensor dropout, clamping) cannot produce inf.
MIN_REFLECTANCE = 1e-3

# REIP is a wavelength; it must lie between the bands it interpolates.
REIP_MIN_NM = 705.0
REIP_MAX_NM = 783.0


def compute_ndre(b) -> np.ndarray:
    """NDRE = (783nm - 705nm) / (783nm + 705nm)"""
    return (b['red_edge_783'] - b['red_edge_705']) / \
           (b['red_edge_783'] + b['red_edge_705'] + EPS)


def compute_ndvi(b) -> np.ndarray:
    """NDVI (red-edge variant) = (NIR_842 - RedEdge_705) / (NIR_842 + RedEdge_705)"""
    return (b['nir_842'] - b['red_edge_705']) / \
           (b['nir_842'] + b['red_edge_705'] + EPS)


def compute_msi(b) -> np.ndarray:
    """Moisture Stress Index = SWIR_1610 / NIR_842 (higher = more water stress)."""
    return b['swir_1610'] / np.maximum(b['nir_842'], MIN_REFLECTANCE)


def compute_swir_ratio(b) -> np.ndarray:
    """SWIR_ratio = SWIR_2190 / SWIR_1610."""
    return b['swir_2190'] / np.maximum(b['swir_1610'], MIN_REFLECTANCE)


def compute_reip(b) -> np.ndarray:
    """
    Red Edge Inflection Point (linear interpolation, Guyot & Baret style).

    The inflection reflectance is the midpoint of the red-edge shoulders,
    R_ip = (R705 + R783) / 2, and REIP is the wavelength between 705nm and
    740nm where the reflectance curve crosses R_ip:

        REIP = 705 + 35 * (R_ip - R705) / (R740 - R705)

    The previous version divided by (R783 - R740), a gap that is routinely
    ~0, which produced values from -177,881 to +4,568,169 nm. Here the
    denominator is floored and the result is clipped to the physically
    meaningful 705-783nm range.
    """
    r_ip = (b['red_edge_705'] + b['red_edge_783']) / 2.0
    slope = np.maximum(b['red_edge_740'] - b['red_edge_705'], MIN_REFLECTANCE)
    reip = 705.0 + 35.0 * (r_ip - b['red_edge_705']) / slope
    return np.clip(reip, REIP_MIN_NM, REIP_MAX_NM)


def compute_cri(b) -> np.ndarray:
    """CRI = (1 / RedEdge_705) - (1 / RedEdge_740)."""
    return (1.0 / np.maximum(b['red_edge_705'], MIN_REFLECTANCE)) - \
           (1.0 / np.maximum(b['red_edge_740'], MIN_REFLECTANCE))


_INDEX_FUNCS = {
    'NDRE': compute_ndre,
    'NDVI': compute_ndvi,
    'MSI': compute_msi,
    'SWIR_ratio': compute_swir_ratio,
    'REIP_approx': compute_reip,
    'CRI': compute_cri,
}


def compute_index_values(bands) -> dict:
    """Compute every index from a mapping of band name -> value(s)."""
    return {name: _INDEX_FUNCS[name](bands) for name in INDICES}


def compute_indices(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Compute all vegetation stress indices and add them as new columns.

    Raises
    ------
    ValueError
        If any computed index is NaN or infinite.
    """
    if not inplace:
        df = df.copy()

    for name, values in compute_index_values(df).items():
        df[name] = values

    values = df[INDICES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c in INDICES if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
        raise ValueError(f"Non-finite vegetation index values in: {bad}")

    return df


def raw_to_features(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw band readings of shape (..., 6) into model features of
    shape (..., 12): the 6 bands followed by the 6 indices.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1] != len(BANDS):
        raise ValueError(f"Expected {len(BANDS)} raw bands, got {raw.shape[-1]}")
    bands = {name: raw[..., i] for i, name in enumerate(BANDS)}
    idx = compute_index_values(bands)
    return np.concatenate(
        [raw, np.stack([idx[name] for name in INDICES], axis=-1)], axis=-1
    )


def get_feature_columns(include_raw: bool = True,
                        include_indices: bool = True) -> list[str]:
    """Feature column names in model input order."""
    cols = []
    if include_raw:
        cols.extend(BANDS)
    if include_indices:
        cols.extend(INDICES)
    return cols
