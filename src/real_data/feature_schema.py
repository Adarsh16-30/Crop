"""
Sensor-Independent Feature Schema
=================================
Each sensor names its bands differently and covers a different part of the
spectrum. This module maps every sensor onto canonical spectral regions and
defines indices over those regions, so features computed from the
MicaSense UAV camera, Sentinel-2 and the project's six-band model input share
one vocabulary.

An index is only computed for a sensor that has every region it needs
(no silent substitution of one band for another). `common_indices()`
returns the indices two or more sensors can all provide; those are the
only features that may be compared across sensors.

Note that a shared name is not a shared measurement: the MicaSense red edge
band is centred at 717 nm (FWHM 12 nm), the Sentinel-2 B05 at 705 nm (FWHM
15 nm). The `wavelength_nm` column of `schema_table()` makes those
differences visible.
"""

import numpy as np
import pandas as pd

EPS = 1e-8

# Canonical spectral regions, ordered by wavelength.
REGIONS = ['blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3',
           'nir', 'swir_1', 'swir_2']

# sensor -> region -> (band name in that sensor's data, centre wavelength nm)
SENSORS: dict[str, dict[str, tuple[str, float]]] = {
    'sentinel2': {
        'blue': ('B02', 492.0), 'green': ('B03', 560.0), 'red': ('B04', 665.0),
        'red_edge_1': ('B05', 704.0), 'red_edge_2': ('B06', 740.0),
        'red_edge_3': ('B07', 783.0), 'nir': ('B08', 833.0),
        'swir_1': ('B11', 1610.0), 'swir_2': ('B12', 2190.0),
    },
    'micasense_rededge_p': {
        'blue': ('blue', 475.0), 'green': ('green', 560.0), 'red': ('red', 668.0),
        'red_edge_1': ('red_edge', 717.0), 'nir': ('nir', 842.0),
    },
    # The band set the API, the live stream and the trained models use.
    'project': {
        'red_edge_1': ('red_edge_705', 705.0), 'red_edge_2': ('red_edge_740', 740.0),
        'red_edge_3': ('red_edge_783', 783.0), 'nir': ('nir_842', 842.0),
        'swir_1': ('swir_1610', 1610.0), 'swir_2': ('swir_2190', 2190.0),
    },
}

# Sentinel-2 asset -> project band. B05/B06/B07/B08/B11/B12 are exactly the
# six project bands, so Sentinel-2 observations feed the existing models.
S2_TO_PROJECT = {
    'B05': 'red_edge_705', 'B06': 'red_edge_740', 'B07': 'red_edge_783',
    'B08': 'nir_842', 'B11': 'swir_1610', 'B12': 'swir_2190',
}


def _nd(a, b):
    return (a - b) / (a + b + EPS)


# index -> (required regions, function of a region -> values mapping)
INDICES: dict[str, tuple[tuple[str, ...], callable]] = {
    'NDVI': (('nir', 'red'), lambda r: _nd(r['nir'], r['red'])),
    'GNDVI': (('nir', 'green'), lambda r: _nd(r['nir'], r['green'])),
    'NDRE': (('nir', 'red_edge_1'), lambda r: _nd(r['nir'], r['red_edge_1'])),
    'CI_red_edge': (('nir', 'red_edge_1'),
                    lambda r: r['nir'] / np.maximum(r['red_edge_1'], 1e-3) - 1.0),
    'CI_green': (('nir', 'green'),
                 lambda r: r['nir'] / np.maximum(r['green'], 1e-3) - 1.0),
    'NDRE_783_705': (('red_edge_3', 'red_edge_1'),
                     lambda r: _nd(r['red_edge_3'], r['red_edge_1'])),
    'MSI': (('swir_1', 'nir'), lambda r: r['swir_1'] / np.maximum(r['nir'], 1e-3)),
    'NDMI': (('nir', 'swir_1'), lambda r: _nd(r['nir'], r['swir_1'])),
    'SWIR_ratio': (('swir_2', 'swir_1'),
                   lambda r: r['swir_2'] / np.maximum(r['swir_1'], 1e-3)),
}


def sensor_regions(sensor: str) -> list[str]:
    if sensor not in SENSORS:
        raise KeyError(f"Unknown sensor {sensor!r}; known: {sorted(SENSORS)}")
    return [r for r in REGIONS if r in SENSORS[sensor]]


def available_indices(sensor: str) -> list[str]:
    regions = set(sensor_regions(sensor))
    return [name for name, (needs, _) in INDICES.items() if set(needs) <= regions]


def common_indices(*sensors: str) -> list[str]:
    """Indices every listed sensor can compute (the cross-sensor comparable set)."""
    sets = [set(available_indices(s)) for s in sensors]
    return [name for name in INDICES if all(name in s for s in sets)]


def to_regions(bands, sensor: str) -> dict:
    """
    Rename a sensor's bands to canonical regions.

    `bands` is a mapping (dict / DataFrame / structured record) keyed by the
    sensor's own band names; regions the data does not contain are skipped.
    """
    out = {}
    for region, (band, _) in SENSORS[sensor].items():
        if band in bands:
            out[region] = np.asarray(bands[band], dtype=float)
    return out


def compute_canonical_features(bands, sensor: str,
                               indices: list[str] | None = None) -> pd.DataFrame:
    """
    Canonical region reflectances plus every requested index the data supports.

    Returns a DataFrame with columns `refl_<region>` and one column per index.
    """
    regions = to_regions(bands, sensor)
    wanted = indices if indices is not None else list(INDICES)
    cols = {f'refl_{r}': np.atleast_1d(v) for r, v in regions.items()}
    for name in wanted:
        needs, func = INDICES[name]
        if all(r in regions for r in needs):
            cols[name] = np.atleast_1d(func(regions))
    return pd.DataFrame(cols)


def schema_table() -> pd.DataFrame:
    """Availability matrix: one row per (sensor, region) and per (sensor, index)."""
    rows = []
    for sensor, mapping in SENSORS.items():
        for region in REGIONS:
            band, wl = mapping.get(region, (None, np.nan))
            rows.append({'sensor': sensor, 'kind': 'region', 'name': region,
                         'sensor_band': band, 'wavelength_nm': wl,
                         'available': band is not None})
        avail = set(available_indices(sensor))
        for name, (needs, _) in INDICES.items():
            rows.append({'sensor': sensor, 'kind': 'index', 'name': name,
                         'sensor_band': '+'.join(needs), 'wavelength_nm': np.nan,
                         'available': name in avail})
    return pd.DataFrame(rows)
