"""
Acquisition Context Features
============================
Per-timestep context channels appended after the spectral features:

    doy_sin, doy_cos   day of year of the acquisition, on the unit circle
                       (31 Dec and 1 Jan are neighbours)
    lat, lon           location of the field, in degrees

Every real reading has a date and a location, so these are legitimate
inputs at inference time, not leaked labels. They let a model learn when
and where outbreaks happen (season, latitude, region), which on the CDM
data carries most of the predictable signal; the spectral channels then
only have to explain the remainder. The pipeline reports a context-only
baseline so the spectral contribution stays visible.

This is the single implementation used by preprocessing, the API, the MQTT
bridge and the history backfill.
"""

import numpy as np
import pandas as pd

CONTEXT_FEATURES = ['doy_sin', 'doy_cos', 'lat', 'lon']


def day_of_year(dates) -> np.ndarray:
    """Day of year (1-366) from ISO date strings, datetimes or day-of-year ints."""
    arr = np.asarray(dates)
    if np.issubdtype(arr.dtype, np.number):
        doy = arr.astype(np.float64)
    else:
        parsed = pd.to_datetime(pd.Series(arr.ravel()), format='ISO8601')
        doy = parsed.dt.dayofyear.to_numpy(dtype=np.float64).reshape(arr.shape)
    if np.any((doy < 1) | (doy > 366)) or not np.isfinite(doy).all():
        raise ValueError("Day of year must be in 1..366")
    return doy


def context_features(dates, lat, lon) -> np.ndarray:
    """
    Context channels for one or more sequences.

    Parameters
    ----------
    dates : (..., seq_len) acquisition dates (ISO strings or day-of-year).
    lat, lon : scalar or (...,) field location in degrees, constant over a
        sequence (broadcast across its timesteps).

    Returns
    -------
    (..., seq_len, 4) in CONTEXT_FEATURES order.
    """
    doy = day_of_year(dates)
    angle = 2 * np.pi * doy / 365.25
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError("lat must be in [-90, 90] and lon in [-180, 180]")
    lat = np.broadcast_to(lat[..., None], doy.shape)
    lon = np.broadcast_to(lon[..., None], doy.shape)
    return np.stack([np.sin(angle), np.cos(angle), lat, lon], axis=-1)


def add_context_columns(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Add CONTEXT_FEATURES columns to a long table with date, lat, lon per row."""
    df = df.copy()
    angle = 2 * np.pi * day_of_year(df[date_col].to_numpy()) / 365.25
    df['doy_sin'], df['doy_cos'] = np.sin(angle), np.cos(angle)
    return df
