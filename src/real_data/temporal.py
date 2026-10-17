"""
CDM Reports + Sentinel-2 -> Labelled Sequences
==============================================
Turns each usable CDM outbreak report into Sentinel-2 sequences labelled by
their position relative to the report date, written in the same raw CSV
format the pipeline consumes (sample_id, timestamp, label, lat, lon +
the six project bands). The existing preprocessing, models, evaluation and
risk maps therefore run on it unchanged.

Labelling (weak, time-relative)
-------------------------------
CDM reports are positives only, so each site is its own control. A sample
is the last `seq_len` QC-passing acquisitions up to an anchor acquisition;
with d = anchor date - report date:

    d <  -early_window            -> 0 "Healthy"        (before the outbreak)
    -early_window <= d < 0        -> 1 "Early Stress"   (pre-report window)
    0 <= d <= post_window         -> 2 "Active Disease" (reported outbreak)
    d >  post_window              -> excluded

Caveats, which the reports state next to every result:
- the report date is when someone noticed CDM, not the infection date, so
  the class boundaries are noisy by up to the latent period (~4-12 days)
  plus reporting delay;
- time relative to the report also tracks crop growth (and the calendar),
  so a classifier can score above chance from phenology alone.
  `seasonality_features` gives the anchor day-of-year and latitude for a
  seasonality-only baseline that measures this;
- consecutive samples of one event overlap, and one site can host several
  events, so splits must be grouped by `site_id`.
"""

import logging
import os

import numpy as np
import pandas as pd

from src.constants import BANDS
from src.real_data.feature_schema import S2_TO_PROJECT

logger = logging.getLogger(__name__)


def label_from_offset(days: int, early_window: int = 21, post_window: int = 45) -> int:
    """Class for an anchor `days` after the report (negative = before); -1 = excluded."""
    if days < -early_window:
        return 0
    if days < 0:
        return 1
    if days <= post_window:
        return 2
    return -1


def event_points(events: pd.DataFrame, days_before: int = 120,
                 days_after: int = 45) -> pd.DataFrame:
    """Sentinel-2 query table (point_id, lat, lon, start, end) for each event."""
    d0 = pd.to_datetime(events['observation_date'])
    return pd.DataFrame({
        'point_id': events['event_id'].to_numpy(),
        'lat': events['latitude'].to_numpy(),
        'lon': events['longitude'].to_numpy(),
        'start': (d0 - pd.Timedelta(days=days_before)).dt.strftime('%Y-%m-%d').to_numpy(),
        'end': (d0 + pd.Timedelta(days=days_after)).dt.strftime('%Y-%m-%d').to_numpy(),
    })


def build_sequences(
    events: pd.DataFrame,
    observations: pd.DataFrame,
    seq_len: int = 6,
    early_window: int = 21,
    post_window: int = 45,
    max_gap_days: int = 45,
    require_purity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build labelled sequences.

    Returns
    -------
    raw : one row per (sample, timestep) in the generator's raw CSV format,
        plus `date`, `site_id`, `event_id` and `days_to_report`.
    samples : one row per sample (sample_id, label, site_id, event_id,
        anchor_date, days_to_report, anchor_doy, lat, lon, span_days).
    """
    obs = observations.copy()
    obs['date'] = pd.to_datetime(obs['date'])
    if require_purity:
        obs = obs[obs['purity_pass'].astype(bool)]
    obs = obs.dropna(subset=list(S2_TO_PROJECT))
    obs_by_event = {pid: g.sort_values('date') for pid, g in obs.groupby('point_id')}

    raw_rows, sample_rows = [], []
    skipped_gap = 0
    for ev in events.itertuples(index=False):
        series = obs_by_event.get(ev.event_id)
        if series is None or len(series) < seq_len:
            continue
        d0 = pd.Timestamp(ev.observation_date)
        dates = series['date'].to_numpy()
        values = series[list(S2_TO_PROJECT)].to_numpy(dtype=float)
        for k in range(seq_len - 1, len(series)):
            window = slice(k - seq_len + 1, k + 1)
            gaps = np.diff(dates[window]).astype('timedelta64[D]').astype(int)
            if len(gaps) and gaps.max() > max_gap_days:
                skipped_gap += 1
                continue
            anchor = pd.Timestamp(dates[k])
            days = int((anchor - d0).days)
            label = label_from_offset(days, early_window, post_window)
            if label < 0:
                continue
            sample_id = f"{ev.event_id}_{anchor:%Y%m%d}"
            for t, (date, vals) in enumerate(zip(dates[window], values[window])):
                raw_rows.append({
                    'sample_id': sample_id, 'timestamp': t, 'label': label,
                    'lat': ev.latitude, 'lon': ev.longitude,
                    **dict(zip(BANDS, vals)),
                    'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                    'site_id': ev.site_id, 'event_id': ev.event_id,
                    'days_to_report': int((pd.Timestamp(date) - d0).days),
                })
            sample_rows.append({
                'sample_id': sample_id, 'label': label, 'site_id': ev.site_id,
                'event_id': ev.event_id, 'anchor_date': anchor.strftime('%Y-%m-%d'),
                'days_to_report': days, 'anchor_doy': anchor.dayofyear,
                'lat': ev.latitude, 'lon': ev.longitude,
                'span_days': int((pd.Timestamp(dates[k]) -
                                  pd.Timestamp(dates[k - seq_len + 1])).days),
            })

    raw = pd.DataFrame(raw_rows)
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        logger.warning("No sequences built (seq_len=%d); too few QC-passing acquisitions",
                       seq_len)
        return raw, samples
    logger.info(
        "Built %d sequences from %d events at %d sites (seq_len=%d, %d windows skipped for "
        "gaps > %d d) | label counts %s", len(samples), samples['event_id'].nunique(),
        samples['site_id'].nunique(), seq_len, skipped_gap, max_gap_days,
        samples['label'].value_counts().sort_index().to_dict())
    return raw, samples


def seasonality_features(samples: pd.DataFrame) -> np.ndarray:
    """(n, 3): sin/cos of anchor day-of-year and latitude -- no spectral information."""
    angle = 2 * np.pi * samples['anchor_doy'].to_numpy(dtype=float) / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle), samples['lat'].to_numpy(dtype=float)])


def observation_summary(events: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Per-event acquisition and QC counts (how much usable signal each report has)."""
    if observations.empty:
        return pd.DataFrame()
    obs = observations.copy()
    obs['purity_pass'] = obs['purity_pass'].astype(bool)
    per = obs.groupby('point_id').agg(
        acquisitions=('date', 'size'),
        qc_pass=('purity_pass', 'sum'),
        median_vegetation_fraction=('vegetation_fraction', 'median'),
        median_nir_cv=('nir_cv', 'median'),
    )
    reasons = (obs[~obs['purity_pass']].groupby(['point_id', 'qc_reason']).size()
               .unstack(fill_value=0).add_prefix('fail_'))
    per = per.join(reasons, how='left').fillna(0)
    meta = events.set_index('event_id')[['site_id', 'observation_date', 'state', 'habitat']]
    return meta.join(per, how='left').fillna({'acquisitions': 0, 'qc_pass': 0}).reset_index()


def write_dataset(raw: pd.DataFrame, samples: pd.DataFrame, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    paths = {'raw_csv': os.path.join(output_dir, 'cdm_s2_sequences.csv'),
             'samples_csv': os.path.join(output_dir, 'cdm_s2_samples.csv')}
    raw.to_csv(paths['raw_csv'], index=False)
    samples.to_csv(paths['samples_csv'], index=False)
    return paths
