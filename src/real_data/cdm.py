"""
Cucurbit Downy Mildew (CDM) Field Reports
=========================================
Source: the ipmPIPE CDM reporting platform (https://cdm.ipmpipe.org/), whose
"current reports" table is served by the public Bugwood occurrence API
(project 101). Each record is a confirmed CDM outbreak with an observation
date, state / county and, for recent years, GPS coordinates.

What the public API does and does not provide (checked October 2026):
- dates, state, county, county FIPS, habitat and GPS are returned;
- host, incidence, severity, gross area and verification method are NOT
  returned (the fields come back empty or are dropped), although the web
  page lists them. Pass an export that has them via `extra_csv` (matched on
  `report_id`) and they are carried through;
- only ~90 records (2021 onward) have GPS. Older records are county-level,
  which is far too coarse for 20 m Sentinel-2 pixels, so they are kept in
  the CSV but marked unusable for the satellite pipeline;
- every record is a positive report. There are no negative (healthy)
  reports, so controls must come from the time axis (observations before
  the report), see src/real_data/temporal.py.

Reporter names are personal data and are not stored.
"""

import logging
import os

import numpy as np
import pandas as pd

from src.real_data.common import data_dir, http_session

logger = logging.getLogger(__name__)

API_URL = 'https://api.bugwood.org/rest/api/occurrence.json'
PROJECT_ID = 101

# Requested from the API. Unsupported names are silently dropped by the API;
# the response's `columns` list says what actually came back.
API_FIELDS = [
    'objectid', 'Observationdate', 'dateentered', 'habitat', 'state', 'county',
    'countyfips', 'latitude', 'longitude', 'scientificname', 'commonname',
    'host', 'hostscientificname', 'HostName', 'Incidence', 'IncidenceUnit',
    'Severity', 'severityunit', 'Grossarea', 'grossareaunits', 'VerificationMethod',
]

# Normalised column -> accepted source names (case-insensitive), for both the
# API response and user-supplied exports.
COLUMN_ALIASES = {
    'report_id': ['objectid', 'report_id', 'id'],
    'observation_date': ['observationdate', 'observation_date', 'date', 'report_date'],
    'entered_date': ['dateentered', 'entered_date'],
    'state': ['state'],
    'county': ['county'],
    'county_fips': ['countyfips', 'county_fips', 'fips'],
    'habitat': ['habitat'],
    'host': ['hostname', 'host', 'hostcommonname', 'hostscientificname', 'crop'],
    'incidence': ['incidence'],
    'incidence_unit': ['incidenceunit', 'incidence_unit'],
    'severity': ['severity'],
    'severity_unit': ['severityunit', 'severity_unit'],
    'gross_area': ['grossarea', 'gross_area'],
    'gross_area_unit': ['grossareaunits', 'gross_area_unit'],
    'verification': ['verificationmethod', 'verification'],
    'latitude': ['latitude', 'lat'],
    'longitude': ['longitude', 'lon', 'lng', 'long'],
    'pathogen': ['scientificname', 'pathogen'],
}

NORMALISED_COLUMNS = list(COLUMN_ALIASES)

# Generous North-America bounding box; CDM is reported in the US and Canada.
LAT_RANGE = (14.0, 62.0)
LON_RANGE = (-170.0, -50.0)

# Two reports closer than this (degrees, ~22 m N-S) are treated as one site,
# i.e. one Sentinel-2 20 m pixel neighbourhood.
SITE_ROUND_DEG = 0.0002


def fetch_year(year: int, session=None, page_size: int = 1000) -> pd.DataFrame:
    """All CDM reports observed in `year`, as returned by the API (raw columns)."""
    session = session or http_session()
    params = {
        'observationdatestart': f'01/01/{year}',
        'observationdateend': f'31/12/{year}',
        'includeonly': ','.join(API_FIELDS),
        'project': PROJECT_ID,
        'draw': 1,
    }
    frames, start, total = [], 0, None
    while total is None or start < total:
        r = session.get(API_URL, params={**params, 'start': start, 'length': page_size},
                        timeout=60)
        r.raise_for_status()
        payload = r.json()
        total = int(payload.get('recordsFiltered', 0))
        rows = payload.get('data', [])
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=payload['columns']))
        start += len(rows)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("CDM %d: %d reports", year, len(raw))
    return raw


def normalise(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map raw API / export columns onto NORMALISED_COLUMNS and clean the types."""
    lower = {c.lower(): c for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for col, aliases in COLUMN_ALIASES.items():
        match = next((lower[a] for a in aliases if a in lower), None)
        out[col] = raw[match] if match is not None else np.nan

    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = (out[col].astype(object).where(out[col].notna(), '').astype(str)
                        .str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan}))
    out['report_id'] = out['report_id'].astype(str)
    for col in ('observation_date', 'entered_date'):
        out[col] = pd.to_datetime(out[col], errors='coerce').dt.normalize()
    for col in ('latitude', 'longitude', 'incidence', 'severity', 'gross_area'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['source'] = source
    return out


def merge_extra(reports: pd.DataFrame, extra_csv: str) -> pd.DataFrame:
    """
    Fill host / incidence / severity (and anything else missing) from a
    user-supplied export, matched on report_id. Values already present in
    `reports` win.
    """
    extra = normalise(pd.read_csv(extra_csv, dtype=str), source='extra_csv')
    extra = extra.drop_duplicates('report_id').set_index('report_id')
    merged = reports.set_index('report_id')
    known = merged.index.intersection(extra.index)
    for col in NORMALISED_COLUMNS:
        if col == 'report_id':
            continue
        filled = merged[col].astype(object)
        filled.loc[known] = filled.loc[known].where(filled.loc[known].notna(),
                                                    extra.loc[known, col].astype(object))
        numeric = col in ('latitude', 'longitude', 'incidence', 'severity', 'gross_area')
        merged[col] = pd.to_numeric(filled, errors='coerce') if numeric else filled
    new = extra.loc[extra.index.difference(merged.index)]
    if len(new):
        logger.info("extra_csv adds %d reports not returned by the API", len(new))
    return pd.concat([merged, new]).reset_index()


def site_id(lat: float, lon: float) -> str:
    """Stable id for a ~20 m neighbourhood."""
    return f"S{round(lat / SITE_ROUND_DEG):+d}_{round(lon / SITE_ROUND_DEG):+d}"


def select_usable(reports: pd.DataFrame, min_days_between: int = 30) -> pd.DataFrame:
    """
    Flag which reports can drive the satellite pipeline.

    Adds `usable` (bool), `reject_reason`, `site_id` and `event_id`. A report
    is usable when it has a date and valid GPS inside the North-America box,
    and is not a repeat report from the same site within `min_days_between`
    days of an earlier one (repeat visits describe the same outbreak).
    """
    df = reports.copy()
    reason = pd.Series('', index=df.index, dtype=object)

    def flag(mask, text):
        reason[mask & (reason == '')] = text

    flag(df['observation_date'].isna(), 'no_date')
    flag(df['latitude'].isna() | df['longitude'].isna(), 'no_gps')
    zero = (df['latitude'].abs() < 1e-6) | (df['longitude'].abs() < 1e-6)
    flag(zero, 'zero_gps')
    in_box = (df['latitude'].between(*LAT_RANGE) & df['longitude'].between(*LON_RANGE))
    flag(~in_box, 'gps_outside_region')

    ok = reason == ''
    df['site_id'] = np.where(ok, [site_id(a, b) if o else '' for a, b, o in
                                  zip(df['latitude'].fillna(0), df['longitude'].fillna(0), ok)],
                             '')

    # Collapse repeat reports from the same site into one event (earliest date).
    df = df.sort_values('observation_date', kind='stable')
    last_event_date: dict[str, pd.Timestamp] = {}
    for idx in df.index[ok.loc[df.index]]:
        site, date = df.at[idx, 'site_id'], df.at[idx, 'observation_date']
        prev = last_event_date.get(site)
        if prev is not None and (date - prev).days < min_days_between:
            reason[idx] = 'repeat_report_same_site'
        else:
            last_event_date[site] = date

    df['reject_reason'] = reason.loc[df.index].replace('', np.nan)
    df['usable'] = df['reject_reason'].isna()
    df['event_id'] = np.where(
        df['usable'],
        df['site_id'] + '_' + df['observation_date'].dt.strftime('%Y%m%d').fillna(''),
        '')
    return df.sort_index()


def build_cdm_table(years: list[int], extra_csv: str | None = None,
                    min_days_between: int = 30, output_dir: str | None = None) -> pd.DataFrame:
    """
    Fetch, normalise and quality-flag CDM reports for `years`.

    Writes `cdm_reports.csv` (every record, with `usable` / `reject_reason`)
    and `cdm_events.csv` (usable events only) to `output_dir`.
    """
    output_dir = output_dir or data_dir('cdm')
    session = http_session()
    frames = [normalise(fetch_year(y, session), source='bugwood_api') for y in years]
    frames = [f for f in frames if len(f)]
    reports = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=NORMALISED_COLUMNS + ['source'])
    reports = reports.drop_duplicates('report_id')
    if extra_csv:
        reports = merge_extra(reports, extra_csv)

    flagged = select_usable(reports, min_days_between=min_days_between)
    os.makedirs(output_dir, exist_ok=True)
    flagged.to_csv(os.path.join(output_dir, 'cdm_reports.csv'), index=False,
                   date_format='%Y-%m-%d')
    events = flagged[flagged['usable']].reset_index(drop=True)
    events.to_csv(os.path.join(output_dir, 'cdm_events.csv'), index=False,
                  date_format='%Y-%m-%d')

    counts = flagged['reject_reason'].fillna('usable').value_counts().to_dict()
    logger.info("CDM reports: %d total -> %d usable events at %d sites | %s",
                len(flagged), len(events), events['site_id'].nunique(), counts)
    return flagged


def load_events(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(data_dir('cdm'), 'cdm_events.csv')
    events = pd.read_csv(path, dtype={'report_id': str})
    events['observation_date'] = pd.to_datetime(events['observation_date'])
    return events
