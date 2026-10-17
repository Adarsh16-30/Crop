"""
EthCT2020 Spatial-Quality Check
===============================
EthCT2020 (Mendeley Data mfpvmk8cnm, v1): 2,793 georeferenced smallholder
field samples from Ethiopia's 2020/21 Meher season with crop types. The
polygons are 10 m-radius circles around the surveyed point (314 m^2, smaller
than one 20 m Sentinel-2 pixel), so they mark a location, not a field
boundary.

It has no disease labels. It is used to check the Sentinel-2 extraction
itself on many independent, confidently located fields:

1. Purity statistics: what share of in-season acquisitions pass each QC
   step (cloud / centre pixel / vegetation / NIR homogeneity), overall and
   per crop. Smallholder plots are often narrower than 60 m, so this is a
   hard test of the footprint purity rule.
2. Positive control: crop type should be predictable from in-season
   spectra. A Random Forest with spatially grouped CV (0.1 degree zones held
   out) is trained on per-field seasonal medians, once from all
   acquisitions and once from QC-passing acquisitions only. If extraction
   or masking were broken, crop type would be near chance.
"""

import logging
import os

import numpy as np
import pandas as pd

from src.constants import zone_key
from src.real_data.common import data_dir, download_file, http_session, reports_dir
from src.real_data.feature_schema import S2_TO_PROJECT, compute_canonical_features
from src.real_data.sentinel2 import PurityConfig, extract_points, make_backend

logger = logging.getLogger(__name__)

MENDELEY_DATASET = 'mfpvmk8cnm'
FILE_IDS = {
    'shp': 'afeded7b-b64f-4fbb-a37f-7ea0101e3900',
    'shx': 'cd5c981e-b5dd-406a-b3df-b96ff9acae80',
    'dbf': '1eaff03f-5ef3-447f-936c-293d8379d5eb',
    'prj': 'd36b44eb-c092-499c-879c-89d74d6520c0',
}
FILE_URL = 'https://data.mendeley.com/public-files/datasets/{ds}/files/{fid}/file_downloaded'
SEPARABILITY_INDICES = ['NDRE', 'CI_red_edge', 'NDRE_783_705', 'MSI', 'NDMI', 'SWIR_ratio']


def find_ethct_zip(extra: str | None = None) -> str | None:
    """The dataset ZIP as downloaded from Mendeley Data, if one is lying around."""
    import glob

    if extra:
        return extra if os.path.exists(extra) else None
    for folder in (data_dir('ethct'), '.', os.path.expanduser('~/Downloads')):
        hits = sorted(glob.glob(os.path.join(folder, '*EthCT2020*.zip')))
        if hits:
            return hits[0]
    return None


def extract_ethct_zip(zip_path: str, dest: str) -> str:
    """Extract the four EthCT2020.* shapefile parts (any folder inside the ZIP)."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if name.startswith('EthCT2020.') and name.rsplit('.', 1)[-1] in FILE_IDS:
                with zf.open(info) as src, open(os.path.join(dest, name), 'wb') as out:
                    out.write(src.read())
    shp = os.path.join(dest, 'EthCT2020.shp')
    if not os.path.exists(shp):
        raise FileNotFoundError(f"{zip_path} contains no EthCT2020.shp")
    logger.info("EthCT2020 extracted from %s", zip_path)
    return shp


def fetch_ethct(dest: str | None = None, zip_path: str | None = None) -> str:
    """
    Make sure the shapefile parts (~8 MB) are in `dest`; returns the .shp path.

    Order: an explicit `zip_path`; parts already in `dest`; an EthCT2020 ZIP
    found by find_ethct_zip(); a direct download. Mendeley Data sits behind a
    Cloudflare bot check that usually refuses scripted downloads, so the
    normal route is to download the dataset ZIP from
    https://data.mendeley.com/datasets/mfpvmk8cnm/1 in a browser.
    """
    import requests

    dest = dest or data_dir('ethct')
    if zip_path:
        return extract_ethct_zip(zip_path, dest)
    if all(os.path.exists(os.path.join(dest, f'EthCT2020.{e}')) for e in FILE_IDS):
        return os.path.join(dest, 'EthCT2020.shp')
    found = find_ethct_zip()
    if found:
        return extract_ethct_zip(found, dest)
    session = http_session(retries=1)
    for ext, fid in FILE_IDS.items():
        path = os.path.join(dest, f'EthCT2020.{ext}')
        if os.path.exists(path):
            continue
        try:
            download_file(FILE_URL.format(ds=MENDELEY_DATASET, fid=fid), path, session=session)
        except requests.HTTPError as exc:
            raise FileNotFoundError(
                f"Could not download {os.path.basename(path)} ({exc}). Download EthCT2020.shp, "
                f".shx, .dbf and .prj from https://data.mendeley.com/datasets/"
                f"{MENDELEY_DATASET}/1 and place them in {dest}/") from exc
    return os.path.join(dest, 'EthCT2020.shp')


def load_fields(shp_path: str) -> pd.DataFrame:
    """One row per field: id, crop class / group, source, lat/lon of the centre (WGS84)."""
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    centres = gdf.geometry.centroid.to_crs(4326)
    return pd.DataFrame({
        'field_id': gdf['id'].astype(int).to_numpy(),
        'crop_class': gdf['c_class'].to_numpy(),
        'crop_group': gdf['c_group'].to_numpy(),
        'source': gdf['sorc_nm'].to_numpy(),
        'quality': gdf['quality'].to_numpy(),
        'location_confidence': gdf['locatin'].to_numpy(),
        'lat': centres.y.to_numpy(), 'lon': centres.x.to_numpy(),
    })


def sample_fields(fields: pd.DataFrame, n_fields: int, min_per_class: int = 15,
                  max_classes: int = 6, seed: int = 42) -> pd.DataFrame:
    """
    Balanced sample over the most common crop classes (wheat alone is 74%
    of the dataset, so a plain random sample would be almost all wheat).
    """
    ok = fields[(fields['quality'] == 'very good') & (fields['location_confidence'] == 'confident')]
    counts = ok['crop_class'].value_counts()
    classes = counts[counts >= min_per_class].index[:max_classes]
    per_class = max(min_per_class, n_fields // max(len(classes), 1))
    parts = [g.sample(min(len(g), per_class), random_state=seed)
             for c, g in ok[ok['crop_class'].isin(classes)].groupby('crop_class')]
    return pd.concat(parts).reset_index(drop=True)


def purity_summary(obs: pd.DataFrame, fields: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QC outcome shares overall and per crop class."""
    o = obs.merge(fields[['field_id', 'crop_class']].assign(point_id=lambda d: 'eth_' +
                                                              d['field_id'].astype(str)),
                  on='point_id', how='left')
    o['outcome'] = o['qc_reason'].fillna('').replace('', 'pass')
    overall = o['outcome'].value_counts(normalize=True).rename('share').to_frame()
    overall['count'] = o['outcome'].value_counts()
    overall.loc['fields_with_any_pass', ['count']] = \
        o[o['outcome'] == 'pass']['point_id'].nunique()
    overall.loc['fields_total', ['count']] = o['point_id'].nunique()
    # Of the cloud-free acquisitions, how many fail purity -- the part of the
    # loss that a smaller footprint or better geolocation could recover.
    clear = o[~o['outcome'].isin(['no_valid_pixels', 'cloud_or_invalid'])]
    overall.loc['purity_fail_share_of_clear', 'share'] = (clear['outcome'] != 'pass').mean()
    by_class = (o.groupby('crop_class')['outcome'].value_counts(normalize=True)
                .unstack(fill_value=0))
    by_class['fields'] = o.groupby('crop_class')['point_id'].nunique()
    by_class['median_nir_cv'] = o.groupby('crop_class')['nir_cv'].median()
    return overall.reset_index(names='outcome'), by_class.reset_index()


def field_features(obs: pd.DataFrame, require_purity: bool) -> pd.DataFrame:
    """Per-field seasonal median of the six bands and the canonical indices."""
    o = obs.dropna(subset=list(S2_TO_PROJECT))
    if require_purity:
        o = o[o['purity_pass'].astype(bool)]
    else:
        o = o[o['n_valid'] > 0]
    if o.empty:
        return pd.DataFrame()
    canon = compute_canonical_features({b: o[b].to_numpy() for b in S2_TO_PROJECT}, 'sentinel2',
                                       indices=SEPARABILITY_INDICES)
    canon.index = o.index
    o = pd.concat([o[['point_id', 'date']], canon], axis=1)
    o['month'] = pd.to_datetime(o['date']).dt.month
    # Seasonal median plus per-month medians (crop calendars differ by month).
    season = o.drop(columns=['date', 'month']).groupby('point_id').median()
    monthly = (o.drop(columns=['date']).groupby(['point_id', 'month']).median()
               .unstack('month'))
    monthly.columns = [f'{c}_m{m}' for c, m in monthly.columns]
    feats = season.join(monthly)
    feats['n_acquisitions'] = o.groupby('point_id').size()
    return feats


def separability(obs: pd.DataFrame, fields: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Crop-type RF with spatial-zone grouped CV, with and without the purity filter."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    meta = fields.assign(point_id='eth_' + fields['field_id'].astype(str)).set_index('point_id')
    rows = []
    for require in (False, True):
        feats = field_features(obs, require_purity=require)
        if feats.empty:
            continue
        feats = feats.join(meta[['crop_class', 'lat', 'lon']], how='inner')
        counts = feats['crop_class'].value_counts()
        feats = feats[feats['crop_class'].isin(counts[counts >= 10].index)]
        if feats['crop_class'].nunique() < 2:
            continue
        y = feats['crop_class'].to_numpy()
        X = feats.drop(columns=['crop_class', 'lat', 'lon']).to_numpy(dtype=float)
        groups = np.array([hash(zone_key(a, b)) for a, b in zip(feats['lat'], feats['lon'])])
        n_splits = min(5, len(np.unique(groups)))
        model = make_pipeline(SimpleImputer(strategy='median'),
                              RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                     random_state=seed, n_jobs=-1))
        pred = np.empty(len(y), dtype=object)
        for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                           random_state=seed).split(X, y, groups):
            pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
        chance = 1.0 / feats['crop_class'].nunique()
        rows.append({'acquisitions': 'qc_pass_only' if require else 'all_cloud_free',
                     'fields': len(y), 'classes': feats['crop_class'].nunique(),
                     'balanced_accuracy': balanced_accuracy_score(y, pred),
                     'f1_macro': f1_score(y, pred, average='macro'),
                     'chance_balanced_accuracy': chance, 'cv': f'{n_splits}-fold spatial zone'})
    return pd.DataFrame(rows)


def run_ethct_check(n_fields: int = 300, start: str = '2020-09-01', end: str = '2020-11-30',
                    cfg: PurityConfig | None = None, backend_name: str = 'earthsearch',
                    workers: int = 8, point_workers: int = 4, seed: int = 42,
                    zip_path: str | None = None) -> dict:
    cfg = cfg or PurityConfig()
    fields = load_fields(fetch_ethct(zip_path=zip_path))
    sample = sample_fields(fields, n_fields, seed=seed)
    logger.info("EthCT2020: %d fields total, %d sampled | %s", len(fields), len(sample),
                sample['crop_class'].value_counts().to_dict())
    points = pd.DataFrame({'point_id': 'eth_' + sample['field_id'].astype(str),
                           'lat': sample['lat'], 'lon': sample['lon'],
                           'start': start, 'end': end})
    obs = extract_points(points, make_backend(backend_name, workers=workers), cfg,
                         cache_dir=os.path.join(data_dir('ethct'), 'cache'),
                         point_workers=point_workers)
    out = reports_dir('ethct')
    obs.to_csv(os.path.join(data_dir('ethct'), 'ethct_observations.csv'), index=False)
    overall, by_class = purity_summary(obs, sample)
    overall.to_csv(os.path.join(out, 'purity_summary.csv'), index=False)
    by_class.to_csv(os.path.join(out, 'purity_by_crop.csv'), index=False)
    sep = separability(obs, sample, seed=seed)
    sep.to_csv(os.path.join(out, 'crop_separability.csv'), index=False)
    logger.info("Purity outcomes:\n%s", overall.round(3).to_string(index=False))
    logger.info("Crop-type separability (positive control):\n%s",
                sep.round(3).to_string(index=False))
    return {'observations': obs, 'purity': overall, 'by_crop': by_class, 'separability': sep}
