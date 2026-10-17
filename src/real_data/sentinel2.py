"""
Sentinel-2 Level-2A Point Time Series
=====================================
Extracts surface reflectance for the six project bands (B05, B06, B07 red
edge; B08 NIR; B11, B12 SWIR) around a point, for every acquisition in a
date range, with cloud masking and pixel-purity checks.

Analysis grid
-------------
Everything is summarised on the native 20 m grid of B05/B06/B07/B11/B12 and
the scene classification (SCL). The 10 m B08 is block-averaged onto that
grid, so it does not pretend to carry finer spatial detail than the
red-edge and SWIR bands. The footprint is an n x n block of 20 m pixels
(default 3 x 3 = 60 m) centred on the pixel that contains the point;
averaging over it also absorbs Sentinel-2's ~10 m geolocation error.

Quality control (per observation)
---------------------------------
- scene cloud cover below `max_scene_cloud` (pre-filter on metadata);
- per pixel: SCL class in VALID_SCL (vegetation or bare soil, i.e. not
  cloud, shadow, water, snow, saturated or no-data) and every band in
  (0, 1];
- purity: at least `min_valid_fraction` of the footprint valid, the centre
  pixel valid, at least `min_vegetation_fraction` of the valid pixels
  classed as vegetation, and a NIR coefficient of variation across the
  valid pixels below `max_nir_cv` (a mixed field / road / hedge footprint
  shows up as high NIR variation).
Observations are kept with their QC metrics; `purity_pass` says whether
they meet every threshold, and `qc_reason` says which one failed.

Backends
--------
- ``earthsearch`` (default): Element 84 Earth Search STAC,
  collection ``sentinel-2-c1-l2a``: the Copernicus L2A archive as Cloud-
  Optimised GeoTIFFs on AWS, readable without an account. Only the needed
  pixel windows are fetched. Scale and offset come from each asset's
  ``raster:bands`` metadata, so the processing-baseline 04.00 offset is
  handled.
- ``cdse``: Copernicus Data Space Ecosystem, Sentinel Hub Statistical API.
  Needs a free CDSE account and an OAuth client; set CDSE_CLIENT_ID and
  CDSE_CLIENT_SECRET. It returns window statistics directly; centre-pixel
  validity is not available from it, so that check is skipped.
"""

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.real_data.common import http_session
from src.real_data.feature_schema import S2_TO_PROJECT

logger = logging.getLogger(__name__)

S2_BANDS = list(S2_TO_PROJECT)                  # B05 B06 B07 B08 B11 B12
VALID_SCL = (4, 5)                              # vegetation, not-vegetated
VEGETATION_SCL = 4
NATIVE_GRID_M = 20.0

EARTH_SEARCH_URL = 'https://earth-search.aws.element84.com/v1'
EARTH_SEARCH_COLLECTION = 'sentinel-2-c1-l2a'
EARTH_SEARCH_ASSETS = {
    'B05': 'rededge1', 'B06': 'rededge2', 'B07': 'rededge3', 'B08': 'nir',
    'B11': 'swir16', 'B12': 'swir22', 'SCL': 'scl',
}

# GDAL settings for anonymous, range-request reads of public COGs.
_GDAL_ENV = {
    'AWS_NO_SIGN_REQUEST': 'YES',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.TIF',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2',
    'GDAL_HTTP_MULTIRANGE': 'YES',
    'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
    'VSI_CACHE': 'TRUE',
}


@dataclass
class PurityConfig:
    window_px: int = 3
    min_valid_fraction: float = 0.67
    require_center_valid: bool = True
    min_vegetation_fraction: float = 0.5
    max_nir_cv: float = 0.25
    max_scene_cloud: float = 80.0

    def __post_init__(self):
        if self.window_px < 1 or self.window_px % 2 == 0:
            raise ValueError("window_px must be a positive odd number")


# ---------------------------------------------------------------------------
# Pure QC on one footprint
# ---------------------------------------------------------------------------

def summarise_window(bands: dict[str, np.ndarray], scl: np.ndarray | None,
                     cfg: PurityConfig) -> dict:
    """
    Reduce an n x n footprint to one observation plus QC metrics.

    Parameters
    ----------
    bands : {S2 band -> (n, n) reflectance array}, NaN where no data.
    scl : (n, n) scene classification, or None if unavailable (then every
        pixel with finite reflectance counts as valid and vegetated).
    """
    stack = np.stack([np.asarray(bands[b], dtype=float) for b in S2_BANDS])
    n_pix = stack[0].size
    in_range = np.all(np.isfinite(stack) & (stack > 0) & (stack <= 1.0), axis=0)
    if scl is None:
        valid = in_range
        veg = valid
        scl_center = np.nan
    else:
        scl = np.asarray(scl)
        valid = in_range & np.isin(scl, VALID_SCL)
        veg = valid & (scl == VEGETATION_SCL)
        scl_center = float(scl[scl.shape[0] // 2, scl.shape[1] // 2])

    n_valid = int(valid.sum())
    center_valid = bool(valid[valid.shape[0] // 2, valid.shape[1] // 2])
    out = {
        'n_pixels': n_pix, 'n_valid': n_valid,
        'valid_fraction': n_valid / n_pix,
        'vegetation_fraction': float(veg.sum() / n_valid) if n_valid else 0.0,
        'center_valid': center_valid, 'center_scl': scl_center,
    }
    for i, band in enumerate(S2_BANDS):
        out[band] = float(stack[i][valid].mean()) if n_valid else np.nan
    nir = stack[S2_BANDS.index('B08')][valid]
    out['nir_cv'] = float(nir.std() / nir.mean()) if n_valid > 1 and nir.mean() > 0 else np.nan

    out['qc_reason'] = qc_reason(out, cfg, center_known=scl is not None)
    out['purity_pass'] = out['qc_reason'] == ''
    return out


def qc_reason(obs: dict, cfg: PurityConfig, center_known: bool = True) -> str:
    """First failed purity check, or '' if the observation passes."""
    if obs['n_valid'] == 0:
        return 'no_valid_pixels'
    if obs['valid_fraction'] < cfg.min_valid_fraction:
        return 'cloud_or_invalid'
    if cfg.require_center_valid and center_known and not obs['center_valid']:
        return 'center_invalid'
    if obs['vegetation_fraction'] < cfg.min_vegetation_fraction:
        return 'not_vegetated'
    if np.isfinite(obs.get('nir_cv', np.nan)) and obs['nir_cv'] > cfg.max_nir_cv:
        return 'heterogeneous'
    return ''


def snap_footprint(x: float, y: float, n: int, grid: float = NATIVE_GRID_M) -> tuple:
    """Bounds (left, bottom, right, top) of the n x n grid block centred on (x, y)'s cell."""
    cx = math.floor(x / grid) * grid
    cy = math.floor(y / grid) * grid
    half = (n // 2) * grid
    return cx - half, cy - half, cx + grid + half, cy + grid + half


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


# ---------------------------------------------------------------------------
# Earth Search (AWS COGs)
# ---------------------------------------------------------------------------

class EarthSearchBackend:
    name = 'earthsearch'

    def __init__(self, url: str = EARTH_SEARCH_URL,
                 collection: str = EARTH_SEARCH_COLLECTION, workers: int = 8):
        for key, val in _GDAL_ENV.items():
            os.environ.setdefault(key, val)
        self.url = url
        self.collection = collection
        self.workers = workers
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import warnings

            from pystac_client import Client
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self._client = Client.open(self.url)
        return self._client

    def search(self, lat: float, lon: float, start: str, end: str,
               max_cloud: float) -> list:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            search = self.client.search(
                collections=[self.collection],
                intersects={'type': 'Point', 'coordinates': [lon, lat]},
                datetime=f'{start}/{end}',
                query={'eo:cloud_cover': {'lt': max_cloud}},
                max_items=2000,
            )
            return list(search.items())

    @staticmethod
    def _scale_offset(asset) -> tuple[float, float, float | None]:
        info = (asset.extra_fields.get('raster:bands') or [{}])[0]
        return float(info.get('scale', 1e-4)), float(info.get('offset', 0.0)), info.get('nodata')

    def read_footprint(self, item, lat: float, lon: float, n: int) -> dict | None:
        import rasterio
        from pyproj import Transformer
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        scl_href = item.assets[EARTH_SEARCH_ASSETS['SCL']].href
        with rasterio.open(scl_href) as ds:
            x, y = Transformer.from_crs(4326, ds.crs, always_xy=True).transform(lon, lat)
            bounds = snap_footprint(x, y, n, grid=abs(ds.transform.a))
            win = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
            if (win.col_off < 0 or win.row_off < 0 or win.col_off + n > ds.width
                    or win.row_off + n > ds.height):
                return None
            scl = ds.read(1, window=win)
        if not np.isin(scl, VALID_SCL).any():
            return {'scl': scl, 'bands': {b: np.full((n, n), np.nan) for b in S2_BANDS}}

        bands = {}
        for band in S2_BANDS:
            asset = item.assets[EARTH_SEARCH_ASSETS[band]]
            scale, offset, nodata = self._scale_offset(asset)
            with rasterio.open(asset.href) as ds:
                win = from_bounds(*bounds, transform=ds.transform)
                dn = ds.read(1, window=win, out_shape=(n, n),
                             resampling=Resampling.average).astype(float)
            refl = dn * scale + offset
            if nodata is not None:
                refl[dn == nodata] = np.nan
            bands[band] = refl
        return {'scl': scl, 'bands': bands}

    def point_series(self, lat: float, lon: float, start: str, end: str,
                     cfg: PurityConfig) -> list[dict]:
        items = self.search(lat, lon, start, end, cfg.max_scene_cloud)

        def one(item):
            for attempt in range(3):
                try:
                    fp = self.read_footprint(item, lat, lon, cfg.window_px)
                    break
                except Exception as exc:  # network hiccup on a single COG
                    if attempt == 2:
                        logger.warning("Read failed for %s: %s", item.id, exc)
                        return None
                    time.sleep(2 * (attempt + 1))
            if fp is None:
                return None
            obs = summarise_window(fp['bands'], fp['scl'], cfg)
            props = item.properties
            obs.update({
                'datetime': item.datetime.isoformat() if item.datetime else props.get('datetime'),
                'item_id': item.id,
                'platform': props.get('platform'),
                'tile': props.get('grid:code') or props.get('s2:mgrs_tile'),
                'scene_cloud': props.get('eo:cloud_cover'),
                'processing_baseline': props.get('s2:processing_baseline'),
            })
            return obs

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return [o for o in pool.map(one, items) if o is not None]


# ---------------------------------------------------------------------------
# Copernicus Data Space Ecosystem (Sentinel Hub Statistical API)
# ---------------------------------------------------------------------------

CDSE_TOKEN_URL = ('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/'
                  'protocol/openid-connect/token')
CDSE_STATS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/statistics'

CDSE_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B05","B06","B07","B08","B11","B12","SCL","dataMask"]}],
    output: [
      {id: "bands", bands: 6, sampleType: "FLOAT32"},
      {id: "veg", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function inRange(v) { return v > 0 && v <= 1; }
function evaluatePixel(s) {
  var refl = [s.B05, s.B06, s.B07, s.B08, s.B11, s.B12];
  var ok = s.dataMask == 1 && (s.SCL == 4 || s.SCL == 5) && refl.every(inRange);
  return {bands: refl, veg: [s.SCL == 4 ? 1 : 0], dataMask: [ok ? 1 : 0]};
}
"""


class CDSEBackend:
    name = 'cdse'

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or os.environ.get('CDSE_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('CDSE_CLIENT_SECRET')
        if not (self.client_id and self.client_secret):
            raise RuntimeError(
                "CDSE backend needs CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (create an OAuth "
                "client in your Copernicus Data Space dashboard), or use --backend earthsearch.")
        self.session = http_session()
        self._token, self._token_expiry = None, 0.0

    def _auth_header(self) -> dict:
        if self._token is None or time.time() > self._token_expiry - 60:
            r = self.session.post(CDSE_TOKEN_URL, data={
                'grant_type': 'client_credentials', 'client_id': self.client_id,
                'client_secret': self.client_secret}, timeout=60)
            r.raise_for_status()
            tok = r.json()
            self._token = tok['access_token']
            self._token_expiry = time.time() + float(tok.get('expires_in', 600))
        return {'Authorization': f'Bearer {self._token}'}

    def point_series(self, lat: float, lon: float, start: str, end: str,
                     cfg: PurityConfig) -> list[dict]:
        from pyproj import Transformer

        epsg = utm_epsg(lat, lon)
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        left, bottom, right, top = snap_footprint(x, y, cfg.window_px)
        body = {
            'input': {
                'bounds': {'bbox': [left, bottom, right, top],
                           'properties': {'crs': f'http://www.opengis.net/def/crs/EPSG/0/{epsg}'}},
                'data': [{'type': 'sentinel-2-l2a',
                          'dataFilter': {'maxCloudCoverage': cfg.max_scene_cloud,
                                         'mosaickingOrder': 'leastCC'}}],
            },
            'aggregation': {
                'timeRange': {'from': f'{start}T00:00:00Z', 'to': f'{end}T23:59:59Z'},
                'aggregationInterval': {'of': 'P1D'},
                'evalscript': CDSE_EVALSCRIPT,
                'resx': NATIVE_GRID_M, 'resy': NATIVE_GRID_M,
            },
            'calculations': {'default': {}},
        }
        r = self.session.post(CDSE_STATS_URL, json=body, headers=self._auth_header(), timeout=300)
        r.raise_for_status()
        return [o for o in (self._parse_interval(d, cfg) for d in r.json().get('data', []))
                if o is not None]

    @staticmethod
    def _parse_interval(entry: dict, cfg: PurityConfig) -> dict | None:
        outputs = entry.get('outputs', {})
        band_stats = outputs.get('bands', {}).get('bands', {})
        if not band_stats:
            return None

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        first = band_stats['B0']['stats']
        n_pix = int(first.get('sampleCount', 0))
        n_valid = n_pix - int(first.get('noDataCount', n_pix))
        if n_pix == 0:
            return None
        obs = {'n_pixels': n_pix, 'n_valid': n_valid, 'valid_fraction': n_valid / n_pix,
               'center_valid': np.nan, 'center_scl': np.nan}
        for i, band in enumerate(S2_BANDS):
            obs[band] = num(band_stats[f'B{i}']['stats'].get('mean'))
        veg = outputs.get('veg', {}).get('bands', {}).get('B0', {}).get('stats', {})
        obs['vegetation_fraction'] = num(veg.get('mean')) if n_valid else 0.0
        nir = band_stats['B3']['stats']
        mean, sd = num(nir.get('mean')), num(nir.get('stDev'))
        obs['nir_cv'] = sd / mean if n_valid > 1 and mean > 0 else np.nan
        obs['qc_reason'] = qc_reason(obs, cfg, center_known=False)
        obs['purity_pass'] = obs['qc_reason'] == ''
        obs.update({'datetime': entry['interval']['from'], 'item_id': '', 'platform': '',
                    'tile': '', 'scene_cloud': np.nan, 'processing_baseline': ''})
        return obs


def make_backend(name: str, workers: int = 8):
    if name == 'earthsearch':
        return EarthSearchBackend(workers=workers)
    if name == 'cdse':
        return CDSEBackend()
    raise ValueError(f"Unknown backend {name!r}")


# ---------------------------------------------------------------------------
# Many points, cached
# ---------------------------------------------------------------------------

OBS_COLUMNS = ['point_id', 'date', 'datetime', 'backend', 'item_id', 'platform', 'tile',
               'scene_cloud', 'processing_baseline', *S2_BANDS, 'n_pixels', 'n_valid',
               'valid_fraction', 'vegetation_fraction', 'center_valid', 'center_scl',
               'nir_cv', 'purity_pass', 'qc_reason']


def dedupe_acquisitions(obs: pd.DataFrame) -> pd.DataFrame:
    """
    One observation per point and day. The same acquisition can appear in
    two overlapping MGRS tiles; keep the one with the most valid pixels.
    """
    if obs.empty:
        return obs
    obs = obs.sort_values(['point_id', 'date', 'n_valid', 'scene_cloud'],
                          ascending=[True, True, False, True])
    return obs.drop_duplicates(['point_id', 'date'], keep='first').reset_index(drop=True)


def extract_points(points: pd.DataFrame, backend, cfg: PurityConfig,
                   cache_dir: str, point_workers: int = 4) -> pd.DataFrame:
    """
    Time series for every row of `points` (columns point_id, lat, lon, start, end).

    Each point is cached as a CSV keyed by backend, window and date range,
    so an interrupted run resumes where it stopped. Up to `point_workers`
    points are processed at once (each with the backend's own item-level
    parallelism).
    """
    os.makedirs(cache_dir, exist_ok=True)
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    total = len(points)

    def one(arg):
        i, p = arg
        key = f"{p.point_id}_{backend.name}_w{cfg.window_px}_{p.start}_{p.end}.csv"
        path = os.path.join(cache_dir, key)
        if os.path.exists(path):
            return pd.read_csv(path)
        t0 = time.time()
        try:
            rows = backend.point_series(float(p.lat), float(p.lon), str(p.start), str(p.end), cfg)
        except Exception:
            logger.exception("Sentinel-2 extraction failed for %s", p.point_id)
            return pd.DataFrame(columns=OBS_COLUMNS)
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=OBS_COLUMNS)
        else:
            df['point_id'] = p.point_id
            df['backend'] = backend.name
            df['date'] = pd.to_datetime(df['datetime'], utc=True,
                                        format='ISO8601').dt.strftime('%Y-%m-%d')
            df = dedupe_acquisitions(df.reindex(columns=OBS_COLUMNS))
        df.to_csv(path, index=False)
        logger.info("[%d/%d] %s: %d acquisitions, %d pass QC (%.0fs)", i, total,
                    p.point_id, len(df), int(df['purity_pass'].astype(bool).sum()),
                    time.time() - t0)
        return df

    with ThreadPoolExecutor(max_workers=max(1, point_workers)) as pool:
        frames = list(pool.map(one, enumerate(points.itertuples(index=False), 1)))
    frames = [f for f in frames if not f.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OBS_COLUMNS)
    out['purity_pass'] = out['purity_pass'].astype(str).str.lower().isin(['true', '1'])
    out['qc_reason'] = out['qc_reason'].fillna('')     # '' round-trips through CSV as NaN
    return out


def save_config(cfg: PurityConfig, path: str, **extra) -> None:
    with open(path, 'w') as f:
        json.dump({**asdict(cfg), **extra}, f, indent=2)
