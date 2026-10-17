"""
Wheat Septoria VNIR + SWIR hyperspectral dataset (Data INRAE, doi:10.57745/WVP0FJ)
===============================================================================
Durum wheat leaves (genotypes EPO_67, EPO_68) inoculated with Zymoseptoria
tritici ("Infected") or with water + Tween 20 only ("Non-infected"), imaged
by two HySpex cameras on 18 dates from -1 to 20 days post inoculation (dpi):

    VNIR  1024 samples, 216 bands, 411-993 nm    (~440 MB per image, zipped)
    SWIR   320 samples, 256 bands, 964-2494 nm   (~50 MB per image, zipped)

One archive per date and camera (611 GB in total); inside, one folder per
pot side (`<pot>_<G|D>/`) with the image cube (`.hyspex` + ENVI `.hdr`),
an RGB preview and one pixel mask per leaf (`<pot>_<side>_leaf<k>.csv`).

What the files contain (verified on the archives, not assumed):
    - cubes are RAW sensor counts (uint16, BIL, after a binary header of
      `header offset` bytes), not reflectance; a white reference panel is
      scanned first in every image, so relative reflectance = leaf / panel
      per band and column. The counts are already dark-subtracted (the
      darkest pixels are exactly 0 in every band). 'data ignore value' is
      not a saturation level (VNIR counts reach ~27000 with ignore = 4095).
    - leaf masks list 1-based (line, column) pixels; the column axis is
      mirrored in the SWIR images checked and not in the VNIR ones, so the
      orientation is chosen per image (mask_orientation), never assumed.
    - the leaf annotation (Data_description.csv) has ONE infection-confidence
      value per leaf, from a visual symptom assessment, and no per-date
      symptom record (see visible_onset for how that is handled).

Everything is read straight out of the remote ZIPs with HTTP range requests
(src.real_data.common), so only the members that are needed are downloaded.
"""

import io
import logging
import os
import re

import numpy as np
import pandas as pd

from src.real_data.common import data_dir, http_session

logger = logging.getLogger(__name__)

SERVER = 'https://entrepot.recherche.data.gouv.fr'
DOI = 'doi:10.57745/WVP0FJ'
INOCULATION_DATE = pd.Timestamp('2021-03-12')     # 0 dpi (first archive is -1 dpi)
ANNOTATION_FILES = ('Data_description.csv', 'VNIR_data.csv', 'SWIR_data.csv')
SENSORS = ('VNIR', 'SWIR')

MEMBER_RE = re.compile(r'^(?P<date>\d{8})(?:_part\d)?/(?P<pot>\d+)_(?P<side>[GD])/'
                       r'(?P=pot)_(?P=side)(?:_leaf(?P<leaf>\d+))?\.(?P<ext>\w+)$')


def septoria_dir(*parts: str) -> str:
    return data_dir('septoria', *parts)


def file_url(file_id: int) -> str:
    return f'{SERVER}/api/access/datafile/{file_id}'


# Archives downloaded by hand (e.g. on a faster connection) go here, one folder
# per camera because both cameras use the same file names (20210311.zip):
#     data/real/septoria/archives/VNIR/20210311.zip
ARCHIVE_DIR = os.environ.get('SEPTORIA_ARCHIVE_DIR',
                             os.path.join('data', 'real', 'septoria', 'archives'))


def local_archive(sensor: str, archive: str, archive_dir: str | None = None) -> str | None:
    """Path of a hand-downloaded archive, or None. Partial downloads are rejected."""
    import zipfile

    path = os.path.join(archive_dir or ARCHIVE_DIR, sensor, archive)
    if not os.path.exists(path):
        return None
    if not zipfile.is_zipfile(path):
        raise zipfile.BadZipFile(f"{path} is not a complete ZIP archive (still downloading?)")
    return path


def open_archive(sensor: str, archive: str, file_id: int, archive_dir: str | None = None):
    """The archive as a ZipFile: the local copy if there is one, else over HTTP ranges."""
    import zipfile

    from src.real_data.common import open_remote_zip

    path = local_archive(sensor, archive, archive_dir)
    if path:
        logger.info("Using local archive %s", path)
        return zipfile.ZipFile(path)
    return open_remote_zip(file_url(file_id))


def read_member(zf, name: str, workers: int = 6) -> bytes:
    """A member's bytes: parallel range requests for a remote archive, plain read locally."""
    from src.real_data.common import HTTPRangeFile, read_member_parallel

    if isinstance(getattr(zf.fp, 'raw', None), HTTPRangeFile):
        return read_member_parallel(zf, name, workers=workers)
    return zf.read(name)


# ---------------------------------------------------------------------------
# Dataset file list and annotations
# ---------------------------------------------------------------------------

def list_files(session=None) -> pd.DataFrame:
    """Every file of the dataset: id, name, size, folder, sensor, date (cached)."""
    path = os.path.join(septoria_dir(), 'files.csv')
    if os.path.exists(path):
        return pd.read_csv(path, dtype={'date': str})
    session = session or http_session()
    r = session.get(f'{SERVER}/api/datasets/:persistentId/',
                    params={'persistentId': DOI}, timeout=120)
    r.raise_for_status()
    rows = []
    for f in r.json()['data']['latestVersion']['files']:
        df = f['dataFile']
        folder = f.get('directoryLabel', '')
        sensor = folder.rsplit('/', 1)[-1] if folder.startswith('Data/') else None
        rows.append({'file_id': df['id'], 'filename': df['filename'],
                     'size': df.get('filesize', 0), 'folder': folder,
                     'sensor': sensor if sensor in SENSORS else None,
                     'date': df['filename'][:8] if df['filename'][:8].isdigit() else None})
    files = pd.DataFrame(rows).sort_values(['sensor', 'filename'], na_position='first')
    files.to_csv(path, index=False)
    return files


def fetch_annotations(session=None) -> dict[str, str]:
    """Download the three small annotation tables; returns name -> local path."""
    from src.real_data.common import download_file

    files = list_files(session).set_index('filename')
    return {name: download_file(file_url(int(files.loc[name, 'file_id'])),
                                os.path.join(septoria_dir(), name), session)
            for name in ANNOTATION_FILES}


def load_leaves(path: str | None = None) -> pd.DataFrame:
    """
    Data_description.csv -> one row per leaf: pot, side, leaf, genotype,
    treatment, infected (bool) and confidence (1 = high, 2 = low; a visual
    assessment of symptoms).
    """
    path = path or os.path.join(septoria_dir(), 'Data_description.csv')
    raw = pd.read_csv(path, sep=';', skiprows=2, encoding='latin1', usecols=range(6))
    raw.columns = ['pot', 'side', 'leaf', 'genotype', 'treatment', 'confidence']
    raw = raw.dropna(subset=['pot', 'side', 'leaf'])
    raw = raw[pd.to_numeric(raw['pot'], errors='coerce').notna()]
    leaves = raw.astype({'pot': int, 'leaf': int, 'confidence': int})
    leaves['side'] = leaves['side'].str.strip()
    leaves['treatment'] = leaves['treatment'].str.strip()
    leaves['infected'] = leaves['treatment'].eq('Infected')
    return leaves.reset_index(drop=True)


def load_acquisitions(sensor: str, path: str | None = None) -> pd.DataFrame:
    """
    VNIR_data.csv / SWIR_data.csv -> one row per (pot, side, dpi) with the
    number of leaves imaged ('NA' = technical failure -> n_leaves NaN,
    missing = pot side not imaged that day).
    """
    path = path or os.path.join(septoria_dir(), f'{sensor}_data.csv')
    table = pd.read_csv(path, sep=';', header=None, skiprows=2, encoding='latin1',
                        dtype=str)
    dpi_row = table.iloc[0, 3:].tolist()
    side_row = table.iloc[1, 3:].tolist()
    columns, current = [], None
    for dpi, side in zip(dpi_row, side_row):
        if isinstance(dpi, str) and dpi.strip().lstrip('-').isdigit():
            current = int(dpi)
        columns.append((current, str(side).strip()[0] if isinstance(side, str) else None))
    rows = []
    for _, rec in table.iloc[2:].iterrows():
        pot = str(rec.iloc[0]).strip()
        if not pot.isdigit():
            continue
        for (dpi, side), value in zip(columns, rec.iloc[3:].tolist()):
            if dpi is None or side not in ('G', 'D') or not isinstance(value, str) \
                    or not value.strip():
                continue
            value = value.strip()
            rows.append({'pot': int(pot), 'side': side, 'dpi': dpi, 'sensor': sensor,
                         'n_leaves': np.nan if value == 'NA' else int(value)})
    return pd.DataFrame(rows)


def dpi_of(date: str) -> int:
    """'20210331' -> days post inoculation."""
    return int((pd.Timestamp(date) - INOCULATION_DATE).days)


# ---------------------------------------------------------------------------
# Archive contents (central directories only)
# ---------------------------------------------------------------------------

def archive_index(dates: list[str] | None = None, sensors=SENSORS,
                  session=None) -> pd.DataFrame:
    """
    One row per member of the image archives (date, sensor, pot, side, leaf,
    ext, sizes), read from the ZIP central directories without downloading
    the images. Cached per archive in data/real/septoria/index/.
    """
    from src.real_data.common import open_remote_zip

    session = session or http_session()
    files = list_files(session)
    files = files[files['sensor'].isin(sensors) & files['date'].notna()]
    if dates is not None:
        files = files[files['date'].isin(dates)]
    frames = []
    for rec in files.itertuples():
        cache = os.path.join(septoria_dir('index'), f'{rec.sensor}_{rec.filename}.csv')
        if not os.path.exists(cache):
            zf = open_remote_zip(file_url(rec.file_id), session=session)
            rows = []
            for info in zf.infolist():
                m = MEMBER_RE.match(info.filename)
                if m:
                    rows.append({'member': info.filename, 'date': m['date'],
                                 'pot': int(m['pot']), 'side': m['side'],
                                 'leaf': int(m['leaf']) if m['leaf'] else None,
                                 'ext': m['ext'], 'size': info.file_size,
                                 'compressed': info.compress_size})
            if not rows:
                raise ValueError(f"No image members recognised in {rec.filename}; "
                                 "the archive layout differs from MEMBER_RE")
            pd.DataFrame(rows).to_csv(cache, index=False)
            logger.info("Indexed %s %s: %d members", rec.sensor, rec.filename, len(rows))
        frame = pd.read_csv(cache, dtype={'date': str})
        frame['sensor'], frame['file_id'], frame['archive'] = \
            rec.sensor, rec.file_id, rec.filename
        frames.append(frame)
    index = pd.concat(frames, ignore_index=True)
    index['dpi'] = index['date'].map(dpi_of)
    return index


# ---------------------------------------------------------------------------
# Cubes -> per-leaf relative reflectance
# ---------------------------------------------------------------------------

# Project bands (src.constants.BANDS) and the camera that measures each one.
# Each is the mean over +/- BAND_HALF_WIDTH_NM, roughly a Sentinel-2 band width.
PROJECT_BANDS = {'red_edge_705': ('VNIR', 705.0), 'red_edge_740': ('VNIR', 740.0),
                 'red_edge_783': ('VNIR', 783.0), 'nir_842': ('VNIR', 842.0),
                 'swir_1610': ('SWIR', 1610.0), 'swir_2190': ('SWIR', 2190.0)}
BAND_HALF_WIDTH_NM = 10.0
# Extra narrow bands kept for the hyperspectral feature table (RF / XGB / QUBO).
EXTRA_BANDS = {'VNIR': (450, 531, 550, 570, 670, 680, 715, 725, 760, 900, 970),
               'SWIR': (1000, 1200, 1450, 1650, 1940, 2100, 2300)}
PANEL_BAND_NM = {'VNIR': 800.0, 'SWIR': 1650.0}


def parse_hdr(text: str) -> dict:
    """The ENVI header fields the reader needs."""
    def field(name):
        m = re.search(rf'^\s*{name}\s*=\s*(.+)$', text, re.M | re.I)
        return m.group(1).strip() if m else None
    wl = re.search(r'wavelength\s*=\s*\{([^}]*)\}', text, re.I).group(1)
    return {'lines': int(field('lines')), 'samples': int(field('samples')),
            'bands': int(field('bands')), 'offset': int(field('header offset')),
            'dtype': {'12': '<u2', '2': '<i2', '4': '<f4'}[field('data type')],
            'interleave': field('interleave').lower(),
            # 'data ignore value' is NOT a saturation level here (VNIR counts reach
            # ~27000 with ignore = 4095: frames are averaged), so saturation is
            # the integer ceiling of the data type.
            'saturation': (np.iinfo({'12': np.uint16, '2': np.int16}[field('data type')]).max
                           if field('data type') in ('12', '2') else np.inf),
            'wavelength': np.array([float(v) for v in wl.split(',')]),
            'date': field('acquisition date')}


def read_cube(raw: bytes, hdr: dict) -> np.ndarray:
    """Raw .hyspex bytes -> (lines, bands, samples) array (BIL order)."""
    if hdr['interleave'] != 'bil':
        raise ValueError(f"Expected BIL interleave, got {hdr['interleave']}")
    n = hdr['lines'] * hdr['bands'] * hdr['samples']
    data = np.frombuffer(raw, dtype=hdr['dtype'], count=n, offset=hdr['offset'])
    return data.reshape(hdr['lines'], hdr['bands'], hdr['samples'])


def panel_lines(cube: np.ndarray, band: int, search_frac: float = 0.3) -> np.ndarray:
    """
    Lines of the white reference panel: the panel is scanned first, so it is
    the run of bright lines at the start of the scan. A line belongs to it
    when the median of its central samples is >= 85% of the brightest line
    in the first `search_frac` of the scan.
    """
    lines, _, samples = cube.shape
    centre = cube[:, band, samples // 8: samples - samples // 8].astype(np.float64)
    level = np.median(centre, axis=1)
    head = level[:max(1, int(lines * search_frac))]
    bright = np.flatnonzero(head >= 0.85 * head.max())
    runs = [r for r in np.split(bright, np.flatnonzero(np.diff(bright) > 1) + 1) if len(r) >= 5]
    if not runs:
        raise ValueError("No white reference panel found at the start of the scan")
    run = runs[0]
    return run[2:-2] if len(run) > 8 else run                # drop the soft edges


def white_reference(cube: np.ndarray, hdr: dict, sensor: str) -> np.ndarray:
    """(bands, samples) mean panel counts, i.e. the per-column flat field."""
    band = int(np.argmin(np.abs(hdr['wavelength'] - PANEL_BAND_NM[sensor])))
    rows = panel_lines(cube, band)
    panel = cube[rows].astype(np.float64)
    if (panel >= hdr['saturation']).mean() > 0.01:
        raise ValueError("White panel saturated")
    return panel.mean(axis=0)


def read_mask(mask_csv: bytes | str) -> np.ndarray:
    """Leaf mask CSV -> (n, 2) 1-based (line, column) pixel list."""
    import io
    text = mask_csv.decode('latin1') if isinstance(mask_csv, bytes) else mask_csv
    return pd.read_csv(io.StringIO(text), header=None).to_numpy(dtype=np.int64)


def leaf_pixels(rc: np.ndarray, hdr: dict, mirrored: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    1-based (line, column) mask -> (line, sample) indices into the cube.
    The column axis is mirrored relative to the cube in some images and not
    in others (SWIR and VNIR differ), so the orientation is decided per
    image by mask_orientation, never assumed.
    """
    line = rc[:, 0] - 1
    sample = hdr['samples'] - rc[:, 1] if mirrored else rc[:, 1] - 1
    ok = (line >= 0) & (line < hdr['lines']) & (sample >= 0) & (sample < hdr['samples'])
    return line[ok], sample[ok]


# Leaf vs. background: leaves reflect >0.2 in the NIR / SWIR plateau, the
# black board behind them < 0.1.
LEAF_TEST_NM = {'VNIR': 800.0, 'SWIR': 1250.0}
LEAF_TEST_LEVEL = 0.2
# Narrow leaves are mostly edge pixels (mixed with the board, dropped later),
# so the right orientation can fit only ~0.7.
MIN_LEAF_FRACTION = 0.7
# Mask column orientation of each camera, found on every cohort image so far
# (SWIR 96/96 mirrored, VNIR 22/22 not). Broad leaves can fit both ways, so the
# known orientation is used unless the image clearly contradicts it.
EXPECTED_MIRRORED = {'VNIR': False, 'SWIR': True}
MIN_ORIENTATION_MARGIN = 0.2
# Late, drying leaves curl out of masks drawn earlier: the known orientation
# is still accepted from this fit when it beats the other one clearly.
MIN_EXPECTED_FRACTION = 0.6


def mask_orientation(cube: np.ndarray, white: np.ndarray, hdr: dict, masks: list[np.ndarray],
                     sensor: str) -> tuple[bool, float]:
    """
    (mirrored, fraction of mask pixels that look like leaf). The camera's
    known orientation is kept when it fits; otherwise the other one, if it
    fits clearly better. Raises if neither does, so a misregistered image is
    never silently turned into features.
    """
    band = int(np.argmin(np.abs(hdr['wavelength'] - LEAF_TEST_NM[sensor])))
    refl = cube[:, band, :] / np.maximum(white[band][None, :], 1.0)
    scores = {}
    for mirrored in (False, True):
        hits = [refl[leaf_pixels(rc, hdr, mirrored)] > LEAF_TEST_LEVEL for rc in masks]
        scores[mirrored] = float(np.concatenate(hits).mean())
    expected = EXPECTED_MIRRORED.get(sensor)
    if (expected is not None and scores[expected] >= MIN_LEAF_FRACTION
            and scores[not expected] - scores[expected] < MIN_ORIENTATION_MARGIN):
        return expected, scores[expected]
    if (expected is not None and scores[expected] >= MIN_EXPECTED_FRACTION
            and scores[expected] - scores[not expected] >= MIN_ORIENTATION_MARGIN):
        return expected, scores[expected]
    best = max(scores, key=scores.get)
    if scores[best] < MIN_LEAF_FRACTION or scores[best] - scores[not best] < MIN_ORIENTATION_MARGIN:
        raise ValueError(f"Leaf masks do not fit the image (leaf fraction {scores})")
    return best, scores[best]


def leaf_reflectance(cube: np.ndarray, white: np.ndarray, line: np.ndarray,
                     sample: np.ndarray, saturation: float) -> np.ndarray:
    """(n_pixels, bands) relative reflectance; saturated pixels dropped."""
    dn = cube[line, :, sample].astype(np.float64)               # (n, bands)
    keep = ~(dn >= saturation).any(axis=1)
    return dn[keep] / np.maximum(white[:, sample[keep]].T, 1.0)


def band_mean(refl: np.ndarray, wavelength: np.ndarray, centre: float,
              half_width: float = BAND_HALF_WIDTH_NM) -> np.ndarray:
    """Per-pixel mean reflectance over [centre - hw, centre + hw]."""
    sel = np.abs(wavelength - centre) <= half_width
    if not sel.any():
        sel = np.abs(wavelength - centre) == np.abs(wavelength - centre).min()
    return refl[:, sel].mean(axis=1)


def leaf_features(refl: np.ndarray, wavelength: np.ndarray, sensor: str) -> dict:
    """
    Per-leaf features from one camera: the project bands it covers, the
    extra narrow bands, and for VNIR the within-leaf heterogeneity and the
    fraction of chlorotic pixels (a symptom proxy, see septoria_symptoms).
    """
    feats = {'n_pixels': len(refl)}
    for name, (cam, centre) in PROJECT_BANDS.items():
        if cam == sensor:
            feats[name] = float(np.median(band_mean(refl, wavelength, centre)))
    for centre in EXTRA_BANDS[sensor]:
        feats[f'R{centre}'] = float(np.median(band_mean(refl, wavelength, centre, 5.0)))
    if sensor == 'VNIR':
        nir = band_mean(refl, wavelength, 800.0)
        red = band_mean(refl, wavelength, 670.0, 5.0)
        green = band_mean(refl, wavelength, 550.0, 5.0)
        re705 = band_mean(refl, wavelength, 705.0)
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        ndre = (nir - re705) / np.maximum(nir + re705, 1e-6)
        feats.update({
            'ndvi_p10': float(np.percentile(ndvi, 10)), 'ndvi_std': float(ndvi.std()),
            'ndre_p10': float(np.percentile(ndre, 10)), 'ndre_std': float(ndre.std()),
            # chlorotic / necrotic pixel: red no longer well below green
            'chlorotic_frac': float(np.mean(red / np.maximum(green, 1e-6) > 0.85)),
        })
    return feats


def image_cache(sensor: str, date: str, pot: int, side: str) -> str:
    return os.path.join(septoria_dir('spectra', sensor), f'{date}_{pot:02d}_{side}.npz')


def extract_image(index: pd.DataFrame, sensor: str, date: str, pot: int, side: str,
                  zf=None, workers: int = 6) -> str:
    """
    Download one image (cube + header + leaf masks) out of its archive and
    reduce it to per-leaf data, saved as an .npz (the cube is not kept):
    wavelength, per leaf the median / 10th / 90th percentile relative
    reflectance spectrum and the leaf_features. Returns the cache path.
    """
    import json

    path = image_cache(sensor, date, pot, side)
    if os.path.exists(path):
        return path
    rows = index[(index['sensor'] == sensor) & (index['date'] == date) &
                 (index['pot'] == pot) & (index['side'] == side)]
    by_ext = {ext: grp for ext, grp in rows.groupby('ext')}
    if 'hyspex' not in by_ext or 'csv' not in by_ext:
        raise FileNotFoundError(f"{sensor} {date} {pot}_{side}: no cube or no leaf masks")
    zf = zf or open_archive(sensor, rows['archive'].iloc[0], int(rows['file_id'].iloc[0]))
    hdr = parse_hdr(zf.read(by_ext['hdr']['member'].iloc[0]).decode('latin1'))
    cube = read_cube(read_member(zf, by_ext['hyspex']['member'].iloc[0], workers), hdr)
    white = white_reference(cube, hdr, sensor)
    masks = {int(rec.leaf): read_mask(zf.read(rec.member)) for rec in by_ext['csv'].itertuples()}
    mirrored, leaf_fraction = mask_orientation(cube, white, hdr, list(masks.values()), sensor)
    out = {'wavelength': hdr['wavelength'], 'mirrored': np.array(mirrored),
           'leaf_fraction': np.array(leaf_fraction)}
    test_band = int(np.argmin(np.abs(hdr['wavelength'] - LEAF_TEST_NM[sensor])))
    feats = {}
    for leaf, rc in masks.items():
        line, sample = leaf_pixels(rc, hdr, mirrored)
        refl = leaf_reflectance(cube, white, line, sample, hdr['saturation'])
        # Drop mask-edge pixels that are mostly black board (mixed pixels):
        # leaf tissue, including chlorotic / necrotic lesions, stays above
        # LEAF_TEST_LEVEL on the NIR / SWIR plateau; the board does not.
        on_leaf = refl[:, test_band] > LEAF_TEST_LEVEL
        kept = float(on_leaf.mean()) if len(refl) else 0.0
        refl = refl[on_leaf]
        if len(refl) < 50:
            logger.warning("%s %s %d_%s leaf %d: only %d pixels, skipped",
                           sensor, date, pot, side, leaf, len(refl))
            continue
        out[f'leaf{leaf}_p50'] = np.median(refl, axis=0).astype(np.float32)
        out[f'leaf{leaf}_p10'] = np.percentile(refl, 10, axis=0).astype(np.float32)
        out[f'leaf{leaf}_p90'] = np.percentile(refl, 90, axis=0).astype(np.float32)
        feats[leaf] = {**leaf_features(refl, hdr['wavelength'], sensor), 'kept_fraction': kept}
    out['features'] = np.array(json.dumps(feats))
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return path


def load_image_features(path: str) -> dict[int, dict]:
    import json
    with np.load(path) as z:
        return {int(k): v for k, v in json.loads(str(z['features'])).items()}


# ---------------------------------------------------------------------------
# Cohort, observations, symptom onset, labelled sequences
# ---------------------------------------------------------------------------

# The extracted cohort (chosen for download cost; see README): the 12 pots
# imaged on every date, left side, on the pre-symptomatic window and onset.
COHORT_POTS = tuple(range(37, 49))
COHORT_SIDE = 'G'
COHORT_DPI = (-1, 4, 7, 10, 12, 13, 14, 15)

# Visible-abnormality rule (see visible_onset). A leaf observation is visibly
# abnormal when more than ABNORMAL_LEVEL of its pixels are chlorotic or
# necrotic (red no longer well below green). Healthy leaves read 0-0.03.
ABNORMAL_LEVEL = 0.05
# Per-pixel chlorosis test, shared by the VNIR cube (reflectance) and the RGB
# previews (rendered colour): red / green above this = yellow or brown.
CHLOROTIC_RED_GREEN = 0.85
# Preview leaf pixel: coloured (green OR yellow / brown), not the black or grey
# board: max(R, G) > 35 and above blue by 10 (0-255). A brightness test alone
# drops dark-green tissue of narrow leaves; a green test would drop lesions.
PREVIEW_MIN_LEVEL = 35
PREVIEW_MIN_CHROMA = 10
PREVIEW_MIN_FIT = 0.5


# ---------------------------------------------------------------------------
# RGB previews: a symptom measure without the 440 MB VNIR cubes
# ---------------------------------------------------------------------------

def preview_cache(date: str, pot: int, side: str) -> str:
    return os.path.join(septoria_dir('previews'), f'{date}_{pot:02d}_{side}.json')


def preview_features(jpg: bytes, masks: dict[int, np.ndarray]) -> dict:
    """
    Per-leaf chlorotic fraction from the RGB preview of a VNIR image. The
    preview is the cube rendered with x = line, y = sample, so the VNIR leaf
    masks apply; the column orientation is still tested, as for the cubes.
    """
    from PIL import Image

    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert('RGB'), dtype=np.float64)
    height, width, _ = rgb.shape
    top = np.maximum(rgb[:, :, 0], rgb[:, :, 1])
    bright = (top > PREVIEW_MIN_LEVEL) & (top > rgb[:, :, 2] + PREVIEW_MIN_CHROMA)

    def pixels(rc, mirrored):
        x = rc[:, 0] - 1
        y = height - rc[:, 1] if mirrored else rc[:, 1] - 1
        ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return y[ok], x[ok]

    scores = {m: float(np.concatenate([bright[pixels(rc, m)] for rc in masks.values()]).mean())
              for m in (False, True)}
    mirrored = max(scores, key=scores.get)
    if scores[mirrored] < PREVIEW_MIN_FIT:
        raise ValueError(f"Leaf masks do not fit the preview (leaf fraction {scores})")
    out = {'mirrored': mirrored, 'leaf_fraction': scores[mirrored], 'leaves': {}}
    for leaf, rc in masks.items():
        y, x = pixels(rc, mirrored)
        px = rgb[y, x][bright[y, x]]
        out['leaves'][int(leaf)] = {
            'rgb_pixels': int(len(px)),
            'rgb_chlorotic_frac': float(np.mean(px[:, 0] / np.maximum(px[:, 1], 1.0)
                                                > CHLOROTIC_RED_GREEN)) if len(px) else np.nan}
    return out


def fetch_previews(pots=COHORT_POTS, side: str = COHORT_SIDE, dpis=COHORT_DPI) -> int:
    """Preview + VNIR leaf masks of every cohort image (~1 MB each); returns count."""
    import json

    index = archive_index(sensors=('VNIR',))
    done = 0
    for dpi in dpis:
        rows = index[(index['dpi'] == dpi) & (index['side'] == side) &
                     (index['pot'].isin(pots))]
        todo = [p for p in sorted(rows['pot'].unique())
                if not os.path.exists(preview_cache(rows['date'].iloc[0], p, side))]
        if not todo:
            continue
        zf = open_archive('VNIR', rows['archive'].iloc[0], int(rows['file_id'].iloc[0]))
        for pot in todo:
            mine = rows[rows['pot'] == pot]
            try:
                jpg = read_member(zf, mine.loc[mine['ext'] == 'jpg', 'member'].iloc[0], workers=2)
                masks = {int(r.leaf): read_mask(zf.read(r.member))
                         for r in mine[mine['ext'] == 'csv'].itertuples()}
                feats = preview_features(jpg, masks)
            except Exception as exc:
                logger.error("Preview %+d dpi pot %d_%s failed: %s", dpi, pot, side, exc)
                continue
            with open(preview_cache(mine['date'].iloc[0], pot, side), 'w') as f:
                json.dump(feats, f)
            done += 1
            logger.info("Preview %+d dpi pot %d_%s done", dpi, pot, side)
    return done


# ---------------------------------------------------------------------------
# Observations, visible-symptom onset, states
# ---------------------------------------------------------------------------

def build_observations(pots=COHORT_POTS, side=COHORT_SIDE, dpis=COHORT_DPI,
                       sensors=SENSORS) -> pd.DataFrame:
    """
    One row per (pot, side, leaf, dpi): the features of every camera in
    `sensors` (rows need all of them), the leaf annotation, and the symptom
    measure `symptom_frac` -- the VNIR cube's chlorotic fraction when VNIR
    was extracted, else the RGB preview's (`symptom_source` says which).
    """
    import json

    leaves = load_leaves()
    rows = []
    for dpi in dpis:
        date = (INOCULATION_DATE + pd.Timedelta(days=dpi)).strftime('%Y%m%d')
        for pot in pots:
            per_sensor = {s: load_image_features(image_cache(s, date, pot, side))
                          for s in sensors if os.path.exists(image_cache(s, date, pot, side))}
            if len(per_sensor) < len(sensors):
                continue
            preview = {}
            if os.path.exists(preview_cache(date, pot, side)):
                with open(preview_cache(date, pot, side)) as f:
                    preview = {int(k): v for k, v in json.load(f)['leaves'].items()}
            for leaf in sorted(set.intersection(*(set(f) for f in per_sensor.values()))):
                row = {'pot': pot, 'side': side, 'leaf': leaf, 'dpi': dpi, 'date': date,
                       **preview.get(leaf, {})}
                for sensor, feats in per_sensor.items():     # n_pixels is in both cameras
                    row.update({f'{sensor.lower()}_{k}' if k in ('n_pixels', 'kept_fraction')
                                else k: v for k, v in feats[leaf].items()})
                rows.append(row)
    obs = pd.DataFrame(rows)
    if obs.empty:
        return obs
    if 'chlorotic_frac' in obs:
        obs['symptom_frac'], obs['symptom_source'] = obs['chlorotic_frac'], 'vnir_cube'
    elif 'rgb_chlorotic_frac' in obs:
        obs['symptom_frac'], obs['symptom_source'] = obs['rgb_chlorotic_frac'], 'rgb_preview'
    else:
        raise ValueError("No symptom measure: extract VNIR or run fetch_previews first")
    obs = obs.merge(leaves, on=['pot', 'side', 'leaf'], how='left', validate='many_to_one')
    missing = obs['treatment'].isna() | obs['symptom_frac'].isna()
    if missing.any():
        logger.warning("%d observations without annotation or symptom measure dropped",
                       int(missing.sum()))
        obs = obs[~missing]
    return obs.sort_values(['pot', 'side', 'leaf', 'dpi']).reset_index(drop=True)


def visible_onset(obs: pd.DataFrame, level: float = ABNORMAL_LEVEL) -> pd.DataFrame:
    """
    First date each leaf looks abnormal, from the imagery.

    The dataset annotates infection confidence once per leaf, not symptom
    dates, and inoculation alone does not show that a leaf was symptom-free.
    An observation is visibly abnormal when its symptom_frac > `level`
    (absolute, not relative to controls: 'pre-symptomatic' must mean the leaf
    shows NO visible change, whatever its cause -- colour cannot tell
    Septoria lesions from tip necrosis or senescence). Onset = first date
    abnormal on that date and the next cohort date (a single date counts
    only on the last cohort date), so one noisy date does not end a leaf's
    pre-symptomatic period. Applied to controls too: their rate measures
    non-disease change (senescence, handling damage).

    Returns one row per leaf with onset_dpi (NaN = looked normal throughout);
    the per-observation flags are in .attrs['flags'].
    """
    flags = obs[['pot', 'side', 'leaf', 'dpi', 'infected', 'symptom_frac']].copy()
    flags['abnormal'] = flags['symptom_frac'] > level
    last_dpi = obs['dpi'].max()
    onsets = []
    for key, grp in flags.sort_values('dpi').groupby(['pot', 'side', 'leaf']):
        ab, dpis = grp['abnormal'].to_numpy(), grp['dpi'].to_numpy()
        onset = np.nan
        for i in range(len(ab)):
            if ab[i] and ((i + 1 < len(ab) and ab[i + 1]) or dpis[i] == last_dpi):
                onset = dpis[i]
                break
        onsets.append({**dict(zip(('pot', 'side', 'leaf'), key)), 'onset_dpi': onset,
                       'infected': bool(grp['infected'].iloc[0])})
    out = pd.DataFrame(onsets)
    out.attrs['flags'] = flags
    return out


def observation_state(obs: pd.DataFrame, onsets: pd.DataFrame) -> pd.Series:
    """
    Per observation: 0 = healthy (control leaf, or inoculated leaf before
    inoculation, dpi <= 0); 1 = inoculated and still visibly normal
    (pre-symptomatic); 2 = inoculated and visibly abnormal from its onset on.
    """
    o = obs.merge(onsets[['pot', 'side', 'leaf', 'onset_dpi']], on=['pot', 'side', 'leaf'],
                  how='left')
    state = np.where(~o['infected'] | (o['dpi'] <= 0), 0,
                     np.where(o['onset_dpi'].notna() & (o['dpi'] >= o['onset_dpi']), 2, 1))
    return pd.Series(state, index=obs.index, name='state')


def visibly_normal(obs: pd.DataFrame, onsets: pd.DataFrame) -> pd.Series:
    """True while a leaf (control or inoculated) is before its visible onset."""
    o = obs.merge(onsets[['pot', 'side', 'leaf', 'onset_dpi']], on=['pot', 'side', 'leaf'],
                  how='left')
    return pd.Series((o['onset_dpi'].isna() | (o['dpi'] < o['onset_dpi'])).to_numpy(),
                     index=obs.index, name='visibly_normal')


# Observation columns that are identifiers / annotations, not spectral features.
META_COLUMNS = {'pot', 'side', 'leaf', 'dpi', 'date', 'genotype', 'treatment', 'confidence',
                'infected', 'symptom_source', 'state', 'visibly_normal'}


def build_leaf_sequences(obs: pd.DataFrame, state: pd.Series, seq_len: int = 4,
                         normal: pd.Series | None = None) -> pd.DataFrame:
    """
    Chronological per-leaf windows of `seq_len` consecutive cohort dates, in
    the pipeline's long CSV format (one row per timestep, every feature
    column carried along). The label is the state at the window's LAST date
    (what a deployed model would be asked); site_id is the pot, so CV and
    splits never split a pot. `window_normal` = every observation in the
    window looks normal (see visibly_normal).
    """
    obs = obs.assign(state=state, visibly_normal=True if normal is None else normal)
    features = [c for c in obs.columns if c not in META_COLUMNS]
    rows = []
    for (pot, side, leaf), grp in obs.groupby(['pot', 'side', 'leaf']):
        grp = grp.sort_values('dpi')
        for end in range(seq_len - 1, len(grp)):
            win = grp.iloc[end - seq_len + 1:end + 1]
            sid = f'P{pot:02d}{side}L{leaf}_d{int(win["dpi"].iloc[-1]):+03d}'
            window_normal = bool(win['visibly_normal'].all())
            for t, (_, rec) in enumerate(win.iterrows()):
                rows.append({'sample_id': sid, 'timestamp': t, 'label': int(win['state'].iloc[-1]),
                             **{c: rec[c] for c in features},
                             'date': pd.Timestamp(rec['date']).strftime('%Y-%m-%d'),
                             'dpi': rec['dpi'], 'site_id': f'P{pot:02d}{side}',
                             'leaf_id': f'P{pot:02d}{side}L{leaf}',
                             'genotype': rec['genotype'], 'treatment': rec['treatment'],
                             'confidence': rec['confidence'], 'window_normal': window_normal})
    return pd.DataFrame(rows)


def fetch_cohort(pots=COHORT_POTS, side: str = COHORT_SIDE, dpis=COHORT_DPI,
                 sensors=('SWIR', 'VNIR'), workers: int = 6,
                 local_only: bool = False) -> pd.DataFrame:
    """
    Extract every image of the cohort (resumable: cached images are skipped).
    SWIR first (small), then VNIR; one archive is opened per date and camera,
    from ARCHIVE_DIR/<camera>/ when it was downloaded by hand, else remotely.
    Failures are logged and skipped, so one bad member does not stop a
    many-hour run; rerun to retry them. Returns a per-image status table.
    """
    fetch_annotations()
    index = archive_index(sensors=sensors)
    status = []
    for sensor in sensors:
        for dpi in dpis:
            rows = index[(index['sensor'] == sensor) & (index['dpi'] == dpi) &
                         (index['side'] == side) & (index['pot'].isin(pots)) &
                         (index['ext'] == 'hyspex')]
            todo = [r for r in rows.itertuples()
                    if not os.path.exists(image_cache(sensor, r.date, r.pot, side))]
            status += [{'sensor': sensor, 'dpi': dpi, 'pot': r.pot, 'status': 'cached'}
                       for r in rows.itertuples() if r not in todo]
            if todo and local_only and not local_archive(sensor, todo[0].archive):
                logger.info("%s %+d dpi: %s not in %s/%s, skipped (--local-only)", sensor, dpi,
                            todo[0].archive, ARCHIVE_DIR, sensor)
                status += [{'sensor': sensor, 'dpi': dpi, 'pot': r.pot, 'status': 'not local'}
                           for r in todo]
                continue
            if not todo:
                continue
            zf = None
            for r in todo:
                try:
                    zf = zf or open_archive(sensor, r.archive, int(r.file_id))
                    extract_image(index, sensor, r.date, r.pot, side, zf=zf, workers=workers)
                    status.append({'sensor': sensor, 'dpi': dpi, 'pot': r.pot, 'status': 'ok'})
                    logger.info("%s %+d dpi pot %d_%s extracted", sensor, dpi, r.pot, side)
                except Exception as exc:          # keep going; rerun retries the failures
                    logger.error("%s %+d dpi pot %d_%s failed: %s", sensor, dpi, r.pot, side,
                                 exc)
                    status.append({'sensor': sensor, 'dpi': dpi, 'pot': r.pot,
                                   'status': f'failed: {exc}'})
                    zf = None
    return pd.DataFrame(status)
