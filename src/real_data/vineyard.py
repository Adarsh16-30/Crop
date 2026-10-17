"""
Vineyard UAV Dataset (MicaSense RedEdge-P)
==========================================
Zenodo record 14936376: "A Dataset for Vineyard Disease Detection via
Multispectral Imaging". 30 capture sessions on 31 July 2024 in three
Lambrusco vineyards (varieties Marani, Ancellotta, Salamino), annotated per
session for Flavescence doree (FD) and Esca. Sessions 0000 and 0019 image
the reflectance calibration panel.

Why the raw TIFFs and not the processed PNGs
--------------------------------------------
The `processed/` PNGs are band-aligned but 8-bit and contrast-stretched per
image, so their values are not reflectance and ratios between bands are not
meaningful. The raw 16-bit TIFFs carry the full radiometric model in their
metadata, which is applied here (following MicaSense's published
imageprocessing pipeline):

    radiance = V(x, y) * R(y) * (DN - black_level) / 2^16 * a1 / (gain * t_exp)
    reflectance = pi * radiance / E_dls

V is the 2-D vignetting polynomial, R the row-readout gradient
1 / (1 + a2 * y / t_exp - a3 * y), (a1, a2, a3) the radiometric
calibration, and E_dls the Downwelling Light Sensor horizontal irradiance
(XMP value x 0.01 to W/m^2/nm for DLS2). DLS-based reflectance is an
approximation; the calibration-panel sessions are processed too, and their
centre reflectance is written to panel_qc.csv as a check.

Bands are aligned to NIR with an ECC homography on gradient images, a mask
keeps vegetation pixels (NDVI above a threshold, not saturated, not in deep
shadow), and each capture is reduced to robust statistics of the
sensor-independent features in src.real_data.feature_schema.

What the labels allow
---------------------
Labels are per session (a vine row segment), not per plant or pixel, and
all sessions are from one date. So: no temporal modelling, captures of a
session are not independent (CV is leave-one-session-out), and variety and
location are partly confounded with the labels (see label_table.csv).
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.real_data.common import HTTPRangeFile, http_session, open_remote_zip
from src.real_data.feature_schema import compute_canonical_features

logger = logging.getLogger(__name__)

ZENODO_RECORD = '14936376'
ZIP_URL = (f'https://zenodo.org/api/records/{ZENODO_RECORD}/files/'
           'vineyard_diseases_detection_via_multispectral_imaging.zip/content')
ZIP_ROOT = 'vineyard_diseases_detection_via_multispectral_imaging/'

# RedEdge-P file suffix -> band (checked against XMP Camera:BandName on load).
BAND_FILES = {1: 'blue', 2: 'green', 3: 'red', 4: 'nir', 5: 'red_edge', 6: 'panchro'}
SPECTRAL_BANDS = ['blue', 'green', 'red', 'red_edge', 'nir']
REFERENCE_BAND = 'nir'
DLS2_IRRADIANCE_SCALE = 0.01        # XMP irradiance -> W/m^2/nm
SATURATION_DN = 65000
MIN_NIR_REFLECTANCE = 0.05          # below this a vegetation pixel is in deep shadow
BORDER_FRACTION = 0.03              # cropped after alignment

RAW_TIF_RE = re.compile(r'raw/(\d{4})SET/\d{3}/IMG_(\d{4})_(\d)\.tif$', re.IGNORECASE)

FEATURE_INDICES = ['NDVI', 'GNDVI', 'NDRE', 'CI_red_edge', 'CI_green']
FEATURE_STATS = {'median': np.median,
                 'p10': lambda v: np.percentile(v, 10),
                 'p90': lambda v: np.percentile(v, 90),
                 'std': np.std}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _read_member_http(raw_file, info) -> bytes:
    """
    Read one stored or deflated ZIP member straight from an HTTPRangeFile:
    one small request for the local header, one for the data, instead of
    zipfile's many small buffered reads.
    """
    import zlib

    raw_file.seek(info.header_offset)
    header = raw_file.read(30)
    if header[:4] != b'PK\x03\x04':
        raise OSError(f"Bad local header for {info.filename}")
    name_len = int.from_bytes(header[26:28], 'little')
    extra_len = int.from_bytes(header[28:30], 'little')
    raw_file.seek(info.header_offset + 30 + name_len + extra_len)
    data = raw_file.read(info.compress_size)
    if info.compress_type == 8:
        data = zlib.decompress(data, -15)
    elif info.compress_type != 0:
        raise OSError(f"Unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise OSError(f"CRC mismatch for {info.filename}")
    return data


def fetch_vineyard(dest: str, sessions: list[str] | None = None,
                   bands: list[int] | None = None, zip_path: str | None = None,
                   max_captures: int | None = None, workers: int = 6) -> dict:
    """
    Extract annotation.json, the usage notebook and the raw TIFFs of the
    selected sessions / bands into `dest`, without downloading the whole
    4.8 GB archive (unless `zip_path` points at a local copy).

    Files already present with the right size are skipped, so the call
    resumes after an interruption.
    """
    import threading
    import zipfile

    bands = bands or [1, 2, 3, 4, 5]
    zf = zipfile.ZipFile(zip_path) if zip_path else open_remote_zip(ZIP_URL)
    infos = zf.infolist()
    wanted = []
    for info in infos:
        rel = info.filename[len(ZIP_ROOT):] if info.filename.startswith(ZIP_ROOT) else None
        if rel in ('annotation.json', 'usage_examples.ipynb'):
            wanted.append((info, os.path.join(dest, rel)))
    per_session: dict[str, set] = {}
    for info in infos:
        m = RAW_TIF_RE.search(info.filename)
        if not m:
            continue
        sess, cap, band = m.group(1), m.group(2), int(m.group(3))
        if (sessions and sess not in sessions) or band not in bands:
            continue
        caps = per_session.setdefault(sess, set())
        if max_captures and cap not in caps and len(caps) >= max_captures:
            continue
        caps.add(cap)
        wanted.append((info, os.path.join(dest, 'raw', f'{sess}SET', f'IMG_{cap}_{band}.tif')))

    # Round-robin over sessions (first capture of every session, then the
    # second, ...) so an interrupted or partial download still covers every
    # session and both classes.
    def order(job):
        m = RAW_TIF_RE.search(job[0].filename)
        if not m:
            return (-1, '', '', 0)
        rank = sorted(per_session[m.group(1)]).index(m.group(2))
        return (rank, m.group(1), m.group(2), int(m.group(3)))

    wanted.sort(key=order)
    todo = [(i, p) for i, p in wanted
            if not (os.path.exists(p) and os.path.getsize(p) == i.file_size)]
    total_mb = sum(i.file_size for i, _ in todo) / 1e6
    logger.info("Vineyard: %d files selected, %d to fetch (%.0f MB)", len(wanted), len(todo),
                total_mb)

    local = threading.local()

    def fetch(job):
        info, path = job
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if zip_path:
            data = zf.read(info)
        else:
            # one range handle per worker thread
            if not hasattr(local, 'raw'):
                local.raw = HTTPRangeFile(ZIP_URL, session=http_session())
            data = _read_member_http(local.raw, info)
        tmp = path + '.part'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
        return info.file_size

    done = 0
    if zip_path or workers <= 1:
        for job in todo:
            done += fetch(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, size in enumerate(pool.map(fetch, todo), 1):
                done += size
                if i % 50 == 0 or i == len(todo):
                    logger.info("  %d/%d files (%.0f MB)", i, len(todo), done / 1e6)
    return {'selected': len(wanted), 'fetched': len(todo), 'megabytes': done / 1e6}


# ---------------------------------------------------------------------------
# Radiometric calibration
# ---------------------------------------------------------------------------

def _xmp_value(xmp: str, tag: str) -> str | None:
    m = re.search(rf'<{tag}>\s*(?:<rdf:Seq>\s*)?(.*?)\s*(?:</rdf:Seq>\s*)?</{tag}>', xmp, re.S)
    if not m:
        return None
    items = re.findall(r'<rdf:li>(.*?)</rdf:li>', m.group(1), re.S)
    return ','.join(items) if items else m.group(1).strip()


def _floats(text: str | None) -> list[float]:
    return [float(v) for v in re.split(r'[,\s]+', text.strip()) if v] if text else []


def parse_metadata(tiff_bytes_or_path) -> dict:
    """Radiometric metadata of one MicaSense TIFF (XMP + EXIF + TIFF tags)."""
    from PIL import Image

    if isinstance(tiff_bytes_or_path, (bytes, bytearray)):
        import io
        raw = bytes(tiff_bytes_or_path)
        img = Image.open(io.BytesIO(raw))
    else:
        with open(tiff_bytes_or_path, 'rb') as f:
            raw = f.read()
        img = Image.open(tiff_bytes_or_path)
    start, end = raw.find(b'<x:xmpmeta'), raw.find(b'</x:xmpmeta>')
    if start < 0 or end < 0:
        raise ValueError("No XMP metadata: not a MicaSense TIFF")
    xmp = raw[start:end + 12].decode('utf-8', 'replace')

    exif = img.getexif().get_ifd(0x8769)
    tags = img.tag_v2
    black = tags.get(50714)
    meta = {
        'band_name': (_xmp_value(xmp, 'Camera:BandName') or '').strip(),
        'wavelength_nm': float(_xmp_value(xmp, 'Camera:CentralWavelength') or 'nan'),
        'radiometric_cal': _floats(_xmp_value(xmp, 'MicaSense:RadiometricCalibration')),
        'vignette_poly2d': _floats(_xmp_value(xmp, 'Camera:VignettingPolynomial2D')),
        'vignette_exponents': [int(v) for v in
                               _floats(_xmp_value(xmp, 'Camera:VignettingPolynomial2DName'))],
        'vignette_center': _floats(_xmp_value(xmp, 'Camera:VignettingCenter')),
        'vignette_poly': _floats(_xmp_value(xmp, 'Camera:VignettingPolynomial')),
        'black_level': float(np.mean(black)) if black else
        float(np.mean(_floats(_xmp_value(xmp, 'MicaSense:DarkRowValue')) or [0])),
        'exposure_s': float(exif.get(0x829A)),
        'gain': float(exif.get(0x8833, 100)) / 100.0,
        'bits': int((tags.get(258) or (16,))[0]),
        'horizontal_irradiance': float(_xmp_value(xmp, 'DLS:HorizontalIrradiance') or 'nan'),
        'spectral_irradiance': float(_xmp_value(xmp, 'DLS:SpectralIrradiance') or 'nan'),
        'capture_id': (_xmp_value(xmp, 'MicaSense:CaptureId') or '').strip(),
    }
    if len(meta['radiometric_cal']) != 3:
        raise ValueError("Missing MicaSense:RadiometricCalibration")
    return meta


def vignette_map(meta: dict, shape: tuple[int, int]) -> np.ndarray:
    """Multiplicative vignetting correction V (>= ~1 towards the corners)."""
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    coeffs, exps = meta['vignette_poly2d'], meta['vignette_exponents']
    if coeffs and len(exps) == 2 * len(coeffs):
        xv, yv = x / cols, y / rows
        poly = np.zeros(shape)
        for i, c in enumerate(coeffs):
            poly += c * xv ** exps[2 * i] * yv ** exps[2 * i + 1]
        return 1.0 / poly
    if meta['vignette_center'] and meta['vignette_poly']:
        cx, cy = meta['vignette_center']
        r = np.hypot(x - cx, y - cy)
        k = list(reversed(meta['vignette_poly'])) + [1.0]
        return 1.0 / np.polyval(k, r)
    return np.ones(shape)


def dn_to_radiance(dn: np.ndarray, meta: dict) -> np.ndarray:
    """Spectral radiance, W / m^2 / sr / nm."""
    a1, a2, a3 = meta['radiometric_cal']
    te = meta['exposure_s']
    rows = np.arange(dn.shape[0], dtype=np.float64)[:, None]
    row_gradient = 1.0 / (1.0 + a2 * rows / te - a3 * rows)
    signal = np.clip(dn.astype(np.float64) - meta['black_level'], 0, None)
    L = vignette_map(meta, dn.shape) * row_gradient * signal
    return L / (2.0 ** meta['bits']) * a1 / (meta['gain'] * te)


def dls_irradiance(meta: dict) -> float:
    """DLS2 horizontal irradiance in W/m^2/nm (falls back to the spectral value)."""
    e = meta['horizontal_irradiance']
    if not np.isfinite(e) or e <= 0:
        e = meta['spectral_irradiance']
    return e * DLS2_IRRADIANCE_SCALE


def dn_to_reflectance(dn: np.ndarray, meta: dict) -> np.ndarray:
    e = dls_irradiance(meta)
    if not np.isfinite(e) or e <= 0:
        raise ValueError("No DLS irradiance in metadata")
    return np.pi * dn_to_radiance(dn, meta) / e


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _gradient(img: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag / (mag.max() + 1e-12)


def align_to_reference(images: dict[str, np.ndarray], reference: str = REFERENCE_BAND,
                       scale: float = 0.5) -> tuple[dict, np.ndarray, dict]:
    """
    Warp every band onto `reference` with an ECC homography estimated on
    downscaled gradient images. Returns (aligned images, mask of pixels
    covered by every band, per-band success flags).
    """
    import cv2

    ref = images[reference]
    h, w = ref.shape
    small = (int(w * scale), int(h * scale))
    ref_g = _gradient(cv2.resize(ref, small, interpolation=cv2.INTER_AREA))
    S = np.diag([scale, scale, 1.0])
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    aligned, ok = {reference: ref}, {reference: True}
    coverage = np.ones((h, w), dtype=np.float32)
    for band, img in images.items():
        if band == reference:
            continue
        warp = np.eye(3, dtype=np.float32)
        try:
            img_g = _gradient(cv2.resize(img, small, interpolation=cv2.INTER_AREA))
            _, warp = cv2.findTransformECC(ref_g, img_g, warp, cv2.MOTION_HOMOGRAPHY,
                                           criteria, None, 5)
            ok[band] = True
        except cv2.error:
            warp = np.eye(3, dtype=np.float32)
            ok[band] = False
        H = (np.linalg.inv(S) @ warp.astype(np.float64) @ S)
        flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        aligned[band] = cv2.warpPerspective(img.astype(np.float32), H, (w, h), flags=flags,
                                            borderValue=np.nan)
        coverage *= cv2.warpPerspective(np.ones((h, w), np.float32), H, (w, h), flags=flags,
                                        borderValue=0)
    mask = coverage > 0.999
    bh, bw = int(h * BORDER_FRACTION), int(w * BORDER_FRACTION)
    mask[:bh], mask[-bh:], mask[:, :bw], mask[:, -bw:] = False, False, False, False
    return aligned, mask, ok


# ---------------------------------------------------------------------------
# Per-capture features
# ---------------------------------------------------------------------------

def load_capture(paths: dict[int, str]) -> tuple[dict, dict, dict]:
    """Reflectance image, metadata and saturation mask per band for one capture."""
    from PIL import Image

    refl, metas, saturated = {}, {}, {}
    for num, path in sorted(paths.items()):
        expected = BAND_FILES[num]
        if expected not in SPECTRAL_BANDS:
            continue
        meta = parse_metadata(path)
        name = meta['band_name'].lower().replace(' ', '_')
        if name and name != expected:
            raise ValueError(f"{path}: XMP band {meta['band_name']!r}, expected {expected!r}")
        dn = np.array(Image.open(path), dtype=np.float64)
        refl[expected] = dn_to_reflectance(dn, meta).astype(np.float32)
        metas[expected] = meta
        saturated[expected] = dn >= SATURATION_DN
    return refl, metas, saturated


def capture_features(paths: dict[int, str], ndvi_threshold: float = 0.4,
                     band_factors: dict[str, float] | None = None) -> dict:
    """
    Vegetation-masked feature statistics of one capture. `band_factors`
    (from panel_factors) rescale each band's reflectance.
    """
    import cv2

    refl, metas, saturated = load_capture(paths)
    missing = [b for b in SPECTRAL_BANDS if b not in refl]
    if missing:
        raise ValueError(f"capture is missing bands {missing}")
    if band_factors:
        refl = {b: img * np.float32(band_factors.get(b, 1.0)) for b, img in refl.items()}
    sat = np.zeros(refl[REFERENCE_BAND].shape, dtype=np.float32)
    for s in saturated.values():
        sat = np.maximum(sat, s.astype(np.float32))
    images = dict(refl)
    aligned, covered, ok = align_to_reference(images)
    # Warp the saturation flags with the image data: approximate but conservative
    # (any band saturated anywhere nearby removes the pixel).
    sat = cv2.dilate(sat, np.ones((5, 5), np.uint8)) > 0

    nir, red = aligned['nir'], aligned['red']
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = (nir - red) / (nir + red)
    finite = np.all([np.isfinite(aligned[b]) for b in SPECTRAL_BANDS], axis=0)
    usable = covered & finite & ~sat
    veg = usable & (ndvi > ndvi_threshold) & (nir > MIN_NIR_REFLECTANCE)

    row = {
        'n_pixels': int(usable.sum()),
        'n_vegetation': int(veg.sum()),
        'vegetation_fraction': float(veg.sum() / max(usable.sum(), 1)),
        'aligned_all': bool(all(ok.values())),
        **{f'irradiance_{b}': dls_irradiance(metas[b]) for b in SPECTRAL_BANDS},
        **{f'exposure_ms_{b}': metas[b]['exposure_s'] * 1e3 for b in SPECTRAL_BANDS},
    }
    if veg.sum() < 500:
        return row
    pix = {b: aligned[b][veg].astype(np.float64) for b in SPECTRAL_BANDS}
    feats = compute_canonical_features(pix, 'micasense_rededge_p', indices=FEATURE_INDICES)
    for col in feats.columns:
        values = feats[col].to_numpy()
        values = values[np.isfinite(values)]
        for stat, func in FEATURE_STATS.items():
            row[f'{col}_{stat}'] = float(func(values)) if len(values) else np.nan
    return row


def _local_cv(img: np.ndarray, k: int = 15) -> np.ndarray:
    import cv2

    mean = cv2.blur(img, (k, k))
    sq = cv2.blur(img * img, (k, k))
    return np.sqrt(np.maximum(sq - mean * mean, 0)) / np.maximum(mean, 1e-3)


def detect_panel(aligned: dict[str, np.ndarray], covered: np.ndarray,
                 min_area: int = 2000) -> np.ndarray | None:
    """
    Mask of the calibration panel: the largest region that is spectrally
    flat (grey: max/min band ratio < 1.6, every band > 0.15) and locally
    uniform in every band (grass is textured, the QR code is black/white).
    """
    import cv2

    stack = np.stack([np.nan_to_num(aligned[b]).astype(np.float32) for b in SPECTRAL_BANDS])
    uniform = np.all([_local_cv(s) < 0.04 for s in stack], axis=0)
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    flat = (lo > 0.15) & (hi / np.maximum(lo, 1e-6) < 1.6) & covered
    n, labels, stats, _ = cv2.connectedComponentsWithStats((uniform & flat).astype(np.uint8))
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    return cv2.erode((labels == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def panel_reflectance(paths: dict[int, str]) -> dict:
    """DLS-based reflectance of the calibration panel in a panel capture."""
    refl, _, _ = load_capture(paths)
    aligned, covered, _ = align_to_reference(refl)
    mask = detect_panel(aligned, covered)
    out = {'panel_pixels': int(mask.sum()) if mask is not None else 0}
    for band in SPECTRAL_BANDS:
        out[f'panel_{band}'] = float(np.median(aligned[band][mask])) if mask is not None \
            else np.nan
    return out


def panel_factors(panels: pd.DataFrame, nominal_albedo: float) -> pd.DataFrame:
    """
    Per panel session and band: nominal_albedo / median measured panel
    reflectance. Multiplying DLS-based reflectance by this factor corrects
    the DLS bias while the DLS still tracks illumination changes between
    captures.
    """
    cols = [f'panel_{b}' for b in SPECTRAL_BANDS]
    med = panels.groupby('session')[cols].median()
    factors = nominal_albedo / med
    factors.columns = SPECTRAL_BANDS
    return factors


def load_annotations(vineyard_dir: str) -> pd.DataFrame:
    with open(os.path.join(vineyard_dir, 'annotation.json')) as f:
        ann = json.load(f)
    rows = []
    for sess, a in ann.items():
        d = a['date']
        rows.append({'session': sess, 'type': a['type'], 'FD': bool(a['FD']),
                     'ESCA': bool(a['ESCA']), 'annotated_captures': a['captures'],
                     'date': f"{d['year']:04d}-{d['month']:02d}-{d['day']:02d}",
                     'lat': a['gpsCoordinate']['lat'], 'lon': a['gpsCoordinate']['lon']})
    return pd.DataFrame(rows)


def list_captures(vineyard_dir: str) -> dict[tuple[str, str], dict[int, str]]:
    captures: dict[tuple[str, str], dict[int, str]] = {}
    raw_dir = os.path.join(vineyard_dir, 'raw')
    if not os.path.isdir(raw_dir):
        return captures
    for sess_dir in sorted(os.listdir(raw_dir)):
        m = re.match(r'(\d{4})SET$', sess_dir)
        if not m:
            continue
        for name in sorted(os.listdir(os.path.join(raw_dir, sess_dir))):
            fm = re.match(r'IMG_(\d{4})_(\d)\.tif$', name, re.IGNORECASE)
            if fm:
                captures.setdefault((m.group(1), fm.group(1)), {})[int(fm.group(2))] = \
                    os.path.join(raw_dir, sess_dir, name)
    return captures


def assign_panels(sessions: list[str], ann: pd.DataFrame,
                  panel_sessions: list[str]) -> dict[str, str | None]:
    """
    Each session -> the panel session to correct it with: the latest panel
    session at or before it taken at the same GPS location, else the
    nearest panel session at that location, else None.
    """
    out = {}
    for s in sessions:
        here = [p for p in panel_sessions
                if (ann.at[p, 'lat'], ann.at[p, 'lon']) == (ann.at[s, 'lat'], ann.at[s, 'lon'])]
        if not here:
            # Ancellotta and Salamino rows are ~100 m apart and share panel 0019.
            here = sorted(panel_sessions, key=lambda p: abs(ann.at[p, 'lat'] - ann.at[s, 'lat'])
                          + abs(ann.at[p, 'lon'] - ann.at[s, 'lon']))[:1]
        before = [p for p in here if p <= s]
        out[s] = max(before) if before else (min(here) if here else None)
    return out


def extract_features(vineyard_dir: str, ndvi_threshold: float = 0.4, workers: int = 4,
                     panel_albedo: float | None = 0.5) -> pd.DataFrame:
    """
    Features for every downloaded capture, joined with the session labels.

    Panel captures are processed first. With `panel_albedo` set, each
    session's reflectance is rescaled band by band so that its site's panel
    reads `panel_albedo` (replace 0.5 with your panel's calibration values
    if known); with None, plain DLS reflectance is used.

    Writes capture_features.csv and panel_qc.csv into `vineyard_dir`.
    """
    ann = load_annotations(vineyard_dir).set_index('session')
    captures = list_captures(vineyard_dir)
    if not captures:
        raise FileNotFoundError(f"No raw TIFFs under {vineyard_dir}/raw; run vineyard-fetch")
    is_panel = {s: ann.at[s, 'type'] == 'calibration_panel' for s, _ in captures}

    def run(func, items):
        def one(item):
            (sess, cap), paths = item
            try:
                return {'session': sess, 'capture': cap, **func(sess, paths)}
            except Exception as exc:
                logger.warning("Capture %s/%s skipped: %s", sess, cap, exc)
                return {'session': sess, 'capture': cap, 'error': str(exc)}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(one, items))

    items = sorted(captures.items())
    panel_rows = run(lambda s, p: panel_reflectance(p), [i for i in items if is_panel[i[0][0]]])
    panels = pd.DataFrame(panel_rows)
    factors = pd.DataFrame()
    if len(panels) and 'panel_nir' in panels:
        panels.to_csv(os.path.join(vineyard_dir, 'panel_qc.csv'), index=False)
        logger.info("Calibration panel, DLS-based median reflectance per session:\n%s",
                    panels.groupby('session')[[f'panel_{b}' for b in SPECTRAL_BANDS]]
                    .median().round(3).to_string())
        if panel_albedo:
            factors = panel_factors(panels.dropna(subset=['panel_nir']), panel_albedo)
    vine_sessions = sorted({s for (s, _), _ in items if not is_panel[s]})
    mapping = assign_panels(vine_sessions, ann, list(factors.index)) if len(factors) else {}
    if panel_albedo and not mapping:
        logger.warning("No usable panel captures: using uncorrected DLS reflectance")

    def vine(sess, paths):
        panel = mapping.get(sess)
        bf = factors.loc[panel].to_dict() if panel is not None else None
        return {'panel_session': panel, **capture_features(paths, ndvi_threshold, bf)}

    rows = run(vine, [i for i in items if not is_panel[i[0][0]]])
    feats = pd.DataFrame(rows).merge(ann.reset_index(), on='session', how='left')
    feats = feats.dropna(axis=1, how='all')
    feats.to_csv(os.path.join(vineyard_dir, 'capture_features.csv'), index=False)
    n_ok = int(feats.filter(like='NDVI_median').notna().any(axis=1).sum())
    logger.info("Vineyard features: %d captures in %d sessions (%d with enough vegetation)",
                len(feats), feats['session'].nunique(), n_ok)
    return feats


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

TASKS = {
    # task -> (function(row) -> label or None, class names)
    'binary': (lambda r: int(r.FD or r.ESCA), ['Healthy', 'Diseased']),
    'multiclass': (lambda r: None if (r.FD and r.ESCA) else (1 if r.FD else 2 if r.ESCA else 0),
                   ['Healthy', 'FD', 'Esca']),
    'fd': (lambda r: None if r.ESCA else int(r.FD), ['Healthy', 'FD']),
    'esca': (lambda r: None if r.FD else int(r.ESCA), ['Healthy', 'Esca']),
}


def task_labels(df: pd.DataFrame, task: str) -> pd.Series:
    func, _ = TASKS[task]
    return df.apply(lambda r: func(r), axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    stats = tuple(f'_{s}' for s in FEATURE_STATS)
    return [c for c in df.columns if c.endswith(stats)] + ['vegetation_fraction']


def _models(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        'Logistic Regression': make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000)),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                                class_weight='balanced', random_state=seed,
                                                n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                 verbosity=0),
    }


def leave_session_out(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold class probabilities with whole sessions held out."""
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut

    from src.models.classical_models import balanced_sample_weights

    n_classes = len(np.unique(y))
    proba = np.zeros((len(y), n_classes))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model)
        if len(np.unique(y[tr])) < n_classes:
            proba[te] = np.bincount(y[tr], minlength=n_classes) / len(tr)
            continue
        if type(m).__name__ == 'XGBClassifier':
            m.fit(X[tr], y[tr], sample_weight=balanced_sample_weights(y[tr]))
        else:
            m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
    return proba


def score(y: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    pred = proba.argmax(axis=1)
    out = {'n': int(len(y)), 'balanced_accuracy': balanced_accuracy_score(y, pred),
           'f1_macro': f1_score(y, pred, average='macro', zero_division=0)}
    try:
        out['roc_auc'] = (roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2 else
                          roc_auc_score(y, proba, multi_class='ovr', average='macro'))
    except ValueError:
        out['roc_auc'] = np.nan
    return out


def session_level(df_pred: pd.DataFrame, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean out-of-fold probability per session (one vote per session)."""
    cols = [f'prob_{k}' for k in range(n_classes)]
    agg = df_pred.groupby('session').agg({**{c: 'mean' for c in cols}, 'label': 'first'})
    return agg['label'].to_numpy(dtype=int), agg[cols].to_numpy()


def univariate_tests(df: pd.DataFrame, y: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """
    Session-level Mann-Whitney U per feature (binary tasks): session medians
    are compared so that sessions with many captures do not dominate.
    """
    from scipy.stats import mannwhitneyu

    sess = df.assign(_y=y).groupby('session').agg({**{c: 'median' for c in cols}, '_y': 'first'})
    rows = []
    for c in cols:
        a = sess.loc[sess['_y'] == 1, c].dropna()
        b = sess.loc[sess['_y'] == 0, c].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        rows.append({'feature': c, 'median_diseased': a.median(), 'median_healthy': b.median(),
                     'rank_biserial': 1 - 2 * u / (len(a) * len(b)), 'p_value': p,
                     'n_sessions_diseased': len(a), 'n_sessions_healthy': len(b)})
    out = pd.DataFrame(rows).sort_values('p_value')
    if len(out):
        # Benjamini-Hochberg across features
        m = len(out)
        ranked = out['p_value'].to_numpy() * m / np.arange(1, m + 1)
        out['q_value_bh'] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return out


def run_experiment(features: pd.DataFrame, output_dir: str, task: str = 'binary',
                   n_permutations: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Leave-one-session-out classification with three models, a session-level
    label-permutation test, per-feature tests and the label/variety table.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = features.copy()
    df['label'] = task_labels(df, task)
    df = df.dropna(subset=['label'])
    cols = [c for c in feature_columns(df) if c in df.columns]
    df = df.dropna(subset=cols)
    df['label'] = df['label'].astype(int)
    _, class_names = TASKS[task]

    label_table = pd.crosstab([df['type'], df['session']], df['label'].map(dict(enumerate(
        class_names)))).reset_index()
    label_table.to_csv(os.path.join(output_dir, f'label_table_{task}.csv'), index=False)

    X, y, groups = df[cols].to_numpy(), df['label'].to_numpy(), df['session'].to_numpy()
    n_classes = len(class_names)
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        raise ValueError("Need at least two classes and three sessions; fetch more sessions")
    logger.info("Vineyard %s: %d captures, %d sessions, %d features | class counts %s",
                task, len(y), len(np.unique(groups)), len(cols),
                dict(zip(class_names, np.bincount(y, minlength=n_classes))))

    rng = np.random.default_rng(seed)
    sess_labels = df.groupby('session')['label'].first()
    rows, preds = [], []
    for name, model in _models(seed).items():
        proba = leave_session_out(model, X, y, groups)
        pred_df = pd.DataFrame(proba, columns=[f'prob_{k}' for k in range(n_classes)])
        pred_df.insert(0, 'session', groups)
        pred_df.insert(1, 'capture', df['capture'].to_numpy())
        pred_df['label'] = y
        pred_df['model'] = name
        preds.append(pred_df)

        cap = score(y, proba)
        ys, ps = session_level(pred_df, n_classes)
        ses = score(ys, ps)
        row = {'task': task, 'model': name,
               **{f'capture_{k}': v for k, v in cap.items()},
               **{f'session_{k}': v for k, v in ses.items()}}

        if n_permutations and name in ('Logistic Regression', 'Random Forest'):
            null = []
            for _ in range(n_permutations):
                shuffled = dict(zip(sess_labels.index, rng.permutation(sess_labels.to_numpy())))
                y_perm = np.array([shuffled[s] for s in groups])
                p_perm = leave_session_out(model, X, y_perm, groups)
                null.append(score(y_perm, p_perm)['balanced_accuracy'])
            null = np.array(null)
            row['perm_null_mean'] = float(null.mean())
            row['perm_null_p95'] = float(np.quantile(null, 0.95))
            row['perm_p_value'] = float((1 + (null >= cap['balanced_accuracy']).sum()) /
                                        (1 + len(null)))
        rows.append(row)
        logger.info("  %-20s capture bal.acc %.3f AUC %.3f | session bal.acc %.3f AUC %.3f%s",
                    name, cap['balanced_accuracy'], cap['roc_auc'], ses['balanced_accuracy'],
                    ses['roc_auc'],
                    f" | permutation p={row['perm_p_value']:.3f}" if 'perm_p_value' in row
                    else '')

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(output_dir, f'cv_metrics_{task}.csv'), index=False)
    pd.concat(preds).to_csv(os.path.join(output_dir, f'predictions_{task}.csv'), index=False)
    if n_classes == 2:
        univariate_tests(df, y, cols).to_csv(
            os.path.join(output_dir, f'feature_tests_{task}.csv'), index=False)
    return metrics
