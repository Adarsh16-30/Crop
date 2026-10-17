"""
Septoria hyperspectral dataset: annotation parsing on the real tables,
cube reading / white reference / mask orientation on synthetic cubes,
the symptom-onset rule, labelled sequences, the parallel ZIP member reader
and the experiment helpers. No network.
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.real_data import septoria as sep

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'septoria')


# ---------------------------------------------------------------------------
# Annotations (real tables)
# ---------------------------------------------------------------------------

def test_leaf_table_matches_dataset_description():
    leaves = sep.load_leaves(os.path.join(FIX, 'Data_description.csv'))
    assert len(leaves) == 279
    assert set(leaves['treatment']) == {'Infected', 'Non-infected'}
    assert set(leaves['confidence']) == {1, 2}
    assert leaves.duplicated(['pot', 'side', 'leaf']).sum() == 0
    assert leaves['infected'].sum() == 139


@pytest.mark.parametrize('sensor', ['VNIR', 'SWIR'])
def test_acquisition_tables_match_published_totals(sensor):
    """'a total of 1175 images provided information on 3326 leaves' (dataset description)."""
    acq = sep.load_acquisitions(sensor, os.path.join(FIX, f'{sensor}_data.csv'))
    assert acq['dpi'].nunique() == 18 and acq['dpi'].min() == -1 and acq['dpi'].max() == 20
    assert acq['n_leaves'].notna().sum() == 1175
    assert int(acq['n_leaves'].sum()) == 3326


def test_dpi_of_archive_dates():
    assert sep.dpi_of('20210311') == -1
    assert sep.dpi_of('20210312') == 0
    assert sep.dpi_of('20210401') == 20


def test_member_names_including_split_archive():
    m = sep.MEMBER_RE.match('20210401_part1/13_G/13_G_leaf2.csv')
    assert m and m['date'] == '20210401' and m['pot'] == '13' and m['leaf'] == '2'
    assert sep.MEMBER_RE.match('20210331/37_D/37_D.hyspex')['ext'] == 'hyspex'
    assert sep.MEMBER_RE.match('20210331/37_D/38_D.hyspex') is None   # pot mismatch


# ---------------------------------------------------------------------------
# Cubes (synthetic)
# ---------------------------------------------------------------------------

LINES, BANDS, SAMPLES = 60, 4, 40
WAVELENGTHS = np.array([700.0, 800.0, 1250.0, 1650.0])


def synthetic_image(mirrored: bool):
    """Panel on lines 2-11, one off-centre leaf on lines 30-33 / samples 3-25 (so its mirror
    image is board), black board elsewhere."""
    response = np.linspace(8000, 12000, SAMPLES)               # lens fall-off across samples
    cube = np.zeros((LINES, BANDS, SAMPLES))
    cube[:, :, :] = 0.03 * response
    cube[2:12, :, :] = 1.0 * response
    cube[30:34, :, 3:26] = 0.5 * response[3:26]
    header = (f"ENVI\nlines = {LINES}\nbands = {BANDS}\nsamples = {SAMPLES}\n"
              "header offset = 16\ndata type = 12\ninterleave  = bil\n"
              "data ignore value = 4095\nwavelength = {700, 800, 1250, 1650}\n")
    raw = b'\0' * 16 + cube.astype('<u2').tobytes()
    lines, samples = np.meshgrid(np.arange(30, 34), np.arange(3, 26), indexing='ij')
    cols = (SAMPLES - samples) if mirrored else (samples + 1)
    mask = np.column_stack([lines.ravel() + 1, cols.ravel()])
    return header, raw, mask


def test_header_and_cube_reading():
    header, raw, _ = synthetic_image(False)
    hdr = sep.parse_hdr(header)
    assert (hdr['lines'], hdr['bands'], hdr['samples'], hdr['offset']) == (60, 4, 40, 16)
    assert hdr['saturation'] == 65535            # 'data ignore value' is not saturation
    np.testing.assert_array_equal(hdr['wavelength'], WAVELENGTHS)
    cube = sep.read_cube(raw, hdr)
    assert cube.shape == (LINES, BANDS, SAMPLES)


def test_white_reference_is_per_column_flat_field():
    header, raw, _ = synthetic_image(False)
    hdr = sep.parse_hdr(header)
    cube = sep.read_cube(raw, hdr)
    rows = sep.panel_lines(cube, band=1)
    assert rows.min() >= 2 and rows.max() <= 11
    white = sep.white_reference(cube, hdr, 'VNIR')
    line, sample = np.full(23, 31), np.arange(3, 26)
    refl = sep.leaf_reflectance(cube, white, line, sample, hdr['saturation'])
    np.testing.assert_allclose(refl, 0.5, atol=2e-3)       # fall-off removed


def test_no_panel_raises():
    cube = np.full((LINES, BANDS, SAMPLES), 100, dtype=np.uint16)
    cube[::7] = 5000                                     # isolated bright lines, no panel run
    with pytest.raises(ValueError, match='panel'):
        sep.panel_lines(cube, band=1)


@pytest.mark.parametrize('mirrored', [False, True])
def test_mask_orientation_detected_per_image(mirrored):
    header, raw, mask = synthetic_image(mirrored)
    hdr = sep.parse_hdr(header)
    cube = sep.read_cube(raw, hdr)
    white = sep.white_reference(cube, hdr, 'VNIR')
    found, fraction = sep.mask_orientation(cube, white, hdr, [mask], 'VNIR')
    assert found is mirrored and fraction > 0.99
    line, sample = sep.leaf_pixels(mask, hdr, found)
    assert line.min() == 30 and sample.min() == 3 and sample.max() == 25


def test_mask_that_fits_no_orientation_raises():
    header, raw, _ = synthetic_image(False)
    hdr = sep.parse_hdr(header)
    cube = sep.read_cube(raw, hdr)
    white = sep.white_reference(cube, hdr, 'VNIR')
    off_leaf = np.column_stack([np.full(20, 51), np.arange(1, 21)])   # line 50: board
    with pytest.raises(ValueError, match='do not fit'):
        sep.mask_orientation(cube, white, hdr, [off_leaf], 'VNIR')


def test_leaf_features_project_bands_by_camera():
    wl = np.arange(400.0, 1000.0, 5.0)
    refl = np.tile(np.where(wl < 700, 0.05, 0.5), (100, 1))
    refl[:, (wl >= 530) & (wl <= 570)] = 0.15               # green peak above red
    refl[:10, (wl > 640) & (wl < 700)] = 0.3                 # 10% chlorotic pixels
    feats = sep.leaf_features(refl, wl, 'VNIR')
    assert set(sep.PROJECT_BANDS) - set(feats) == {'swir_1610', 'swir_2190'}
    assert feats['nir_842'] == pytest.approx(0.5)
    assert feats['chlorotic_frac'] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Symptom onset and sequences
# ---------------------------------------------------------------------------

def observations():
    dpis = [-1, 4, 7, 10, 12, 13, 14, 15]
    chlor = {  # leaf -> chlorotic fraction per date
        ('ctrl', 1): [0.01, 0.02, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05],
        ('ctrl', 2): [0.02, 0.01, 0.02, 0.01, 0.02, 0.04, 0.05, 0.06],
        ('ctrl', 3): [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04],
        ('inoc', 1): [0.01, 0.02, 0.02, 0.03, 0.04, 0.20, 0.30, 0.40],   # onset 13
        ('inoc', 2): [0.01, 0.02, 0.02, 0.03, 0.15, 0.04, 0.05, 0.06],   # one blip: no onset
        ('inoc', 3): [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.30],   # last date only: 15
    }
    rows = []
    for (kind, leaf), values in chlor.items():
        pot = 40 if kind == 'ctrl' else 37
        for dpi, c in zip(dpis, values):
            rows.append({'pot': pot, 'side': 'G', 'leaf': leaf, 'dpi': dpi,
                         'date': (sep.INOCULATION_DATE + pd.Timedelta(days=dpi)).strftime('%Y%m%d'),
                         'symptom_frac': c, 'genotype': 'EPO_68',
                         'treatment': 'Infected' if kind == 'inoc' else 'Non-infected',
                         'infected': kind == 'inoc', 'confidence': 1,
                         **{b: 0.3 for b in sep.PROJECT_BANDS}})
    return pd.DataFrame(rows)


def test_visible_onset_rule():
    """Absolute rule (symptom_frac > 0.05), persistent for two dates or on the last date."""
    obs = observations()
    onsets = sep.visible_onset(obs).set_index(['pot', 'leaf'])['onset_dpi']
    assert onsets[(37, 1)] == 13
    assert onsets[(37, 2)] == 15        # the 12-dpi blip alone is not an onset; 15 is last
    assert onsets[(37, 3)] == 14
    # applied to controls as well: ctrl 2 crosses the level on the last date
    assert np.isnan(onsets[(40, 1)]) and onsets[(40, 2)] == 15 and np.isnan(onsets[(40, 3)])


def test_states_windows_and_normal_flags():
    obs = observations()
    onsets = sep.visible_onset(obs)
    state = sep.observation_state(obs, onsets)
    normal = sep.visibly_normal(obs, onsets)
    by = obs.assign(state=state).set_index(['pot', 'leaf', 'dpi'])['state']
    assert by[(40, 2, 15)] == 0                    # control stays healthy-class even if abnormal
    assert by[(37, 1, -1)] == 0                    # before inoculation
    assert by[(37, 1, 12)] == 1 and by[(37, 1, 13)] == 2
    nm = obs.assign(normal=normal).set_index(['pot', 'leaf', 'dpi'])['normal']
    assert nm[(40, 2, 14)] and not nm[(40, 2, 15)] and not nm[(37, 1, 14)]

    seqs = sep.build_leaf_sequences(obs, state, seq_len=4, normal=normal)
    per_window = seqs.groupby('sample_id')
    assert per_window.size().eq(4).all()
    assert per_window.ngroups == 6 * (8 - 4 + 1)
    last = per_window.last()
    assert (last['label'] == last.index.map(
        lambda sid: by[(int(sid[1:3]), int(sid[5]), int(sid.split('_d')[1]))])).all()
    assert set(seqs['site_id']) == {'P37G', 'P40G'}          # grouping unit is the pot
    assert 'symptom_frac' in seqs and 'red_edge_705' in seqs  # features carried along
    assert not last.loc['P40GL2_d+15', 'window_normal']
    assert last.loc['P40GL2_d+14', 'window_normal']


def test_preview_features_orientation_and_chlorosis():
    from PIL import Image

    rgb = np.zeros((60, 80, 3), dtype=np.uint8)              # rows = samples, cols = lines
    rgb[5:40, 20:24] = (60, 150, 60)                         # green leaf, off-centre
    rgb[5:12, 20:24] = (180, 160, 60)                        # yellow tip = 7 of 35 rows
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    lines, samples = np.meshgrid(np.arange(20, 24), np.arange(5, 40), indexing='ij')
    mask = np.column_stack([lines.ravel() + 1, samples.ravel() + 1])
    out = sep.preview_features(buf.getvalue(), {1: mask})
    assert out['mirrored'] is False and out['leaf_fraction'] == 1.0
    assert out['leaves'][1]['rgb_chlorotic_frac'] == pytest.approx(7 / 35)


# ---------------------------------------------------------------------------
# Parallel ZIP member reader
# ---------------------------------------------------------------------------

class BytesRange(io.RawIOBase):
    """In-memory stand-in for HTTPRangeFile."""

    def __init__(self, data: bytes):
        self.data, self.pos, self.calls = data, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset,
                    io.SEEK_END: len(self.data) + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read_range(self, start, n):
        self.calls += 1
        return self.data[start:start + n]

    def readinto(self, buffer):
        chunk = self.data[self.pos:self.pos + len(buffer)]
        buffer[:len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


@pytest.mark.parametrize('compression', [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_read_member_parallel(compression):
    from src.real_data.common import read_member_parallel

    payload = np.random.default_rng(0).integers(0, 50, 300_000, dtype=np.uint8).tobytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression) as zf:
        zf.writestr('a/first.txt', b'hello')
        zf.writestr('a/cube.hyspex', payload)
    raw = BytesRange(buf.getvalue())
    zf = zipfile.ZipFile(io.BufferedReader(raw))
    assert read_member_parallel(zf, 'a/cube.hyspex', workers=3, chunk=40_000) == payload
    assert raw.calls > 3                      # really fetched in ranges


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def test_scale_uses_training_statistics_only():
    from src.real_data.septoria_experiment import scale

    train = np.random.default_rng(1).normal(5, 2, (50, 3, 2))
    test = train + 100
    s_train, s_test = scale(train, test)
    np.testing.assert_allclose(s_train.reshape(-1, 2).mean(0), 0, atol=1e-6)
    assert (s_test > 40).all()                # test shift not absorbed by the scaler


def test_full_proba_handles_missing_class():
    from sklearn.ensemble import RandomForestClassifier

    from src.real_data.septoria_experiment import full_proba

    X = np.arange(20, dtype=float)[:, None]
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, [0] * 10 + [2] * 10)
    p = full_proba(model, X)
    assert p.shape == (20, 3) and (p[:, 1] == 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_summarise_pot_bootstrap():
    from src.real_data.septoria_experiment import summarise

    y = np.array([0, 1, 2] * 20)
    pots = np.repeat(np.arange(6), 10)
    row = summarise('perfect', y, y, pots, seed=0, n_boot=200)
    assert row['f1_macro'] == 1 and row['f1_macro_ci_low'] == 1
    noisy = np.roll(y, 1)
    row = summarise('noisy', y, noisy, pots, seed=0, n_boot=200)
    assert row['f1_macro_ci_low'] <= row['f1_macro'] <= row['f1_macro_ci_high']


def test_extract_image_from_local_archive(tmp_path, monkeypatch):
    """A hand-downloaded archive in ARCHIVE_DIR/<camera>/ is used instead of the server."""
    header, raw, mask = synthetic_image(mirrored=False)
    archives = tmp_path / 'archives' / 'VNIR'
    archives.mkdir(parents=True)
    members = {'20210311/37_G/37_G.hdr': header.encode(),
               '20210311/37_G/37_G.hyspex': raw,
               '20210311/37_G/37_G_leaf1.csv': pd.DataFrame(mask).to_csv(
                   index=False, header=False).encode()}
    with zipfile.ZipFile(archives / '20210311.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    index = pd.DataFrame([{'member': m, 'date': '20210311', 'pot': 37, 'side': 'G',
                           'leaf': 1 if m.endswith('.csv') else None,
                           'ext': m.rsplit('.', 1)[1], 'sensor': 'VNIR', 'file_id': 0,
                           'archive': '20210311.zip'} for m in members])
    monkeypatch.setattr(sep, 'ARCHIVE_DIR', str(tmp_path / 'archives'))
    def cache_dir(*parts):
        path = tmp_path.joinpath('cache', *parts)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr(sep, 'septoria_dir', cache_dir)
    monkeypatch.setattr(sep, 'file_url', lambda _: pytest.fail('tried the network'))

    path = sep.extract_image(index, 'VNIR', '20210311', 37, 'G')
    feats = sep.load_image_features(path)
    assert set(feats) == {1}
    assert feats[1]['kept_fraction'] == pytest.approx(1.0)
    assert feats[1]['nir_842'] == pytest.approx(0.5, abs=0.01)
    with np.load(path) as z:
        assert not bool(z['mirrored'])
    assert sep.extract_image(index, 'VNIR', '20210311', 37, 'G') == path   # cached


def test_partial_local_archive_rejected(tmp_path, monkeypatch):
    (tmp_path / 'VNIR').mkdir()
    (tmp_path / 'VNIR' / '20210311.zip').write_bytes(b'PK\x03\x04 truncated download')
    with pytest.raises(zipfile.BadZipFile, match='still downloading'):
        sep.local_archive('VNIR', '20210311.zip', archive_dir=str(tmp_path))
    assert sep.local_archive('VNIR', '20210312.zip', archive_dir=str(tmp_path)) is None
