"""
Real-data integration: CDM parsing, Sentinel-2 QC, temporal labelling,
grouped splits, the feature schema and the MicaSense calibration maths.
Everything here is offline; network responses and images are synthesised.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.constants import BANDS
from src.real_data import cdm, feature_schema, sentinel2, temporal, vineyard

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def test_sentinel2_bands_are_exactly_the_project_bands():
    assert list(feature_schema.S2_TO_PROJECT.values()) == BANDS


def test_index_availability_follows_band_coverage():
    uav = feature_schema.available_indices('micasense_rededge_p')
    assert 'NDRE' in uav and 'NDVI' in uav
    assert 'MSI' not in uav and 'NDRE_783_705' not in uav     # no SWIR, one red edge
    common = feature_schema.common_indices('micasense_rededge_p', 'sentinel2')
    assert common == ['NDVI', 'GNDVI', 'NDRE', 'CI_red_edge', 'CI_green']


def test_canonical_features_values():
    bands = {'nir': np.array([0.5]), 'red_edge': np.array([0.2]), 'red': np.array([0.05]),
             'green': np.array([0.1]), 'blue': np.array([0.04])}
    f = feature_schema.compute_canonical_features(bands, 'micasense_rededge_p')
    assert f['NDRE'].iloc[0] == pytest.approx(0.3 / 0.7, rel=1e-6)
    assert f['CI_red_edge'].iloc[0] == pytest.approx(1.5)
    assert 'MSI' not in f.columns
    table = feature_schema.schema_table()
    assert table.query("sensor == 'project' and name == 'MSI'")['available'].item()


# ---------------------------------------------------------------------------
# CDM reports
# ---------------------------------------------------------------------------

def _raw_reports():
    return pd.DataFrame({
        'objectid': [1, 2, 3, 4, 5, 6],
        'Observationdate': ['2024-06-01T00:00:00', '2024-06-10T00:00:00', '2024-07-01T00:00:00',
                            '2024-06-05T00:00:00', '2024-06-05T00:00:00', '2024-06-05T00:00:00'],
        'state': ['NC'] * 6, 'county': ['Wake'] * 6,
        'latitude': ['  35.0', ' 35.00001', '35.0', '', '0', '10.0'],
        'longitude': [' -78.0', '-78.00001', '-78.0', '', '0', '-78.0'],
        'habitat': ['Ag. Field     '] * 6,
    })


def test_normalise_and_select_usable():
    reports = cdm.normalise(_raw_reports(), source='test')
    assert reports['latitude'].iloc[0] == 35.0 and reports['habitat'].iloc[0] == 'Ag. Field'
    flagged = cdm.select_usable(reports, min_days_between=30).set_index('report_id')
    assert flagged.loc['1', 'usable']
    assert flagged.loc['2', 'reject_reason'] == 'repeat_report_same_site'   # 9 days later
    assert flagged.loc['3', 'usable']                                       # 30 days later
    assert flagged.loc['4', 'reject_reason'] == 'no_gps'
    assert flagged.loc['5', 'reject_reason'] == 'zero_gps'
    assert flagged.loc['6', 'reject_reason'] == 'gps_outside_region'
    assert flagged.loc['1', 'site_id'] == flagged.loc['3', 'site_id']
    assert flagged.loc['1', 'event_id'] != flagged.loc['3', 'event_id']


def test_merge_extra_fills_missing_fields(tmp_path):
    reports = cdm.normalise(_raw_reports(), source='test')
    extra = tmp_path / 'extra.csv'
    pd.DataFrame({'objectid': ['1', '99'], 'HostName': ['cucumber', 'squash'],
                  'Severity': ['20', '5'], 'Observationdate': ['2024-06-01', '2024-08-01'],
                  'latitude': ['35', '36'], 'longitude': ['-78', '-79']}).to_csv(extra, index=False)
    merged = cdm.merge_extra(reports, str(extra)).set_index('report_id')
    assert merged.loc['1', 'host'] == 'cucumber' and merged.loc['1', 'severity'] == 20
    assert merged.loc['99', 'host'] == 'squash'            # report only in the export


def test_fetch_year_paginates():
    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.starts = []

        def get(self, url, params, timeout):
            self.starts.append(params['start'])
            rows = [[i, '2024-06-01T00:00:00'] for i in range(params['start'],
                                                               min(params['start'] + 2, 3))]
            return Resp({'recordsFiltered': 3, 'data': rows,
                         'columns': ['objectid', 'Observationdate']})

    session = Session()
    raw = cdm.fetch_year(2024, session=session, page_size=2)
    assert len(raw) == 3 and session.starts == [0, 2]


# ---------------------------------------------------------------------------
# Sentinel-2 QC
# ---------------------------------------------------------------------------

def _footprint(nir=0.4, scl=4, n=3):
    bands = {b: np.full((n, n), 0.2) for b in sentinel2.S2_BANDS}
    bands['B08'] = np.full((n, n), nir, dtype=float)
    return bands, np.full((n, n), scl)


def test_summarise_window_passes_clean_vegetation():
    bands, scl = _footprint()
    obs = sentinel2.summarise_window(bands, scl, sentinel2.PurityConfig())
    assert obs['purity_pass'] and obs['valid_fraction'] == 1.0
    assert obs['B08'] == pytest.approx(0.4) and obs['nir_cv'] == pytest.approx(0.0)


@pytest.mark.parametrize('mutate, reason', [
    (lambda b, s: s.__setitem__((slice(0, 2), slice(None)), 9), 'cloud_or_invalid'),
    (lambda b, s: s.__setitem__((1, 1), 8), 'center_invalid'),
    (lambda b, s: s.fill(5), 'not_vegetated'),
    (lambda b, s: b['B08'].__setitem__((0, slice(None)), 0.05), 'heterogeneous'),
    (lambda b, s: [b[k].fill(np.nan) for k in b], 'no_valid_pixels'),
])
def test_summarise_window_flags(mutate, reason):
    bands, scl = _footprint()
    mutate(bands, scl)
    obs = sentinel2.summarise_window(bands, scl, sentinel2.PurityConfig())
    assert obs['qc_reason'] == reason and not obs['purity_pass']


def test_out_of_range_reflectance_is_invalid():
    bands, scl = _footprint()
    bands['B05'][0, 0] = -0.01       # L2A offset can push dark pixels below 0
    obs = sentinel2.summarise_window(bands, scl, sentinel2.PurityConfig())
    assert obs['n_valid'] == 8


def test_snap_footprint_is_on_the_20m_grid():
    left, bottom, right, top = sentinel2.snap_footprint(500013.0, 4100029.0, 3)
    assert (right - left, top - bottom) == (60.0, 60.0)
    assert left % 20 == 0 and bottom % 20 == 0
    assert left < 500013.0 < right and bottom < 4100029.0 < top


def test_dedupe_keeps_best_tile_per_day():
    obs = pd.DataFrame({'point_id': ['a', 'a', 'a'], 'date': ['2024-06-01'] * 2 + ['2024-06-06'],
                        'n_valid': [3, 9, 9], 'scene_cloud': [1.0, 5.0, 1.0]})
    out = sentinel2.dedupe_acquisitions(obs)
    assert len(out) == 2 and out.iloc[0]['n_valid'] == 9


def test_cdse_statistics_parsing():
    stats = {f'B{i}': {'stats': {'mean': 0.1 * (i + 1), 'stDev': 0.01, 'sampleCount': 9,
                                 'noDataCount': 1}} for i in range(6)}
    entry = {'interval': {'from': '2024-06-01T00:00:00Z'},
             'outputs': {'bands': {'bands': stats},
                         'veg': {'bands': {'B0': {'stats': {'mean': 0.9}}}}}}
    obs = sentinel2.CDSEBackend._parse_interval(entry, sentinel2.PurityConfig())
    assert obs['n_valid'] == 8 and obs['B08'] == pytest.approx(0.4)
    assert obs['nir_cv'] == pytest.approx(0.01 / 0.4) and obs['purity_pass']


def test_cdse_backend_requires_credentials(monkeypatch):
    monkeypatch.delenv('CDSE_CLIENT_ID', raising=False)
    monkeypatch.delenv('CDSE_CLIENT_SECRET', raising=False)
    with pytest.raises(RuntimeError, match='CDSE_CLIENT_ID'):
        sentinel2.CDSEBackend()


# ---------------------------------------------------------------------------
# Temporal labelling and grouped split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('days, label', [(-60, 0), (-22, 0), (-21, 1), (-1, 1), (0, 2),
                                         (45, 2), (46, -1)])
def test_label_from_offset(days, label):
    assert temporal.label_from_offset(days, early_window=21, post_window=45) == label


def _events_and_obs(n_sites=12, every=5, seed=0):
    rng = np.random.default_rng(seed)
    events, obs = [], []
    for s in range(n_sites):
        d0 = pd.Timestamp('2024-07-01') + pd.Timedelta(days=int(s))
        eid = f'site{s}_{d0:%Y%m%d}'
        events.append({'event_id': eid, 'site_id': f'site{s}', 'observation_date': d0,
                       'latitude': 35 + s * 0.5, 'longitude': -78.0})
        for k, d in enumerate(pd.date_range(d0 - pd.Timedelta(days=120),
                                            d0 + pd.Timedelta(days=45), freq=f'{every}D')):
            obs.append({'point_id': eid, 'date': d.strftime('%Y-%m-%d'),
                        'purity_pass': k % 7 != 3,
                        **{b: 0.2 + 0.01 * rng.random() for b in sentinel2.S2_BANDS}})
    return pd.DataFrame(events), pd.DataFrame(obs)


def test_build_sequences_shapes_and_labels():
    events, obs = _events_and_obs()
    raw, samples = temporal.build_sequences(events, obs, seq_len=4, early_window=21,
                                            post_window=45)
    assert set(BANDS) <= set(raw.columns)
    assert (raw.groupby('sample_id').size() == 4).all()
    assert set(samples['label']) == {0, 1, 2}
    for row in samples.itertuples():
        assert row.label == temporal.label_from_offset(row.days_to_report, 21, 45)
    # QC-failing acquisitions never enter a sequence
    failed = set(zip(obs.loc[~obs['purity_pass'], 'point_id'],
                     obs.loc[~obs['purity_pass'], 'date']))
    assert failed and not failed & set(zip(raw['event_id'], raw['date']))


def test_build_sequences_skips_long_gaps():
    events, obs = _events_and_obs(n_sites=1, every=30)
    _, samples = temporal.build_sequences(events, obs, seq_len=3, max_gap_days=20,
                                          require_purity=False)
    assert samples.empty


def test_grouped_preprocessing_keeps_sites_in_one_split(tmp_path):
    from src.data.preprocessor import run_preprocessing_pipeline

    events, obs = _events_and_obs(n_sites=20)
    raw, _ = temporal.build_sequences(events, obs, seq_len=4)
    csv = tmp_path / 'real.csv'
    raw.to_csv(csv, index=False)
    data = run_preprocessing_pipeline(str(csv), str(tmp_path / 'proc'), seq_len=4, seed=1,
                                      group_col='site_id')
    meta = pd.read_csv(tmp_path / 'proc' / 'sample_meta.csv')
    sites_per_split = meta.groupby('site_id')['split'].nunique()
    assert (sites_per_split == 1).all()
    assert data['X_train'].shape[1:] == (4, 16)       # 12 spectral + 4 context
    assert len(data['X_test']) > 0 and len(data['X_val']) > 0


def test_seasonality_features_have_no_spectra():
    samples = pd.DataFrame({'anchor_doy': [1, 183], 'lat': [35.0, 40.0]})
    X = temporal.seasonality_features(samples)
    assert X.shape == (2, 3) and X[0, 1] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Vineyard: calibration maths, panel, experiment
# ---------------------------------------------------------------------------

def _meta(**kw):
    meta = {'radiometric_cal': [2.6e-4, 0.0, 0.0], 'vignette_poly2d': [1.0],
            'vignette_exponents': [0, 0], 'vignette_center': [], 'vignette_poly': [],
            'black_level': 4000.0, 'exposure_s': 0.002, 'gain': 1.0, 'bits': 16,
            'horizontal_irradiance': 50.0, 'spectral_irradiance': 60.0}
    meta.update(kw)
    return meta


def test_xmp_parsing():
    xmp = ('<MicaSense:RadiometricCalibration><rdf:Seq><rdf:li>0.1</rdf:li><rdf:li>2e-09</rdf:li>'
           '<rdf:li>3</rdf:li></rdf:Seq></MicaSense:RadiometricCalibration>'
           '<Camera:BandName>NIR</Camera:BandName>')
    assert vineyard._floats(vineyard._xmp_value(xmp, 'MicaSense:RadiometricCalibration')) == \
        [0.1, 2e-09, 3.0]
    assert vineyard._xmp_value(xmp, 'Camera:BandName') == 'NIR'


def test_radiance_and_reflectance_formula():
    dn = np.full((4, 5), 4000.0 + 0.25 * 65536)
    meta = _meta()
    L = vineyard.dn_to_radiance(dn, meta)
    assert np.allclose(L, 0.25 * 2.6e-4 / 0.002)
    R = vineyard.dn_to_reflectance(dn, meta)
    assert np.allclose(R, np.pi * L / (50.0 * vineyard.DLS2_IRRADIANCE_SCALE))
    # below the black level clips to zero radiance
    assert vineyard.dn_to_radiance(np.full((2, 2), 100.0), meta).max() == 0


def test_vignette_polynomial_2d():
    meta = _meta(vignette_poly2d=[1.0, -0.5], vignette_exponents=[0, 0, 1, 0])
    V = vineyard.vignette_map(meta, (2, 4))
    # poly = 1 - 0.5 * x/width -> V = 1 / poly grows towards the right edge
    assert V[0, 0] == pytest.approx(1.0) and V[0, 2] == pytest.approx(1 / 0.75)


def test_detect_panel_finds_grey_square():
    rng = np.random.default_rng(0)
    h, w = 200, 240
    grass = {b: (base + 0.05 * rng.random((h, w))).astype(np.float32)
             for b, base in zip(vineyard.SPECTRAL_BANDS, (0.03, 0.08, 0.04, 0.25, 0.45))}
    for img in grass.values():
        img[60:140, 80:180] = 0.52
    mask = vineyard.detect_panel(grass, np.ones((h, w), bool), min_area=500)
    assert mask is not None and mask[100, 130] and not mask[10, 10]


def test_panel_factors_and_assignment():
    panels = pd.DataFrame({'session': ['0000', '0000', '0019'],
                           **{f'panel_{b}': [0.6, 0.6, 0.4] for b in vineyard.SPECTRAL_BANDS}})
    f = vineyard.panel_factors(panels, nominal_albedo=0.5)
    assert f.loc['0000', 'nir'] == pytest.approx(0.5 / 0.6)
    ann = pd.DataFrame({'session': ['0000', '0005', '0019', '0023'],
                        'lat': [44.86, 44.86, 44.79, 44.7935],
                        'lon': [10.81, 10.81, 10.80, 10.8004]}).set_index('session')
    mapping = vineyard.assign_panels(['0005', '0023'], ann, ['0000', '0019'])
    assert mapping == {'0005': '0000', '0023': '0019'}


def _fake_capture_features(n_sessions=10, per=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        diseased = s % 2 == 1
        for c in range(per):
            row = {'session': f'{s:04d}', 'capture': f'{c:04d}', 'type': 'marani',
                   'FD': diseased, 'ESCA': False, 'vegetation_fraction': rng.random()}
            for col in ('NDRE', 'NDVI'):
                for stat in vineyard.FEATURE_STATS:
                    row[f'{col}_{stat}'] = (0.3 if diseased else 0.5) + 0.02 * rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_task_labels():
    df = pd.DataFrame({'FD': [True, False, True, False], 'ESCA': [False, True, True, False]})
    assert vineyard.task_labels(df, 'binary').tolist() == [1, 1, 1, 0]
    multi = vineyard.task_labels(df, 'multiclass').tolist()
    assert multi[:2] == [1, 2] and pd.isna(multi[2]) and multi[3] == 0


def test_vineyard_experiment_on_separable_data(tmp_path):
    feats = _fake_capture_features()
    metrics = vineyard.run_experiment(feats, str(tmp_path), task='binary', n_permutations=5)
    assert set(metrics['model']) == {'Logistic Regression', 'Random Forest', 'XGBoost'}
    assert (metrics['capture_balanced_accuracy'] > 0.9).all()
    assert metrics['perm_p_value'].dropna().between(0, 1).all()
    tests = pd.read_csv(tmp_path / 'feature_tests_binary.csv')
    assert tests['q_value_bh'].between(0, 1).all()
    preds = pd.read_csv(tmp_path / 'predictions_binary.csv', dtype={'session': str})
    # leave-one-session-out: each capture predicted exactly once per model
    assert preds.groupby('model').size().eq(len(feats)).all()


def test_list_captures(tmp_path):
    d = tmp_path / 'raw' / '0003SET'
    d.mkdir(parents=True)
    for b in (1, 2):
        (d / f'IMG_0007_{b}.tif').write_bytes(b'')
    caps = vineyard.list_captures(str(tmp_path))
    assert list(caps) == [('0003', '0007')]
    assert set(caps[('0003', '0007')]) == {1, 2}
    assert os.path.basename(caps[('0003', '0007')][2]) == 'IMG_0007_2.tif'


# ---------------------------------------------------------------------------
# EthCT2020 analysis and CLI wiring
# ---------------------------------------------------------------------------

def _eth_fields_and_obs(per_class=12, seed=0):
    from src.real_data.sentinel2 import S2_BANDS

    rng = np.random.default_rng(seed)
    fields, obs = [], []
    crops = {'wheat': 0.40, 'maize': 0.30, 'teff': 0.20}
    fid = 0
    for crop, nir in crops.items():
        for _ in range(per_class):
            fid += 1
            fields.append({'field_id': fid, 'crop_class': crop, 'crop_group': 'cereals',
                           'source': 'GDCC', 'quality': 'very good',
                           'location_confidence': 'confident',
                           'lat': 8 + rng.random() * 3, 'lon': 38 + rng.random() * 3})
            for d in pd.date_range('2020-09-01', '2020-11-30', freq='10D'):
                ok = rng.random() > 0.3
                obs.append({'point_id': f'eth_{fid}', 'date': d.strftime('%Y-%m-%d'),
                            **{b: 0.2 + 0.01 * rng.random() for b in S2_BANDS},
                            'B08': nir + 0.01 * rng.random(), 'n_valid': 9 if ok else 0,
                            'nir_cv': 0.05, 'purity_pass': ok,
                            'qc_reason': '' if ok else 'cloud_or_invalid'})
    return pd.DataFrame(fields), pd.DataFrame(obs)


def test_ethct_sampling_purity_and_separability():
    from src.real_data import ethct

    fields, obs = _eth_fields_and_obs()
    sample = ethct.sample_fields(fields, n_fields=30, min_per_class=5)
    assert sample['crop_class'].value_counts().max() <= 10
    overall, by_crop = ethct.purity_summary(obs, fields)
    shares = overall.set_index('outcome')['share']
    assert shares['pass'] + shares['cloud_or_invalid'] == pytest.approx(1.0)
    assert set(by_crop['crop_class']) == {'wheat', 'maize', 'teff'}
    feats = ethct.field_features(obs, require_purity=True)
    assert len(feats) == len(fields) and 'NDRE' in feats.columns
    sep = ethct.separability(obs, fields)
    assert set(sep['acquisitions']) == {'all_cloud_free', 'qc_pass_only'}
    assert (sep['balanced_accuracy'] > sep['chance_balanced_accuracy']).all()


def test_cli_parser_and_schema(tmp_path, monkeypatch):
    from src.real_data import __main__ as cli
    from src.real_data import common

    monkeypatch.setattr(common, 'REAL_REPORTS_DIR', str(tmp_path))
    args = cli.build_parser().parse_args(['s2-cdm', '--window-px', '5', '--no-center-check'])
    cfg = cli.purity_config(args)
    assert cfg.window_px == 5 and not cfg.require_center_valid
    with pytest.raises(ValueError):
        sentinel2.PurityConfig(window_px=4)
    assert cli.main(['schema']) == 0
    assert (tmp_path / 'feature_schema.csv').exists()


def test_cli_temporal_builds_dataset(tmp_path, monkeypatch):
    from src.real_data import __main__ as cli
    from src.real_data import common

    monkeypatch.setattr(common, 'REAL_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(common, 'REAL_REPORTS_DIR', str(tmp_path / 'reports'))
    events, obs = _events_and_obs(n_sites=4)
    events['state'], events['habitat'] = 'NC', 'Ag. Field'
    events['observation_date'] = events['observation_date'].dt.strftime('%Y-%m-%d')
    (tmp_path / 'data' / 'cdm').mkdir(parents=True)
    (tmp_path / 'data' / 's2').mkdir(parents=True)
    events.to_csv(tmp_path / 'data' / 'cdm' / 'cdm_events.csv', index=False)
    obs.assign(vegetation_fraction=1.0, nir_cv=0.05,
               qc_reason=np.where(obs['purity_pass'], '', 'heterogeneous')).to_csv(
        tmp_path / 'data' / 's2' / 'cdm_observations.csv', index=False)
    assert cli.main(['temporal', '--seq-len', '4']) == 0
    samples = pd.read_csv(tmp_path / 'data' / 'cdm_s2' / 'cdm_s2_samples.csv')
    assert len(samples) > 0
    summary = pd.read_csv(tmp_path / 'reports' / 'event_qc_summary.csv')
    assert 'fail_heterogeneous' in summary.columns


def test_http_range_file_reads_zip_members(tmp_path):
    import io
    import zipfile

    from src.real_data.common import HTTPRangeFile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('root/annotation.json', '{"a": 1}' * 100)
    blob = buf.getvalue()

    raw = HTTPRangeFile('http://x/f.zip', session=RangeSession(blob))
    z = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=64))
    info = z.getinfo('root/annotation.json')
    assert vineyard._read_member_http(raw, info) == b'{"a": 1}' * 100


class RangeResp:
    def __init__(self, status, content=b'', headers=None, url='http://x/f.zip'):
        self.status_code, self.content = status, content
        self.headers, self.url = headers or {}, url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def close(self):
        pass


class RangeSession:
    """Range-only server (HEAD unsupported, like pre-signed S3 GET links)."""

    def __init__(self, blob: bytes, expire_after: int | None = None):
        self.blob, self.served, self.expire_after = blob, 0, expire_after
        self.resolves = 0

    def get(self, url, headers=None, **kw):
        a, b = map(int, headers['Range'].split('=')[1].split('-'))
        if url == 'http://x/f.zip':                       # the "redirecting" source URL
            self.resolves += 1
            url = f'http://s3/f.zip?sig={self.resolves}'
        elif self.expire_after is not None and self.served == self.expire_after:
            self.expire_after = None
            return RangeResp(403, url=url)                  # link expires once
        self.served += 1
        part = self.blob[a:b + 1]
        return RangeResp(206, part, {'Content-Range': f'bytes {a}-{b}/{len(self.blob)}'},
                         url=url)


def test_http_range_file_reresolves_expired_presigned_link():
    from src.real_data.common import HTTPRangeFile

    blob = bytes(range(256)) * 40
    session = RangeSession(blob, expire_after=3)
    raw = HTTPRangeFile('http://x/f.zip', session=session)
    assert raw.size == len(blob) and raw.url.startswith('http://s3/')
    out = b''.join(raw.read_range(i, 100) for i in range(0, 1000, 100))
    assert out == blob[:1000]
    assert session.resolves == 2                 # initial resolve + one after the 403
