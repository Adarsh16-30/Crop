"""
Streaming Tests — publisher payloads, payload validation, per-cycle sequence
assembly and the bridge's processing path (no broker / InfluxDB needed).
"""

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from src.config import load_influx_config, load_mqtt_config
from src.constants import BANDS
from src.dashboard.mqtt_subscriber import MQTTInfluxBridge, SensorReading, SequenceAssembler
from src.data.mqtt_publisher import SpectralPublisher
from src.inference.predictor import Predictor

from .conftest import SEQ_LEN, context_at

MQTT_CFG = {'host': 'localhost', 'port': 8883, 'topic': 'crop/spectral', 'client_id': 'test',
            'keepalive': 60, 'qos': 1, 'username': None, 'password': None, 'tls_ca_file': None}


def reading(station='S01', cycle=0, t=0, **overrides):
    payload = {'station_id': station, 'cycle_id': cycle, 'timestep': t, 'lat': 12.9,
               'lon': 77.5, **{b: 0.4 for b in BANDS}}
    payload.update(overrides)
    return SensorReading.model_validate(payload)


class TestPublisher:

    def test_cycles_replay_real_sequences_and_carry_ids(self, replay_sites):
        pub = SpectralPublisher(MQTT_CFG, n_stations=2, seed=1, sites=replay_sites)
        assert pub.cycle_length == SEQ_LEN
        station = pub.stations[0]
        real = station['trajectory'][BANDS].to_numpy()
        station_dates = station['trajectory']['date'].to_list()
        msgs = []
        for _ in range(SEQ_LEN + 2):
            msgs.append(pub._generate_reading(station))
            station['timestep'] += 1
            if station['timestep'] >= pub.cycle_length:
                pub._start_cycle(station)
        assert [m['timestep'] for m in msgs[:SEQ_LEN]] == list(range(SEQ_LEN))
        assert {m['cycle_id'] for m in msgs[:SEQ_LEN]} == {0}
        assert msgs[SEQ_LEN]['cycle_id'] == 1 and msgs[SEQ_LEN]['timestep'] == 0
        np.testing.assert_allclose([[m[b] for b in BANDS] for m in msgs[:SEQ_LEN]], real)
        assert msgs[0]['profile'] in ('healthy', 'early_stress', 'active_disease')
        parsed = SensorReading.model_validate(json.loads(json.dumps(msgs[0])))  # valid payload
        assert parsed.acquisition_date() == str(station_dates[0])[:10]

    def test_stations_are_real_sites(self, replay_sites):
        a = SpectralPublisher(MQTT_CFG, n_stations=3, seed=5, sites=replay_sites)
        b = SpectralPublisher(MQTT_CFG, n_stations=3, seed=5, sites=replay_sites)
        assert [s['station_id'] for s in a.stations] == ['S01', 'S02', 'S03']
        assert [s['site_id'] for s in a.stations] == [s['site_id'] for s in b.stations]
        for s in a.stations:
            seq = replay_sites[s['site_id']][0]
            assert (s['lat'], s['lon']) == (round(seq['lat'].iloc[0], 6),
                                            round(seq['lon'].iloc[0], 6))

    def test_replay_restricted_to_split(self, tmp_path):
        import pandas as pd

        from src.data.mqtt_publisher import load_replay_sequences

        from .conftest import SEQUENCES_CSV
        ids = pd.read_csv(SEQUENCES_CSV)['sample_id'].unique()
        meta = tmp_path / 'meta.csv'
        pd.DataFrame({'sample_id': ids, 'split': ['test'] * 3 + ['train'] * (len(ids) - 3)}
                     ).to_csv(meta, index=False)
        sites = load_replay_sequences(SEQUENCES_CSV, str(meta), split='test')
        assert sum(len(v) for v in sites.values()) == 3


class TestPayloadValidation:

    def test_missing_band_rejected(self):
        payload = reading().model_dump()
        del payload['nir_842']
        with pytest.raises(ValidationError):
            SensorReading.model_validate(payload)

    @pytest.mark.parametrize('bad', [{'nir_842': 1.7}, {'nir_842': float('nan')},
                                     {'lat': 200.0}, {'station_id': ''}, {'timestep': -1}])
    def test_bad_values_rejected(self, bad):
        with pytest.raises(ValidationError):
            reading(**bad)


class TestSequenceAssembler:

    def test_emits_once_per_complete_cycle(self):
        asm = SequenceAssembler(seq_len=4)
        out = [asm.add(reading(t=t, nir_842=0.1 * (t + 1))) for t in range(4)]
        assert out[:3] == [None, None, None]
        np.testing.assert_allclose(out[3][:, BANDS.index('nir_842')], [0.1, 0.2, 0.3, 0.4])

    def test_never_mixes_two_cycles(self):
        asm = SequenceAssembler(seq_len=4)
        for t in range(3):
            asm.add(reading(cycle=0, t=t))
        # New cycle starts before cycle 0 finished: cycle 0 is abandoned.
        assert asm.add(reading(cycle=1, t=0)) is None
        assert asm.add(reading(cycle=0, t=3)) is None  # late reading ignored
        results = [asm.add(reading(cycle=1, t=t)) for t in range(1, 4)]
        assert results[-1] is not None and results[-1].shape == (4, 6)

    def test_stations_are_independent(self):
        asm = SequenceAssembler(seq_len=2)
        assert asm.add(reading('S01', t=0)) is None
        assert asm.add(reading('S02', t=0)) is None
        assert asm.add(reading('S01', t=1)) is not None

    def test_out_of_order_within_cycle(self):
        asm = SequenceAssembler(seq_len=3)
        assert asm.add(reading(t=2)) is None
        assert asm.add(reading(t=0)) is None
        assert asm.add(reading(t=1)) is not None


class FakeWriteAPI:
    def __init__(self):
        self.records = []

    def write(self, bucket, org, record):
        self.records.extend(record if isinstance(record, list) else [record])


class TestBridgeProcessing:

    def test_full_cycle_produces_one_prediction(self, tiny_checkpoint, raw_test_sequences,
                                                 raw_test_context):
        bridge = MQTTInfluxBridge.__new__(MQTTInfluxBridge)  # skip network setup
        bridge.predictor = Predictor(tiny_checkpoint, device=torch.device('cpu'))
        bridge.assembler = SequenceAssembler(seq_len=SEQ_LEN)
        bridge.write_api = FakeWriteAPI()
        bridge.bucket, bridge.org = 'b', 'o'
        bridge.stats = {'processed': 0, 'invalid': 0, 'predictions': 0, 'write_errors': 0}

        seq = raw_test_sequences[0]
        ctx = context_at(raw_test_context, 0)
        preds = [bridge.process_reading(reading(t=t, date=str(d), lat=ctx['lat'], lon=ctx['lon'],
                                                **dict(zip(BANDS, map(float, row)))))
                 for t, (row, d) in enumerate(zip(seq, ctx['dates']))]
        assert all(p is None for p in preds[:-1])
        expected = bridge.predictor.predict_raw(seq, **ctx)[0]
        assert preds[-1]['probabilities'] == pytest.approx(expected['probabilities'], abs=1e-6)
        names = [r._name for r in bridge.write_api.records]
        assert names.count('spectral_readings') == SEQ_LEN
        assert names.count('disease_predictions') == 1

    def test_invalid_json_counted_not_crashing(self):
        bridge = MQTTInfluxBridge.__new__(MQTTInfluxBridge)
        bridge.stats = {'processed': 0, 'invalid': 0, 'predictions': 0, 'write_errors': 0}

        class Msg:
            topic = 'crop/spectral'
            payload = b'{"station_id": "S01"}'
        bridge._on_message(None, None, Msg())
        Msg.payload = b'not json'
        bridge._on_message(None, None, Msg())
        assert bridge.stats['invalid'] == 2


class TestConfig:

    def test_env_overrides_ini_and_secrets_only_from_env(self, monkeypatch, tmp_path):
        ini = tmp_path / 'mqtt.ini'
        ini.write_text('[mqtt]\nhost = brokerhost\nport = 1884\nkeepalive = 30\n')
        monkeypatch.setenv('MQTT_PORT', '2000')
        monkeypatch.setenv('MQTT_PASSWORD', 'pw')
        cfg = load_mqtt_config(str(ini))
        assert cfg['host'] == 'brokerhost' and cfg['port'] == 2000
        assert cfg['keepalive'] == 30 and cfg['password'] == 'pw'

    def test_influx_scoped_token_preferred(self, monkeypatch, tmp_path):
        monkeypatch.setenv('INFLUX_TOKEN', 'admin')
        monkeypatch.setenv('INFLUX_WRITE_TOKEN', 'writer')
        assert load_influx_config(str(tmp_path / 'none.ini'),
                                  token_env='INFLUX_WRITE_TOKEN')['token'] == 'writer'


class TestMqttTls:

    @pytest.fixture(scope='class')
    def ca_file(self, tmp_path_factory):
        import os
        import shutil
        import subprocess
        if not shutil.which('openssl'):
            pytest.skip('openssl CLI not available')
        out = tmp_path_factory.mktemp('ca')
        key, crt = out / 'ca.key', out / 'ca.crt'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt',
                        'ec_paramgen_curve:prime256v1', '-nodes', '-keyout', str(key),
                        '-out', str(crt), '-days', '1', '-subj', '/CN=Test CA'],
                       check=True, capture_output=True,
                       env={**os.environ, 'MSYS_NO_PATHCONV': '1'})
        return str(crt)

    def test_context_verifies_cert_and_hostname_tls12_plus(self, ca_file):
        import ssl

        from src.config import make_tls_context
        ctx = make_tls_context(ca_file)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True
        assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_client_uses_tls_when_ca_configured(self, ca_file):
        from src.config import make_mqtt_client
        client = make_mqtt_client({**MQTT_CFG, 'tls_ca_file': ca_file}, client_id='t')
        assert client._ssl_context is not None and client._ssl

    def test_missing_ca_is_a_clear_error(self):
        from src.config import make_mqtt_client
        with pytest.raises(FileNotFoundError, match='gen_mqtt_certs.sh'):
            make_mqtt_client({**MQTT_CFG, 'tls_ca_file': 'nope/ca.crt'}, client_id='t')

    def test_default_config_is_tls(self, monkeypatch):
        for var in ('MQTT_PORT', 'MQTT_TLS_CA_FILE'):
            monkeypatch.delenv(var, raising=False)
        cfg = load_mqtt_config('config/mqtt.ini')
        assert cfg['port'] == 8883
        assert cfg['tls_ca_file'] == 'certs/mqtt/ca.crt'
