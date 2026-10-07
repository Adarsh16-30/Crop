"""
API Tests — run against a REAL loaded model (TestClient used as a context
manager so the lifespan actually loads the checkpoint).
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api

from .conftest import SEQ_LEN

KEY = 'test-key'
HEADERS = {'X-API-Key': KEY}


@pytest.fixture
def client(tiny_checkpoint, monkeypatch):
    monkeypatch.setattr(api.settings, 'checkpoint_path', tiny_checkpoint)
    monkeypatch.setattr(api.settings, 'api_keys', {KEY})
    monkeypatch.setattr(api.settings, 'allow_unauthenticated', False)
    monkeypatch.setattr(api.settings, 'rate_limit_per_minute', 1000)
    monkeypatch.setattr(api, 'rate_limiter', api.RateLimiter())
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def readings(raw_test_sequences):
    return raw_test_sequences[0].round(6).tolist()


class TestHealthAndInfo:

    def test_health_ok_with_model(self, client):
        r = client.get('/health')
        assert r.status_code == 200
        assert r.json()['status'] == 'ok' and r.json()['model_loaded']

    def test_health_503_without_model(self, monkeypatch):
        monkeypatch.setattr(api.settings, 'checkpoint_path', 'does/not/exist.pt')
        with TestClient(api.app) as c:
            r = c.get('/health')
        assert r.status_code == 503
        assert r.json()['status'] == 'degraded'
        assert 'not found' in r.json()['detail']

    def test_info_describes_served_model(self, client):
        info = client.get('/info', headers=HEADERS).json()
        assert info['model_type'] == 'lstm'
        assert info['seq_len'] == SEQ_LEN
        assert info['hidden_dim'] == 16
        assert len(info['raw_bands']) == 6


class TestPredict:

    def test_predict_returns_real_prediction(self, client, readings):
        r = client.post('/predict', json={'readings': readings}, headers=HEADERS)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['label_name'] in ('Healthy', 'Early Stress', 'Active Disease')
        assert sum(body['probabilities']) == pytest.approx(1.0, abs=1e-3)
        assert body['status'] in ('ok', 'low_confidence', 'out_of_distribution')
        assert body['model_type'] == 'lstm'

    def test_batch(self, client, raw_test_sequences):
        seqs = raw_test_sequences[:3].round(6).tolist()
        r = client.post('/predict/batch', json={'sequences': seqs}, headers=HEADERS)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 3

    def test_batch_size_limit(self, client, readings, monkeypatch):
        monkeypatch.setattr(api.settings, 'max_batch_size', 2)
        r = client.post('/predict/batch', json={'sequences': [readings] * 3}, headers=HEADERS)
        assert r.status_code == 422

    @pytest.mark.parametrize('mutate', [
        lambda r: r[:-1],                                 # wrong length
        lambda r: r + [r[0]],                             # wrong length
        lambda r: [row[:5] for row in r],                 # 5 bands
        lambda r: [[1.5] + row[1:] for row in r],         # reflectance > 1
        lambda r: [[-0.1] + row[1:] for row in r],        # reflectance < 0
    ])
    def test_invalid_input_422(self, client, readings, mutate):
        r = client.post('/predict', json={'readings': mutate(readings)}, headers=HEADERS)
        assert r.status_code == 422

    def test_extra_fields_rejected(self, client, readings):
        r = client.post('/predict', json={'readings': readings, 'x': 1}, headers=HEADERS)
        assert r.status_code == 422

    def test_old_12_feature_contract_rejected(self, client):
        old = [[0.35, 0.55, 0.65, 0.80, 0.20, 0.15, 0.45, 0.60, 0.90, 0.78, 0.25, 1.03]] * SEQ_LEN
        assert client.post('/predict', json={'sequence': old}, headers=HEADERS).status_code == 422

    def test_errors_do_not_leak_internals(self, client, readings, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError('secret internal path /srv/x')
        monkeypatch.setattr(api.state.predictor, 'predict_raw', boom)
        r = client.post('/predict', json={'readings': readings}, headers=HEADERS)
        assert r.status_code == 500
        assert 'secret' not in r.text

    def test_latency_under_200ms(self, client, readings):
        client.post('/predict', json={'readings': readings}, headers=HEADERS)  # warm-up
        start = time.perf_counter()
        r = client.post('/predict', json={'readings': readings}, headers=HEADERS)
        assert r.status_code == 200
        assert (time.perf_counter() - start) * 1000 < 200


class TestSecurity:

    def test_missing_key_401(self, client, readings):
        assert client.post('/predict', json={'readings': readings}).status_code == 401

    def test_wrong_key_401(self, client, readings):
        r = client.post('/predict', json={'readings': readings}, headers={'X-API-Key': 'nope'})
        assert r.status_code == 401

    def test_no_auth_configured_refuses(self, client, readings, monkeypatch):
        monkeypatch.setattr(api.settings, 'api_keys', set())
        assert client.post('/predict', json={'readings': readings}).status_code == 503

    def test_explicit_dev_mode_allows_open_access(self, client, readings, monkeypatch):
        monkeypatch.setattr(api.settings, 'api_keys', set())
        monkeypatch.setattr(api.settings, 'allow_unauthenticated', True)
        assert client.post('/predict', json={'readings': readings}).status_code == 200

    def test_rate_limit(self, client, readings, monkeypatch):
        monkeypatch.setattr(api.settings, 'rate_limit_per_minute', 3)
        codes = [client.post('/predict', json={'readings': readings}, headers=HEADERS).status_code
                 for _ in range(5)]
        assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]

    def test_body_size_limit(self, client):
        big = {'readings': np.full((5000, 6), 0.4).tolist()}
        r = client.post('/predict', json=big, headers=HEADERS)
        assert r.status_code in (413, 422)
        assert client.post('/predict', content=b'x' * 2_000_000, headers={
            **HEADERS, 'Content-Type': 'application/json'}).status_code == 413


def test_prometheus_metrics(client, readings):
    client.post('/predict', json={'readings': readings}, headers=HEADERS)
    text = client.get('/metrics').text
    assert 'api_requests_total' in text
    assert 'api_inference_seconds_bucket' in text
    assert 'api_model_loaded 1.0' in text
