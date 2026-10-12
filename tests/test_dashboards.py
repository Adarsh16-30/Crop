"""
Grafana integration tests — dashboards-as-code, InfluxDB schema consistency,
history backfill and pipeline-metrics export (no running services needed).
"""

import json
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import BANDS, INDICES
from src.dashboard.grafana import build_dashboards as bd
from src.dashboard.mqtt_subscriber import (
    SensorReading,
    build_prediction_point,
    build_reading_point,
)

from .conftest import SEQ_LEN

DASHBOARDS = {name: json.loads(content) for name, content in bd.render().items()}


def _fields(point) -> set:
    return set(point._fields)


def _reading(**overrides):
    payload = {'station_id': 'S01', 'cycle_id': 0, 'timestep': 0, 'lat': 12.9, 'lon': 77.5,
               'timestamp': 1_700_000_000.0, 'profile': 'early_stress', 'source': 'simulator',
               **{b: 0.4 for b in BANDS}}
    payload.update(overrides)
    return SensorReading.model_validate(payload)


PREDICTION = {'predicted_label': 1, 'label_name': 'Early Stress', 'confidence': 0.9,
              'risk_score': 0.5, 'out_of_distribution': False,
              'probabilities': [0.05, 0.9, 0.05]}


@pytest.fixture(scope='module')
def schema(tmp_path_factory):
    """measurement -> field names, taken from the code that actually writes them."""
    from src.dashboard.publish_pipeline_metrics import build_points

    reports = tmp_path_factory.mktemp('reports')
    processed = tmp_path_factory.mktemp('processed')
    pd.DataFrame([{'Model': 'LSTM', 'n_test': 10, 'accuracy': 0.9, 'accuracy_ci_low': 0.8,
                   'accuracy_ci_high': 0.95, 'f1_macro': 0.9, 'f1_macro_ci_low': 0.8,
                   'f1_macro_ci_high': 0.95, 'f1_weighted': 0.9, 'precision_macro': 0.9,
                   'recall_macro': 0.9, 'f1_healthy': 0.9, 'f1_early_stress': 0.9,
                   'f1_active_disease': 0.9}]).to_csv(reports / 'model_comparison.csv',
                                                      index=False)
    pd.DataFrame([{'model_a': 'A', 'model_b': 'B', 'only_a_correct': 1, 'only_b_correct': 2,
                   'p_value': 0.5, 'significant_at_0.05': False}]).to_csv(
        reports / 'model_significance.csv', index=False)
    pd.DataFrame({'epoch': [1, 2], 'train_loss': [1, .5], 'train_acc': [.5, .8],
                  'val_loss': [1, .6], 'val_acc': [.5, .7], 'val_macro_f1': [.4, .7],
                  'lr': [1e-3, 1e-3]}).to_csv(reports / 'lstm_training_log.csv', index=False)
    pd.DataFrame([{'model': 'Random Forest', 'cv_split': s, 'cv_folds': 5,
                   'cv_accuracy_mean': .9, 'cv_accuracy_std': .01, 'cv_f1_macro_mean': .9,
                   'cv_f1_macro_std': .01} for s in ('random', 'spatial_zone')]).to_csv(
        reports / 'classical_cv.csv', index=False)
    pd.DataFrame([{'method': 'mrmr', 'n_features': 8, 'cv_folds': 5, 'cv_accuracy_mean': .9,
                   'cv_accuracy_std': .01, 'cv_f1_macro_mean': .9, 'cv_f1_macro_std': .01,
                   'test_accuracy': .9, 'test_f1_macro': .9}]).to_csv(
        reports / 'feature_selection_comparison.csv', index=False)
    pd.DataFrame([{'sample_id': 3, 'split': 'test', 'prob_class_0': .1, 'prob_class_1': .2,
                   'prob_class_2': .7, 'true_label': 2, 'predicted_label': 2,
                   'confidence': .7, 'lat': 12.9, 'lon': 77.5, 'model': 'lstm'}]).to_csv(
        processed / 'predictions.csv', index=False)

    out: dict[str, set] = {}
    for point in build_points(str(reports), str(processed)):
        out.setdefault(point._name, set()).update(point._fields)
    out['spectral_readings'] = _fields(build_reading_point(_reading()))
    out['disease_predictions'] = _fields(build_prediction_point(_reading(), PREDICTION))
    out['bridge_stats'] = {'processed', 'invalid', 'predictions', 'write_errors'}
    return out


class TestDashboardDefinitions:

    def test_committed_json_matches_generator(self):
        """Dashboards are generated; hand edits to the JSON would be overwritten."""
        for name, content in bd.render().items():
            with open(f'{bd.OUT_DIR}/{name}', encoding='utf-8') as f:
                assert f.read() == content, f'{name} is stale: run build_dashboards'

    def test_four_linked_dashboards(self):
        uids = {d['uid'] for d in DASHBOARDS.values()}
        assert uids == {'crop-overview', 'crop-spectral', 'crop-models', 'crop-system'}
        for d in DASHBOARDS.values():
            assert d['tags'] == ['crop'] and d['links'][0]['tags'] == ['crop']

    @pytest.mark.parametrize('name', sorted(DASHBOARDS))
    def test_layout_and_ids(self, name):
        panels = DASHBOARDS[name]['panels']
        ids = [p['id'] for p in panels]
        assert len(ids) == len(set(ids))
        for p in panels:
            g = p['gridPos']
            assert 0 <= g['x'] and g['x'] + g['w'] <= 24, p['title']

    @pytest.mark.parametrize('name', sorted(DASHBOARDS))
    def test_queries_use_provisioned_datasource_and_defined_variables(self, name):
        dash = DASHBOARDS[name]
        defined = {v['name'] for v in dash['templating']['list']}
        for p in dash['panels']:
            for t in p.get('targets', []):
                assert t['datasource']['uid'] == 'influxdb-crop'
                used = set(re.findall(r'\$\{(\w+)(?::\w+)?\}', t['query']))
                assert used <= defined, (p['title'], used - defined)

    def test_queries_reference_fields_that_are_written(self, schema):
        """Every r._field / r._measurement in a panel must exist in what the code writes."""
        for dash in DASHBOARDS.values():
            for p in dash['panels']:
                for t in p.get('targets', []):
                    measurements = re.findall(r'r\._measurement == "(\w+)"', t['query'])
                    fields = set(re.findall(r'r\._field == "(\w+)"', t['query']))
                    for m in measurements:
                        assert m in schema, (p['title'], m)
                        missing = fields - schema[m]
                        assert not missing, (p['title'], m, missing)

    def test_spectral_fields_complete(self, schema):
        assert set(BANDS) | set(INDICES) <= schema['spectral_readings']


class TestBridgePoints:

    def test_reading_point_uses_sensor_timestamp_and_numeric_coords(self):
        point = build_reading_point(_reading())
        assert point._time == datetime.fromtimestamp(1_700_000_000.0, tz=UTC)
        assert point._fields['latitude'] == 12.9 and point._fields['longitude'] == 77.5

    def test_prediction_point_records_ground_truth(self):
        point = build_prediction_point(_reading(), PREDICTION)
        assert point._tags['true_profile'] == 'early_stress'
        assert point._fields['true_label'] == 1 and point._fields['correct'] == 1
        wrong = build_prediction_point(_reading(profile='healthy'), PREDICTION)
        assert wrong._fields['correct'] == 0

    def test_no_ground_truth_for_real_sensors(self):
        point = build_prediction_point(_reading(profile=None), PREDICTION)
        assert 'true_label' not in point._fields and 'true_profile' not in point._tags


class TestSeedHistory:

    def test_backfill_runs_through_the_real_bridge_path(self, tiny_checkpoint):
        from src.dashboard.seed_history import simulate_history
        from src.inference.predictor import Predictor

        predictor = Predictor(tiny_checkpoint, device=torch.device('cpu'))
        end = 1_800_000_000.0
        out = list(simulate_history(predictor, hours=0.5, interval_s=30, n_stations=2,
                                    end_time=end))
        readings = [recs[0] for recs, _ in out]
        preds = [p for _, p in out if p is not None]

        assert len(readings) == 2 * 60
        # 60 readings per station / SEQ_LEN per cycle -> complete cycles only
        assert len(preds) == 2 * (60 // SEQ_LEN)
        assert all(r._tags['source'] == 'backfill' for r in readings)
        times = [r._time.timestamp() for r in readings]
        assert max(times) < end and times == sorted(times)
        pred_points = [recs[1] for recs, p in out if p is not None]
        assert all('correct' in p._fields for p in pred_points)

    def test_backfill_stations_match_live_simulator(self, tiny_checkpoint):
        from src.dashboard.seed_history import simulate_history
        from src.data.mqtt_publisher import SpectralPublisher
        from src.inference.predictor import Predictor

        predictor = Predictor(tiny_checkpoint, device=torch.device('cpu'))
        first = next(simulate_history(predictor, hours=0.01, interval_s=30, n_stations=3))
        live = SpectralPublisher({'topic': 't', 'client_id': 'c', 'qos': 1, 'keepalive': 60,
                                  'username': None, 'password': None, 'tls_ca_file': None},
                                 n_stations=3, seed=42)
        assert first[0][0]._tags['lat'] == f"{live.stations[0]['lat']:.4f}"


class TestPipelineMetrics:

    def test_points_cover_all_reports(self, schema):
        for m in ('model_evaluation', 'model_significance', 'training_history',
                  'cross_validation', 'feature_selection', 'batch_risk', 'pipeline_run'):
            assert m in schema, m

    def test_batch_risk_is_probability_weighted(self, tmp_path):
        from src.dashboard.publish_pipeline_metrics import build_points

        pd.DataFrame([{'Model': 'X', 'n_test': 1, 'accuracy': 1.0, 'f1_macro': 1.0}]).to_csv(
            tmp_path / 'model_comparison.csv', index=False)
        pd.DataFrame([{'sample_id': i, 'split': 'test', 'prob_class_0': .2,
                       'prob_class_1': .4, 'prob_class_2': .4, 'true_label': 1,
                       'predicted_label': 1, 'confidence': .4, 'lat': 1.0, 'lon': 2.0,
                       'model': 'lstm'} for i in range(3)]).to_csv(
            tmp_path / 'predictions.csv', index=False)
        points = [p for p in build_points(str(tmp_path), str(tmp_path))
                  if p._name == 'batch_risk']
        assert len(points) == 3
        assert len({p._time for p in points}) == 3  # distinct -> no overwrite
        np.testing.assert_allclose([p._fields['risk_score'] for p in points], 0.6)

    def test_missing_reports_is_a_clear_error(self, tmp_path):
        from src.dashboard.publish_pipeline_metrics import build_points
        with pytest.raises(FileNotFoundError, match='run_pipeline'):
            build_points(str(tmp_path), str(tmp_path))
