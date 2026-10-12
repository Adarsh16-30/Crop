"""
Publish offline pipeline results to InfluxDB
=============================================
Turns the report files written by run_pipeline.py into InfluxDB points, so
Grafana's Model Performance dashboard can show model quality next to the
live data, and track it across pipeline runs.

Measurements (all tagged with run_id):
    pipeline_run         one point per run: best model, sample counts
    model_evaluation     test metrics + 95% CIs per model
    model_significance   pairwise McNemar results
    training_history     per-epoch loss / accuracy / macro-F1 per sequence model
    cross_validation     random vs spatial-zone CV for the classical models
    feature_selection    QUBO vs top-K / mRMR / random selectors
    batch_risk           every sample's batch prediction with coordinates

    python -m src.dashboard.publish_pipeline_metrics
"""

import argparse
import logging
import math
import os
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.config import load_influx_config
from src.constants import expected_risk

logger = logging.getLogger(__name__)

TRAINING_LOGS = {'lstm': 'LSTM', 'cnn_lstm': 'CNN-LSTM'}


def _fields(row: pd.Series, skip=()) -> dict:
    """Numeric / boolean columns of a row as InfluxDB fields (NaN dropped)."""
    out = {}
    for key, value in row.items():
        if key in skip or value is None:
            continue
        if hasattr(value, 'item'):  # numpy scalar -> python
            value = value.item()
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)) and not math.isnan(value):
            out[key] = float(value)
    return out


def _point(measurement: str, tags: dict, fields: dict, when: datetime):
    from influxdb_client import Point

    point = Point(measurement).time(when)
    for k, v in tags.items():
        point = point.tag(k, str(v))
    for k, v in fields.items():
        point = point.field(k, v)
    return point


def build_points(reports_dir: str = 'reports', processed_dir: str = 'data/processed',
                 run_time: datetime | None = None) -> list:
    """Build every point from the report files that exist. Missing files are skipped."""
    comparison_csv = os.path.join(reports_dir, 'model_comparison.csv')
    if not os.path.exists(comparison_csv):
        raise FileNotFoundError(f"{comparison_csv} not found; run `python run_pipeline.py` first")
    if run_time is None:
        run_time = datetime.fromtimestamp(os.path.getmtime(comparison_csv), tz=UTC)
    run_id = run_time.strftime('%Y%m%d-%H%M%S')
    base = {'run_id': run_id}
    points = []

    comparison = pd.read_csv(comparison_csv)
    for _, row in comparison.iterrows():
        points.append(_point('model_evaluation', {**base, 'model': row['Model']},
                             _fields(row, skip=('Model',)), run_time))
    best = comparison.loc[comparison['f1_macro'].idxmax()]

    path = os.path.join(reports_dir, 'model_significance.csv')
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            points.append(_point(
                'model_significance',
                {**base, 'model_a': row['model_a'], 'model_b': row['model_b']},
                {'only_a_correct': float(row['only_a_correct']),
                 'only_b_correct': float(row['only_b_correct']),
                 'p_value': float(row['p_value']),
                 'significant': bool(row['significant_at_0.05'])}, run_time))

    for model_type, name in TRAINING_LOGS.items():
        path = os.path.join(reports_dir, f'{model_type}_training_log.csv')
        if not os.path.exists(path):
            continue
        log = pd.read_csv(path)
        last_epoch = int(log['epoch'].max())
        for _, row in log.iterrows():
            # One point per epoch, spaced 1s apart and ending at run_time,
            # so every epoch is a distinct point in time.
            when = run_time - timedelta(seconds=last_epoch - int(row['epoch']))
            points.append(_point('training_history', {**base, 'model': name},
                                 _fields(row), when))

    path = os.path.join(reports_dir, 'classical_cv.csv')
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            points.append(_point('cross_validation',
                                 {**base, 'model': row['model'], 'cv_split': row['cv_split']},
                                 _fields(row, skip=('model', 'cv_split')), run_time))

    path = os.path.join(reports_dir, 'feature_selection_comparison.csv')
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            points.append(_point('feature_selection', {**base, 'method': row['method']},
                                 _fields(row, skip=('method',)), run_time))

    n_samples = 0
    path = os.path.join(processed_dir, 'predictions.csv')
    if os.path.exists(path):
        preds = pd.read_csv(path).dropna(subset=['lat', 'lon'])
        prob_cols = [c for c in preds.columns if c.startswith('prob_class_')]
        risks = expected_risk(preds[prob_cols].to_numpy())
        for i, (row, risk) in enumerate(zip(preds.itertuples(index=False), risks)):
            # Distinct timestamps per sample: points sharing a series and a
            # timestamp would overwrite each other.
            when = run_time - timedelta(milliseconds=i)
            points.append(_point(
                'batch_risk', {**base, 'split': row.split, 'model': row.model},
                {'sample_id': int(row.sample_id), 'latitude': float(row.lat),
                 'longitude': float(row.lon), 'risk_score': float(risk),
                 'predicted_label': int(row.predicted_label),
                 'true_label': int(row.true_label), 'confidence': float(row.confidence),
                 'correct': int(row.predicted_label == row.true_label)}, when))
        n_samples = len(preds)

    points.append(_point('pipeline_run', base, {
        'best_model': str(best['Model']), 'best_f1_macro': float(best['f1_macro']),
        'best_accuracy': float(best['accuracy']), 'n_models': float(len(comparison)),
        'n_test': float(best['n_test']), 'n_samples_mapped': float(n_samples),
    }, run_time))
    return points


def publish(influx_cfg: dict, reports_dir: str = 'reports',
            processed_dir: str = 'data/processed') -> int:
    """Write the pipeline's results to InfluxDB. Returns the number of points."""
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS

    if not influx_cfg.get('token'):
        raise RuntimeError("No InfluxDB token: set INFLUX_WRITE_TOKEN or INFLUX_TOKEN")
    points = build_points(reports_dir, processed_dir)
    with InfluxDBClient(url=influx_cfg['url'], token=influx_cfg['token'],
                        org=influx_cfg['org']) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        for start in range(0, len(points), 5000):
            write_api.write(bucket=influx_cfg['bucket'], org=influx_cfg['org'],
                            record=points[start:start + 5000])
    logger.info("Published %d pipeline metric points to InfluxDB (%s)", len(points),
                influx_cfg['bucket'])
    return len(points)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Publish pipeline results to InfluxDB')
    parser.add_argument('--reports-dir', default='reports')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--influx-config', default='config/influxdb.ini')
    args = parser.parse_args()
    publish(load_influx_config(args.influx_config, token_env='INFLUX_WRITE_TOKEN'),
            args.reports_dir, args.processed_dir)


if __name__ == '__main__':
    main()
