"""
Shared fixtures. Tests run on a committed snapshot of the real CDM +
Sentinel-2 sequences (tests/fixtures/, built by `python -m src.real_data
temporal`), preprocessed once with the site-grouped split, plus a tiny
trained LSTM checkpoint, so API / inference / streaming tests exercise a
REAL model on real readings instead of silently accepting "model not loaded".
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessor import run_preprocessing_pipeline

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
SEQUENCES_CSV = os.path.join(FIXTURES, 'cdm_s2_sequences.csv')
SAMPLES_CSV = os.path.join(FIXTURES, 'cdm_s2_samples.csv')
SEQ_LEN = int(pd.read_csv(SEQUENCES_CSV, usecols=['sample_id'])
              .groupby('sample_id').size().iloc[0])


@pytest.fixture(scope='session')
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp('data')
    data = run_preprocessing_pipeline(SEQUENCES_CSV, str(root / 'processed'), seq_len=SEQ_LEN,
                                      seed=7, group_col='site_id')
    data['raw_csv'] = SEQUENCES_CSV
    data['processed_dir'] = str(root / 'processed')
    return data


@pytest.fixture(scope='session')
def tiny_checkpoint(tiny_data, tmp_path_factory):
    from src.training.train import train_from_arrays

    out = tmp_path_factory.mktemp('ckpt')
    logging.disable(logging.INFO)
    try:
        history = train_from_arrays(
            'lstm', tiny_data, tiny_data['scaler_params'], torch.device('cpu'),
            epochs=3, batch_size=16, hidden_dim=16, patience=3, seed=7,
            checkpoint_dir=str(out), log_dir=str(out),
        )
    finally:
        logging.disable(logging.NOTSET)
    return history['checkpoint_path']


@pytest.fixture(scope='session')
def raw_test_sequences(tiny_data):
    """Raw (unscaled) band readings for the test split, in ids_test order."""
    from src.constants import BANDS

    df = pd.read_csv(tiny_data['raw_csv'])
    seqs = [df[df.sample_id == sid].sort_values('timestamp')[BANDS].to_numpy()[:SEQ_LEN]
            for sid in tiny_data['ids_test']]
    return np.stack(seqs)


@pytest.fixture(scope='session')
def raw_test_context(tiny_data):
    """Acquisition dates (n, seq_len) and site lat / lon (n,) for the test split."""
    df = pd.read_csv(tiny_data['raw_csv'])
    seqs = [df[df.sample_id == sid].sort_values('timestamp').iloc[:SEQ_LEN]
            for sid in tiny_data['ids_test']]
    return {'dates': np.stack([s['date'].to_numpy() for s in seqs]),
            'lat': np.array([s['lat'].iloc[0] for s in seqs]),
            'lon': np.array([s['lon'].iloc[0] for s in seqs])}


def context_at(ctx: dict, i: int) -> dict:
    """The context of test sequence i, as predict_raw keyword arguments."""
    return {'dates': ctx['dates'][i], 'lat': float(ctx['lat'][i]), 'lon': float(ctx['lon'][i])}


@pytest.fixture
def scaler_params(tiny_data):
    with open(f"{tiny_data['processed_dir']}/scaler_params.json") as f:
        return json.load(f)


@pytest.fixture(scope='session')
def replay_sites():
    """Real sequences grouped by site, as the MQTT replay publisher loads them."""
    from src.data.mqtt_publisher import load_replay_sequences

    return load_replay_sequences(SEQUENCES_CSV, sample_meta_csv=None, split=None)
