"""
Shared fixtures: a tiny synthetic dataset, preprocessed once, and a tiny
trained LSTM checkpoint, so API / inference / streaming tests exercise a
REAL model instead of silently accepting "model not loaded".
"""

import json
import logging

import numpy as np
import pytest
import torch

from src.data.data_generator import generate_dataset
from src.data.preprocessor import run_preprocessing_pipeline

SEQ_LEN = 12


@pytest.fixture(scope='session')
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp('data')
    csv = root / 'raw.csv'
    generate_dataset(n_samples=150, seq_len=SEQ_LEN, seed=7).to_csv(csv, index=False)
    data = run_preprocessing_pipeline(str(csv), str(root / 'processed'), seq_len=SEQ_LEN, seed=7)
    data['raw_csv'] = str(csv)
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
    import pandas as pd

    from src.constants import BANDS

    df = pd.read_csv(tiny_data['raw_csv'])
    seqs = [df[df.sample_id == sid].sort_values('timestamp')[BANDS].to_numpy()[:SEQ_LEN]
            for sid in tiny_data['ids_test']]
    return np.stack(seqs)


@pytest.fixture
def scaler_params(tiny_data):
    with open(f"{tiny_data['processed_dir']}/scaler_params.json") as f:
        return json.load(f)
