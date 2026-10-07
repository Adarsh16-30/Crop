"""
Data Preprocessing Pipeline
============================
load -> compute indices -> split by sample -> fit scaler on TRAIN rows only
-> build temporal sequences (LSTM / CNN-LSTM) and aggregated features
(RF / XGBoost) from the same sample-id splits -> save.

Every sample is split exactly once by `sample_id`, so the temporal and
aggregated datasets always share identical train/val/test membership.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.spectral_indices import compute_indices, get_feature_columns

logger = logging.getLogger(__name__)

AGG_STATS = ['mean', 'std', 'slope', 'delta']
# Number of timesteps averaged at each end of a sequence for the delta stat.
DELTA_WINDOW = 3


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler normalization to feature columns.

    If `scaler` is None a new one is fitted on `df`; callers must only ever
    pass training rows in that case.
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 30,
    feature_cols: list[str] | None = None,
    sample_ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create fixed-length (seq_len, n_features) sequences, one per sample.

    Samples with fewer than `seq_len` timesteps are dropped (with a warning)
    rather than zero-padded: after standardization a zero is a plausible
    "average" reading, so padding would silently inject fake data. Longer
    samples are truncated to their first `seq_len` timesteps.

    Parameters
    ----------
    sample_ids : np.ndarray or None
        If given, output rows follow this order (ids missing from df, or too
        short, are skipped). Otherwise the order of first appearance in df.

    Returns
    -------
    X : (n_samples, seq_len, n_features) float32
    y : (n_samples,) int64
    sample_ids : (n_samples,)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    groups = {sid: g for sid, g in df.groupby('sample_id', sort=False)}
    order = list(groups) if sample_ids is None else list(sample_ids)

    X_list, y_list, id_list, short = [], [], [], []
    for sid in order:
        sample = groups.get(sid)
        if sample is None:
            continue
        sample = sample.sort_values('timestamp')
        if len(sample) < seq_len:
            short.append(sid)
            continue
        X_list.append(sample[feature_cols].to_numpy()[:seq_len])
        y_list.append(int(sample['label'].iloc[0]))
        id_list.append(sid)

    if short:
        logger.warning("Dropped %d samples shorter than seq_len=%d", len(short), seq_len)

    X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, len(feature_cols))
    return X, np.array(y_list, dtype=np.int64), np.array(id_list)


def aggregate_feature_names(feature_cols: list[str] | None = None) -> list[str]:
    """Names of the aggregated features, matching create_aggregated_features."""
    if feature_cols is None:
        feature_cols = get_feature_columns()
    return [f'{c}_{stat}' for stat in AGG_STATS for c in feature_cols]


def create_aggregated_features(X_seq: np.ndarray) -> np.ndarray:
    """
    Summarize each sequence into a flat vector for classical ML.

    Per feature: mean, std, slope (least-squares trend per timestep) and
    delta (mean of the last few timesteps minus mean of the first few).
    Slope and delta give RF / XGBoost the temporal trend information that a
    plain mean/std summary throws away, so the comparison against the
    sequence models is not simply "has time vs. has no time".

    Parameters
    ----------
    X_seq : (n_samples, seq_len, n_features)

    Returns
    -------
    (n_samples, n_features * 4), columns ordered as aggregate_feature_names().
    """
    X_seq = np.asarray(X_seq, dtype=np.float64)
    n, seq_len, _ = X_seq.shape
    t = np.arange(seq_len, dtype=np.float64)
    t_centered = t - t.mean()

    mean = X_seq.mean(axis=1)
    std = X_seq.std(axis=1, ddof=1) if seq_len > 1 else np.zeros_like(mean)
    slope = np.einsum('t,ntf->nf', t_centered, X_seq - mean[:, None, :]) / \
        max(float((t_centered ** 2).sum()), 1e-12)
    w = min(DELTA_WINDOW, seq_len)
    delta = X_seq[:, -w:].mean(axis=1) - X_seq[:, :w].mean(axis=1)

    return np.concatenate([mean, std, slope, delta], axis=1).astype(np.float32)


def split_sample_ids(
    sample_table: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """
    Stratified split of sample ids into train / val / test.

    Parameters
    ----------
    sample_table : DataFrame with one row per sample and columns
        `sample_id`, `label`.
    """
    ids = sample_table['sample_id'].to_numpy()
    labels = sample_table['label'].to_numpy()

    ids_temp, ids_test, y_temp, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    ids_train, ids_val = train_test_split(
        ids_temp, test_size=val_size / (1 - test_size),
        stratify=y_temp, random_state=seed
    )
    return {'train': ids_train, 'val': ids_val, 'test': ids_test}


def run_preprocessing_pipeline(
    input_csv: str = 'data/raw/spectral_dataset.csv',
    output_dir: str = 'data/processed',
    seq_len: int = 30,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> dict[str, np.ndarray]:
    """
    Full preprocessing pipeline. See module docstring for the steps.

    Saves to `output_dir`:
        X_{split}.npy, y_{split}.npy, ids_{split}.npy          (sequences)
        X_{split}_agg.npy, y_{split}_agg.npy, ids_{split}_agg.npy
        scaler_params.json     (fitted on train rows only)
        agg_feature_names.json
        sample_meta.csv        (sample_id, label, lat, lon, split)

    Returns
    -------
    dict with X_/y_/ids_ keys for train/val/test (sequences) plus the same
    keys suffixed with `_agg`, and `agg_feature_names`.
    """
    logger.info("Loading data from %s", input_csv)
    df = pd.read_csv(input_csv)

    logger.info("Computing vegetation indices")
    df = compute_indices(df, inplace=True)
    feature_cols = get_feature_columns()

    sample_table = (
        df.groupby('sample_id', sort=True)
          .agg(label=('label', 'first'), lat=('lat', 'first'), lon=('lon', 'first'))
          .reset_index()
    )
    split_ids = split_sample_ids(sample_table, val_size, test_size, seed)

    # Fit the scaler on training rows only -- val/test statistics must not
    # leak into the normalization the model is trained with.
    train_rows = df['sample_id'].isin(split_ids['train'])
    _, scaler = normalize_features(df.loc[train_rows], feature_cols)
    df_norm, _ = normalize_features(df, feature_cols, scaler=scaler)

    out: dict[str, np.ndarray] = {}
    split_col = pd.Series(index=sample_table['sample_id'], dtype=object)
    for split, ids in split_ids.items():
        X, y, kept_ids = create_sequences(df_norm, seq_len, feature_cols, sample_ids=ids)
        out[f'X_{split}'], out[f'y_{split}'], out[f'ids_{split}'] = X, y, kept_ids
        out[f'X_{split}_agg'] = create_aggregated_features(X)
        out[f'y_{split}_agg'], out[f'ids_{split}_agg'] = y, kept_ids
        split_col.loc[kept_ids] = split
    agg_names = aggregate_feature_names(feature_cols)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving processed data to %s/", output_dir)
    for key, arr in out.items():
        np.save(os.path.join(output_dir, f'{key}.npy'), arr)

    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': feature_cols,
        'seq_len': seq_len,
        'fitted_on': 'train',
    }
    with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)
    with open(os.path.join(output_dir, 'agg_feature_names.json'), 'w') as f:
        json.dump(agg_names, f, indent=2)

    sample_table['split'] = sample_table['sample_id'].map(split_col)
    sample_table.dropna(subset=['split']).to_csv(
        os.path.join(output_dir, 'sample_meta.csv'), index=False
    )

    logger.info(
        "Preprocessing complete: train=%s val=%s test=%s | agg features=%d",
        out['X_train'].shape, out['X_val'].shape, out['X_test'].shape, len(agg_names)
    )
    out['agg_feature_names'] = agg_names
    out['scaler_params'] = scaler_params
    return out


def main():
    """CLI entrypoint for preprocessing pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess spectral dataset')
    parser.add_argument('--input', type=str, default='data/raw/spectral_dataset.csv')
    parser.add_argument('--output', type=str, default='data/processed')
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_preprocessing_pipeline(
        input_csv=args.input, output_dir=args.output,
        seq_len=args.seq_len, seed=args.seed
    )


if __name__ == '__main__':
    main()
