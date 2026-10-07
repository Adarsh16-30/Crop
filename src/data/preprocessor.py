"""
Data Preprocessing Pipeline
============================
Handles normalization, sliding window creation, train/val/test splitting,
and feature aggregation for both temporal (LSTM) and classical (RF/XGBoost) models.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.features.spectral_indices import compute_indices, get_feature_columns


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler normalization to feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Columns to normalize.
    scaler : StandardScaler or None
        Pre-fitted scaler. If None, a new one is fitted on df.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Normalized dataframe and the fitted scaler.
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
    feature_cols: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create fixed-length sequences for temporal models (LSTM, CNN-LSTM).

    Groups data by sample_id and extracts a (seq_len, n_features) matrix
    per sample. Samples with fewer timesteps than seq_len are padded;
    those with more are truncated.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with sample_id, timestamp, features, and label.
    seq_len : int
        Target sequence length.
    feature_cols : list[str] or None
        Feature columns to include. If None, uses all raw + index columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        X : shape (n_samples, seq_len, n_features)
        y : shape (n_samples,) — integer labels
        sample_ids : shape (n_samples,) — corresponding sample IDs
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(include_raw=True, include_indices=True)

    X_list = []
    y_list = []
    id_list = []

    for sid, sample in df.groupby('sample_id', sort=False):
        sample = sample.sort_values('timestamp')
        features = sample[feature_cols].values
        label = int(sample['label'].iloc[0])

        # Pad or truncate to seq_len
        if len(features) < seq_len:
            pad = np.zeros((seq_len - len(features), features.shape[1]))
            features = np.vstack([features, pad])
        elif len(features) > seq_len:
            features = features[:seq_len]

        X_list.append(features)
        y_list.append(label)
        id_list.append(sid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    sample_ids_arr = np.array(id_list)

    return X, y, sample_ids_arr


def create_aggregated_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create aggregated (mean + std per sample) feature vectors for
    classical ML models (Random Forest, XGBoost).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with sample_id and features.
    feature_cols : list[str] or None
        Feature columns. If None, uses all raw + index columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        X_agg : shape (n_samples, n_features * 2) — mean + std per feature
        y : shape (n_samples,)
        sample_ids : shape (n_samples,)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(include_raw=True, include_indices=True)

    grouped = df.groupby('sample_id')

    means = grouped[feature_cols].mean()
    stds = grouped[feature_cols].std().fillna(0)
    labels = grouped['label'].first()

    # Rename columns to distinguish mean vs std
    mean_cols = [f'{c}_mean' for c in feature_cols]
    std_cols = [f'{c}_std' for c in feature_cols]
    means.columns = mean_cols
    stds.columns = std_cols

    combined = pd.concat([means, stds], axis=1)

    X_agg = combined.values.astype(np.float32)
    y = labels.values.astype(np.int64)
    sample_ids = combined.index.values

    return X_agg, y, sample_ids


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """
    Split data into train/val/test sets with stratification.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Label array.
    sample_ids : np.ndarray
        Sample ID array.
    val_size : float
        Fraction for validation set.
    test_size : float
        Fraction for test set.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test,
        ids_train, ids_val, ids_test.
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, sample_ids,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # Second split: train vs val (adjusted ratio)
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=adjusted_val_size,
        stratify=y_temp,
        random_state=seed
    )

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'ids_train': ids_train, 'ids_val': ids_val, 'ids_test': ids_test,
    }


def run_preprocessing_pipeline(
    input_csv: str = 'data/raw/spectral_dataset.csv',
    output_dir: str = 'data/processed',
    seq_len: int = 30,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """
    Full preprocessing pipeline: load → compute indices → normalize →
    create sequences → split → save.

    Parameters
    ----------
    input_csv : str
        Path to raw spectral CSV.
    output_dir : str
        Directory to save processed numpy arrays.
    seq_len : int
        Sequence length for temporal models.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, np.ndarray]
        Split data dictionary.
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Compute vegetation indices once and reuse the same normalized frame
    print("Computing vegetation indices...")
    df = compute_indices(df, inplace=True)

    # Get feature columns
    feature_cols = get_feature_columns(include_raw=True, include_indices=True)

    # Normalize once for both temporal and aggregated features
    print("Normalizing features...")
    df_norm, scaler = normalize_features(df, feature_cols)

    # Create temporal sequences for LSTM / CNN-LSTM
    print(f"Creating temporal sequences (seq_len={seq_len})...")
    X_seq, y_seq, ids_seq = create_sequences(
        df_norm,
        seq_len=seq_len,
        feature_cols=feature_cols
    )

    # Create aggregated features for classical ML from the same frame
    print("Creating aggregated features for classical ML...")
    X_agg, y_agg, ids_agg = create_aggregated_features(
        df_norm,
        feature_cols=feature_cols
    )

    # Split temporal data
    print("Splitting temporal data (70/15/15)...")
    splits_seq = split_data(X_seq, y_seq, ids_seq, seed=seed)

    # Split aggregated data (same split by aligning sample IDs)
    print("Splitting aggregated data...")
    splits_agg = split_data(X_agg, y_agg, ids_agg, seed=seed)

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving processed data to {output_dir}/...")

    # Temporal sequences
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        np.save(os.path.join(output_dir, f'{key}.npy'), splits_seq[key])
    for key in ['ids_train', 'ids_val', 'ids_test']:
        np.save(os.path.join(output_dir, f'{key}.npy'), splits_seq[key])

    # Aggregated features
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        np.save(os.path.join(output_dir, f'{key}_agg.npy'), splits_agg[key])
    for key in ['ids_train', 'ids_val', 'ids_test']:
        np.save(os.path.join(output_dir, f'{key}_agg.npy'), splits_agg[key])

    # Save scaler parameters for inference
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': feature_cols
    }
    import json
    with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)

    # Print summary
    print(f"\nPreprocessing complete!")
    print(f"  Temporal sequences: X_train={splits_seq['X_train'].shape}, "
          f"X_val={splits_seq['X_val'].shape}, X_test={splits_seq['X_test'].shape}")
    print(f"  Aggregated features: X_train={splits_agg['X_train'].shape}, "
          f"X_val={splits_agg['X_val'].shape}, X_test={splits_agg['X_test'].shape}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    return splits_seq


def main():
    """CLI entrypoint for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess spectral dataset')
    parser.add_argument('--input', type=str, default='data/raw/spectral_dataset.csv',
                        help='Input CSV path')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--seq-len', type=int, default=30,
                        help='Sequence length for temporal models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    run_preprocessing_pipeline(
        input_csv=args.input,
        output_dir=args.output,
        seq_len=args.seq_len,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
