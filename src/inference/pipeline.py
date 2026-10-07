"""
Feature pipeline: raw band readings -> model-ready standardized features.

This is the ONLY place raw sensor values are turned into model input at
inference time. It mirrors training exactly: compute the vegetation
indices, then standardize with the scaler fitted on the training split.
"""

import json
from dataclasses import dataclass

import numpy as np

from src.constants import BANDS
from src.features.spectral_indices import get_feature_columns, raw_to_features

# |z| above this on any feature marks an input as out-of-distribution.
DEFAULT_OOD_Z = 8.0


@dataclass
class FeaturePipeline:
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]
    seq_len: int

    @classmethod
    def from_dict(cls, params: dict) -> 'FeaturePipeline':
        names = list(params['feature_names'])
        if names != get_feature_columns():
            raise ValueError(
                f"Scaler feature order {names} does not match the code's "
                f"feature order {get_feature_columns()}"
            )
        return cls(
            mean=np.asarray(params['mean'], dtype=np.float64),
            scale=np.asarray(params['scale'], dtype=np.float64),
            feature_names=names,
            seq_len=int(params['seq_len']),
        )

    @classmethod
    def from_json(cls, path: str) -> 'FeaturePipeline':
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            'mean': self.mean.tolist(),
            'scale': self.scale.tolist(),
            'feature_names': list(self.feature_names),
            'seq_len': self.seq_len,
        }

    @property
    def n_raw(self) -> int:
        return len(BANDS)

    def standardize(self, features: np.ndarray) -> np.ndarray:
        """Standardize already-computed (..., 12) features."""
        z = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return z.astype(np.float32)

    def transform_raw(self, raw: np.ndarray) -> np.ndarray:
        """
        Raw bands (..., seq_len, 6) -> standardized features (..., seq_len, 12).

        Raises
        ------
        ValueError
            On wrong band count, wrong sequence length or non-finite values.
        """
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim < 2 or raw.shape[-1] != self.n_raw:
            raise ValueError(
                f"Expected readings shaped (seq_len, {self.n_raw}), got {raw.shape}"
            )
        if raw.shape[-2] != self.seq_len:
            raise ValueError(
                f"Model was trained on sequences of exactly {self.seq_len} "
                f"timesteps, got {raw.shape[-2]}"
            )
        if not np.isfinite(raw).all():
            raise ValueError("Readings contain NaN or infinite values")
        return self.standardize(raw_to_features(raw))

    @staticmethod
    def max_abs_z(scaled: np.ndarray) -> np.ndarray:
        """Largest |z| per sequence; input (n, seq_len, F) -> (n,)."""
        scaled = np.asarray(scaled)
        return np.abs(scaled).reshape(scaled.shape[0], -1).max(axis=1)
