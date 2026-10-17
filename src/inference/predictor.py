"""
Predictor: loads a self-describing checkpoint and runs calibrated inference.

A checkpoint written by src.training.train contains:
    model_state_dict  -- weights
    config            -- architecture, seq_len, scaler params, temperature,
                         label names and library versions (plain types only)
so it can be loaded with torch.load(weights_only=True) and the model
rebuilt without hardcoding any hyperparameters.
"""

import logging

import numpy as np
import torch

from src.constants import LABEL_NAMES, expected_risk
from src.inference.pipeline import DEFAULT_OOD_Z, FeaturePipeline
from src.models.factory import build_model_from_config

logger = logging.getLogger(__name__)


def load_checkpoint_dict(path: str, device: torch.device | str = 'cpu') -> dict:
    """Load a checkpoint safely (no arbitrary pickle execution)."""
    return torch.load(path, map_location=device, weights_only=True)


class Predictor:
    """Calibrated inference for one trained sequence model."""

    def __init__(self, checkpoint_path: str, device: torch.device | None = None,
                 ood_z: float = DEFAULT_OOD_Z):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = load_checkpoint_dict(checkpoint_path, self.device)
        if 'config' not in ckpt:
            raise ValueError(
                f"{checkpoint_path} has no 'config' metadata. It was written by an "
                "older version of the training script; retrain the model."
            )
        self.config: dict = ckpt['config']
        self.checkpoint_path = checkpoint_path
        self.metrics = {k: ckpt.get(k) for k in ('epoch', 'val_loss', 'val_acc', 'val_macro_f1')}

        self.model = build_model_from_config(self.config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

        self.features = FeaturePipeline.from_dict(self.config['scaler'])
        self.temperature = float(self.config.get('temperature', 1.0))
        self.ood_z = ood_z
        self.label_names = {int(k): v for k, v in
                            self.config.get('label_names', LABEL_NAMES).items()}

    @property
    def seq_len(self) -> int:
        return self.features.seq_len

    @property
    def model_type(self) -> str:
        return self.config['model_type']

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def predict_proba_scaled(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Calibrated probabilities for standardized features (n, seq_len, F)."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = []
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start:start + batch_size]).to(self.device)
                logits = self.model(xb) / self.temperature
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if not out:
            return np.empty((0, len(self.label_names)))
        return np.concatenate(out)

    @property
    def uses_context(self) -> bool:
        return self.features.uses_context

    def predict_raw(self, readings: np.ndarray, dates=None, lat=None, lon=None) -> list[dict]:
        """
        Predict from raw band readings: (seq_len, 6) or (n, seq_len, 6).

        Models trained with acquisition context also need `dates` ((seq_len,)
        or (n, seq_len)) and the field location `lat` / `lon` (scalar or (n,)).

        Each result has the predicted label, calibrated probabilities, the
        probability-weighted risk score and an out-of-distribution flag.
        """
        readings = np.asarray(readings, dtype=np.float64)
        if readings.ndim == 2:
            readings = readings[None]
            if dates is not None:
                dates = np.asarray(dates)[None]
        scaled = self.features.transform_raw(readings, dates, lat, lon)
        probs = self.predict_proba_scaled(scaled)
        max_z = FeaturePipeline.max_abs_z(scaled)
        risks = np.atleast_1d(expected_risk(probs))

        results = []
        for p, z, risk in zip(probs, max_z, risks):
            label = int(np.argmax(p))
            results.append({
                'predicted_label': label,
                'label_name': self.label_names[label],
                'confidence': float(p[label]),
                'risk_score': float(risk),
                'probabilities': [float(v) for v in p],
                'out_of_distribution': bool(z > self.ood_z),
                'max_abs_z': float(z),
            })
        return results
