"""
Shared constants used across training, inference, mapping and streaming.

Keep label names, band order and risk weights in exactly one place so the
API, MQTT bridge, geospatial mapper and evaluation can never drift apart.
"""

import numpy as np

# Raw spectral bands, in the order every model expects them.
BANDS = [
    'red_edge_705', 'red_edge_740', 'red_edge_783',
    'nir_842', 'swir_1610', 'swir_2190',
]

# Derived vegetation indices, in model input order (appended after BANDS).
INDICES = ['NDRE', 'NDVI', 'MSI', 'SWIR_ratio', 'REIP_approx', 'CRI']

LABEL_NAMES = {0: 'Healthy', 1: 'Early Stress', 2: 'Active Disease'}
NUM_CLASSES = len(LABEL_NAMES)

# Per-class risk weight. A prediction's risk is the probability-weighted
# sum of these (see expected_risk), not weight[argmax] * confidence.
RISK_WEIGHTS = np.array([0.0, 0.5, 1.0])

# Risk category bins shared by point maps, zone maps and summaries.
RISK_BINS = [-0.01, 0.2, 0.5, 0.7, 1.01]
RISK_CATEGORIES = ['Low', 'Moderate', 'High', 'Critical']


def expected_risk(probabilities) -> np.ndarray | float:
    """
    Probability-weighted risk score in [0, 1].

    Accepts a single probability vector (n_classes,) or a batch
    (n, n_classes). Unlike weight[argmax] * confidence, an uncertain
    "Healthy" prediction still carries the risk mass of the other classes.
    """
    probs = np.asarray(probabilities, dtype=float)
    risk = probs @ RISK_WEIGHTS
    return float(risk) if risk.ndim == 0 else risk


def zone_key(lat: float, lon: float) -> tuple[int, int]:
    """~0.1 degree spatial zone, matching the generator's field zones."""
    return int(round(lat * 10)), int(round(lon * 10))
