"""
Synthetic Spectral Data Generator
=================================
Generates realistic multispectral time-series data mimicking AS7265x + SWIR
sensor readings for crop disease detection research.

Classes:
    0 = Healthy crop
    1 = Early stress (pre-symptomatic)
    2 = Active disease

Spectral Bands:
    - Red-Edge: 705nm, 740nm, 783nm
    - NIR: 842nm
    - SWIR: 1610nm, 2190nm

Signal model
------------
Each simulated plant gets its own trajectory rather than a single shared
linear ramp, so that same-label samples are not near-duplicates of one
another and disease is not trivially linearly separable in a scatter plot:

- **Nonlinear onset**: disease severity follows a logistic (S-curve) growth
  from a random per-sample onset timestep and rate, instead of declining
  from t=0 at a fixed slope. "Early Stress" samples get a late, slow,
  low-amplitude onset (genuinely pre-symptomatic — often mostly flat over
  the observed window), while "Active Disease" samples onset earlier and
  climb faster/higher.
- **Plant-level heterogeneity**: a per-sample random baseline offset per
  band (genotype / canopy density / age variability) so healthy plants
  don't all share one reflectance signature.
- **Field-level spatial structure**: a small baseline shift derived from
  the sample's GPS zone (soil type / cultivar), so nearby fields look more
  alike than distant ones — feeds the geospatial risk-mapping story.
- **Environmental confounder**: a smooth, per-sample, label-independent
  oscillation (irrigation cycles / weather) that perturbs the
  moisture-sensitive SWIR bands regardless of disease state, so SWIR
  movement alone is not a reliable disease signal.
- **Correlated + heteroscedastic sensor noise**: a shared per-timestep
  illumination factor multiplies all bands together (as real changing
  light conditions would), on top of independent per-band noise whose
  scale differs by band (SWIR channels are noisier than Red-Edge/NIR).
"""

import argparse
import os
import numpy as np
import pandas as pd


# Spectral band names matching AS7265x + SWIR sensor configuration
BANDS = [
    'red_edge_705', 'red_edge_740', 'red_edge_783',
    'nir_842', 'swir_1610', 'swir_2190'
]

# Base reflectance values for healthy vegetation (typical values)
BASE_REFLECTANCE = {
    'red_edge_705': 0.35,
    'red_edge_740': 0.55,
    'red_edge_783': 0.65,
    'nir_842': 0.80,
    'swir_1610': 0.20,
    'swir_2190': 0.15
}

# Disease class labels
LABEL_NAMES = {0: 'Healthy', 1: 'Early Stress', 2: 'Active Disease'}

# Per-band maximum absolute reflectance shift at full severity (severity=1.0).
# Negative = degrades with disease (chlorophyll/canopy breakdown), positive =
# rises with disease (water-stress signature in SWIR).
STRESS_MAX_EFFECT = {
    'red_edge_705': -0.10,
    'red_edge_740': -0.14,
    'red_edge_783': -0.18,
    'nir_842':       -0.22,
    'swir_1610':      0.09,
    'swir_2190':      0.07,
}

# Per-band sensitivity to the label-independent environmental confounder
# (irrigation / weather), mostly a SWIR (moisture) effect.
ENV_SENSITIVITY = {
    'red_edge_705': 0.05,
    'red_edge_740': 0.05,
    'red_edge_783': 0.05,
    'nir_842':       0.05,
    'swir_1610':     1.00,
    'swir_2190':     0.85,
}

# Per-band relative sensor-noise scale (SWIR channels are noisier).
BAND_NOISE_SCALE = {
    'red_edge_705': 1.0,
    'red_edge_740': 1.0,
    'red_edge_783': 1.0,
    'nir_842':       0.9,
    'swir_1610':     1.4,
    'swir_2190':     1.6,
}


def field_zone_baseline(lat: float, lon: float, seed: int = 0) -> dict[str, float]:
    """
    Derive a small, spatially-smooth per-band baseline offset from a GPS
    location, simulating soil type / cultivar zone effects so that nearby
    fields share a similar baseline instead of every sample being i.i.d.

    Deterministic given (lat, lon, seed) — does not use Python's built-in
    hash() so results are stable across processes/runs.

    Parameters
    ----------
    lat, lon : float
        GPS coordinates of the sample.
    seed : int
        Global seed to decorrelate zone baselines across dataset runs.

    Returns
    -------
    dict[str, float]
        Band name -> small additive reflectance offset.
    """
    zone_lat = round(lat * 10)  # ~0.1 degree (~11km) zones
    zone_lon = round(lon * 10)
    zone_seed = (zone_lat * 73856093) ^ (zone_lon * 19349663) ^ (seed * 83492791)
    zone_rng = np.random.default_rng(abs(int(zone_seed)) % (2**31))
    return {b: float(zone_rng.normal(0, 0.02)) for b in BASE_REFLECTANCE}


def generate_spectral_sequence(
    n_timesteps: int = 30,
    label: int = 0,
    noise: float = 0.02,
    seed: int | None = None,
    field_baseline: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Generate a single spectral time-series for one crop sample.

    For diseased crops (label > 0), NIR and Red-Edge bands degrade along a
    nonlinear (logistic) onset curve with a per-sample random start time,
    rate, and severity ceiling — simulating that real disease progression
    is not a fixed-rate linear decline shared by every plant. A
    label-independent environmental confounder and correlated sensor noise
    are layered on top, so the disease signal must be disentangled from
    ordinary field/sensor variability rather than read off directly.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps in the sequence.
    label : int
        Disease class: 0=Healthy, 1=Early Stress, 2=Active Disease.
    noise : float
        Base standard deviation of Gaussian sensor noise (scaled per band).
    seed : int or None
        Random seed for reproducibility.
    field_baseline : dict[str, float] or None
        Optional per-band additive offset from `field_zone_baseline`,
        representing this sample's soil/cultivar zone. Defaults to no
        offset when generating a sequence in isolation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each spectral band, label, and timestamp.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    field_baseline = field_baseline or {}

    # Plant-level heterogeneity: each sample gets its own baseline offset
    # per band (genotype / canopy density / age), fixed for the sequence.
    plant_offset = {b: rng.normal(0, 0.04) for b in BASE_REFLECTANCE}

    # Nonlinear, per-sample disease trajectory.
    if label == 0:
        max_severity, onset, rate = 0.0, float(n_timesteps), 0.0
    elif label == 1:
        # Early Stress: pre-symptomatic — subtle, late-onset, slow-building.
        max_severity = rng.uniform(0.12, 0.32)
        onset = rng.uniform(0.45, 0.85) * n_timesteps
        rate = rng.uniform(0.15, 0.35)
    else:
        # Active Disease: earlier onset, stronger and faster progression.
        max_severity = rng.uniform(0.45, 0.85)
        onset = rng.uniform(0.05, 0.35) * n_timesteps
        rate = rng.uniform(0.25, 0.55)

    # Label-independent environmental confounder (irrigation / weather),
    # smooth and oscillatory, mainly perturbing moisture-sensitive SWIR.
    env_phase = rng.uniform(0, 2 * np.pi)
    env_amplitude = rng.uniform(0.0, 0.035)
    env_period = rng.uniform(6, 14)

    illum_std = 0.015  # shared per-timestep illumination noise (correlates bands)

    sequence = []
    for t in range(n_timesteps):
        severity = (
            max_severity / (1.0 + np.exp(-rate * (t - onset)))
            if max_severity > 0 else 0.0
        )
        env_effect = env_amplitude * np.sin(2 * np.pi * t / env_period + env_phase)
        illum = rng.normal(1.0, illum_std)

        row = {}
        for band, base_val in BASE_REFLECTANCE.items():
            value = base_val * (1.0 + plant_offset[band]) + field_baseline.get(band, 0.0)
            value += STRESS_MAX_EFFECT[band] * severity
            value += ENV_SENSITIVITY[band] * env_effect
            value *= illum
            value += rng.normal(0, noise * BAND_NOISE_SCALE[band])
            row[band] = round(max(0.0, min(1.0, value)), 6)

        row['label'] = label
        row['timestamp'] = t
        sequence.append(row)

    return pd.DataFrame(sequence)


def generate_dataset(
    n_samples: int = 500,
    seq_len: int = 30,
    noise: float = 0.02,
    class_distribution: tuple[float, ...] = (0.5, 0.3, 0.2),
    center_lat: float = 12.97,
    center_lon: float = 77.59,
    gps_spread: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a full synthetic spectral dataset with multiple samples.

    Each sample gets a random GPS coordinate within `gps_spread` degrees
    of the center coordinates (simulating a farm region), and a per-band
    baseline offset derived from that location's zone (see
    `field_zone_baseline`) so spatially close samples share soil/cultivar
    characteristics rather than being fully independent.

    Parameters
    ----------
    n_samples : int
        Number of crop samples to generate.
    seq_len : int
        Number of timesteps per sample.
    noise : float
        Gaussian noise level.
    class_distribution : tuple
        Probability distribution for classes [Healthy, Early Stress, Active Disease].
    center_lat : float
        Central latitude for GPS simulation.
    center_lon : float
        Central longitude for GPS simulation.
    gps_spread : float
        Max offset in degrees from center for GPS coordinates.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Combined dataset with all samples, including sample_id, lat, lon.
    """
    rng = np.random.default_rng(seed)
    dfs = []

    for i in range(n_samples):
        label = int(rng.choice([0, 1, 2], p=class_distribution))
        lat = round(center_lat + rng.uniform(-gps_spread, gps_spread), 6)
        lon = round(center_lon + rng.uniform(-gps_spread, gps_spread), 6)
        field_baseline = field_zone_baseline(lat, lon, seed=seed)

        df = generate_spectral_sequence(
            n_timesteps=seq_len,
            label=label,
            noise=noise,
            seed=int(rng.integers(0, 2**31)),
            field_baseline=field_baseline
        )
        df['sample_id'] = i
        df['lat'] = lat
        df['lon'] = lon
        dfs.append(df)

    dataset = pd.concat(dfs, ignore_index=True)

    # Reorder columns for readability
    col_order = ['sample_id', 'timestamp'] + BANDS + ['label', 'lat', 'lon']
    dataset = dataset[col_order]

    return dataset


def main():
    """CLI entrypoint for generating synthetic spectral data."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic spectral crop disease dataset'
    )
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of crop samples (default: 500)')
    parser.add_argument('--seq-len', type=int, default=30,
                        help='Timesteps per sample (default: 30)')
    parser.add_argument('--noise', type=float, default=0.02,
                        help='Gaussian noise level (default: 0.02)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='data/raw/spectral_dataset.csv',
                        help='Output CSV path (default: data/raw/spectral_dataset.csv)')
    args = parser.parse_args()

    print(f"Generating synthetic spectral dataset...")
    print(f"  Samples: {args.samples}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Noise: {args.noise}")
    print(f"  Seed: {args.seed}")

    dataset = generate_dataset(
        n_samples=args.samples,
        seq_len=args.seq_len,
        noise=args.noise,
        seed=args.seed
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dataset.to_csv(args.output, index=False)

    # Print summary
    total_rows = len(dataset)
    label_counts = dataset.groupby('sample_id')['label'].first().value_counts()
    print(f"\nDataset saved to: {args.output}")
    print(f"  Total rows: {total_rows}")
    print(f"  Total samples: {args.samples}")
    print(f"  Label distribution:")
    for label_val, count in sorted(label_counts.items()):
        print(f"    {LABEL_NAMES[label_val]}: {count} ({count/args.samples*100:.1f}%)")


if __name__ == '__main__':
    main()
