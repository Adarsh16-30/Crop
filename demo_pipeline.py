"""
Demo Pipeline Runner (Phase 1 + Phase 2 + Phase 3 + Phase 4)
===============================================================
Runs data generation, preprocessing, model training (Random Forest,
XGBoost, LSTM, CNN-LSTM), quantum-inspired QUBO feature selection, and
geo-spatial risk mapping for a live walkthrough. Reuses the exact phase
functions from run_pipeline.py -- no source files are modified, this is
just a thinner entrypoint that stops after Phase 4 and skips the emoji
banner/summary.

Usage:
    python demo_pipeline.py --samples 1500 --seq-len 30 --epochs 50
"""

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    from run_pipeline import (
        phase1_data_foundation, phase2_model_training,
        phase3_quantum_optimization, phase4_geospatial_mapping
    )

    parser = argparse.ArgumentParser(
        description='Crop Disease Detection Pipeline (Data + Models + Quantum + Geo-Spatial)'
    )
    parser.add_argument('--samples', type=int, default=1500,
                        help='Number of synthetic samples (default: 1500)')
    parser.add_argument('--seq-len', type=int, default=30,
                        help='Sequence length (default: 30)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-quantum', action='store_true',
                        help='Skip quantum feature selection phase')
    args = parser.parse_args()

    start = time.time()
    logger.info("Crop Disease Detection Pipeline")
    logger.info(f"   Samples: {args.samples} | Seq Length: {args.seq_len} | "
                f"Epochs: {args.epochs}")

    # Phase 1: synthetic data generation + preprocessing
    splits = phase1_data_foundation(args)

    # Phase 2: classical ML + deep learning model training
    phase2_model_training(args, splits)

    # Phase 3: quantum-inspired QUBO feature selection
    quantum_results = phase3_quantum_optimization(args)

    # Phase 4: geo-spatial risk mapping
    phase4_geospatial_mapping(args)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete! Total time: {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Generated artifacts:")
    logger.info("  data/raw/spectral_dataset.csv")
    logger.info("  data/processed/*.npy")
    logger.info("  models/checkpoints/*.pt, *.joblib")
    logger.info("  reports/model_comparison.csv")
    logger.info("  reports/*_confusion_matrix.png, *_roc_curves.png, *_training_curves.png")
    if quantum_results:
        logger.info("  reports/qubo_feature_selection_results.csv")
        logger.info("  reports/qubo_comparison.json")
    logger.info("  reports/disease_risk_map.html")
    logger.info("  reports/zone_risk_map.html")
    logger.info("  reports/zone_risk_scores.geojson")


if __name__ == '__main__':
    main()
