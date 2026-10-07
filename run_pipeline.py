"""
End-to-End Pipeline Runner
============================
Orchestrates the complete workflow from data generation through
model training, evaluation, quantum optimization, geo-spatial mapping,
and dashboard preparation.

Usage:
    python run_pipeline.py --samples 500 --seq-len 30
    python run_pipeline.py --skip-quantum --skip-mqtt
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def phase1_data_foundation(args) -> dict:
    """Phase 1: Generate synthetic data and preprocess."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Foundation")
    logger.info("=" * 60)

    from src.data.data_generator import generate_dataset
    from src.features.spectral_indices import compute_indices
    from src.data.preprocessor import run_preprocessing_pipeline

    # Generate synthetic dataset
    logger.info(f"Generating {args.samples} samples with seq_len={args.seq_len}...")
    dataset = generate_dataset(
        n_samples=args.samples,
        seq_len=args.seq_len,
        seed=args.seed
    )

    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/spectral_dataset.csv'
    dataset.to_csv(csv_path, index=False)
    logger.info(f"Dataset saved: {csv_path} ({len(dataset)} rows)")

    # Preprocess
    splits = run_preprocessing_pipeline(
        input_csv=csv_path,
        output_dir='data/processed',
        seq_len=args.seq_len,
        seed=args.seed
    )

    return splits


def phase2_model_training(args, splits: dict) -> list:
    """Phase 2: Train classical ML and deep learning models."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Model Training")
    logger.info("=" * 60)

    import torch
    from src.models.classical_models import (
        train_random_forest, train_xgboost, evaluate_model, save_model
    )
    from src.training.train import (
        get_device, create_data_loaders, build_model, train_model, load_checkpoint
    )
    from src.evaluation.evaluate import (
        evaluate_dl_model, plot_confusion_matrix, plot_roc_curves,
        plot_training_curves, compare_models
    )

    device = get_device()
    all_results = []

    # ── Classical ML (aggregated features) ────────────
    logger.info("Training classical ML models (RF, XGBoost)...")
    X_train_agg = np.load('data/processed/X_train_agg.npy')
    y_train_agg = np.load('data/processed/y_train_agg.npy')
    X_test_agg = np.load('data/processed/X_test_agg.npy')
    y_test_agg = np.load('data/processed/y_test_agg.npy')

    # Random Forest
    rf = train_random_forest(X_train_agg, y_train_agg)
    rf_results = evaluate_model(rf, X_test_agg, y_test_agg, model_name='Random Forest')
    save_model(rf, 'models/checkpoints/rf_model.joblib')
    all_results.append(rf_results)

    # XGBoost
    xgb = train_xgboost(X_train_agg, y_train_agg)
    xgb_results = evaluate_model(xgb, X_test_agg, y_test_agg, model_name='XGBoost')
    save_model(xgb, 'models/checkpoints/xgb_model.joblib')
    all_results.append(xgb_results)

    # ── Deep Learning (temporal sequences) ────────────
    logger.info("Training deep learning models (LSTM, CNN-LSTM)...")
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )

    for model_type in ['lstm', 'cnn_lstm']:
        logger.info(f"Training {model_type}...")
        model = build_model(
            model_type=model_type,
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=128,
            dropout=0.3
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=0.001,
            patience=10,
            checkpoint_dir='models/checkpoints',
            model_name=model_type,
            log_dir='reports'
        )

        # Load best checkpoint and evaluate
        model = build_model(model_type, input_dim, seq_len)
        model = load_checkpoint(model, f'models/checkpoints/{model_type}_best.pt', device)

        dl_results = evaluate_dl_model(
            model, X_test, y_test, device,
            model_name=model_type.upper().replace('_', '-')
        )
        all_results.append(dl_results)

        # Plot confusion matrix and ROC curves
        plot_confusion_matrix(
            dl_results['confusion_matrix'],
            dl_results['model_name'],
            f'reports/{model_type}_confusion_matrix.png'
        )
        if dl_results['probabilities'] is not None:
            plot_roc_curves(
                y_test, dl_results['probabilities'],
                dl_results['model_name'],
                f'reports/{model_type}_roc_curves.png'
            )

        # Plot training curves
        log_path = f'reports/{model_type}_training_log.csv'
        if os.path.exists(log_path):
            plot_training_curves(log_path, f'reports/{model_type}_training_curves.png',
                               model_name=dl_results['model_name'])

    # Model comparison
    compare_models(all_results)

    return all_results


def phase3_quantum_optimization(args) -> dict | None:
    """Phase 3: QUBO-based feature selection."""
    if args.skip_quantum:
        logger.info("Skipping Phase 3 (quantum optimization)")
        return None

    logger.info("=" * 60)
    logger.info("PHASE 3: Quantum Feature Selection")
    logger.info("=" * 60)

    from src.models.classical_models import load_model
    from src.quantum.integration import run_quantum_feature_selection
    from src.features.spectral_indices import get_feature_columns

    rf = load_model('models/checkpoints/rf_model.joblib')
    X_train_agg = np.load('data/processed/X_train_agg.npy')
    y_train_agg = np.load('data/processed/y_train_agg.npy')
    X_test_agg = np.load('data/processed/X_test_agg.npy')
    y_test_agg = np.load('data/processed/y_test_agg.npy')

    # Feature names: mean + std for each raw + index column
    base_cols = get_feature_columns(include_raw=True, include_indices=True)
    feature_names = [f'{c}_mean' for c in base_cols] + [f'{c}_std' for c in base_cols]

    results = run_quantum_feature_selection(
        rf_model=rf,
        X_train_agg=X_train_agg,
        y_train_agg=y_train_agg,
        X_test_agg=X_test_agg,
        y_test_agg=y_test_agg,
        feature_names=feature_names,
        max_features=8,
        reports_dir='reports'
    )

    return results


def phase4_geospatial_mapping(args) -> None:
    """Phase 4: Generate geo-spatial risk maps."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Geo-Spatial Risk Mapping")
    logger.info("=" * 60)

    import torch
    from src.models.lstm_model import SpectralLSTM
    from src.training.train import load_checkpoint, get_device
    from src.geospatial.risk_mapper import (
        build_geodataframe, aggregate_zone_risk, export_geojson, get_risk_summary
    )
    from src.geospatial.heatmap import generate_risk_heatmap, generate_zone_map

    device = get_device()

    # Load test data and model
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    ids_test_path = 'data/processed/ids_test.npy'
    ids_test = np.load(ids_test_path) if os.path.exists(ids_test_path) else None

    # Load original dataset for GPS coordinates
    raw_df = pd.read_csv('data/raw/spectral_dataset.csv')
    gps_lookup = raw_df.groupby('sample_id', sort=False)[['lat', 'lon']].first()

    # Load best model
    model = SpectralLSTM(input_dim=X_test.shape[2], hidden_dim=128,
                          num_layers=2, num_classes=3)
    model_path = 'models/checkpoints/lstm_best.pt'
    if os.path.exists(model_path):
        model = load_checkpoint(model, model_path, device)
    else:
        logger.warning("LSTM checkpoint not found, using untrained model")
        model = model.to(device)
        model.eval()

    # Run predictions
    logger.info("Running predictions on test set...")
    x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    if ids_test is not None and len(ids_test) == len(X_test):
        gps_values = gps_lookup.reindex(ids_test)[['lat', 'lon']].to_numpy()
    else:
        gps_values = gps_lookup.iloc[:len(X_test)].to_numpy()

    predictions = []
    for i in range(len(X_test)):
        label = int(np.argmax(probs[i]))
        confidence = float(probs[i][label])
        lat, lon = gps_values[i]

        if np.isnan(lat) or np.isnan(lon):
            lat = 12.97 + np.random.uniform(-0.5, 0.5)
            lon = 77.59 + np.random.uniform(-0.5, 0.5)

        predictions.append({
            'lat': lat, 'lon': lon,
            'predicted_label': label,
            'confidence': confidence,
            'sample_id': i
        })

    pred_df = pd.DataFrame(predictions)

    # Save predictions for dashboard
    os.makedirs('data/processed', exist_ok=True)
    pred_df.to_csv('data/processed/predictions.csv', index=False)

    # Build GeoDataFrame
    gdf = build_geodataframe(pred_df)

    # Generate heatmap
    generate_risk_heatmap(gdf, 'reports/disease_risk_map.html')

    # Zone aggregation
    zone_gdf = aggregate_zone_risk(gdf, grid_size=0.05)
    if len(zone_gdf) > 0:
        export_geojson(zone_gdf, 'reports/zone_risk_scores.geojson')
        generate_zone_map(zone_gdf, 'reports/zone_risk_map.html')

    # Print summary
    summary = get_risk_summary(gdf)
    logger.info(f"Risk Summary:")
    for key, val in summary.items():
        logger.info(f"  {key}: {val}")


def phase5_dashboard_prep(args) -> None:
    """Phase 5: Prepare dashboard infrastructure."""
    if args.skip_mqtt:
        logger.info("Skipping MQTT setup")
        return

    logger.info("=" * 60)
    logger.info("PHASE 5: Dashboard Preparation")
    logger.info("=" * 60)

    # Verify InfluxDB connection
    try:
        from src.dashboard.influxdb_setup import setup_influxdb
        success = setup_influxdb('config/influxdb.ini')
        if not success:
            logger.warning("InfluxDB setup failed. Run 'docker compose up -d' first.")
    except Exception as e:
        logger.warning(f"InfluxDB setup skipped: {e}")

    logger.info("\nTo start the full dashboard stack:")
    logger.info("  1. docker compose up -d")
    logger.info("  2. python -m src.dashboard.mqtt_subscriber --model-path models/checkpoints/lstm_best.pt")
    logger.info("  3. python -m src.data.mqtt_publisher --profile mixed")
    logger.info("  4. Open Grafana: http://localhost:3000 (admin / cropdetection)")
    logger.info("  5. uvicorn src.api.main:app --reload --port 8000")


def main():
    parser = argparse.ArgumentParser(
        description='Quantum-Enhanced Crop Disease Detection Pipeline'
    )
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of synthetic samples (default: 500)')
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
    parser.add_argument('--skip-mqtt', action='store_true',
                        help='Skip MQTT/InfluxDB setup')
    parser.add_argument('--skip-geo', action='store_true',
                        help='Skip geo-spatial mapping phase')
    args = parser.parse_args()

    start = time.time()
    logger.info("🌱 Quantum-Enhanced Crop Disease Detection Pipeline")
    logger.info(f"   Samples: {args.samples} | Seq Length: {args.seq_len} | "
                f"Epochs: {args.epochs}")

    # Phase 1
    splits = phase1_data_foundation(args)

    # Phase 2
    all_results = phase2_model_training(args, splits)

    # Phase 3
    quantum_results = phase3_quantum_optimization(args)

    # Phase 4
    if not args.skip_geo:
        phase4_geospatial_mapping(args)
    else:
        logger.info("Skipping Phase 4 (geo-spatial mapping)")

    # Phase 5
    phase5_dashboard_prep(args)

    # Summary
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"🎉 Pipeline complete! Total time: {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("\nGenerated artifacts:")
    logger.info("  data/raw/spectral_dataset.csv")
    logger.info("  data/processed/*.npy")
    logger.info("  models/checkpoints/*.pt, *.joblib")
    logger.info("  reports/model_comparison.csv")
    logger.info("  reports/disease_risk_map.html")
    logger.info("  reports/zone_risk_scores.geojson")
    if quantum_results:
        logger.info("  reports/qubo_feature_selection_results.csv")


if __name__ == '__main__':
    main()
