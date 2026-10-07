"""
End-to-End Pipeline Runner
============================
Phase 1  synthetic data generation + preprocessing (one shared split)
Phase 2  classical models (RF, XGBoost) with random and spatial CV
Phase 3  deep models (LSTM, CNN-LSTM) with calibration
Phase 4  quantum-inspired QUBO feature selection vs. classical selectors
         -> model comparison with 95% CIs and McNemar tests
Phase 5  geo-spatial risk maps from the best model (by validation macro-F1)
Phase 6  InfluxDB connectivity check for the live dashboard

Usage:
    python run_pipeline.py --samples 1500 --seq-len 30 --epochs 50
    python run_pipeline.py --skip-quantum --skip-mqtt
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pipeline')

RAW_CSV = 'data/raw/spectral_dataset.csv'
PROCESSED_DIR = 'data/processed'
CHECKPOINT_DIR = 'models/checkpoints'
REPORTS_DIR = 'reports'
DL_MODELS = ('lstm', 'cnn_lstm')
DL_DISPLAY = {'lstm': 'LSTM', 'cnn_lstm': 'CNN-LSTM'}


def banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def phase1_data_foundation(args) -> dict:
    """Generate synthetic data and preprocess it into a single shared split."""
    banner("PHASE 1: Data Foundation")
    from src.data.data_generator import generate_dataset
    from src.data.preprocessor import run_preprocessing_pipeline

    dataset = generate_dataset(n_samples=args.samples, seq_len=args.seq_len, seed=args.seed)
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
    dataset.to_csv(RAW_CSV, index=False)
    logger.info("Dataset saved: %s (%d rows)", RAW_CSV, len(dataset))

    data = run_preprocessing_pipeline(input_csv=RAW_CSV, output_dir=PROCESSED_DIR,
                                      seq_len=args.seq_len, seed=args.seed)
    data['sample_meta'] = pd.read_csv(os.path.join(PROCESSED_DIR, 'sample_meta.csv'))
    return data


def phase2_classical_models(args, data: dict) -> list[dict]:
    """Train / evaluate RF and XGBoost; report random and spatially grouped CV."""
    banner("PHASE 2: Classical Models")
    from src.constants import zone_key
    from src.models.classical_models import (
        cross_validate,
        evaluate_model,
        make_random_forest,
        make_xgboost,
        save_model,
        train_random_forest,
        train_xgboost,
    )

    X_tr, y_tr = data['X_train_agg'], data['y_train_agg']
    X_te, y_te = data['X_test_agg'], data['y_test_agg']

    # CV on train+val; spatial groups are the generator's ~0.1 degree zones.
    X_cv = np.concatenate([X_tr, data['X_val_agg']])
    y_cv = np.concatenate([y_tr, data['y_val_agg']])
    ids_cv = np.concatenate([data['ids_train_agg'], data['ids_val_agg']])
    meta = data['sample_meta'].set_index('sample_id').loc[ids_cv]
    groups = np.array([hash(zone_key(lat, lon)) for lat, lon in zip(meta['lat'], meta['lon'])])

    results, cv_rows = [], []
    for name, trainer, factory, path in (
        ('Random Forest', train_random_forest, make_random_forest, 'rf_model.joblib'),
        ('XGBoost', train_xgboost, make_xgboost, 'xgb_model.joblib'),
    ):
        model = trainer(X_tr, y_tr, seed=args.seed)
        results.append(evaluate_model(model, X_te, y_te, model_name=name))
        save_model(model, os.path.join(CHECKPOINT_DIR, path))
        for split_name, grp in (('random', None), ('spatial_zone', groups)):
            cv = cross_validate(factory(seed=args.seed), X_cv, y_cv, groups=grp, seed=args.seed)
            cv_rows.append({'model': name, 'cv_split': split_name, **cv})

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(REPORTS_DIR, 'classical_cv.csv'), index=False)
    logger.info("Classical CV (random vs spatial zone hold-out):\n%s",
                cv_df.round(4).to_string(index=False))
    return results


def phase3_deep_models(args, data: dict) -> list[dict]:
    """Train, calibrate and evaluate the sequence models."""
    banner("PHASE 3: Deep Learning Models")
    from src.evaluation.evaluate import (
        evaluate_dl_model,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_training_curves,
    )
    from src.inference.predictor import Predictor
    from src.training.train import get_device, train_from_arrays

    device = get_device()
    results = []
    for model_type in args.models:
        history = train_from_arrays(
            model_type, data, data['scaler_params'], device, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            weight_decay=args.weight_decay, hidden_dim=args.hidden_dim,
            dropout=args.dropout, patience=args.patience, seed=args.seed,
            checkpoint_dir=CHECKPOINT_DIR, log_dir=REPORTS_DIR
        )
        predictor = Predictor(history['checkpoint_path'], device=device)
        name = DL_DISPLAY[model_type]
        res = evaluate_dl_model(predictor, data['X_test'], data['y_test'], model_name=name)
        results.append(res)

        plot_confusion_matrix(res['confusion_matrix'], name,
                              f'{REPORTS_DIR}/{model_type}_confusion_matrix.png')
        plot_roc_curves(data['y_test'], res['probabilities'], name,
                        f'{REPORTS_DIR}/{model_type}_roc_curves.png')
        plot_training_curves(f'{REPORTS_DIR}/{model_type}_training_log.csv',
                             f'{REPORTS_DIR}/{model_type}_training_curves.png', model_name=name)
    return results


def phase4_quantum_selection(args, data: dict) -> tuple[dict | None, dict | None]:
    """QUBO feature selection; returns (results, test metrics of the QUBO-subset RF)."""
    if args.skip_quantum:
        logger.info("Skipping Phase 4 (quantum-inspired feature selection)")
        return None, None
    banner("PHASE 4: Quantum-Inspired Feature Selection")
    from src.models.classical_models import evaluate_model, load_model
    from src.quantum.integration import run_quantum_feature_selection

    rf = load_model(os.path.join(CHECKPOINT_DIR, 'rf_model.joblib'))
    results = run_quantum_feature_selection(
        rf_model=rf,
        X_train_agg=data['X_train_agg'], y_train_agg=data['y_train_agg'],
        X_test_agg=data['X_test_agg'], y_test_agg=data['y_test_agg'],
        feature_names=data['agg_feature_names'], max_features=args.max_features,
        use_qaoa=args.use_qaoa, reports_dir=REPORTS_DIR, model_dir=CHECKPOINT_DIR,
        seed=args.seed
    )
    rf_qubo = load_model(results['rf_qubo_model_path'])
    idx = results['selected_indices']
    metrics = evaluate_model(rf_qubo, data['X_test_agg'][:, idx], data['y_test_agg'],
                             model_name=f'RF (QUBO {len(idx)} features)')
    return results, metrics


def select_best_checkpoint(model_types) -> str:
    """Pick the trained sequence model with the best validation macro-F1."""
    from src.inference.predictor import load_checkpoint_dict

    candidates = {}
    for model_type in model_types:
        path = os.path.join(CHECKPOINT_DIR, f'{model_type}_best.pt')
        if os.path.exists(path):
            candidates[path] = load_checkpoint_dict(path).get('val_macro_f1', -1.0)
    if not candidates:
        raise FileNotFoundError(
            f"No trained sequence model in {CHECKPOINT_DIR}; refusing to build risk maps "
            "from an untrained model. Run the training phases first."
        )
    best = max(candidates, key=candidates.get)
    logger.info("Using %s for risk mapping (val macro-F1 %.4f)", best, candidates[best])
    return best


def phase5_geospatial_mapping(args, data: dict) -> None:
    """Risk maps for every sample, labelled with its real id and split."""
    if args.skip_geo:
        logger.info("Skipping Phase 5 (geo-spatial mapping)")
        return
    banner("PHASE 5: Geo-Spatial Risk Mapping")
    from src.geospatial.heatmap import generate_risk_heatmap, generate_zone_map
    from src.geospatial.risk_mapper import (
        aggregate_zone_risk,
        build_geodataframe,
        export_geojson,
        get_risk_summary,
    )
    from src.inference.predictor import Predictor

    predictor = Predictor(select_best_checkpoint(args.models))
    meta = data['sample_meta'].set_index('sample_id')

    frames = []
    for split in ('train', 'val', 'test'):
        probs = predictor.predict_proba_scaled(data[f'X_{split}'])
        ids = data[f'ids_{split}']
        frame = pd.DataFrame(probs, columns=[f'prob_class_{i}' for i in range(probs.shape[1])])
        frame.insert(0, 'sample_id', ids)
        frame.insert(1, 'split', split)
        frame['true_label'] = data[f'y_{split}']
        frame['predicted_label'] = probs.argmax(axis=1)
        frame['confidence'] = probs.max(axis=1)
        frame['lat'] = meta.loc[ids, 'lat'].to_numpy()
        frame['lon'] = meta.loc[ids, 'lon'].to_numpy()
        frames.append(frame)
    pred_df = pd.concat(frames, ignore_index=True)
    pred_df['model'] = predictor.model_type
    pred_df.to_csv(os.path.join(PROCESSED_DIR, 'predictions.csv'), index=False)

    gdf = build_geodataframe(pred_df)
    generate_risk_heatmap(gdf, f'{REPORTS_DIR}/disease_risk_map.html',
                          title=f'Crop Disease Risk Map ({predictor.model_type}, all splits)')
    zone_gdf = aggregate_zone_risk(gdf, grid_size_m=args.grid_size_m)
    if len(zone_gdf) > 0:
        export_geojson(zone_gdf, f'{REPORTS_DIR}/zone_risk_scores.geojson')
        generate_zone_map(zone_gdf, f'{REPORTS_DIR}/zone_risk_map.html')

    for key, val in get_risk_summary(gdf).items():
        logger.info("  %s: %s", key, val)


def phase6_dashboard_check(args) -> None:
    """Verify InfluxDB is reachable for the live dashboard."""
    if args.skip_mqtt:
        logger.info("Skipping Phase 6 (dashboard infrastructure check)")
        return
    banner("PHASE 6: Dashboard Infrastructure Check")
    try:
        from src.dashboard.influxdb_setup import setup_influxdb
        if not setup_influxdb():
            logger.warning("InfluxDB check failed. Start it with: docker compose up -d")
    except Exception:
        logger.exception("InfluxDB check could not run")

    logger.info("To start the live stack:")
    logger.info("  1. docker compose up -d")
    logger.info("  2. python -m src.dashboard.mqtt_subscriber --model-path %s/lstm_best.pt",
                CHECKPOINT_DIR)
    logger.info("  3. python -m src.data.mqtt_publisher --profile mixed")
    logger.info("  4. uvicorn src.api.main:app --port 8000")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Crop Disease Detection Pipeline')
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='+', default=list(DL_MODELS), choices=DL_MODELS)
    parser.add_argument('--max-features', type=int, default=8,
                        help='K for QUBO feature selection')
    parser.add_argument('--use-qaoa', action='store_true',
                        help='Also run QAOA on a pre-filtered QUBO sub-problem')
    parser.add_argument('--grid-size-m', type=float, default=5000.0,
                        help='Zone grid cell size in metres')
    parser.add_argument('--skip-quantum', action='store_true')
    parser.add_argument('--skip-geo', action='store_true')
    parser.add_argument('--skip-mqtt', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.evaluation.evaluate import compare_models

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start = time.time()
    logger.info("Crop Disease Detection Pipeline | samples=%d seq_len=%d epochs=%d seed=%d",
                args.samples, args.seq_len, args.epochs, args.seed)

    data = phase1_data_foundation(args)
    results = phase2_classical_models(args, data)
    results += phase3_deep_models(args, data)
    quantum_results, qubo_metrics = phase4_quantum_selection(args, data)
    if qubo_metrics:
        results.append(qubo_metrics)
    compare_models(results, data['y_test'],
                   output_csv=f'{REPORTS_DIR}/model_comparison.csv',
                   output_plot=f'{REPORTS_DIR}/model_comparison.png',
                   output_significance=f'{REPORTS_DIR}/model_significance.csv')
    phase5_geospatial_mapping(args, data)
    phase6_dashboard_check(args)

    banner(f"Pipeline complete in {time.time() - start:.1f}s")
    logger.info("Artifacts: %s, %s/*.npy, %s/, %s/", RAW_CSV, PROCESSED_DIR,
                CHECKPOINT_DIR, REPORTS_DIR)
    return 0


if __name__ == '__main__':
    sys.exit(main())
