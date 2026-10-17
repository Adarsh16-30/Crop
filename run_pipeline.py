"""
End-to-End Pipeline Runner
============================
Trains and evaluates every model on the real CDM + Sentinel-2 sequences
(built by `python -m src.real_data temporal`).

Phase 1  load sequences + preprocessing (site-grouped train/val/test split)
Phase 2  classical models (RF, XGBoost): grouped-CV tuning, site and
         spatial-zone CV, plus a seasonality-only baseline
Phase 3  sequence models (LSTM, CNN-LSTM, CNN-LSTM-Attention): grouped-CV
         tuning, refit on train+val for the CV-chosen number of epochs,
         temperature calibrated on out-of-fold predictions
Phase 4  quantum-inspired QUBO feature selection vs. classical selectors
         -> model comparison with 95% CIs and McNemar tests
Phase 5  geo-spatial risk maps from the best sequence model (by CV macro-F1)
Phase 6  publish results to InfluxDB for the Grafana dashboards

Overfitting control: hyperparameters are chosen by site-grouped CV on
train+val with the one-standard-error rule (src/training/tuning.py); the
test split is touched exactly once, at the end; train-vs-CV gaps are
written to reports/hyperparameter_tuning.csv.

Usage:
    python run_pipeline.py
    python run_pipeline.py --no-tune --hidden-dim 32 --dropout 0.5
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

RAW_CSV = 'data/real/cdm_s2/cdm_s2_sequences.csv'
SAMPLES_CSV = 'data/real/cdm_s2/cdm_s2_samples.csv'
GROUP_COL = 'site_id'
PROCESSED_DIR = 'data/processed'
CHECKPOINT_DIR = 'models/checkpoints'
REPORTS_DIR = 'reports'
DL_MODELS = ('lstm', 'cnn_lstm', 'cnn_lstm_attention')


def banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def sequence_length(raw_csv: str) -> int:
    """Timesteps per sample (all samples have the same length)."""
    lengths = pd.read_csv(raw_csv, usecols=['sample_id']).groupby('sample_id').size()
    if lengths.nunique() != 1:
        raise ValueError(f"{raw_csv}: samples have different lengths {sorted(lengths.unique())}")
    return int(lengths.iloc[0])


def phase1_data_foundation(args) -> dict:
    """Load the real sequences and preprocess them into one site-grouped split."""
    banner("PHASE 1: Data Foundation (CDM reports + Sentinel-2)")
    from src.data.preprocessor import run_preprocessing_pipeline

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Build it with: python -m src.real_data cdm && "
            "python -m src.real_data s2-cdm && python -m src.real_data temporal")
    logger.info("Dataset: %s (seq_len=%d, split grouped by %s)", RAW_CSV, args.seq_len,
                GROUP_COL)
    data = run_preprocessing_pipeline(input_csv=RAW_CSV, output_dir=PROCESSED_DIR,
                                      seq_len=args.seq_len, seed=args.seed, group_col=GROUP_COL,
                                      context=not args.no_context)
    data['sample_meta'] = pd.read_csv(os.path.join(PROCESSED_DIR, 'sample_meta.csv'))
    data['samples'] = pd.read_csv(SAMPLES_CSV).set_index('sample_id')
    return data


def train_val_groups(data: dict, suffix: str = '') -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                            np.ndarray]:
    """Train+val pooled (for tuning / CV): X, y, sample ids, site groups."""
    X = np.concatenate([data[f'X_train{suffix}'], data[f'X_val{suffix}']])
    y = np.concatenate([data[f'y_train{suffix}'], data[f'y_val{suffix}']])
    ids = np.concatenate([data[f'ids_train{suffix}'], data[f'ids_val{suffix}']])
    groups = data['sample_meta'].set_index('sample_id').loc[ids, GROUP_COL].to_numpy()
    return X, y, ids, groups


def phase2_classical_models(args, data: dict, tuning: list) -> list[dict]:
    """Tune, train and evaluate RF and XGBoost; site / spatial CV; seasonality baseline."""
    banner("PHASE 2: Classical Models")
    from src.constants import zone_key
    from src.models.classical_models import (
        cross_validate,
        evaluate_model,
        fit,
        make_random_forest,
        make_xgboost,
        save_model,
    )
    from src.training.tuning import RF_GRID, XGB_GRID, tune_classical

    X_cv, y_cv, ids_cv, sites = train_val_groups(data, '_agg')
    meta = data['sample_meta'].set_index('sample_id').loc[ids_cv]
    zones = np.array([hash(zone_key(lat, lon)) for lat, lon in zip(meta['lat'], meta['lon'])])

    results, cv_rows = [], []
    for name, factory, grid, path in (
        ('Random Forest', make_random_forest, RF_GRID, 'rf_model.joblib'),
        ('XGBoost', make_xgboost, XGB_GRID, 'xgb_model.joblib'),
    ):
        params = {}
        if not args.no_tune:
            params, table = tune_classical(name, factory, grid, X_cv, y_cv, sites,
                                           n_splits=args.cv_folds, seed=args.seed)
            tuning.append(table)
        # Classical models need no early-stopping split, so they fit on train+val.
        model = fit(factory(seed=args.seed, **params), X_cv, y_cv)
        results.append(evaluate_model(model, data['X_test_agg'], data['y_test_agg'],
                                      model_name=name))
        save_model(model, os.path.join(CHECKPOINT_DIR, path))
        for split_name, grp in (('site', sites), ('spatial_zone', zones)):
            cv = cross_validate(factory(seed=args.seed, **params), X_cv, y_cv, groups=grp,
                                seed=args.seed)
            cv_rows.append({'model': name, 'cv_split': split_name, **cv, 'params': str(params)})

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(REPORTS_DIR, 'classical_cv.csv'), index=False)
    logger.info("Classical CV (site vs spatial-zone hold-out):\n%s",
                cv_df.drop(columns='params').round(4).to_string(index=False))
    results.extend(ablation_baselines(args, data))
    return results


def ablation_baselines(args, data: dict) -> list[dict]:
    """
    How much comes from *when and where* vs. *what the spectra show*.

    The labels are defined by time relative to a report, so part of them is
    predictable from the calendar and location alone. A context-only RF
    (acquisition day of year + location, no spectra) and -- when the models
    use context -- a spectral-only RF bracket the full models. Both use the
    same regularized settings (depth 4, min leaf 15) and are refit on
    train+val; their site-grouped CV goes to reports/ablation_cv.csv.
    """
    from src.features.context import CONTEXT_FEATURES
    from src.models.classical_models import (
        cross_validate,
        evaluate_model,
        fit,
        make_random_forest,
    )
    from src.real_data.temporal import seasonality_features

    X_cv, y_cv, ids_cv, sites = train_val_groups(data, '_agg')
    X_test = data['X_test_agg']
    n_ctx = len(CONTEXT_FEATURES)
    if args.no_context:
        samples = data['samples']
        variants = {'Context-only RF (DOY, lat)': (
            seasonality_features(samples.loc[ids_cv]),
            seasonality_features(samples.loc[data['ids_test_agg']]))}
    else:
        variants = {'Context-only RF (date, location)': (X_cv[:, -n_ctx:], X_test[:, -n_ctx:]),
                    'Spectral-only RF': (X_cv[:, :-n_ctx], X_test[:, :-n_ctx])}
    results, rows = [], []
    for name, (X_fit, X_eval) in variants.items():
        make = lambda: make_random_forest(seed=args.seed, max_depth=4,  # noqa: E731
                                          min_samples_leaf=15)
        rows.append({'model': name, **cross_validate(make(), X_fit, y_cv, groups=sites,
                                                     seed=args.seed)})
        results.append(evaluate_model(fit(make(), X_fit, y_cv), X_eval, data['y_test_agg'],
                                      model_name=name))
    pd.DataFrame(rows).to_csv(os.path.join(REPORTS_DIR, 'ablation_cv.csv'), index=False)
    return results


def phase3_deep_models(args, data: dict, tuning: list) -> list[dict]:
    """Tune, train, calibrate and evaluate the sequence models."""
    banner("PHASE 3: Deep Learning Models")
    import torch

    from src.evaluation.evaluate import (
        evaluate_dl_model,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_training_curves,
    )
    from src.inference.predictor import Predictor
    from src.models.factory import DISPLAY_NAMES
    from src.training.train import fit_temperature, get_device, train_from_arrays
    from src.training.tuning import DL_GRID, tune_sequence_model

    device = get_device()
    X_cv, y_cv, _, sites = train_val_groups(data)
    results = []
    for model_type in args.models:
        params = {'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
                  'weight_decay': args.weight_decay, 'noise_std': args.noise_std}
        if args.no_tune:
            # Early stopping and calibration on the validation sites.
            train_data, refit = data, {'epochs': args.epochs, 'early_stopping': True,
                                       'temperature': None, 'extra_config': None}
        else:
            logger.info("Tuning %s (%d configurations x %d site folds)", model_type,
                        int(np.prod([len(v) for v in DL_GRID.values()])), args.cv_folds)
            params, table, info = tune_sequence_model(
                model_type, X_cv, y_cv, sites, data['scaler_params'], n_splits=args.cv_folds,
                epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                learning_rate=args.lr, seed=args.seed, device=device)
            tuning.append(table)
            # Refit on all of train+val for the CV-chosen epoch count; calibrate on
            # out-of-fold logits. The 8-site validation split alone is too small to
            # pick an epoch or a temperature reliably.
            # The OOF logits come from the fold models, which are early-stopped on
            # less data and so less confident than the refit model. Their
            # temperature may soften the refit model, never sharpen it (T >= 1):
            # sharpening an already sharper model makes it overconfident.
            temperature = max(1.0, fit_temperature(torch.from_numpy(info['oof_logits']),
                                                   torch.from_numpy(info['oof_labels'])))
            train_data = {**data, 'X_train': X_cv, 'y_train': y_cv}
            refit = {'epochs': info['epochs'], 'early_stopping': False,
                     'temperature': temperature,
                     'extra_config': {'cv_macro_f1': float(info['cv_f1']),
                                      'refit_on': 'train+val'}}
            logger.info("%s: refit on train+val for %d epochs, OOF temperature %.3f",
                        model_type, info['epochs'], temperature)
        history = train_from_arrays(
            model_type, train_data, data['scaler_params'], device, epochs=refit['epochs'],
            batch_size=args.batch_size, learning_rate=args.lr,
            weight_decay=params['weight_decay'], hidden_dim=params['hidden_dim'],
            dropout=params['dropout'], patience=args.patience, seed=args.seed,
            checkpoint_dir=CHECKPOINT_DIR, log_dir=REPORTS_DIR, noise_std=params['noise_std'],
            early_stopping=refit['early_stopping'], temperature=refit['temperature'],
            extra_config=refit['extra_config']
        )
        predictor = Predictor(history['checkpoint_path'], device=device)
        name = DISPLAY_NAMES[model_type]
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
    X_cv, y_cv, _, sites = train_val_groups(data, '_agg')
    tuned = rf.get_params()
    rf_params = {k: tuned[k] for k in ('n_estimators', 'max_depth', 'min_samples_leaf',
                                       'max_features')}
    # Same data as the tuned RF (train+val), site-grouped CV, tuned RF settings.
    results = run_quantum_feature_selection(
        rf_model=rf,
        X_train_agg=X_cv, y_train_agg=y_cv,
        X_test_agg=data['X_test_agg'], y_test_agg=data['y_test_agg'],
        feature_names=data['agg_feature_names'], max_features=args.max_features,
        use_qaoa=args.use_qaoa, reports_dir=REPORTS_DIR, model_dir=CHECKPOINT_DIR,
        seed=args.seed, groups_train=sites, rf_params=rf_params
    )
    rf_qubo = load_model(results['rf_qubo_model_path'])
    idx = results['selected_indices']
    metrics = evaluate_model(rf_qubo, data['X_test_agg'][:, idx], data['y_test_agg'],
                             model_name=f'RF (QUBO {len(idx)} features)')
    return results, metrics


def select_best_checkpoint(model_types) -> str:
    """
    Pick the trained sequence model with the best CV macro-F1 (tuned runs),
    else the best validation macro-F1.
    """
    from src.inference.predictor import load_checkpoint_dict

    candidates = {}
    for model_type in model_types:
        path = os.path.join(CHECKPOINT_DIR, f'{model_type}_best.pt')
        if os.path.exists(path):
            ckpt = load_checkpoint_dict(path)
            candidates[path] = ckpt['config'].get('cv_macro_f1',
                                                  ckpt.get('val_macro_f1', -1.0))
    if not candidates:
        raise FileNotFoundError(
            f"No trained sequence model in {CHECKPOINT_DIR}; refusing to build risk maps "
            "from an untrained model. Run the training phases first."
        )
    best = max(candidates, key=candidates.get)
    logger.info("Using %s for risk mapping (macro-F1 %.4f)", best, candidates[best])
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
                          title=f'CDM Risk Map ({predictor.model_type}, all splits)')
    zone_gdf = aggregate_zone_risk(gdf, grid_size_m=args.grid_size_m)
    if len(zone_gdf) > 0:
        export_geojson(zone_gdf, f'{REPORTS_DIR}/zone_risk_scores.geojson')
        generate_zone_map(zone_gdf, f'{REPORTS_DIR}/zone_risk_map.html')

    for key, val in get_risk_summary(gdf).items():
        logger.info("  %s: %s", key, val)


def phase6_dashboard_check(args) -> None:
    """Verify InfluxDB and publish this run's results for the Grafana dashboards."""
    if args.skip_mqtt:
        logger.info("Skipping Phase 6 (dashboard infrastructure check)")
        return
    banner("PHASE 6: Dashboard (InfluxDB check + publish results)")
    try:
        from src.config import load_influx_config
        from src.dashboard.influxdb_setup import setup_influxdb
        from src.dashboard.publish_pipeline_metrics import publish

        if not setup_influxdb():
            logger.warning("InfluxDB check failed. Start it with: docker compose up -d")
        else:
            publish(load_influx_config(token_env='INFLUX_WRITE_TOKEN'),
                    reports_dir=REPORTS_DIR, processed_dir=PROCESSED_DIR)
    except Exception:
        logger.exception("Publishing results to InfluxDB failed")

    logger.info("To start the live stack:")
    logger.info("  1. docker compose up -d")
    logger.info("  2. python -m src.dashboard.mqtt_subscriber --model-path %s/"
                "cnn_lstm_attention_best.pt", CHECKPOINT_DIR)
    logger.info("  3. python -m src.data.mqtt_publisher")
    logger.info("  4. uvicorn src.api.main:app --port 8000")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Crop Disease Detection Pipeline (real data)')
    parser.add_argument('--seq-len', type=int, default=None,
                        help='timesteps per sample; inferred from the dataset')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='+', default=list(DL_MODELS), choices=DL_MODELS)
    parser.add_argument('--no-tune', action='store_true',
                        help='skip grouped-CV tuning; use the hyperparameters below')
    parser.add_argument('--cv-folds', type=int, default=5, help='site-grouped CV folds')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--noise-std', type=float, default=0.1,
                        help='Gaussian input noise during training (standardized units)')
    parser.add_argument('--max-features', type=int, default=8,
                        help='K for QUBO feature selection')
    parser.add_argument('--use-qaoa', action='store_true',
                        help='Also run QAOA on a pre-filtered QUBO sub-problem')
    parser.add_argument('--grid-size-m', type=float, default=50000.0,
                        help='Zone grid cell size in metres')
    parser.add_argument('--no-context', action='store_true',
                        help='spectral features only: no acquisition day-of-year / location')
    parser.add_argument('--skip-quantum', action='store_true')
    parser.add_argument('--skip-geo', action='store_true')
    parser.add_argument('--skip-mqtt', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.evaluation.evaluate import compare_models
    from src.training.tuning import save_tuning

    if os.path.exists(RAW_CSV):
        inferred = sequence_length(RAW_CSV)
        if args.seq_len not in (None, inferred):
            raise SystemExit(f"--seq-len {args.seq_len} does not match {RAW_CSV} ({inferred}); "
                             "rebuild it with python -m src.real_data temporal --seq-len N")
        args.seq_len = inferred

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start = time.time()
    logger.info("Crop Disease Detection Pipeline | seq_len=%s epochs=%d tune=%s seed=%d",
                args.seq_len, args.epochs, not args.no_tune, args.seed)

    data = phase1_data_foundation(args)
    tuning: list = []
    results = phase2_classical_models(args, data, tuning)
    results += phase3_deep_models(args, data, tuning)
    if tuning:
        logger.info("Tuning results: %s", save_tuning(tuning, REPORTS_DIR))
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
    logger.info("Artifacts: %s/*.npy, %s/, %s/", PROCESSED_DIR, CHECKPOINT_DIR, REPORTS_DIR)
    return 0


if __name__ == '__main__':
    sys.exit(main())
