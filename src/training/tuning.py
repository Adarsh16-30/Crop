"""
Hyperparameter Tuning without Overfitting
=========================================
The real dataset is small (a few hundred overlapping sequences from ~50
sites), so a tuned model can easily fit noise, and a selection made on one
small validation split mostly reflects luck. Every candidate configuration
is therefore scored by site-grouped K-fold cross-validation on train+val
(no site ever in both the fitting and the scoring fold), and:

- the selection uses the one-standard-error rule: among configurations
  whose mean CV macro-F1 is within one standard error of the best, pick
  the simplest (fewest parameters / shallowest trees / strongest
  regularization). That trades a statistically indistinguishable bit of CV
  score for a model less likely to have fit the folds by chance;
- every result row also has the training macro-F1, so the train-vs-CV gap
  (how much a configuration overfits) is reported next to its score;
- deep models stop early on an inner, site-grouped hold-out carved out of
  each training fold, never on the fold they are scored on;
- for the chosen deep configuration the search also returns its
  out-of-fold logits and the median best epoch, so the final model can be
  refit on all of train+val for a fixed number of epochs and calibrated on
  predictions it never trained on (see run_pipeline.py, Phase 3), instead
  of early-stopping and calibrating on one small validation split.

The test split is not used here at all.
"""

import itertools
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.constants import NUM_CLASSES

logger = logging.getLogger(__name__)

# Classical grids: (factory kwargs, complexity) -- lower complexity = simpler.
# The grids lean towards strong regularization: with a few hundred
# overlapping sequences, unconstrained models reach near-perfect training
# scores at no gain in CV score.
RF_GRID = {'max_depth': [2, 3, 5, 8], 'min_samples_leaf': [5, 15, 30],
           'max_features': ['sqrt', 0.5, 1.0]}
XGB_GRID = {'max_depth': [1, 2, 3], 'learning_rate': [0.03, 0.1],
            'n_estimators': [50, 150], 'min_child_weight': [5, 20],
            'reg_lambda': [1.0, 10.0]}
# Deep-model grid (shared by LSTM / CNN-LSTM / CNN-LSTM-Attention).
DL_GRID = {'hidden_dim': [8, 16, 32], 'dropout': [0.3, 0.5],
           'weight_decay': [1e-2, 1e-1], 'noise_std': [0.1, 0.3]}


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average='macro', labels=list(range(NUM_CLASSES)),
                    zero_division=0)


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                  seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def one_standard_error_choice(results: pd.DataFrame, complexity_col: str = 'complexity',
                              score_col: str = 'cv_f1_mean',
                              se_col: str = 'cv_f1_se') -> pd.Series:
    """Simplest configuration whose CV score is within 1 SE of the best."""
    best = results.loc[results[score_col].idxmax()]
    eligible = results[results[score_col] >= best[score_col] - best[se_col]]
    return eligible.sort_values([complexity_col, score_col], ascending=[True, False]).iloc[0]


def _summarise(rows: list[dict], params: dict, complexity: float) -> dict:
    cv = np.array([r['cv_f1'] for r in rows])
    train = np.array([r['train_f1'] for r in rows])
    return {**params, 'complexity': complexity,
            'cv_f1_mean': float(cv.mean()), 'cv_f1_std': float(cv.std()),
            'cv_f1_se': float(cv.std(ddof=1) / np.sqrt(len(cv))) if len(cv) > 1 else 0.0,
            'train_f1_mean': float(train.mean()),
            'overfit_gap': float(train.mean() - cv.mean()), 'folds': len(cv)}


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

def _classical_complexity(name: str, p: dict) -> float:
    if name == 'Random Forest':
        depth = p['max_depth'] or 30
        spread = {'sqrt': 1.0, 0.5: 2.0, 1.0: 3.0}.get(p['max_features'], 2.0)
        return depth / p['min_samples_leaf'] * spread
    return p['max_depth'] * p['n_estimators'] * p['learning_rate'] / \
        (p['min_child_weight'] * np.sqrt(p['reg_lambda']))


def tune_classical(name: str, factory, grid: dict, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, n_splits: int = 5,
                   seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Grid search for an sklearn-style model. Returns (chosen params, all results)."""
    from src.models.classical_models import fit

    folds = grouped_folds(y, groups, n_splits, seed)
    rows = []
    for params in expand_grid(grid):
        base = factory(seed=seed, **params)
        fold_rows = []
        for tr, te in folds:
            model = fit(clone(base), X[tr], y[tr])
            fold_rows.append({'cv_f1': macro_f1(y[te], model.predict(X[te])),
                              'train_f1': macro_f1(y[tr], model.predict(X[tr]))})
        rows.append(_summarise(fold_rows, params, _classical_complexity(name, params)))
    results = pd.DataFrame(rows)
    chosen = one_standard_error_choice(results)
    params = expand_grid(grid)[chosen.name]     # original types (a row turns ints to floats)
    logger.info("%s tuned: %s | CV macro-F1 %.3f +/- %.3f (best %.3f) | train-CV gap %.3f",
                name, params, chosen['cv_f1_mean'], chosen['cv_f1_se'],
                results['cv_f1_mean'].max(), chosen['overfit_gap'])
    return params, results.assign(model=name, chosen=results.index == chosen.name)


# ---------------------------------------------------------------------------
# Sequence models
# ---------------------------------------------------------------------------

def _inner_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """~20% of a training fold's sites held out for early stopping."""
    folds = grouped_folds(y, groups, n_splits=5, seed=seed)
    return folds[0]


def train_sequence_fold(model_type: str, params: dict, X_tr, y_tr, X_es, y_es,
                        scaler_params: dict, epochs: int, patience: int, batch_size: int,
                        learning_rate: float, seed: int,
                        device) -> tuple[torch.nn.Module, int]:
    """
    Train one model with early stopping on (X_es, y_es); checkpoints go to a
    temp dir. Returns (model with raw logits, best epoch).
    """
    from src.training.train import train_from_arrays

    data = {'X_train': X_tr, 'y_train': y_tr, 'X_val': X_es, 'y_val': y_es}
    with tempfile.TemporaryDirectory() as tmp:
        history = train_from_arrays(
            model_type, data, scaler_params, device, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=params['weight_decay'],
            hidden_dim=params['hidden_dim'], dropout=params['dropout'], patience=patience,
            seed=seed, checkpoint_dir=tmp, log_dir=tmp, noise_std=params['noise_std'])
        from src.training.train import load_checkpoint
        return load_checkpoint(history['checkpoint_path'], device), int(history['best_epoch'])


@torch.no_grad()
def _logits(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
                 ).cpu().numpy()


def _predict(model, X: np.ndarray, device) -> np.ndarray:
    return _logits(model, X, device).argmax(axis=1)


def tune_sequence_model(model_type: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        scaler_params: dict, grid: dict | None = None, n_splits: int = 5,
                        epochs: int = 60, patience: int = 10, batch_size: int = 32,
                        learning_rate: float = 1e-3, seed: int = 42,
                        device=None) -> tuple[dict, pd.DataFrame, dict]:
    """
    Grouped-CV grid search for one sequence model type.

    Returns (chosen params, results table, refit info for the chosen
    configuration: `epochs` = median best epoch over folds, `oof_logits` /
    `oof_labels` = out-of-fold logits for calibration, `cv_f1`).
    """
    from src.models.factory import build_model, count_parameters

    device = device or torch.device('cpu')
    grid = grid or DL_GRID
    folds = grouped_folds(y, groups, n_splits, seed)
    rows, extras = [], []
    quiet = logging.getLogger('src.training.train')
    level = quiet.level
    quiet.setLevel(logging.WARNING)
    try:
        for params in expand_grid(grid):
            n_params = count_parameters(build_model(model_type, input_dim=X.shape[2],
                                                    seq_len=X.shape[1],
                                                    hidden_dim=params['hidden_dim'],
                                                    dropout=params['dropout']))
            # simpler = fewer parameters, then more regularization
            complexity = n_params / (1 + 10 * params['weight_decay']) / \
                (1 + params['dropout']) / (1 + params['noise_std'])
            fold_rows, best_epochs = [], []
            oof = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
            for k, (tr, te) in enumerate(folds):
                fit_idx, es_idx = _inner_split(y[tr], groups[tr], seed + k)
                model, best_epoch = train_sequence_fold(
                    model_type, params, X[tr][fit_idx], y[tr][fit_idx], X[tr][es_idx],
                    y[tr][es_idx], scaler_params, epochs, patience, batch_size,
                    learning_rate, seed, device)
                oof[te] = _logits(model, X[te], device)
                best_epochs.append(best_epoch)
                fold_rows.append({'cv_f1': macro_f1(y[te], oof[te].argmax(axis=1)),
                                  'train_f1': macro_f1(y[tr][fit_idx],
                                                       _predict(model, X[tr][fit_idx], device))})
            rows.append({**_summarise(fold_rows, params, complexity), 'n_params': n_params,
                         'median_best_epoch': int(np.median(best_epochs))})
            extras.append({'epochs': max(1, int(np.median(best_epochs))), 'oof_logits': oof,
                           'oof_labels': y.copy(), 'cv_f1': rows[-1]['cv_f1_mean']})
            logger.info("  %s %s: CV F1 %.3f +/- %.3f | train %.3f", model_type, params,
                        rows[-1]['cv_f1_mean'], rows[-1]['cv_f1_se'], rows[-1]['train_f1_mean'])
    finally:
        quiet.setLevel(level)
    results = pd.DataFrame(rows)
    chosen = one_standard_error_choice(results)
    params = expand_grid(grid)[chosen.name]
    logger.info("%s tuned: %s (%d params) | CV macro-F1 %.3f +/- %.3f (best %.3f) | "
                "train-CV gap %.3f", model_type, params, int(chosen['n_params']),
                chosen['cv_f1_mean'], chosen['cv_f1_se'], results['cv_f1_mean'].max(),
                chosen['overfit_gap'])
    return (params, results.assign(model=model_type, chosen=results.index == chosen.name),
            extras[chosen.name])


def save_tuning(frames: list[pd.DataFrame], reports_dir: str) -> str:
    path = os.path.join(reports_dir, 'hyperparameter_tuning.csv')
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path
