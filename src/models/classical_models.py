"""
Classical ML Models for Crop Disease Classification
====================================================
Random Forest and XGBoost baselines on aggregated per-sample features
(mean, std, slope and delta of every band / index), plus cross-validation
under both a random and a spatially grouped (zone-held-out) split.
"""

import logging
import os

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from src.evaluation.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def make_random_forest(n_estimators: int = 200, max_depth: int = 10,
                       seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced',
        random_state=seed, n_jobs=-1
    )


def make_xgboost(n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.05,
                 seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
        random_state=seed, verbosity=0
    )


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency per-sample weights (XGBoost has no class_weight)."""
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y])


def fit(model, X: np.ndarray, y: np.ndarray):
    """Fit with class balancing appropriate to the model type."""
    if isinstance(model, XGBClassifier):
        model.fit(X, y, sample_weight=balanced_sample_weights(y))
    else:
        model.fit(X, y)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10, seed=42):
    """Train a Random Forest with balanced class weights."""
    return fit(make_random_forest(n_estimators, max_depth, seed), X_train, y_train)


def train_xgboost(X_train, y_train, n_estimators=300, max_depth=6,
                  learning_rate=0.05, seed=42):
    """Train an XGBoost classifier with balanced sample weights."""
    return fit(make_xgboost(n_estimators, max_depth, learning_rate, seed), X_train, y_train)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = 'Model') -> dict:
    """Evaluate a fitted sklearn-compatible classifier (shared metrics + CIs)."""
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    return compute_metrics(y_test, model.predict(X_test), y_proba, model_name)


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42
) -> dict:
    """
    Stratified k-fold CV. If `groups` is given, folds hold out whole groups
    (e.g. spatial zones), so no zone appears in both train and test.

    Returns mean/std of accuracy and macro-F1 across folds.
    """
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        n_groups = len(np.unique(groups))
        splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True,
                                        random_state=seed)
        split_iter = splitter.split(X, y, groups)

    accs, f1s = [], []
    for train_idx, test_idx in split_iter:
        fold_model = fit(clone(model), X[train_idx], y[train_idx])
        pred = fold_model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average='macro', zero_division=0))

    return {
        'cv_folds': len(accs),
        'cv_accuracy_mean': float(np.mean(accs)), 'cv_accuracy_std': float(np.std(accs)),
        'cv_f1_macro_mean': float(np.mean(f1s)), 'cv_f1_macro_std': float(np.std(f1s)),
    }


def save_model(model, filepath: str) -> None:
    """
    Save a trained model with joblib.

    joblib files are pickles: only ever load ones this pipeline produced.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model saved to: %s", filepath)


def load_model(filepath: str):
    """Load a joblib model. Trusted, locally produced files only (pickle)."""
    return joblib.load(filepath)


def get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Feature name -> importance, sorted descending."""
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
