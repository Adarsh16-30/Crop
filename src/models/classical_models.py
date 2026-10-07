"""
Classical ML Models for Crop Disease Classification
====================================================
Random Forest and XGBoost baselines operating on aggregated
(mean + std) spectral features.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
from xgboost import XGBClassifier

from src.data.data_generator import LABEL_NAMES


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 10,
    seed: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with balanced class weights.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (aggregated features).
    y_train : np.ndarray
        Training labels.
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum tree depth.
    seed : int
        Random seed.

    Returns
    -------
    RandomForestClassifier
        Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    seed: int = 42
) -> XGBClassifier:
    """
    Train an XGBoost classifier for multi-class disease prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    seed : int
        Random seed.

    Returns
    -------
    XGBClassifier
        Trained model.
    """
    # Compute sample weights for class balancing
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weight_map[y] for y in y_train])

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=seed,
        verbosity=1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'Model'
) -> dict:
    """
    Evaluate a trained model and return detailed metrics.

    Parameters
    ----------
    model : classifier
        Trained sklearn-compatible model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.
    model_name : str
        Name for display.

    Returns
    -------
    dict
        Dictionary containing accuracy, per-class metrics, and confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Probabilities (if available)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)

    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred,
        'probabilities': y_proba,
    }

    # Per-class metrics
    for label, name in LABEL_NAMES.items():
        if name in report:
            results[f'f1_{name.lower().replace(" ", "_")}'] = report[name]['f1-score']
            results[f'precision_{name.lower().replace(" ", "_")}'] = report[name]['precision']
            results[f'recall_{name.lower().replace(" ", "_")}'] = report[name]['recall']

    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:       {results['accuracy']:.4f}")
    print(f"  F1 (macro):     {results['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {results['f1_weighted']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    return results


def save_model(model, filepath: str) -> None:
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk."""
    return joblib.load(filepath)


def get_feature_importances(
    model,
    feature_names: list[str]
) -> dict[str, float]:
    """
    Extract feature importances from a tree-based model.

    Returns
    -------
    dict[str, float]
        Feature name → importance score, sorted descending.
    """
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
