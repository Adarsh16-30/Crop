"""
Quantum-Inspired Feature Selection Integration
================================================
1. Relevance = feature importances of a Random Forest trained on TRAIN only
2. Redundancy = |correlation| between aggregated features on TRAIN only
3. Build a QUBO with a built-in "select exactly K" penalty
4. Solve with simulated annealing (optionally QAOA on a pre-filtered
   sub-problem small enough for a statevector simulator)
5. Compare against non-quantum baselines that pick the same K:
   all features, top-K by importance, greedy mRMR, random K
   using 5-fold CV on the training set AND the held-out test set
6. Save the RF trained on the QUBO subset so the selection is actually used
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score

from src.models.classical_models import (
    cross_validate,
    make_random_forest,
    save_model,
    train_random_forest,
)
from src.quantum.annealing_optimizer import get_solver_info, run_qaoa, run_simulated_annealing
from src.quantum.qubo_feature_selector import (
    QISKIT_AVAILABLE,
    build_qubo_matrix,
    build_qubo_problem,
    compute_correlation_matrix,
    compute_relevance_scores,
    interpret_qubo_result,
)

logger = logging.getLogger(__name__)

QAOA_MAX_CANDIDATES = 12


def select_top_k(relevance: np.ndarray, k: int) -> list[int]:
    return sorted(np.argsort(relevance)[::-1][:k].tolist())


def select_mrmr(X: np.ndarray, y: np.ndarray, k: int, seed: int = 42) -> list[int]:
    """Greedy mRMR: maximize MI(feature, label) - mean |corr| with already-selected."""
    mi = mutual_info_classif(X, y, random_state=seed)
    corr = compute_correlation_matrix(X)
    selected = [int(np.argmax(mi))]
    while len(selected) < k:
        remaining = [i for i in range(X.shape[1]) if i not in selected]
        scores = [mi[i] - corr[i, selected].mean() for i in remaining]
        selected.append(remaining[int(np.argmax(scores))])
    return sorted(selected)


def _score_subset(name, idx, X_train, y_train, X_test, y_test, seed) -> dict:
    cv = cross_validate(make_random_forest(seed=seed), X_train[:, idx], y_train, seed=seed)
    model = train_random_forest(X_train[:, idx], y_train, seed=seed)
    pred = model.predict(X_test[:, idx])
    return {
        'method': name, 'n_features': len(idx),
        **cv,
        'test_accuracy': accuracy_score(y_test, pred),
        'test_f1_macro': f1_score(y_test, pred, average='macro', zero_division=0),
    }


def run_quantum_feature_selection(
    rf_model,
    X_train_agg: np.ndarray,
    y_train_agg: np.ndarray,
    X_test_agg: np.ndarray,
    y_test_agg: np.ndarray,
    feature_names: list[str] | None = None,
    max_features: int = 8,
    lambda_penalty: float = 0.3,
    correlation_threshold: float = 0.7,
    use_qaoa: bool = False,
    reports_dir: str = 'reports',
    model_dir: str = 'models/checkpoints',
    seed: int = 42,
    n_random_draws: int = 10
) -> dict:
    """
    Run QUBO feature selection and benchmark it against classical selectors.

    Returns a dict with the selected indices / names, the comparison table
    and the path of the saved QUBO-subset Random Forest.
    """
    n_features = X_train_agg.shape[1]
    feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
    logger.info("QUBO feature selection: %d features -> K=%d | lambda=%.2f tau=%.2f | %s",
                n_features, max_features, lambda_penalty, correlation_threshold,
                get_solver_info())

    relevance = compute_relevance_scores(rf_model)
    corr = compute_correlation_matrix(X_train_agg)

    qubo = build_qubo_matrix(corr, relevance, n_features, lambda_penalty=lambda_penalty,
                             correlation_threshold=correlation_threshold,
                             max_features=max_features)
    selected_sa = run_simulated_annealing(qubo, n_reads=1000, seed=seed,
                                          max_features=max_features)
    if not selected_sa:
        logger.warning("Annealer selected nothing; falling back to top-K by importance")
        selected_sa = select_top_k(relevance, max_features)

    # Optional QAOA on the most relevant candidates only (statevector cost is 2^n).
    selected_qaoa = None
    if use_qaoa:
        if not QISKIT_AVAILABLE:
            logger.warning("--use-qaoa requested but Qiskit is not installed; skipping")
        else:
            candidates = select_top_k(relevance, min(QAOA_MAX_CANDIDATES, n_features))
            k = min(max_features, len(candidates) - 1)
            qp = build_qubo_problem(corr[np.ix_(candidates, candidates)], relevance[candidates],
                                    len(candidates), max_features=k,
                                    lambda_penalty=lambda_penalty,
                                    correlation_threshold=correlation_threshold)
            try:
                local = run_qaoa(qp, reps=2, max_vars=QAOA_MAX_CANDIDATES, seed=seed)
                selected_qaoa = sorted(candidates[i] for i in local)
            except Exception:
                logger.exception("QAOA failed; continuing with simulated annealing result")

    # Benchmark against non-quantum selectors at the same K.
    subsets = {
        'all_features': list(range(n_features)),
        'qubo_simulated_annealing': selected_sa,
        'top_k_importance': select_top_k(relevance, max_features),
        'mrmr': select_mrmr(X_train_agg, y_train_agg, max_features, seed=seed),
    }
    if selected_qaoa:
        subsets['qubo_qaoa'] = selected_qaoa

    rows = [_score_subset(name, idx, X_train_agg, y_train_agg, X_test_agg, y_test_agg, seed)
            for name, idx in subsets.items()]

    rng = np.random.default_rng(seed)
    random_rows = [
        _score_subset('random_k', sorted(rng.choice(n_features, max_features, replace=False)),
                      X_train_agg, y_train_agg, X_test_agg, y_test_agg, seed)
        for _ in range(n_random_draws)
    ]
    random_summary = pd.DataFrame(random_rows).drop(columns='method').mean().to_dict()
    rows.append({'method': f'random_k (mean of {n_random_draws})', **random_summary})

    comparison = pd.DataFrame(rows)
    logger.info("Feature-selection comparison:\n%s", comparison.round(4).to_string(index=False))

    # Close the loop: keep a model that actually uses the QUBO subset.
    rf_qubo = train_random_forest(X_train_agg[:, selected_sa], y_train_agg, seed=seed)
    rf_qubo_path = os.path.join(model_dir, 'rf_qubo_model.joblib')
    save_model(rf_qubo, rf_qubo_path)

    os.makedirs(reports_dir, exist_ok=True)
    interpret_qubo_result(selected_sa, feature_names, relevance).to_csv(
        os.path.join(reports_dir, 'qubo_feature_selection_results.csv'), index=False)
    comparison.to_csv(os.path.join(reports_dir, 'feature_selection_comparison.csv'), index=False)

    results = {
        'solver_used': 'simulated_annealing (classical, quantum-inspired)',
        'selected_indices': [int(i) for i in selected_sa],
        'selected_names': [feature_names[i] for i in selected_sa],
        'n_features_total': n_features,
        'n_features_selected': len(selected_sa),
        'rf_qubo_model_path': rf_qubo_path,
        'comparison': comparison.to_dict(orient='records'),
    }
    if selected_qaoa:
        results['qaoa_selected_indices'] = [int(i) for i in selected_qaoa]
        results['qaoa_selected_names'] = [feature_names[i] for i in selected_qaoa]

    with open(os.path.join(reports_dir, 'qubo_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("QUBO selected: %s", results['selected_names'])
    return results
