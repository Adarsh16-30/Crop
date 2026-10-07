"""
Quantum Feature Selection Integration
=======================================
Orchestrates the full QUBO-based feature selection workflow:

1. Load trained Random Forest → extract feature importances
2. Build correlation matrix from training data
3. Construct QUBO problem
4. Solve via simulated annealing (or QAOA)
5. Re-train models on selected feature subset
6. Compare accuracy: full features vs QUBO-selected
7. Save results
"""

import os
import json

import numpy as np
import pandas as pd

from src.quantum.qubo_feature_selector import (
    compute_relevance_scores,
    compute_correlation_matrix,
    build_qubo_matrix,
    build_qubo_problem,
    interpret_qubo_result,
    QISKIT_AVAILABLE
)
from src.quantum.annealing_optimizer import (
    run_simulated_annealing,
    run_qaoa,
    get_solver_info
)
from src.features.spectral_indices import get_feature_columns


def run_quantum_feature_selection(
    rf_model,
    X_train_agg: np.ndarray,
    y_train_agg: np.ndarray,
    X_test_agg: np.ndarray,
    y_test_agg: np.ndarray,
    feature_names: list[str] | None = None,
    max_features: int = 6,
    lambda_penalty: float = 0.3,
    correlation_threshold: float = 0.7,
    use_qaoa: bool = False,
    reports_dir: str = 'reports'
) -> dict:
    """
    Full quantum feature selection pipeline.

    Parameters
    ----------
    rf_model : RandomForestClassifier
        Trained RF model for feature importance extraction.
    X_train_agg : np.ndarray
        Aggregated training features.
    y_train_agg : np.ndarray
        Training labels.
    X_test_agg : np.ndarray
        Aggregated test features.
    y_test_agg : np.ndarray
        Test labels.
    feature_names : list[str] or None
        Feature names. If None, generates generic names.
    max_features : int
        Maximum features to select.
    lambda_penalty : float
        QUBO correlation penalty weight.
    correlation_threshold : float
        Minimum correlation to trigger penalty.
    use_qaoa : bool
        Whether to also run QAOA (requires Qiskit, ≤15 features).
    reports_dir : str
        Directory for saving reports.

    Returns
    -------
    dict
        Results including selected features, accuracy comparison,
        and the QUBO summary.
    """
    n_features = X_train_agg.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    print(f"\n{'='*60}")
    print(f"  Quantum-Inspired Feature Selection")
    print(f"{'='*60}")
    print(f"  Total features: {n_features}")
    print(f"  Max features to select: {max_features}")
    print(f"  Lambda penalty: {lambda_penalty}")
    print(f"  Correlation threshold: {correlation_threshold}")

    solver_info = get_solver_info()
    print(f"  Recommended solver: {solver_info['recommended']}")

    # Step 1: Compute relevance scores from RF
    print(f"\nStep 1: Computing feature relevance scores...")
    relevance = compute_relevance_scores(rf_model, feature_names)
    print(f"  Top 5 features by relevance:")
    sorted_idx = np.argsort(relevance)[::-1]
    for i in sorted_idx[:5]:
        print(f"    {feature_names[i]}: {relevance[i]:.4f}")

    # Step 2: Build correlation matrix
    print(f"\nStep 2: Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(X_train_agg)
    n_high_corr = np.sum(corr_matrix > correlation_threshold) - n_features  # exclude diagonal
    print(f"  Highly correlated pairs (>{correlation_threshold}): {n_high_corr // 2}")

    # Step 3: Build and solve QUBO
    print(f"\nStep 3: Building QUBO problem...")
    qubo_dict = build_qubo_matrix(
        corr_matrix, relevance, n_features,
        lambda_penalty=lambda_penalty,
        correlation_threshold=correlation_threshold
    )

    # Solve with simulated annealing
    print(f"\nStep 4: Solving QUBO via simulated annealing...")
    selected_indices_sa = run_simulated_annealing(
        qubo_dict,
        n_reads=1000,
        max_features=max_features
    )

    if not selected_indices_sa:
        selected_indices_sa = list(np.argsort(relevance)[::-1][:max_features])

    # Optionally solve with QAOA
    selected_indices_qaoa = None
    if use_qaoa and QISKIT_AVAILABLE and n_features <= 15:
        print(f"\nStep 4b: Solving QUBO via QAOA...")
        qp = build_qubo_problem(
            corr_matrix, relevance, n_features,
            max_features=max_features,
            lambda_penalty=lambda_penalty,
            correlation_threshold=correlation_threshold
        )
        if qp is not None:
            try:
                selected_indices_qaoa = run_qaoa(qp, reps=2)
            except Exception as e:
                print(f"  QAOA failed: {e}")

    # Step 5: Evaluate — compare full vs selected features
    print(f"\nStep 5: Comparing baseline vs QUBO-selected features...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    # Baseline: all features
    rf_baseline = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_baseline.fit(X_train_agg, y_train_agg)
    y_pred_baseline = rf_baseline.predict(X_test_agg)
    acc_baseline = accuracy_score(y_test_agg, y_pred_baseline)
    f1_baseline = f1_score(y_test_agg, y_pred_baseline, average='macro')

    # QUBO-selected features
    X_train_selected = X_train_agg[:, selected_indices_sa]
    X_test_selected = X_test_agg[:, selected_indices_sa]

    rf_qubo = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_qubo.fit(X_train_selected, y_train_agg)
    y_pred_qubo = rf_qubo.predict(X_test_selected)
    acc_qubo = accuracy_score(y_test_agg, y_pred_qubo)
    f1_qubo = f1_score(y_test_agg, y_pred_qubo, average='macro')

    # Print comparison
    acc_delta = acc_qubo - acc_baseline
    f1_delta = f1_qubo - f1_baseline
    print(f"\n  {'Metric':<20} {'Baseline (all)':<18} {'QUBO-selected':<18} {'Delta':<10}")
    print(f"  {'-'*66}")
    print(f"  {'Accuracy':<20} {acc_baseline:<18.4f} {acc_qubo:<18.4f} {acc_delta:+.4f}")
    print(f"  {'F1 (macro)':<20} {f1_baseline:<18.4f} {f1_qubo:<18.4f} {f1_delta:+.4f}")
    print(f"  {'Num features':<20} {n_features:<18d} {len(selected_indices_sa):<18d}")

    # Build results
    feature_summary = interpret_qubo_result(selected_indices_sa, feature_names, relevance)
    selected_names = [feature_names[i] for i in selected_indices_sa]

    results = {
        'selected_indices': selected_indices_sa,
        'selected_names': selected_names,
        'n_features_total': n_features,
        'n_features_selected': len(selected_indices_sa),
        'accuracy_baseline': acc_baseline,
        'accuracy_qubo': acc_qubo,
        'accuracy_delta': acc_delta,
        'f1_baseline': f1_baseline,
        'f1_qubo': f1_qubo,
        'f1_delta': f1_delta,
        'feature_summary': feature_summary,
        'solver_used': 'simulated_annealing',
    }

    if selected_indices_qaoa is not None:
        results['qaoa_selected_indices'] = selected_indices_qaoa
        results['qaoa_selected_names'] = [feature_names[i] for i in selected_indices_qaoa]

    # Save results
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, 'qubo_feature_selection_results.csv')
    feature_summary.to_csv(csv_path, index=False)
    print(f"\nFeature selection results saved: {csv_path}")

    comparison_path = os.path.join(reports_dir, 'qubo_comparison.json')
    comparison_data = {
        k: v for k, v in results.items()
        if k != 'feature_summary'
    }
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Comparison data saved: {comparison_path}")

    return results
