"""
QUBO Feature Selection Formulation
====================================
Casts spectral feature selection as a Quadratic Unconstrained Binary
Optimization (QUBO) problem:

    Minimize: -Σ(relevance_i * x_i) + λ * Σ(corr_ij * x_i * x_j)
    Subject to: Σ(x_i) <= max_features

where x_i ∈ {0, 1} indicates whether feature i is selected.

Supports two backends:
    1. Qiskit QuadraticProgram (primary)
    2. Pure numpy QUBO matrix (fallback for Python 3.14 compatibility)
"""

import numpy as np
import pandas as pd

# Try importing Qiskit — fall back gracefully if unavailable
QISKIT_AVAILABLE = False
try:
    from qiskit_optimization import QuadraticProgram
    QISKIT_AVAILABLE = True
except ImportError:
    pass


def compute_relevance_scores(
    model,
    feature_names: list[str]
) -> np.ndarray:
    """
    Extract feature relevance scores from a trained tree-based model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with `feature_importances_` attribute.
    feature_names : list[str]
        Names of features corresponding to importance values.

    Returns
    -------
    np.ndarray
        Normalized relevance scores (sum to 1).
    """
    importances = model.feature_importances_
    # Normalize to [0, 1]
    if importances.max() > 0:
        importances = importances / importances.max()
    return importances


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Pearson correlation matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Absolute correlation matrix of shape (n_features, n_features).
    """
    corr = np.corrcoef(X, rowvar=False)
    # Handle NaN (constant features)
    corr = np.nan_to_num(corr, nan=0.0)
    return np.abs(corr)


def build_qubo_problem(
    correlation_matrix: np.ndarray,
    relevance_scores: np.ndarray,
    n_features: int,
    max_features: int = 6,
    lambda_penalty: float = 0.3,
    correlation_threshold: float = 0.7
):
    """
    Build QUBO problem for feature selection using Qiskit.

    The objective penalizes:
    1. NOT selecting relevant features (via negative relevance weights)
    2. Selecting highly correlated feature pairs (via quadratic penalty)

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Absolute pairwise correlation matrix.
    relevance_scores : np.ndarray
        Feature relevance/importance scores.
    n_features : int
        Total number of features.
    max_features : int
        Maximum number of features to select.
    lambda_penalty : float
        Weight of the correlation penalty term.
    correlation_threshold : float
        Only penalize pairs with correlation above this threshold.

    Returns
    -------
    QuadraticProgram or None
        Qiskit QuadraticProgram if available, else None.
    """
    if not QISKIT_AVAILABLE:
        print("WARNING: Qiskit not available. Use build_qubo_matrix() instead.")
        return None

    qp = QuadraticProgram('feature_selection')

    # Binary decision variables: x_i = 1 if feature i is selected
    for i in range(n_features):
        qp.binary_var(name=f'x{i}')

    # Linear terms: prefer features with high relevance
    linear = {f'x{i}': -float(relevance_scores[i]) for i in range(n_features)}

    # Quadratic terms: penalize selecting correlated pairs
    quadratic = {}
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i][j]) > correlation_threshold:
                quadratic[(f'x{i}', f'x{j}')] = (
                    lambda_penalty * float(correlation_matrix[i][j])
                )

    qp.minimize(linear=linear, quadratic=quadratic)

    # Constraint: select at most max_features
    constraint_linear = {f'x{i}': 1 for i in range(n_features)}
    qp.linear_constraint(
        linear=constraint_linear,
        sense='LE',
        rhs=max_features,
        name='max_features'
    )

    return qp


def build_qubo_matrix(
    correlation_matrix: np.ndarray,
    relevance_scores: np.ndarray,
    n_features: int,
    lambda_penalty: float = 0.3,
    correlation_threshold: float = 0.7
) -> dict[tuple[int, int], float]:
    """
    Build QUBO matrix as a dictionary (pure numpy fallback).

    This format is compatible with D-Wave's neal sampler.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Absolute pairwise correlation matrix.
    relevance_scores : np.ndarray
        Feature relevance scores.
    n_features : int
        Number of features.
    lambda_penalty : float
        Correlation penalty weight.
    correlation_threshold : float
        Threshold for penalizing correlations.

    Returns
    -------
    dict[tuple[int, int], float]
        QUBO matrix as {(i, j): weight} dictionary.
    """
    Q = {}

    # Linear terms (diagonal): negative relevance = prefer selecting
    for i in range(n_features):
        Q[(i, i)] = -float(relevance_scores[i])

    # Quadratic terms (off-diagonal): penalize correlated pairs
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i][j]) > correlation_threshold:
                Q[(i, j)] = lambda_penalty * float(correlation_matrix[i][j])

    return Q


def interpret_qubo_result(
    selected_indices: list[int],
    feature_names: list[str],
    relevance_scores: np.ndarray
) -> pd.DataFrame:
    """
    Create a summary of selected features with their relevance scores.

    Parameters
    ----------
    selected_indices : list[int]
        Indices of selected features.
    feature_names : list[str]
        All feature names.
    relevance_scores : np.ndarray
        All feature relevance scores.

    Returns
    -------
    pd.DataFrame
        Summary with feature name, index, relevance, and selection status.
    """
    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            'feature': name,
            'index': i,
            'relevance': float(relevance_scores[i]),
            'selected': i in selected_indices
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('relevance', ascending=False)
    return df
