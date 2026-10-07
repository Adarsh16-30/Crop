"""
QUBO Feature Selection Formulation
====================================
Casts feature selection as a Quadratic Unconstrained Binary Optimization
problem over x_i in {0, 1} (x_i = 1 means feature i is selected):

    E(x) = - sum_i r_i x_i                              (relevance)
           + lambda * sum_{i<j, |c_ij| > tau} |c_ij| x_i x_j   (redundancy)
           + P * (sum_i x_i - K)^2                       (select exactly K)

The cardinality term is expanded into the QUBO (x_i^2 = x_i):
    diagonal  += P * (1 - 2K)
    off-diag  += 2P          for every pair i < j
    constant  P * K^2        (dropped; does not change the argmin)

With P larger than the biggest single-feature gain, any solution with
sum(x) != K costs more than the best size-K solution, so the annealer
solves the constrained problem directly instead of having its answer
truncated afterwards.

This is a classical, "quantum-inspired" formulation: it is solved by
simulated annealing on a CPU (and optionally QAOA on a statevector
simulator for small sub-problems), not on quantum hardware.
"""

import numpy as np
import pandas as pd

QISKIT_AVAILABLE = False
try:
    from qiskit_optimization import QuadraticProgram
    QISKIT_AVAILABLE = True
except ImportError:
    pass


def compute_relevance_scores(model, feature_names: list[str] | None = None) -> np.ndarray:
    """Tree-model feature importances scaled so the most important feature is 1."""
    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.max() > 0:
        importances = importances / importances.max()
    return importances


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Absolute pairwise Pearson correlation (constant features -> 0)."""
    corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
    return np.abs(corr)


def default_cardinality_penalty(relevance_scores: np.ndarray) -> float:
    """A penalty strictly larger than any single feature's relevance gain."""
    return 1.5 * float(np.max(relevance_scores)) + 1e-3


def build_qubo_matrix(
    correlation_matrix: np.ndarray,
    relevance_scores: np.ndarray,
    n_features: int,
    lambda_penalty: float = 0.3,
    correlation_threshold: float = 0.7,
    max_features: int | None = None,
    cardinality_penalty: float | None = None
) -> dict[tuple[int, int], float]:
    """
    Build the QUBO as an upper-triangular {(i, j): weight} dict (D-Wave format).

    If `max_features` (K) is given, the (sum x - K)^2 cardinality penalty is
    folded in. Without it, only relevance and redundancy terms are used.
    """
    Q: dict[tuple[int, int], float] = {}
    for i in range(n_features):
        Q[(i, i)] = -float(relevance_scores[i])
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i][j]) > correlation_threshold:
                Q[(i, j)] = lambda_penalty * float(correlation_matrix[i][j])

    if max_features is not None:
        P = cardinality_penalty if cardinality_penalty is not None \
            else default_cardinality_penalty(relevance_scores)
        K = int(max_features)
        for i in range(n_features):
            Q[(i, i)] += P * (1 - 2 * K)
            for j in range(i + 1, n_features):
                Q[(i, j)] = Q.get((i, j), 0.0) + 2 * P
    return Q


def qubo_energy(Q: dict[tuple[int, int], float], x: np.ndarray) -> float:
    """Energy of a binary vector under a QUBO dict."""
    x = np.asarray(x)
    return float(sum(w * x[i] * x[j] for (i, j), w in Q.items()))


def build_qubo_problem(
    correlation_matrix: np.ndarray,
    relevance_scores: np.ndarray,
    n_features: int,
    max_features: int = 6,
    lambda_penalty: float = 0.3,
    correlation_threshold: float = 0.7
):
    """
    Same objective as a Qiskit QuadraticProgram with an explicit
    sum(x) == max_features constraint (Qiskit converts it to a penalty).

    Returns None if Qiskit is unavailable.
    """
    if not QISKIT_AVAILABLE:
        return None

    qp = QuadraticProgram('feature_selection')
    for i in range(n_features):
        qp.binary_var(name=f'x{i}')

    linear = {f'x{i}': -float(relevance_scores[i]) for i in range(n_features)}
    quadratic = {
        (f'x{i}', f'x{j}'): lambda_penalty * float(correlation_matrix[i][j])
        for i in range(n_features) for j in range(i + 1, n_features)
        if abs(correlation_matrix[i][j]) > correlation_threshold
    }
    qp.minimize(linear=linear, quadratic=quadratic)
    qp.linear_constraint(linear={f'x{i}': 1 for i in range(n_features)},
                         sense='==', rhs=max_features, name='cardinality')
    return qp


def interpret_qubo_result(
    selected_indices: list[int],
    feature_names: list[str],
    relevance_scores: np.ndarray
) -> pd.DataFrame:
    """Per-feature summary: name, index, relevance, selected flag (sorted by relevance)."""
    selected = set(int(i) for i in selected_indices)
    df = pd.DataFrame([
        {'feature': name, 'index': i, 'relevance': float(relevance_scores[i]),
         'selected': i in selected}
        for i, name in enumerate(feature_names)
    ])
    return df.sort_values('relevance', ascending=False)
