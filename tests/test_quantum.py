"""
Unit Tests — QUBO Feature Selection
"""

import numpy as np
import pytest

from src.quantum.annealing_optimizer import (
    get_solver_info,
    numpy_fallback,
    run_simulated_annealing,
)
from src.quantum.integration import select_mrmr, select_top_k
from src.quantum.qubo_feature_selector import (
    build_qubo_matrix,
    compute_correlation_matrix,
    interpret_qubo_result,
    qubo_energy,
)


def _random_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, n))
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=200)  # 0 and 1 nearly duplicate
    return compute_correlation_matrix(X), rng.random(n)


class TestQUBOConstruction:

    def test_diagonal_is_negative_relevance_without_cardinality(self):
        relevance = np.array([0.5, 0.3, 0.8])
        Q = build_qubo_matrix(np.eye(3), relevance, n_features=3)
        for i in range(3):
            assert Q[(i, i)] == pytest.approx(-relevance[i])

    def test_correlated_pairs_penalized_and_low_ignored(self):
        hi = build_qubo_matrix(np.array([[1, .9], [.9, 1]]), np.array([.5, .5]), 2)
        lo = build_qubo_matrix(np.array([[1, .3], [.3, 1]]), np.array([.5, .5]), 2)
        assert hi[(0, 1)] > 0 and (0, 1) not in lo

    def test_cardinality_penalty_makes_size_k_optimal(self):
        """Brute force over the QUBO must land on exactly K features."""
        corr, relevance = _random_problem(10)
        for k in (2, 4, 7):
            Q = build_qubo_matrix(corr, relevance, 10, max_features=k)
            assert len(numpy_fallback(Q)) == k

    def test_energy_function(self):
        Q = {(0, 0): -1.0, (1, 1): -2.0, (0, 1): 5.0}
        assert qubo_energy(Q, [1, 1]) == pytest.approx(2.0)
        assert qubo_energy(Q, [0, 1]) == pytest.approx(-2.0)


class TestSolvers:

    def test_annealer_returns_exactly_k(self):
        corr, relevance = _random_problem(24, seed=1)
        Q = build_qubo_matrix(corr, relevance, 24, max_features=8)
        selected = run_simulated_annealing(Q, n_reads=50, num_sweeps=500, max_features=8)
        assert len(selected) == 8

    def test_annealer_avoids_redundant_pair_when_penalized(self):
        relevance = np.array([1.0, 0.99, 0.2, 0.2])
        corr = np.eye(4)
        corr[0, 1] = corr[1, 0] = 0.99
        Q = build_qubo_matrix(corr, relevance, 4, lambda_penalty=5.0, max_features=2)
        selected = run_simulated_annealing(Q, n_reads=50, num_sweeps=500, max_features=2)
        assert not {0, 1} <= set(selected)

    def test_brute_force_small_problem(self):
        Q = {(0, 0): -1.0, (1, 1): -0.1, (2, 2): -0.8, (0, 1): 0.5}
        assert numpy_fallback(Q) == [0, 2]

    def test_fallback_handles_more_than_20_variables(self):
        """Previously raised ValueError for n > 20."""
        corr, relevance = _random_problem(30, seed=2)
        Q = build_qubo_matrix(corr, relevance, 30, max_features=6)
        assert len(numpy_fallback(Q, n_starts=5)) == 6

    def test_solver_info(self):
        info = get_solver_info()
        assert info['recommended'] in ('simulated_annealing', 'qiskit', 'numpy')


class TestBaselines:

    def test_top_k(self):
        assert select_top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2) == [1, 3]

    def test_mrmr_skips_duplicate_feature(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        signal = y + 0.3 * rng.normal(size=300)
        X = np.column_stack([signal, signal + 1e-3 * rng.normal(size=300),
                             y + 0.6 * rng.normal(size=300), rng.normal(size=300)])
        selected = select_mrmr(X, y, 2)
        assert not {0, 1} <= set(selected)

    def test_interpret_result(self):
        df = interpret_qubo_result([0, 2], ['f0', 'f1', 'f2'], np.array([0.9, 0.1, 0.8]))
        assert df['selected'].sum() == 2 and len(df) == 3
