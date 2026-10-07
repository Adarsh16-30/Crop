"""
Unit Tests — Quantum Feature Selection
"""

import pytest
import numpy as np

from src.quantum.qubo_feature_selector import (
    build_qubo_matrix, compute_relevance_scores,
    compute_correlation_matrix, interpret_qubo_result
)
from src.quantum.annealing_optimizer import get_solver_info


class TestQUBOConstruction:
    """Tests for QUBO problem construction."""

    def test_qubo_matrix_builds_without_error(self):
        corr = np.random.rand(10, 10)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        relevance = np.random.rand(10)
        Q = build_qubo_matrix(corr, relevance, n_features=10)
        assert Q is not None
        assert isinstance(Q, dict)

    def test_qubo_has_diagonal_entries(self):
        """Linear terms should appear as diagonal entries."""
        relevance = np.array([0.5, 0.3, 0.8])
        corr = np.eye(3)  # No correlations
        Q = build_qubo_matrix(corr, relevance, n_features=3)
        # Diagonal entries should be negative relevance
        for i in range(3):
            assert (i, i) in Q
            assert Q[(i, i)] == pytest.approx(-relevance[i])

    def test_qubo_penalizes_correlated_pairs(self):
        """Highly correlated pairs should have positive off-diagonal entries."""
        relevance = np.array([0.5, 0.5])
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])  # High correlation
        Q = build_qubo_matrix(corr, relevance, n_features=2,
                               correlation_threshold=0.7)
        assert (0, 1) in Q
        assert Q[(0, 1)] > 0  # Penalty should be positive

    def test_qubo_ignores_low_correlation(self):
        """Low correlation pairs should not appear in QUBO."""
        relevance = np.array([0.5, 0.5])
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])  # Low correlation
        Q = build_qubo_matrix(corr, relevance, n_features=2,
                               correlation_threshold=0.7)
        assert (0, 1) not in Q


class TestCorrelationAndRelevance:
    """Tests for relevance and correlation computation."""

    def test_correlation_matrix_shape(self):
        X = np.random.rand(100, 10)
        corr = compute_correlation_matrix(X)
        assert corr.shape == (10, 10)

    def test_correlation_matrix_symmetric(self):
        X = np.random.rand(100, 5)
        corr = compute_correlation_matrix(X)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_correlation_diagonal_is_one(self):
        X = np.random.rand(100, 5)
        corr = compute_correlation_matrix(X)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(5))

    def test_correlation_values_in_range(self):
        X = np.random.rand(100, 5)
        corr = compute_correlation_matrix(X)
        assert (corr >= 0).all()  # We use absolute values
        assert (corr <= 1.0 + 1e-8).all()


class TestQUBOInterpretation:
    """Tests for result interpretation."""

    def test_interpret_result(self):
        selected = [0, 2, 4]
        names = ['f0', 'f1', 'f2', 'f3', 'f4']
        relevance = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        df = interpret_qubo_result(selected, names, relevance)
        assert len(df) == 5
        assert df[df['selected']].shape[0] == 3
        assert 'feature' in df.columns
        assert 'relevance' in df.columns


class TestSolverInfo:
    """Tests for solver availability."""

    def test_solver_info_returns_dict(self):
        info = get_solver_info()
        assert isinstance(info, dict)
        assert 'recommended' in info
        assert info['recommended'] in ['neal', 'qiskit', 'scipy']


class TestBruteForceFallback:
    """Tests for the scipy fallback solver."""

    def test_small_problem_solvable(self):
        from src.quantum.annealing_optimizer import _scipy_fallback
        # Simple 3-variable problem: prefer x0 and x2
        Q = {
            (0, 0): -1.0,  # Strong preference for x0
            (1, 1): -0.1,  # Weak preference for x1
            (2, 2): -0.8,  # Moderate preference for x2
            (0, 1): 0.5,   # Penalize x0 + x1 together
        }
        selected = _scipy_fallback(Q)
        assert isinstance(selected, list)
        assert len(selected) <= 3
        # x0 should definitely be selected (strongest preference)
        assert 0 in selected
