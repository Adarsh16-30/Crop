"""
QUBO Solvers
============
1. Simulated annealing via `dwave.samplers` (falls back to the deprecated
   `neal` package if that is all that is installed).
2. Qiskit QAOA on a statevector simulator (small problems only).
3. Pure-numpy fallback: exact brute force for n <= 20, otherwise a
   multi-start greedy + single-flip local search.

All solvers return a sorted list of selected feature indices.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SA_BACKEND = None
try:
    from dwave.samplers import SimulatedAnnealingSampler
    SA_BACKEND = 'dwave.samplers'
except ImportError:
    try:
        from neal import SimulatedAnnealingSampler  # deprecated upstream
        SA_BACKEND = 'neal'
    except ImportError:
        SimulatedAnnealingSampler = None

QISKIT_AVAILABLE = False
try:
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    QISKIT_AVAILABLE = True
except ImportError:
    pass

# Kept for backwards compatibility with callers checking this flag.
NEAL_AVAILABLE = SA_BACKEND is not None
BRUTE_FORCE_MAX_VARS = 20


def _num_vars(qubo: dict[tuple[int, int], float]) -> int:
    return max((max(i, j) for i, j in qubo), default=-1) + 1


def _to_matrix(qubo: dict[tuple[int, int], float], n: int) -> np.ndarray:
    Q = np.zeros((n, n))
    for (i, j), w in qubo.items():
        Q[i, j] += w
    return Q


def _energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x)


def _repair(selected: list[int], qubo: dict, max_features: int | None) -> list[int]:
    """Last-resort trim if a solver still returns more than max_features."""
    if max_features is None or len(selected) <= max_features:
        return sorted(selected)
    logger.warning("Solver returned %d features (> %d); trimming by relevance",
                   len(selected), max_features)
    ranked = sorted(selected, key=lambda idx: qubo.get((idx, idx), 0.0))
    return sorted(ranked[:max_features])


def run_simulated_annealing(
    qubo_matrix: dict[tuple[int, int], float],
    n_reads: int = 1000,
    num_sweeps: int = 2000,
    seed: int = 42,
    max_features: int | None = None
) -> list[int]:
    """Solve a QUBO by simulated annealing; numpy fallback if no sampler is installed."""
    if SimulatedAnnealingSampler is None:
        logger.warning("No simulated-annealing package installed; using numpy fallback")
        return numpy_fallback(qubo_matrix, max_features=max_features, seed=seed)

    response = SimulatedAnnealingSampler().sample_qubo(
        qubo_matrix, num_reads=n_reads, num_sweeps=num_sweeps, seed=seed
    )
    selected = [k for k, v in response.first.sample.items() if v == 1]
    logger.info("Simulated annealing (%s): %d selected, energy %.4f, %d reads x %d sweeps",
                SA_BACKEND, len(selected), response.first.energy, n_reads, num_sweeps)
    return _repair(selected, qubo_matrix, max_features)


def run_qaoa(qubo_problem, reps: int = 2, max_vars: int = 15, seed: int = 42) -> list[int]:
    """Solve a Qiskit QuadraticProgram with QAOA on a statevector simulator."""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is not available. Install qiskit-algorithms "
                           "and qiskit-optimization.")
    n_vars = qubo_problem.get_num_vars()
    if n_vars > max_vars:
        raise ValueError(f"QAOA is impractical for {n_vars} variables (limit {max_vars}).")

    logger.info("Running QAOA (reps=%d) on %d variables", reps, n_vars)
    qaoa = QAOA(sampler=StatevectorSampler(seed=seed), optimizer=COBYLA(maxiter=200), reps=reps)
    result = MinimumEigenOptimizer(qaoa).solve(qubo_problem)
    selected = [i for i, v in enumerate(result.x) if round(v) == 1]
    logger.info("QAOA: selected %s, objective %.4f, status %s", selected, result.fval,
                result.status)
    return selected


def numpy_fallback(
    qubo_matrix: dict[tuple[int, int], float],
    max_features: int | None = None,
    seed: int = 42,
    n_starts: int = 50
) -> list[int]:
    """
    Exact brute force for n <= 20; otherwise multi-start greedy descent with
    single-bit-flip local search (no exponential blow-up).
    """
    n = _num_vars(qubo_matrix)
    if n == 0:
        return []
    Q = _to_matrix(qubo_matrix, n)

    if n <= BRUTE_FORCE_MAX_VARS:
        states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        energies = np.einsum('si,ij,sj->s', states, Q, states)
        best = states[int(np.argmin(energies))]
        logger.info("Brute force over %d variables: energy %.4f", n, energies.min())
    else:
        rng = np.random.default_rng(seed)
        best, best_e = None, np.inf
        for _ in range(n_starts):
            x = (rng.random(n) < 0.5).astype(float)
            improved = True
            while improved:
                improved = False
                for i in rng.permutation(n):
                    x_flip = x.copy()
                    x_flip[i] = 1 - x_flip[i]
                    if _energy(Q, x_flip) < _energy(Q, x) - 1e-12:
                        x, improved = x_flip, True
            e = _energy(Q, x)
            if e < best_e:
                best, best_e = x, e
        logger.info("Local search over %d variables: energy %.4f", n, best_e)

    selected = [i for i in range(n) if best[i] == 1]
    return _repair(selected, qubo_matrix, max_features)


# Old name, kept so existing imports keep working.
_scipy_fallback = numpy_fallback


def get_solver_info() -> dict:
    """Report which solvers are available."""
    if SA_BACKEND:
        recommended = 'simulated_annealing'
    elif QISKIT_AVAILABLE:
        recommended = 'qiskit'
    else:
        recommended = 'numpy'
    return {
        'simulated_annealing_backend': SA_BACKEND,
        'neal_available': NEAL_AVAILABLE,
        'qiskit_available': QISKIT_AVAILABLE,
        'recommended': recommended,
    }
