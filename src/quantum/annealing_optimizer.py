"""
Quantum / Simulated Annealing Optimizer
========================================
Solves the QUBO feature selection problem using:
    1. D-Wave Neal Simulated Annealing (primary)
    2. Qiskit QAOA (optional, for small problems ≤15 features)
    3. SciPy brute-force fallback (for very small problems)

All solvers return a list of selected feature indices.
"""

import numpy as np

# Try importing D-Wave Neal
NEAL_AVAILABLE = False
try:
    import neal
    NEAL_AVAILABLE = True
except ImportError:
    pass

# Try importing Qiskit for QAOA
QISKIT_AVAILABLE = False
try:
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA
    from qiskit.primitives import StatevectorSampler
    QISKIT_AVAILABLE = True
except ImportError:
    pass


def run_simulated_annealing(
    qubo_matrix: dict[tuple[int, int], float],
    n_reads: int = 1000,
    num_sweeps: int = 2000,
    seed: int = 42,
    max_features: int | None = None
) -> list[int]:
    """
    Solve QUBO using D-Wave's Neal Simulated Annealing sampler.

    Parameters
    ----------
    qubo_matrix : dict
        QUBO dictionary of {(i, j): weight} pairs.
    n_reads : int
        Number of annealing runs.
    num_sweeps : int
        Number of sweeps per run.
    seed : int
        Random seed.

    Returns
    -------
    list[int]
        Indices of selected features (where x_i = 1).
    """
    if NEAL_AVAILABLE:
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(
            qubo_matrix,
            num_reads=n_reads,
            num_sweeps=num_sweeps,
            seed=seed
        )
        best_sample = response.first.sample
        selected = [k for k, v in sorted(best_sample.items()) if v == 1]
        if max_features is not None and len(selected) > max_features:
            # Diagonal value = -relevance (more negative = more relevant), so
            # an ascending sort ranks the most relevant features first.
            ranked = sorted(
                selected,
                key=lambda idx: qubo_matrix.get((idx, idx), 0.0)
            )
            selected = ranked[:max_features]
        energy = response.first.energy

        print(f"Simulated Annealing Result:")
        print(f"  Selected features: {selected}")
        print(f"  Energy: {energy:.4f}")
        print(f"  Num reads: {n_reads}, Num sweeps: {num_sweeps}")

        return selected
    else:
        print("WARNING: D-Wave Neal not available. Using scipy fallback.")
        return _scipy_fallback(qubo_matrix, max_features=max_features)


def run_qaoa(
    qubo_problem,
    reps: int = 2,
    max_features: int = 15
) -> list[int]:
    """
    Solve QUBO using Qiskit QAOA (for small problems only).

    Parameters
    ----------
    qubo_problem : QuadraticProgram
        Qiskit QuadraticProgram instance.
    reps : int
        Number of QAOA repetitions.
    max_features : int
        Safety limit — refuse problems larger than this.

    Returns
    -------
    list[int]
        Selected feature indices.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is not available. Install qiskit-algorithms "
                          "and qiskit-optimization.")

    n_vars = qubo_problem.get_num_vars()
    if n_vars > max_features:
        raise ValueError(
            f"QAOA is impractical for {n_vars} variables (limit: {max_features}). "
            f"Use simulated annealing instead."
        )

    print(f"Running QAOA with {reps} repetitions on {n_vars} variables...")

    sampler = StatevectorSampler()
    qaoa = QAOA(sampler=sampler, reps=reps)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qubo_problem)

    selected = [i for i, v in enumerate(result.x) if v == 1]

    print(f"QAOA Result:")
    print(f"  Selected features: {selected}")
    print(f"  Objective value: {result.fval:.4f}")
    print(f"  Status: {result.status}")

    return selected


def _scipy_fallback(
    qubo_matrix: dict[tuple[int, int], float],
    max_features: int | None = None
) -> list[int]:
    """
    Brute-force QUBO solver for small problems using numpy.

    Evaluates all 2^n binary combinations. Only practical for n ≤ 20.

    Parameters
    ----------
    qubo_matrix : dict
        QUBO dictionary.

    Returns
    -------
    list[int]
        Selected feature indices.
    """
    # Determine number of variables
    all_vars = set()
    for (i, j) in qubo_matrix.keys():
        all_vars.add(i)
        all_vars.add(j)
    n = max(all_vars) + 1 if all_vars else 0

    if n > 20:
        raise ValueError(f"Brute-force is impractical for {n} variables. "
                        f"Install dwave-neal for simulated annealing.")

    print(f"Scipy fallback: brute-force search over {n} variables ({2**n} states)...")

    best_energy = float('inf')
    best_state = None

    for state_int in range(2 ** n):
        # Convert integer to binary vector
        x = np.array([(state_int >> bit) & 1 for bit in range(n)])

        # Evaluate QUBO energy
        energy = 0.0
        for (i, j), weight in qubo_matrix.items():
            energy += weight * x[i] * x[j]

        if energy < best_energy:
            best_energy = energy
            best_state = x.copy()

    selected = [i for i in range(n) if best_state[i] == 1]
    if max_features is not None and len(selected) > max_features:
        # Diagonal value = -relevance (more negative = more relevant), so
        # an ascending sort ranks the most relevant features first.
        ranked = sorted(
            selected,
            key=lambda idx: qubo_matrix.get((idx, idx), 0.0)
        )
        selected = ranked[:max_features]

    print(f"Brute-force Result:")
    print(f"  Selected features: {selected}")
    print(f"  Energy: {best_energy:.4f}")

    return selected


def get_solver_info() -> dict:
    """Report which solvers are available."""
    return {
        'neal_available': NEAL_AVAILABLE,
        'qiskit_available': QISKIT_AVAILABLE,
        'recommended': 'neal' if NEAL_AVAILABLE else ('qiskit' if QISKIT_AVAILABLE else 'scipy')
    }
