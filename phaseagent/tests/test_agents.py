"""Unit tests for query policies. Uses small inline DataFrames as fixtures."""
import numpy as np
import pandas as pd
import pytest

from phaseagent.agents import (
    BoundaryGreedyPolicy,
    PhaseAgentPolicy,
    RandomPolicy,
    UncertaintyShellPolicy,
    UniformShellPolicy,
    simulate_boundary_discovery,
)


@pytest.fixture
def pool():
    """Inline fixture: a deterministic table with shells 0..10 of 50 rows each."""
    rng = np.random.default_rng(42)
    rows = []
    for d in range(11):
        for i in range(50):
            p = 1.0 / (1.0 + np.exp(d - 5))
            rows.append({
                "variant_id": f"v{d}_{i}",
                "mutation_distance": d,
                "fitness_norm": float(rng.beta(2 if rng.random() < p else 1, 5)),
                "viable": int(rng.random() < p),
                "mutation_notation": "A1V" * (d if d > 0 else 1),
            })
    return pd.DataFrame(rows)


def test_random_policy_returns_batch(pool):
    rng = np.random.default_rng(0)
    pol = RandomPolicy(rng)
    batch = pol.select_batch(pool.iloc[:10], pool.iloc[10:], batch_size=5)
    assert len(batch) == 5
    assert set(batch.columns) >= {"mutation_distance", "viable"}


def test_uniform_shell_diverse(pool):
    rng = np.random.default_rng(0)
    pol = UniformShellPolicy(rng)
    batch = pol.select_batch(pool.iloc[:10], pool.iloc[10:], batch_size=22)
    assert batch["mutation_distance"].nunique() >= 5


def test_uncertainty_shell_returns_batch(pool):
    rng = np.random.default_rng(0)
    pol = UncertaintyShellPolicy(rng)
    batch = pol.select_batch(pool.iloc[:10], pool.iloc[10:], batch_size=10)
    assert 1 <= len(batch) <= 10


def test_boundary_greedy_returns_batch(pool):
    rng = np.random.default_rng(0)
    pol = BoundaryGreedyPolicy(rng)
    observed = pool.sample(80, random_state=0)
    rest = pool.drop(index=observed.index)
    batch = pol.select_batch(observed, rest, batch_size=10)
    assert 1 <= len(batch) <= 10


def test_phaseagent_returns_batch(pool):
    rng = np.random.default_rng(0)
    pol = PhaseAgentPolicy(rng)
    observed = pool.sample(80, random_state=0)
    rest = pool.drop(index=observed.index)
    batch = pol.select_batch(observed, rest, batch_size=10)
    assert 1 <= len(batch) <= 10


def test_simulator_runs(pool):
    rng = np.random.default_rng(0)
    res = simulate_boundary_discovery(
        pool,
        policy=PhaseAgentPolicy(rng),
        initial_n=20,
        batch_size=10,
        n_steps=3,
        n_repeats=2,
    )
    assert len(res) > 0
    assert {"dc_hat", "dc_true", "n_queries", "policy", "repeat", "step"} <= set(res.columns)
