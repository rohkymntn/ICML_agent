"""Uncertainty-aware active querying for phase-boundary discovery."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import BoundaryGreedyPolicy, RandomPolicy, UniformShellPolicy
from .flow_posterior import posterior_curve_samples, sample_phase_posterior


def candidate_shells(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize query availability by mutation-distance shell."""
    if len(pool_df) == 0:
        return pd.DataFrame(columns=["mutation_distance", "n"])
    return (
        pool_df.groupby("mutation_distance")
        .size()
        .rename("n")
        .reset_index()
        .sort_values("mutation_distance")
        .reset_index(drop=True)
    )


def _binary_entropy(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def expected_information_gain(
    phase_samples: pd.DataFrame,
    candidate_batch: pd.DataFrame,
    cost_per_query: float = 0.0,
) -> float:
    """Approximate MI by disagreement in sampled survival probabilities."""
    if len(candidate_batch) == 0 or len(phase_samples) == 0:
        return 0.0
    depths = candidate_batch["mutation_distance"].to_numpy(dtype=float)
    probs = []
    for _, z in phase_samples.iterrows():
        probs.append(1.0 / (1.0 + np.exp(z["alpha"] * (depths - z["dc"]))))
    p = np.vstack(probs)
    marginal_entropy = _binary_entropy(p.mean(axis=0)).sum()
    conditional_entropy = _binary_entropy(p).mean(axis=0).sum()
    return float(marginal_entropy - conditional_entropy - cost_per_query * len(candidate_batch))


class SpectralPriorPhaseAgent:
    """Batch selector using a survival prior plus posterior uncertainty."""

    def __init__(
        self,
        n_posterior_samples: int = 512,
        n_boot: int = 25,
        cost_per_query: float = 0.0,
        seed: int = 0,
    ):
        self.n_posterior_samples = int(n_posterior_samples)
        self.n_boot = int(n_boot)
        self.cost_per_query = float(cost_per_query)
        self.seed = int(seed)

    def select_batch(
        self,
        observed_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        batch_size: int,
        spectrum=None,
        structure=None,
    ) -> pd.DataFrame:
        """Greedily select variants maximizing approximate phase information gain."""
        if len(pool_df) == 0 or batch_size <= 0:
            return pool_df.iloc[:0].copy()
        samples = sample_phase_posterior(
            spectrum=spectrum,
            structure=structure,
            observed_df=observed_df,
            n_samples=self.n_posterior_samples,
            n_boot=self.n_boot,
            seed=self.seed,
        )
        selected = []
        remaining = pool_df.copy()
        for _ in range(min(batch_size, len(pool_df))):
            best_idx = None
            best_score = -np.inf
            # Score shell representatives first; this keeps selection stable and cheap.
            for _, shell in candidate_shells(remaining).iterrows():
                shell_df = remaining[remaining["mutation_distance"] == shell["mutation_distance"]]
                candidate = shell_df.head(1)
                score = expected_information_gain(samples, candidate, cost_per_query=self.cost_per_query)
                if score > best_score:
                    best_score = score
                    best_idx = candidate.index[0]
            selected.append(best_idx)
            remaining = remaining.drop(index=best_idx)
        return pool_df.loc[selected].copy()

    def posterior_curves(self, observed_df: pd.DataFrame, depths) -> pd.DataFrame:
        samples = sample_phase_posterior(
            observed_df=observed_df,
            n_samples=self.n_posterior_samples,
            n_boot=self.n_boot,
            seed=self.seed,
        )
        return posterior_curve_samples(samples, depths)


def random_batch(observed_df: pd.DataFrame, pool_df: pd.DataFrame, batch_size: int, seed: int = 0) -> pd.DataFrame:
    del observed_df
    return RandomPolicy(np.random.default_rng(seed)).select_batch(pd.DataFrame(), pool_df, batch_size)


def uniform_shell_batch(observed_df: pd.DataFrame, pool_df: pd.DataFrame, batch_size: int, seed: int = 0) -> pd.DataFrame:
    return UniformShellPolicy(np.random.default_rng(seed)).select_batch(observed_df, pool_df, batch_size)


def boundary_greedy_batch(observed_df: pd.DataFrame, pool_df: pd.DataFrame, batch_size: int, seed: int = 0) -> pd.DataFrame:
    return BoundaryGreedyPolicy(np.random.default_rng(seed)).select_batch(observed_df, pool_df, batch_size)


def old_phaseagent_heuristic(observed_df: pd.DataFrame, pool_df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Expose the old heuristic as an explicit negative baseline."""
    return boundary_greedy_batch(observed_df, pool_df, batch_size=batch_size)


def posterior_thompson_batch(
    observed_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    batch_size: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample a boundary and query variants near it."""
    if len(pool_df) == 0 or batch_size <= 0:
        return pool_df.iloc[:0].copy()
    z = sample_phase_posterior(observed_df=observed_df, n_samples=1, seed=seed).iloc[0]
    out = pool_df.copy()
    out["_distance_to_sampled_boundary"] = np.abs(out["mutation_distance"].to_numpy(dtype=float) - float(z["dc"]))
    return out.nsmallest(min(batch_size, len(out)), "_distance_to_sampled_boundary").drop(columns=["_distance_to_sampled_boundary"])


def mutual_information_phaseagent(
    observed_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    batch_size: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Convenience function for the spectral-prior MI selector."""
    agent = SpectralPriorPhaseAgent(seed=seed)
    return agent.select_batch(observed_df, pool_df, batch_size=batch_size)


POLICIES = {
    "random": random_batch,
    "uniform_shell": uniform_shell_batch,
    "boundary_greedy": boundary_greedy_batch,
    "old_phaseagent_heuristic": old_phaseagent_heuristic,
    "posterior_thompson": posterior_thompson_batch,
    "mutual_information_phaseagent": mutual_information_phaseagent,
    "spectral_prior_phaseagent": mutual_information_phaseagent,
}
