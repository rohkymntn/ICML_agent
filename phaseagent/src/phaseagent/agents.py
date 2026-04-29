"""Active boundary-discovery policies and the simulator that compares them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .phase import (
    compute_viability_by_distance,
    fit_phase_boundary,
    sigmoid,
)


class QueryPolicy:
    name = "base"

    def select_batch(
        self,
        observed_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        raise NotImplementedError


def _sample(rng: np.random.Generator, df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n <= 0 or len(df) == 0:
        return df.iloc[:0]
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx]


class RandomPolicy(QueryPolicy):
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_batch(self, observed_df, pool_df, batch_size):
        return _sample(self.rng, pool_df, batch_size)


class UniformShellPolicy(QueryPolicy):
    name = "uniform_shell"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_batch(self, observed_df, pool_df, batch_size):
        if len(pool_df) == 0:
            return pool_df.iloc[:0]
        shells = sorted(pool_df["mutation_distance"].unique())
        per_shell = max(1, batch_size // max(1, len(shells)))
        rows, remaining = [], batch_size
        for d in shells:
            if remaining <= 0:
                break
            sub = pool_df[pool_df["mutation_distance"] == d]
            take = min(per_shell, len(sub), remaining)
            picked = _sample(self.rng, sub, take)
            if len(picked):
                rows.append(picked)
                remaining -= len(picked)
        if remaining > 0:
            already = pd.concat(rows).index if rows else pd.Index([])
            leftover = pool_df.drop(index=already, errors="ignore")
            picked = _sample(self.rng, leftover, remaining)
            if len(picked):
                rows.append(picked)
        return pd.concat(rows) if rows else pool_df.iloc[:0]


class UncertaintyShellPolicy(QueryPolicy):
    name = "uncertainty_shell"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_batch(self, observed_df, pool_df, batch_size):
        if len(pool_df) == 0:
            return pool_df.iloc[:0]
        if len(observed_df):
            stats = observed_df.groupby("mutation_distance")["viable"].agg(["mean", "size"])
        else:
            stats = pd.DataFrame(columns=["mean", "size"])
        shells = sorted(pool_df["mutation_distance"].unique())
        scores = {}
        for d in shells:
            if d in stats.index:
                p, n = float(stats.loc[d, "mean"]), int(stats.loc[d, "size"])
            else:
                p, n = 0.5, 0
            scores[d] = (p * (1 - p) + 0.05) / (n + 1)
        return _allocate_by_score(self.rng, pool_df, scores, batch_size)


class BoundaryGreedyPolicy(QueryPolicy):
    name = "boundary_greedy"

    def __init__(self, rng: np.random.Generator, min_count: int = 3):
        self.rng = rng
        self.min_count = min_count

    def select_batch(self, observed_df, pool_df, batch_size):
        if len(observed_df) < 10 or len(pool_df) == 0:
            return UniformShellPolicy(self.rng).select_batch(observed_df, pool_df, batch_size)
        v_by_d = compute_viability_by_distance(observed_df, min_count_per_distance=self.min_count)
        fit = fit_phase_boundary(v_by_d)
        if not fit["fit_success"]:
            return UniformShellPolicy(self.rng).select_batch(observed_df, pool_df, batch_size)
        dc = fit["dc"]
        shells = sorted(pool_df["mutation_distance"].unique())
        scores = {d: float(np.exp(-abs(d - dc))) for d in shells}
        return _allocate_by_score(self.rng, pool_df, scores, batch_size)


class PhaseAgentPolicy(QueryPolicy):
    name = "phaseagent"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_batch(self, observed_df, pool_df, batch_size):
        if len(pool_df) == 0:
            return pool_df.iloc[:0]
        if len(observed_df):
            stats = observed_df.groupby("mutation_distance")["viable"].agg(["mean", "size"])
        else:
            stats = pd.DataFrame(columns=["mean", "size"])

        fit_dc, fit_alpha, fit_success = float("nan"), float("nan"), False
        if len(observed_df) >= 10:
            v_by_d = compute_viability_by_distance(observed_df, min_count_per_distance=3)
            fit = fit_phase_boundary(v_by_d)
            fit_dc, fit_alpha, fit_success = fit["dc"], fit["alpha"], fit["fit_success"]

        shells = sorted(pool_df["mutation_distance"].unique())
        scores = {}
        for d in shells:
            if d in stats.index:
                p, n = float(stats.loc[d, "mean"]), int(stats.loc[d, "size"])
            else:
                p, n = 0.5, 0
            uncertainty = (p * (1 - p) + 0.05) / (n + 1)
            if fit_success and np.isfinite(fit_dc) and np.isfinite(fit_alpha):
                v_d = float(sigmoid(d, fit_dc, fit_alpha))
                susceptibility = abs(fit_alpha) * v_d * (1.0 - v_d) + 1e-3
            else:
                susceptibility = 1.0
            availability = float(np.log1p(len(pool_df[pool_df["mutation_distance"] == d])))
            scores[d] = uncertainty * susceptibility * availability
        return _allocate_by_score(self.rng, pool_df, scores, batch_size)


def _allocate_by_score(
    rng: np.random.Generator,
    pool_df: pd.DataFrame,
    scores: dict[int, float],
    batch_size: int,
) -> pd.DataFrame:
    total = sum(max(0.0, s) for s in scores.values())
    if total <= 0 or len(pool_df) == 0:
        return UniformShellPolicy(rng).select_batch(pool_df.iloc[:0], pool_df, batch_size)
    rows, remaining = [], batch_size
    for d in sorted(scores, key=lambda k: -scores[k]):
        if remaining <= 0:
            break
        sub = pool_df[pool_df["mutation_distance"] == d]
        if len(sub) == 0:
            continue
        quota = int(np.ceil(scores[d] / total * batch_size))
        take = min(max(1, quota), len(sub), remaining)
        picked = _sample(rng, sub, take)
        if len(picked):
            rows.append(picked)
            remaining -= len(picked)
    return pd.concat(rows) if rows else pool_df.iloc[:0]


def _fit_truth(df: pd.DataFrame, min_count: int = 10) -> dict:
    return fit_phase_boundary(compute_viability_by_distance(df, min_count_per_distance=min_count))


def simulate_boundary_discovery(
    df: pd.DataFrame,
    policy: QueryPolicy,
    initial_n: int = 20,
    batch_size: int = 10,
    n_steps: int = 20,
    n_repeats: int = 20,
    random_state: int = 0,
) -> pd.DataFrame:
    truth = _fit_truth(df)
    rows = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(random_state + rep)
        if hasattr(policy, "rng"):
            policy.rng = rng
        pool = df.reset_index(drop=True)
        n0 = min(initial_n, len(pool))
        init_idx = rng.choice(pool.index.to_numpy(), size=n0, replace=False)
        observed = pool.loc[init_idx]
        pool_remaining = pool.drop(index=init_idx)

        for step in range(n_steps + 1):
            v_by_d = compute_viability_by_distance(observed, min_count_per_distance=3)
            fit = fit_phase_boundary(v_by_d)
            dc_err = (
                abs(fit["dc"] - truth["dc"])
                if np.isfinite(fit["dc"]) and np.isfinite(truth["dc"])
                else float("nan")
            )
            rows.append({
                "policy": policy.name,
                "repeat": rep,
                "step": step,
                "n_queries": int(len(observed)),
                "dc_hat": fit["dc"],
                "alpha_hat": fit["alpha"],
                "dc_true": truth["dc"],
                "alpha_true": truth["alpha"],
                "dc_error": dc_err,
                "fit_success": fit["fit_success"],
            })
            if step == n_steps or len(pool_remaining) == 0:
                break
            batch = policy.select_batch(observed, pool_remaining, batch_size)
            if len(batch) == 0:
                break
            observed = pd.concat([observed, batch])
            pool_remaining = pool_remaining.drop(index=batch.index, errors="ignore")
    return pd.DataFrame(rows)
