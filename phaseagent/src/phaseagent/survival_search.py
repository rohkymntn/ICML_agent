"""Survival-aware design search utilities."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import evaluate_search
from .search import _featurize, _fit_surrogate, phase_aware_search, random_search, novelty_search, fitness_proxy_search


def _lookup_survival(distances: np.ndarray, survival_by_d: dict[int, float], eps: float) -> np.ndarray:
    if not survival_by_d:
        return np.ones_like(distances, dtype=float)
    known_depths = np.array(sorted(survival_by_d), dtype=float)
    known_values = np.array([survival_by_d[int(d)] for d in known_depths], dtype=float)
    values = np.interp(distances.astype(float), known_depths, known_values, left=known_values[0], right=known_values[-1])
    return np.clip(values, eps, 1.0)


def survival_curve_to_dict(
    curve: pd.DataFrame,
    survival_col: str = "survival_pred",
) -> dict[int, float]:
    """Convert a survival-curve dataframe into a depth -> probability map."""
    if curve is None or len(curve) == 0:
        return {}
    return {
        int(r["mutation_distance"]): float(r[survival_col])
        for _, r in curve.dropna(subset=["mutation_distance", survival_col]).iterrows()
    }


def score_candidates_survival_aware(
    pool_df: pd.DataFrame,
    f_hat: Iterable[float],
    v_hat_by_d: dict[int, float],
    beta: float = 0.0,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Score candidates with ``f_hat + beta*d + gamma*log V(d)``."""
    out = pool_df.copy()
    distance = out["mutation_distance"].to_numpy(dtype=float)
    survival = _lookup_survival(distance, v_hat_by_d, eps=eps)
    out["pred_score"] = np.asarray(list(f_hat), dtype=float)
    out["survival_prior"] = survival
    out["survival_adjusted_score"] = out["pred_score"] + beta * distance + gamma * np.log(survival)
    return out


def survival_prior_search(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    survival_curve: pd.DataFrame | dict[int, float],
    budget: int,
    beta: float = 0.0,
    gamma: float = 1.0,
    survival_col: str = "survival_pred",
) -> pd.DataFrame:
    """Select top candidates under the survival-aware score."""
    if len(pool_df) == 0:
        return pool_df.iloc[:0].copy()
    model = _fit_surrogate(train_df)
    f_hat = model.predict(_featurize(pool_df))
    v_by_d = survival_curve if isinstance(survival_curve, dict) else survival_curve_to_dict(survival_curve, survival_col)
    scored = score_candidates_survival_aware(pool_df, f_hat, v_by_d, beta=beta, gamma=gamma)
    return scored.nlargest(min(budget, len(scored)), "survival_adjusted_score")


def run_frontier_sweep(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    survival_curve: pd.DataFrame | dict[int, float],
    budget: int,
    beta_grid: Iterable[float],
    gamma_grid: Iterable[float],
    dc_true: float = np.nan,
    survival_col: str = "survival_pred",
) -> pd.DataFrame:
    """Sweep beta/gamma to trace the novelty-hit frontier."""
    rows = []
    for beta in beta_grid:
        for gamma in gamma_grid:
            selected = survival_prior_search(
                train_df,
                pool_df,
                survival_curve=survival_curve,
                budget=budget,
                beta=float(beta),
                gamma=float(gamma),
                survival_col=survival_col,
            )
            metrics = evaluate_search(selected, dc_true=dc_true)
            rows.append(
                {
                    "policy": "survival_prior",
                    "budget": budget,
                    "beta": float(beta),
                    "gamma": float(gamma),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_survival_search_benchmark(
    datasets: dict[str, pd.DataFrame],
    survival_curves: dict[str, pd.DataFrame | dict[int, float]],
    boundaries: dict[str, dict] | None = None,
    budget: int = 50,
    seeds: Iterable[int] = (0, 1, 2),
    beta_grid: Iterable[float] = (0.0, 0.02, 0.05),
    gamma_grid: Iterable[float] = (0.1, 0.5, 1.0),
    train_fraction: float = 0.5,
    lambda_boundary: float = 0.5,
    survival_col: str = "survival_pred",
) -> pd.DataFrame:
    """Compare baseline search, old hard phase penalty, and survival priors."""
    rows = []
    boundaries = boundaries or {}
    for dataset_id, df in datasets.items():
        if dataset_id not in survival_curves:
            continue
        df = df.reset_index(drop=True)
        bd = boundaries.get(dataset_id, {})
        dc_true = bd.get("dc_true", bd.get("dc", np.nan))
        dc_hat = bd.get("dc", np.nan)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(df))
            n_train = max(1, int(train_fraction * len(df)))
            train = df.iloc[idx[:n_train]]
            pool = df.iloc[idx[n_train:]]
            baseline = {
                "random": random_search(pool, budget, seed=seed),
                "novelty": novelty_search(pool, budget, seed=seed),
                "fitness_proxy": fitness_proxy_search(train, pool, budget, seed=seed),
                "old_phase_aware_hard_penalty": phase_aware_search(
                    train,
                    pool,
                    dc_hat=dc_hat,
                    budget=budget,
                    lambda_boundary=lambda_boundary,
                    seed=seed,
                ),
            }
            for policy, selected in baseline.items():
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "policy": policy,
                        "budget": budget,
                        "seed": seed,
                        "beta": np.nan,
                        "gamma": np.nan,
                        **evaluate_search(selected, dc_true=dc_true),
                    }
                )
            frontier = run_frontier_sweep(
                train,
                pool,
                survival_curves[dataset_id],
                budget=budget,
                beta_grid=beta_grid,
                gamma_grid=gamma_grid,
                dc_true=dc_true,
                survival_col=survival_col,
            )
            frontier["dataset_id"] = dataset_id
            frontier["seed"] = seed
            rows.extend(frontier.to_dict(orient="records"))
    return pd.DataFrame(rows)
