"""Additive and large-deviation survival baselines."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .spectrum import compute_single_mutant_effects, compute_wt_fitness


def _clean_effects(single_effects: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(single_effects), dtype=float)
    return x[np.isfinite(x)]


def monte_carlo_additive_survival(
    single_effects: Iterable[float],
    depths: Iterable[int],
    threshold: float,
    n_samples: int = 100_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate ``P(sum_{j=1}^d X_j >= threshold)`` by sampling effects."""
    effects = _clean_effects(single_effects)
    depths_arr = np.asarray(list(depths), dtype=int)
    rng = np.random.default_rng(seed)
    rows = []
    for d in depths_arr:
        if d <= 0:
            survival = 1.0 if threshold <= 0 else 0.0
        elif effects.size == 0:
            survival = np.nan
        else:
            draws = rng.choice(effects, size=(int(n_samples), int(d)), replace=True)
            survival = float(np.mean(draws.sum(axis=1) >= threshold))
        rows.append({"mutation_distance": int(d), "survival_additive_mc": survival})
    return pd.DataFrame(rows)


def fft_additive_survival(
    single_effects: Iterable[float],
    depths: Iterable[int],
    threshold: float,
    bins: int = 2048,
) -> pd.DataFrame:
    """Approximate additive survival by repeated discrete convolution."""
    effects = _clean_effects(single_effects)
    depths_arr = np.asarray(list(depths), dtype=int)
    if effects.size == 0:
        return pd.DataFrame(
            {"mutation_distance": depths_arr, "survival_additive_fft": np.nan}
        )
    bins = int(max(32, bins))
    lo, hi = float(effects.min()), float(effects.max())
    if lo == hi:
        vals = [
            1.0 if int(d) * lo >= threshold else 0.0
            for d in depths_arr
        ]
        return pd.DataFrame({"mutation_distance": depths_arr, "survival_additive_fft": vals})

    hist, edges = np.histogram(effects, bins=bins, range=(lo, hi), density=False)
    pmf = hist.astype(float) / hist.sum()
    dx = float(edges[1] - edges[0])
    center0 = float((edges[0] + edges[1]) / 2.0)
    conv_by_depth = {1: pmf}
    rows = []
    for d in depths_arr:
        d = int(d)
        if d <= 0:
            survival = 1.0 if threshold <= 0 else 0.0
        else:
            while max(conv_by_depth) < d:
                prev_d = max(conv_by_depth)
                conv_by_depth[prev_d + 1] = np.convolve(conv_by_depth[prev_d], pmf)
            conv = conv_by_depth[d]
            centers = d * center0 + dx * np.arange(len(conv))
            survival = float(conv[centers >= threshold].sum())
        rows.append({"mutation_distance": d, "survival_additive_fft": survival})
    return pd.DataFrame(rows)


def cumulant_generating_function(
    single_effects: Iterable[float],
    lambdas: Iterable[float],
) -> pd.DataFrame:
    """Estimate ``K(lambda) = log E[exp(lambda X)]`` stably."""
    effects = _clean_effects(single_effects)
    lam = np.asarray(list(lambdas), dtype=float)
    vals = []
    for l in lam:
        if effects.size == 0:
            vals.append(np.nan)
            continue
        z = np.clip(l * effects, -700, 700)
        m = float(np.max(z))
        vals.append(float(m + np.log(np.mean(np.exp(z - m)))))
    return pd.DataFrame({"lambda": lam, "cgf": vals})


def rate_function(
    single_effects: Iterable[float],
    s_grid: Iterable[float],
    lambda_grid: Iterable[float],
) -> pd.DataFrame:
    """Estimate Cramer rate function ``I(s)=sup_lambda lambda*s-K(lambda)``."""
    s = np.asarray(list(s_grid), dtype=float)
    lam = np.asarray(list(lambda_grid), dtype=float)
    cgf = cumulant_generating_function(single_effects, lam)["cgf"].to_numpy(dtype=float)
    rows = []
    for val in s:
        objective = lam * val - cgf
        finite = objective[np.isfinite(objective)]
        rows.append({"s": float(val), "rate": float(np.max(finite)) if finite.size else np.nan})
    return pd.DataFrame(rows)


def large_deviation_survival(
    single_effects: Iterable[float],
    depths: Iterable[int],
    threshold: float,
    lambda_grid: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Approximate additive-tail survival with a large-deviation rate."""
    effects = _clean_effects(single_effects)
    depths_arr = np.asarray(list(depths), dtype=int)
    if lambda_grid is None:
        lambda_grid = np.linspace(-80.0, 80.0, 1601)
    lam = np.asarray(list(lambda_grid), dtype=float)
    cgf = cumulant_generating_function(effects, lam)["cgf"].to_numpy(dtype=float)
    rows = []
    for d in depths_arr:
        d = int(d)
        if d <= 0:
            survival = 1.0 if threshold <= 0 else 0.0
            rate = 0.0
        elif effects.size == 0:
            survival = np.nan
            rate = np.nan
        else:
            s = threshold / d
            objective = lam * s - cgf
            finite = objective[np.isfinite(objective)]
            rate = max(0.0, float(np.max(finite))) if finite.size else np.nan
            survival = float(np.exp(-d * rate)) if np.isfinite(rate) else np.nan
            survival = float(np.clip(survival, 0.0, 1.0))
        rows.append(
            {
                "mutation_distance": d,
                "survival_large_deviation": survival,
                "rate": rate,
            }
        )
    return pd.DataFrame(rows)


def additive_survival_curve(
    df: pd.DataFrame,
    depths: Iterable[int] | None = None,
    threshold: float | None = None,
    fitness_col: str = "fitness_norm",
    n_mc: int = 100_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute additive MC, FFT, and large-deviation survival curves for a dataset."""
    if depths is None:
        max_depth = int(df["mutation_distance"].max()) if "mutation_distance" in df.columns and len(df) else 1
        depths = range(0, max_depth + 1)
    if threshold is None:
        if "viable" in df.columns and fitness_col in df.columns:
            viable = df[df["viable"] == 1]
            threshold_abs = float(viable[fitness_col].min()) if len(viable) else float(df[fitness_col].quantile(0.75))
        else:
            threshold_abs = float(df[fitness_col].quantile(0.75))
    else:
        threshold_abs = float(threshold)
    wt_fitness = compute_wt_fitness(df, fitness_col=fitness_col)
    effect_threshold = threshold_abs - wt_fitness
    effects_df = compute_single_mutant_effects(df, fitness_col=fitness_col, wt_fitness=wt_fitness)
    effects = effects_df["delta_f"].to_numpy(dtype=float) if len(effects_df) else np.array([])

    depths_list = list(depths)
    out = monte_carlo_additive_survival(
        effects,
        depths_list,
        threshold=effect_threshold,
        n_samples=n_mc,
        seed=seed,
    )
    out = out.merge(
        fft_additive_survival(effects, depths_list, threshold=effect_threshold),
        on="mutation_distance",
        how="outer",
    )
    out = out.merge(
        large_deviation_survival(effects, depths_list, threshold=effect_threshold),
        on="mutation_distance",
        how="outer",
    )
    if "dataset_id" in df.columns and len(df):
        out["dataset_id"] = df["dataset_id"].iloc[0]
    out["fitness_threshold"] = threshold_abs
    out["wt_fitness"] = wt_fitness
    out["effect_threshold"] = effect_threshold
    out["n_single_mutants"] = int(len(effects))
    return out.sort_values("mutation_distance").reset_index(drop=True)
