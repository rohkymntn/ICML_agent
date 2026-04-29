"""Phase-aware search simulator."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .metrics import evaluate_search

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    _HAS_SK = True
except Exception:
    _HAS_SK = False


def _featurize(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    feats = np.zeros((n, 4))
    feats[:, 0] = df["mutation_distance"].to_numpy()
    feats[:, 1] = df["mutation_distance"].to_numpy() ** 2
    if "mutation_notation" in df.columns:
        nota = df["mutation_notation"].astype(str)
        feats[:, 2] = nota.str.len().to_numpy()
        feats[:, 3] = nota.apply(lambda s: sum(ord(c) for c in s) % 1000 / 1000.0).to_numpy()
    return feats


def _fit_surrogate(train_df: pd.DataFrame):
    X = _featurize(train_df)
    y = train_df["fitness_norm"].to_numpy()
    if _HAS_SK and len(train_df) >= 30:
        model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0)
        model.fit(X, y)
        return model
    if _HAS_SK:
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        return model
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)

    class _Lin:
        def predict(self, Z):
            Zb = np.hstack([Z, np.ones((Z.shape[0], 1))])
            return Zb @ coef

    return _Lin()


def random_search(pool_df: pd.DataFrame, budget: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(budget, len(pool_df))
    if n <= 0:
        return pool_df.iloc[:0].copy()
    idx = rng.choice(pool_df.index.to_numpy(), size=n, replace=False)
    return pool_df.loc[idx].copy()


def novelty_search(
    pool_df: pd.DataFrame,
    budget: int,
    min_distance: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub = pool_df if min_distance is None else pool_df[pool_df["mutation_distance"] >= min_distance]
    if len(sub) == 0:
        sub = pool_df
    sorted_pool = sub.sort_values("mutation_distance", ascending=False)
    top = sorted_pool.head(min(budget * 4, len(sorted_pool)))
    n = min(budget, len(top))
    if n <= 0:
        return pool_df.iloc[:0].copy()
    idx = rng.choice(top.index.to_numpy(), size=n, replace=False)
    return pool_df.loc[idx].copy()


def fitness_proxy_search(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    budget: int,
    seed: int = 0,
) -> pd.DataFrame:
    if len(pool_df) == 0:
        return pool_df.iloc[:0].copy()
    model = _fit_surrogate(train_df)
    pred = model.predict(_featurize(pool_df))
    out = pool_df.copy()
    out["pred_score"] = pred
    return out.nlargest(min(budget, len(out)), "pred_score")


def phase_aware_search(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    dc_hat: float,
    budget: int,
    lambda_boundary: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    if len(pool_df) == 0:
        return pool_df.iloc[:0].copy()
    model = _fit_surrogate(train_df)
    pred = model.predict(_featurize(pool_df))
    out = pool_df.copy()
    out["pred_score"] = pred
    if np.isfinite(dc_hat):
        beyond = np.maximum(0.0, out["mutation_distance"].to_numpy() - dc_hat)
        out["adjusted_score"] = out["pred_score"] - lambda_boundary * beyond
    else:
        out["adjusted_score"] = out["pred_score"]
    return out.nlargest(min(budget, len(out)), "adjusted_score")


def run_search_benchmark(
    datasets: dict[str, pd.DataFrame],
    boundaries: dict[str, dict],
    budgets: Iterable[int] = (50,),
    seeds: Iterable[int] = (0, 1, 2),
    train_fraction: float = 0.5,
    lambda_boundary: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for ds_id, df in datasets.items():
        df = df.reset_index(drop=True)
        bd = boundaries.get(ds_id, {})
        dc_true = bd.get("dc_true", bd.get("dc", float("nan")))
        dc_hat = bd.get("dc", float("nan"))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(df))
            n_train = int(train_fraction * len(df))
            train = df.iloc[idx[:n_train]]
            pool = df.iloc[idx[n_train:]]
            for budget in budgets:
                policies = {
                    "random": random_search(pool, budget, seed),
                    "novelty": novelty_search(pool, budget, seed=seed),
                    "fitness_proxy": fitness_proxy_search(train, pool, budget, seed=seed),
                    "phase_aware": phase_aware_search(
                        train, pool, dc_hat=dc_hat, budget=budget, lambda_boundary=lambda_boundary, seed=seed
                    ),
                }
                for name, sel in policies.items():
                    metrics = evaluate_search(sel, dc_true=dc_true)
                    rows.append({
                        "dataset_id": ds_id,
                        "policy": name,
                        "budget": budget,
                        "seed": seed,
                        **metrics,
                    })
    return pd.DataFrame(rows)
