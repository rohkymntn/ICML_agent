"""Few-shot subsampling for the DMS function prior.

Real protein engineers do not have full DMS datasets for their target — they
have on the order of 50-500 hand-measured variants. This module simulates that
regime by subsampling the train-split DMS table to a target ``n_train`` and
returning a deterministic, reproducible subsample suitable for fitting
``DMSFunctionPrior``. Three subsampling strategies are supported:

- ``random``: uniform random sample over all train variants.
- ``stratified``: sample within (dataset_id, distance_bin) cells so the
  subsample preserves the per-dataset and per-mutation-distance composition
  of the full train pool.
- ``active``: initial random batch followed by greedy acquisition rounds
  using model uncertainty (Thompson-style) on the unlabeled pool.

The active-learning protocol intentionally mirrors the EVOLVEpro/ALDE family of
directed-evolution acquisition strategies: a small initial batch, then
batched UCB selection, with a fixed total label budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SubsampleStrategy = str  # "random" | "stratified" | "active"


@dataclass(frozen=True)
class FewShotConfig:
    """Configuration for one few-shot training subsample."""

    n_train: int
    strategy: SubsampleStrategy = "random"
    seed: int = 0
    n_distance_bins: int = 4
    # Active-learning specific:
    initial_frac: float = 0.2  # fraction of n_train allocated to the initial random batch
    n_rounds: int = 5
    acquisition: str = "ucb"  # "ucb" | "uncertainty" | "greedy"


def _distance_bins(distances: np.ndarray, n_bins: int) -> np.ndarray:
    """Return integer bin assignments using quantile cuts on distance.

    Falls back to a single bin when there is too little variance to bin.
    """
    distances = np.asarray(distances, dtype=float)
    if len(distances) == 0:
        return np.zeros(0, dtype=int)
    finite = distances[np.isfinite(distances)]
    if len(np.unique(finite)) <= 1:
        return np.zeros(len(distances), dtype=int)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(finite, qs))
    if len(edges) <= 1:
        return np.zeros(len(distances), dtype=int)
    bins = np.clip(np.digitize(distances, edges[1:-1]), 0, len(edges) - 2)
    return bins.astype(int)


def random_subsample(
    df: pd.DataFrame,
    n_train: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Uniform random sample of ``n_train`` rows.

    If ``n_train >= len(df)``, return the full frame (deduplicated and shuffled).
    """
    if n_train >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n=int(n_train), random_state=seed).reset_index(drop=True)


def stratified_subsample(
    df: pd.DataFrame,
    n_train: int,
    seed: int = 0,
    n_distance_bins: int = 4,
) -> pd.DataFrame:
    """Sample within (dataset_id, distance_bin) cells.

    Allocates ``n_train`` proportionally to cell size, with a per-cell minimum
    of 1 so every cell present in the train pool is represented when possible.
    """
    if n_train >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    work = df.copy()
    if "mutation_distance" not in work.columns:
        return random_subsample(work, n_train, seed)
    work["_dbin"] = _distance_bins(
        work["mutation_distance"].to_numpy(dtype=float), n_bins=n_distance_bins
    )
    if "dataset_id" not in work.columns:
        work["_ds"] = "_"
    else:
        work["_ds"] = work["dataset_id"].astype(str)
    cells = list(work.groupby(["_ds", "_dbin"]))
    sizes = np.array([len(c[1]) for c in cells], dtype=float)
    if sizes.sum() == 0:
        return random_subsample(df, n_train, seed)
    targets = np.maximum(1, np.floor(sizes / sizes.sum() * n_train).astype(int))
    # If we under-allocated (due to flooring), distribute remainder to largest cells.
    remaining = int(n_train) - int(targets.sum())
    if remaining > 0:
        order = np.argsort(-sizes)
        for j in order[:remaining]:
            targets[j] += 1
    elif remaining < 0:
        order = np.argsort(sizes)
        for j in order[: -remaining]:
            if targets[j] > 1:
                targets[j] -= 1
    rng = np.random.default_rng(seed)
    out_frames = []
    for (_, sub), k in zip(cells, targets):
        k = int(min(k, len(sub)))
        if k <= 0:
            continue
        idx = rng.choice(len(sub), size=k, replace=False)
        out_frames.append(sub.iloc[idx].drop(columns=["_dbin", "_ds"]))
    if not out_frames:
        return random_subsample(df, n_train, seed)
    out = pd.concat(out_frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def active_subsample(
    df: pd.DataFrame,
    n_train: int,
    seed: int = 0,
    initial_frac: float = 0.2,
    n_rounds: int = 5,
    acquisition: str = "ucb",
) -> pd.DataFrame:
    """Active acquisition: initial random batch + greedy rounds.

    Uses model uncertainty/UCB on the unlabeled pool. The acquisition model
    is a fresh ``DMSFunctionPrior`` retrained at each round on the labeled
    set so far. This intentionally simulates an experimentalist's batched
    measurement loop.
    """
    from .editguard_prior import DMSFunctionPrior

    if n_train >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_initial = max(20, int(round(n_train * float(initial_frac))))
    n_initial = min(n_initial, n_train)
    n_remaining = n_train - n_initial
    per_round = max(1, int(round(n_remaining / max(1, n_rounds))))

    pool = df.reset_index(drop=True).copy()
    init_idx = rng.choice(len(pool), size=n_initial, replace=False)
    labeled_mask = np.zeros(len(pool), dtype=bool)
    labeled_mask[init_idx] = True

    # Iterative acquisition.
    for _ in range(n_rounds):
        if labeled_mask.sum() >= n_train:
            break
        labeled = pool.iloc[labeled_mask]
        if labeled["viable"].nunique() < 2:
            # Cannot fit a meaningful classifier yet — fall back to random.
            unlabeled_idx = np.flatnonzero(~labeled_mask)
            take = rng.choice(unlabeled_idx, size=min(per_round, len(unlabeled_idx)), replace=False)
            labeled_mask[take] = True
            continue
        try:
            prior = DMSFunctionPrior(random_state=int(seed)).fit(labeled, dms_context=labeled)
        except Exception:
            unlabeled_idx = np.flatnonzero(~labeled_mask)
            take = rng.choice(unlabeled_idx, size=min(per_round, len(unlabeled_idx)), replace=False)
            labeled_mask[take] = True
            continue
        unlabeled_idx = np.flatnonzero(~labeled_mask)
        if len(unlabeled_idx) == 0:
            break
        unlabeled = pool.iloc[unlabeled_idx]
        if acquisition == "uncertainty":
            scores = prior.uncertainty(unlabeled)
        elif acquisition == "ucb":
            mean = prior.predict_proba(unlabeled)
            unc = prior.uncertainty(unlabeled)
            scores = mean + 1.0 * np.sqrt(unc)
        else:  # greedy: pick highest predicted probability of being viable
            scores = prior.predict_proba(unlabeled)
        k = min(per_round, len(unlabeled_idx), n_train - int(labeled_mask.sum()))
        if k <= 0:
            break
        top = unlabeled_idx[np.argsort(-scores)[:k]]
        labeled_mask[top] = True

    out = pool.iloc[labeled_mask].reset_index(drop=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def subsample(
    df: pd.DataFrame,
    config: FewShotConfig,
) -> pd.DataFrame:
    """Dispatch on ``config.strategy``."""
    if config.strategy == "random":
        return random_subsample(df, config.n_train, config.seed)
    if config.strategy == "stratified":
        return stratified_subsample(
            df, config.n_train, config.seed, config.n_distance_bins
        )
    if config.strategy == "active":
        return active_subsample(
            df,
            config.n_train,
            seed=config.seed,
            initial_frac=config.initial_frac,
            n_rounds=config.n_rounds,
            acquisition=config.acquisition,
        )
    raise ValueError(f"Unknown few-shot strategy: {config.strategy!r}")


def standard_few_shot_grid(seeds: Iterable[int] = (0, 1, 2, 3, 4)) -> list[FewShotConfig]:
    """The grid of (n_train, strategy, seed) used for the headline few-shot table."""
    grid: list[FewShotConfig] = []
    for n in (50, 100, 200, 500):
        for strat in ("random", "stratified", "active"):
            for s in seeds:
                grid.append(FewShotConfig(n_train=int(n), strategy=strat, seed=int(s)))
    return grid
