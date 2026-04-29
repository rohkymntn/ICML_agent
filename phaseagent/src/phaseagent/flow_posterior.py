"""Posterior interfaces for phase parameters and survival curves."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .phase import bootstrap_phase_boundary, compute_viability_by_distance, fit_phase_boundary


@dataclass
class GaussianPhasePosterior:
    """Simple Gaussian posterior over ``(dc, alpha)``."""

    mean: np.ndarray
    cov: np.ndarray

    def sample(self, n_samples: int = 1000, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cov = self.cov + 1e-8 * np.eye(self.cov.shape[0])
        samples = rng.multivariate_normal(self.mean, cov, size=n_samples)
        return pd.DataFrame(samples, columns=["dc", "alpha"])


def fit_gaussian_phase_posterior(
    df: pd.DataFrame,
    n_boot: int = 200,
    min_count_per_distance: int = 10,
    seed: int = 0,
) -> GaussianPhasePosterior:
    """Bootstrap empirical data and fit a Gaussian approximation to phase parameters."""
    boot = bootstrap_phase_boundary(
        df,
        n_boot=n_boot,
        min_count_per_distance=min_count_per_distance,
        random_state=seed,
    )
    vals = boot[["dc", "alpha"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 2:
        v = compute_viability_by_distance(df, min_count_per_distance=min_count_per_distance)
        fit = fit_phase_boundary(v)
        mean = np.array([
            fit["dc"] if np.isfinite(fit["dc"]) else 0.0,
            fit["alpha"] if np.isfinite(fit["alpha"]) else 1.0,
        ])
        cov = np.diag([1.0, 1.0])
    else:
        mean = vals.mean().to_numpy(dtype=float)
        cov = np.cov(vals.to_numpy(dtype=float), rowvar=False)
        if cov.ndim == 0:
            cov = np.diag([float(cov), float(cov)])
    return GaussianPhasePosterior(mean=mean, cov=np.asarray(cov, dtype=float))


def sample_phase_posterior(
    spectrum=None,
    structure=None,
    n_samples: int = 1000,
    observed_df: pd.DataFrame | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample phase parameters using the best available lightweight posterior.

    ``spectrum`` and ``structure`` are accepted to preserve the future flow API.
    """
    del spectrum, structure
    if observed_df is None or len(observed_df) == 0:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "dc": rng.normal(loc=2.0, scale=1.0, size=n_samples).clip(min=0.0),
                "alpha": rng.lognormal(mean=0.0, sigma=0.5, size=n_samples),
            }
        )
    posterior = fit_gaussian_phase_posterior(observed_df, n_boot=n_boot, seed=seed)
    samples = posterior.sample(n_samples=n_samples, seed=seed)
    samples["dc"] = samples["dc"].clip(lower=0.0)
    samples["alpha"] = samples["alpha"].clip(lower=0.01)
    return samples


def posterior_curve_samples(
    phase_samples: pd.DataFrame,
    depths,
) -> pd.DataFrame:
    """Convert sampled ``dc, alpha`` parameters into sampled sigmoid survival curves."""
    d = np.asarray(list(depths), dtype=float)
    rows = []
    for sample_id, row in phase_samples.reset_index(drop=True).iterrows():
        v = 1.0 / (1.0 + np.exp(row["alpha"] * (d - row["dc"])))
        rows.extend(
            {
                "sample_id": int(sample_id),
                "mutation_distance": int(depth),
                "survival_pred": float(value),
            }
            for depth, value in zip(d, v)
        )
    return pd.DataFrame(rows)
