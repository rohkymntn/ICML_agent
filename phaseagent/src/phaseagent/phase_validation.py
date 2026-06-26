"""Theorem 1 (Spectrum->Phase) empirical validation.

Given single-mutant DMS measurements with raw fitness/ddG values, we predict
the multi-mutant phase boundary parameters (d_c, alpha) closed-form from the
first two moments of the single-mutant effect distribution, then compare to
observed multi-mutant survival.

Two prediction levers:
  * Berry-Esseen (CLT) bulk: phase params from (mean, std) only.
  * Cramer / large-deviation tail: rate function from full empirical CGF.

Two empirical validations:
  * Megascale (singles + doubles only): predict V(2) from singles, compare to
    observed V(2) on doubles. No multi-distance sigmoid fit possible.
  * ProteinGym multi-mutant assays (d in {1, 2, 3, ...}): fit V(d) sigmoid on
    observed data; predict (d_c, alpha) from singles spectrum; compare.

Conventions
-----------
We work in raw effect units (negative for damaging) consistent with the
theorem statement. For Megascale we use ddG_ML directly with sign convention
"X = -ddG_ML" so positive X means damaging (sign-flip to match the theorem,
where the sum of damage S_d = sum X_i and viable iff WT_fitness - S_d >=
threshold_fitness, i.e., S_d <= Delta_g).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm


@dataclass(frozen=True)
class SpectrumStats:
    """First three moments of the single-mutant effect distribution."""

    n: int
    mean_X: float          # E[X], expected damage per random mutation (positive => damaging)
    std_X: float           # std[X]
    third_abs_moment: float  # E[|X|^3] for Berry-Esseen rate
    median_X: float
    quantiles: tuple[float, float, float]  # 25th, 50th, 75th of effects

    @property
    def m_g(self) -> float:
        """The 'mean damage' parameter from Theorem 1; positive = average damaging."""
        return float(self.mean_X)

    @property
    def sigma_g(self) -> float:
        return float(self.std_X)

    @property
    def hardness(self) -> float:
        """H_g = sigma_g / m_g, the per-protein hardness from Sec 8."""
        if self.m_g <= 0:
            return float("nan")
        return self.std_X / self.mean_X


def compute_spectrum_stats(effects: np.ndarray) -> SpectrumStats:
    """Compute the moments of an effect array.

    Parameters
    ----------
    effects
        1-D array of single-mutant *damage* values (positive = damaging in our
        sign convention). For Megascale ddG_ML > 0 = stabilizing, so callers
        should pass ``-ddG_ML``.
    """
    e = np.asarray(effects, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 5:
        return SpectrumStats(
            n=int(len(e)),
            mean_X=float("nan"),
            std_X=float("nan"),
            third_abs_moment=float("nan"),
            median_X=float("nan"),
            quantiles=(float("nan"),) * 3,
        )
    mean_X = float(np.mean(e))
    std_X = float(np.std(e, ddof=1))
    third = float(np.mean(np.abs(e - mean_X) ** 3))
    median_X = float(np.median(e))
    q = tuple(float(v) for v in np.quantile(e, [0.25, 0.50, 0.75]))
    return SpectrumStats(
        n=int(len(e)),
        mean_X=mean_X,
        std_X=std_X,
        third_abs_moment=third,
        median_X=median_X,
        quantiles=q,
    )


def predict_dc_alpha_from_spectrum(
    stats: SpectrumStats,
    delta_g: float,
) -> dict:
    """Closed-form Berry-Esseen prediction of (d_c, alpha) from spectrum moments.

    Parameters
    ----------
    stats
        Single-mutant spectrum stats.
    delta_g
        Viability gap = WT_fitness - viability_threshold > 0. In raw
        ddG units, e.g., 1.0 if "viable iff total destabilization < 1 kcal/mol".

    Returns
    -------
    dict with predicted ``dc_pred``, ``alpha_pred`` and standard
    Berry-Esseen rate ``be_rate``.
    """
    if not np.isfinite(stats.m_g) or stats.m_g <= 0:
        return {
            "dc_pred": float("nan"),
            "alpha_pred": float("nan"),
            "be_rate": float("nan"),
        }
    dc_pred = float(delta_g) / stats.m_g
    if dc_pred <= 0:
        return {
            "dc_pred": float("nan"),
            "alpha_pred": float("nan"),
            "be_rate": float("nan"),
        }
    # alpha = 4 m / (sigma sqrt(2 pi d_c))
    if stats.sigma_g <= 0:
        return {
            "dc_pred": float(dc_pred),
            "alpha_pred": float("inf"),
            "be_rate": 0.0,
        }
    alpha_pred = 4.0 * stats.m_g / (stats.sigma_g * np.sqrt(2.0 * np.pi * dc_pred))
    be_rate = (
        stats.third_abs_moment / (stats.sigma_g**3 * np.sqrt(dc_pred))
        if stats.sigma_g > 0
        else float("nan")
    )
    return {
        "dc_pred": float(dc_pred),
        "alpha_pred": float(alpha_pred),
        "be_rate": float(be_rate),
    }


def predict_V_at_distance(
    stats: SpectrumStats,
    d: int,
    delta_g: float,
) -> float:
    """Predict V(d) = P[viable | d random mutations] via Berry-Esseen Gaussian.

    V(d) = Phi((delta_g - d * m_g) / (sigma_g * sqrt(d))).

    For d >> d_c this gives small probability; for d << d_c probability ~ 1.
    """
    if d <= 0:
        return 1.0
    if not np.isfinite(stats.m_g) or stats.sigma_g <= 0:
        return float("nan")
    z = (float(delta_g) - d * stats.m_g) / (stats.sigma_g * np.sqrt(d))
    return float(norm.cdf(z))


def predict_V_at_distance_mc(
    effects: np.ndarray,
    d: int,
    delta_g: float,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Monte-Carlo simulate V(d) directly from the empirical effect distribution.

    For finite-d / non-Gaussian spectra this is more accurate than the
    Berry-Esseen Gaussian approximation. For d=2 (Megascale doubles) we use
    this as the LDP-tail-aware predictor.
    """
    if d <= 0:
        return 1.0
    e = np.asarray(effects, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(e, size=(int(n_samples), int(d)), replace=True)
    sums = draws.sum(axis=1)
    return float(np.mean(sums <= float(delta_g)))


def fit_observed_phase_boundary(
    df: pd.DataFrame,
    distance_col: str = "mutation_distance",
    viable_col: str = "viable",
    min_count_per_distance: int = 30,
) -> dict:
    """Fit V(d) = sigmoid(alpha * (d_c - d)) to observed data."""
    g = (
        df.groupby(distance_col)[viable_col]
        .agg(["sum", "size"])
        .reset_index()
        .rename(columns={"sum": "n_viable", "size": "n_total"})
    )
    g["v"] = g["n_viable"].astype(float) / g["n_total"].clip(lower=1)
    g = g[g["n_total"] >= int(min_count_per_distance)].copy()
    if len(g) < 3:
        return {
            "dc_obs": float("nan"),
            "alpha_obs": float("nan"),
            "fit_success": False,
            "r2": float("nan"),
            "n_distances": int(len(g)),
            "max_distance": int(g[distance_col].max()) if len(g) else 0,
        }
    d = g[distance_col].to_numpy(dtype=float)
    v = g["v"].to_numpy(dtype=float)
    d_min, d_max = float(d.min()), float(d.max())
    p0 = [float(np.median(d)), 1.0]
    bounds = ([max(d_min, 1e-3), 1e-3], [d_max + 5, 20.0])

    def _sig(d_, dc, alpha):
        return 1.0 / (1.0 + np.exp(alpha * (d_ - dc)))

    try:
        popt, _ = curve_fit(_sig, d, v, p0=p0, bounds=bounds, maxfev=20000)
        dc, alpha = float(popt[0]), float(popt[1])
        v_pred = _sig(d, dc, alpha)
        ss_res = float(np.sum((v - v_pred) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "dc_obs": dc,
            "alpha_obs": alpha,
            "fit_success": True,
            "r2": float(r2),
            "n_distances": int(len(g)),
            "max_distance": int(g[distance_col].max()),
        }
    except Exception as exc:
        return {
            "dc_obs": float("nan"),
            "alpha_obs": float("nan"),
            "fit_success": False,
            "r2": float("nan"),
            "n_distances": int(len(g)),
            "max_distance": int(g[distance_col].max()) if len(g) else 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================================
# Per-protein validation drivers
# ============================================================================

def validate_megascale_protein(
    singles_df: pd.DataFrame,
    doubles_df: pd.DataFrame,
    raw_label_col: str = "ddG_ML",
    delta_g_kcal: float = 1.0,
    n_mc_samples: int = 200_000,
    seed: int = 0,
) -> dict:
    """Validate Theorem 1 for one Megascale protein: predict V(2) from singles.

    Sign convention: damage = -ddG_ML (so a damaging mutation has ddG_ML < 0
    in the Megascale convention; passing -ddG_ML makes mean(damage) > 0 for
    most proteins). We predict the probability that the *total damage* over
    d random mutations stays below the viability gap delta_g_kcal.

    Returns one row of metrics for this protein.
    """
    if raw_label_col not in singles_df.columns:
        return {"protein": None, "error": f"missing {raw_label_col} in singles"}

    eff_singles = -pd.to_numeric(singles_df[raw_label_col], errors="coerce").to_numpy()
    eff_singles = eff_singles[np.isfinite(eff_singles)]
    if len(eff_singles) < 30:
        return {"protein": None, "n_singles": int(len(eff_singles)), "error": "too few singles"}

    stats = compute_spectrum_stats(eff_singles)
    pred = predict_dc_alpha_from_spectrum(stats, delta_g=delta_g_kcal)

    # V(d) predictions at d=1 and d=2
    V1_be = predict_V_at_distance(stats, d=1, delta_g=delta_g_kcal)
    V2_be = predict_V_at_distance(stats, d=2, delta_g=delta_g_kcal)
    V1_mc = predict_V_at_distance_mc(eff_singles, d=1, delta_g=delta_g_kcal, n_samples=n_mc_samples, seed=seed)
    V2_mc = predict_V_at_distance_mc(eff_singles, d=2, delta_g=delta_g_kcal, n_samples=n_mc_samples, seed=seed)

    # V(d) observed
    eff_singles_for_V1 = eff_singles  # damage values
    V1_obs = float(np.mean(eff_singles_for_V1 <= delta_g_kcal))

    if doubles_df is not None and len(doubles_df) > 0 and raw_label_col in doubles_df.columns:
        eff_doubles = -pd.to_numeric(doubles_df[raw_label_col], errors="coerce").to_numpy()
        eff_doubles = eff_doubles[np.isfinite(eff_doubles)]
        n_doubles = int(len(eff_doubles))
        V2_obs = float(np.mean(eff_doubles <= delta_g_kcal)) if n_doubles else float("nan")
    else:
        n_doubles = 0
        V2_obs = float("nan")

    return {
        "n_singles": int(stats.n),
        "n_doubles": int(n_doubles),
        "m_g": float(stats.m_g),
        "sigma_g": float(stats.sigma_g),
        "third_abs_moment": float(stats.third_abs_moment),
        "hardness_H": float(stats.hardness),
        "delta_g": float(delta_g_kcal),
        "dc_pred": float(pred["dc_pred"]),
        "alpha_pred": float(pred["alpha_pred"]),
        "be_rate": float(pred["be_rate"]),
        "V1_pred_BE": float(V1_be),
        "V2_pred_BE": float(V2_be),
        "V1_pred_MC": float(V1_mc),
        "V2_pred_MC": float(V2_mc),
        "V1_obs": float(V1_obs),
        "V2_obs": float(V2_obs),
        "abs_err_V1_BE": float(abs(V1_be - V1_obs)) if np.isfinite(V1_obs) and np.isfinite(V1_be) else float("nan"),
        "abs_err_V1_MC": float(abs(V1_mc - V1_obs)) if np.isfinite(V1_obs) and np.isfinite(V1_mc) else float("nan"),
        "abs_err_V2_BE": float(abs(V2_be - V2_obs)) if np.isfinite(V2_obs) and np.isfinite(V2_be) else float("nan"),
        "abs_err_V2_MC": float(abs(V2_mc - V2_obs)) if np.isfinite(V2_obs) and np.isfinite(V2_mc) else float("nan"),
    }


def validate_proteingym_assay(
    df: pd.DataFrame,
    fitness_col: str = "DMS_score",
    distance_col: str = "mutation_distance",
    quantile_threshold: float = 0.50,
    delta_quantile_for_threshold: float = 0.50,
    n_mc_samples: int = 200_000,
    seed: int = 0,
) -> dict:
    """Validate Theorem 1 for one ProteinGym multi-mutant assay.

    Strategy: fit V(d) sigmoid on observed data (multi-distance!) -> get
    (dc_obs, alpha_obs). Then from d=1 subset compute spectrum -> predict
    (dc_pred, alpha_pred). Compare.

    Sign convention: damage = WT_DMS - DMS_score (so damaging mutations have
    positive damage, mean(damage) > 0).

    Parameters
    ----------
    df
        Per-assay DataFrame with mutation_distance, DMS_score (or fitness_col),
        and a per-row viability label that we'll define here from the
        empirical fitness distribution.
    quantile_threshold
        Per-assay viability threshold, defined as the per-assay fitness
        quantile (default 0.5 = median).
    """
    # Define viable per row.
    if fitness_col not in df.columns:
        return {"error": f"missing {fitness_col}"}
    if distance_col not in df.columns:
        return {"error": f"missing {distance_col}"}
    f = pd.to_numeric(df[fitness_col], errors="coerce")
    valid = f.notna()
    work = df.loc[valid].copy()
    work[fitness_col] = f.loc[valid].astype(float)
    if len(work) == 0:
        return {"error": "no valid rows"}

    # Derive WT fitness empirically (best per-assay; use top-percentile as proxy).
    wt_fitness = float(work[fitness_col].quantile(0.95))
    threshold = float(work[fitness_col].quantile(quantile_threshold))
    work["damage"] = wt_fitness - work[fitness_col]
    work["viable"] = (work[fitness_col] >= threshold).astype(int)
    delta_g = wt_fitness - threshold  # always >= 0 since WT > median.

    # Per-distance counts.
    dist_counts = (
        work.groupby(distance_col)
        .size()
        .reset_index(name="n")
        .sort_values(distance_col)
    )

    # Spectrum from d=1 only.
    singles = work[work[distance_col] == 1]
    if len(singles) < 30:
        return {"error": f"too few singles ({len(singles)})"}
    eff_singles = singles["damage"].to_numpy(dtype=float)
    stats = compute_spectrum_stats(eff_singles)
    pred = predict_dc_alpha_from_spectrum(stats, delta_g=delta_g)

    # Observed sigmoid fit on the multi-distance V(d) curve.
    obs = fit_observed_phase_boundary(work, distance_col=distance_col, viable_col="viable")

    # Predicted V(d) curve at integer distances 1..max_d.
    max_d = int(work[distance_col].max())
    V_pred_curve = []
    V_obs_curve = []
    for d in range(1, max_d + 1):
        sub = work[work[distance_col] == d]
        n_d = len(sub)
        if n_d >= 10:
            V_obs = float(sub["viable"].mean())
        else:
            V_obs = float("nan")
        V_BE = predict_V_at_distance(stats, d=d, delta_g=delta_g)
        V_MC = (
            predict_V_at_distance_mc(eff_singles, d=d, delta_g=delta_g, n_samples=n_mc_samples, seed=seed)
            if d > 0 else 1.0
        )
        V_pred_curve.append({
            "mutation_distance": int(d),
            "V_obs": float(V_obs),
            "V_pred_BE": float(V_BE),
            "V_pred_MC": float(V_MC),
            "n": int(n_d),
        })
    curve_df = pd.DataFrame(V_pred_curve)

    return {
        "n_singles": int(stats.n),
        "n_total": int(len(work)),
        "max_distance": int(max_d),
        "wt_fitness": float(wt_fitness),
        "threshold": float(threshold),
        "delta_g": float(delta_g),
        "m_g": float(stats.m_g),
        "sigma_g": float(stats.sigma_g),
        "hardness_H": float(stats.hardness),
        "dc_pred": float(pred["dc_pred"]),
        "alpha_pred": float(pred["alpha_pred"]),
        "dc_obs": float(obs["dc_obs"]),
        "alpha_obs": float(obs["alpha_obs"]),
        "obs_fit_r2": float(obs["r2"]),
        "obs_fit_success": bool(obs["fit_success"]),
        "abs_err_dc": (
            float(abs(pred["dc_pred"] - obs["dc_obs"]))
            if np.isfinite(pred["dc_pred"]) and np.isfinite(obs["dc_obs"])
            else float("nan")
        ),
        "rel_err_alpha": (
            float(abs(pred["alpha_pred"] - obs["alpha_obs"]) / obs["alpha_obs"])
            if np.isfinite(pred["alpha_pred"]) and np.isfinite(obs["alpha_obs"]) and obs["alpha_obs"] > 0
            else float("nan")
        ),
        "_curve": curve_df,
    }
