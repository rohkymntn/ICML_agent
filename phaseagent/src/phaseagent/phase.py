"""Phase-boundary estimation: V(d), sigmoid fit, susceptibility, bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def sigmoid(d, dc, alpha):
    return 1.0 / (1.0 + np.exp(alpha * (d - dc)))


def compute_viability_by_distance(
    df: pd.DataFrame,
    min_count_per_distance: int = 10,
) -> pd.DataFrame:
    g = (
        df.groupby("mutation_distance")
        .agg(n=("viable", "size"), viable_count=("viable", "sum"))
        .reset_index()
    )
    g["viability_density"] = g["viable_count"] / g["n"]
    p = g["viability_density"]
    g["stderr"] = np.sqrt(p * (1.0 - p) / g["n"].clip(lower=1))
    g["sufficient"] = g["n"] >= min_count_per_distance
    return g.sort_values("mutation_distance").reset_index(drop=True)


def estimate_susceptibility(v_by_d: pd.DataFrame) -> pd.DataFrame:
    out = v_by_d.copy().sort_values("mutation_distance").reset_index(drop=True)
    if len(out) < 2:
        out["susceptibility"] = np.nan
        return out
    v = out["viability_density"].to_numpy(dtype=float)
    d = out["mutation_distance"].to_numpy(dtype=float)
    out["susceptibility"] = np.abs(np.gradient(v, d))
    return out


def fit_phase_boundary(v_by_d: pd.DataFrame) -> dict:
    sufficient = v_by_d[v_by_d["sufficient"]] if "sufficient" in v_by_d.columns else v_by_d
    if len(sufficient) < 3:
        return {
            "dc": float("nan"),
            "alpha": float("nan"),
            "fit_success": False,
            "r2": float("nan"),
            "n_points": int(len(sufficient)),
            "fallback": "insufficient_data",
        }
    d = sufficient["mutation_distance"].to_numpy(dtype=float)
    v = sufficient["viability_density"].to_numpy(dtype=float)
    d_min, d_max = float(d.min()), float(d.max())
    if d_min == d_max:
        return {
            "dc": float("nan"),
            "alpha": float("nan"),
            "fit_success": False,
            "r2": float("nan"),
            "n_points": int(len(sufficient)),
            "fallback": "single_distance",
        }
    p0 = [float(np.median(d)), 1.0]
    bounds = ([d_min, 0.01], [d_max, 10.0])
    try:
        popt, pcov = curve_fit(sigmoid, d, v, p0=p0, bounds=bounds, maxfev=20000)
        dc, alpha = float(popt[0]), float(popt[1])
        v_pred = sigmoid(d, dc, alpha)
        ss_res = float(np.sum((v - v_pred) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if np.all(np.isfinite(pcov)):
            stderr = np.sqrt(np.diag(pcov))
        else:
            stderr = [float("nan"), float("nan")]
        return {
            "dc": dc,
            "alpha": alpha,
            "fit_success": True,
            "r2": float(r2),
            "dc_stderr": float(stderr[0]),
            "alpha_stderr": float(stderr[1]),
            "n_points": int(len(sufficient)),
        }
    except Exception as exc:
        chi = estimate_susceptibility(v_by_d)
        if chi["susceptibility"].notna().any():
            argmax_idx = chi["susceptibility"].idxmax()
            dc_fallback = float(chi.loc[argmax_idx, "mutation_distance"])
        else:
            dc_fallback = float("nan")
        return {
            "dc": dc_fallback,
            "alpha": float("nan"),
            "fit_success": False,
            "r2": float("nan"),
            "n_points": int(len(sufficient)),
            "fallback": f"curve_fit_failed:{type(exc).__name__}",
        }


def bootstrap_phase_boundary(
    df: pd.DataFrame,
    n_boot: int = 200,
    min_count_per_distance: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = len(df)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = df.iloc[idx]
        v = compute_viability_by_distance(boot, min_count_per_distance=min_count_per_distance)
        fit = fit_phase_boundary(v)
        rows.append({"bootstrap": b, **fit})
    return pd.DataFrame(rows)


def classify_phase_regime(dc, alpha, r2, v_by_d) -> str:
    if not np.isfinite(dc) or not np.isfinite(alpha):
        return "weak_signal"
    if r2 is None or not np.isfinite(r2) or r2 < 0.5:
        if v_by_d["viability_density"].std() < 0.05:
            return "weak_signal"
        return "noisy_or_multistep"
    if alpha >= 1.5:
        return "sharp_collapse"
    if alpha < 0.5:
        return "smooth_decay"
    d_max = v_by_d["mutation_distance"].max()
    if dc >= 0.6 * d_max:
        return "delayed_collapse"
    return "smooth_decay"
