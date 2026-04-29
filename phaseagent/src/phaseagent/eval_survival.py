"""Survival-curve evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .phase import compute_viability_by_distance, fit_phase_boundary


def true_survival_curve(
    df: pd.DataFrame,
    min_count_per_distance: int = 1,
) -> pd.DataFrame:
    """Compute empirical ``V(d)`` from canonical viable labels."""
    v = compute_viability_by_distance(df, min_count_per_distance=min_count_per_distance)
    return v.rename(columns={"viability_density": "survival_true"})


def align_survival_curves(
    pred: pd.DataFrame,
    true: pd.DataFrame,
    pred_col: str = "survival_pred",
    true_col: str = "survival_true",
) -> pd.DataFrame:
    """Align predicted and true curves by mutation depth."""
    merged = true[["mutation_distance", true_col]].merge(
        pred[["mutation_distance", pred_col]],
        on="mutation_distance",
        how="inner",
    )
    return merged.sort_values("mutation_distance").reset_index(drop=True)


def area_under_survival(curve: pd.DataFrame, survival_col: str) -> float:
    """Trapezoidal area under a survival curve over mutation depth."""
    if len(curve) == 0:
        return np.nan
    d = curve["mutation_distance"].to_numpy(dtype=float)
    v = curve[survival_col].to_numpy(dtype=float)
    mask = np.isfinite(d) & np.isfinite(v)
    if mask.sum() == 0:
        return np.nan
    if mask.sum() == 1:
        return float(v[mask][0])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(v[mask], d[mask]))


def susceptibility_peak(curve: pd.DataFrame, survival_col: str) -> float:
    """Return depth with maximal absolute finite-difference slope."""
    if len(curve) < 2:
        return np.nan
    d = curve["mutation_distance"].to_numpy(dtype=float)
    v = curve[survival_col].to_numpy(dtype=float)
    mask = np.isfinite(d) & np.isfinite(v)
    if mask.sum() < 2:
        return np.nan
    chi = np.abs(np.gradient(v[mask], d[mask]))
    return float(d[mask][int(np.argmax(chi))])


def boundary_weighted_mse(
    aligned: pd.DataFrame,
    pred_col: str = "survival_pred",
    true_col: str = "survival_true",
    weight_scale: float = 5.0,
) -> float:
    """MSE that upweights depths near high empirical susceptibility."""
    if len(aligned) == 0:
        return np.nan
    d = aligned["mutation_distance"].to_numpy(dtype=float)
    true = aligned[true_col].to_numpy(dtype=float)
    pred = aligned[pred_col].to_numpy(dtype=float)
    if len(aligned) > 1:
        chi = np.abs(np.gradient(true, d))
        denom = float(np.nanmax(chi)) if np.isfinite(chi).any() else 0.0
        weights = 1.0 + weight_scale * (chi / denom) if denom > 0 else np.ones_like(true)
    else:
        weights = np.ones_like(true)
    return float(np.average((pred - true) ** 2, weights=weights))


def evaluate_survival_prediction(
    pred: pd.DataFrame,
    true: pd.DataFrame,
    pred_col: str = "survival_pred",
    true_col: str = "survival_true",
) -> dict[str, float]:
    """Compute curve, AUC, and phase-boundary errors."""
    aligned = align_survival_curves(pred, true, pred_col=pred_col, true_col=true_col)
    if len(aligned) == 0:
        return {
            "curve_mse": np.nan,
            "curve_mae": np.nan,
            "auc_survival_error": np.nan,
            "boundary_weighted_mse": np.nan,
            "dc_error": np.nan,
            "alpha_error": np.nan,
            "susceptibility_peak_error": np.nan,
        }
    err = aligned[pred_col].to_numpy(dtype=float) - aligned[true_col].to_numpy(dtype=float)

    pred_for_fit = pred.rename(columns={pred_col: "viability_density"}).copy()
    true_for_fit = true.rename(columns={true_col: "viability_density"}).copy()
    pred_for_fit["sufficient"] = True
    true_for_fit["sufficient"] = True
    pred_fit = fit_phase_boundary(pred_for_fit)
    true_fit = fit_phase_boundary(true_for_fit)

    pred_auc = area_under_survival(pred, pred_col)
    true_auc = area_under_survival(true, true_col)
    return {
        "curve_mse": float(np.mean(err**2)),
        "curve_mae": float(np.mean(np.abs(err))),
        "auc_survival_error": float(abs(pred_auc - true_auc)) if np.isfinite(pred_auc) and np.isfinite(true_auc) else np.nan,
        "boundary_weighted_mse": boundary_weighted_mse(aligned, pred_col=pred_col, true_col=true_col),
        "dc_error": (
            float(abs(pred_fit["dc"] - true_fit["dc"]))
            if np.isfinite(pred_fit["dc"]) and np.isfinite(true_fit["dc"])
            else np.nan
        ),
        "alpha_error": (
            float(abs(pred_fit["alpha"] - true_fit["alpha"]))
            if np.isfinite(pred_fit["alpha"]) and np.isfinite(true_fit["alpha"])
            else np.nan
        ),
        "susceptibility_peak_error": abs(
            susceptibility_peak(pred, pred_col) - susceptibility_peak(true, true_col)
        ),
    }


def evaluate_many_survival_predictions(
    predictions: dict[str, pd.DataFrame],
    datasets: dict[str, pd.DataFrame],
    pred_col: str = "survival_pred",
) -> pd.DataFrame:
    """Evaluate per-dataset predictions against empirical survival curves."""
    rows = []
    for dataset_id, pred in predictions.items():
        if dataset_id not in datasets:
            continue
        true = true_survival_curve(datasets[dataset_id])
        metrics = evaluate_survival_prediction(pred, true, pred_col=pred_col)
        rows.append({"dataset_id": dataset_id, **metrics})
    return pd.DataFrame(rows)
