"""Evaluation utilities for EditGuard editing experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .editing_tasks import EditingTask, constraint_satisfied


def evaluate_edit_selection(selected: pd.DataFrame, task: EditingTask) -> dict:
    """Evaluate selected edits against measured DMS labels."""
    if len(selected) == 0:
        return {
            "n_selected": 0,
            "functional_hit_rate": np.nan,
            "mean_fitness": np.nan,
            "best_fitness": np.nan,
            "mean_edit_distance": np.nan,
            "constraint_satisfaction_rate": np.nan,
            "joint_success_rate": np.nan,
        }
    ok = constraint_satisfied(selected, task)
    viable = selected["viable"].to_numpy(dtype=float) if "viable" in selected.columns else np.full(len(selected), np.nan)
    joint = ok & (viable == 1)
    return {
        "n_selected": int(len(selected)),
        "functional_hit_rate": float(np.nanmean(viable)),
        "mean_fitness": float(selected["fitness_norm"].mean()) if "fitness_norm" in selected.columns else np.nan,
        "best_fitness": float(selected["fitness_norm"].max()) if "fitness_norm" in selected.columns else np.nan,
        "mean_edit_distance": float(selected["mutation_distance"].mean()),
        "constraint_satisfaction_rate": float(np.mean(ok)),
        "joint_success_rate": float(np.mean(joint)),
    }


def evaluate_generated_selection(selected: pd.DataFrame, task: EditingTask) -> dict:
    """Evaluate generated edits, accounting for candidates without DMS labels."""
    base = {
        "n_selected": int(len(selected)),
        "n_labeled": 0,
        "labeled_fraction": 0.0,
        "functional_hit_rate_labeled": np.nan,
        "joint_success_rate_labeled": np.nan,
        "first_functional_rank": np.nan,
        "mean_edit_distance": np.nan,
        "mean_prior_function_prob": np.nan,
        "mean_plm_score": np.nan,
    }
    if len(selected) == 0:
        return base
    base["mean_edit_distance"] = float(selected["mutation_distance"].mean()) if "mutation_distance" in selected.columns else np.nan
    if "prior_function_prob" in selected.columns:
        base["mean_prior_function_prob"] = float(selected["prior_function_prob"].mean())
    if "plm_score" in selected.columns:
        base["mean_plm_score"] = float(selected["plm_score"].mean())
    labeled = selected[selected.get("has_dms_label", pd.Series(False, index=selected.index)).astype(bool)].copy()
    base["n_labeled"] = int(len(labeled))
    base["labeled_fraction"] = float(len(labeled) / len(selected))
    if len(labeled) == 0:
        return base
    metrics = evaluate_edit_selection(labeled, task)
    base["functional_hit_rate_labeled"] = metrics["functional_hit_rate"]
    base["joint_success_rate_labeled"] = metrics["joint_success_rate"]
    if "generated_rank" in labeled.columns:
        functional = labeled[labeled["viable"] == 1]
        if len(functional):
            base["first_functional_rank"] = int(functional["generated_rank"].min())
    return base


def first_success_rank(selected: pd.DataFrame, success_col: str = "viable") -> float:
    """Return generated rank of the first successful candidate, if any."""
    if len(selected) == 0 or success_col not in selected.columns:
        return np.nan
    functional = selected[selected[success_col] == 1]
    if len(functional) == 0:
        return np.nan
    if "generated_rank" in functional.columns:
        return float(functional["generated_rank"].min())
    return float(functional.index.min())


def evaluate_methods(selections: dict[str, pd.DataFrame], task: EditingTask) -> pd.DataFrame:
    rows = []
    for method, selected in selections.items():
        rows.append({"method": method, **evaluate_edit_selection(selected, task)})
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 1000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarize_with_ci(
    results: pd.DataFrame,
    group_cols=("method",),
    metric: str = "functional_hit_rate",
) -> pd.DataFrame:
    rows = []
    for key, sub in results.groupby(list(group_cols)):
        vals = sub[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                f"{metric}_mean": float(np.nanmean(vals)),
                f"{metric}_ci_low": lo,
                f"{metric}_ci_high": hi,
                "n": int(np.isfinite(vals).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_method_test(
    results: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "functional_hit_rate",
    pair_cols=("dataset_id", "objective", "edit_budget", "seed"),
) -> dict:
    """Paired test on matched task-level metrics."""
    a = results[results["method"] == method_a][list(pair_cols) + [metric]]
    b = results[results["method"] == method_b][list(pair_cols) + [metric]]
    merged = a.merge(b, on=list(pair_cols), suffixes=("_a", "_b"))
    if len(merged) == 0:
        return {"n_pairs": 0, "mean_delta": np.nan, "p_value": np.nan}
    delta = merged[f"{metric}_a"].to_numpy(dtype=float) - merged[f"{metric}_b"].to_numpy(dtype=float)
    try:
        from scipy.stats import wilcoxon

        p_value = float(wilcoxon(delta, zero_method="wilcox").pvalue) if np.any(delta != 0) else 1.0
    except Exception:
        p_value = np.nan
    return {"n_pairs": int(len(delta)), "mean_delta": float(np.nanmean(delta)), "p_value": p_value}
