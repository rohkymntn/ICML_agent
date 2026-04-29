"""Epistasis proxies and search-evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mutations import parse_mutation_notation


def compute_additive_expectation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mutation_notation" not in out.columns:
        out["fitness_additive_pred"] = np.nan
        out["epistasis_residual"] = np.nan
        return out

    out["fitness_additive_pred"] = np.nan
    for _, sub in out.groupby("dataset_id"):
        wt_rows = sub[sub["mutation_distance"] == 0]
        f_wt = float(wt_rows["fitness_norm"].mean()) if len(wt_rows) else 0.5
        singles = sub[sub["mutation_distance"] == 1]
        delta_by_token: dict[str, float] = {}
        for _, r in singles.iterrows():
            toks = parse_mutation_notation(r["mutation_notation"])
            if len(toks) == 1:
                delta_by_token[toks[0]] = float(r["fitness_norm"]) - f_wt
        preds = []
        for _, r in sub.iterrows():
            toks = parse_mutation_notation(r["mutation_notation"])
            if len(toks) == 0:
                preds.append(f_wt)
            elif all(t in delta_by_token for t in toks):
                preds.append(f_wt + sum(delta_by_token[t] for t in toks))
            else:
                preds.append(np.nan)
        out.loc[sub.index, "fitness_additive_pred"] = preds
    out["epistasis_residual"] = out["fitness_norm"] - out["fitness_additive_pred"]
    return out


def shell_epistasis_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if "epistasis_residual" not in df.columns:
        return pd.DataFrame(columns=["mutation_distance", "mean_abs_epistasis", "var_epistasis", "n"])
    sub = df.dropna(subset=["epistasis_residual"])
    if len(sub) == 0:
        return pd.DataFrame(columns=["mutation_distance", "mean_abs_epistasis", "var_epistasis", "n"])
    g = (
        sub.groupby("mutation_distance")["epistasis_residual"]
        .agg(
            mean_abs_epistasis=lambda x: float(np.mean(np.abs(x))),
            var_epistasis=lambda x: float(np.var(x)),
            n="size",
        )
        .reset_index()
    )
    return g


def shell_fitness_ruggedness_proxy(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("mutation_distance")["fitness_norm"]
        .agg(var_fitness="var", n="size")
        .reset_index()
    )
    g["ruggedness"] = g["var_fitness"]
    return g


def evaluate_search(selected_df: pd.DataFrame, dc_true: float) -> dict:
    if len(selected_df) == 0:
        return {
            "best_fitness": np.nan,
            "mean_fitness": np.nan,
            "functional_hit_rate": np.nan,
            "mean_mutation_distance": np.nan,
            "beyond_boundary_fraction": np.nan,
            "diversity_proxy": np.nan,
        }
    return {
        "best_fitness": float(selected_df["fitness_norm"].max()),
        "mean_fitness": float(selected_df["fitness_norm"].mean()),
        "functional_hit_rate": float(selected_df["viable"].mean()),
        "mean_mutation_distance": float(selected_df["mutation_distance"].mean()),
        "beyond_boundary_fraction": (
            float(np.mean(selected_df["mutation_distance"] > dc_true))
            if np.isfinite(dc_true)
            else np.nan
        ),
        "diversity_proxy": float(selected_df["mutation_distance"].std()),
    }
