"""Calibration metrics for the DMS function prior.

Implements claim C3 from the implementation plan: a well-calibrated prior
on held-out test proteins. Reports:

- ECE (Expected Calibration Error) with 15 equal-frequency bins
- Brier score
- Log loss (clipped at 1e-6 to avoid -inf on rare zero-probability rows)
- A pooled-and-per-dataset reliability diagram

ECE here is the standard "binned absolute miscalibration" definition:
   ECE = sum_b (n_b / N) * |empirical_rate_b - mean_predicted_b|
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe(arr: Iterable, dtype=float) -> np.ndarray:
    a = np.asarray(list(arr), dtype=dtype)
    return a


def expected_calibration_error(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
    bin_strategy: str = "quantile",
) -> tuple[float, pd.DataFrame]:
    """Return ``(ece, bins)`` where ``bins`` has one row per bin.

    ``bin_strategy="quantile"`` uses equal-frequency bins (more reliable when
    predictions are concentrated in a narrow range, common with calibrated
    classifiers). ``bin_strategy="uniform"`` gives the textbook fixed-edge
    version, kept available for diagnostics.
    """
    p = np.clip(_safe(p), 0.0, 1.0)
    y = _safe(y, dtype=int)
    if p.size == 0:
        return float("nan"), pd.DataFrame()
    if bin_strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # Equal-frequency: use quantiles; deduplicate edges in case of ties.
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    edges[0] = 0.0
    edges[-1] = 1.0
    rows = []
    n = float(len(p))
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        emp = float(y[mask].mean())
        pred = float(p[mask].mean())
        gap = abs(emp - pred)
        weighted = (n_bin / n) * gap
        ece += weighted
        rows.append(
            {
                "bin_idx": i,
                "lo": float(lo),
                "hi": float(hi),
                "n": n_bin,
                "fraction": n_bin / n,
                "mean_predicted": pred,
                "empirical_rate": emp,
                "gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(_safe(p), 0.0, 1.0)
    y = _safe(y, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(_safe(p), eps, 1.0 - eps)
    y = _safe(y, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate_calibration(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
) -> tuple[dict, pd.DataFrame]:
    """Return ``(scalar_metrics, bin_table)`` for one prediction set."""
    ece, bins = expected_calibration_error(p, y, n_bins=n_bins, bin_strategy="quantile")
    metrics = {
        "n": int(len(p)),
        "ece": ece,
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "mean_predicted": float(np.mean(p)) if len(p) else float("nan"),
        "empirical_rate": float(np.mean(y)) if len(y) else float("nan"),
    }
    return metrics, bins


def evaluate_calibration_per_split(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    split_col: str = "split",
    n_bins: int = 15,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute calibration per split-value plus pooled.

    Returns a (metrics_df, bin_tables_by_split) tuple. ``metrics_df`` has one
    row per split + a final ``pooled`` row.
    """
    metrics_rows = []
    bin_tables: dict[str, pd.DataFrame] = {}
    for split, sub in df.groupby(split_col):
        m, b = evaluate_calibration(sub[pred_col].to_numpy(), sub[label_col].to_numpy(), n_bins=n_bins)
        m[split_col] = str(split)
        metrics_rows.append(m)
        bin_tables[str(split)] = b
    pooled_metrics, pooled_bins = evaluate_calibration(df[pred_col].to_numpy(), df[label_col].to_numpy(), n_bins=n_bins)
    pooled_metrics[split_col] = "pooled"
    metrics_rows.append(pooled_metrics)
    bin_tables["pooled"] = pooled_bins
    cols_first = [split_col, "n", "ece", "brier", "log_loss", "mean_predicted", "empirical_rate"]
    out = pd.DataFrame(metrics_rows)[cols_first]
    return out, bin_tables


def evaluate_calibration_per_dataset(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    dataset_col: str = "dataset_id",
    n_bins: int = 15,
) -> pd.DataFrame:
    """Per-dataset calibration scalar metrics."""
    rows = []
    for ds_id, sub in df.groupby(dataset_col):
        m, _ = evaluate_calibration(sub[pred_col].to_numpy(), sub[label_col].to_numpy(), n_bins=n_bins)
        m[dataset_col] = str(ds_id)
        rows.append(m)
    cols_first = [dataset_col, "n", "ece", "brier", "log_loss", "mean_predicted", "empirical_rate"]
    return pd.DataFrame(rows)[cols_first]


def plot_reliability_diagram(
    bin_tables: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str | None = None,
    splits_to_plot: Iterable[str] | None = ("train", "val", "test", "pooled"),
) -> Path:
    """Render a reliability diagram with one curve per split and a y=x identity line.

    Bins with very small ``n`` are still plotted but their marker size scales
    with ``fraction`` so reviewers can see where mass concentrates.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=140)
    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1, label="perfect")

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "black", "tab:red", "tab:purple"]
    plotted = []
    for i, split in enumerate(splits_to_plot or sorted(bin_tables.keys())):
        if split not in bin_tables:
            continue
        b = bin_tables[split]
        if len(b) == 0:
            continue
        sizes = (b["fraction"] * 600).clip(lower=8.0).to_numpy()
        ax.scatter(
            b["mean_predicted"],
            b["empirical_rate"],
            s=sizes,
            color=color_cycle[i % len(color_cycle)],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            label=split,
        )
        ax.plot(
            b["mean_predicted"],
            b["empirical_rate"],
            color=color_cycle[i % len(color_cycle)],
            linewidth=1.0,
            alpha=0.6,
        )
        plotted.append(split)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P(viable)")
    ax.set_ylabel("empirical P(viable)")
    ax.set_title(title or "Reliability diagram (DMS function prior)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
