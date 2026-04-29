"""Advanced figure helpers for Spectral PhaseAgent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _prep_out(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_distance_shell_histogram(df: pd.DataFrame, out: str | Path) -> Path:
    """Figure 1A: histogram of mutation-distance shell counts per assay."""
    import matplotlib.pyplot as plt

    shell_counts = df.groupby("dataset_id")["mutation_distance"].nunique()
    out = _prep_out(out)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(shell_counts, bins=np.arange(1, shell_counts.max() + 2) - 0.5, color="#4c78a8")
    ax.set_xlabel("Number of mutation-distance shells")
    ax.set_ylabel("Assays")
    ax.set_title("ProteinGym substitution assays are broad but shallow")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_spectrum_histograms(summary_or_effects: pd.DataFrame, out: str | Path) -> Path:
    """Figure 2: distributions of single-mutant effects."""
    import matplotlib.pyplot as plt

    out = _prep_out(out)
    fig, ax = plt.subplots(figsize=(6, 4))
    if "delta_f" in summary_or_effects.columns:
        for ds_id, sub in summary_or_effects.groupby("dataset_id"):
            ax.hist(sub["delta_f"].dropna(), bins=40, alpha=0.35, label=str(ds_id)[:24])
        ax.legend(fontsize=7)
    elif "effect_mean" in summary_or_effects.columns:
        ax.hist(summary_or_effects["effect_mean"].dropna(), bins=40, color="#59a14f")
    ax.set_xlabel("Single-mutant effect")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_survival_curves(
    curves: pd.DataFrame,
    out: str | Path,
    true_col: str = "survival_true",
    pred_col: str | None = None,
) -> Path:
    """Figures 3/5: empirical and predicted survival curves, one dataset per panel."""
    import matplotlib.pyplot as plt

    out = _prep_out(out)
    groups = list(curves.groupby("dataset_id")) if "dataset_id" in curves.columns else [("dataset", curves)]
    n = len(groups)
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False, sharey=True)
    for ax, (ds_id, sub) in zip(axes.ravel(), groups):
        label = str(ds_id).replace("_", " ")
        if true_col in sub.columns:
            ax.plot(
                sub["mutation_distance"],
                sub[true_col],
                marker="o",
                linewidth=1.7,
                label="empirical V(d)",
            )
        if pred_col and pred_col in sub.columns:
            ax.plot(
                sub["mutation_distance"],
                sub[pred_col],
                linestyle="--",
                linewidth=1.7,
                label=pred_col.replace("survival_", ""),
            )
        if "survival_additive_mc" in sub.columns and pred_col != "survival_additive_mc":
            ax.plot(
                sub["mutation_distance"],
                sub["survival_additive_mc"],
                linestyle=":",
                linewidth=1.3,
                label="additive MC",
            )
        ax.set_title(label[:42], fontsize=9)
        ax.set_xlabel("Mutation distance")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Survival V(d)")
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_dc_scatter(metrics_or_boundaries: pd.DataFrame, out: str | Path) -> Path:
    """Figure 6: predicted vs. true critical radii."""
    import matplotlib.pyplot as plt

    out = _prep_out(out)
    x_col = "dc_true"
    y_col = "dc_pred" if "dc_pred" in metrics_or_boundaries.columns else "dc"
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(metrics_or_boundaries[x_col], metrics_or_boundaries[y_col], alpha=0.8)
    lo = float(np.nanmin(metrics_or_boundaries[[x_col, y_col]].to_numpy()))
    hi = float(np.nanmax(metrics_or_boundaries[[x_col, y_col]].to_numpy()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True dc")
    ax.set_ylabel("Predicted dc")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_search_frontier(results: pd.DataFrame, out: str | Path) -> Path:
    """Figure 7: hit rate vs. novelty frontier."""
    import matplotlib.pyplot as plt

    out = _prep_out(out)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for policy, sub in results.groupby("policy"):
        ax.scatter(
            sub["mean_mutation_distance"],
            sub["functional_hit_rate"],
            label=policy,
            alpha=0.75,
        )
    ax.set_xlabel("Mean mutation distance")
    ax.set_ylabel("Functional hit rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
