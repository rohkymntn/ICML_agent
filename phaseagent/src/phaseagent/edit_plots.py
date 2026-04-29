"""Publication-quality plots for EditGuard experiments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def set_nature_style():
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="white", context="paper")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.0,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _finish_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(width=1.0)


def _save_all(fig, out_prefix: str | Path):
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _method_palette(methods):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    return {m: cmap(i / max(len(methods) - 1, 1)) for i, m in enumerate(methods)}


def plot_editing_frontier(results: pd.DataFrame, out_prefix: str | Path):
    """Deprecated alias: plot aggregated Fig. 2 replacement, not raw spaghetti."""
    plot_fig2_replacement(results, out_prefix)


def plot_fig2_replacement(results: pd.DataFrame, out_prefix: str | Path):
    """Clean Fig. 2 replacement with aggregated bars and uncertainty."""
    import matplotlib.pyplot as plt
    import numpy as np

    set_nature_style()
    methods = sorted(results["method"].unique())
    colors = _method_palette(methods)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)

    metric_specs = [
        ("functional_hit_rate", "Functional hit rate"),
        ("joint_success_rate", "Joint success rate"),
        ("mean_edit_distance", "Mean edit distance"),
    ]
    for ax, (metric, label) in zip(axes, metric_specs):
        grouped = results.groupby("method")[metric].agg(["mean", "sem"]).reindex(methods)
        x = np.arange(len(methods))
        ax.bar(
            x,
            grouped["mean"].fillna(0),
            yerr=grouped["sem"].fillna(0),
            color=[colors[m] for m in methods],
            edgecolor="black",
            linewidth=1.0,
            capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel(label)
        if "rate" in metric:
            ax.set_ylim(0, 1.05)
        _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_grouped_metric_bars(
    results: pd.DataFrame,
    out_prefix: str | Path,
    x_col: str = "edit_budget",
    metric: str = "functional_hit_rate",
):
    """Grouped bar plot for method comparisons across discrete categories."""
    import matplotlib.pyplot as plt
    import numpy as np

    set_nature_style()
    df = results.dropna(subset=[metric]).copy()
    methods = sorted(df["method"].unique())
    x_vals = sorted(df[x_col].unique())
    colors = _method_palette(methods)
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    positions = np.arange(len(x_vals))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        grouped = sub.groupby(x_col)[metric].agg(["mean", "sem"]).reindex(x_vals)
        ax.bar(
            positions + (i - (len(methods) - 1) / 2) * width,
            grouped["mean"].fillna(0),
            width=width,
            yerr=grouped["sem"].fillna(0),
            label=method,
            color=colors[method],
            edgecolor="black",
            linewidth=1.0,
            capsize=2,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    if "rate" in metric:
        ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_method_boxplot(results: pd.DataFrame, out_prefix: str | Path, metric: str = "functional_hit_rate"):
    """Plot per-task metric distributions as boxplots."""
    import matplotlib.pyplot as plt

    set_nature_style()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    methods = sorted(results["method"].unique())
    data = [results[results["method"] == m][metric].dropna().to_numpy() for m in methods]
    box = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.6)
    cmap = plt.get_cmap("magma")
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(cmap(i / max(len(methods) - 1, 1)))
        patch.set_alpha(0.7)
        patch.set_linewidth(1.0)
    for key in ["whiskers", "caps", "medians"]:
        for artist in box[key]:
            artist.set_linewidth(1.0)
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(axis="x", labelrotation=30)
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_calibration(predictions: pd.DataFrame, out_prefix: str | Path, n_bins: int = 10):
    """Reliability plot for predicted function probabilities."""
    import matplotlib.pyplot as plt
    import numpy as np

    set_nature_style()
    df = predictions.dropna(subset=["prior_function_prob", "viable"]).copy()
    df["bin"] = pd.cut(df["prior_function_prob"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    cal = df.groupby("bin", observed=True).agg(pred=("prior_function_prob", "mean"), obs=("viable", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1.0)
    ax.plot(cal["pred"], cal["obs"], marker="o", color=plt.get_cmap("viridis")(0.65), linewidth=1.0)
    ax.set_xlabel("Predicted functional probability")
    ax.set_ylabel("Observed functional fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_ablation_bar(results: pd.DataFrame, out_prefix: str | Path, metric: str = "functional_hit_rate"):
    """Plot ablation mean performance with standard-error bars."""
    import matplotlib.pyplot as plt

    set_nature_style()
    label_col = "ablation" if "ablation" in results.columns else "method"
    grouped = results.groupby(label_col)[metric].agg(["mean", "sem", "count"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    cmap = plt.get_cmap("cividis")
    colors = [cmap(i / max(len(grouped) - 1, 1)) for i in range(len(grouped))]
    ax.bar(
        grouped[label_col],
        grouped["mean"],
        yerr=grouped["sem"].fillna(0),
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        capsize=2,
    )
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, max(1.0, float(grouped["mean"].max()) * 1.15))
    ax.tick_params(axis="x", labelrotation=30)
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_sample_efficiency(results: pd.DataFrame, out_prefix: str | Path):
    """Plot labeled functional hit rate vs number of generated candidates."""
    import matplotlib.pyplot as plt

    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    methods = sorted(results["method"].unique())
    cmap = plt.get_cmap("viridis")
    for i, method in enumerate(methods):
        sub = results[results["method"] == method]
        grouped = (
            sub.groupby("n_generated")["functional_hit_rate_labeled"]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("n_generated")
        )
        color = cmap(i / max(len(methods) - 1, 1))
        ax.plot(
            grouped["n_generated"],
            grouped["mean"],
            marker="o",
            color=color,
            linewidth=1.0,
            label=method,
        )
        ax.fill_between(
            grouped["n_generated"],
            grouped["mean"] - grouped["sem"].fillna(0),
            grouped["mean"] + grouped["sem"].fillna(0),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Generated candidates")
    ax.set_ylabel("Functional hit rate among labeled")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_coverage_vs_hit_rate(results: pd.DataFrame, out_prefix: str | Path):
    """Plot label coverage against labeled hit rate."""
    import matplotlib.pyplot as plt

    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    methods = sorted(results["method"].unique())
    colors = _method_palette(methods)
    for method in methods:
        sub = results[results["method"] == method]
        ax.scatter(
            sub["labeled_fraction"],
            sub["functional_hit_rate_labeled"],
            s=12,
            alpha=0.55,
            color=colors[method],
            label=method,
            linewidth=0,
        )
    ax.set_xlabel("DMS label coverage")
    ax.set_ylabel("Functional hit rate among labeled")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)


def plot_guidance_distribution_shift(
    selections: pd.DataFrame,
    out_prefix: str | Path,
    score_col: str = "prior_function_prob",
):
    """Density plot showing guided vs unguided score distributions."""
    import matplotlib.pyplot as plt

    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    methods = sorted(selections["method"].dropna().unique())
    colors = _method_palette(methods)
    for method in methods:
        vals = selections.loc[selections["method"] == method, score_col].dropna()
        if len(vals) < 2:
            continue
        vals.plot(kind="density", ax=ax, color=colors[method], linewidth=1.0, label=method)
    ax.set_xlabel(score_col.replace("_", " "))
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _finish_axes(ax)
    _save_all(fig, out_prefix)
    plt.close(fig)
