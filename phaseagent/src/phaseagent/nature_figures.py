"""Nature-tier figure helpers for the EditGuard ICML paper.

Design choices follow the Nature figure guide (https://www.nature.com/nature/for-authors/final-submission)
and current best practice in computational biology figure design:

- **Vector format** for everything that goes to the paper: PDF + SVG. PNG is
  exported only for slide previews / GitHub thumbnails.
- **Color-blind safe palette**: Okabe & Ito 8-color set
  (https://jfly.uni-koeln.de/color/) — the universally-recommended palette.
- **Sans-serif type**: Helvetica/Arial fallbacks, base size 7pt for
  axis labels / 8pt for legends, matching Nature's figure type sizes.
- **Multipanel layouts**: 1- or 2-column figures sized to Nature's 89mm
  (single) and 183mm (double) text widths, with consistent inter-panel
  whitespace and bold lower-case panel labels (a, b, c, d).
- **Statistical annotations**: significance bars use Holm-corrected p-values
  passed in, never recomputed inside the plotting function.
- **No chart-junk**: no gridlines beyond a subtle horizontal at zero, no
  background tints, no 3D effects, no legend boxes.

The module exposes two high-level entry points:

- ``setup_nature_style()``: configure matplotlib rcParams once.
- ``save_figure(fig, path)``: save to both PDF and SVG with consistent dpi.

Plus three figure builders aligned to the EditGuard story:

- ``figure_headline_leaderboard``: Fig. 1 — leaderboard with bootstrap CIs,
  per-protein breakdown, and stratified p-values.
- ``figure_few_shot_scaling``: Fig. 2 — performance vs train-set size for
  every method, with shaded 95% CI band.
- ``figure_calibration_and_abstention``: Fig. 3 — per-protein reliability
  diagrams + the conformal-abstention recovery curve.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


# Okabe-Ito 8-color palette + neutral grey. CB-safe, photocopy-safe, journal-safe.
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermilion": "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}


# Nature single-column = 89mm = 3.504 in; double-column = 183mm = 7.205 in.
NATURE_SINGLE_IN = 3.504
NATURE_DOUBLE_IN = 7.205


def setup_nature_style(font: str = "Helvetica") -> None:
    """Configure matplotlib for Nature-tier output.

    Idempotent. Safe to call multiple times. Falls back to Arial if
    Helvetica is not installed.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for Nature figures")
    fallbacks = [font, "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fallbacks,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,  # editable text in vector output
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, path: str | Path, formats: Sequence[str] = ("pdf", "svg", "png")) -> list[Path]:
    """Save a figure to multiple vector + raster formats.

    Returns the list of paths written. PNG always rendered at 600 dpi for
    slide / web previews; PDF + SVG carry the editable vectors for the paper.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for Nature figures")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = p.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=600 if fmt == "png" else None)
        written.append(out)
    return written


def _panel_label(ax, label: str, dx: float = -0.18, dy: float = 1.05) -> None:
    """Place a bold lower-case panel label in the upper-left, Nature style."""
    ax.text(
        dx, dy, label,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        ha="left", va="bottom",
    )


def figure_headline_leaderboard(
    leaderboard: pd.DataFrame,
    per_protein: pd.DataFrame | None = None,
    pairwise_pvals: pd.DataFrame | None = None,
    *,
    metric: str = "functional_hit_rate",
    method_order: Sequence[str] | None = None,
    palette: dict[str, str] | None = None,
):
    """Fig. 1: leaderboard with CIs, per-protein breakdown, p-value annotations.

    ``leaderboard`` is the output of build_headline_leaderboard.py with at least
    the columns: method, metric, mean, ci_lo, ci_hi, n.

    ``per_protein`` optionally adds a second panel showing the per-protein
    breakdown so reviewers can see whether one protein drives the headline.
    Expected columns: method, dataset_id, metric, mean.

    ``pairwise_pvals`` adds significance bars across the bars in panel a.
    Expected columns: method_a, method_b, p_holm.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required")
    setup_nature_style()
    palette = palette or _default_palette()
    data = leaderboard[leaderboard["metric"] == metric].copy()
    if method_order is None:
        method_order = list(data.sort_values("mean", ascending=False)["method"])
    data = data.set_index("method").reindex(method_order).reset_index()

    has_per_protein = per_protein is not None and len(per_protein) > 0
    width = NATURE_DOUBLE_IN if has_per_protein else NATURE_SINGLE_IN
    height = 3.0
    fig = plt.figure(figsize=(width, height))
    if has_per_protein:
        gs = GridSpec(1, 2, width_ratios=[1.6, 1.0], wspace=0.35)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
    else:
        ax_a = fig.add_subplot(1, 1, 1)
        ax_b = None

    # Panel a: leaderboard with horizontal CI bars.
    y = np.arange(len(data))[::-1]
    colors = [palette.get(m, OKABE_ITO["grey"]) for m in data["method"]]
    means = data["mean"].astype(float).to_numpy()
    lo = (data["mean"] - data["ci_lo"]).astype(float).to_numpy()
    hi = (data["ci_hi"] - data["mean"]).astype(float).to_numpy()
    ax_a.barh(y, means, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax_a.errorbar(means, y, xerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6, capsize=2)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(data["method"])
    ax_a.set_xlabel(_pretty_metric(metric))
    ax_a.set_xlim(0, 1.0)
    _panel_label(ax_a, "a")

    if pairwise_pvals is not None and len(pairwise_pvals) > 0:
        _annotate_significance(ax_a, data, pairwise_pvals, y_positions=y, metric=metric)

    # Panel b: per-protein breakdown.
    if ax_b is not None and has_per_protein:
        pp = per_protein[per_protein["metric"] == metric].copy()
        proteins = sorted(pp["dataset_id"].unique())
        method_subset = list(data["method"])
        x = np.arange(len(proteins))
        bar_w = 0.8 / max(1, len(method_subset))
        for j, m in enumerate(method_subset):
            ys = []
            for ds in proteins:
                row = pp[(pp["method"] == m) & (pp["dataset_id"] == ds)]
                ys.append(float(row["mean"].iloc[0]) if len(row) else float("nan"))
            ax_b.bar(
                x + (j - (len(method_subset) - 1) / 2) * bar_w,
                ys,
                width=bar_w * 0.95,
                color=palette.get(m, OKABE_ITO["grey"]),
                edgecolor="black",
                linewidth=0.3,
                alpha=0.9,
                label=m,
            )
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([_short_protein(p) for p in proteins], rotation=30, ha="right")
        ax_b.set_ylabel(_pretty_metric(metric))
        ax_b.set_ylim(0, 1.0)
        _panel_label(ax_b, "b")

    fig.tight_layout()
    return fig


def figure_few_shot_scaling(
    metrics: pd.DataFrame,
    *,
    metric: str = "functional_hit_rate",
    methods: Sequence[str] | None = None,
    palette: dict[str, str] | None = None,
):
    """Fig. 2: performance vs train-set size, with shaded 95% CI bands.

    ``metrics`` columns: method, n_train, mean, ci_lo, ci_hi (one row per
    (method, n_train)). Plots one line per method with a shaded band.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required")
    setup_nature_style()
    palette = palette or _default_palette()
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_IN, 2.6))
    methods = methods or sorted(metrics["method"].unique())
    for m in methods:
        sub = metrics[metrics["method"] == m].sort_values("n_train")
        x = sub["n_train"].astype(int).to_numpy()
        y = sub["mean"].astype(float).to_numpy()
        lo = sub["ci_lo"].astype(float).to_numpy()
        hi = sub["ci_hi"].astype(float).to_numpy()
        c = palette.get(m, OKABE_ITO["grey"])
        ax.plot(x, y, marker="o", color=c, label=m, markeredgecolor="white", markeredgewidth=0.4)
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Number of training variants per assay")
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_xticks([50, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    return fig


def figure_calibration_and_abstention(
    reliability: pd.DataFrame,
    abstention_curve: pd.DataFrame | None = None,
    *,
    palette: dict[str, str] | None = None,
):
    """Fig. 3: per-protein reliability diagrams + abstention recovery.

    ``reliability`` columns: dataset_id, bin_center, empirical, n.
    ``abstention_curve`` columns (optional): abstain_rate, hit_rate,
    hit_rate_lo, hit_rate_hi.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required")
    setup_nature_style()
    palette = palette or _default_palette()
    proteins = sorted(reliability["dataset_id"].unique())

    has_abstain = abstention_curve is not None and len(abstention_curve) > 0
    width = NATURE_DOUBLE_IN if has_abstain else NATURE_SINGLE_IN
    fig = plt.figure(figsize=(width, 2.6))
    if has_abstain:
        gs = GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.35)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
    else:
        ax_a = fig.add_subplot(1, 1, 1)
        ax_b = None

    # Panel a: per-protein reliability.
    ax_a.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6)
    for i, ds in enumerate(proteins):
        sub = reliability[reliability["dataset_id"] == ds].sort_values("bin_center")
        c = palette.get(f"protein:{ds}", list(OKABE_ITO.values())[(i + 1) % len(OKABE_ITO)])
        ax_a.plot(
            sub["bin_center"], sub["empirical"],
            marker="o", color=c, label=_short_protein(ds),
            markeredgecolor="white", markeredgewidth=0.4,
        )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Predicted P(viable)")
    ax_a.set_ylabel("Empirical fraction viable")
    ax_a.legend(loc="lower right")
    _panel_label(ax_a, "a")

    # Panel b: abstention recovery curve.
    if ax_b is not None and has_abstain:
        ac = abstention_curve.sort_values("abstain_rate")
        x = ac["abstain_rate"].astype(float).to_numpy()
        y = ac["hit_rate"].astype(float).to_numpy()
        lo = ac.get("hit_rate_lo", pd.Series(y - 0.02)).astype(float).to_numpy()
        hi = ac.get("hit_rate_hi", pd.Series(y + 0.02)).astype(float).to_numpy()
        ax_b.plot(x, y, color=OKABE_ITO["blue"], marker="o", markeredgecolor="white", markeredgewidth=0.4)
        ax_b.fill_between(x, lo, hi, color=OKABE_ITO["blue"], alpha=0.18, linewidth=0)
        ax_b.set_xlabel("Abstention rate")
        ax_b.set_ylabel("Hit rate (selective)")
        _panel_label(ax_b, "b")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_DEFAULT_METHOD_COLORS = {
    "dms_prior_rerank":            OKABE_ITO["blue"],
    "editguard_guided":            OKABE_ITO["vermilion"],
    "editguard_dplm":              OKABE_ITO["vermilion"],
    "esm2_masked_marginal":        OKABE_ITO["green"],
    "esm2_masked_marginal_dms_rerank": OKABE_ITO["skyblue"],
    "tranception_l_rerank":        OKABE_ITO["orange"],
    "eve_ensemble_rerank":         OKABE_ITO["yellow"],
    "esm_if_rerank":               OKABE_ITO["purple"],
    "dms_pool_guided":             OKABE_ITO["grey"],
    "additive_baseline":           OKABE_ITO["black"],
    "thermompnn":                  OKABE_ITO["green"],
    "thermompnn_d":                OKABE_ITO["skyblue"],
    "venusrem":                    OKABE_ITO["orange"],
    "saprot":                      OKABE_ITO["yellow"],
    "novelty":                     OKABE_ITO["grey"],
    "objective_only":              OKABE_ITO["grey"],
    "random":                      OKABE_ITO["grey"],
}


def _default_palette() -> dict[str, str]:
    return dict(_DEFAULT_METHOD_COLORS)


def _pretty_metric(metric: str) -> str:
    return {
        "functional_hit_rate": "Functional hit rate",
        "joint_success_rate":  "Joint success rate",
        "constraint_satisfaction_rate": "Constraint satisfaction",
        "spearman": "Spearman ρ",
    }.get(metric, metric.replace("_", " ").capitalize())


def _short_protein(ds_id: str) -> str:
    """Short, paper-friendly label for a ProteinGym dataset id."""
    s = str(ds_id)
    base = s.split("_")[0]
    short = {
        "F7YBW8": "F7YBW8 (toxin)",
        "GCN4":   "GCN4 (TF)",
        "GFP":    "GFP",
        "HIS7":   "HIS7",
        "PHOT":   "Phot LOV",
        "SPG1":   "Protein G B1",
        "CAPSD":  "AAV2",
        "D7PM05": "FP-Clygr",
        "Q8WTC7": "FP-9CNID",
        "Q6WV12": "FP-9MAXI",
    }
    return short.get(base, base)


def _annotate_significance(ax, data: pd.DataFrame, pvals: pd.DataFrame, y_positions: np.ndarray, metric: str):
    """Light-touch significance bars for a horizontal bar plot.

    For brevity we annotate only the most-stringent comparisons (p_holm < 0.001
    → ***, < 0.01 → **, < 0.05 → *) against the top-ranked method.
    """
    if "p_holm" not in pvals.columns:
        return
    top = data.iloc[0]["method"]
    rel = pvals[(pvals["method_a"] == top) | (pvals["method_b"] == top)]
    method_y = {m: y_positions[i] for i, m in enumerate(data["method"])}
    x_max = float(data["ci_hi"].max()) + 0.02
    for _, row in rel.iterrows():
        other = row["method_a"] if row["method_b"] == top else row["method_b"]
        if other not in method_y:
            continue
        p = float(row["p_holm"])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if not stars:
            continue
        ax.text(x_max, method_y[other], stars, va="center", ha="left", fontsize=7)
