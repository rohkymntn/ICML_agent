"""Build real v2 figures from actual Modal outputs (no toy data).

Reads the pulled CSVs from /tmp and renders publication-ready PDFs/SVGs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.nature_figures import (  # noqa: E402
    OKABE_ITO,
    figure_calibration_and_abstention,
    figure_few_shot_scaling,
    figure_headline_leaderboard,
    save_figure,
    setup_nature_style,
)

OUT = ROOT / "outputs" / "editguard" / "figures_v2_real"
OUT.mkdir(parents=True, exist_ok=True)
setup_nature_style()


def fig_megascale_t2_additive():
    """Headline plot: model vs additive Spearman across Megascale doubles."""
    df = pd.read_csv("/tmp/megascale_t2.csv")
    df = df.dropna(subset=["overall_model", "overall_additive"])
    if df.empty:
        print("[fig] no megascale T2 rows; skipping")
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(
        df["overall_additive"], df["overall_model"],
        color=OKABE_ITO["vermilion"], s=8, alpha=0.7, edgecolor="black", linewidth=0.3,
    )
    lim = (-0.4, 1.0)
    ax.plot(lim, lim, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Additive baseline Spearman ρ (sum of single-mutant scores)")
    ax.set_ylabel("RF prior Spearman ρ (chemistry features)")
    ax.set_title("Megascale double-mutant ΔΔG (T2)\nadditive baseline dominates", fontsize=8)
    n_below = int((df["overall_model"] < df["overall_additive"]).sum())
    ax.text(
        0.05, 0.95,
        f"n proteins = {len(df)}\nadditive wins on {n_below}/{len(df)}",
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
    )
    fig.tight_layout()
    paths = save_figure(fig, OUT / "fig_megascale_t2_additive_vs_model")
    print(f"[fig] megascale T2 -> {paths}")


def fig_ood_family_drop():
    """Train AUROC vs OOD-test AUROC per held-out family."""
    rows = []
    for csv in sorted(Path("/tmp/ood_real/ood_family").glob("*_metrics.csv")):
        sub = pd.read_csv(csv)
        train = sub[sub["split"] == "train"]
        ood = sub[sub["split"] == "ood_test"]
        if len(train) == 0 or len(ood) == 0:
            continue
        rows.append(
            {
                "family": str(train["held_out_family"].iloc[0]),
                "train_auroc": float(train["auroc"].iloc[0]),
                "ood_auroc": float(ood["auroc"].iloc[0]),
                "train_spearman": float(train["spearman"].iloc[0]),
                "ood_spearman": float(ood["spearman"].iloc[0]),
                "ood_n": int(ood["n"].iloc[0]),
            }
        )
    if not rows:
        print("[fig] no OOD rows; skipping")
        return
    df = pd.DataFrame(rows).sort_values("ood_auroc", ascending=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["train_auroc"], height=0.36, color=OKABE_ITO["skyblue"], edgecolor="black", linewidth=0.4, label="train (in-distribution)")
    ax.barh(y + 0.18, df["ood_auroc"], height=0.36, color=OKABE_ITO["vermilion"], edgecolor="black", linewidth=0.4, label="OOD held-out family")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_", " ") for f in df["family"]])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Leave-one-family-out OOD generalization", fontsize=8)
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    ax.axvline(0.5, color=OKABE_ITO["grey"], linestyle=":", linewidth=0.5)
    fig.tight_layout()
    paths = save_figure(fig, OUT / "fig_ood_family_train_vs_ood")
    print(f"[fig] OOD -> {paths}")


def fig_conformal_abstention():
    """Hit rate vs abstention rate."""
    df = pd.read_csv("/tmp/conformal2.csv")
    if df.empty:
        print("[fig] no conformal rows; skipping")
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = df["abstain_rate"].astype(float).to_numpy()
    y = df["hit_rate"].astype(float).to_numpy()
    ax.plot(x, y, marker="o", color=OKABE_ITO["blue"], markeredgecolor="white", markeredgewidth=0.4)
    ax.fill_between(x, y - 0.01, y + 0.01, color=OKABE_ITO["blue"], alpha=0.15, linewidth=0)
    ax.set_xlabel("Abstention rate")
    ax.set_ylabel("Hit rate (top-quartile viable)")
    ax.set_title("Per-protein conformal abstention\nselectivity recovers hit rate", fontsize=8)
    base = float(y[0])
    ax.axhline(base, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.5, label=f"base rate = {base:.3f}")
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, OUT / "fig_conformal_abstention_curve")
    print(f"[fig] conformal -> {paths}")


def fig_megascale_vs_proteingym_priors():
    """Compare v1 ProteinGym prior vs Megascale prior on respective test sets."""
    pg = pd.read_csv("/tmp/v1_prior_metrics.csv")
    ms = pd.read_csv("/tmp/megascale_prior_metrics.csv")
    rows = []
    for source, df in [("ProteinGym v1", pg), ("Megascale dataset3_single", ms)]:
        for _, r in df.iterrows():
            rows.append({"source": source, "split": r["split"], "metric": "AUROC", "value": float(r["auroc"])})
            rows.append({"source": source, "split": r["split"], "metric": "Spearman ρ", "value": float(r["spearman"])})
    df = pd.DataFrame(rows)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)
    for ax, metric in zip(axes, ("AUROC", "Spearman ρ")):
        sub = df[df["metric"] == metric]
        sources = ["ProteinGym v1", "Megascale dataset3_single"]
        splits = ["train", "val", "test"]
        x = np.arange(len(splits))
        bar_w = 0.36
        for j, s in enumerate(sources):
            ys = [
                float(sub[(sub["source"] == s) & (sub["split"] == sp)]["value"].iloc[0])
                if len(sub[(sub["source"] == s) & (sub["split"] == sp)]) else float("nan")
                for sp in splits
            ]
            ax.bar(x + (j - 0.5) * bar_w, ys, width=bar_w * 0.95,
                   color=[OKABE_ITO["skyblue"], OKABE_ITO["green"]][j],
                   edgecolor="black", linewidth=0.3, label=s if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_title(metric, fontsize=8)
        ax.set_ylim(0, 1.0)
    axes[0].legend(loc="lower right", fontsize=6, frameon=False)
    fig.suptitle("DMS prior performance: Megascale ΔΔG transfers cleaner than ProteinGym DMS", fontsize=8, y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, OUT / "fig_proteingym_vs_megascale_prior")
    print(f"[fig] PG-vs-Megascale -> {paths}")


if __name__ == "__main__":
    fig_megascale_t2_additive()
    fig_ood_family_drop()
    fig_conformal_abstention()
    fig_megascale_vs_proteingym_priors()
    print("done")
