"""Bio-ML case-study figures: specific epistasis in protein FUNCTION.

  fig5_function_decomposition  the additive -> +global-link ladder across five
                               function assays (fluorescence + binding), showing
                               the three-layer structure generalizes from
                               stability to function.
  fig6_gb1_gain_of_function    GB1 binding: additive prediction vs measured
                               binding, with gain-of-function epistatic pairs
                               (individually weak, together strong) highlighted.

Field-standard language throughout (binding, fluorescence, additive baseline,
specific epistasis, gain of function). Saves PDF + PNG to paper/figures_epistasis/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phaseagent.mutations import parse_mutation_notation
from phaseagent.epistasis_decomposition import (
    add_global_specific_layers,
    decompose_multimutants,
)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
INK, BLUE, RED, GREY, GREEN = "#222222", "#3B6FB6", "#C0504D", "#9A9A9A", "#4C9A6B"

ASSAYS = [
    ("GFP\nfluorescence", "/tmp/GFP_AEQVI_Sarkisyan_2016.csv"),
    ("GB1 binding\n(Olson)", "/tmp/SPG1_STRSG_Olson_2014.csv"),
    ("GB1 binding\n(Wu)", "/tmp/SPG1_STRSG_Wu_2016.csv"),
    ("GRB2\nbinding", "/tmp/GRB2_HUMAN_Faure_2021.csv"),
    ("PABP\nbinding", "/tmp/PABP_YEAST_Melamed_2013.csv"),
]


def _save(fig, name):
    out = Path("paper/figures_epistasis")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def load_assay(path, name):
    d = pd.read_csv(path).rename(columns={"mutant": "mutation_notation"})
    d["dataset_id"] = name
    d["mutation_distance"] = d["mutation_notation"].astype(str).apply(lambda s: len(parse_mutation_notation(s)))
    d["DMS_score"] = pd.to_numeric(d["DMS_score"], errors="coerce")
    return d


def decompose_doubles(d):
    dd = d[d["mutation_distance"] == 2]
    if len(dd) > 40000:
        d = pd.concat([d[d["mutation_distance"] == 1], dd.sample(40000, random_state=0)])
    dec = add_global_specific_layers(
        decompose_multimutants(d, label_col="DMS_score", max_distance=2), label_col="DMS_score"
    )
    return dec[np.isfinite(dec["epsilon_specific"])].copy()


def fig5_ladder():
    rows = []
    for name, path in ASSAYS:
        b = decompose_doubles(load_assay(path, name))
        obs = b["DMS_score"].to_numpy(float)
        sstot = np.sum((obs - obs.mean()) ** 2)
        r2a = 1 - np.sum((obs - b["ddG_additive"].to_numpy(float)) ** 2) / sstot
        r2g = 1 - np.sum((obs - b["ddG_global"].to_numpy(float)) ** 2) / sstot
        share = float(np.var(b["epsilon_specific"], ddof=1) / np.var(obs, ddof=1))
        rows.append((name, max(r2a, -5), r2g, share))
    palette = ["#3B6FB6", "#C0504D", "#4C9A6B", "#E08214", "#8073AC"]
    rows = sorted(rows, key=lambda r: -r[3])  # by specific share, descending
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for i, (name, a, g, share) in enumerate(rows):
        c = palette[i % len(palette)]
        lw = 2.4 if share > 0.3 else 1.3
        ax.plot([0, 1], [a, g], "-", color=c, lw=lw, zorder=2,
                label=f"{name.replace(chr(10),' ')}  ({share*100:.0f}% specific)")
        ax.scatter([0, 1], [a, g], color=c, s=26, zorder=3)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.text(0.02, 0.06, "predicting the mean (R²=0)", fontsize=8, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Additive\n(sum of singles)", "Additive\n+ global link"])
    ax.set_ylabel("Variance of measured function explained (R²)")
    ax.set_title("Three-layer decomposition holds for protein FUNCTION")
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-5.6, 1.15)
    ax.legend(loc="lower right", fontsize=8.2)
    _save(fig, "fig5_function_decomposition")


def fig6_gb1():
    d = load_assay("/tmp/SPG1_STRSG_Wu_2016.csv", "GB1")
    sing = d[d["mutation_distance"] == 1]
    slk = dict(zip(sing["mutation_notation"].astype(str), sing["DMS_score"]))
    b = decompose_doubles(d)
    b["s1"] = b["mutation_notation"].apply(lambda n: slk.get(parse_mutation_notation(n)[0], np.nan))
    b["s2"] = b["mutation_notation"].apply(lambda n: slk.get(parse_mutation_notation(n)[1], np.nan))
    b = b[np.isfinite(b["s1"]) & np.isfinite(b["s2"])].copy()
    add = b["ddG_additive"].to_numpy(float)
    obs = b["DMS_score"].to_numpy(float)
    eps = b["epsilon_specific"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    sc = ax.scatter(add, obs, c=eps, cmap="coolwarm", vmin=-3, vmax=3, s=8, alpha=0.6, linewidths=0)
    lim = [min(add.min(), obs.min()) - 0.3, max(add.max(), obs.max()) + 0.3]
    ax.plot(lim, lim, color=INK, lw=1.0, ls="--", label="additive (y = x)")
    # annotate strongest gain-of-function pairs
    top = b.assign(ae=np.abs(eps)).sort_values("ae", ascending=False).head(4)
    for _, r in top.iterrows():
        ax.annotate(
            r["mutation_notation"],
            (r["ddG_additive"], r["DMS_score"]),
            fontsize=7.5, color=INK,
            xytext=(r["ddG_additive"] - 2.4, r["DMS_score"] + 0.15),
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6),
        )
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Additive prediction  (sum of single-mutant binding)")
    ax.set_ylabel("Measured binding of the double mutant")
    ax.set_title("GB1: gain-of-function epistasis\nindividually weak, together strong (above the line)")
    ax.legend(loc="lower right", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("specific epistasis", fontsize=9)
    _save(fig, "fig6_gb1_gain_of_function")


if __name__ == "__main__":
    print("[function-figs] building ...")
    fig5_ladder()
    fig6_gb1()
    print("[function-figs] done -> paper/figures_epistasis/")
