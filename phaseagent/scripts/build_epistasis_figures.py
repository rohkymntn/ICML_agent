"""Build clean, Bio-ML-legible figures for the three-layer epistasis decomposition.

Reads the decomposed multi-mutant table (ddG_observed, ddG_additive) produced by
``run_epistasis_atlas``, fits the per-protein global-epistasis link, and renders:

  fig1_additivity_saturation   measured vs additive ddG -> additivity ranks well
                               but overshoots the assay floor (global epistasis)
  fig2_three_layer_r2          per-protein variance explained: additive baseline
                               -> + global (saturation) link
  fig3_specific_epistasis      distribution of the specific-epistasis residual
                               (kcal/mol) that a learned model must predict

Language is deliberately field-standard: additive baseline, global / specific
epistasis, measured stability (ddG), variance explained. No "phase/viability"
jargon. Saves PDF + PNG to ``paper/figures_epistasis/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phaseagent.epistasis_decomposition import add_global_specific_layers


# --- clean publication style (minimal, no chartjunk) -----------------------
def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


INK = "#222222"
BLUE = "#3B6FB6"
RED = "#C0504D"
GREY = "#9A9A9A"


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # metadata CreationDate=None drops the embedded timestamp so the PDF is
    # byte-reproducible (regenerate-and-diff actually verifies the figure).
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def load_decomposed(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    for c in ("ddG_ML", "ddG_additive", "epsilon"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[np.isfinite(d["ddG_ML"]) & np.isfinite(d["ddG_additive"])].copy()
    return add_global_specific_layers(d)


def three_layer_atlas(d: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for ds, sub in d.groupby("dataset_id"):
        obs = sub["ddG_ML"].to_numpy(float)
        add = sub["ddG_additive"].to_numpy(float)
        glob = sub["ddG_global"].to_numpy(float)
        m = np.isfinite(obs) & np.isfinite(add) & np.isfinite(glob)
        if m.sum() < min_n:
            continue
        obs, add, glob = obs[m], add[m], glob[m]
        sstot = np.sum((obs - obs.mean()) ** 2)
        r2_add = 1 - np.sum((obs - add) ** 2) / sstot
        r2_glob = 1 - np.sum((obs - glob) ** 2) / sstot
        spec = obs - glob
        rows.append({
            "dataset_id": ds, "n": int(m.sum()),
            "r2_additive": r2_add, "r2_global": r2_glob,
            "spec_std": float(np.std(spec, ddof=1)),
            "spec_var_share": float(np.var(spec, ddof=1) / np.var(obs, ddof=1)),
        })
    return pd.DataFrame(rows)


def fig1_saturation(d: pd.DataFrame, outdir: Path) -> None:
    obs = d["ddG_ML"].to_numpy(float)
    add = d["ddG_additive"].to_numpy(float)
    floor = np.nanmin(obs)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    hb = ax.hexbin(add, obs, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    lim = [-9, 4]
    ax.plot(lim, lim, color=INK, lw=1.0, ls="--", label="additive (y = x)")
    # pooled monotonic link, for illustration of the squashing
    order = np.argsort(add)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(add, obs)
    xx = np.linspace(lim[0], lim[1], 200)
    ax.plot(xx, iso.predict(xx), color=RED, lw=2.0, label="global-epistasis link")
    ax.axhline(floor, color=GREY, lw=0.9, ls=":")
    ax.text(lim[0] + 0.2, floor + 0.15, "stability assay floor", color=GREY, fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Additive prediction:  ΔΔG(mut₁) + ΔΔG(mut₂)  (kcal/mol)")
    ax.set_ylabel("Measured ΔΔG of the double mutant  (kcal/mol)")
    ax.set_title("Additivity ranks well but overshoots\n(global epistasis = saturation)")
    ax.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("multi-mutants (log count)", fontsize=9)
    _save(fig, outdir, "fig1_additivity_saturation")


def fig2_three_layer(atlas: pd.DataFrame, outdir: Path) -> None:
    a = atlas.copy()
    add = np.clip(a["r2_additive"].to_numpy(float), -4, 1)
    glob = np.clip(a["r2_global"].to_numpy(float), -4, 1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    x0, x1 = 0.0, 1.0
    rng = np.random.default_rng(0)
    jit = (rng.random(len(add)) - 0.5) * 0.12
    for i in range(len(add)):
        ax.plot([x0 + jit[i], x1 + jit[i]], [add[i], glob[i]], color=GREY, lw=0.4, alpha=0.35)
    ax.scatter(np.full(len(add), x0) + jit, add, s=14, color=BLUE, zorder=3, label="additive baseline")
    ax.scatter(np.full(len(glob), x1) + jit, glob, s=14, color=RED, zorder=3, label="+ global link")
    for x, v in ((x0, add), (x1, glob)):
        ax.plot([x - 0.18, x + 0.18], [np.median(v)] * 2, color=INK, lw=2.2, zorder=4)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.text(0.5, 0.03, "predicting the mean (R²=0)", ha="center", color=INK, fontsize=8)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Additive\n(sum of singles)", "Additive\n+ global link"])
    ax.set_ylabel("Variance of measured ΔΔG explained  (R²)")
    ax.set_title(f"A saturation link recovers most non-additivity\n(per protein, n={len(add)} domains)")
    ax.set_xlim(-0.4, 1.4)
    _save(fig, outdir, "fig2_three_layer_r2")


def fig3_specific(atlas: pd.DataFrame, outdir: Path) -> None:
    spec = atlas["spec_std"].to_numpy(float)
    spec = spec[np.isfinite(spec)]
    med = float(np.median(spec))
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.hist(spec, bins=28, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvspan(0.1, 0.3, color=GREY, alpha=0.18)
    ax.text(0.2, ax.get_ylim()[1] * 0.92, "measurement\nnoise", ha="center", va="top", color=GREY, fontsize=8.5)
    ax.axvline(med, color=RED, lw=2.0)
    ax.text(med + 0.02, ax.get_ylim()[1] * 0.8, f"median\n{med:.2f} kcal/mol", color=RED, fontsize=9)
    ax.set_xlabel("Specific-epistasis residual, std per protein  (kcal/mol)")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Specific epistasis left after additive + global\n(the signal a learned model must predict)")
    _save(fig, outdir, "fig3_specific_epistasis")


def fig4_headroom(hr: pd.DataFrame, outdir: Path) -> None:
    from scipy.stats import spearmanr
    pp = []
    zrows = []
    for _, g in hr.groupby("dataset_id"):
        x = pd.to_numeric(g["dplm_epistasis"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g["eps_specific"], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 20:
            continue
        x, y = x[m], y[m]
        pp.append(spearmanr(x, y).statistic)
        if x.std() > 0 and y.std() > 0:
            zrows.append(np.c_[(x - x.mean()) / x.std(), (y - y.mean()) / y.std()])
    pp = np.array(pp)
    Z = np.vstack(zrows)
    qs = np.quantile(Z[:, 0], np.linspace(0, 1, 11))
    cent, mu, se = [], [], []
    for i in range(10):
        sel = (Z[:, 0] >= qs[i]) & (Z[:, 0] <= qs[i + 1])
        if sel.sum() > 5:
            cent.append((qs[i] + qs[i + 1]) / 2)
            mu.append(Z[sel, 1].mean())
            se.append(Z[sel, 1].std() / np.sqrt(sel.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].hist(pp, bins=16, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax[0].axvline(0, color=INK, lw=0.8, ls=":")
    ax[0].axvline(np.median(pp), color=RED, lw=2.0)
    ax[0].text(np.median(pp) + 0.02, ax[0].get_ylim()[1] * 0.78, f"median\nρ={np.median(pp):.2f}", color=RED, fontsize=9)
    ax[0].set_xlabel("Per-protein rank correlation (Spearman ρ)")
    ax[0].set_ylabel("Number of proteins")
    ax[0].set_title("DPLM zero-shot epistasis predicts\nthe measured specific residual")
    ax[1].errorbar(cent, mu, yerr=se, fmt="o-", color=BLUE, lw=1.6, ms=5, capsize=2)
    ax[1].axhline(0, color=INK, lw=0.8, ls=":")
    ax[1].set_xlabel("DPLM predicted epistasis  (z-scored per protein)")
    ax[1].set_ylabel("Measured specific epistasis  (z-scored)")
    ax[1].set_title("Predicted tracks measured\n(label-free, a floor for Model 1)")
    fig.tight_layout()
    _save(fig, outdir, "fig4_headroom_dplm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomposed", default="/tmp/epi_decomposed.parquet")
    ap.add_argument("--outdir", default="paper/figures_epistasis")
    ap.add_argument("--atlas-out", default="paper/figures_epistasis/three_layer_atlas.parquet")
    ap.add_argument("--headroom", default="/tmp/epi_headroom.parquet")
    ap.add_argument("--from-committed", action="store_true",
                    help="Regenerate the PAPER-CITED figures (fig2, fig3, fig4) from the "
                         "committed per-protein atlas + headroom parquet, no /tmp inputs.")
    args = ap.parse_args()

    _style()
    outdir = Path(args.outdir)
    if args.from_committed:
        # The paper cites fig2/fig3 (decomposition) + fig4 (zero-shot headroom); all three
        # regenerate from committed artifacts under paper/figures_epistasis + outputs/epistasis.
        atlas = pd.read_parquet(args.atlas_out)
        print(f"[figures] committed atlas: {len(atlas)} proteins; "
              f"median R2 additive {atlas['r2_additive'].median():.3f} -> "
              f"+global {atlas['r2_global'].median():.3f}; "
              f"specific std {atlas['spec_std'].median():.3f} kcal/mol")
        fig2_three_layer(atlas, outdir)
        fig3_specific(atlas, outdir)
        hr = pd.read_parquet("outputs/epistasis/headroom_dplm.parquet")
        fig4_headroom(hr, outdir)
        print(f"[figures] done (from committed) -> {outdir}")
        return

    print("[figures] loading + fitting global link ...")
    d = load_decomposed(Path(args.decomposed))
    atlas = three_layer_atlas(d)
    Path(args.atlas_out).parent.mkdir(parents=True, exist_ok=True)
    atlas.to_parquet(args.atlas_out, index=False)
    print(f"[figures] {len(d):,} multi-mutants, {len(atlas)} proteins")
    print(f"[figures] median R2: additive {atlas['r2_additive'].median():.3f} -> +global {atlas['r2_global'].median():.3f}; "
          f"specific std {atlas['spec_std'].median():.3f} kcal/mol")

    fig1_saturation(d, outdir)
    fig2_three_layer(atlas, outdir)
    fig3_specific(atlas, outdir)
    if Path(args.headroom).exists():
        hr = pd.read_parquet(args.headroom)
        fig4_headroom(hr, outdir)
    print(f"[figures] done -> {outdir}")


if __name__ == "__main__":
    main()
