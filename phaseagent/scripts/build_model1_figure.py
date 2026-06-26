"""Proof figure (fig8): a trained coupling head (Model 1) predicts FUNCTION
epistasis on held-out doubles, where the zero-shot DPLM PLL score cannot.

Regenerates from the COMMITTED out-of-fold artifacts
``outputs/epistasis/model1_oof_<assay>.csv`` (columns
``y_true, pll, oof_doubles, oof_position, ...``) — no retraining, no GPU, no
volume intermediates. The numbers it draws are exactly the committed
``model1_<assay>_summary.json`` values (guarded by tests/test_model1_committed.py).

  Panel A: held-out Spearman, zero-shot DPLM PLL vs Model 1, on GB1 + GFP.
  Panel B: Model 1 out-of-fold prediction vs measured specific epistasis on
           held-out GB1 doubles.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
BLUE, RED, GREY = "#3B6FB6", "#C0504D", "#9A9A9A"

OOF = Path("outputs/epistasis")
ASSAYS = {"GB1 binding": "GB1_Olson", "GFP fluorescence": "GFP"}


def load_oof(tag):
    df = pd.read_csv(OOF / f"model1_oof_{tag}.csv")
    y, pll, oof = df["y_true"].to_numpy(), df["pll"].to_numpy(), df["oof_doubles"].to_numpy()
    return dict(y=y, oof=oof, pll=pll,
                m1=spearmanr(y, oof).statistic, zs=spearmanr(y, pll).statistic)


def main(out="paper/figures_epistasis"):
    res = {name: load_oof(tag) for name, tag in ASSAYS.items()}
    for name in res:
        print(f"{name}: Model1={res[name]['m1']:.3f}  zero-shot={res[name]['zs']:.3f}")
    # self-check: the figure must draw the committed summary numbers, not drift.
    import json
    for name, tag in ASSAYS.items():
        s = json.loads((OOF / f"model1_{tag}_summary.json").read_text())
        assert abs(res[name]["m1"] - s["model1_doubles_spearman"]) < 1e-6
        assert abs(res[name]["zs"] - s["zeroshot_spearman"]) < 1e-6

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # Panel A — bars
    names = list(res.keys())
    x = np.arange(len(names))
    zs = [res[n]["zs"] for n in names]
    m1 = [res[n]["m1"] for n in names]
    ax[0].bar(x - 0.2, zs, 0.4, color=GREY, label="zero-shot DPLM PLL")
    ax[0].bar(x + 0.2, m1, 0.4, color=RED, label="Model 1 (trained coupling head)")
    for xi, v in zip(x - 0.2, zs):
        ax[0].text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.2, m1):
        ax[0].text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("Held-out Spearman (specific epistasis)")
    ax[0].set_title("A   Model 1 predicts function epistasis;\nzero-shot DPLM cannot", loc="left", fontsize=12, fontweight="bold")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_ylim(0, max(m1) * 1.25)

    # Panel B — GB1 scatter
    g = res["GB1 binding"]
    ok = np.isfinite(g["oof"])
    ax[1].hexbin(g["oof"][ok], g["y"][ok], gridsize=40, bins="log", cmap="Reds", mincnt=1, linewidths=0)
    rho = spearmanr(g["y"][ok], g["oof"][ok]).statistic
    ax[1].set_xlabel("Model 1 prediction (held-out)")
    ax[1].set_ylabel("Measured specific epistasis")
    ax[1].set_title(f"B   GB1 held-out doubles  (ρ = {rho:.2f})", loc="left", fontsize=12, fontweight="bold")

    fig.tight_layout()
    outp = Path(out)
    fig.savefig(outp / "fig8_model1_function.pdf", bbox_inches="tight")
    fig.savefig(outp / "fig8_model1_function.png", bbox_inches="tight")
    print("saved fig8_model1_function")


if __name__ == "__main__":
    main()
