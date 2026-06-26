"""Proof figure: a trained coupling head (Model 1) predicts FUNCTION epistasis
held-out, where zero-shot DPLM cannot.

  Panel A: held-out Spearman, zero-shot DPLM vs Model 1, on GB1 binding + GFP.
  Panel B: Model 1 out-of-fold prediction vs measured specific epistasis on
           held-out GB1 doubles.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from train_model1 import load, train_fold

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
BLUE, RED, GREY = "#3B6FB6", "#C0504D", "#9A9A9A"


def held_out_double(path):
    F_, y, pll, pos, D, use_shift = load(path)
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
        oof[te] = train_fold(F_, y, tr, te, D, epochs=200)
    return y, oof, pll


assays = {"GB1 binding": "/tmp/feat_GB1_Olson.npz", "GFP fluorescence": "/tmp/feat_GFP.npz"}
res = {}
for name, path in assays.items():
    y, oof, pll = held_out_double(path)
    res[name] = dict(y=y, oof=oof, pll=pll,
                     m1=spearmanr(y, oof).statistic, zs=spearmanr(y, pll).statistic)
    print(f"{name}: Model1={res[name]['m1']:.3f}  zero-shot={res[name]['zs']:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))

# Panel A — bars
names = list(res.keys())
x = np.arange(len(names))
zs = [res[n]["zs"] for n in names]
m1 = [res[n]["m1"] for n in names]
ax[0].bar(x - 0.2, zs, 0.4, color=GREY, label="zero-shot DPLM")
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
out = Path("paper/figures_epistasis")
fig.savefig(out / "fig8_model1_function.pdf", bbox_inches="tight")
fig.savefig(out / "fig8_model1_function.png", bbox_inches="tight")
print("saved fig8_model1_function")
