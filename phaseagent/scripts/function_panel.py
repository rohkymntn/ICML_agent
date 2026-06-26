"""Model 1 across ALL function multi-mutant assays + function cross-protein transfer.

Turns "GB1 0.37, GFP 0.14" into a distribution over ~12 function proteins, and
tests whether training on a pool of function proteins predicts epistasis in a
HELD-OUT function protein (leave-one-assay-out) -- the function analog of the
0.335 stability cross-protein result.
"""
import glob
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from train_model1 import load, train_fold

# include GB1/GFP we already have, plus the newly extracted assays
paths = sorted(set(glob.glob("/tmp/func_feats/feat_*.npz")) |
               {"/tmp/feat_GB1_Olson.npz", "/tmp/feat_GFP.npz"})
A = {}
for fp in paths:
    name = fp.split("feat_")[-1].replace(".npz", "")
    try:
        F_, y, pll, pos, D, _ = load(fp)
        if len(y) >= 60:
            A[name] = {"F": F_, "y": y, "pll": pll, "D": D, "n": len(y)}
    except Exception as e:
        print(f"skip {fp}: {e}")
print(f"[panel] {len(A)} function assays: {list(A)}")

# per-assay held-out-double
rows = []
for name, a in A.items():
    oof = np.full(a["n"], np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(a["n"])):
        oof[te] = train_fold(a["F"], a["y"], tr, te, a["D"], epochs=150)
    m1 = spearmanr(a["y"], oof).statistic
    zs = spearmanr(a["y"], a["pll"]).statistic
    rows.append((name, a["n"], m1, zs))
    print(f"[panel] {name:12s} n={a['n']:<6d} Model1={m1:.3f}  zero-shot={zs:.3f}")

m1s = [r[2] for r in rows]
zss = [r[3] for r in rows]
print(f"[panel] WITHIN-PROTEIN: Model1 median={np.median(m1s):.3f}  zero-shot median={np.median(zss):.3f}  (n_assays={len(rows)})")

# function cross-protein: leave-one-assay-out (pooled, per-assay z-scored target)
names = list(A)
D = A[names[0]]["D"]
poolF = {k: np.concatenate([A[n]["F"][k] for n in names]) for k in ("Hi", "Hj", "ids", "pll")}
poolY = np.concatenate([(A[n]["y"] - A[n]["y"].mean()) / (A[n]["y"].std() + 1e-6) for n in names])
asg = np.concatenate([[i] * A[n]["n"] for i, n in enumerate(names)])
loo = []
for i, n in enumerate(names):
    tr = np.where(asg != i)[0]
    te = np.where(asg == i)[0]
    pred = train_fold(poolF, poolY, tr, te, D, epochs=120)
    rho = spearmanr(A[n]["y"], pred).statistic
    loo.append((n, rho))
    print(f"[panel] cross-protein (train others -> {n}): {rho:.3f}")
loo_med = float(np.median([r[1] for r in loo]))
print(f"[panel] FUNCTION CROSS-PROTEIN (leave-one-assay-out) median Spearman = {loo_med:.3f}")

# figure
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 4.0))
order = np.argsort(m1s)[::-1]
xs = np.arange(len(rows))
ax.bar(xs - 0.2, [m1s[i] for i in order], 0.4, color="#C0504D", label="Model 1 (within-protein)")
ax.bar(xs + 0.2, [zss[i] for i in order], 0.4, color="#9A9A9A", label="zero-shot DPLM")
ax.axhline(loo_med, color="#3B6FB6", ls="--", lw=1.6, label=f"function cross-protein (median {loo_med:.2f})")
ax.set_xticks(xs); ax.set_xticklabels([rows[i][0] for i in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Held-out Spearman (specific epistasis)")
ax.set_title("Model 1 across function multi-mutant assays")
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
from pathlib import Path
Path("paper/figures_epistasis").mkdir(parents=True, exist_ok=True)
fig.savefig("paper/figures_epistasis/fig11_function_panel.pdf", bbox_inches="tight")
fig.savefig("paper/figures_epistasis/fig11_function_panel.png", bbox_inches="tight")
print("[panel] saved fig11_function_panel")
