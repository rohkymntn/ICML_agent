"""Epistasis-guided design (the RIGHT way to use Model 1 for generation).

Don't fine-tune DPLM to imitate a deleterious-heavy landscape. Instead, use an
epistasis-aware predictor (DPLM embeddings -> function) to SELECT the designed
library. On GB1's complete 4-site landscape: train each predictor on a fraction
of variants, predict function for held-out variants, select the top 5% as the
"designed library", and report its measured function vs additive selection,
pairwise selection, random, and the oracle -- as a function of training size.

Win = the DPLM-embedding (epistasis-aware) selector designs higher-function
libraries than additive, especially data-efficiently (small training fraction).
"""
from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("/tmp/gb1wu_feat.npz")
emb = z["emb"].astype(np.float32)
combo = z["combo"].astype(int)
score = z["score"].astype(float)
keep = (combo < 20).all(1)
emb, combo, score = emb[keep], combo[keep], score[keep]
N, P, D = emb.shape
print(f"[guided] {N} GB1 combos, {P} positions")

def oh(X, order):
    blocks = []
    for cp in combinations(range(P), order):
        idx = np.zeros(len(X), dtype=np.int64)
        for c in cp:
            idx = idx * 20 + X[:, c]
        blocks.append(sparse.csr_matrix((np.ones(len(X)), (np.arange(len(X)), idx)), shape=(len(X), 20 ** order)))
    return sparse.hstack(blocks).tocsr()

F1 = oh(combo, 1)
F2 = sparse.hstack([F1, oh(combo, 2)]).tocsr()
PLM = emb.reshape(N, P * D)
PLM = (PLM - PLM.mean(0)) / (PLM.std(0) + 1e-6)

rng = np.random.default_rng(0)
perm = rng.permutation(N)
nte = int(0.4 * N)
te, pool = perm[:nte], perm[nte:]

def design_mean(pred, frac=0.05):
    k = max(1, int(len(te) * frac))
    return score[te[np.argsort(-pred[te])[:k]]].mean()

fracs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25]
res = {"additive": [], "pairwise": [], "PLM (Model 1)": []}
for f in fracs:
    tr = pool[: max(80, int(f * len(pool)))]
    pa = np.zeros(N); pa[te] = Ridge(alpha=10.0).fit(F1[tr], score[tr]).predict(F1[te])
    pp = np.zeros(N); pp[te] = Ridge(alpha=10.0).fit(F2[tr], score[tr]).predict(F2[te])
    pm = np.zeros(N); pm[te] = Ridge(alpha=100.0).fit(PLM[tr], score[tr]).predict(PLM[te])
    res["additive"].append(design_mean(pa))
    res["pairwise"].append(design_mean(pp))
    res["PLM (Model 1)"].append(design_mean(pm))
    print(f"[guided] f={f:<6} n_train={len(tr):<6} additive={res['additive'][-1]:.2f}  pairwise={res['pairwise'][-1]:.2f}  PLM={res['PLM (Model 1)'][-1]:.2f}")

k = max(1, int(len(te) * 0.05))
oracle = float(np.sort(score[te])[-k:].mean())
randm = float(score[te].mean())
print(f"[guided] oracle-top={oracle:.2f}  random/avg={randm:.2f}  (unconditioned DPLM lib mean was 1.10)")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "font.size": 11})
fig, ax = plt.subplots(figsize=(5.4, 4.0))
xs = [int(f * len(pool)) for f in fracs]
ax.axhline(oracle, color="#222", ls=":", lw=1.2, label=f"oracle ({oracle:.1f})")
for name, c in [("PLM (Model 1)", "#C0504D"), ("pairwise", "#4C9A6B"), ("additive", "#9A9A9A")]:
    ax.plot(xs, res[name], "o-", color=c, lw=1.8, label=name)
ax.axhline(randm, color="#9A9A9A", ls="--", lw=1.0, label=f"random ({randm:.2f})")
ax.set_xscale("log")
ax.set_xlabel("Training variants")
ax.set_ylabel("Measured function of designed top-5% library")
ax.set_title("Epistasis-guided design (GB1 complete landscape)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
from pathlib import Path
Path("paper/figures_epistasis").mkdir(parents=True, exist_ok=True)
fig.savefig("paper/figures_epistasis/fig12_guided_design.pdf", bbox_inches="tight")
fig.savefig("paper/figures_epistasis/fig12_guided_design.png", bbox_inches="tight")
print("[guided] saved fig12_guided_design")
