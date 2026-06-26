"""Data-efficiency of predicting 3rd-order epistasis from DPLM embeddings.

Target: the higher-order residual of GB1 binding = score - (additive+pairwise fit
on the full landscape). Question: trained on a FRACTION of variants, can
context-aware DPLM embeddings predict this residual on held-out variants better
than a one-hot 3rd-order model (which needs dense combinatorial sampling) or an
additive+pairwise model (which is ~0 on the residual by construction)?
"""
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
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
print(f"[order3] {N} variants, {P} positions, dim {D}")

def oh(X, order):
    blocks = []
    for cp in combinations(range(P), order):
        idx = np.zeros(len(X), dtype=np.int64)
        for c in cp:
            idx = idx * 20 + X[:, c]
        blocks.append(sparse.csr_matrix((np.ones(len(X)), (np.arange(len(X)), idx)), shape=(len(X), 20 ** order)))
    return sparse.hstack(blocks).tocsr()

F2 = sparse.hstack([oh(combo, 1), oh(combo, 2)]).tocsr()
resid = score - Ridge(alpha=10.0).fit(F2, score).predict(F2)  # ground-truth higher-order residual
print(f"[order3] higher-order residual is {np.var(resid)/np.var(score)*100:.0f}% of binding variance")

PLM = emb.reshape(N, P * D)
PLM = (PLM - PLM.mean(0)) / (PLM.std(0) + 1e-6)
OH3 = oh(combo, 3)

rng = np.random.default_rng(0)
perm = rng.permutation(N)
nte = int(0.3 * N)
te, pool = perm[:nte], perm[nte:]
fracs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
sp_plm, sp_oh, sp_ap = [], [], []
for f in fracs:
    tr = pool[: max(50, int(f * len(pool)))]
    pp = Ridge(alpha=100.0).fit(PLM[tr], resid[tr]).predict(PLM[te])
    po = Ridge(alpha=10.0).fit(OH3[tr], resid[tr]).predict(OH3[te])
    pa = Ridge(alpha=10.0).fit(F2[tr], resid[tr]).predict(F2[te])
    sp_plm.append(spearmanr(resid[te], pp).statistic)
    sp_oh.append(spearmanr(resid[te], po).statistic)
    sp_ap.append(spearmanr(resid[te], pa).statistic)
    print(f"[order3] f={f:<5} n_train={len(tr):<6}  PLM={sp_plm[-1]:.3f}  one-hot-3rd={sp_oh[-1]:.3f}  add+pair={sp_ap[-1]:.3f}")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "font.size": 11})
fig, ax = plt.subplots(figsize=(5.2, 4.0))
xs = [f * len(pool) for f in fracs]
ax.plot(xs, sp_plm, "o-", color="#C0504D", lw=2, label="DPLM embeddings (context-aware)")
ax.plot(xs, sp_oh, "o-", color="#3B6FB6", lw=1.6, label="one-hot 3rd-order model")
ax.plot(xs, sp_ap, "o--", color="#9A9A9A", lw=1.4, label="additive + pairwise")
ax.set_xscale("log")
ax.set_xlabel("Training variants")
ax.set_ylabel("Held-out Spearman on 3rd-order residual")
ax.set_title("Data efficiency: predicting 3rd-order epistasis\n(GB1 binding, complete 4-site landscape)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
from pathlib import Path
Path("paper/figures_epistasis").mkdir(parents=True, exist_ok=True)
fig.savefig("paper/figures_epistasis/fig10_order3_dataeff.pdf", bbox_inches="tight")
fig.savefig("paper/figures_epistasis/fig10_order3_dataeff.png", bbox_inches="tight")
print("[order3] saved fig10_order3_dataeff")
