"""Cross-domain transfer: does a coupling head trained on one protein/assay
predict epistasis in a DIFFERENT protein AND a different readout?

Train on GB1 binding -> predict GFP fluorescence epistasis (and vice versa).
If transfer Spearman > 0, the head has learned protein-general epistasis rules,
not assay-specific memorization -- the framework-defining result.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from train_model1 import CouplingHead, load


def fit(Fd, y, D, epochs=200, seed=0):
    torch.manual_seed(seed)
    m = CouplingHead(D)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    ymu, ysd = y.mean(), y.std() + 1e-6
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    li = lambda a: torch.tensor(a, dtype=torch.long)
    Hi, Hj, ids, pll = t(Fd["Hi"]), t(Fd["Hj"]), li(Fd["ids"]), t(Fd["pll"])
    yt = t((y - ymu) / ysd)
    n = len(y)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 1024):
            b = perm[s : s + 1024]
            opt.zero_grad()
            F.smooth_l1_loss(m(Hi[b], Hj[b], ids[b], pll[b]), yt[b]).backward()
            opt.step()
    return m


def predict(m, Fd):
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Fd["Hi"], dtype=torch.float32), torch.tensor(Fd["Hj"], dtype=torch.float32),
              torch.tensor(Fd["ids"], dtype=torch.long), torch.tensor(Fd["pll"], dtype=torch.float32))
    return p.numpy()


Fg, yg, _, _, Dg, _ = load("/tmp/feat_GB1_Olson.npz")
Ff, yf, _, _, Df, _ = load("/tmp/feat_GFP.npz")
print(f"GB1 n={len(yg)}, GFP n={len(yf)}, dim={Dg}")

m_gb1 = fit(Fg, yg, Dg)
t_gb1_gfp = spearmanr(yf, predict(m_gb1, Ff)).statistic
m_gfp = fit(Ff, yf, Df)
t_gfp_gb1 = spearmanr(yg, predict(m_gfp, Fg)).statistic

print(f"[cross-domain] train GB1 binding -> predict GFP fluorescence:  Spearman = {t_gb1_gfp:.3f}")
print(f"[cross-domain] train GFP fluorescence -> predict GB1 binding:  Spearman = {t_gfp_gb1:.3f}")
print(f"[reference] within-assay held-out: GB1 0.37, GFP 0.14 ; zero-shot ~0.0")
