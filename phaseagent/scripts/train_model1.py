"""Model 1: a learned coupling operator that BEATS zero-shot DPLM at predicting
the specific-epistasis residual, held out.

Features per double:
  - DPLM hidden states h_i, h_j of the two mutated positions (+ bilinear coupling)
  - amino-acid identity embeddings (mut + wt at both sites)
  - zero-shot pseudo-LL epistasis (feature + residual base)
  - OPTIONAL (--shift): representation shifts  H^(i->a)[j]-H^wt[j]  and
    H^(j->b)[i]-H^wt[i], a direct coupling signal meant to transfer to UNSEEN
    positions (the hard held-out-position case).

Eval = proof: held-out DOUBLES (unseen pair label) and held-out POSITIONS
(unseen sites), vs zero-shot on the same rows.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F


class CouplingHead(nn.Module):
    def __init__(self, D, dproj=128, n_aa=21, demb=16, hidden=256, p=0.3, use_shift=False):
        super().__init__()
        self.use_shift = use_shift
        self.pi = nn.Linear(D, dproj)
        self.pj = nn.Linear(D, dproj)
        self.aa = nn.Embedding(n_aa + 1, demb)
        extra = 0
        if use_shift:
            self.si = nn.Linear(D, dproj)
            self.sj = nn.Linear(D, dproj)
            extra = dproj * 2
        in_dim = dproj * 3 + demb * 4 + 1 + extra
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(p), nn.Linear(hidden // 2, 1),
        )
        self.pll_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, hi, hj, ids, pll, shi=None, shj=None):
        pi, pj = F.gelu(self.pi(hi)), F.gelu(self.pj(hj))
        a = self.aa(ids).reshape(ids.shape[0], -1)
        parts = [pi, pj, pi * pj, a, pll.unsqueeze(-1)]
        if self.use_shift:
            parts += [F.gelu(self.si(shi)), F.gelu(self.sj(shj))]
        x = torch.cat(parts, -1)
        return self.mlp(x).squeeze(-1) + self.pll_gain * pll


def train_fold(F_, y, tr, te, D, use_shift=False, epochs=200, lr=2e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CouplingHead(D, use_shift=use_shift).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    li = lambda a: torch.tensor(a, dtype=torch.long, device=dev)
    Hi, Hj, ids, pll = F_["Hi"], F_["Hj"], F_["ids"], F_["pll"]
    shi = F_.get("shi"); shj = F_.get("shj")

    def batch(idx):
        out = [t(Hi[idx]), t(Hj[idx]), li(ids[idx]), t(pll[idx])]
        if use_shift:
            out += [t(shi[idx]), t(shj[idx])]
        return out

    y_tr = t((y[tr] - ymu) / ysd)
    m.train()
    n = len(tr)
    for _ in range(epochs):
        perm = tr[torch.randperm(n).numpy()]
        for s in range(0, n, 1024):
            b = perm[s : s + 1024]
            opt.zero_grad()
            pred = m(*batch(b))
            F.smooth_l1_loss(pred, t((y[b] - ymu) / ysd)).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(*batch(te))
    return pred.cpu().numpy() * ysd + ymu


def load(feat, shift=None):
    z = np.load(feat)
    Hi, Hj, meta = z["Hi"].astype(np.float32), z["Hj"].astype(np.float32), z["meta"]
    y, pll = meta[:, 0], meta[:, 1]
    F_ = {
        "Hi": (Hi - Hi.mean(0)) / (Hi.std(0) + 1e-6),
        "Hj": (Hj - Hj.mean(0)) / (Hj.std(0) + 1e-6),
        "ids": meta[:, 2:6].astype(int),
        "pll": (pll - pll.mean()) / (pll.std() + 1e-6),
    }
    pos = meta[:, 6:8].astype(int)
    use_shift = False
    if shift:
        zs = np.load(shift)
        assert np.allclose(zs["meta"][:, 0], y, atol=1e-4), "shift rows misaligned with feat rows"
        si, sj = zs["shift_i"].astype(np.float32), zs["shift_j"].astype(np.float32)
        F_["shi"] = (si - si.mean(0)) / (si.std(0) + 1e-6)
        F_["shj"] = (sj - sj.mean(0)) / (sj.std(0) + 1e-6)
        use_shift = True
    return F_, y, pll, pos, Hi.shape[1], use_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--shift", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    F_, y, pll, pos, D, use_shift = load(args.feat, args.shift)
    tag = "DPLM + representation-shifts" if use_shift else "DPLM hidden states"
    print(f"[model1] {len(y)} doubles, dim={D}, features = {tag}")
    print(f"[model1] zero-shot PLL Spearman: {spearmanr(y, pll).statistic:.3f}")

    def run(splits, label):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            oof[te] = train_fold(F_, y, tr, te, D, use_shift=use_shift, epochs=args.epochs)
        m1, zs = spearmanr(y, oof).statistic, spearmanr(y, pll).statistic
        print(f"[{label}] Model 1={m1:.3f}  zero-shot={zs:.3f}  "
              f"{'WIN +' + format(m1 - zs, '.3f') if m1 > zs else 'lose'}")

    run(list(KFold(5, shuffle=True, random_state=0).split(y)), "held-out DOUBLES ")
    run(list(GroupKFold(5).split(y, groups=pos[:, 0])), "held-out POSITION")


if __name__ == "__main__":
    main()
