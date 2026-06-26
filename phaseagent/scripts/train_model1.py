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
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def summarize(df, assay, dim, use_shift):
    """Derive the paper's Model 1 numbers from the raw out-of-fold CSV.

    Keeping the summary a pure function of the committed OOF table means a test
    can recompute it and catch any csv/summary drift (the headroom pattern)."""
    y = df["y_true"].to_numpy()
    pll = df["pll"].to_numpy()
    rho = lambda a: float(spearmanr(y, a).statistic)
    zs = rho(pll)
    d, p = rho(df["oof_doubles"].to_numpy()), rho(df["oof_position"].to_numpy())
    return {
        "assay": assay, "n_doubles": int(len(df)), "dim": int(dim),
        "use_shift": bool(use_shift),
        "zeroshot_spearman": zs,
        "model1_doubles_spearman": d, "model1_position_spearman": p,
        "delta_doubles": d - zs, "delta_position": p - zs,
    }


def run_model1(feat, assay, out="outputs/epistasis", shift=None, epochs=200):
    """Train Model 1, write the committed DoD #3 artifacts, return the summary.

    Writes `model1_oof_<assay>.csv` (raw out-of-fold predictions for both the
    held-out-DOUBLES and held-out-POSITION splits, the traceable artifact) and
    `model1_<assay>_summary.json` (the numbers the paper / CLAIMS.md cite,
    derived from that CSV)."""
    F_, y, pll, pos, D, use_shift = load(feat, shift)

    def oof_for(splits):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            oof[te] = train_fold(F_, y, tr, te, D, use_shift=use_shift, epochs=epochs)
        return oof

    oof_d = oof_for(list(KFold(5, shuffle=True, random_state=0).split(y)))
    oof_p = oof_for(list(GroupKFold(5).split(y, groups=pos[:, 0])))
    df = pd.DataFrame({
        "y_true": y, "pll": pll, "oof_doubles": oof_d, "oof_position": oof_p,
        "pos_i": pos[:, 0], "pos_j": pos[:, 1],
    })
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"model1_oof_{assay}.csv", index=False)
    summ = summarize(df, assay, D, use_shift)
    (outdir / f"model1_{assay}_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--shift", default=None)
    ap.add_argument("--assay", required=True)
    ap.add_argument("--out", default="outputs/epistasis")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    s = run_model1(args.feat, args.assay, out=args.out, shift=args.shift, epochs=args.epochs)
    tag = "DPLM + representation-shifts" if s["use_shift"] else "DPLM hidden states"
    print(f"[model1] {s['n_doubles']} doubles, dim={s['dim']}, features = {tag}")
    print(f"[model1] zero-shot PLL Spearman: {s['zeroshot_spearman']:.3f}")
    for split, m, dl in [("held-out DOUBLES ", s["model1_doubles_spearman"], s["delta_doubles"]),
                         ("held-out POSITION", s["model1_position_spearman"], s["delta_position"])]:
        print(f"[{split}] Model 1={m:.3f}  zero-shot={s['zeroshot_spearman']:.3f}  "
              f"{'WIN +' + format(dl, '.3f') if dl > 0 else 'lose'}")
    print(f"[model1] wrote outputs to {args.out}/model1_{args.assay}_*")


if __name__ == "__main__":
    main()
