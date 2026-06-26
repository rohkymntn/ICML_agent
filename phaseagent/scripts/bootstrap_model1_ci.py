"""Bootstrap 95% CIs for the Model 1 held-out-doubles Spearman vs the zero-shot
PLL floor, from the committed out-of-fold CSVs.

Adds uncertainty quantification to the paper's headline prediction numbers
(tab:model1): a point Spearman with no CI is not enough for a reviewer to judge
whether Model 1 genuinely beats the near-zero zero-shot floor. Pure committed
data, no GPU.

Writes outputs/epistasis/model1_ci.json. The run() self-check asserts the point
estimates equal the committed model1_<assay>_summary.json (no drift) and that the
gain over zero-shot is significant (paired-resample 95% CI on the per-resample
Spearman difference excludes 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSAYS = ["GB1_Olson", "GFP"]
B = 2000
SEED = 0


def _ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_assay(df, b=B, seed=SEED):
    """Percentile-bootstrap CIs over the held-out doubles of one assay.

    Resamples (y, model1, zero-shot) triples with replacement and recomputes both
    Spearmans on the SAME resample so their difference is paired (a clean
    significance test for the gain over the zero-shot floor)."""
    y = df["y_true"].to_numpy()
    m1 = df["oof_doubles"].to_numpy()
    zs = df["pll"].to_numpy()
    point_m1 = float(spearmanr(y, m1).statistic)
    point_zs = float(spearmanr(y, zs).statistic)

    rng = np.random.default_rng(seed)
    n = len(y)
    bm1 = np.empty(b)
    bzs = np.empty(b)
    bdelta = np.empty(b)
    for k in range(b):
        idx = rng.integers(0, n, n)
        a = float(spearmanr(y[idx], m1[idx]).statistic)
        c = float(spearmanr(y[idx], zs[idx]).statistic)
        bm1[k], bzs[k], bdelta[k] = a, c, a - c
    return {
        "n_doubles": int(n),
        "model1_doubles_spearman": point_m1,
        "model1_doubles_ci95": _ci(bm1),
        "zeroshot_spearman": point_zs,
        "zeroshot_ci95": _ci(bzs),
        "delta_doubles": point_m1 - point_zs,
        "delta_ci95": _ci(bdelta),
    }


def run(out="outputs/epistasis"):
    out = Path(out)
    res = {"n_bootstrap": B, "seed": SEED, "assays": {}}
    for a in ASSAYS:
        r = bootstrap_assay(pd.read_csv(out / f"model1_oof_{a}.csv"))
        summ = json.loads((out / f"model1_{a}_summary.json").read_text())
        # no-drift: bootstrap point estimates must equal the committed summary
        assert abs(r["model1_doubles_spearman"] - summ["model1_doubles_spearman"]) < 1e-9, a
        assert abs(r["zeroshot_spearman"] - summ["zeroshot_spearman"]) < 1e-9, a
        # significance: the gain over the zero-shot floor excludes 0
        assert r["delta_ci95"][0] > 0, a
        res["assays"][a] = r
    (out / "model1_ci.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    r = run()
    for a, v in r["assays"].items():
        ci = lambda k: tuple(round(x, 3) for x in v[k])
        print(f"[{a}] Model1 {v['model1_doubles_spearman']:.3f} CI{ci('model1_doubles_ci95')}  "
              f"zero-shot {v['zeroshot_spearman']:.3f} CI{ci('zeroshot_ci95')}  "
              f"delta {v['delta_doubles']:.3f} CI{ci('delta_ci95')}")
