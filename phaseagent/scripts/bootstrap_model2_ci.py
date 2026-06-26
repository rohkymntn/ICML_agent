"""Bootstrap 95% CIs for the Model 2 design result, from the committed
out-of-fold table.

Model 1's prediction numbers gained CIs (bootstrap_model1_ci.py); the Model 2
*design* result -- the paper's central design claim -- was still a set of bare
point estimates. A reviewer asks two significance questions of it:
  (1) is the matched-additive control Spearman (0.28) distinguishable from 0?
      it is the control that proves the recovered gain is genuine epistasis and
      not relearned additivity, so it must be significantly positive.
  (2) does Model 1's gain-of-function lift (+1.09) significantly beat both zero
      (random pool selection) and the zero-shot DPLM PLL best-of-N rerank reward
      (the PLM-as-reward design baseline, lift -0.22)?

Pure committed data, no GPU. Resamples the 12,000 held-out GB1 doubles with
replacement and recomputes the EXACT paper statistic (summarize_model2) on each
resample, so the matched-additive median-over-deciles and the top-10%-of-pool
lift are bootstrapped self-consistently. Writes outputs/epistasis/model2_ci.json.

The run() self-check asserts the point estimates equal the committed
model2_GB1_summary.json (no drift) and that all three significance claims hold
(each lower CI bound excludes 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from model2_design import summarize_model2  # noqa: E402

ASSAY = "GB1"
B = 2000
SEED = 0


def _ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_model2(df, b=B, seed=SEED):
    """Percentile-bootstrap CIs over the held-out doubles of the design pool.

    Each resample recomputes the full paper statistic via summarize_model2, so
    the matched-additive control and the gain-of-function lifts are bootstrapped
    through the same estimation procedure the paper reports. The Model 1 vs
    zero-shot lift difference is computed on the SAME resample (paired), a clean
    test that ranking by predicted epistasis beats the PLM-as-reward rerank."""
    point = summarize_model2(df)
    rng = np.random.default_rng(seed)
    n = len(df)
    bmatched = np.empty(b)
    blift_m1 = np.empty(b)
    blift_zs = np.empty(b)
    bdelta = np.empty(b)
    for k in range(b):
        idx = rng.integers(0, n, n)
        s = summarize_model2(df.iloc[idx].reset_index(drop=True))
        bmatched[k] = s["matched_additive_spearman"]
        blift_m1[k] = s["gof_lift_model1"]
        blift_zs[k] = s["gof_lift_zeroshot"]
        bdelta[k] = s["gof_lift_model1"] - s["gof_lift_zeroshot"]
    return {
        "n_doubles": int(n),
        "matched_additive_spearman": point["matched_additive_spearman"],
        "matched_additive_ci95": _ci(bmatched),
        "gof_lift_model1": point["gof_lift_model1"],
        "gof_lift_model1_ci95": _ci(blift_m1),
        "gof_lift_zeroshot": point["gof_lift_zeroshot"],
        "gof_lift_zeroshot_ci95": _ci(blift_zs),
        "gof_lift_delta_model1_minus_zeroshot": point["gof_lift_model1"] - point["gof_lift_zeroshot"],
        "gof_lift_delta_ci95": _ci(bdelta),
    }


def run(out="outputs/epistasis"):
    out = Path(out)
    df = pd.read_csv(out / f"model2_oof_{ASSAY}.csv")
    r = bootstrap_model2(df)
    summ = json.loads((out / f"model2_{ASSAY}_summary.json").read_text())
    # no-drift: bootstrap point estimates must equal the committed summary
    assert abs(r["matched_additive_spearman"] - summ["matched_additive_spearman"]) < 1e-9
    assert abs(r["gof_lift_model1"] - summ["gof_lift_model1"]) < 1e-9
    assert abs(r["gof_lift_zeroshot"] - summ["gof_lift_zeroshot"]) < 1e-9
    # significance: the matched-additive control is positive (genuine epistasis,
    # not relearned additivity); the design lift beats both 0 and the zero-shot rerank.
    assert r["matched_additive_ci95"][0] > 0
    assert r["gof_lift_model1_ci95"][0] > 0
    assert r["gof_lift_delta_ci95"][0] > 0
    res = {"n_bootstrap": B, "seed": SEED, "assay": ASSAY, **r}
    (out / "model2_ci.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    r = run()
    cr = lambda k: tuple(round(x, 3) for x in r[k])
    print(f"matched-additive control rho {r['matched_additive_spearman']:.3f} CI{cr('matched_additive_ci95')}")
    print(f"Model 1 design lift {r['gof_lift_model1']:.3f} CI{cr('gof_lift_model1_ci95')}")
    print(f"zero-shot rerank lift {r['gof_lift_zeroshot']:.3f} CI{cr('gof_lift_zeroshot_ci95')}")
    print(f"Model1 - zero-shot lift {r['gof_lift_delta_model1_minus_zeroshot']:.3f} CI{cr('gof_lift_delta_ci95')}")
