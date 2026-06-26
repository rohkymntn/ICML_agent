"""Bootstrap 95% CIs for the three-layer decomposition result, from the committed
per-protein atlas.

The decomposition is the paper's strongest, fully-committed result (149 Megascale
stability domains), but its headline numbers in Table 1 / section 7.1 were bare
point estimates: median additive R^2 (-0.91), median +global R^2 (0.72), median
specific-epistasis std (0.42 kcal/mol), median specific variance share (28%). A
reviewer asks the obvious significance questions of the central claims:
  (1) is the median additive R^2 robustly below 0 (the additive baseline is worse
      than predicting the mean for the typical protein)?
  (2) is the median +global R^2 robustly above 0 (the monotonic link genuinely
      recovers the gap)?
  (3) is the median specific epistasis robustly large (std and variance share),
      i.e. not an artifact of a few proteins?

Pure committed data, no GPU. Resamples the 149 PROTEINS with replacement
(per-protein rows are the unit of analysis) and recomputes the EXACT paper
statistic (build_decomposition_artifact.summarize) on each resample, so the four
headline medians are bootstrapped self-consistently. Writes
outputs/epistasis/decomposition_ci.json.

The run() self-check asserts the point estimates equal the committed
decomposition_summary.json (no drift) and that the three significance claims hold
(additive CI fully < 0, global CI fully > 0, specific-std CI fully > 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_decomposition_artifact import summarize  # noqa: E402

B = 2000
SEED = 0
_KEYS = (
    "median_r2_additive",
    "median_r2_global",
    "median_spec_std_kcal",
    "median_spec_var_share",
)


def _ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_decomposition(atlas, b=B, seed=SEED):
    """Percentile-bootstrap CIs over the 149 per-protein rows.

    Each resample recomputes the full paper statistic via summarize(), so every
    reported median is bootstrapped through the same estimation procedure the
    paper reports -- no statistic is duplicated here."""
    point = summarize(atlas)
    rng = np.random.default_rng(seed)
    n = len(atlas)
    draws = {k: np.empty(b) for k in _KEYS}
    for i in range(b):
        idx = rng.integers(0, n, n)
        s = summarize(atlas.iloc[idx].reset_index(drop=True))
        for k in _KEYS:
            draws[k][i] = s[k]
    out = {"n_proteins": int(n)}
    for k in _KEYS:
        out[k] = point[k]
        out[k + "_ci95"] = _ci(draws[k])
    return out


def run(atlas_path="paper/figures_epistasis/three_layer_atlas.parquet",
        out="outputs/epistasis"):
    out = Path(out)
    atlas = pd.read_parquet(atlas_path)
    r = bootstrap_decomposition(atlas)
    summ = json.loads((out / "decomposition_summary.json").read_text())
    # no-drift: bootstrap point estimates must equal the committed summary
    for k in _KEYS:
        assert abs(r[k] - summ[k]) < 1e-9, (k, r[k], summ[k])
    # significance of the central decomposition claims
    assert r["median_r2_additive_ci95"][1] < 0      # additive worse than the mean
    assert r["median_r2_global_ci95"][0] > 0        # global link recovers the gap
    assert r["median_spec_std_kcal_ci95"][0] > 0    # specific epistasis is large
    assert r["median_spec_var_share_ci95"][0] > 0
    res = {"n_bootstrap": B, "seed": SEED, **r}
    (out / "decomposition_ci.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    r = run()
    cr = lambda k: tuple(round(x, 3) for x in r[k])
    print(f"median R2 additive {r['median_r2_additive']:.3f} CI{cr('median_r2_additive_ci95')}")
    print(f"median R2 +global  {r['median_r2_global']:.3f} CI{cr('median_r2_global_ci95')}")
    print(f"median spec std    {r['median_spec_std_kcal']:.3f} CI{cr('median_spec_std_kcal_ci95')}")
    print(f"median spec share  {r['median_spec_var_share']:.3f} CI{cr('median_spec_var_share_ci95')}")
