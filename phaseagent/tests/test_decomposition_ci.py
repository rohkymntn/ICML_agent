"""Guard the committed decomposition bootstrap-CI artifact the paper cites (DoD #8).

`outputs/epistasis/decomposition_ci.json` carries the 95% CIs section 7.1 reports
for the four headline decomposition medians (additive R^2, +global R^2, specific
std, specific variance share). We assert (a) its point estimates reproduce exactly
from the committed decomposition_summary.json (no drift), (b) each CI brackets its
point estimate, and (c) the three significance claims folded into section 7.1 hold
-- the additive R^2 CI is entirely below 0 (additive worse than the mean), the
+global R^2 CI is entirely above 0 (the link recovers the gap), and the specific
std / share CIs are entirely above 0 (the residual is robustly large). A small
seeded re-bootstrap confirms the script still runs without the full 2000-resample
cost.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_decomposition_ci import bootstrap_decomposition  # noqa: E402

OUT = ROOT / "outputs" / "epistasis"
CI = OUT / "decomposition_ci.json"
ATLAS = ROOT / "paper" / "figures_epistasis" / "three_layer_atlas.parquet"
_KEYS = ("median_r2_additive", "median_r2_global",
         "median_spec_std_kcal", "median_spec_var_share")


def test_ci_point_estimates_no_drift_and_significant():
    if not CI.exists():
        pytest.skip("decomposition_ci.json not committed")
    v = json.loads(CI.read_text())
    summ = json.loads((OUT / "decomposition_summary.json").read_text())
    for k in _KEYS:
        # point estimate equals the committed decomposition summary (no drift)
        assert v[k] == summ[k], f"{k} drift"
        lo, hi = v[k + "_ci95"]
        assert lo <= v[k] <= hi, f"{k} outside its CI"
    # the three significance claims the paper makes in section 7.1
    assert v["median_r2_additive_ci95"][1] < 0, "additive R2 CI includes 0"
    assert v["median_r2_global_ci95"][0] > 0, "+global R2 CI includes 0"
    assert v["median_spec_std_kcal_ci95"][0] > 0, "specific std CI includes 0"
    assert v["median_spec_var_share_ci95"][0] > 0, "specific share CI includes 0"


def test_bootstrap_reproduces_committed_point_estimates():
    """A cheap re-run (small B, same seed) confirms the script reads the committed
    atlas and recovers the committed point estimates and ordered CIs."""
    if not CI.exists():
        pytest.skip("decomposition_ci.json not committed")
    v = json.loads(CI.read_text())
    atlas = pd.read_parquet(ATLAS)
    r = bootstrap_decomposition(atlas, b=100, seed=0)
    for k in _KEYS:
        assert abs(r[k] - v[k]) < 1e-12, f"{k} not reproduced"
    lo, hi = r["median_r2_global_ci95"]
    assert lo < r["median_r2_global"] < hi
