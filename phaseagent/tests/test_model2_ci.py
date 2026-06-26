"""Guard the committed Model 2 design bootstrap-CI artifact the paper cites (DoD #8).

`outputs/epistasis/model2_ci.json` carries the 95% CIs the paper reports for the
matched-additive control Spearman and the gain-of-function design lift. We assert
(a) its point estimates reproduce exactly from the committed model2_GB1_summary.json
(no drift), (b) each CI brackets its point estimate, and (c) the three significance
claims folded into section 7.4 hold -- the matched-additive control excludes 0
(genuine epistasis, not relearned additivity), the design lift excludes 0, and the
lift over the zero-shot PLL rerank excludes 0 (Model 1 beats the PLM-as-reward
baseline). A small seeded re-bootstrap confirms the script still runs without
paying the full 2000-resample cost.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_model2_ci import bootstrap_model2  # noqa: E402

OUT = ROOT / "outputs" / "epistasis"
CI = OUT / "model2_ci.json"


def test_ci_point_estimates_no_drift_and_significant():
    if not CI.exists():
        pytest.skip("model2_ci.json not committed")
    v = json.loads(CI.read_text())
    summ = json.loads((OUT / "model2_GB1_summary.json").read_text())
    # point estimates equal the committed Model 2 summary (no drift)
    assert v["matched_additive_spearman"] == summ["matched_additive_spearman"]
    assert v["gof_lift_model1"] == summ["gof_lift_model1"]
    assert v["gof_lift_zeroshot"] == summ["gof_lift_zeroshot"]
    # each CI brackets its point estimate
    for key, ci in [("matched_additive_spearman", "matched_additive_ci95"),
                    ("gof_lift_model1", "gof_lift_model1_ci95"),
                    ("gof_lift_zeroshot", "gof_lift_zeroshot_ci95"),
                    ("gof_lift_delta_model1_minus_zeroshot", "gof_lift_delta_ci95")]:
        lo, hi = v[ci]
        assert lo <= v[key] <= hi, f"{key} outside {ci}"
    # the three significance claims the paper makes
    assert v["matched_additive_ci95"][0] > 0, "matched-additive control CI includes 0"
    assert v["gof_lift_model1_ci95"][0] > 0, "design lift CI includes 0"
    assert v["gof_lift_delta_ci95"][0] > 0, "Model1 - zero-shot lift CI includes 0"


def test_bootstrap_reproduces_committed_point_estimates():
    """A cheap re-run (small B, same seed) confirms the script reads the committed
    CSV and recovers the committed point estimates and ordered CIs."""
    if not CI.exists():
        pytest.skip("model2_ci.json not committed")
    v = json.loads(CI.read_text())
    df = pd.read_csv(OUT / "model2_oof_GB1.csv")
    r = bootstrap_model2(df, b=100, seed=0)
    assert abs(r["matched_additive_spearman"] - v["matched_additive_spearman"]) < 1e-12
    assert abs(r["gof_lift_model1"] - v["gof_lift_model1"]) < 1e-12
    lo, hi = r["matched_additive_ci95"]
    assert lo < r["matched_additive_spearman"] < hi
