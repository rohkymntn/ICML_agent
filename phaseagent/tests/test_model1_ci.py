"""Guard the committed Model 1 bootstrap-CI artifact the paper cites (DoD #8).

`outputs/epistasis/model1_ci.json` carries the 95% CIs the paper reports for the
held-out-doubles Spearman and the gain over the zero-shot floor. We assert (a) its
point estimates reproduce exactly from the committed OOF CSV and equal the
committed model1_<assay>_summary.json (no drift), (b) each CI brackets its point
estimate, and (c) the gain over zero-shot is significant (delta CI excludes 0) --
the significance claim folded into section 7.3. A small seeded re-bootstrap
confirms the script still runs without paying the full 2000-resample cost.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_model1_ci import bootstrap_assay  # noqa: E402

OUT = ROOT / "outputs" / "epistasis"
CI = OUT / "model1_ci.json"


def test_ci_point_estimates_no_drift_and_significant():
    if not CI.exists():
        pytest.skip("model1_ci.json not committed")
    res = json.loads(CI.read_text())
    for assay, v in res["assays"].items():
        summ = json.loads((OUT / f"model1_{assay}_summary.json").read_text())
        # point estimates equal the committed Model 1 summary (no drift)
        assert v["model1_doubles_spearman"] == summ["model1_doubles_spearman"]
        assert v["zeroshot_spearman"] == summ["zeroshot_spearman"]
        # each CI brackets its point estimate
        for key, ci in [("model1_doubles_spearman", "model1_doubles_ci95"),
                        ("zeroshot_spearman", "zeroshot_ci95"),
                        ("delta_doubles", "delta_ci95")]:
            lo, hi = v[ci]
            assert lo <= v[key] <= hi, f"{assay}: {key} outside {ci}"
        # the gain over the zero-shot floor is significant
        assert v["delta_ci95"][0] > 0, f"{assay}: delta CI includes 0"


def test_bootstrap_reproduces_committed_point_estimates():
    """A cheap re-run (small B, same seed) confirms the script reads the committed
    CSV and recovers the committed point estimates and ordered CIs."""
    if not CI.exists():
        pytest.skip("model1_ci.json not committed")
    res = json.loads(CI.read_text())
    for assay, v in res["assays"].items():
        df = pd.read_csv(OUT / f"model1_oof_{assay}.csv")
        r = bootstrap_assay(df, b=100, seed=0)
        assert abs(r["model1_doubles_spearman"] - v["model1_doubles_spearman"]) < 1e-12
        lo, hi = r["model1_doubles_ci95"]
        assert lo < r["model1_doubles_spearman"] < hi
